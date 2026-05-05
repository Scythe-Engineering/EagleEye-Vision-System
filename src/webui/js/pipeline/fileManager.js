// File manager popup for listing, uploading, selecting, and deleting operation files.
import { BACKEND_BASE_URL } from "../config.js";
import { confirmDialog } from "../ui/confirmationDialog.js";
import {
    closeOnBackdropClick,
    closeOnEscape,
    createElement,
    getOrCreateModalElements,
    hideModal,
    showModal,
} from "../ui/modal.js";

/**
 * Registers and returns the singleton file manager popup instance.
 * @returns {{init: Function, open: Function, close: Function}}
 */
export function registerFileManagerPopup() {
    if (globalThis.FileManagerPopup) {
        return globalThis.FileManagerPopup;
    }

    const OVERLAY_ID = "fileManagerOverlay";
    const MODAL_ID = "fileManagerModal";

    /**
     * Gets or creates the overlay and modal elements used by the popup.
     * @returns {{overlay: HTMLElement, modal: HTMLElement}}
     */
    function findOverlayElements() {
        return getOrCreateModalElements({
            overlayId: OVERLAY_ID,
            modalId: MODAL_ID,
            overlayClassName:
                "fixed inset-0 bg-black bg-opacity-50 z-50 hidden flex items-center justify-center",
            overlayStyle: null,
            modalClassName:
                "bg-[#1a1a1a] rounded-lg shadow-xl max-w-2xl w-full mx-4 max-h-[90vh] flex flex-col",
        });
    }

    let currentOperationName = null;
    let currentParameterName = null;
    let currentValue = null;
    let onFileSelectedCallback = null;
    let filesList = [];

    /**
     * Loads the files for the current operation and parameter.
     * @returns {Promise<void>}
     */
    async function fetchFiles() {
        if (!currentOperationName || !currentParameterName) return;

        try {
            const response = await fetch(
                `${BACKEND_BASE_URL}/get-operation-files/${encodeURIComponent(currentOperationName)}/${encodeURIComponent(currentParameterName)}`,
            );
            if (response.ok) {
                const data = await response.json();
                filesList =
                    data.file_details ||
                    data.files.map((f) => ({
                        filename: f,
                        size: 0,
                        modified: 0,
                    }));
                renderFileList();
            } else {
                console.error("Failed to fetch files");
            }
        } catch (error) {
            console.error("Error fetching files:", error);
        }
    }

    /**
     * Formats a byte count into a human-readable file size.
     * @param {number} bytes
     * @returns {string}
     */
    function formatFileSize(bytes) {
        if (bytes === 0) return "0 B";
        const k = 1024;
        const sizes = ["B", "KB", "MB", "GB"];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return (
            Math.round((bytes / Math.pow(k, i)) * 100) / 100 + " " + sizes[i]
        );
    }

    /**
     * Formats a Unix timestamp into a locale string.
     * @param {number} timestamp
     * @returns {string}
     */
    function formatDate(timestamp) {
        if (!timestamp) return "Unknown";
        const date = new Date(timestamp * 1000);
        return date.toLocaleString();
    }

    /**
     * Renders the current file list into the modal.
     */
    function renderFileList() {
        const fileListContainer = document.getElementById(
            "fileManagerFileList",
        );
        if (!fileListContainer) return;

        fileListContainer.innerHTML = "";

        if (filesList.length === 0) {
            const emptyMessage = createElement("div", {
                className: "text-center text-[#ac8a2f] py-8",
                text: "No files available. Upload a file to get started.",
            });
            fileListContainer.appendChild(emptyMessage);
            return;
        }

        filesList.forEach((file) => {
            const fileRow = createElement("div", {
                className:
                    "flex items-center justify-between p-3 border-b border-[#414141] hover:bg-[#2a2a2a]",
            });

            const fileInfo = createElement("div", {
                className: "flex-1",
            });

            const fileName = createElement("div", {
                className: "text-white font-medium",
                text: file.filename,
            });

            const fileMeta = createElement("div", {
                className: "text-xs text-[#ac8a2f] mt-1",
                text: `${formatFileSize(file.size)} • ${formatDate(file.modified)}`,
            });

            fileInfo.appendChild(fileName);
            fileInfo.appendChild(fileMeta);

            const actions = createElement("div", {
                className: "flex gap-2",
            });

            const selectButton = createElement("button", {
                type: "button",
                className:
                    "px-3 py-1 bg-[#f9c845] text-[#232323] rounded hover:bg-[#d4a83a] text-sm",
                text: "Select",
                onclick: () => {
                    if (onFileSelectedCallback) {
                        onFileSelectedCallback(file.filename);
                    }
                    close();
                },
            });

            const deleteButton = createElement("button", {
                type: "button",
                className:
                    "px-3 py-1 bg-red-600 text-white rounded hover:bg-red-700 text-sm",
                text: "Delete",
                onclick: async () => {
                    const shouldDelete = await confirmDialog({
                        title: "Delete Operation File?",
                        message: `Delete "${file.filename}"?`,
                        detail: "This action cannot be undone.",
                        confirmText: "Delete",
                    });
                    if (shouldDelete) {
                        deleteFile(file.filename);
                    }
                },
            });

            actions.appendChild(selectButton);
            actions.appendChild(deleteButton);

            fileRow.appendChild(fileInfo);
            fileRow.appendChild(actions);
            fileListContainer.appendChild(fileRow);
        });
    }

    /**
     * Deletes the specified file for the current operation and parameter.
     * @param {string} filename
     * @returns {Promise<void>}
     */
    async function deleteFile(filename) {
        if (!currentOperationName || !currentParameterName) return;

        try {
            const response = await fetch(
                `${BACKEND_BASE_URL}/delete-operation-file/${encodeURIComponent(currentOperationName)}/${encodeURIComponent(currentParameterName)}/${encodeURIComponent(filename)}`,
                { method: "DELETE" },
            );

            if (response.ok) {
                await fetchFiles();
                if (onFileSelectedCallback) {
                    onFileSelectedCallback(null);
                }
                if (globalThis.refreshPathDropdown) {
                    globalThis.refreshPathDropdown(null);
                }
            } else {
                const error = await response.json();
                alert(
                    `Failed to delete file: ${error.error || "Unknown error"}`,
                );
            }
        } catch (error) {
            console.error("Error deleting file:", error);
            alert("Failed to delete file");
        }
    }

    /**
     * Uploads a file for the current operation and parameter.
     * @param {File} file
     * @returns {Promise<void>}
     */
    async function uploadFile(file) {
        if (!currentOperationName || !currentParameterName) return;

        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch(
                `${BACKEND_BASE_URL}/upload-operation-file/${encodeURIComponent(currentOperationName)}/${encodeURIComponent(currentParameterName)}`,
                {
                    method: "POST",
                    body: formData,
                },
            );

            if (response.ok) {
                const data = await response.json();
                await fetchFiles();
                if (onFileSelectedCallback) {
                    onFileSelectedCallback(data.filename);
                }
                if (globalThis.refreshPathDropdown) {
                    globalThis.refreshPathDropdown(data.filename);
                }
            } else {
                const error = await response.json();
                alert(
                    `Failed to upload file: ${error.error || "Unknown error"}`,
                );
            }
        } catch (error) {
            console.error("Error uploading file:", error);
            alert("Failed to upload file");
        }
    }

    /**
     * Renders the popup contents.
     */
    function render() {
        const { overlay, modal } = findOverlayElements();

        modal.innerHTML = "";

        const header = createElement("div", {
            className: "p-6 border-b border-[#414141]",
        });

        const title = createElement("h2", {
            className: "text-xl font-bold text-[#f9c845]",
            text: `Manage Files: ${currentParameterName || "Unknown"}`,
        });

        const closeButton = createElement("button", {
            type: "button",
            className: "absolute top-4 right-4 text-[#ac8a2f] hover:text-white",
            text: "×",
            onclick: close,
            style: "font-size: 2rem; line-height: 1;",
        });

        header.appendChild(title);
        header.style.position = "relative";
        header.appendChild(closeButton);

        const body = createElement("div", {
            className: "p-6 flex-1 overflow-y-auto",
        });

        const uploadSection = createElement("div", {
            className:
                "mb-6 p-4 border border-[#414141] rounded-lg bg-[#232323]",
        });

        const uploadLabel = createElement("label", {
            className: "block text-sm font-medium text-[#f9c845] mb-2",
            text: "Upload File",
        });

        const fileInput = createElement("input", {
            type: "file",
            id: "fileManagerFileInput",
            className:
                "w-full text-white file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-[#f9c845] file:text-[#232323] hover:file:bg-[#d4a83a]",
        });

        fileInput.addEventListener("change", (e) => {
            const file = e.target.files[0];
            if (file) {
                uploadFile(file);
                e.target.value = "";
            }
        });

        uploadSection.appendChild(uploadLabel);
        uploadSection.appendChild(fileInput);

        const fileListSection = createElement("div", {
            className: "mb-4",
        });

        const fileListLabel = createElement("h3", {
            className: "text-lg font-medium text-[#f9c845] mb-3",
            text: "Available Files",
        });

        const fileListContainer = createElement("div", {
            id: "fileManagerFileList",
            className:
                "border border-[#414141] rounded-lg bg-[#232323] max-h-96 overflow-y-auto",
        });

        fileListSection.appendChild(fileListLabel);
        fileListSection.appendChild(fileListContainer);

        body.appendChild(uploadSection);
        body.appendChild(fileListSection);

        const footer = createElement("div", {
            className: "p-6 border-t border-[#414141] flex justify-end",
        });

        const cancelButton = createElement("button", {
            type: "button",
            className:
                "px-4 py-2 bg-[#414141] text-white rounded hover:bg-[#515151]",
            text: "Close",
            onclick: close,
        });

        footer.appendChild(cancelButton);

        modal.appendChild(header);
        modal.appendChild(body);
        modal.appendChild(footer);
    }

    /**
     * Initializes backdrop and escape handling for the popup.
     */
    function init() {
        const { overlay } = findOverlayElements();
        closeOnBackdropClick(overlay, close);
        closeOnEscape(overlay, close);
    }

    /**
     * Opens the popup for a specific operation parameter.
     * @param {string} operationName
     * @param {string} parameterName
     * @param {*} currentValueParam
     * @param {Function} onFileSelected
     */
    function open(
        operationName,
        parameterName,
        currentValueParam,
        onFileSelected,
    ) {
        currentOperationName = operationName;
        currentParameterName = parameterName;
        currentValue = currentValueParam;
        onFileSelectedCallback = onFileSelected;

        render();
        fetchFiles();

        const { overlay } = findOverlayElements();
        showModal(overlay);
    }

    /**
     * Closes the popup and clears its state.
     */
    function close() {
        const { overlay } = findOverlayElements();
        hideModal(overlay);
        currentOperationName = null;
        currentParameterName = null;
        currentValue = null;
        onFileSelectedCallback = null;
        filesList = [];
    }

    const popup = {
        init,
        open,
        close,
    };
    globalThis.FileManagerPopup = popup;

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }

    return popup;
}

registerFileManagerPopup();
