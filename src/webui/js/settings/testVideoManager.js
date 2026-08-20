import { BACKEND_BASE_URL } from "../config.js";

/**
 * Manages the test video modal UI, uploads, deletions, and backend restart prompts.
 */
import {
    closeOnBackdropClick,
    closeOnEscape,
    createElement,
    getOrCreateModalElements,
    hideModal,
    showModal,
} from "../ui/modal.js";
import { confirmDialog } from "../ui/confirmationDialog.js";
import {
    showDanger,
    showSuccess,
    showUploadToast,
    showWarning,
} from "../ui/notificationSystem.js";
import { uploadWithProgress } from "../ui/uploadWithProgress.js";

const OVERLAY_ID = "testVideoManagerOverlay";
const MODAL_ID = "testVideoManagerModal";

let videos = [];
let restartRequired = false;
let initialized = false;

/**
 * Gets or creates the modal overlay and dialog elements for the test video manager.
 */
function getOverlayElements() {
    return getOrCreateModalElements({
        overlayId: OVERLAY_ID,
        modalId: MODAL_ID,
        modalClassName:
            "bg-[#1a1a1a] rounded-lg shadow-xl max-w-3xl w-full mx-4 max-h-[90vh] flex flex-col border border-[#414141]",
    });
}

/**
 * Formats a byte count into a human-readable file size string.
 */
function formatFileSize(bytes) {
    if (!Number.isFinite(bytes) || bytes <= 0) {
        return "0 B";
    }
    const unitSize = 1024;
    const units = ["B", "KB", "MB", "GB"];
    const unitIndex = Math.min(
        Math.floor(Math.log(bytes) / Math.log(unitSize)),
        units.length - 1,
    );
    return `${Math.round((bytes / Math.pow(unitSize, unitIndex)) * 100) / 100} ${units[unitIndex]}`;
}

/**
 * Formats a Unix timestamp into a localized date string.
 */
function formatDate(timestamp) {
    if (!Number.isFinite(timestamp)) {
        return "Unknown";
    }
    return new Date(timestamp * 1000).toLocaleString();
}

/**
 * Fetches JSON from the backend and throws a normalized error on failure.
 */
async function fetchJson(path, options = {}) {
    const response = await fetch(`${BACKEND_BASE_URL}${path}`, options);
    let payload = {};
    try {
        payload = await response.json();
    } catch {
        payload = {};
    }
    if (!response.ok) {
        const error = new Error(
            payload.error || `Request failed: ${response.status}`,
        );
        error.status = response.status;
        error.payload = payload;
        throw error;
    }
    return payload;
}

/**
 * Marks the UI and backend as requiring a restart.
 */
async function markRestartRequired() {
    restartRequired = true;
    render();

    try {
        await fetch(`${BACKEND_BASE_URL}/set_restart_required`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ required: true }),
        });
    } catch (error) {
        console.warn("Failed to mark backend restart required:", error);
        showWarning(
            "Restart is required, but the backend flag was not updated.",
        );
    }
}

/**
 * Requests a backend restart and reloads the page shortly after.
 */
async function restartBackend(button = null) {
    if (button) {
        button.disabled = true;
        button.textContent = "Restarting...";
    }

    try {
        await fetch(`${BACKEND_BASE_URL}/restart-backend`, {
            method: "POST",
        });
    } catch (error) {
        console.warn("Restart request failed or connection closed:", error);
    } finally {
        setTimeout(() => {
            globalThis.location.reload();
        }, 1000);
    }
}

/**
 * Confirms and requests a full host reboot (Linux only).
 *
 * @param {HTMLButtonElement | null} [button=null]
 */
async function rebootComputer(button = null) {
    const confirmed = await confirmDialog({
        title: "Reboot Computer?",
        message: "This will reboot the entire machine. Are you sure?",
        detail: "All processes will stop and the host will restart. Linux only.",
        confirmText: "Reboot",
        variant: "warning",
    });
    if (!confirmed) {
        return;
    }

    if (button) {
        button.disabled = true;
        button.textContent = "Rebooting...";
    }

    try {
        const response = await fetch(`${BACKEND_BASE_URL}/reboot-system`, {
            method: "POST",
        });
        let payload = {};
        try {
            payload = await response.json();
        } catch {
            payload = {};
        }
        if (!response.ok) {
            throw new Error(payload.error || payload.message || "Reboot failed");
        }
        showWarning("System reboot initiated. The page will become unavailable.");
    } catch (error) {
        console.error("Failed to reboot computer:", error);
        showDanger(error.message || "Failed to reboot computer");
        if (button) {
            button.disabled = false;
            button.textContent = "Reboot Computer";
        }
    }
}

/**
 * Loads the current test video list from the backend.
 */
async function loadVideos() {
    try {
        const payload = await fetchJson("/test-videos");
        videos = Array.isArray(payload.videos) ? payload.videos : [];
        render();
    } catch (error) {
        console.error("Failed to load test videos:", error);
        showDanger(error.payload?.error || "Failed to load test videos");
    }
}

/**
 * Uploads a test video, optionally overwriting an existing file.
 */
async function uploadVideo(file, overwrite = false) {
    const formData = new FormData();
    formData.append("file", file);
    if (overwrite) {
        formData.append("overwrite", "true");
    }

    const uploadToast = showUploadToast({
        label: `Uploading ${file.name}...`,
    });

    try {
        await uploadWithProgress({
            url: "/test-videos",
            formData,
            onProgress: uploadToast.setProgress,
        });
        uploadToast.complete("Test video uploaded.");
        await markRestartRequired();
        await loadVideos();
    } catch (error) {
        if (error.status === 409 && error.payload?.requires_overwrite) {
            uploadToast.dismiss();
            const shouldOverwrite = await confirmDialog({
                title: "Replace Test Video?",
                message: `"${error.payload.filename}" already exists. Replace it?`,
                confirmText: "Replace",
                variant: "warning",
            });
            if (shouldOverwrite) {
                await uploadVideo(file, true);
            }
            return;
        }

        console.error("Failed to upload test video:", error);
        uploadToast.fail(error.payload?.error || "Failed to upload test video");
    }
}

/**
 * Deletes a test video, optionally forcing removal when referenced.
 */
async function deleteVideo(filename, force = false) {
    const forceQuery = force ? "?force=true" : "";

    try {
        await fetchJson(
            `/test-videos/${encodeURIComponent(filename)}${forceQuery}`,
            { method: "DELETE" },
        );
        showSuccess("Test video deleted.");
        await markRestartRequired();
        await loadVideos();
    } catch (error) {
        if (error.status === 409 && error.payload?.requires_force) {
            const references = error.payload.pipeline_references || [];
            const referenceText =
                references.length > 0
                    ? `\n\nReferenced by: ${references.join(", ")}`
                    : "";
            const shouldDelete = await confirmDialog({
                title: "Delete Referenced Video?",
                message: `"${filename}" is used by configured pipelines.${referenceText}`,
                detail: "Delete it anyway?",
                confirmText: "Delete Anyway",
            });
            if (shouldDelete) {
                await deleteVideo(filename, true);
            }
            return;
        }

        console.error("Failed to delete test video:", error);
        showDanger(error.payload?.error || "Failed to delete test video");
    }
}

/**
 * Renders the test video list into the provided container.
 */
function renderVideoRows(container) {
    container.innerHTML = "";

    if (videos.length === 0) {
        container.appendChild(
            createElement("div", {
                className: "text-center text-[#ac8a2f] py-8",
                text: "No test videos uploaded.",
            }),
        );
        return;
    }

    videos.forEach((video) => {
        const references = Array.isArray(video.pipeline_references)
            ? video.pipeline_references
            : [];
        const referenceText =
            references.length > 0
                ? `Referenced by ${references.join(", ")}`
                : "No pipeline references";

        const fileInfo = createElement("div", { className: "flex-1 min-w-0" }, [
            createElement("div", {
                className: "text-white font-medium truncate",
                text: video.filename,
                title: video.filename,
            }),
            createElement("div", {
                className: "text-xs text-[#ac8a2f] mt-1",
                text: `bus_id: ${video.bus_id} | ${formatFileSize(video.size)} | ${formatDate(video.modified)}`,
            }),
            createElement("div", {
                className:
                    references.length > 0
                        ? "text-xs text-yellow-300 mt-1"
                        : "text-xs text-gray-400 mt-1",
                text: referenceText,
            }),
        ]);

        const deleteButton = createElement("button", {
            type: "button",
            className:
                "px-3 py-1 bg-red-700 text-white rounded-md hover:bg-red-600 text-sm disabled:opacity-60",
            text: "Delete",
            onclick: async () => {
                const shouldDelete = await confirmDialog({
                    title: "Delete Test Video?",
                    message: `Delete "${video.filename}"?`,
                    detail: "This action cannot be undone.",
                    confirmText: "Delete",
                });
                if (shouldDelete) {
                    deleteVideo(video.filename);
                }
            },
        });

        const row = createElement(
            "div",
            {
                className:
                    "flex items-center justify-between gap-3 p-3 border-b border-[#414141] hover:bg-[#232323]",
            },
            [fileInfo, deleteButton],
        );

        container.appendChild(row);
    });
}

/**
 * Renders the test video manager modal contents.
 */
function render() {
    const { modal } = getOverlayElements();
    modal.innerHTML = "";

    const closeButton = createElement("button", {
        type: "button",
        className: "absolute top-4 right-4 text-[#ac8a2f] hover:text-white",
        text: "x",
        onclick: close,
        style: "font-size: 1.5rem; line-height: 1;",
    });

    const header = createElement(
        "div",
        {
            className: "p-6 border-b border-[#414141] relative",
        },
        [
            createElement("h2", {
                className: "text-xl font-bold text-[#f9c845]",
                text: "Manage Test Videos",
            }),
            createElement("p", {
                className: "text-sm text-gray-300 mt-2",
                text: "Uploaded MP4 files become selectable camera sources after a backend restart.",
            }),
            closeButton,
        ],
    );

    const bodyChildren = [];

    if (restartRequired) {
        bodyChildren.push(
            createElement("div", {
                className:
                    "mb-4 p-3 rounded-md border border-yellow-500 bg-yellow-900 bg-opacity-40 text-yellow-100 text-sm",
                text: "Backend restart required before test video changes appear as camera sources.",
            }),
        );
    }

    const fileInput = createElement("input", {
        type: "file",
        accept: ".mp4,video/mp4",
        className:
            "w-full text-white file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-[#f9c845] file:text-[#232323] hover:file:bg-[#d4a83a]",
    });
    fileInput.addEventListener("change", (event) => {
        const file = event.target.files?.[0];
        if (file) {
            uploadVideo(file);
            event.target.value = "";
        }
    });

    bodyChildren.push(fileInput);

    const listContainer = createElement("div", {
        id: "testVideoManagerList",
        className:
            "mt-6 border border-[#414141] rounded-lg bg-[#1f1f1f] max-h-96 overflow-y-auto",
    });
    bodyChildren.push(listContainer);

    const body = createElement(
        "div",
        {
            className: "p-6 flex-1 overflow-y-auto",
        },
        bodyChildren,
    );

    const restartButton = createElement("button", {
        type: "button",
        className:
            "px-4 py-2 bg-red-900 text-white rounded-md border border-red-700 hover:bg-red-800 disabled:opacity-60",
        text: "Restart Backend",
        onclick: (event) => restartBackend(event.currentTarget),
    });

    const footer = createElement(
        "div",
        {
            className: "p-6 border-t border-[#414141] flex justify-end gap-3",
        },
        [
            createElement("button", {
                type: "button",
                className:
                    "px-4 py-2 bg-[#414141] text-white rounded-md hover:bg-[#515151]",
                text: "Close",
                onclick: close,
            }),
            restartButton,
        ],
    );

    modal.appendChild(header);
    modal.appendChild(body);
    modal.appendChild(footer);

    renderVideoRows(listContainer);
}

/**
 * Opens the test video manager modal and loads the latest videos.
 */
function open() {
    const { overlay } = getOverlayElements();
    render();
    showModal(overlay);
    loadVideos();
}

/**
 * Closes the test video manager modal.
 */
function close() {
    const { overlay } = getOverlayElements();
    hideModal(overlay);
}

/**
 * Initializes event wiring and global access for the test video manager.
 */
export function initializeTestVideoManager() {
    if (initialized) {
        return;
    }
    initialized = true;

    const { overlay } = getOverlayElements();
    closeOnBackdropClick(overlay, close);
    closeOnEscape(overlay, close);

    const manageButton = document.getElementById("manageTestVideosBtn");
    if (manageButton) {
        manageButton.addEventListener("click", open);
    }

    const rebootComputerButton = document.getElementById("rebootComputerBtn");
    if (rebootComputerButton) {
        rebootComputerButton.addEventListener("click", () => {
            void rebootComputer(rebootComputerButton);
        });
    }

    globalThis.TestVideoManager = {
        open,
        close,
        loadVideos,
    };
    globalThis.restartBackend = () => restartBackend();
    globalThis.rebootComputer = () => rebootComputer();
}
