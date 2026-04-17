import { BACKEND_BASE_URL } from "../config.js";
import { populateFieldDropdown } from "../dropdown/fieldDropdown.js";
import { populateRobotDropdown } from "../dropdown/robotDropdown.js";
import { apply3DAssetScale } from "../init3DView.js";
import { showDanger, showSuccess } from "../ui/notificationSystem.js";

const OVERLAY_ID = "assetFileManagerOverlay";
const MODAL_ID = "assetFileManagerModal";
const SCALE_POPUP_ID = "assetScalePopup";
const ASSET_TYPES = {
    robot: {
        label: "Robot Files",
        uploadLabel: "Upload Robot GLB",
        endpoint: "/robot-files",
        emptyText: "No robot files uploaded.",
    },
    field: {
        label: "Field Files",
        uploadLabel: "Upload Field GLB",
        endpoint: "/field-files",
        emptyText: "No field files available.",
    },
};

let initialized = false;
let activeType = "robot";
let robotFiles = [];
let fieldFiles = [];
let selectedFieldYear = "";
let selectedFieldApriltagMapFile = null;

function createElement(tag, attrs = {}, children = []) {
    const el = document.createElement(tag);
    Object.entries(attrs).forEach(([key, value]) => {
        if (key === "className") {
            el.className = value;
        } else if (key === "text") {
            el.textContent = value;
        } else if (key === "html") {
            el.innerHTML = value;
        } else if (key.startsWith("on") && typeof value === "function") {
            el.addEventListener(key.substring(2).toLowerCase(), value);
        } else if (value !== undefined && value !== null) {
            el.setAttribute(key, String(value));
        }
    });
    children.forEach((child) => el.appendChild(child));
    return el;
}

function getOverlayElements() {
    let overlay = document.getElementById(OVERLAY_ID);
    let modal = document.getElementById(MODAL_ID);

    if (!overlay) {
        overlay = createElement("div", {
            id: OVERLAY_ID,
            className:
                "fixed inset-0 z-50 hidden flex items-center justify-center",
            style: "background-color: rgba(0, 0, 0, 0.25); backdrop-filter: blur(6px);",
        });
        document.body.appendChild(overlay);
    }

    if (!modal) {
        modal = createElement("div", {
            id: MODAL_ID,
            className:
                "bg-[#1a1a1a] rounded-lg shadow-xl max-w-4xl w-full mx-4 max-h-[90vh] flex flex-col border border-[#414141]",
        });
        overlay.appendChild(modal);
    }

    return { overlay, modal };
}

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

function formatDate(timestamp) {
    if (!Number.isFinite(timestamp)) {
        return "Unknown";
    }
    return new Date(timestamp * 1000).toLocaleString();
}

function normalizeScale(scale) {
    const numericScale = Number.parseFloat(scale);
    return Number.isFinite(numericScale) && numericScale > 0 ? numericScale : 1;
}

function formatScale(scale) {
    return normalizeScale(scale).toLocaleString(undefined, {
        maximumFractionDigits: 6,
    });
}

function getCurrentFieldYear() {
    const yearSelect = document.getElementById("yearSelect");
    if (yearSelect && yearSelect.selectedIndex > 0) {
        return yearSelect.value;
    }
    return "";
}

function scaleEndpoint(file) {
    if (activeType === "robot") {
        return `${ASSET_TYPES.robot.endpoint}/${encodeURIComponent(file.filename)}/scale`;
    }

    return `${ASSET_TYPES.field.endpoint}/${encodeURIComponent(file.year)}/${encodeURIComponent(file.filename)}/scale`;
}

function is3DViewActive() {
    const view = document.getElementById("view-3d");
    return Boolean(view && !view.classList.contains("hidden"));
}

function getSelectableFieldYears() {
    const uploadedYears = fieldFiles
        .map((file) => String(file.year))
        .filter((year) => /^\d+$/.test(year))
        .map((year) => Number.parseInt(year, 10))
        .filter((year) => Number.isInteger(year));

    if (uploadedYears.length === 0) {
        const currentYear = new Date().getFullYear();
        return [currentYear - 1, currentYear, currentYear + 1].map(String);
    }

    const minYear = Math.min(...uploadedYears);
    const maxYear = Math.max(...uploadedYears);
    const years = [];
    for (let year = minYear - 1; year <= maxYear + 1; year += 1) {
        years.push(String(year));
    }
    return years;
}

function normalizeSelectedFieldYear() {
    const selectableYears = getSelectableFieldYears();
    if (selectableYears.includes(selectedFieldYear)) {
        return;
    }

    const currentFieldYear = getCurrentFieldYear();
    if (selectableYears.includes(currentFieldYear)) {
        selectedFieldYear = currentFieldYear;
        return;
    }

    selectedFieldYear =
        fieldFiles.find((file) => selectableYears.includes(file.year))?.year ||
        selectableYears[Math.min(1, selectableYears.length - 1)] ||
        "";
}

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

async function loadAssets() {
    try {
        const [robotPayload, fieldPayload] = await Promise.all([
            fetchJson(ASSET_TYPES.robot.endpoint),
            fetchJson(ASSET_TYPES.field.endpoint),
        ]);
        robotFiles = Array.isArray(robotPayload.file_details)
            ? robotPayload.file_details
            : [];
        fieldFiles = Array.isArray(fieldPayload.file_details)
            ? fieldPayload.file_details
            : [];

        if (!selectedFieldYear) {
            selectedFieldYear =
                getCurrentFieldYear() ||
                fieldFiles[0]?.year ||
                new Date().getFullYear().toString();
        }
        normalizeSelectedFieldYear();

        render();
    } catch (error) {
        console.error("Failed to load 3D assets:", error);
        showDanger(error.payload?.error || "Failed to load 3D asset files");
    }
}

async function refresh3DAssetDropdowns(selectedFilename = null) {
    if (activeType === "robot") {
        await populateRobotDropdown(selectedFilename);
        return;
    }

    await populateFieldDropdown({
        selectedYear: selectedFieldYear,
        selectedFile: selectedFilename,
        loadSelected: is3DViewActive(),
    });
}

async function uploadAsset(file, overwrite = false) {
    const formData = new FormData();
    formData.append("file", file);
    if (activeType === "field") {
        formData.append("year", selectedFieldYear);
        if (selectedFieldApriltagMapFile) {
            formData.append("apriltag_map", selectedFieldApriltagMapFile);
        }
    }
    if (overwrite) {
        formData.append("overwrite", "true");
    }

    try {
        const payload = await fetchJson(ASSET_TYPES[activeType].endpoint, {
            method: "POST",
            body: formData,
        });
        showSuccess(
            `${activeType === "robot" ? "Robot" : "Field"} file uploaded.`,
        );
        selectedFieldApriltagMapFile = null;
        await loadAssets();
        await refresh3DAssetDropdowns(payload.file?.filename || file.name);
    } catch (error) {
        if (error.status === 409 && error.payload?.requires_overwrite) {
            const shouldOverwrite = globalThis.confirm(
                `"${error.payload.filename}" already exists. Replace it?`,
            );
            if (shouldOverwrite) {
                await uploadAsset(file, true);
            }
            return;
        }

        console.error("Failed to upload 3D asset:", error);
        showDanger(error.payload?.error || "Failed to upload 3D asset file");
    }
}

async function deleteAsset(file) {
    const path =
        activeType === "robot"
            ? `${ASSET_TYPES.robot.endpoint}/${encodeURIComponent(file.filename)}`
            : `${ASSET_TYPES.field.endpoint}/${encodeURIComponent(file.year)}/${encodeURIComponent(file.filename)}`;

    try {
        await fetchJson(path, { method: "DELETE" });
        showSuccess(
            `${activeType === "robot" ? "Robot" : "Field"} file deleted.`,
        );
        await loadAssets();
        await refresh3DAssetDropdowns();
    } catch (error) {
        console.error("Failed to delete 3D asset:", error);
        showDanger(error.payload?.error || "Failed to delete 3D asset file");
    }
}

async function saveAssetScale(file, scale) {
    try {
        const payload = await fetchJson(scaleEndpoint(file), {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ scale }),
        });
        const savedScale = normalizeScale(payload.file?.scale ?? scale);
        showSuccess(
            `${activeType === "robot" ? "Robot" : "Field"} scale saved.`,
        );
        apply3DAssetScale(
            activeType,
            {
                year: file.year,
                filename: file.filename,
            },
            savedScale,
        );
        await loadAssets();
        await refresh3DAssetDropdowns(file.filename);
        return true;
    } catch (error) {
        console.error("Failed to save 3D asset scale:", error);
        showDanger(error.payload?.error || "Failed to save 3D asset scale");
        return false;
    }
}

function openScalePopup(file) {
    const existingPopup = document.getElementById(SCALE_POPUP_ID);
    if (existingPopup) {
        existingPopup.remove();
    }

    const popup = createElement("div", {
        id: SCALE_POPUP_ID,
        className: "fixed inset-0 z-[60] flex items-center justify-center px-4",
        style: "background-color: rgba(0, 0, 0, 0.35); backdrop-filter: blur(6px);",
    });
    popup.addEventListener("click", (event) => {
        if (event.target.id === SCALE_POPUP_ID) {
            popup.remove();
        }
    });

    const closeButton = createElement("button", {
        type: "button",
        className: "absolute top-4 right-4 text-[#ac8a2f] hover:text-white",
        text: "x",
        onclick: () => popup.remove(),
        style: "font-size: 1.5rem; line-height: 1;",
    });

    const input = createElement("input", {
        id: "assetScale",
        name: "scale",
        type: "number",
        step: "any",
        value: formatScale(file.scale),
        className:
            "bg-[#2a2a2a] border border-[#414141] text-white rounded-md px-3 py-2 focus:outline-none focus:border-[#f9c845]",
    });

    const form = createElement(
        "form",
        {
            className:
                "bg-[#1a1a1a] rounded-lg shadow-xl max-w-2xl w-full border border-[#414141]",
            onsubmit: async (event) => {
                event.preventDefault();
                try {
                    const scale = Number.parseFloat(input.value);
                    if (!Number.isFinite(scale) || scale <= 0) {
                        throw new Error("Scale must be a positive number.");
                    }
                    const saved = await saveAssetScale(file, scale);
                    if (saved) {
                        popup.remove();
                    }
                } catch (error) {
                    showDanger(error.message || "Invalid scale value.");
                }
            },
        },
        [
            createElement(
                "div",
                {
                    className: "p-6 border-b border-[#414141] relative",
                },
                [
                    createElement("h3", {
                        className: "text-xl font-bold text-[#f9c845]",
                        text: `Scale ${file.filename}`,
                    }),
                    createElement("p", {
                        className: "text-sm text-gray-300 mt-2",
                        text: "Enter a positive scale factor.",
                    }),
                    closeButton,
                ],
            ),
            createElement(
                "div",
                {
                    className: "p-6 flex flex-col gap-5",
                },
                [
                    createElement(
                        "label",
                        {
                            className:
                                "flex flex-col gap-2 text-sm text-[#f9c845] font-medium",
                        },
                        [
                            createElement("span", { text: "Scale factor" }),
                            input,
                        ],
                    ),
                ],
            ),
            createElement(
                "div",
                {
                    className:
                        "p-6 border-t border-[#414141] flex justify-end gap-3",
                },
                [
                    createElement("button", {
                        type: "button",
                        className:
                            "px-4 py-2 bg-[#414141] text-white rounded-md hover:bg-[#515151]",
                        text: "Cancel",
                        onclick: () => popup.remove(),
                    }),
                    createElement("button", {
                        type: "submit",
                        className:
                            "px-4 py-2 rounded-md bg-[#f9c845] text-[#232323] font-semibold hover:bg-[#d4a83a]",
                        text: "Save",
                    }),
                ],
            ),
        ],
    );

    popup.appendChild(form);
    document.body.appendChild(popup);
    input.focus();
    input.select();
}

function renderTabs() {
    return createElement(
        "div",
        {
            className: "flex gap-2 mt-4",
        },
        Object.entries(ASSET_TYPES).map(([type, config]) =>
            createElement("button", {
                type: "button",
                className:
                    activeType === type
                        ? "px-4 py-2 rounded-md bg-[#f9c845] text-[#232323] font-semibold"
                        : "px-4 py-2 rounded-md bg-[#2a2a2a] text-[#f9c845] border border-[#414141] hover:bg-[#3a3a3a]",
                text: config.label,
                onclick: () => {
                    activeType = type;
                    render();
                },
            }),
        ),
    );
}

function renderUploadControls() {
    const controls = [];

    if (activeType === "field") {
        const yearSelect = createElement("select", {
            className:
                "bg-[#2a2a2a] border border-[#414141] text-white rounded-md px-3 py-2 focus:outline-none focus:border-[#f9c845]",
            onchange: (event) => {
                selectedFieldYear = event.currentTarget.value;
                render();
            },
        });

        getSelectableFieldYears().forEach((year) => {
            yearSelect.appendChild(
                createElement("option", {
                    value: year,
                    text: year,
                }),
            );
        });
        yearSelect.value = selectedFieldYear;

        controls.push(
            createElement(
                "label",
                {
                    className:
                        "flex flex-col gap-2 text-sm text-[#f9c845] font-medium",
                },
                [createElement("span", { text: "Field Year" }), yearSelect],
            ),
        );
    }

    if (activeType === "field") {
        const apriltagMapInput = createElement("input", {
            type: "file",
            accept: ".fmap,.json,application/json",
            className:
                "w-full text-white file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-[#f9c845] file:text-[#232323] hover:file:bg-[#d4a83a]",
        });
        apriltagMapInput.addEventListener("change", (event) => {
            selectedFieldApriltagMapFile = event.target.files?.[0] || null;
        });

        controls.push(
            createElement(
                "label",
                {
                    className:
                        "flex flex-col gap-2 text-sm text-[#f9c845] font-medium",
                },
                [
                    createElement("span", {
                        text: "AprilTag fmap",
                    }),
                    apriltagMapInput,
                    createElement("span", {
                        className: "text-xs text-gray-400",
                        text: "Optional. Select before choosing the field GLB to upload both together.",
                    }),
                ],
            ),
        );
    }

    const fileInput = createElement("input", {
        type: "file",
        accept: ".glb,model/gltf-binary",
        className:
            "w-full text-white file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-[#f9c845] file:text-[#232323] hover:file:bg-[#d4a83a]",
    });
    fileInput.addEventListener("change", (event) => {
        const file = event.target.files?.[0];
        if (file) {
            uploadAsset(file);
            event.target.value = "";
        }
    });

    controls.push(fileInput);

    return createElement(
        "div",
        {
            className: "flex flex-col gap-4",
        },
        controls,
    );
}

function renderFileRows(container) {
    container.innerHTML = "";

    const files =
        activeType === "robot"
            ? robotFiles
            : fieldFiles.filter((file) => file.year === selectedFieldYear);
    if (files.length === 0) {
        container.appendChild(
            createElement("div", {
                className: "text-center text-[#ac8a2f] py-8",
                text:
                    activeType === "field"
                        ? `No field files for ${selectedFieldYear}.`
                        : ASSET_TYPES[activeType].emptyText,
            }),
        );
        return;
    }

    files.forEach((file) => {
        const detailText =
            activeType === "field"
                ? `${file.year} | ${formatFileSize(file.size)} | ${formatDate(file.modified)}`
                : `${formatFileSize(file.size)} | ${formatDate(file.modified)}`;
        const scaleText = `Scale: ${formatScale(file.scale)}`;
        const apriltagMapText =
            activeType === "field"
                ? `AprilTag map: ${file.apriltag_map?.filename || "none"}`
                : "";

        const fileInfo = createElement("div", { className: "flex-1 min-w-0" }, [
            createElement("div", {
                className: "text-white font-medium truncate",
                text: file.filename,
                title: file.filename,
            }),
            createElement("div", {
                className: "text-xs text-[#ac8a2f] mt-1",
                text: detailText,
            }),
            createElement("div", {
                className: "text-xs text-gray-300 mt-1",
                text: scaleText,
            }),
            ...(activeType === "field"
                ? [
                      createElement("div", {
                          className: file.apriltag_map
                              ? "text-xs text-green-300 mt-1"
                              : "text-xs text-gray-400 mt-1",
                          text: apriltagMapText,
                      }),
                  ]
                : []),
        ]);

        const settingsButton = createElement(
            "button",
            {
                type: "button",
                className:
                    "p-2 bg-[#2a2a2a] border border-[#414141] text-white rounded-md hover:border-[#f9c845] hover:bg-[#3a3a3a]",
                title: "Change scale",
                onclick: () => openScalePopup(file),
            },
            [
                createElement("img", {
                    src: "./assets/settings.svg",
                    alt: "Scale settings",
                    className: "w-4 h-4",
                    style: "filter: grayscale(100%);",
                }),
            ],
        );

        const deleteButton = createElement("button", {
            type: "button",
            className:
                "px-3 py-1 bg-red-700 text-white rounded-md hover:bg-red-600 text-sm",
            text: "Delete",
            onclick: () => {
                const shouldDelete = globalThis.confirm(
                    `Delete "${file.filename}"?`,
                );
                if (shouldDelete) {
                    deleteAsset(file);
                }
            },
        });

        container.appendChild(
            createElement(
                "div",
                {
                    className:
                        "flex items-center justify-between gap-3 p-3 border-b border-[#414141] hover:bg-[#232323]",
                },
                [fileInfo, settingsButton, deleteButton],
            ),
        );
    });
}

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
                text: "Manage Robot and Field Files",
            }),
            createElement("p", {
                className: "text-sm text-gray-300 mt-2",
                text: "Upload GLB files here. Draco compression is prepared automatically before files are served in the 3D view.",
            }),
            renderTabs(),
            closeButton,
        ],
    );

    const listContainer = createElement("div", {
        id: "assetFileManagerList",
        className:
            "mt-6 border border-[#414141] rounded-lg bg-[#1f1f1f] max-h-96 overflow-y-auto",
    });

    const body = createElement(
        "div",
        {
            className: "p-6 flex-1 overflow-y-auto",
        },
        [
            createElement("h3", {
                className: "text-lg font-medium text-[#f9c845] mb-3",
                text:
                    activeType === "field"
                        ? `${ASSET_TYPES[activeType].uploadLabel} for ${selectedFieldYear}`
                        : ASSET_TYPES[activeType].uploadLabel,
            }),
            renderUploadControls(),
            listContainer,
        ],
    );

    const footer = createElement(
        "div",
        {
            className: "p-6 border-t border-[#414141] flex justify-end",
        },
        [
            createElement("button", {
                type: "button",
                className:
                    "px-4 py-2 bg-[#414141] text-white rounded-md hover:bg-[#515151]",
                text: "Close",
                onclick: close,
            }),
        ],
    );

    modal.appendChild(header);
    modal.appendChild(body);
    modal.appendChild(footer);

    renderFileRows(listContainer);
}

function open() {
    const { overlay } = getOverlayElements();
    selectedFieldYear =
        selectedFieldYear ||
        getCurrentFieldYear() ||
        new Date().getFullYear().toString();
    normalizeSelectedFieldYear();
    render();
    overlay.classList.remove("hidden");
    loadAssets();
}

function close() {
    const { overlay } = getOverlayElements();
    overlay.classList.add("hidden");
}

export function initializeAssetFileManager() {
    if (initialized) {
        return;
    }
    initialized = true;

    const { overlay } = getOverlayElements();
    overlay.addEventListener("click", (event) => {
        if (event.target.id === OVERLAY_ID) {
            close();
        }
    });

    const manageButton = document.getElementById("manage3DAssetsBtn");
    if (manageButton) {
        manageButton.addEventListener("click", open);
    }

    globalThis.AssetFileManager = {
        open,
        close,
        loadAssets,
    };
}
