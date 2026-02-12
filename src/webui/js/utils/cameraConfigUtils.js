import { BACKEND_BASE_URL } from "../config.js";
import { showDanger, showSuccess, showWarning } from "../ui/notificationSystem.js";

const EXTRINSICS_KEYS = [
    "horizontal_fov",
    "vertical_fov",
    "pitch",
    "yaw",
    "roll",
    "x_offset",
    "y_offset",
    "z_offset",
];

const INPUT_ID_BY_KEY = {
    horizontal_fov: "utils-horizontal-fov",
    vertical_fov: "utils-vertical-fov",
    pitch: "utils-pitch",
    yaw: "utils-yaw",
    roll: "utils-roll",
    x_offset: "utils-x-offset",
    y_offset: "utils-y-offset",
    z_offset: "utils-z-offset",
};

let initialized = false;
let currentCameraBusId = "";

function getElement(id) {
    return document.getElementById(id);
}

function getExtrinsicsPayload() {
    const payload = {};
    EXTRINSICS_KEYS.forEach((key) => {
        const input = getElement(INPUT_ID_BY_KEY[key]);
        const numericValue = Number.parseFloat(input?.value ?? "0");
        payload[key] = Number.isFinite(numericValue) ? numericValue : 0;
    });
    return payload;
}

function setExtrinsicsInputs(extrinsics = {}) {
    EXTRINSICS_KEYS.forEach((key) => {
        const input = getElement(INPUT_ID_BY_KEY[key]);
        if (!input) {
            return;
        }
        const value = extrinsics[key];
        input.value = Number.isFinite(value) ? String(value) : "0";
    });
}

function setIntrinsicsStatus(text) {
    const status = getElement("utilsIntrinsicsStatus");
    if (status) {
        status.textContent = text;
    }
}

function setCameraMeta(camera) {
    const meta = getElement("utilsCameraMeta");
    if (!meta) {
        return;
    }

    if (!camera) {
        meta.textContent = "";
        return;
    }

    meta.textContent = `Selected: ${camera.name} (bus_id: ${camera.bus_id})`;
}

async function fetchJson(path, options = {}) {
    const response = await fetch(`${BACKEND_BASE_URL}${path}`, options);
    let data = null;
    try {
        data = await response.json();
    } catch {
        data = null;
    }

    if (!response.ok) {
        const errorMessage = data?.error || data?.message || `Request failed: ${response.status}`;
        throw new Error(errorMessage);
    }

    return data;
}

async function loadCameraList() {
    const select = getElement("utilsCameraSelect");
    if (!select) {
        return;
    }

    try {
        const payload = await fetchJson("/camera-config/cameras");
        const cameras = Array.isArray(payload?.cameras) ? payload.cameras : [];

        select.innerHTML = "";
        if (cameras.length === 0) {
            const option = document.createElement("option");
            option.value = "";
            option.textContent = "No active cameras";
            select.appendChild(option);
            currentCameraBusId = "";
            setExtrinsicsInputs({});
            setIntrinsicsStatus("No camera selected.");
            setCameraMeta(null);
            return;
        }

        cameras.forEach((camera) => {
            const option = document.createElement("option");
            option.value = String(camera.bus_id);
            option.textContent = `${camera.name} (${camera.bus_id})`;
            select.appendChild(option);
        });

        if (!currentCameraBusId || !cameras.some((camera) => String(camera.bus_id) === currentCameraBusId)) {
            currentCameraBusId = String(cameras[0].bus_id);
        }

        select.value = currentCameraBusId;
        const selected = cameras.find((camera) => String(camera.bus_id) === currentCameraBusId);
        setCameraMeta(selected || null);
        await loadCameraConfig(currentCameraBusId);
    } catch (error) {
        showDanger(`Failed to load camera list: ${error.message}`);
    }
}

async function loadCameraConfig(cameraBusId) {
    if (!cameraBusId) {
        return;
    }

    try {
        const payload = await fetchJson(`/camera-config/${encodeURIComponent(cameraBusId)}`);
        setExtrinsicsInputs(payload?.extrinsics || {});
        if (payload?.intrinsics_exists) {
            setIntrinsicsStatus(`Current intrinsics: ${payload.intrinsics_path || "intrinsics.json"}`);
        } else {
            setIntrinsicsStatus("No intrinsics file currently set.");
        }
    } catch (error) {
        showDanger(`Failed to load camera config: ${error.message}`);
    }
}

async function saveExtrinsics() {
    if (!currentCameraBusId) {
        showWarning("Select a camera first");
        return;
    }

    try {
        await fetchJson(`/camera-config/${encodeURIComponent(currentCameraBusId)}/extrinsics`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(getExtrinsicsPayload()),
        });
        showSuccess("Camera extrinsics saved");
        await loadCameraConfig(currentCameraBusId);
    } catch (error) {
        showDanger(`Failed to save extrinsics: ${error.message}`);
    }
}

async function uploadIntrinsicsFile(file) {
    if (!currentCameraBusId) {
        showWarning("Select a camera first");
        return;
    }

    if (!file) {
        showWarning("Select a file first");
        return;
    }

    if (!file.name.toLowerCase().endsWith(".json")) {
        showWarning("Only .json files are supported for intrinsics");
        return;
    }

    try {
        const formData = new FormData();
        formData.append("file", file);
        await fetchJson(`/camera-config/${encodeURIComponent(currentCameraBusId)}/intrinsics`, {
            method: "POST",
            body: formData,
        });
        showSuccess("Intrinsics file uploaded");
        await loadCameraConfig(currentCameraBusId);
    } catch (error) {
        showDanger(`Failed to upload intrinsics: ${error.message}`);
    }
}

async function deleteIntrinsics() {
    if (!currentCameraBusId) {
        showWarning("Select a camera first");
        return;
    }

    try {
        await fetchJson(`/camera-config/${encodeURIComponent(currentCameraBusId)}/intrinsics`, {
            method: "DELETE",
        });
        showSuccess("Intrinsics file deleted");
        await loadCameraConfig(currentCameraBusId);
    } catch (error) {
        showDanger(`Failed to delete intrinsics: ${error.message}`);
    }
}

function setupDropzone() {
    const dropzone = getElement("utilsIntrinsicsDropzone");
    if (!dropzone) {
        return;
    }

    const setHover = (isHover) => {
        dropzone.classList.toggle("border-[#f9c845]", isHover);
        dropzone.classList.toggle("text-[#f9c845]", isHover);
    };

    dropzone.addEventListener("dragover", (event) => {
        event.preventDefault();
        setHover(true);
    });

    dropzone.addEventListener("dragleave", () => {
        setHover(false);
    });

    dropzone.addEventListener("drop", (event) => {
        event.preventDefault();
        setHover(false);
        const file = event.dataTransfer?.files?.[0];
        if (file) {
            void uploadIntrinsicsFile(file);
        }
    });
}

export function initCameraConfigUtils() {
    if (initialized) {
        return;
    }

    const cameraSelect = getElement("utilsCameraSelect");
    const saveButton = getElement("utilsSaveExtrinsicsBtn");
    const refreshButton = getElement("utilsRefreshConfigBtn");
    const uploadButton = getElement("utilsUploadIntrinsicsBtn");
    const deleteButton = getElement("utilsDeleteIntrinsicsBtn");
    const fileInput = getElement("utilsIntrinsicsFileInput");

    if (!cameraSelect) {
        return;
    }

    cameraSelect.addEventListener("change", () => {
        currentCameraBusId = cameraSelect.value;
        void loadCameraConfig(currentCameraBusId);
        const selectedText = cameraSelect.options[cameraSelect.selectedIndex]?.text || "";
        setCameraMeta(
            currentCameraBusId
                ? {
                      name: selectedText.split(" (")[0] || selectedText,
                      bus_id: currentCameraBusId,
                  }
                : null,
        );
    });

    saveButton?.addEventListener("click", () => {
        void saveExtrinsics();
    });

    refreshButton?.addEventListener("click", () => {
        void loadCameraList();
    });

    uploadButton?.addEventListener("click", () => {
        fileInput?.click();
    });

    fileInput?.addEventListener("change", () => {
        const file = fileInput.files?.[0];
        if (file) {
            void uploadIntrinsicsFile(file);
            fileInput.value = "";
        }
    });

    deleteButton?.addEventListener("click", () => {
        void deleteIntrinsics();
    });

    setupDropzone();
    initialized = true;
}

export function refreshCameraConfigUtils() {
    void loadCameraList();
}
