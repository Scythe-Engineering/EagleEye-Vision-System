// Live camera hardware controls and the AprilTag detection preview on the Utils tab.
import { BACKEND_BASE_URL } from "../config.js";
import {
    showDanger,
    showSuccess,
    showWarning,
} from "../ui/notificationSystem.js";
import { subscribeConnectionStatus } from "../ui/connectionStatus.js";
import { fetchJson } from "./fetchJson.js";

// Slider drags send at most one request per interval.
const APPLY_DELAY_MS = 120;

let initialized = false;
let busId = "";
let streamName = "";
let feedActive = false;
let overlayOn = false;
let controls = [];
let saved = {};
let pendingValues = {};
let applyTimer = null;
let requestChain = Promise.resolve();
let loadFailed = false;
let loadGeneration = 0;
let backendConnected = false;

const getElement = (id) => document.getElementById(id);
const controlsPath = (suffix = "") =>
    `/camera-config/${encodeURIComponent(busId)}/controls${suffix}`;

/**
 * Point the live view at the raw camera stream or the AprilTag preview.
 */
function updateFeed() {
    const image = getElement("utilsCameraSettingsFeed");
    if (!image) return;
    let url = "";
    if (feedActive && backendConnected && busId) {
        url = overlayOn
            ? `${BACKEND_BASE_URL}/camera-config/${encodeURIComponent(busId)}/apriltag/feed`
            : streamName
              ? `${BACKEND_BASE_URL}/feed/${encodeURIComponent(streamName)}`
              : "";
    }
    if (image.dataset.src === url) return;
    image.dataset.src = url;
    // Dropping src closes the MJPEG connection, which browsers limit per host.
    if (url) image.src = url;
    else image.removeAttribute("src");
}

/**
 * Stop the live view while the Utils tab is hidden.
 */
export function pauseCameraSettingsFeed() {
    feedActive = false;
    updateFeed();
}

/**
 * Start the live view when the Utils tab is shown.
 */
export function resumeCameraSettingsFeed() {
    feedActive = true;
    updateFeed();
}

function setOverlay(enabled) {
    overlayOn = enabled;
    const button = getElement("utilsAprilTagOverlayBtn");
    if (button) {
        button.textContent = enabled
            ? "Hide AprilTag Detections"
            : "Show AprilTag Detections";
        button.setAttribute("aria-pressed", String(enabled));
    }
    updateFeed();
}

/**
 * Show a control value the way the driver names it.
 * @param {object} control - Control description from the backend.
 * @param {number} value - Raw control value.
 * @returns {string}
 */
function valueLabel(control, value) {
    if (control.type === "boolean") return value ? "on" : "off";
    const item = control.menu.find((entry) => entry.value === value);
    return item ? String(item.label) : String(value);
}

function controlHint(control) {
    const parts = [`default ${valueLabel(control, control.default)}`];
    if (control.key in saved) {
        parts.push(`saved ${valueLabel(control, saved[control.key])}`);
    }
    if (control.inactive) parts.push("set by an auto mode");
    if (control.read_only) parts.push("read-only");
    return parts.join(" · ");
}

function queueValue(key, value) {
    pendingValues[key] = value;
    clearTimeout(applyTimer);
    applyTimer = setTimeout(flushPendingValues, APPLY_DELAY_MS);
}

function flushPendingValues() {
    clearTimeout(applyTimer);
    const values = pendingValues;
    pendingValues = {};
    if (Object.keys(values).length > 0) {
        void sendValues(values);
    }
    return requestChain;
}

/**
 * Build the input row for one control.
 * @param {object} control - Control description from the backend.
 * @returns {HTMLElement}
 */
function buildControlRow(control) {
    const row = document.createElement("div");
    row.dataset.key = control.key;
    row.className = "text-sm text-[#f9c845]";

    const title = document.createElement("div");
    title.className = "flex items-center justify-between gap-2";
    const name = document.createElement("span");
    name.textContent = control.name;
    title.appendChild(name);
    row.appendChild(title);

    const inputClass =
        "bg-[#232323] border border-[#414141] text-white rounded-md px-2 py-1 disabled:opacity-50";
    const inputs = [];
    if (control.type === "boolean") {
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.className = "h-4 w-4 accent-[#f9c845]";
        checkbox.addEventListener("change", () =>
            queueValue(control.key, checkbox.checked ? 1 : 0),
        );
        title.appendChild(checkbox);
        inputs.push(checkbox);
    } else if (control.type === "menu") {
        const select = document.createElement("select");
        select.className = `mt-1 w-full ${inputClass}`;
        for (const item of control.menu) {
            const option = document.createElement("option");
            option.value = String(item.value);
            option.textContent = String(item.label);
            select.appendChild(option);
        }
        select.addEventListener("change", () =>
            queueValue(control.key, Number(select.value)),
        );
        row.appendChild(select);
        inputs.push(select);
    } else {
        const line = document.createElement("div");
        line.className = "mt-1 flex items-center gap-2";
        const slider = document.createElement("input");
        slider.type = "range";
        slider.className = "flex-1 accent-[#f9c845] disabled:opacity-50";
        const number = document.createElement("input");
        number.type = "number";
        number.className = `w-24 ${inputClass}`;
        for (const input of [slider, number]) {
            input.min = String(control.minimum);
            input.max = String(control.maximum);
            input.step = String(control.step);
        }
        slider.addEventListener("input", () => {
            number.value = slider.value;
            queueValue(control.key, Number(slider.value));
        });
        number.addEventListener("change", () => {
            slider.value = number.value;
            queueValue(control.key, Number(number.value));
        });
        line.append(slider, number);
        row.appendChild(line);
        inputs.push(slider, number);
    }
    for (const input of inputs) input.setAttribute("aria-label", control.name);

    const hint = document.createElement("p");
    hint.className = "mt-1 text-xs text-[#ac8a2f]";
    hint.dataset.role = "hint";
    row.appendChild(hint);
    row.inputs = inputs;
    return row;
}

/**
 * Show a control's current value and state in its row.
 * @param {HTMLElement} row - Row built by buildControlRow.
 * @param {object} control - Control description from the backend.
 */
function syncControlRow(row, control) {
    const editing = control.key in pendingValues;
    for (const input of row.inputs) {
        input.disabled = control.inactive || control.read_only;
        // Leave a value alone while the user is still changing it.
        if (editing || control.value === null) continue;
        if (input.type === "checkbox") input.checked = Boolean(control.value);
        else input.value = String(control.value);
    }
    row.querySelector("[data-role='hint']").textContent = controlHint(control);
}

/**
 * Render the latest control list, reusing rows when the set is unchanged.
 * @param {{controls?: object[], saved?: object}} payload - Backend response.
 */
function showControls(payload) {
    const container = getElement("utilsCameraControls");
    const status = getElement("utilsCameraControlsStatus");
    controls = Array.isArray(payload?.controls) ? payload.controls : [];
    saved = payload?.saved || {};
    if (!container) return;

    const rows = [...container.children];
    const sameControls =
        rows.length === controls.length &&
        rows.every((row, index) => row.dataset.key === controls[index].key);
    if (!sameControls) {
        container.replaceChildren(...controls.map(buildControlRow));
    }
    controls.forEach((control, index) =>
        syncControlRow(container.children[index], control),
    );
    if (status) {
        status.textContent =
            controls.length > 0
                ? `${controls.length} controls reported by the camera driver.`
                : "This camera has no adjustable controls.";
    }
}

function setControlsMessage(message) {
    showControls({});
    const status = getElement("utilsCameraControlsStatus");
    if (status) status.textContent = message;
}

/**
 * Apply control values live, one request at a time.
 * @param {object} values - Control values keyed by control key.
 * @returns {Promise<void>}
 */
function sendValues(values) {
    const target = busId;
    const path = controlsPath();
    requestChain = requestChain.then(async () => {
        try {
            const payload = await fetchJson(path, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(values),
            });
            if (target !== busId) return;
            showControls(payload);
            if (payload?.refused?.length) {
                showWarning(
                    `The camera refused: ${payload.refused.join(", ")}`,
                );
            }
        } catch (error) {
            showDanger(`Failed to apply camera settings: ${error.message}`);
        }
    });
    return requestChain;
}

async function loadControls() {
    const generation = ++loadGeneration;
    loadFailed = false;
    try {
        const payload = await fetchJson(controlsPath());
        if (generation === loadGeneration) showControls(payload);
    } catch (error) {
        if (generation !== loadGeneration) return;
        loadFailed = true;
        setControlsMessage(error.message);
    }
}

async function saveControls() {
    if (!busId) return;
    const target = busId;
    const path = controlsPath("/save");
    await flushPendingValues();
    if (target !== busId) return;
    try {
        const payload = await fetchJson(path, { method: "POST" });
        if (target === busId) showControls(payload);
        showSuccess("Camera settings saved.");
    } catch (error) {
        showDanger(`Failed to save camera settings: ${error.message}`);
    }
}

function revertToSaved() {
    if (Object.keys(saved).length === 0) {
        showWarning("No saved settings for this camera yet.");
        return;
    }
    flushPendingValues();
    void sendValues(saved);
}

function applyDriverDefaults() {
    flushPendingValues();
    const values = Object.fromEntries(
        controls
            .filter((control) => !control.read_only)
            .map((control) => [control.key, control.default]),
    );
    if (Object.keys(values).length > 0) void sendValues(values);
}

/**
 * Show the settings for another camera.
 * @param {string} cameraBusId - Stable camera bus identifier, or "" for none.
 * @param {string} cameraStreamName - URL-safe name of the camera's view stream.
 */
export function setCameraSettingsCamera(cameraBusId, cameraStreamName) {
    if (cameraBusId === busId && cameraStreamName === streamName) {
        if (loadFailed) void loadControls();
        return;
    }
    clearTimeout(applyTimer);
    pendingValues = {};
    busId = cameraBusId;
    streamName = cameraStreamName;
    updateFeed();
    if (busId) {
        setControlsMessage("Loading camera controls...");
        void loadControls();
    } else {
        setControlsMessage("No camera selected.");
    }
}

/**
 * Wire the Camera Settings buttons once.
 */
export function initCameraSettings() {
    if (initialized) return;
    initialized = true;
    // The live view is open only while the tab is shown and the backend is up.
    subscribeConnectionStatus((status) => {
        backendConnected = status.backendConnected;
        updateFeed();
    });
    getElement("utilsAprilTagOverlayBtn")?.addEventListener("click", () =>
        setOverlay(!overlayOn),
    );
    getElement("utilsCameraSettingsFeed")?.addEventListener("error", () => {
        if (!overlayOn || !feedActive || !backendConnected) return;
        showWarning("AprilTag preview is unavailable on this device.");
        setOverlay(false);
    });
    getElement("utilsSaveCameraControlsBtn")?.addEventListener(
        "click",
        () => void saveControls(),
    );
    getElement("utilsRevertCameraControlsBtn")?.addEventListener(
        "click",
        revertToSaved,
    );
    getElement("utilsDefaultCameraControlsBtn")?.addEventListener(
        "click",
        applyDriverDefaults,
    );
}
