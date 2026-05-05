// Manages the system update modal, status checks, and backend restart flow.
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
import { showDanger } from "../ui/notificationSystem.js";

const OVERLAY_ID = "systemUpdateOverlay";
const MODAL_ID = "systemUpdateModal";

let initialized = false;
let statusTimer = null;
let updateAvailable = false;
let updating = false;
let statusReason = "Checking update availability...";

/**
 * Gets or creates the modal overlay elements used by the system update UI.
 * @returns {{overlay: HTMLElement, modal: HTMLElement}}
 */
function getOverlayElements() {
    return getOrCreateModalElements({
        overlayId: OVERLAY_ID,
        modalId: MODAL_ID,
        modalClassName:
            "bg-[#1a1a1a] rounded-lg shadow-xl max-w-lg w-full mx-4 max-h-[90vh] flex flex-col border border-[#414141]",
    });
}

/**
 * Fetches JSON from the backend and throws on non-OK responses.
 * @param {string} path
 * @param {RequestInit} [options={}]
 * @returns {Promise<any>}
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
            payload.error || payload.message || `Request failed: ${response.status}`,
        );
        error.payload = payload;
        error.status = response.status;
        throw error;
    }
    return payload;
}

/**
 * Closes the system update modal unless an update is currently running.
 */
function close() {
    if (updating) {
        return;
    }
    hideModal(getOverlayElements().overlay);
}

/**
 * Updates the system update button to reflect current availability and state.
 */
function setButtonState() {
    const button = document.getElementById("updateSystemBtn");
    if (!button) {
        return;
    }

    button.disabled = updating || !updateAvailable;
    button.title = updateAvailable
        ? "Pull code updates, install apt upgrades, and restart the backend"
        : statusReason;
    button.textContent = updating ? "Updating..." : "Update System";
}

/**
 * Refreshes the backend-reported system update availability and reason.
 */
async function refreshUpdateStatus() {
    if (updating) {
        return;
    }

    try {
        const payload = await fetchJson("/system-update/status");
        updateAvailable = payload.available === true;
        statusReason = payload.reason || "Update requires WiFi with internet access";
    } catch (error) {
        updateAvailable = false;
        statusReason = error.payload?.error || "Unable to check WiFi internet access";
    }
    setButtonState();
}

/**
 * Shows the confirmation dialog before starting the update flow.
 */
async function renderConfirm() {
    const confirmed = await confirmDialog({
        title: "Update System?",
        message: "This will restart the system. Are you sure?",
        detail: "The backend will run git pull, apt update, and non-interactive apt upgrade before restarting.",
        confirmText: "Update and Restart",
        variant: "warning",
    });
    if (confirmed) {
        runUpdate();
    }
}

/**
 * Renders the in-modal progress state for the update flow.
 * @param {string} message
 * @param {string} [detail=""]
 */
function renderProgress(message, detail = "") {
    const { overlay, modal } = getOverlayElements();
    modal.innerHTML = "";
    showModal(overlay);
    modal.appendChild(
        createElement("div", { className: "p-6" }, [
            createElement("h3", {
                className: "text-xl font-bold text-yellow-400 mb-4",
                text: "Updating System",
            }),
            createElement("div", {
                className: "mb-3 text-gray-200",
                text: message,
            }),
            createElement("div", {
                className: "h-3 w-full overflow-hidden rounded-full bg-[#2a2a2a] border border-[#414141]",
                html: '<div class="h-full w-1/3 rounded-full bg-yellow-400 animate-pulse"></div>',
            }),
            createElement("pre", {
                className:
                    "mt-4 max-h-48 overflow-y-auto whitespace-pre-wrap rounded bg-[#101010] p-3 text-xs text-gray-300 border border-[#414141]",
                text: detail,
            }),
        ]),
    );
}

/**
 * Renders the in-modal error state for failed update attempts.
 * @param {string} message
 */
function renderError(message) {
    const { overlay, modal } = getOverlayElements();
    modal.innerHTML = "";
    showModal(overlay);
    modal.appendChild(
        createElement("div", { className: "p-6" }, [
            createElement("h3", {
                className: "text-xl font-bold text-red-300 mb-3",
                text: "Update Failed",
            }),
            createElement("pre", {
                className:
                    "max-h-64 overflow-y-auto whitespace-pre-wrap rounded bg-[#101010] p-3 text-sm text-red-100 border border-red-700/60 mb-5",
                text: message,
            }),
            createElement("div", { className: "flex justify-end" }, [
                createElement("button", {
                    type: "button",
                    className:
                        "px-4 py-2 bg-[#2a2a2a] text-[#f9c845] rounded-md border border-[#414141] hover:bg-[#3a3a3a]",
                    text: "Close",
                    onclick: close,
                }),
            ]),
        ]),
    );
}

/**
 * Runs the system update sequence and restarts the backend.
 */
async function runUpdate() {
    updating = true;
    setButtonState();
    renderProgress("Checking WiFi internet access...");

    try {
        const status = await fetchJson("/system-update/status");
        if (status.available !== true) {
            throw new Error(status.reason || "WiFi internet access is required.");
        }

        renderProgress("Pulling latest changes and installing apt upgrades...");
        const updateResult = await fetchJson("/system-update/run", { method: "POST" });
        renderProgress("Restarting backend...", updateResult.output || "Update completed.");

        try {
            await fetchJson("/restart-backend", { method: "POST" });
        } catch (error) {
            console.warn("Restart request failed or connection closed:", error);
        }

        setTimeout(() => {
            globalThis.location.reload();
        }, 2500);
    } catch (error) {
        updating = false;
        const message = error.payload?.error || error.payload?.output || error.message || "Update failed";
        renderError(message);
        setButtonState();
    }
}

/**
 * Initializes the system update manager UI and event handlers.
 */
export function initializeSystemUpdateManager() {
    if (initialized) {
        return;
    }
    initialized = true;

    const button = document.getElementById("updateSystemBtn");
    if (!button) {
        return;
    }

    button.addEventListener("click", () => {
        if (!button.disabled) {
            renderConfirm();
        } else if (statusReason) {
            showDanger(statusReason);
        }
    });

    const { overlay } = getOverlayElements();
    closeOnBackdropClick(overlay, close);
    closeOnEscape(overlay, close);

    setButtonState();
    refreshUpdateStatus();
    statusTimer = setInterval(refreshUpdateStatus, 30000);

    window.addEventListener("beforeunload", () => {
        if (statusTimer) {
            clearInterval(statusTimer);
        }
    });
}
