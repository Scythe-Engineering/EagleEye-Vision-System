import { BACKEND_BASE_URL } from "./config.js";

let demoModeEnabled = false;
let demoModeResolved = false;
/** @type {Promise<boolean>|null} */
let demoModeLoadPromise = null;

/**
 * Apply demo-mode CSS class and banner to the document body.
 *
 * @param {boolean} enabled - Whether demo mode is active.
 */
function applyDemoModeDocumentState(enabled) {
    document.body.classList.toggle("demo-mode", enabled);

    let banner = document.getElementById("demo-mode-banner");
    if (!enabled) {
        banner?.remove();
        return;
    }

    if (!banner) {
        banner = document.createElement("div");
        banner.id = "demo-mode-banner";
        banner.className =
            "fixed bottom-3 left-1/2 -translate-x-1/2 z-[120] px-4 py-2 rounded-md border border-[#f9c845]/60 bg-[#1b1b1b]/95 text-[#f9c845] text-sm font-semibold shadow-lg pointer-events-none";
        banner.textContent = "Demo mode — view only (changes are not saved)";
        document.body.appendChild(banner);
    }
}

/**
 * Disable common mutable controls across settings and pipeline chrome.
 */
function disableMutableControls() {
    const controlIds = [
        "saveSettingsBtn",
        "restartBackendBtn",
        "restartBackendButton",
        "newPipelineButton",
        "deletePipelineButton",
        "undoButton",
        "redoButton",
        "pipelineJsonEditorButton",
        "pipelineSettingsButton",
    ];

    for (const controlId of controlIds) {
        const control = document.getElementById(controlId);
        if (!control) {
            continue;
        }
        control.setAttribute("disabled", "true");
        control.classList.add("opacity-40", "cursor-not-allowed", "pointer-events-none");
        control.classList.add("hidden");
    }

    const editableInputs = [
        "robotAddressInput",
        "viewStreamDownscaleInput",
    ];
    for (const inputId of editableInputs) {
        const input = document.getElementById(inputId);
        if (!input) {
            continue;
        }
        input.setAttribute("readonly", "true");
        input.setAttribute("disabled", "true");
    }

    const manageButtonSelectors = [
        "#manageTestVideosBtn",
        "#manageRobotFilesBtn",
        "#manageFieldFilesBtn",
        "#wifiConnectBtn",
        "#wifiDisconnectBtn",
        "#runSystemUpdateBtn",
        "#testNotificationsBtn",
    ];
    for (const selector of manageButtonSelectors) {
        const button = document.querySelector(selector);
        if (!button) {
            continue;
        }
        button.setAttribute("disabled", "true");
        button.classList.add("opacity-40", "cursor-not-allowed", "pointer-events-none");
    }

    const operationsList = document.getElementById("operationsList");
    if (operationsList) {
        operationsList.classList.add("pointer-events-none", "opacity-60");
        operationsList.setAttribute("aria-disabled", "true");
    }

    const pipelineSubtitle = document.getElementById("pipelineBuilderSubtitle");
    if (pipelineSubtitle) {
        pipelineSubtitle.textContent =
            "View pipeline configuration (read-only demo)";
    }
}

/**
 * Load demo mode state from the backend general configuration.
 *
 * @returns {Promise<boolean>} Whether demo mode is enabled.
 */
export async function loadDemoMode() {
    if (demoModeResolved) {
        return demoModeEnabled;
    }
    if (demoModeLoadPromise) {
        return demoModeLoadPromise;
    }

    demoModeLoadPromise = (async () => {
        try {
            const response = await fetch(`${BACKEND_BASE_URL}/get-general-conf`, {
                method: "GET",
                headers: { "Content-Type": "application/json" },
            });
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const settings = await response.json();
            demoModeEnabled = Boolean(settings?.demo_mode);
        } catch (error) {
            console.warn("Failed to load demo mode flag; assuming disabled", error);
            demoModeEnabled = false;
        }

        demoModeResolved = true;
        applyDemoModeDocumentState(demoModeEnabled);
        if (demoModeEnabled) {
            disableMutableControls();
        }
        return demoModeEnabled;
    })();

    return demoModeLoadPromise;
}

/**
 * Return whether demo/read-only mode is currently active.
 *
 * @returns {boolean} True when the UI should be read-only.
 */
export function isDemoMode() {
    return demoModeEnabled;
}

/**
 * Wait until demo mode has been resolved from the backend.
 *
 * @returns {Promise<boolean>} Whether demo mode is enabled.
 */
export async function whenDemoModeReady() {
    return loadDemoMode();
}
