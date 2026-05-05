/**
 * Controls backend restart state for the pipeline creator UI.
 */
import { creatorContext } from "./context.js";
import {
    getRestartRequired,
    restartBackend,
    setRestartRequired,
} from "./dataApi.js";
import { hideAllProfilingBadges } from "./profilingController.js";
import { showDanger } from "../../ui/notificationSystem.js";

/**
 * Updates the restart indicator UI and optionally syncs the state to the backend.
 *
 * @param {boolean} show - Whether the restart warning should be shown.
 * @param {Object} options - Optional behavior flags.
 * @param {boolean} [options.syncBackend=false] - Whether to persist the state to the backend.
 */
async function updateRestartIndicator(show = false, options = {}) {
    const restartIndicator = creatorContext.elements.restartIndicator;
    if (options.syncBackend) {
        try {
            const data = await setRestartRequired(show);
            applyBackendRestartState(data);
            return;
        } catch (error) {
            showDanger("Failed to notify backend about restart requirement");
            console.error(
                "Failed to notify backend about restart requirement:",
                error,
            );
        }
    }

    creatorContext.restartRequired = show;

    if (!restartIndicator) return;

    const restartMessage =
        restartIndicator.querySelector(".text-red-100") ||
        restartIndicator.querySelector("span");

    if (show) {
        restartIndicator.classList.remove("hidden");
        if (restartMessage) restartMessage.textContent = "Backend restart required";
        restartIndicator.classList.add("backend-state-warning");
    } else {
        restartIndicator.classList.add("hidden");
        restartIndicator.classList.remove("backend-state-warning");
    }

    if (show) {
        hideAllProfilingBadges();
    }
}

/**
 * Applies backend restart metadata to the local UI context.
 *
 * @param {Object} data - Backend restart response data.
 */
function applyBackendRestartState(data = {}) {
    const show = Boolean(data.restart_required);
    if (typeof data.runtime_id === "string" && data.runtime_id) {
        creatorContext.backendRuntimeId = data.runtime_id;
    }
    updateRestartIndicator(show);
}

/**
 * Requests a backend restart and waits for the runtime to change before reloading.
 */
async function handleRestartBackend() {
    try {
        const restartButton = creatorContext.elements.restartIndicator?.querySelector(
            "#restartBackendButton",
        );

        if (restartButton) {
            restartButton.disabled = true;
            restartButton.textContent = "Restarting...";
        }

        const previousRuntimeId = creatorContext.backendRuntimeId;
        try {
            await restartBackend();
        } catch (error) {
            console.warn("Failed to send restart request:", error);
        }

        await waitForBackendRuntimeChange(previousRuntimeId);
        globalThis.location.reload();
    } catch (error) {
        console.error("Failed to restart backend:", error);
    }
}

/**
 * Waits until the backend runtime ID changes or the timeout elapses.
 *
 * @param {string|undefined|null} previousRuntimeId - Runtime ID observed before the restart.
 */
async function waitForBackendRuntimeChange(previousRuntimeId) {
    const deadline = Date.now() + 30000;
    while (Date.now() < deadline) {
        await new Promise((resolve) => setTimeout(resolve, 750));
        try {
            const data = await getRestartRequired();
            if (!previousRuntimeId || data.runtime_id !== previousRuntimeId) {
                applyBackendRestartState(data);
                return;
            }
        } catch (_error) {
            // The backend may be between shutdown and startup.
        }
    }
}

/**
 * Fetches and applies the current backend restart status.
 */
async function checkBackendRestartStatus() {
    try {
        const data = await getRestartRequired();
        applyBackendRestartState(data);
    } catch (error) {
        console.error("Error checking backend restart status:", error);
    }
}

/**
 * Checks whether the current pipeline operation requires a backend restart.
 *
 * @param {*} operationItem - The pipeline operation being evaluated.
 * @param {string|null} changedParamName - The changed parameter name.
 * @param {*} changedValue - The changed parameter value.
 */
async function checkPipelineRestartRequirements(
    operationItem = null,
    changedParamName = null,
    changedValue = null,
) {
    await checkBackendRestartStatus();
}

export {
    applyBackendRestartState,
    checkBackendRestartStatus,
    checkPipelineRestartRequirements,
    handleRestartBackend,
    updateRestartIndicator,
};
