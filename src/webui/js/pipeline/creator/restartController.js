import { creatorContext } from "./context.js";
import {
    getRestartRequired,
    restartBackend,
    setRestartRequired,
} from "./dataApi.js";
import { hideAllProfilingBadges } from "./profilingController.js";
import { showDanger } from "../../ui/notificationSystem.js";

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

function applyBackendRestartState(data = {}) {
    const show = Boolean(data.restart_required);
    if (typeof data.runtime_id === "string" && data.runtime_id) {
        creatorContext.backendRuntimeId = data.runtime_id;
    }
    updateRestartIndicator(show);
}

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

async function checkBackendRestartStatus() {
    try {
        const data = await getRestartRequired();
        applyBackendRestartState(data);
    } catch (error) {
        console.error("Error checking backend restart status:", error);
    }
}

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
