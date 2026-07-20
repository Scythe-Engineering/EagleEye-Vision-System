// Controls the pipeline-wide settings modal.
import {
    closeOnBackdropClick,
    closeOnEscape,
    createElement,
    getOrCreateModalElements,
    hideModal,
    showModal,
} from "../../ui/modal.js";
import { showDanger, showSuccess } from "../../ui/notificationSystem.js";
import { fetchPipelineSettings, savePipelineSettings } from "./dataApi.js";
import { updateRestartIndicator } from "./restartController.js";
import { getSelectedPipeline } from "./stateHelpers.js";

const OVERLAY_ID = "pipelineSettingsOverlay";
const MODAL_ID = "pipelineSettingsModal";

/**
 * Creates the controller used to open and save pipeline-wide settings.
 *
 * @returns {Function} Opens the settings modal for the selected pipeline.
 */
export function createPipelineSettingsController() {
    const { overlay, modal } = getOrCreateModalElements({
        overlayId: OVERLAY_ID,
        modalId: MODAL_ID,
        modalClassName:
            "bg-[#1f1f1f] rounded-2xl shadow-2xl w-full max-w-lg mx-4 flex flex-col border border-[#414141] overflow-hidden",
    });

    const title = createElement("h2", {
        className: "text-xl font-semibold text-[#f9c845]",
        text: "Pipeline Settings",
    });
    const pipelineName = createElement("p", {
        className: "text-xs text-[#999] mt-1",
    });
    const closeButton = createElement("button", {
        type: "button",
        className:
            "w-9 h-9 rounded-lg border border-[#414141] text-gray-400 hover:text-white hover:bg-[#2a2a2a] transition-colors text-xl",
        text: "×",
        title: "Close pipeline settings",
        "aria-label": "Close pipeline settings",
    });
    const header = createElement(
        "div",
        {
            className:
                "flex items-center justify-between px-5 py-4 bg-[#1e1e1e] border-b border-[#414141]",
        },
        [createElement("div", {}, [title, pipelineName]), closeButton],
    );

    const limitFramesToggle = createElement("input", {
        type: "checkbox",
        className: "w-5 h-5 shrink-0",
        "aria-describedby": "pipelineLimitFramesDescription",
    });
    const limitFramesLabel = createElement("label", {
        className: "text-sm font-medium text-[#f9c845] cursor-pointer",
        text: "Limit frames to camera capture speed",
    });
    const limitFramesDescription = createElement("p", {
        id: "pipelineLimitFramesDescription",
        className: "mt-1 text-sm leading-5 text-[#aaa]",
        text: "When enabled, this pipeline will not process frames faster than the camera captures them. Disable it to process available frames as quickly as possible.",
    });
    const settingsBody = createElement("div", { className: "px-5 py-5" }, [
        createElement("div", { className: "flex items-start gap-3" }, [
            limitFramesToggle,
            createElement("div", { className: "min-w-0" }, [
                limitFramesLabel,
                limitFramesDescription,
            ]),
        ]),
    ]);
    limitFramesLabel.htmlFor = "pipelineLimitFramesToggle";
    limitFramesToggle.id = "pipelineLimitFramesToggle";

    const cancelButton = createElement("button", {
        type: "button",
        className:
            "px-4 py-2 rounded-lg bg-[#2b2b2b] text-white hover:bg-[#3a3a3a] transition-colors",
        text: "Cancel",
    });
    const saveButton = createElement("button", {
        type: "button",
        className:
            "px-4 py-2 rounded-lg bg-[#f9c845] text-black font-semibold hover:bg-[#e6b73c] disabled:opacity-40 disabled:cursor-not-allowed transition-colors",
        text: "Save",
    });
    const footer = createElement(
        "div",
        {
            className:
                "px-5 py-4 bg-[#1e1e1e] border-t border-[#414141] flex items-center justify-end gap-2",
        },
        [cancelButton, saveButton],
    );
    modal.replaceChildren(header, settingsBody, footer);

    let selectedPipelineName = null;
    let settingsRequestId = 0;
    let saving = false;

    /** Close the modal unless settings are currently being saved. */
    const closeSettings = () => {
        if (saving) return;
        settingsRequestId += 1;
        selectedPipelineName = null;
        hideModal(overlay);
    };

    /** Save the displayed settings for the pipeline that opened the modal. */
    const saveSettings = async () => {
        if (!selectedPipelineName || saving) return;

        saving = true;
        saveButton.disabled = true;
        cancelButton.disabled = true;
        saveButton.textContent = "Saving...";
        try {
            const result = await savePipelineSettings(selectedPipelineName, {
                limit_frames_to_camera_capture_speed: limitFramesToggle.checked,
            });
            await updateRestartIndicator(Boolean(result?.restart_required));
            showSuccess("Pipeline settings saved. Restart required to apply.");
            hideModal(overlay);
        } catch (error) {
            console.error("Failed to save pipeline settings:", error);
            showDanger("Failed to save pipeline settings.");
        } finally {
            saving = false;
            saveButton.disabled = false;
            cancelButton.disabled = false;
            saveButton.textContent = "Save";
        }
    };

    /** Open the modal and load settings for the currently selected pipeline. */
    const openPipelineSettings = async () => {
        const selectedPipeline = getSelectedPipeline();
        if (!selectedPipeline) return;

        selectedPipelineName = selectedPipeline.name;
        const requestedPipelineName = selectedPipeline.name;
        const requestId = ++settingsRequestId;
        pipelineName.textContent = selectedPipeline.displayName;
        limitFramesToggle.checked = false;
        limitFramesToggle.disabled = true;
        saveButton.disabled = true;
        showModal(overlay);

        try {
            const settings = await fetchPipelineSettings(requestedPipelineName);
            if (
                requestId !== settingsRequestId ||
                selectedPipelineName !== requestedPipelineName
            ) {
                return;
            }
            limitFramesToggle.checked = Boolean(
                settings?.limit_frames_to_camera_capture_speed,
            );
            limitFramesToggle.disabled = false;
            saveButton.disabled = false;
        } catch (error) {
            if (requestId !== settingsRequestId) return;
            console.error("Failed to load pipeline settings:", error);
            showDanger("Failed to load pipeline settings.");
            hideModal(overlay);
        }
    };

    closeButton.addEventListener("click", closeSettings);
    cancelButton.addEventListener("click", closeSettings);
    saveButton.addEventListener("click", () => void saveSettings());
    closeOnBackdropClick(overlay, closeSettings);
    closeOnEscape(overlay, closeSettings);

    return openPipelineSettings;
}
