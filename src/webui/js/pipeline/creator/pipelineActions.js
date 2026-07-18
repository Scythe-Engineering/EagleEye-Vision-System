// Pipeline creator actions: loading, rendering, selection, and persistence.
import { renderOperations } from "../rendering.js";
import { pipelineStore } from "../PipelineStore.js";
import { confirmDialog } from "../../ui/confirmationDialog.js";
import { showDanger, showSuccess, showWarning } from "../../ui/notificationSystem.js";
import { creatorContext } from "./context.js";
import {
    getOperations,
    getPipeline,
    getPipelines,
    getSelectedPipeline,
} from "./stateHelpers.js";
import {
    fetchAvailableOperations,
    fetchAvailableCameras,
    fetchPipelineConfig,
    fetchPipelines,
    savePipelineConfig,
    deletePipelineConfig,
} from "./dataApi.js";
import { applySelectedPipelineProfiling } from "./profilingController.js";
import { postFlowchartStructureRefresh } from "./flowchartController.js";
import { updateRestartIndicator } from "./restartController.js";
import { updatePipelineCameraNote } from "./stateHelpers.js";

/**
 * Populate the pipeline select dropdown from the current pipeline list.
 *
 * @param {string|null} selectedPipelineName - Pipeline name to preselect.
 */
function populatePipelineDropdown(selectedPipelineName = null) {
    const pipelineSelect = creatorContext.elements.pipelineSelect;
    if (!pipelineSelect) return;
    pipelineSelect.innerHTML = "";

    const defaultOption = document.createElement("option");
    defaultOption.disabled = true;
    defaultOption.textContent = "Select Pipeline";
    pipelineSelect.appendChild(defaultOption);

    let foundSelectedPipeline = false;
    const pipelines = getPipelines();

    for (let index = 0; index < pipelines.length; index++) {
        const pipelineItem = pipelines[index];
        const option = document.createElement("option");
        option.value = pipelineItem.name;
        option.textContent = pipelineItem.displayName;

        if (
            selectedPipelineName === pipelineItem.name ||
            (selectedPipelineName === null && index === 0)
        ) {
            option.selected = true;
            pipelineStore.setCurrentPipeline(pipelineItem.name);
            foundSelectedPipeline = true;
        }

        pipelineSelect.appendChild(option);
    }

    if (
        selectedPipelineName &&
        !foundSelectedPipeline &&
        pipelines.length > 0
    ) {
        console.warn(
            `Pipeline "${selectedPipelineName}" not found in pipelines list, selecting first pipeline`,
        );
        const firstOption = pipelineSelect.querySelector(
            "option:not([disabled])",
        );
        if (firstOption) {
            firstOption.selected = true;
            pipelineStore.setCurrentPipeline(pipelines[0].name);
        }
    }

    if (pipelines.length === 0) {
        pipelineStore.setCurrentPipeline(null);
    }
}

/**
 * Handle changes to the selected pipeline and sync UI state.
 *
 * @param {{loadPipelineIntoBuilder?: Function}} options - Selection handlers.
 */
async function handlePipelineSelection({ loadPipelineIntoBuilder }) {
    const pipelineSelect = creatorContext.elements.pipelineSelect;
    const selectedValue = pipelineSelect?.value;
    const pipelines = getPipelines();
    const selectedPipeline = pipelines.find(
        (pipelineItem) => pipelineItem.name === selectedValue,
    );
    console.log("Selected pipeline:", selectedPipeline);

    if (selectedPipeline) {
        pipelineStore.setCurrentPipeline(selectedPipeline.name);
        await loadPipelineIntoBuilder?.(selectedPipeline.name);
        applySelectedPipelineProfiling();
    }

    updateDeleteButtonVisibility();
}

/**
 * Load a pipeline configuration into the builder state.
 *
 * @param {string} pipelineName - Pipeline identifier to load.
 * @param {{renderCurrentPipeline?: Function, centerView?: boolean}} [options={}] - Optional render callback.
 */
async function loadPipelineIntoBuilder(pipelineName, { renderCurrentPipeline, centerView = true } = {}) {
    try {
        const operations = getOperations();
        if (operations.length === 0) {
            console.warn("Operations not loaded yet, cannot load pipeline");
            return;
        }

        const pipelineConfig = await fetchPipelineConfig(pipelineName);
        const allConnections = [];
        pipelineConfig.forEach((configItem) => {
            if (configItem.connections && Array.isArray(configItem.connections)) {
                allConnections.push(...configItem.connections);
            }
        });

        pipelineStore.loadPipelineData(pipelineConfig, allConnections);
        await renderCurrentPipeline?.({ centerView });
        updateRunButton();
        updatePipelineCameraNote();
    } catch (error) {
        showDanger("Failed to load pipeline");
        console.error("Failed to load pipeline:", error);
    }
}

/**
 * Render the current pipeline into the flowchart view.
 *
 * @param {{centerView?: boolean}} [options={}] - Render options.
 */
async function renderCurrentPipeline({ centerView = true } = {}) {
    const flowchartRenderer = creatorContext.flowchartRenderer;
    if (!flowchartRenderer) return;
    const pipeline = getPipeline();
    const connections = pipelineStore.getConnectionsForRenderer();
    await flowchartRenderer.renderPipeline(pipeline, {
        connections,
        centerView,
    });
    await postFlowchartStructureRefresh();
}

/**
 * Auto-load the selected pipeline when the creator is initialized.
 *
 * @param {{loadPipelineIntoBuilder?: Function}} options - Auto-fill callbacks.
 */
async function checkAndTriggerAutoFill({ loadPipelineIntoBuilder }) {
    try {
        const pipelines = getPipelines();
        const pipelineSelect = creatorContext.elements.pipelineSelect;
        if (!pipelineSelect?.value) {
            console.log("No pipeline selected, skipping auto-fill");
            return;
        }
        const selectedPipelineName = pipelineSelect.value;
        const pipelineObj = pipelines.find(
            (p) => p.name === selectedPipelineName,
        );
        if (!pipelineObj) {
            console.log("Selected pipeline not found in pipelines list");
            return;
        }
        pipelineStore.setCurrentPipeline(pipelineObj.name);
        console.log("Pipeline pre-selected, triggering auto-fill");
        await loadPipelineIntoBuilder?.(pipelineObj.name);
    } catch (error) {
        console.error("Error during auto-fill check:", error);
    }
}

/**
 * Update the run button enabled state based on whether nodes exist.
 */
function updateRunButton() {
    const runButton = creatorContext.elements.runButton;
    if (runButton) runButton.disabled = pipelineStore.getNodes().length === 0;
}

/**
 * Persist the current pipeline configuration to the backend.
 *
 * @param {{showNotification?: boolean, requiresRestart?: boolean}} [options={}] - Save options.
 */
async function autoSavePipelineImpl(options = {}) {
    const selectedPipeline = getSelectedPipeline();
    if (!selectedPipeline) {
        console.log("No pipeline selected, skipping auto-save");
        return;
    }
    try {
        const pipelineConfig = pipelineStore.exportToConfig();
        const result = await savePipelineConfig(
            selectedPipeline.name,
            pipelineConfig,
        );
        console.log("Pipeline auto-saved successfully");
        const restartRequired = Boolean(
            result?.restart_required ||
                options.requiresRestart ||
                result?.live_update_status === "unsupported",
        );
        if (restartRequired) {
            await updateRestartIndicator(true, { syncBackend: !result?.restart_required });
        } else if (result?.restart_required === false) {
            await updateRestartIndicator(false);
        }
        if (options.showNotification) {
            if (result?.live_update_status === "failed") {
                showDanger("Operation settings saved, but live apply failed.");
            } else if (
                restartRequired
            ) {
                showWarning(
                    "Operation settings saved. Restart required to apply.",
                );
            } else {
                showSuccess("Operation settings applied.");
            }
        }
        return result;
    } catch (error) {
        console.error("Failed to auto-save pipeline:", error);
        if (options.showNotification) {
            showDanger("Failed to save operation settings.");
        }
        return null;
    }
}

/**
 * Convenience wrapper for auto-saving the current pipeline.
 *
 * @param {object} [options={}] - Auto-save options.
 * @returns {Promise<*|null>} Save result or null on failure.
 */
const autoSavePipeline = (options = {}) => autoSavePipelineImpl(options);

/**
 * Create a new pipeline, optionally overwriting an existing one.
 *
 * @param {{renderCurrentPipeline?: Function, updateDeleteButtonVisibility?: Function, autoSavePipeline?: Function}} options - Creation callbacks.
 */
async function createNewPipeline({
    renderCurrentPipeline,
    updateDeleteButtonVisibility,
    autoSavePipeline,
}) {
    const newPipelineName = prompt("Enter a name for the new pipeline:");
    if (!newPipelineName || newPipelineName.trim() === "") return;
    const pipelineFileName = newPipelineName.trim().replaceAll(/\s+/g, "_");
    const pipelines = getPipelines();
    const existingPipeline = pipelines.find((p) => p.name === pipelineFileName);
    if (existingPipeline) {
        const shouldOverwrite = await confirmDialog({
            title: "Overwrite Pipeline?",
            message: `Pipeline "${newPipelineName}" already exists.`,
            detail: "Do you want to overwrite it?",
            confirmText: "Overwrite",
            variant: "warning",
        });
        if (!shouldOverwrite) return;
    }
    try {
        pipelineStore.clearPipeline();
        const newPipelineObj = {
            name: pipelineFileName,
            displayName: newPipelineName.trim(),
        };
        const currentPipelines = pipelineStore.state.pipelines;
        const existingIndex = currentPipelines.findIndex(
            (p) => p.name === pipelineFileName,
        );
        if (existingIndex >= 0) currentPipelines[existingIndex] = newPipelineObj;
        else currentPipelines.push(newPipelineObj);
        pipelineStore.setCurrentPipeline(pipelineFileName);
        populatePipelineDropdown(pipelineFileName);
        setTimeout(() => {
            const pipelineSelect = creatorContext.elements.pipelineSelect;
            if (pipelineSelect) pipelineSelect.value = pipelineFileName;
        }, 10);
        const operations = getOperations();
        const deviceInputOp = operations.find((op) => op.id === "device_input.py");
        if (deviceInputOp) {
            pipelineStore.addNode(
                { id: deviceInputOp.id, config: {} },
                { x: 100, y: 100 },
            );
        }
        await renderCurrentPipeline?.();
        updateRunButton();
        updateDeleteButtonVisibility?.();
        await autoSavePipeline?.();
    } catch (error) {
        console.error("Failed to create new pipeline:", error);
        alert(
            "Failed to create new pipeline. Please check the console for details.",
        );
    }
}

/**
 * Delete the currently selected pipeline after confirmation.
 *
 * @param {{renderCurrentPipeline?: Function, updateDeleteButtonVisibility?: Function, updateRunButton?: Function}} options - Deletion callbacks.
 */
async function deleteCurrentPipeline({
    renderCurrentPipeline,
    updateDeleteButtonVisibility,
    updateRunButton,
}) {
    const selectedPipeline = getSelectedPipeline();
    if (!selectedPipeline) {
        alert("No pipeline selected to delete.");
        return;
    }
    const confirmed = await confirmDialog({
        title: "Delete Pipeline?",
        message: `Delete the pipeline "${selectedPipeline.displayName}"?`,
        detail: "This action cannot be undone.",
        confirmText: "Delete Pipeline",
    });
    if (!confirmed) return;
    try {
        const result = await deletePipelineConfig(selectedPipeline.name);
        console.log("Pipeline deleted from backend:", result);
        const currentPipelines = pipelineStore.state.pipelines;
        const pipelineIndex = currentPipelines.findIndex(
            (p) => p.name === selectedPipeline.name,
        );
        if (pipelineIndex === -1) {
            console.error("Pipeline not found in pipelines array");
            alert("Failed to delete pipeline. Pipeline not found.");
            return;
        }
        currentPipelines.splice(pipelineIndex, 1);
        pipelineStore.clearPipeline();
        pipelineStore.setCurrentPipeline(null);
        populatePipelineDropdown();
        await renderCurrentPipeline?.();
        updateRunButton?.();
        updateDeleteButtonVisibility?.();
    } catch (error) {
        console.error("Failed to delete pipeline:", error);
        alert(
            "Failed to delete pipeline. Please check the console for details.",
        );
    }
}

/**
 * Toggle delete button visibility based on whether a pipeline is selected.
 */
function updateDeleteButtonVisibility() {
    const deletePipelineButton = creatorContext.elements.deletePipelineButton;
    if (!deletePipelineButton) return;
    const selectedPipeline = getSelectedPipeline();
    deletePipelineButton.classList.toggle("hidden", !selectedPipeline);
}

/**
 * Refresh pipeline creator data and re-render the UI after reconnecting.
 *
 * @param {{openOperationSettings?: Function, handleDragStartWithLogging?: Function, checkBackendRestartStatus?: Function, loadPipelineIntoBuilder?: Function}} options - Refresh callbacks.
 */
async function refreshPipelineCreator({
    openOperationSettings,
    handleDragStartWithLogging,
    checkBackendRestartStatus,
    loadPipelineIntoBuilder,
}) {
    try {
        console.log("[PIPELINE] Refreshing pipeline creator after reconnection");
        const flowchartCanvas = creatorContext.flowchartRenderer?.canvas;
        const savedViewport = flowchartCanvas?.getViewportState?.() || null;

        await fetchAvailableOperations(pipelineStore);
        const operations = getOperations();
        const operationsList = creatorContext.elements.operationsList;
        if (operationsList && operations.length > 0) {
            renderOperations(
                operations,
                operationsList,
                openOperationSettings,
                handleDragStartWithLogging,
            );
        }
        await fetchAvailableCameras(pipelineStore);
        await fetchPipelines(pipelineStore);
        populatePipelineDropdown();
        const selectedPipeline = getSelectedPipeline();
        if (selectedPipeline) {
            await loadPipelineIntoBuilder?.(selectedPipeline.name, {
                centerView: false,
            });
        }
        if (savedViewport && flowchartCanvas?.setViewportState) {
            flowchartCanvas.setViewportState(savedViewport);
        }
        updateDeleteButtonVisibility();
        await checkBackendRestartStatus?.();
    } catch (error) {
        console.error("[PIPELINE] Error refreshing pipeline creator:", error);
    }
}

/**
 * Trigger the pipeline run action for the current configuration.
 */
function runPipeline() {
    console.log("Running pipeline:", getPipeline());
    alert("Pipeline run! Check console for details.");
}

export {
    autoSavePipeline,
    autoSavePipelineImpl,
    checkAndTriggerAutoFill,
    createNewPipeline,
    deleteCurrentPipeline,
    handlePipelineSelection,
    loadPipelineIntoBuilder,
    populatePipelineDropdown,
    refreshPipelineCreator,
    renderCurrentPipeline,
    runPipeline,
    updateDeleteButtonVisibility,
    updateRunButton,
};
