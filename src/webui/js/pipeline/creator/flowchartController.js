// Coordinates flowchart updates, node changes, and related UI refreshes in the pipeline creator.
import { FlowchartRenderer } from "../rendering.js";
import { handleDragStart } from "../dragDrop.js";
import { confirmDialog } from "../../ui/confirmationDialog.js";
import { showDanger, showWarning } from "../../ui/notificationSystem.js";
import { pipelineStore } from "../PipelineStore.js";
import { creatorContext } from "./context.js";
import { getDeviceInputNodes, getSelectedPipeline } from "./stateHelpers.js";
import { fetchPipelineThreadInfo } from "./dataApi.js";
import { applyPipelineErrorHighlights } from "./errorController.js";
import { applySelectedPipelineProfiling, clearProfilingUI } from "./profilingController.js";
import { updatePipelineCameraNote } from "./stateHelpers.js";
import { updateRestartIndicator } from "./restartController.js";

/**
 * Starts a drag operation while logging diagnostic details.
 *
 * @param {DragEvent} event - The drag event.
 * @param {object} item - The dragged item.
 * @param {number|null} [fromIndex=null] - Source index, if any.
 * @param {any} [collection=null] - Source collection, if any.
 * @returns {any} The underlying drag-start handler result.
 */
function handleDragStartWithLogging(event, item, fromIndex = null, collection = null) {
    console.log("[PIPELINE] Drag start initiated", {
        draggedElement: event.target,
        itemInstanceId: item?.instanceId || null,
        fromIndex,
        timestamp: new Date().toISOString(),
    });
    return handleDragStart(event, item, collection, fromIndex);
}

/**
 * Hides thread badges for all rendered nodes.
 */
function hideAllThreadBadges() {
    const flowchartRenderer = creatorContext.flowchartRenderer;
    if (!flowchartRenderer) return;
    for (const node of flowchartRenderer.nodes.values()) {
        node.hideThreadBadge();
    }
}

/**
 * Hides thread badges and clears profiling UI.
 */
function hideAllThreadAndProfilingBadges() {
    hideAllThreadBadges();
    clearProfilingUI();
}

/**
 * Fetches thread information for the selected pipeline and updates node badges.
 *
 * @returns {Promise<void>}
 */
async function fetchAndUpdateThreadInfo() {
    const selectedPipeline = getSelectedPipeline();
    const flowchartRenderer = creatorContext.flowchartRenderer;

    if (!selectedPipeline || pipelineStore.isRestartRequired()) {
        hideAllThreadBadges();
        return;
    }

    try {
        const data = await fetchPipelineThreadInfo(selectedPipeline.name);
        if (flowchartRenderer) {
            for (const [instanceId, node] of flowchartRenderer.nodes) {
                const uuid = pipelineStore.instanceIdToUuid.get(instanceId);
                if (uuid && data.operations) {
                    node.updateThreadInfo(data.operations[uuid]);
                } else {
                    node.hideThreadBadge();
                }
            }
        }
    } catch (error) {
        console.error("Error fetching thread info:", error);
        hideAllThreadBadges();
    }
}

/**
 * Refreshes flowchart-related overlays after structural pipeline changes.
 *
 * @returns {Promise<void>}
 */
async function postFlowchartStructureRefresh() {
    applyPipelineErrorHighlights();
    await fetchAndUpdateThreadInfo();
    applySelectedPipelineProfiling();
    updatePipelineCameraNote();
}

/**
 * Removes a pipeline node and refreshes related UI state.
 *
 * @param {string|number} instanceId - The node instance identifier.
 * @returns {Promise<void>}
 */
async function removeFromPipeline(instanceId) {
    const removedNode = pipelineStore.getNode(instanceId);
    const deviceInputCountBefore = getDeviceInputNodes().length;

    console.log("[PIPELINE] Removing operation from pipeline", {
        removedOperation: removedNode
            ? {
                  id: removedNode.operationId,
                  name: removedNode.name,
                  instanceId: removedNode.instanceId,
              }
            : null,
        pipelineLengthBefore: pipelineStore.getNodes().length,
        timestamp: new Date().toISOString(),
    });

    pipelineStore.removeNode(instanceId);
    creatorContext.flowchartRenderer?.removeNode(instanceId);

    console.log("[PIPELINE] Pipeline after removal", {
        pipelineLengthAfter: pipelineStore.getNodes().length,
        remainingOperations: pipelineStore.getNodes().map((node) => ({
            id: node.operationId,
            name: node.name,
            instanceId: node.instanceId,
        })),
        timestamp: new Date().toISOString(),
    });

    await postFlowchartStructureRefresh();

    const deviceInputCountAfter = getDeviceInputNodes().length;
    if (deviceInputCountBefore > 0 && deviceInputCountAfter === 0) {
        showWarning("No device_input nodes configured; bus_id required for camera input.");
    }

    console.log("Operation removed from pipeline - requiring backend restart");
    await updateRestartIndicator(true);
}

/**
 * Applies a flowchart-driven pipeline change.
 *
 * @param {object} changeEvent - The change payload.
 * @param {object} [options={}] - Change handlers and callbacks.
 * @param {Function} [options.autoSavePipeline] - Saves the pipeline automatically.
 * @param {Function} [options.renderCurrentPipeline] - Unused rendering callback.
 * @param {Function} [options.updateRunButton] - Updates the run button state.
 * @param {Function} [options.onCreatePipeline] - Creates a pipeline when needed.
 * @returns {Promise<void>}
 */
async function handleFlowchartPipelineChange(changeEvent, { autoSavePipeline, renderCurrentPipeline, updateRunButton, onCreatePipeline } = {}) {
    const selectedPipeline = getSelectedPipeline();
    if (!selectedPipeline) {
        const shouldCreate = await confirmDialog({
            title: "Create Pipeline?",
            message: "You need to create a pipeline before adding operations.",
            detail: "Would you like to create a new pipeline now?",
            confirmText: "Create Pipeline",
            variant: "warning",
        });
        if (!shouldCreate) return;
        await onCreatePipeline?.();
        if (!getSelectedPipeline()) return;
    }

    if (changeEvent.type === "add") {
        const node = pipelineStore.addNode(
            { id: changeEvent.operationId },
            changeEvent.position,
        );
        if (!node) {
            console.warn(`Operation ${changeEvent.operationId} not found`);
            return;
        }
        creatorContext.flowchartRenderer?.addNodeFromStore(node.instanceId);
        autoSavePipeline?.();
        await updateRestartIndicator(true);
        hideAllThreadBadges();
        await postFlowchartStructureRefresh();
    }
}

/**
 * Initializes the flowchart renderer for the pipeline creator.
 *
 * @param {object} params - Renderer setup parameters.
 * @returns {FlowchartRenderer|null}
 */
function initFlowchartRenderer({ openOperationSettings, updateRunButton, removeFromPipeline, autoSavePipeline, onPipelineChange }) {
    const flowchartCanvas = creatorContext.elements.flowchartCanvas;
    if (!flowchartCanvas) {
        console.error("Flowchart canvas not found (#flowchartCanvas)");
        showDanger("Pipeline builder could not start: the flowchart canvas is missing from the page.");
        return null;
    }

    const flowchartRenderer = new FlowchartRenderer(flowchartCanvas, {
        gridSpacing: 20,
        nodeSpacingX: 300,
        nodeSpacingY: 150,
        openOperationSettings,
        updateRunButton,
        removeFromPipeline,
        onPipelineChange,
        autoSavePipeline,
    });

    creatorContext.flowchartRenderer = flowchartRenderer;
    globalThis.flowchartRenderer = flowchartRenderer;
    return flowchartRenderer;
}

export {
    fetchAndUpdateThreadInfo,
    handleDragStartWithLogging,
    handleFlowchartPipelineChange,
    hideAllThreadBadges,
    hideAllThreadAndProfilingBadges,
    initFlowchartRenderer,
    postFlowchartStructureRefresh,
    removeFromPipeline,
};
