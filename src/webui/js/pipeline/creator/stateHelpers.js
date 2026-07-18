import { creatorContext } from "./context.js";
import { pipelineStore } from "../PipelineStore.js";

/**
 * Helpers for reading pipeline creator state and deriving UI display values.
 */

/**
 * Returns the current operations list from the pipeline store.
 *
 * @returns {Array}
 */
export function getOperations() {
    return pipelineStore.state.operations;
}

/**
 * Returns the nodes prepared for renderer consumption.
 *
 * @returns {Array}
 */
export function getPipeline() {
    return pipelineStore.getNodesForRenderer();
}

/**
 * Returns the full pipeline collection from the store.
 *
 * @returns {Array}
 */
export function getPipelines() {
    return pipelineStore.state.pipelines;
}

/**
 * Returns the pipeline matching the current pipeline name, or null.
 *
 * @returns {Object|null}
 */
export function getSelectedPipeline() {
    const pipelineName = pipelineStore.state.currentPipeline?.pipelineName;
    if (!pipelineName) {
        return null;
    }
    const selectedPipeline = pipelineStore.state.pipelines.find(
        (p) => p.name === pipelineName,
    );
    return selectedPipeline ?? null;
}

/**
 * Returns nodes whose normalized operation id is device_input.
 *
 * @returns {Array}
 */
export function getDeviceInputNodes() {
    return pipelineStore.getNodes().filter((node) => {
        return (
            pipelineStore.normalizeOperationId(node.operationId) ===
            "device_input"
        );
    });
}

/**
 * Returns unique bus IDs from device input nodes.
 *
 * @returns {Array<string>}
 */
export function getDeviceInputBusIds() {
    const busIds = new Set();
    pipelineStore.getNodes().forEach((node) => {
        const operationId = pipelineStore.normalizeOperationId(
            node.operationId,
        );
        if (operationId === "device_input") {
            const busId = node.config?.camera_bus_id;
            if (busId !== undefined && busId !== null) {
                busIds.add(String(busId));
            }
        }
    });
    return Array.from(busIds);
}

/**
 * Formats the camera note text and title for a list of bus IDs.
 *
 * @param {Array<string>} busIds
 * @returns {{ text: string, title: string }}
 */
export function formatPipelineCameraNote(busIds) {
    if (busIds.length === 0) {
        return { text: "No camera bus IDs configured", title: "" };
    }
    const sortedBusIds = [...busIds].sort((first, second) =>
        first.localeCompare(second, undefined, { sensitivity: "accent" }),
    );
    if (sortedBusIds.length <= 2) {
        return {
            text: `Bus IDs: ${sortedBusIds.join(", ")}`,
            title: sortedBusIds.join(", "),
        };
    }
    const visibleBusIds = sortedBusIds.slice(0, 2).join(", ");
    return {
        text: `Bus IDs: ${visibleBusIds} (+${sortedBusIds.length - 2} more)`,
        title: sortedBusIds.join(", "),
    };
}

/**
 * Updates the pipeline camera note element from the current bus IDs.
 */
export function updatePipelineCameraNote() {
    const pipelineCameraNote = creatorContext.elements.pipelineCameraNote;
    if (!pipelineCameraNote) {
        return;
    }
    const busIds = getDeviceInputBusIds();
    const note = formatPipelineCameraNote(busIds);
    pipelineCameraNote.textContent = note.text;
    pipelineCameraNote.title = note.title;
}
