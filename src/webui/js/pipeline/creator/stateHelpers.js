import { creatorContext } from "./context.js";
import { pipelineStore } from "../PipelineStore.js";

export function getOperations() {
    return pipelineStore.state.operations;
}

export function getPipeline() {
    return pipelineStore.getNodesForRenderer();
}

export function getPipelines() {
    return pipelineStore.state.pipelines;
}

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

export function getDeviceInputNodes() {
    return pipelineStore.getNodes().filter((node) => {
        return (
            pipelineStore.normalizeOperationId(node.operationId) ===
            "device_input"
        );
    });
}

export function getDeviceInputBusIds() {
    const busIds = new Set();
    pipelineStore.getNodes().forEach((node) => {
        const operationId = pipelineStore.normalizeOperationId(
            node.operationId,
        );
        if (operationId === "device_input") {
            const busId = node.config?.bus_id;
            if (busId !== undefined && busId !== null) {
                busIds.add(String(busId));
            }
        }
    });
    return Array.from(busIds);
}

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
