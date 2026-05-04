import { creatorContext } from "./context.js";

export function cachePipelineCreatorElements() {
    const { elements } = creatorContext;

    elements.pipelineArea = document.getElementById("pipelineArea");
    elements.pipelinePlaceholder = document.getElementById(
        "pipelinePlaceholder",
    );
    elements.operationsList = document.getElementById("operationsList");
    elements.runButton = document.getElementById("runButton");
    elements.pipelineSelect = document.getElementById("pipelineSelect");
    elements.pipelineCameraNote = document.getElementById(
        "pipelineCameraNote",
    );
    elements.newPipelineButton = document.getElementById("newPipelineButton");
    elements.deletePipelineButton = document.getElementById(
        "deletePipelineButton",
    );
    elements.undoButton = document.getElementById("undoButton");
    elements.redoButton = document.getElementById("redoButton");
    elements.restartIndicator = document.getElementById("restartIndicator");
    elements.flowchartCanvas = document.getElementById("flowchartCanvas");
    elements.executionTimestepsList = document.getElementById(
        "executionTimestepsList",
    );
    elements.executionSummaryContent = document.getElementById(
        "executionSummaryContent",
    );
    elements.profilingDetailsOverlay = document.getElementById(
        "profilingDetailsOverlay",
    );
    elements.profilingDetailsModal = document.getElementById(
        "profilingDetailsModal",
    );
    elements.profilingDetailsBackdrop = document.getElementById(
        "profilingDetailsBackdrop",
    );
    elements.profilingDetailsBody = document.getElementById(
        "profilingDetailsBody",
    );
    elements.profilingDetailsTitle = document.getElementById(
        "profilingDetailsTitle",
    );
    elements.profilingDetailsCloseButton = document.getElementById(
        "profilingDetailsCloseButton",
    );
    elements.profilingDetailsInfoButton = document.getElementById(
        "profilingDetailsInfoButton",
    );
    elements.profilingDetailsAverageCheckbox = document.getElementById(
        "profilingDetailsAverageCheckbox",
    );

    return elements;
}
