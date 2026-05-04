import { pipelineHistory } from "../pipelineHistory.js";

export function initializePipelineHistory(pipelineStore, callbacks) {
    pipelineHistory.init(pipelineStore, callbacks);
}

export function bindHistoryButtons(undoButton, redoButton) {
    pipelineHistory.setButtons(undoButton, redoButton);
}

export function attachHistoryKeyboardShortcuts() {
    document.addEventListener("keydown", (event) => {
        const pipelineView = document.getElementById("view-pipeline");
        if (pipelineView?.classList.contains("hidden")) return;

        const target = event.target;
        if (
            target.tagName === "INPUT" ||
            target.tagName === "TEXTAREA" ||
            target.tagName === "SELECT" ||
            target.isContentEditable
        )
            return;

        const key = event.key.toLowerCase();
        const isUndo =
            (event.ctrlKey || event.metaKey) && key === "z" && !event.shiftKey;
        const isRedo =
            (event.ctrlKey || event.metaKey) && key === "z" && event.shiftKey;

        if (!isUndo && !isRedo) return;

        if (
            globalThis.flowchartRenderer?.connections?.manualPathCreator
                ?.isActive
        )
            return;

        event.preventDefault();

        if (isUndo) {
            pipelineHistory.undo();
        } else {
            pipelineHistory.redo();
        }
    });
}
