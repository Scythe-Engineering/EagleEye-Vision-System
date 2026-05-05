import { pipelineHistory } from "../pipelineHistory.js";

/**
 * History controller for wiring pipeline undo/redo state to UI controls and keyboard shortcuts.
 */
export function initializePipelineHistory(pipelineStore, callbacks) {
    pipelineHistory.init(pipelineStore, callbacks);
}

/**
 * Bind the undo and redo buttons to the shared pipeline history instance.
 */
export function bindHistoryButtons(undoButton, redoButton) {
    pipelineHistory.setButtons(undoButton, redoButton);
}

/**
 * Attach global keyboard shortcuts for pipeline history actions.
 */
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
