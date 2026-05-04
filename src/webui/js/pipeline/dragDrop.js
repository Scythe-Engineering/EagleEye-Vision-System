// Drag-and-drop helpers for the pipeline operations panel.
/**
 * Drag payload for the operations panel → flowchart canvas. The canvas drop
 * handler reads `application/pipeline` / `text/plain` (see FlowchartRenderer).
 *
 * @param {DragEvent} e
 * @param {Object} item
 * @param {*} _operations
 * @param {number|null} fromIndex
 */
export function handleDragStart(e, item, _operations, fromIndex = null) {
    e.dataTransfer.effectAllowed = fromIndex !== null ? "move" : "copy";

    try {
        const payload = JSON.stringify({
            id: item.id,
            instanceId: item.instanceId || null,
            fromIndex,
        });
        e.dataTransfer.setData("application/pipeline", payload);
        e.dataTransfer.setData("text/plain", payload);
    } catch (err) {
        console.warn("Failed to set drag data:", err);
    }

    if (e.currentTarget instanceof HTMLElement) {
        e.currentTarget.classList.add("dragging");
    }
}
