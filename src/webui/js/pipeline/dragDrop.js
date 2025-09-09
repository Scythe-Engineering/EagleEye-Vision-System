import { uid, parseDropPayload } from "./utils.js";

// Drop indicator creation/removal
export function createDropIndicator() {
    const div = document.createElement("div");
    div.className = "h-1 bg-orange-500 rounded-full my-2 animate-pulse";
    div.style.pointerEvents = "none";
    return div;
}

export function removeDropIndicator() {
    const dropIndicatorElem = document.querySelector(".drop-indicator");
    if (dropIndicatorElem?.parentNode) {
        dropIndicatorElem.parentNode.removeChild(dropIndicatorElem);
    }
}

// Global drag state
let draggedItem = null;
let draggedFromIndex = null;
let dropIndicatorIndex = null;
let dropIndicatorElem = null;
let isProcessingDrop = false;
let pipelineDragDepth = 0;
let isDragOverScheduled = false;
let lastDragOverEvent = null;

// Drag/drop handlers
export function handleDragStart(e, item, operations, fromIndex = null) {
    draggedItem = item;
    draggedFromIndex = fromIndex;
    e.dataTransfer.effectAllowed = fromIndex !== null ? "move" : "copy";

    // Store a small payload so if external listeners inspect dataTransfer
    // they get something useful.
    try {
        const payload = JSON.stringify({
            id: item.id,
            instanceId: item.instanceId || null,
            fromIndex,
        });
        e.dataTransfer.setData("application/pipeline", payload);
        e.dataTransfer.setData("text/plain", payload);
    } catch (err) {
        // some browsers restrict certain mime types during drag
        console.warn("Failed to set drag data:", err);
    }

    // Visual cue
    if (e.currentTarget instanceof HTMLElement) {
        e.currentTarget.classList.add("dragging");
    }
}

export function handleDragEnd(
    e,
    pipelineContainer,
    pipelinePlaceholder,
    pipeline,
) {
    if (e.currentTarget instanceof HTMLElement) {
        e.currentTarget.classList.remove("dragging");
        e.currentTarget.style.opacity = "";
    }
    draggedItem = null;
    draggedFromIndex = null;
    removeDropIndicator();
}

export function handleDragEnterPipeline(e) {
    e.stopPropagation();
    pipelineDragDepth += 1;
    // Hide placeholder on first enter to ensure correct indicator positioning
    const pipelinePlaceholder = document.getElementById("pipelinePlaceholder");
    if (pipelineDragDepth === 1 && pipelinePlaceholder) {
        pipelinePlaceholder.classList.add("hidden");
    }
}

export function handleDragOverPipeline(e, pipeline, pipelineContainer) {
    e.preventDefault();
    e.stopPropagation();
    if (e.target === dropIndicatorElem) return;

    lastDragOverEvent = e;
    if (isDragOverScheduled) return;
    isDragOverScheduled = true;
    requestAnimationFrame(() => {
        isDragOverScheduled = false;
        const evt = lastDragOverEvent;
        if (!evt) return;

        evt.dataTransfer.dropEffect =
            draggedFromIndex !== null ? "move" : "copy";

        // Build array of items excluding the dragged one (so positions are stable)
        const nonDragged = pipeline.filter(
            (it) => !(draggedItem && draggedItem.instanceId === it.instanceId),
        );

        // compute insertion index among non-dragged items
        const mouseY = evt.clientY;
        let k = nonDragged.length;
        for (let i = 0; i < nonDragged.length; i++) {
            const id = nonDragged[i].instanceId;
            const el = pipelineContainer.querySelector(
                `[data-instance-id="${id}"]`,
            );
            if (!el) continue;
            const box = el.getBoundingClientRect();
            if (mouseY < box.top + box.height / 2) {
                k = i;
                break;
            }
        }

        if (dropIndicatorIndex === k) return;
        dropIndicatorIndex = k;

        removeDropIndicator();
        dropIndicatorElem = createDropIndicator();
        dropIndicatorElem.classList.add("drop-indicator");

        if (nonDragged.length === 0) {
            pipelineContainer.appendChild(dropIndicatorElem);
        } else if (k < nonDragged.length) {
            const refId = nonDragged[k].instanceId;
            const refEl = pipelineContainer.querySelector(
                `[data-instance-id="${refId}"]`,
            );
            if (refEl) {
                pipelineContainer.insertBefore(dropIndicatorElem, refEl);
            } else {
                pipelineContainer.appendChild(dropIndicatorElem);
            }
        } else {
            pipelineContainer.appendChild(dropIndicatorElem);
        }
    });
}

export function handleDragLeavePipeline(e, pipeline, pipelinePlaceholder) {
    e.stopPropagation();
    pipelineDragDepth = Math.max(0, pipelineDragDepth - 1);
    if (pipelineDragDepth === 0) {
        removeDropIndicator();
        dropIndicatorIndex = null;
        if (pipeline.length === 0) {
            pipelinePlaceholder.classList.remove("hidden");
        }
    }
}

export function handleDropOnPipeline(
    e,
    pipeline,
    operations,
    pipelineContainer,
    pipelinePlaceholder,
    callbacks,
) {
    e.preventDefault();
    e.stopPropagation();
    if (isProcessingDrop) return;
    isProcessingDrop = true;

    let localDraggedItem = draggedItem;
    let localFromIndex = draggedFromIndex;

    if (!localDraggedItem) {
        const payload = parseDropPayload(e.dataTransfer);
        if (!payload) {
            isProcessingDrop = false;
            return;
        }
        if (payload.instanceId) {
            const idx = pipeline.findIndex(
                (it) => it.instanceId === payload.instanceId,
            );
            if (idx === -1) {
                isProcessingDrop = false;
                return;
            }
            localDraggedItem = pipeline[idx];
            localFromIndex = idx;
        } else if (payload.id) {
            const op = operations.find((o) => o.id === payload.id);
            if (!op) {
                isProcessingDrop = false;
                return;
            }
            localDraggedItem = op;
            localFromIndex = null;
        } else {
            isProcessingDrop = false;
            return;
        }
    }

    // compute insertion index among non-dragged items
    let k = 0;
    if (
        dropIndicatorElem &&
        dropIndicatorElem.parentNode === pipelineContainer
    ) {
        const children = Array.from(pipelineContainer.childNodes);
        for (const node of children) {
            if (node === dropIndicatorElem) break;
            if (
                node.nodeType === Node.ELEMENT_NODE &&
                node.classList.contains("pipeline-item")
            ) {
                k += 1;
            }
        }
    } else {
        // fallback: compute among non-dragged items by mouse position
        const nonDragged = pipeline.filter(
            (it) =>
                !(
                    localDraggedItem &&
                    localDraggedItem.instanceId === it.instanceId
                ),
        );
        const mouseY = e.clientY;
        k = nonDragged.length;
        for (let i = 0; i < nonDragged.length; i++) {
            const id = nonDragged[i].instanceId;
            const el = pipelineContainer.querySelector(
                `[data-instance-id="${id}"]`,
            );
            if (!el) continue;
            const box = el.getBoundingClientRect();
            if (mouseY < box.top + box.height / 2) {
                k = i;
                break;
            }
        }
    }

    // derive final insertion index into original pipeline array
    let finalIndex = k;

    if (localFromIndex !== null) {
        // Reordering existing item
        if (localFromIndex !== finalIndex) {
            const removedItem = pipeline.splice(localFromIndex, 1)[0];
            const newPipeline = pipeline.slice();
            newPipeline.splice(finalIndex, 0, removedItem);
            pipeline.length = 0;
            pipeline.push(...newPipeline);
            callbacks.renderPipeline();

            // Auto-save when reordering items
            setTimeout(() => {
                if (window.pipelineCreator?.autoSavePipeline) {
                    window.pipelineCreator.autoSavePipeline();
                }
            }, 100);
        }
    } else {
        // Adding new item from operations panel
        const newItem = {
            ...localDraggedItem,
            instanceId: uid(localDraggedItem.id + "-"),
        };
        const newPipeline = pipeline.slice();
        newPipeline.splice(finalIndex, 0, newItem);
        pipeline.length = 0;
        pipeline.push(...newPipeline);
        callbacks.renderPipeline();

        // Auto-save when adding new items
        setTimeout(() => {
            if (window.pipelineCreator?.autoSavePipeline) {
                window.pipelineCreator.autoSavePipeline();
            }
        }, 100);
    }

    removeDropIndicator();
    draggedItem = null;
    draggedFromIndex = null;
    dropIndicatorIndex = null;

    // If still empty (should not be after add), show placeholder
    if (pipeline.length === 0) {
        pipelinePlaceholder.classList.remove("hidden");
    }

    isProcessingDrop = false;
}
