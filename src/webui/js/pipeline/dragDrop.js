import { uid, parseDropPayload } from "./utils.js";

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

let draggedItem = null;
let draggedFromIndex = null;
let dropIndicatorIndex = null;
let dropIndicatorElem = null;
let isProcessingDrop = false;
let pipelineDragDepth = 0;
let isDragOverScheduled = false;
let lastDragOverEvent = null;

export function handleDragStart(e, item, operations, fromIndex = null) {
    draggedItem = item;
    draggedFromIndex = fromIndex;
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

        const nonDragged = pipeline.filter(
            (it) => !(draggedItem && draggedItem.instanceId === it.instanceId),
        );

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

    let finalIndex = k;

    if (localFromIndex !== null) {
        if (localFromIndex < finalIndex) {
            finalIndex -= 1;
        }

        if (localFromIndex !== finalIndex) {

            const removedItem = pipeline.splice(localFromIndex, 1)[0];
            const newPipeline = pipeline.slice();
            newPipeline.splice(finalIndex, 0, removedItem);
            pipeline.length = 0;
            pipeline.push(...newPipeline);
            callbacks.renderPipeline();

            setTimeout(() => {
                if (callbacks.autoSavePipeline) {
                    callbacks.autoSavePipeline();
                }
            }, 100);
        }
    } else {
        const newItem = {
            ...localDraggedItem,
            instanceId: uid(localDraggedItem.id + "-"),
        };
        const newPipeline = pipeline.slice();
        newPipeline.splice(finalIndex, 0, newItem);
        pipeline.length = 0;
        pipeline.push(...newPipeline);
        callbacks.renderPipeline();

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

    if (pipeline.length === 0) {
        pipelinePlaceholder.classList.remove("hidden");
    }

    isProcessingDrop = false;
}

export class FlowchartDragDropHandler {
    constructor(canvas, options = {}) {
        this.canvas = canvas;
        this.gridSpacing = options.gridSpacing || 20;
        this.onDrop = options.onDrop || (() => {});
        this.onReorder = options.onReorder || (() => {});
        
        this.isDragging = false;
        this.draggedOperation = null;
        this.dragPreview = null;
        
        this.init();
    }

    init() {
        this.setupOperationsPanelDrag();
    }

    setupOperationsPanelDrag() {
        const operationsList = document.getElementById("operationsList");
        if (!operationsList) return;

        operationsList.addEventListener("dragstart", (e) => {
            const operationEl = e.target.closest("[draggable]");
            if (!operationEl) return;

            this.isDragging = true;
            this.createDragPreview(e);
        });

        operationsList.addEventListener("dragend", () => {
            this.isDragging = false;
            this.removeDragPreview();
        });
    }

    createDragPreview(e) {
        this.dragPreview = document.createElement("div");
        this.dragPreview.className = "fixed pointer-events-none z-50 opacity-80";
        this.dragPreview.style.cssText = `
            background-color: #232323;
            border: 2px dashed #f9c845;
            border-radius: 8px;
            padding: 8px 16px;
            color: white;
            font-size: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        `;
        this.dragPreview.textContent = "Drop to add operation";
        
        document.body.appendChild(this.dragPreview);
        
        const updatePosition = (event) => {
            if (this.dragPreview) {
                this.dragPreview.style.left = `${event.clientX + 15}px`;
                this.dragPreview.style.top = `${event.clientY + 15}px`;
            }
        };
        
        document.addEventListener("drag", updatePosition);
        document.addEventListener("dragend", () => {
            document.removeEventListener("drag", updatePosition);
        }, { once: true });
    }

    removeDragPreview() {
        if (this.dragPreview) {
            this.dragPreview.remove();
            this.dragPreview = null;
        }
    }

    snapToGrid(value) {
        return Math.round(value / this.gridSpacing) * this.gridSpacing;
    }

    getDropPosition(e) {
        const worldPos = this.canvas.screenToWorld(e.clientX, e.clientY);
        return {
            x: this.snapToGrid(worldPos.x),
            y: this.snapToGrid(worldPos.y)
        };
    }

    calculateInsertIndex(pipeline, dropY) {
        if (pipeline.length === 0) return 0;

        for (let i = 0; i < pipeline.length; i++) {
            const item = pipeline[i];
            const itemY = item.position?.y || 0;
            if (dropY < itemY) {
                return i;
            }
        }

        return pipeline.length;
    }

    destroy() {
        this.removeDragPreview();
    }
}
