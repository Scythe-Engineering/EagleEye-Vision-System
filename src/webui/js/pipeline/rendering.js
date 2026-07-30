// Renders and manages the web UI pipeline/flowchart view.
import { escapeHtml } from "./utils.js";
import { FlowchartCanvas } from "./flowchartCanvas.js";
import { FlowchartNode } from "./flowchartNode.js";
import { FlowchartConnections } from "./flowchartConnections.js";
import { FlowchartMinimap } from "./flowchartMinimap.js";
import { findCycles, findUnreachableIslands } from "./graphUtils.js";
import { pipelineStore } from "./PipelineStore.js";
import { prefetchConfigs } from "./operationConfigCache.js";
import { hideTooltip, showTooltip } from "../ui/tooltip.js";

/**
 * Kept for compatibility with existing callers; actual tooltips are created lazily by the shared UI helper.
 */
export function createDescriptionPopup() {
    // Kept for existing callers; tooltips are lazily created by the shared UI helper.
}

/**
 * Shows a tooltip for an operation description.
 *
 * @param {string} name Operation name.
 * @param {string} description Operation description.
 * @param {Event} event Triggering event.
 */
export function showDescriptionPopup(name, description, event) {
    showTooltip(event.currentTarget, {
        html: `
            <div class="text-[#f9c845] font-semibold text-sm mb-2 border-b border-[#404040] pb-2">${escapeHtml(name)}</div>
            <div class="text-white text-xs">${escapeHtml(description)}</div>
        `,
    });
}

/**
 * Hides the shared description tooltip.
 */
export function hideDescriptionPopup() {
    hideTooltip();
}

/**
 * Attaches tooltip hover listeners to an element.
 *
 * @param {HTMLElement} element Target element.
 * @param {string} name Operation name.
 * @param {string} description Operation description.
 */
export function addHoverListeners(element, name, description) {
    element.addEventListener("mouseenter", (e) => {
        showDescriptionPopup(name, description, e);
    });

    element.addEventListener("mouseleave", () => {
        hideTooltip(element);
    });
}

const FOLDER_ORDER = [
    "Input",
    "Detection",
    "Preprocessing",
    "Localization",
    "Filtering",
    "Networking",
    "Output",
];

/**
 * Returns the SVG icon used for a folder label.
 *
 * @param {string} folderName Folder name.
 * @returns {string} SVG markup.
 */
function getFolderIcon(folderName) {
    const icons = {
        Input: `<svg class="w-3.5 h-3.5 shrink-0" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="1" y="4" width="14" height="10" rx="1.5"/><path d="M5 4V3a2 2 0 0 1 4 0v1"/></svg>`,
        Detection: `<svg class="w-3.5 h-3.5 shrink-0" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="8" cy="8" r="3"/><path d="M1 8h2M13 8h2M8 1v2M8 13v2"/></svg>`,
        Preprocessing: `<svg class="w-3.5 h-3.5 shrink-0" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M3 3h2v10H3zM7 6h2v7H7zM11 1h2v12h-2z"/></svg>`,
        Localization: `<svg class="w-3.5 h-3.5 shrink-0" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="8" cy="7" r="3"/><path d="M8 10c0 0-5 3-5 0a5 5 0 0 1 10 0c0 3-5 0-5 0z"/></svg>`,
        Filtering: `<svg class="w-3.5 h-3.5 shrink-0" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M2 4h12M4 8h8M6 12h4"/></svg>`,
        Networking: `<svg class="w-3.5 h-3.5 shrink-0" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="8" cy="8" r="3"/><path d="M8 1a7 7 0 1 0 0 14A7 7 0 0 0 8 1zM1 8h14M8 1c-2 2-3 4-3 7s1 5 3 7M8 1c2 2 3 4 3 7s-1 5-3 7"/></svg>`,
        Output: `<svg class="w-3.5 h-3.5 shrink-0" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M8 2v9M4 7l4 4 4-4M2 13h12"/></svg>`,
    };
    return (
        icons[folderName] ??
        `<svg class="w-3.5 h-3.5 shrink-0" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M2 5h5l2 2h5v7H2z"/></svg>`
    );
}

/**
 * Creates a draggable operation card for the operations list.
 *
 * @param {object} op Operation metadata.
 * @param {Array<object>} operations All operations.
 * @param {Function} openOperationSettings Settings callback.
 * @param {Function} handleDragStart Drag-start callback.
 * @returns {HTMLElement} The created card element.
 */
function createOperationCard(
    op,
    operations,
    openOperationSettings,
    handleDragStart,
) {
    const el = document.createElement("div");
    el.draggable = true;
    el.className =
        "bg-[#232323] border-2 border-[#404040] rounded-xl p-4 cursor-move hover:border-[#f9c845] transition-all transform hover:scale-105 hover:shadow-lg mb-2 group";
    el.style.boxShadow = "4px 4px 8px rgba(0, 0, 0, 0.4)";
    el.innerHTML = `
        <div class="flex items-center gap-3">
          <div>
            <h3 class="font-medium text-white truncate max-w-[210px]">${escapeHtml(op.name)}</h3>
          </div>
          <div class="ml-auto">
            <button class="op-settings-btn p-2 hover:bg-[#404040] rounded-lg transition-all" title="Settings">
              <img src="../../../assets/settings.svg" alt="Settings" class="w-4 h-4 icon-grayscale" />
            </button>
          </div>
        </div>
    `;

    el.addEventListener("dragstart", (e) => {
        const rect = e.currentTarget.getBoundingClientRect();
        const offsetX = e.clientX - rect.left;
        const offsetY = e.clientY - rect.top;

        if (globalThis.flowchartRenderer) {
            globalThis.flowchartRenderer.setDragOffset(offsetX, offsetY);
        }

        handleDragStart(e, op, null, operations);
    });
    el.addEventListener("dragend", (e) => {
        if (e.currentTarget instanceof HTMLElement) {
            e.currentTarget.classList.remove("dragging");
            e.currentTarget.style.opacity = "";
        }

        if (globalThis.flowchartRenderer) {
            globalThis.flowchartRenderer.setDragOffset(0, 0);
        }
    });

    addHoverListeners(el, op.name, op.description);

    const settingsBtn = el.querySelector(".op-settings-btn");
    if (settingsBtn) {
        settingsBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            e.preventDefault();
            openOperationSettings(op);
        });
    }

    return el;
}

/**
 * Renders the grouped operations list.
 *
 * @param {Array<object>} operations Operations to render.
 * @param {HTMLElement} operationsList Container element.
 * @param {Function} openOperationSettings Settings callback.
 * @param {Function} handleDragStart Drag-start callback.
 */
export function renderOperations(
    operations,
    operationsList,
    openOperationSettings,
    handleDragStart,
) {
    operationsList.innerHTML = "";

    const byFolder = new Map();
    for (const op of operations) {
        const folder = op.folder || "Uncategorized";
        if (!byFolder.has(folder)) byFolder.set(folder, []);
        byFolder.get(folder).push(op);
    }

    const orderedFolders = [
        ...FOLDER_ORDER.filter((f) => byFolder.has(f)),
        ...[...byFolder.keys()].filter((f) => !FOLDER_ORDER.includes(f)).sort(),
    ];

    let isFirstFolder = true;
    for (const folderName of orderedFolders) {
        const ops = byFolder.get(folderName);

        const section = document.createElement("div");
        section.className = isFirstFolder ? "mb-1" : "mt-1 mb-1";

        const header = document.createElement("button");
        header.type = "button";
        header.className =
            "w-full flex items-center gap-2 px-2 py-2 rounded-lg hover:bg-[#2a2a2a] transition-colors group/folder select-none";

        const arrow = document.createElement("span");
        arrow.className =
            "text-[#666] transition-transform duration-200 group-hover/folder:text-[#888]";
        arrow.innerHTML = `<svg class="w-3 h-3" viewBox="0 0 12 12" fill="currentColor"><path d="M2 4l4 4 4-4"/></svg>`;

        const iconEl = document.createElement("span");
        iconEl.className = "text-[#f9c845]";
        iconEl.innerHTML = getFolderIcon(folderName);

        const label = document.createElement("span");
        label.className =
            "text-[#c0c0c0] text-xs font-semibold uppercase tracking-widest group-hover/folder:text-white transition-colors";
        label.textContent = folderName;

        const count = document.createElement("span");
        count.className =
            "ml-auto text-[#555] text-xs font-mono group-hover/folder:text-[#666] transition-colors";
        count.textContent = ops.length;

        header.appendChild(arrow);
        header.appendChild(iconEl);
        header.appendChild(label);
        header.appendChild(count);

        const contentWrapper = document.createElement("div");
        contentWrapper.className = "relative mt-1";

        const braceCol = document.createElement("div");
        braceCol.style.cssText =
            "position:absolute;left:0;top:4px;bottom:4px;width:12px;pointer-events:none;";

        const braceSvg = document.createElementNS(
            "http://www.w3.org/2000/svg",
            "svg",
        );
        braceSvg.setAttribute("width", "12");
        braceSvg.setAttribute("viewBox", "0 0 12 100");
        braceSvg.setAttribute("preserveAspectRatio", "none");
        braceSvg.style.cssText =
            "display:block;width:12px;height:100%;transition:opacity 0.25s ease;";

        const bracePath = document.createElementNS(
            "http://www.w3.org/2000/svg",
            "path",
        );
        bracePath.setAttribute(
            "d",
            "M11,2 C11,2 3,2 3,12 L3,44 C3,44 1,50 3,56 L3,88 C3,88 3,98 11,98",
        );
        bracePath.setAttribute("fill", "none");
        bracePath.setAttribute("stroke", "#484848");
        bracePath.setAttribute("stroke-width", "1.5");
        bracePath.setAttribute("stroke-linecap", "round");
        bracePath.setAttribute("vector-effect", "non-scaling-stroke");

        braceSvg.appendChild(bracePath);
        braceCol.appendChild(braceSvg);

        const body = document.createElement("div");
        body.className = "pl-4 pr-0.5";

        for (const op of ops) {
            body.appendChild(
                createOperationCard(
                    op,
                    operations,
                    openOperationSettings,
                    handleDragStart,
                ),
            );
        }

        body.addEventListener("mouseenter", () => {
            braceSvg.style.opacity = "0";
        });
        body.addEventListener("mouseleave", () => {
            braceSvg.style.opacity = "1";
        });

        contentWrapper.appendChild(braceCol);
        contentWrapper.appendChild(body);

        header.addEventListener("click", () => {
            const isOpen = !contentWrapper.classList.contains("hidden");
            if (isOpen) {
                contentWrapper.classList.add("hidden");
                arrow.style.transform = "rotate(-90deg)";
            } else {
                contentWrapper.classList.remove("hidden");
                arrow.style.transform = "";
            }
        });

        section.appendChild(header);
        section.appendChild(contentWrapper);

        if (!isFirstFolder) {
            const divider = document.createElement("div");
            divider.className = "h-px bg-[#2a2a2a] my-1";
            operationsList.appendChild(divider);
        }
        operationsList.appendChild(section);

        isFirstFolder = false;
    }
}

export class FlowchartRenderer {
    /**
     * Creates a renderer for the pipeline canvas.
     *
     * @param {HTMLElement} canvasContainer Canvas container element.
     * @param {object} [options={}] Renderer options.
     */
    constructor(canvasContainer, options = {}) {
        this.canvasContainer = canvasContainer;
        this.canvas = null;
        this.connections = null;
        this.minimap = null;
        this.nodes = new Map();
        this.pipeline = [];
        this.islandBlocks = [];

        this.callbacks = {
            openOperationSettings: options.openOperationSettings || (() => {}),
            updateRunButton: options.updateRunButton || (() => {}),
            removeFromPipeline: options.removeFromPipeline || (() => {}),
            onPipelineChange: options.onPipelineChange || (() => {}),
            autoSavePipeline: options.autoSavePipeline || (() => {}),
        };

        this.gridSpacing = options.gridSpacing || 20;
        this.nodeSpacingX = options.nodeSpacingX || 300;
        this.nodeSpacingY = options.nodeSpacingY || 150;

        this.dragOffsetX = 0;
        this.dragOffsetY = 0;
        this.pendingPositionFrame = null;
        this.pendingDragNodeIds = new Set();
        this.lastLayoutChromeUpdateMs = 0;
        this.lastHighlightedInputPort = null;

        this.init();
    }

    /**
     * Initializes the canvas, minimap, and connection manager.
     */
    init() {
        this.canvas = new FlowchartCanvas(this.canvasContainer, {
            gridSpacing: this.gridSpacing,
            onViewportChange: this.handleViewportChange.bind(this),
        });

        this.minimap = new FlowchartMinimap(this.canvas, {
            width: 180,
            height: 120,
            padding: 10,
            connectionColor: "#f9c845",
        });
        this.minimap.attachTo(this.canvasContainer);

        this.connections = new FlowchartConnections(
            this.canvas.getConnectionsLayer(),
            {
                connectionColor: "#f9c845",
                connectionWidth: 2,
                onConnectionRemoved: this.handleConnectionRemoved.bind(this),
                onConnectionChanged: this.handleConnectionChanged.bind(this),
                onGetNode: (nodeId) => this.nodes.get(nodeId),
                onCheckDefaultAllowed: (nodeId, portName) => {
                    const node = this.nodes.get(nodeId);
                    if (!node) return false;
                    return node.canInputPortBeDefault(portName);
                },
                canvas: this.canvas,
            },
        );

        this.dragGhost = null;
        this.setupDropZone();
    }

    /**
     * Stores the current drag offset used for ghost placement.
     *
     * @param {number} offsetX Horizontal offset.
     * @param {number} offsetY Vertical offset.
     */
    setDragOffset(offsetX, offsetY) {
        this.dragOffsetX = offsetX;
        this.dragOffsetY = offsetY;
    }

    /**
     * Registers drag-and-drop handlers for the pipeline area.
     */
    setupDropZone() {
        const pipelineArea = document.getElementById("pipelineArea");
        const canvasContainer = this.canvasContainer;

        const handleDragOver = (e) => {
            e.preventDefault();
            e.stopPropagation();
            e.dataTransfer.dropEffect = "copy";

            this.updateDragGhost(e);
        };

        const handleDragEnter = (e) => {
            e.preventDefault();
            e.stopPropagation();
            this.createDragGhost(e);
        };

        const handleDragLeave = (e) => {
            e.preventDefault();
            e.stopPropagation();

            // Only remove if we're actually leaving the canvas area
            const rect = canvasContainer.getBoundingClientRect();
            if (
                e.clientX <= rect.left ||
                e.clientX >= rect.right ||
                e.clientY <= rect.top ||
                e.clientY >= rect.bottom
            ) {
                this.removeDragGhost();
            }
        };

        const handleDrop = async (e) => {
            e.preventDefault();
            e.stopPropagation();
            this.removeDragGhost();
            await this.handleDrop(e);
        };

        // Add listeners to both the area and the container to be safe
        [pipelineArea, canvasContainer].forEach((elem) => {
            if (elem) {
                elem.addEventListener("dragover", handleDragOver);
                elem.addEventListener("dragenter", handleDragEnter);
                elem.addEventListener("dragleave", handleDragLeave);
                elem.addEventListener("drop", handleDrop);
            }
        });
    }

    /**
     * Creates the drag ghost shown while placing a node.
     *
     * @param {DragEvent} e Drag event.
     */
    createDragGhost(e) {
        if (this.dragGhost) return;

        this.dragGhost = document.createElement("div");
        this.dragGhost.className = "flowchart-node-ghost";
        Object.assign(this.dragGhost.style, {
            position: "absolute",
            width: "200px",
            height: "80px",
            backgroundColor: "rgba(249, 200, 69, 0.05)",
            border: "2px dashed #f9c845",
            borderRadius: "12px",
            zIndex: "1000",
            pointerEvents: "none",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "#f9c845",
            fontWeight: "bold",
            fontSize: "12px",
            boxShadow: "0 0 15px rgba(249, 200, 69, 0.2)",
        });
        this.dragGhost.textContent = "Place Operation";

        this.canvas.getNodesLayer().appendChild(this.dragGhost);
    }

    /**
     * Updates the drag ghost position.
     *
     * @param {DragEvent} e Drag event.
     */
    updateDragGhost(e) {
        if (!this.dragGhost) {
            this.createDragGhost(e);
        }

        const scale = this.canvas.scale || 1;
        const worldPos = this.canvas.screenToWorld(e.clientX, e.clientY);
        const adjustedX = worldPos.x - this.dragOffsetX / scale;
        const adjustedY = worldPos.y - this.dragOffsetY / scale;
        const snappedPos = this.canvas.snapPositionToGrid(adjustedX, adjustedY);

        this.dragGhost.style.left = `${snappedPos.x}px`;
        this.dragGhost.style.top = `${snappedPos.y}px`;
    }

    /**
     * Removes the drag ghost if it exists.
     */
    removeDragGhost() {
        if (this.dragGhost) {
            this.dragGhost.remove();
            this.dragGhost = null;
        }
    }

    /**
     * Handles dropping an operation onto the canvas.
     *
     * @param {DragEvent} e Drop event.
     */
    async handleDrop(e) {
        let dropData = null;

        try {
            const dataTransfer = e.dataTransfer;
            const jsonData =
                dataTransfer?.getData("application/pipeline") ||
                dataTransfer?.getData("text/plain");
            if (jsonData) {
                dropData = JSON.parse(jsonData);
            }
        } catch (err) {
            console.warn("[FLOWCHART] Failed to parse drop data:", err);
            return;
        }

        if (!dropData?.id) {
            console.warn("[FLOWCHART] Invalid drop data", dropData);
            return;
        }

        if (dropData.instanceId) {
            return;
        }

        const scale = this.canvas.scale || 1;
        const worldPos = this.canvas.screenToWorld(e.clientX, e.clientY);
        const adjustedX = worldPos.x - this.dragOffsetX / scale;
        const adjustedY = worldPos.y - this.dragOffsetY / scale;
        const snappedPos = this.canvas.snapPositionToGrid(adjustedX, adjustedY);

        await this.callbacks.onPipelineChange({
            type: "add",
            operationId: dropData.id,
            position: snappedPos,
        });
    }

    /**
     * Renders a pipeline into the canvas.
     *
     * @param {Array<object>} pipeline Pipeline nodes.
     * @param {object} [options={}] Render options.
     */
    async renderPipeline(pipeline, options = {}) {
        this.pipeline = pipeline;

        this.nodes.forEach((node) => node.destroy());
        this.nodes.clear();
        this.clearIslandBlocks();

        // Only clear connections if we're not preserving them (default behavior for loading saved connections)
        if (!options.preserveConnections) {
            this.connections.clearAllConnections();
        }

        const placeholder = document.getElementById("pipelinePlaceholder");

        if (placeholder) {
            const selectedPipeline =
                pipelineStore.state.currentPipeline?.pipelineName;
            const shouldShow = !selectedPipeline;
            placeholder.classList.toggle("hidden", !shouldShow);

            if (shouldShow) {
                const mainText = placeholder.querySelector("p.text-lg");
                const hintText = placeholder.querySelector("p:last-child");
                if (mainText) {
                    mainText.textContent = "Make a new pipeline";
                }
                if (hintText) {
                    hintText.textContent = "Use the New Pipeline button above";
                    hintText.style.color = "#f9c845";
                }

                // Set focus area on grid to a 50x50 square at center of placeholder
                const interactiveGrid = this.canvas?.getInteractiveGrid?.();
                if (interactiveGrid) {
                    const placeholderRect = placeholder.getBoundingClientRect();
                    const containerRect =
                        this.canvasContainer.getBoundingClientRect();

                    // Calculate center of placeholder in screen coordinates
                    const centerScreenX =
                        placeholderRect.left + placeholderRect.width / 2;
                    const centerScreenY =
                        placeholderRect.top + placeholderRect.height / 2;

                    // Create 50x50 unit square centered on placeholder center in screen coordinates
                    const squareScreenLeft = centerScreenX - 25;
                    const squareScreenTop = centerScreenY - 25;

                    // Convert to world coordinates accounting for canvas transforms
                    const relativeRect = {
                        x:
                            (squareScreenLeft -
                                containerRect.left -
                                this.canvas.translateX) /
                            this.canvas.scale,
                        y:
                            (squareScreenTop -
                                containerRect.top -
                                this.canvas.translateY) /
                            this.canvas.scale,
                        width: 50 / this.canvas.scale,
                        height: 50 / this.canvas.scale,
                    };
                    interactiveGrid.setFocusArea(relativeRect);
                }

                // Disable zoom and pan when placeholder is shown
                this.canvas.setPlaceholderVisible(true);

                // Clear minimap when no pipeline is selected
                if (this.minimap) {
                    this.minimap.updateNodes([]);
                    this.minimap.updateConnections([]);
                }
            } else {
                // Clear focus area when placeholder is hidden
                const interactiveGrid = this.canvas?.getInteractiveGrid?.();
                if (interactiveGrid) {
                    interactiveGrid.clearFocusArea();
                }

                // Enable zoom and pan when placeholder is hidden
                this.canvas.setPlaceholderVisible(false);
            }
        }

        if (pipeline.length === 0) {
            // Update grid with empty operation positions to clear caches
            this.updateGridOperationPositions();
            this.updateIslandBlocks();
            // Clear minimap when pipeline is empty
            if (this.minimap) {
                this.minimap.updateNodes([]);
                this.minimap.updateConnections([]);
            }
            this.callbacks.updateRunButton();
            return;
        }

        pipeline.forEach((item, i) => {
            if (!item.position) {
                item.position = this.calculateDefaultPosition(
                    i,
                    pipeline.length,
                );
            }
        });

        const uniqueOps = [
            ...new Map(
                pipeline.map((item) => {
                    const isSecondary = item.isSecondary || false;
                    return [
                        `${item.id}:${isSecondary ? 1 : 0}`,
                        { name: item.id, isSecondary },
                    ];
                }),
            ).values(),
        ];
        await prefetchConfigs(uniqueOps);

        await Promise.all(pipeline.map((item) => this.createNode(item)));

        // Restore connections if provided
        if (options.connections && options.connections.length > 0) {
            this.restoreConnections(options.connections);
        }

        this.syncAllDynamicNodes();
        this.updateDockingLayout();

        this.refreshLayoutChrome();
        this.updateIslandBlocks();

        if (options.centerView !== false) {
            this.centerViewOnNodes();
        }
    }

    /**
     * Refreshes minimap, run button, and island overlays.
     */
    refreshLayoutChrome() {
        if (this.minimap) {
            const nodeDataList = Array.from(this.nodes.values()).map(
                (node) => ({
                    instanceId: node.instanceId,
                    position: node.getPosition(),
                    width: 200,
                    height: 80,
                }),
            );
            this.minimap.updateNodes(nodeDataList);
            this.minimap.updateConnections(
                this.connections.getConnectionData(),
            );
        }
        this.updateGridOperationPositions();
        this.callbacks.updateRunButton();
        this.updateIslandBlocks();
    }

    /**
     * Synchronizes the renderer's pipeline snapshot from the store.
     */
    syncPipelineArrayFromStore() {
        this.pipeline = pipelineStore.getNodesForRenderer();
    }

    /**
     * Append a single new node and wire connections, without full teardown.
     * Keeps pan/zoom. Centers the view only when the graph was previously empty.
     */
    async addNodeFromStore(instanceId) {
        if (!instanceId) {
            return;
        }
        const wasEmpty = this.nodes.size === 0;
        this.syncPipelineArrayFromStore();
        if (this.nodes.has(instanceId)) {
            this.refreshLayoutChrome();
            if (wasEmpty) {
                this.centerViewOnNodes();
            }
            return;
        }

        const item = this.pipeline.find((p) => p.instanceId === instanceId);
        if (!item) {
            return;
        }

        if (!item.position) {
            const index = this.pipeline.findIndex(
                (p) => p.instanceId === item.instanceId,
            );
            item.position = this.calculateDefaultPosition(
                index >= 0 ? index : 0,
                this.pipeline.length,
            );
        }

        await this.createNode(item);

        this.restoreConnections(pipelineStore.getConnectionsForRenderer());
        this.syncAllDynamicNodes();
        this.updateDockingLayout();
        this.updateCycleHighlights();
        this.refreshLayoutChrome();
        this.updateIslandBlocks();
        if (wasEmpty) {
            this.centerViewOnNodes();
        }
    }

    /**
     * Calculates a default node position for a pipeline item.
     *
     * @param {number} index Node index.
     * @param {number} total Total node count.
     * @returns {{x:number,y:number}} Default position.
     */
    calculateDefaultPosition(index, total) {
        const startX = 100;
        const startY = 100;

        return {
            x: this.canvas.snapToGrid(startX + index * this.nodeSpacingX),
            y: this.canvas.snapToGrid(startY + Math.sin(index * 0.5) * 50),
        };
    }

    /**
     * Creates and mounts a node for a pipeline item.
     *
     * @param {object} item Pipeline item.
     * @returns {Promise<FlowchartNode>} Created node.
     */
    async createNode(item) {
        const node = new FlowchartNode(item, {
            gridSpacing: this.gridSpacing,
            onDragStart: this.handleNodeDragStart.bind(this),
            onDragEnd: this.handleNodeDragEnd.bind(this),
            onPositionChange: this.handleNodePositionChange.bind(this),
            onSettingsClick: this.callbacks.openOperationSettings,
            onRemoveClick: this.handleNodeRemove.bind(this),
            onPortHover: this.handlePortHover.bind(this),
            onPortClick: this.handlePortClick.bind(this),
        });

        const element = await node.createElement();
        const nodesLayer = this.canvas.getNodesLayer();

        nodesLayer.appendChild(element);

        this.nodes.set(item.instanceId, node);

        return node;
    }

    /**
     * Synchronizes dynamic ports for a single node.
     *
     * @param {FlowchartNode} node Node to sync.
     * @returns {boolean} Whether ports changed.
     */
    syncNodeDynamicPorts(node) {
        if (!node || !node.dynamicGroup) {
            return false;
        }

        const didChange = node.syncDynamicPorts(
            this.connections.getConnectionData(),
        );
        if (!didChange) {
            return false;
        }

        const changedNodeIds = new Set([node.instanceId]);
        this.connections.updateAllConnections(
            this.nodes,
            changedNodeIds,
            false,
        );
        return true;
    }

    /**
     * Synchronizes dynamic ports for all nodes.
     *
     * @returns {boolean} Whether any node changed.
     */
    syncAllDynamicNodes() {
        let anyChanged = false;
        for (let pass = 0; pass < 3; pass += 1) {
            let changedThisPass = false;
            for (const node of this.nodes.values()) {
                changedThisPass =
                    this.syncNodeDynamicPorts(node) || changedThisPass;
            }
            anyChanged = anyChanged || changedThisPass;
            if (!changedThisPass) {
                break;
            }
        }
        return anyChanged;
    }

    /**
     * Returns a target's docking metadata, including the MX3 contract while
     * config metadata is still loading.
     *
     * @param {FlowchartNode|null} node Target node.
     * @returns {object|null} Docking metadata.
     */
    getDockingMetadata(node) {
        if (node?.docking) return node.docking;
        return String(node?.operationData?.id || "")
            .replace(/\.py$/, "")
            .toLowerCase() === "mx3_async_object_detection"
            ? {
                  source_action: "device_input",
                  source_port: "frame",
                  target_port: "frame",
              }
            : null;
    }

    /**
     * Returns whether a rendered connection matches the destination's docking
     * metadata. Docking remains an ordinary serialized connection.
     *
     * @param {object} connection Renderer connection data.
     * @returns {boolean} Whether this is a docking connection.
     */
    isDockingConnection(connection) {
        const source = this.nodes.get(connection?.fromNodeId);
        const target = this.nodes.get(connection?.toNodeId);
        const docking = this.getDockingMetadata(target);
        const normalize = (value) =>
            String(value || "")
                .replace(/\.py$/, "")
                .toLowerCase()
                .replace(/\s+/g, "_");
        return Boolean(
            source &&
                docking &&
                normalize(source.operationData.id) ===
                    normalize(docking.source_action) &&
                connection.fromPortName === docking.source_port &&
                connection.toPortName === docking.target_port,
        );
    }

    /**
     * Snaps docked detectors beside their Device Input and updates dock chrome.
     *
     * @returns {Set<string>} Instance IDs whose position changed.
     */
    updateDockingLayout() {
        const dockedSources = new Map();
        this.connections.getConnectionData().forEach((connection) => {
            if (this.isDockingConnection(connection)) {
                dockedSources.set(connection.toNodeId, connection.fromNodeId);
            }
        });

        const changedNodeIds = new Set();
        this.nodes.forEach((node, instanceId) => {
            const sourceId = dockedSources.get(instanceId);
            node.setDockState({
                docked: Boolean(sourceId),
                invalid: Boolean(this.getDockingMetadata(node)) && !sourceId,
            });
            if (!sourceId) return;

            const source = this.nodes.get(sourceId);
            if (!source) return;
            const sourcePosition = source.getPosition();
            const sourceWidth =
                source.element?.offsetWidth || source.cachedElementWidth || 200;
            const position = this.canvas.snapPositionToGrid(
                sourcePosition.x + sourceWidth + this.gridSpacing * 2,
                sourcePosition.y,
            );
            const currentPosition = node.getPosition();
            if (
                currentPosition.x === position.x &&
                currentPosition.y === position.y
            ) {
                return;
            }

            node.setPosition(position.x, position.y);
            const item = this.pipeline.find(
                (candidate) => candidate.instanceId === instanceId,
            );
            if (item) item.position = position;
            pipelineStore.updateNodePosition(instanceId, position);
            changedNodeIds.add(instanceId);
        });

        if (changedNodeIds.size > 0) {
            this.connections.updateAllConnections(
                this.nodes,
                changedNodeIds,
                false,
            );
        }
        return changedNodeIds;
    }

    /**
     * Handles the start of a node drag.
     *
     * @param {FlowchartNode} node Dragged node.
     * @param {DragEvent} event Drag event.
     */
    handleNodeDragStart(node, event) {
        node.element.style.zIndex = "100";
    }

    /**
     * Handles the end of a node drag.
     *
     * @param {FlowchartNode} node Dragged node.
     * @param {{x:number,y:number}} position Final position.
     */
    handleNodeDragEnd(node, position) {
        node.element.style.zIndex = "10";
        const dockedNodeIds = this.updateDockingLayout();

        const item = this.pipeline.find(
            (p) => p.instanceId === node.instanceId,
        );
        if (item) {
            item.position = position;
        }
        pipelineStore.updateNodePosition(node.instanceId, position);

        if (this.positionChangeDebounce) {
            clearTimeout(this.positionChangeDebounce);
            this.positionChangeDebounce = null;
        }
        if (this.pendingPositionFrame) {
            cancelAnimationFrame(this.pendingPositionFrame);
            this.pendingPositionFrame = null;
            this.pendingDragNodeIds.clear();
        }

        this.connections.connections.forEach((connection) => {
            if (
                connection.fromNodeId === node.instanceId ||
                connection.toNodeId === node.instanceId
            ) {
                delete connection.lastPosKey;
            }
        });

        const changedNodeIds = new Set([node.instanceId, ...dockedNodeIds]);
        this.connections.updateAllConnections(
            this.nodes,
            changedNodeIds,
            false,
        );

        if (this.minimap) {
            const nodeDataList = Array.from(this.nodes.values()).map((n) => ({
                instanceId: n.instanceId,
                position: n.getPosition(),
                width: 200,
                height: 80,
            }));
            this.minimap.updateNodes(nodeDataList);
            this.minimap.updateConnections(
                this.connections.getConnectionData(),
            );
        }

        this.updateGridOperationPositions();
        this.updateIslandBlocks();

        this.callbacks.autoSavePipeline();
    }

    /**
     * Handles in-progress node position updates.
     *
     * @param {FlowchartNode} node Node being moved.
     * @param {{x:number,y:number}} position Current position.
     */
    handleNodePositionChange(node, position) {
        const dockedNodeIds = this.updateDockingLayout();
        this.pendingDragNodeIds.add(node.instanceId);
        dockedNodeIds.forEach((instanceId) =>
            this.pendingDragNodeIds.add(instanceId),
        );

        if (!this.pendingPositionFrame) {
            this.pendingPositionFrame = requestAnimationFrame(() => {
                const changedNodeIds = new Set(this.pendingDragNodeIds);
                this.pendingDragNodeIds.clear();
                this.pendingPositionFrame = null;
                this.connections.updateAllConnections(
                    this.nodes,
                    changedNodeIds,
                    true,
                );
            });
        }

        const now = performance.now();
        if (
            !this.positionChangeDebounce &&
            now - this.lastLayoutChromeUpdateMs > 90
        ) {
            this.positionChangeDebounce = setTimeout(() => {
                this.lastLayoutChromeUpdateMs = performance.now();
                if (this.minimap) {
                    const nodeDataList = Array.from(this.nodes.values()).map(
                        (n) => ({
                            instanceId: n.instanceId,
                            position: n.getPosition(),
                            width: 200,
                            height: 80,
                        }),
                    );
                    this.minimap.updateNodes(nodeDataList);
                    this.minimap.updateConnections(
                        this.connections.getConnectionData(),
                    );
                }
                this.updateGridOperationPositions();
                this.updateIslandBlocks();
                this.positionChangeDebounce = null;
            }, 90);
        }
    }

    /**
     * Helper method to update the interactive grid with current node positions
     */
    updateGridOperationPositions() {
        if (this.canvas) {
            const nodeDataList = Array.from(this.nodes.values()).map((n) => ({
                position: n.getPosition(),
            }));
            this.canvas.updateOperationPositions(nodeDataList);
        }
    }

    /**
     * Highlights or unhighlights connections for a port.
     *
     * @param {FlowchartNode} node Node owning the port.
     * @param {string} portName Port name.
     * @param {string} portType Port type.
     * @param {boolean} isHovering Whether the port is hovered.
     */
    handlePortHover(node, portName, portType, isHovering) {
        const connectionIds = this.connections.getConnectionsForPort(
            node.instanceId,
            portName,
            portType,
        );

        connectionIds.forEach((id) => {
            this.connections.highlightConnection(id, isHovering);
        });
    }

    /**
     * Handles clicks on input/output ports.
     *
     * @param {FlowchartNode} node Node owning the port.
     * @param {string} portName Port name.
     * @param {string} portType Port type.
     * @param {MouseEvent} event Click event.
     */
    handlePortClick(node, portName, portType, event) {
        if (portType === "output") {
            // Check for existing connections from this output port
            const existingConnections = this.connections.getConnectionsForPort(
                node.instanceId,
                portName,
                "output",
            );

            if (existingConnections.length > 0) {
                // Time-based click vs drag detection
                // < 0.125s = click = delete connection
                // > 0.125s = drag = reconnect
                const mouseDownTime = Date.now();
                const startX = event.clientX;
                const startY = event.clientY;
                let hasMoved = false;
                let connectingStarted = false;

                const onMouseMove = (e) => {
                    const dx = e.clientX - startX;
                    const dy = e.clientY - startY;
                    if (Math.abs(dx) > 5 || Math.abs(dy) > 5) {
                        hasMoved = true;
                        if (!connectingStarted) {
                            connectingStarted = true;
                            this.startConnecting(node, portName, event, true);
                        }
                    }
                };

                const onMouseUp = (e) => {
                    const elapsed = Date.now() - mouseDownTime;
                    globalThis.removeEventListener("mousemove", onMouseMove);
                    globalThis.removeEventListener("mouseup", onMouseUp);

                    if (elapsed < 125 && !hasMoved) {
                        // Quick click without movement - delete all connections from this output
                        if (connectingStarted) {
                            this.cancelConnecting();
                        }
                        existingConnections.forEach((id) =>
                            this.connections.removeConnection(id),
                        );
                        this.callbacks.autoSavePipeline();
                    } else if (!connectingStarted) {
                        // Held long enough but didn't move - start connecting now
                        // Only start connecting if mouse button is still pressed
                        if (e.buttons & 1) {
                            this.startConnecting(node, portName, event, true);
                        }
                    }
                    // If connectingStarted is true and not a quick click, startConnecting handles its own cleanup
                };

                globalThis.addEventListener("mousemove", onMouseMove);
                globalThis.addEventListener("mouseup", onMouseUp);
            } else {
                // No existing connections - just start connecting immediately
                this.startConnecting(node, portName, event);
            }
        } else if (portType === "input") {
            // Check if there's already a connection to this input port
            const existingConnections = this.connections.getConnectionsForPort(
                node.instanceId,
                portName,
                "input",
            );

            // If we're currently connecting, complete the connection
            if (this.connectingState) {
                // Remove existing connection if present before completing new one
                if (existingConnections.length > 0) {
                    existingConnections.forEach((id) =>
                        this.connections.removeConnection(id),
                    );
                }
                this.completeConnection(node, portName);
            } else if (existingConnections.length > 0) {
                // Time-based click vs drag detection for input ports with existing connections
                // < 0.125s = click = delete connection
                // > 0.125s = drag = reconnect
                const mouseDownTime = Date.now();
                const startX = event.clientX;
                const startY = event.clientY;
                let hasMoved = false;
                let reconnectingStarted = false;

                const onMouseMove = (e) => {
                    const dx = e.clientX - startX;
                    const dy = e.clientY - startY;
                    if (Math.abs(dx) > 5 || Math.abs(dy) > 5) {
                        hasMoved = true;
                        if (!reconnectingStarted) {
                            reconnectingStarted = true;
                            // Start reconnecting from the original output port
                            const connectionData =
                                this.connections.getConnectionData();
                            const existingConn = connectionData.find(
                                (c) => c.id === existingConnections[0],
                            );
                            if (existingConn) {
                                const fromNode = this.nodes.get(
                                    existingConn.fromNodeId,
                                );
                                if (fromNode) {
                                    // Remove the existing connection
                                    existingConnections.forEach((id) =>
                                        this.connections.removeConnection(id),
                                    );
                                    this.callbacks.autoSavePipeline();
                                    this.startConnecting(
                                        fromNode,
                                        existingConn.fromPortName,
                                        event,
                                        true,
                                    );
                                }
                            }
                        }
                    }
                };

                const onMouseUp = (e) => {
                    const elapsed = Date.now() - mouseDownTime;
                    globalThis.removeEventListener("mousemove", onMouseMove);
                    globalThis.removeEventListener("mouseup", onMouseUp);

                    if (elapsed < 125 && !hasMoved) {
                        // Quick click without movement - delete the connection
                        if (reconnectingStarted) {
                            this.cancelConnecting();
                        }
                        existingConnections.forEach((id) =>
                            this.connections.removeConnection(id),
                        );
                        this.callbacks.autoSavePipeline();
                    } else if (!reconnectingStarted) {
                        // Held long enough but didn't move - start reconnecting now
                        const connectionData =
                            this.connections.getConnectionData();
                        const existingConn = connectionData.find(
                            (c) => c.id === existingConnections[0],
                        );
                        if (existingConn) {
                            const fromNode = this.nodes.get(
                                existingConn.fromNodeId,
                            );
                            if (fromNode) {
                                // Remove the existing connection
                                existingConnections.forEach((id) =>
                                    this.connections.removeConnection(id),
                                );
                                this.callbacks.autoSavePipeline();
                                // Only start connecting if mouse button is still pressed
                                if (e.buttons & 1) {
                                    this.startConnecting(
                                        fromNode,
                                        existingConn.fromPortName,
                                        event,
                                        true,
                                    );
                                }
                            }
                        }
                    }
                    // If reconnectingStarted is true and not a quick click, startConnecting handles its own cleanup
                };

                globalThis.addEventListener("mousemove", onMouseMove);
                globalThis.addEventListener("mouseup", onMouseUp);
            }
        }
    }

    /**
     * Starts a temporary connection drag.
     *
     * @param {FlowchartNode} node Source node.
     * @param {string} portName Source port name.
     * @param {MouseEvent} event Pointer event.
     * @param {boolean} [isReconnecting=false] Whether this is a reconnect flow.
     */
    startConnecting(node, portName, event, isReconnecting = false) {
        if (this.connectingState) {
            if (this.connectingState.cleanup) {
                this.connectingState.cleanup();
            }
            this.connectingState.temp.remove();
        }

        const startPos = node.getOutputPortPosition(portName);
        const temp = this.connections.createTemporaryConnection(startPos, {
            fromHover: isReconnecting,
        });

        this.connectingState = {
            fromNode: node,
            fromPort: portName,
            temp: temp,
            cleanup: null,
        };

        // Immediately update temp connection to current mouse position
        const initialWorldPos = this.canvas.screenToWorld(
            event.clientX,
            event.clientY,
        );
        temp.update(initialWorldPos);

        const onMouseMove = (e) => {
            if (!this.connectingState) return;
            const worldPos = this.canvas.screenToWorld(e.clientX, e.clientY);
            this.connectingState.temp.update(worldPos);

            // Visual feedback for potential connection
            const target = document
                .elementFromPoint(e.clientX, e.clientY)
                ?.closest(".port-connector");
            if (target && target.dataset.portType === "input") {
                if (this.lastHighlightedInputPort !== target) {
                    if (this.lastHighlightedInputPort) {
                        this.lastHighlightedInputPort.style.backgroundColor =
                            "#404040";
                        this.lastHighlightedInputPort.style.transform =
                            "scale(1)";
                    }
                    this.lastHighlightedInputPort = target;
                    target.style.backgroundColor = "#f9c845";
                    target.style.transform = "scale(1.3)";
                }
            } else {
                this.clearHighlightedInputPort();
            }
        };

        const onMouseUp = (e) => {
            // Find the element at the release point more reliably
            const target = document
                .elementFromPoint(e.clientX, e.clientY)
                ?.closest(".port-connector");
            const isInput = target?.dataset.portType === "input";

            if (isInput) {
                const nodeElement = target.closest(".flowchart-node");
                const instanceId = nodeElement?.dataset.instanceId;
                const targetNode = this.nodes.get(instanceId);
                const targetPortName = target.dataset.portName;

                if (targetNode && targetPortName) {
                    this.completeConnection(targetNode, targetPortName);
                } else {
                    this.cancelConnecting();
                }
            } else {
                // Preserve click-click: release on the same source output port without
                // landing on an input keeps the in-progress wire (mousedown was on output).
                // Compare to the actual output port element, not event.target (which may be
                // an input when reconnecting was started from the destination port).
                const sourceOutputEl =
                    this.connectingState?.fromNode?.outputPorts?.get(
                        this.connectingState.fromPort,
                    );
                const releasedOnSourceOutput =
                    Boolean(sourceOutputEl && target) &&
                    (target === sourceOutputEl ||
                        sourceOutputEl.contains(target));
                if (!releasedOnSourceOutput) {
                    this.cancelConnecting();
                }
            }

            this.clearHighlightedInputPort();

            globalThis.removeEventListener("mousemove", onMouseMove);
            globalThis.removeEventListener("mouseup", onMouseUp);
            if (this.connectingState) {
                this.connectingState.cleanup = () => {};
            }
        };

        const cleanup = () => {
            globalThis.removeEventListener("mousemove", onMouseMove);
            globalThis.removeEventListener("mouseup", onMouseUp);
        };

        this.connectingState.cleanup = cleanup;

        globalThis.addEventListener("mousemove", onMouseMove);
        globalThis.addEventListener("mouseup", onMouseUp);

        event.stopPropagation();
    }

    /**
     * Finalizes the current connection.
     *
     * @param {FlowchartNode} toNode Target node.
     * @param {string} toPort Target port name.
     */
    completeConnection(toNode, toPort) {
        if (!this.connectingState) return;

        const { fromNode, fromPort } = this.connectingState;

        // Don't connect to self
        if (fromNode.instanceId === toNode.instanceId) {
            this.cancelConnecting();
            return;
        }

        const connectionId = `${fromNode.instanceId}-${fromPort}-${toNode.instanceId}-${toPort}`;
        const storeConnectionKey = pipelineStore.addConnection(
            fromNode.instanceId,
            fromPort,
            toNode.instanceId,
            toPort,
            fromPort,
            false,
        );
        if (!storeConnectionKey) {
            this.cancelConnecting();
            return;
        }

        // Only replace the visual input wire after the store accepts the new
        // connection (important for the one-MX3-per-Device-Input limit).
        const existingConnections = this.connections.getConnectionsForPort(
            toNode.instanceId,
            toPort,
            "input",
        );
        existingConnections.forEach((id) =>
            this.connections.removeConnection(id),
        );

        this.connections.createConnection({
            connectionId,
            fromNode,
            fromPortName: fromPort,
            toNode,
            toPortName: toPort,
            dataType: fromPort,
            isDefault: false,
            isDocked: this.isDockingConnection({
                fromNodeId: fromNode.instanceId,
                fromPortName: fromPort,
                toNodeId: toNode.instanceId,
                toPortName: toPort,
            }),
        });

        this.syncAllDynamicNodes();
        this.updateDockingLayout();

        if (this.minimap) {
            this.minimap.updateConnections(
                this.connections.getConnectionData(),
            );
        }

        this.cancelConnecting();
        this.callbacks.updateRunButton();
        this.callbacks.autoSavePipeline();
        this.updateCycleHighlights();
        this.updateIslandBlocks();
    }

    /**
     * Updates cycle highlighting for all connections.
     */
    updateCycleHighlights() {
        const connectionsData = this.connections.getConnectionData();
        const cycleConnectionIds = findCycles(this.nodes, connectionsData);

        // Reset all highlights first
        for (const connectionId of this.connections.connections.keys()) {
            this.connections.setCycleHighlight(connectionId, false);
        }

        // Highlight cycles
        cycleConnectionIds.forEach((id) => {
            this.connections.setCycleHighlight(id, true);
        });
    }

    /**
     * Cancels any in-progress connection drag.
     */
    cancelConnecting() {
        if (this.connectingState) {
            if (this.connectingState.cleanup) {
                this.connectingState.cleanup();
            }
            this.connectingState.temp.remove();
            this.connectingState = null;
        }
        this.clearHighlightedInputPort();
    }

    /**
     * Clears the currently highlighted input port.
     */
    clearHighlightedInputPort() {
        if (!this.lastHighlightedInputPort) {
            return;
        }
        this.lastHighlightedInputPort.style.backgroundColor = "#404040";
        this.lastHighlightedInputPort.style.transform = "scale(1)";
        this.lastHighlightedInputPort = null;
    }

    /**
     * Handles node removal requests from child nodes.
     *
     * @param {string} instanceId Node instance id.
     */
    handleNodeRemove(instanceId) {
        this.callbacks.removeFromPipeline(instanceId);
    }

    /**
     * Handles removal of a connection from the canvas.
     *
     * @param {string} connectionId Connection id.
     */
    handleConnectionRemoved(connectionId, connection = null) {
        // Instance IDs contain hyphens, so use stored connection metadata rather
        // than parsing the visual ID. This makes removing a dock an explicit,
        // reliable detach of its ordinary serialized connection.
        if (connection) {
            const fromUuid = pipelineStore.resolveToUuid(connection.fromNodeId);
            const toUuid = pipelineStore.resolveToUuid(connection.toNodeId);
            if (fromUuid && toUuid) {
                pipelineStore.removeConnection(
                    `${fromUuid}-${connection.fromPortName}-${toUuid}-${connection.toPortName}`,
                );
            }
        }

        if (this.minimap) {
            this.minimap.updateConnections(
                this.connections.getConnectionData(),
            );
        }
        this.syncAllDynamicNodes();
        this.updateDockingLayout();
        this.callbacks.updateRunButton();
        this.callbacks.autoSavePipeline();
        this.updateCycleHighlights();
        this.updateIslandBlocks();
    }

    /**
     * Handles updates to an existing connection.
     *
     * @param {string} connectionId Connection id.
     */
    handleConnectionChanged(connectionId) {
        const parts = connectionId.split("-");
        if (parts.length >= 4) {
            const fromId = parts[0];
            const fromPort = parts[1];
            const toId = parts[2];
            const toPort = parts[3];

            const fromUuid = pipelineStore.instanceIdToUuid.get(fromId);
            const toUuid = pipelineStore.instanceIdToUuid.get(toId);

            if (fromUuid && toUuid) {
                const storeConnectionKey = `${fromUuid}-${fromPort}-${toUuid}-${toPort}`;

                const customWaypoints =
                    this.connections.getCustomWaypoints(connectionId);
                if (customWaypoints !== undefined) {
                    pipelineStore.updateConnectionWaypoints(
                        storeConnectionKey,
                        customWaypoints,
                    );
                }

                const connectionData =
                    this.connections.connections.get(connectionId);
                if (connectionData) {
                    const storeConnection =
                        pipelineStore.state.currentPipeline.connections.get(
                            storeConnectionKey,
                        );
                    if (
                        storeConnection &&
                        storeConnection.isDefault !== connectionData.isDefault
                    ) {
                        pipelineStore.toggleConnectionDefault(
                            storeConnectionKey,
                        );
                    }
                }
            }
        }

        if (this.minimap) {
            this.minimap.updateConnections(
                this.connections.getConnectionData(),
            );
        }
        this.callbacks.autoSavePipeline();
        this.updateCycleHighlights();
        this.updateIslandBlocks();
    }

    /**
     * Propagates viewport changes to the minimap.
     *
     * @param {object} viewportState Viewport state.
     */
    handleViewportChange(viewportState) {
        if (this.minimap) {
            this.minimap.onViewportChange(viewportState);
        }
    }

    /**
     * Removes a node from the renderer.
     *
     * @param {string} instanceId Node instance id.
     */
    removeNode(instanceId) {
        const node = this.nodes.get(instanceId);
        if (node) {
            this.connections.removeConnectionsForNode(instanceId);
            node.destroy();
            this.nodes.delete(instanceId);

            // Trigger cycle detection after node removal
            this.updateCycleHighlights();

            this.syncPipelineArrayFromStore();
            this.refreshLayoutChrome();
            this.updateIslandBlocks();
        }
    }

    /**
     * Returns the current positions of all rendered nodes.
     *
     * @returns {Object<string, {x:number,y:number}>} Node positions.
     */
    getNodePositions() {
        const positions = {};
        this.nodes.forEach((node, instanceId) => {
            positions[instanceId] = node.getPosition();
        });
        return positions;
    }

    /**
     * Centers the canvas view on all rendered nodes.
     */
    centerViewOnNodes() {
        if (this.nodes.size === 0) return;

        let minX = Infinity;
        let minY = Infinity;
        let maxX = -Infinity;
        let maxY = -Infinity;

        this.nodes.forEach((node) => {
            const pos = node.getPosition();
            const nodeWidth = 200;
            const nodeHeight = 80;

            minX = Math.min(minX, pos.x - nodeWidth / 2);
            minY = Math.min(minY, pos.y - nodeHeight / 2);
            maxX = Math.max(maxX, pos.x + nodeWidth / 2);
            maxY = Math.max(maxY, pos.y + nodeHeight / 2);
        });

        const centerX = (minX + maxX) / 2;
        const centerY = (minY + maxY) / 2;
        const width = maxX - minX;
        const height = maxY - minY;

        const containerRect = this.canvasContainer.getBoundingClientRect();
        const padding = 100;

        const scaleX = (containerRect.width - padding * 2) / width;
        const scaleY = (containerRect.height - padding * 2) / height;
        const scale = Math.min(scaleX, scaleY, 1);

        this.canvas.scale = scale;
        this.canvas.translateX = containerRect.width / 2 - centerX * scale;
        this.canvas.translateY = containerRect.height / 2 - centerY * scale;

        this.canvas.updateTransform();
    }

    /**
     * Fits the canvas view to the rendered content.
     */
    fitToContent() {
        this.canvas.fitToContent();
    }

    /**
     * Resets the canvas view to its default transform.
     */
    resetView() {
        this.canvas.resetView();
    }

    /**
     * Tears down the renderer and its child components.
     */
    destroy() {
        this.nodes.forEach((node) => node.destroy());
        this.nodes.clear();
        this.clearIslandBlocks();
        this.connections.destroy();
        if (this.minimap) {
            this.minimap.destroy();
        }
        this.canvas.destroy();
    }

    /**
     * Restores connections from serialized connection data.
     *
     * @param {Array<object>} connectionsData Serialized connections.
     */
    restoreConnections(connectionsData) {
        connectionsData.forEach((conn) => {
            const fromNode = this.nodes.get(conn.fromNodeId);
            const toNode = this.nodes.get(conn.toNodeId);

            if (fromNode && toNode) {
                fromNode.ensureDynamicPortsForConnectionPort(
                    conn.fromPortName,
                    "output",
                );
                toNode.ensureDynamicPortsForConnectionPort(
                    conn.toPortName,
                    "input",
                );

                const connectionId = `${conn.fromNodeId}-${conn.fromPortName}-${conn.toNodeId}-${conn.toPortName}`;
                this.connections.createConnection({
                    connectionId,
                    fromNode,
                    fromPortName: conn.fromPortName,
                    toNode,
                    toPortName: conn.toPortName,
                    dataType: conn.dataType || conn.fromPortName,
                    isDefault: conn.isDefault || false,
                    customWaypoints: conn.customWaypoints || null,
                    isDocked: this.isDockingConnection(conn),
                });
            }
        });

        this.updateCycleHighlights();
        this.updateIslandBlocks();
    }

    /**
     * Removes all island overlays.
     */
    clearIslandBlocks() {
        const islandLayer = this.canvas?.getIslandLayer?.();
        if (islandLayer) {
            islandLayer.innerHTML = "";
        }
        this.islandBlocks = [];
        hideTooltip();
    }

    /**
     * Recomputes and renders disconnected-operation island overlays.
     */
    updateIslandBlocks() {
        const islandLayer = this.canvas?.getIslandLayer?.();
        if (!islandLayer) {
            return;
        }

        islandLayer.innerHTML = "";
        this.islandBlocks = [];
        this.nodes.forEach((node) => {
            node.setIslandInactive?.(false);
        });

        const islands = findUnreachableIslands(
            this.nodes,
            this.connections.getConnectionData(),
        );
        const padding = 28;

        islands.forEach((island, index) => {
            let minX = Infinity;
            let minY = Infinity;
            let maxX = -Infinity;
            let maxY = -Infinity;

            island.forEach((instanceId) => {
                const node = this.nodes.get(instanceId);
                if (!node?.element) {
                    return;
                }
                const position = node.getPosition();
                const width = node.element.offsetWidth || 200;
                const height = node.element.offsetHeight || 80;

                minX = Math.min(minX, position.x);
                minY = Math.min(minY, position.y);
                maxX = Math.max(maxX, position.x + width);
                maxY = Math.max(maxY, position.y + height);
            });

            if (
                !Number.isFinite(minX) ||
                !Number.isFinite(minY) ||
                !Number.isFinite(maxX) ||
                !Number.isFinite(maxY)
            ) {
                return;
            }

            const block = document.createElement("div");
            block.className = "pipeline-island-block";
            block.setAttribute(
                "aria-label",
                `Disconnected operation island ${index + 1}`,
            );
            Object.assign(block.style, {
                position: "absolute",
                left: `${minX - padding}px`,
                top: `${minY - padding}px`,
                width: `${maxX - minX + padding * 2}px`,
                height: `${maxY - minY + padding * 2}px`,
                border: "2px dotted rgba(255, 194, 74, 0.75)",
                borderRadius: "16px",
                pointerEvents: "none",
                background:
                    "repeating-linear-gradient(135deg, rgba(255, 194, 74, 0.16) 0, rgba(255, 194, 74, 0.16) 2px, transparent 2px, transparent 14px)",
                boxShadow: "0 0 0 1px rgba(255, 194, 74, 0.05)",
                opacity: "0.95",
            });

            const infoDot = this.createIslandInfoDot();
            block.appendChild(infoDot);
            islandLayer.appendChild(block);
            this.islandBlocks.push(block);

            island.forEach((instanceId) => {
                this.nodes.get(instanceId)?.setIslandInactive?.(true);
            });
        });
    }

    /**
     * Creates the info button shown on disconnected island overlays.
     *
     * @returns {HTMLButtonElement} Info button.
     */
    createIslandInfoDot() {
        const message =
            "Operation Island: these operations will not execute in the current configuration.";
        const dot = document.createElement("button");
        dot.type = "button";
        dot.className = "pipeline-island-info-dot";
        dot.setAttribute("aria-label", message);
        dot.title = message;
        Object.assign(dot.style, {
            position: "absolute",
            top: "-11px",
            left: "-11px",
            width: "24px",
            height: "24px",
            borderRadius: "50%",
            border: "2px solid rgba(26, 26, 26, 0.95)",
            background: "#f9c845",
            color: "#1a1a1a",
            fontSize: "13px",
            fontWeight: "800",
            lineHeight: "20px",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            cursor: "help",
            pointerEvents: "auto",
            zIndex: "3",
            boxShadow: "0 2px 8px rgba(0, 0, 0, 0.35)",
        });
        dot.textContent = "i";

        const showInfoTooltip = () => {
            dot.dataset.tooltip = message;
            dot.dataset.tooltipPlacement = "top";
            showTooltip(dot);
        };
        const hideInfoTooltip = () => {
            hideTooltip(dot);
        };

        dot.addEventListener("mouseenter", showInfoTooltip);
        dot.addEventListener("mouseleave", hideInfoTooltip);
        dot.addEventListener("click", (event) => {
            event.preventDefault();
            event.stopPropagation();
            showInfoTooltip();
        });

        return dot;
    }
}
