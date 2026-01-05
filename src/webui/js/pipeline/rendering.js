import { escapeHtml, getIconSVG } from "./utils.js";
import { FlowchartCanvas } from "./flowchartCanvas.js";
import { FlowchartNode } from "./flowchartNode.js";
import { FlowchartConnections } from "./flowchartConnections.js";

let descriptionPopup = null;

export function createDescriptionPopup() {
    if (descriptionPopup) return;

    descriptionPopup = document.createElement("div");
    descriptionPopup.id = "description-popup";
    descriptionPopup.className =
        "fixed z-50 bg-[#232323] border-2 border-[#f9c845] rounded-lg p-3 shadow-lg max-w-xs pointer-events-none opacity-0 transition-opacity duration-200";
    descriptionPopup.style.fontSize = "0.875rem";
    descriptionPopup.style.lineHeight = "1.25rem";
    descriptionPopup.style.boxShadow =
        "4px 4px 12px rgba(0,0,0,0.45), 8px 8px 20px rgba(0,0,0,0.25), 2px 2px 6px rgba(249,196,69,0.06)";

    document.body.appendChild(descriptionPopup);
}

export function showDescriptionPopup(name, description, event) {
    if (!descriptionPopup) createDescriptionPopup();

    descriptionPopup.innerHTML = `
        <div class="text-[#f9c845] font-semibold text-sm mb-2 border-b border-[#404040] pb-2">${escapeHtml(name)}</div>
        <div class="text-white text-xs">${escapeHtml(description)}</div>
    `;

    const mouseX = event.clientX;
    const mouseY = event.clientY;

    descriptionPopup.style.left = mouseX + 10 + "px";
    descriptionPopup.style.top = mouseY + 10 + "px";

    descriptionPopup.classList.remove("opacity-0");
    descriptionPopup.classList.add("opacity-100");
}

export function hideDescriptionPopup() {
    if (!descriptionPopup) return;
    descriptionPopup.classList.remove("opacity-100");
    descriptionPopup.classList.add("opacity-0");
}

export function addHoverListeners(element, name, description) {
    element.addEventListener("mouseenter", (e) => {
        showDescriptionPopup(name, description, e);
    });

    element.addEventListener("mousemove", (e) => {
        if (descriptionPopup?.classList.contains("opacity-100")) {
            descriptionPopup.style.left = e.clientX + 10 + "px";
            descriptionPopup.style.top = e.clientY + 10 + "px";
        }
    });

    element.addEventListener("mouseleave", () => {
        hideDescriptionPopup();
    });
}

export function renderOperations(
    operations,
    operationsList,
    openOperationSettings,
    handleDragStart,
) {
    operationsList.innerHTML = "";
    operations.forEach((op, index) => {
        const el = document.createElement("div");
        el.draggable = true;
        el.className =
            "bg-[#232323] border-2 border-[#404040] rounded-xl p-4 cursor-move hover:border-[#f9c845] transition-all transform hover:scale-105 hover:shadow-lg mb-2 group";
        el.style.boxShadow = "4px 4px 8px rgba(0, 0, 0, 0.4)";
        el.innerHTML = `
        <div class="flex items-center gap-3">
          <div class="bg-[#995e19] text-white text-xs font-semibold px-2 py-1 rounded-md uppercase tracking-wider">${escapeHtml(op.type)}</div>
          <div>
            <h3 class="font-medium text-white truncate max-w-[190px]">${escapeHtml(op.name)}</h3>
            ${index === 0 ? '<p class="text-xs text-gray-500 tracking-wider">Hover for description</p>' : ""}
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

            if (window.flowchartRenderer) {
                window.flowchartRenderer.setDragOffset(offsetX, offsetY);
            }

            handleDragStart(e, op, null, operations);
        });
        el.addEventListener("dragend", (e) => {
            if (e.currentTarget instanceof HTMLElement) {
                e.currentTarget.classList.remove("dragging");
                e.currentTarget.style.opacity = "";
            }

            if (window.flowchartRenderer) {
                window.flowchartRenderer.setDragOffset(0, 0);
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

        operationsList.appendChild(el);
    });
}

export class FlowchartRenderer {
    constructor(canvasContainer, options = {}) {
        this.canvasContainer = canvasContainer;
        this.canvas = null;
        this.connections = null;
        this.nodes = new Map();
        this.pipeline = [];

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

        this.init();
    }

    init() {
        this.canvas = new FlowchartCanvas(this.canvasContainer, {
            gridSpacing: this.gridSpacing
        });

        this.connections = new FlowchartConnections(
            this.canvas.getConnectionsLayer(),
            {
                connectionColor: "#f9c845",
                connectionWidth: 2,
                onConnectionRemoved: this.handleConnectionRemoved.bind(this)
            }
        );

        this.dragGhost = null;
        this.setupDropZone();
    }

    setDragOffset(offsetX, offsetY) {
        this.dragOffsetX = offsetX;
        this.dragOffsetY = offsetY;
    }

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
            if (e.clientX <= rect.left || e.clientX >= rect.right || e.clientY <= rect.top || e.clientY >= rect.bottom) {
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
        [pipelineArea, canvasContainer].forEach(elem => {
            if (elem) {
                elem.addEventListener("dragover", handleDragOver);
                elem.addEventListener("dragenter", handleDragEnter);
                elem.addEventListener("dragleave", handleDragLeave);
                elem.addEventListener("drop", handleDrop);
            }
        });
    }

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
            boxShadow: "0 0 15px rgba(249, 200, 69, 0.2)"
        });
        this.dragGhost.textContent = "Place Operation";
        
        this.canvas.getNodesLayer().appendChild(this.dragGhost);
    }

    updateDragGhost(e) {
        if (!this.dragGhost) {
            this.createDragGhost(e);
        }

        const scale = this.canvas.scale || 1;
        const worldPos = this.canvas.screenToWorld(e.clientX, e.clientY);
        const adjustedX = worldPos.x - (this.dragOffsetX / scale);
        const adjustedY = worldPos.y - (this.dragOffsetY / scale);
        const snappedPos = this.canvas.snapPositionToGrid(adjustedX, adjustedY);

        this.dragGhost.style.left = `${snappedPos.x}px`;
        this.dragGhost.style.top = `${snappedPos.y}px`;
    }

    removeDragGhost() {
        if (this.dragGhost) {
            this.dragGhost.remove();
            this.dragGhost = null;
        }
    }

    async handleDrop(e) {
        let dropData = null;

        try {
            const jsonData = e.dataTransfer.getData("application/pipeline") ||
                           e.dataTransfer.getData("text/plain");
            if (jsonData) {
                dropData = JSON.parse(jsonData);
            }
        } catch (err) {
            console.warn("[FLOWCHART] Failed to parse drop data:", err);
            return;
        }

        if (!dropData || !dropData.id) {
            console.warn("[FLOWCHART] Invalid drop data", dropData);
            return;
        }

        if (dropData.instanceId) {
            return;
        }

        const scale = this.canvas.scale || 1;
        const worldPos = this.canvas.screenToWorld(e.clientX, e.clientY);
        const adjustedX = worldPos.x - (this.dragOffsetX / scale);
        const adjustedY = worldPos.y - (this.dragOffsetY / scale);
        const snappedPos = this.canvas.snapPositionToGrid(adjustedX, adjustedY);

        await this.callbacks.onPipelineChange({
            type: "add",
            operationId: dropData.id,
            position: snappedPos
        });
    }

    async renderPipeline(pipeline) {
        this.pipeline = pipeline;
        
        this.nodes.forEach(node => node.destroy());
        this.nodes.clear();
        this.connections.clearAllConnections();
        
        const placeholder = document.getElementById("pipelinePlaceholder");
        if (placeholder) {
            const shouldShow = pipeline.length === 0;
            placeholder.classList.toggle("hidden", !shouldShow);
            
            if (shouldShow) {
                const hintText = placeholder.querySelector("p:last-child");
                if (hintText && !window.pipelineCreator?.selectedPipeline) {
                    hintText.textContent = "Create a new pipeline first using the button above";
                    hintText.style.color = "#f9c845";
                } else if (hintText) {
                    hintText.textContent = "Double-click to fit view • Scroll to zoom • Drag to pan";
                    hintText.style.color = "#666";
                }
            }
        }
        
        if (pipeline.length === 0) {
            this.callbacks.updateRunButton();
            return;
        }
        
        for (let i = 0; i < pipeline.length; i++) {
            const item = pipeline[i];

            if (!item.position) {
                item.position = this.calculateDefaultPosition(i, pipeline.length);
            }

            await this.createNode(item);
        }
        
        this.callbacks.updateRunButton();
    }

    calculateDefaultPosition(index, total) {
        const startX = 100;
        const startY = 100;
        
        return {
            x: this.canvas.snapToGrid(startX + index * this.nodeSpacingX),
            y: this.canvas.snapToGrid(startY + Math.sin(index * 0.5) * 50)
        };
    }

    async createNode(item) {
        const node = new FlowchartNode(item, {
            gridSpacing: this.gridSpacing,
            onDragStart: this.handleNodeDragStart.bind(this),
            onDragEnd: this.handleNodeDragEnd.bind(this),
            onPositionChange: this.handleNodePositionChange.bind(this),
            onSettingsClick: this.callbacks.openOperationSettings,
            onRemoveClick: this.handleNodeRemove.bind(this),
            onPortHover: this.handlePortHover.bind(this),
            onPortClick: this.handlePortClick.bind(this)
        });
        
        const element = await node.createElement();
        const nodesLayer = this.canvas.getNodesLayer();

        nodesLayer.appendChild(element);

        this.nodes.set(item.instanceId, node);
        
        return node;
    }

    handleNodeDragStart(node, event) {
        node.element.style.zIndex = "100";
    }

    handleNodeDragEnd(node, position) {
        node.element.style.zIndex = "10";
        
        const item = this.pipeline.find(p => p.instanceId === node.instanceId);
        if (item) {
            item.position = position;
        }
        
        this.connections.updateAllConnections(this.nodes);
        this.callbacks.autoSavePipeline();
    }

    handleNodePositionChange(node, position) {
        this.connections.updateAllConnections(this.nodes);
    }

    handlePortHover(node, portName, portType, isHovering) {
        const connectionIds = this.connections.getConnectionsForPort(
            node.instanceId,
            portName,
            portType
        );
        
        connectionIds.forEach(id => {
            this.connections.highlightConnection(id, isHovering);
        });
    }

    handlePortClick(node, portName, portType, event) {
        if (portType === "output") {
            this.startConnecting(node, portName, event);
        } else if (portType === "input" && this.connectingState) {
            this.completeConnection(node, portName);
        }
    }

    startConnecting(node, portName, event) {
        if (this.connectingState) {
            this.connectingState.temp.remove();
        }

        const startPos = node.getOutputPortPosition(portName);
        const temp = this.connections.createTemporaryConnection(startPos);
        
        this.connectingState = {
            fromNode: node,
            fromPort: portName,
            temp: temp
        };

        const onMouseMove = (e) => {
            if (!this.connectingState) return;
            const worldPos = this.canvas.screenToWorld(e.clientX, e.clientY);
            this.connectingState.temp.update(worldPos);
            
            // Visual feedback for potential connection
            const target = document.elementFromPoint(e.clientX, e.clientY)?.closest(".port-connector");
            if (target && target.dataset.portType === "input") {
                target.style.backgroundColor = "#f9c845";
                target.style.transform = "scale(1.3)";
            } else {
                // Reset other ports (simple way)
                document.querySelectorAll(".input-connector").forEach(p => {
                    if (p !== target) {
                        p.style.backgroundColor = "#404040";
                        p.style.transform = "scale(1)";
                    }
                });
            }
        };

        const onMouseUp = (e) => {
            // Find the element at the release point more reliably
            const target = document.elementFromPoint(e.clientX, e.clientY)?.closest(".port-connector");
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
                // Only cancel if it's not a click-click flow (mousedown and mouseup on the same port)
                const isOriginalOutput = target === event.target;
                if (!isOriginalOutput) {
                    this.cancelConnecting();
                }
            }
            
            // Reset visual feedback
            document.querySelectorAll(".input-connector").forEach(p => {
                p.style.backgroundColor = "#404040";
                p.style.transform = "scale(1)";
            });
            
            window.removeEventListener("mousemove", onMouseMove);
            window.removeEventListener("mouseup", onMouseUp);
        };

        window.addEventListener("mousemove", onMouseMove);
        window.addEventListener("mouseup", onMouseUp);
        
        event.stopPropagation();
    }

    completeConnection(toNode, toPort) {
        if (!this.connectingState) return;

        const { fromNode, fromPort, temp } = this.connectingState;
        
        // Don't connect to self
        if (fromNode.instanceId === toNode.instanceId) {
            this.cancelConnecting();
            return;
        }

        const connectionId = `${fromNode.instanceId}-${fromPort}-${toNode.instanceId}-${toPort}`;
        
        this.connections.createConnection(
            connectionId,
            fromNode,
            fromPort,
            toNode,
            toPort,
            fromPort // Use output port name as data type for now
        );

        this.cancelConnecting();
        this.callbacks.autoSavePipeline();
    }

    cancelConnecting() {
        if (this.connectingState) {
            this.connectingState.temp.remove();
            this.connectingState = null;
        }
    }

    handleNodeRemove(instanceId) {
        this.callbacks.removeFromPipeline(instanceId);
    }

    handleConnectionRemoved(connectionId) {
        this.callbacks.autoSavePipeline();
    }

    removeNode(instanceId) {
        const node = this.nodes.get(instanceId);
        if (node) {
            this.connections.removeConnectionsForNode(instanceId);
            node.destroy();
            this.nodes.delete(instanceId);
        }
    }

    getNodePositions() {
        const positions = {};
        this.nodes.forEach((node, instanceId) => {
            positions[instanceId] = node.getPosition();
        });
        return positions;
    }

    fitToContent() {
        this.canvas.fitToContent();
    }

    resetView() {
        this.canvas.resetView();
    }

    destroy() {
        this.nodes.forEach(node => node.destroy());
        this.nodes.clear();
        this.connections.destroy();
        this.canvas.destroy();
    }
}

export function renderPipeline(
    pipeline,
    pipelineContainer,
    pipelinePlaceholder,
    callbacks,
) {
    pipelineContainer.innerHTML = "";

    if (pipeline.length === 0) {
        pipelinePlaceholder.classList.remove("hidden");
        callbacks.updateRunButton();
        return;
    }

    pipelinePlaceholder.classList.add("hidden");

    pipeline.forEach((item, index) => {
        const wrapper = document.createElement("div");
        wrapper.dataset.instanceId = item.instanceId;
        wrapper.draggable = true;
        wrapper.className =
            "pipeline-item group relative bg-[#232323] border-2 border-[#404040] rounded-xl p-4 cursor-move hover:border-[#f9c845] transition-all transform hover:scale-105 hover:shadow-lg";
        wrapper.style.boxShadow = "4px 4px 8px rgba(0, 0, 0, 0.4)";

        wrapper.innerHTML = `
        <div class="flex items-center gap-3">
          <div class="text-gray-600">${getIconSVG("grip")}</div>
          <div class="bg-[#995e19] text-white text-xs font-semibold px-2 py-1 rounded-md uppercase tracking-wider">${escapeHtml(item.type)}</div>
          <div class="flex-1">
            <h3 class="font-semibold text-white truncate max-w-[230px]">${escapeHtml(item.name)}</h3>
            <p class="text-xs text-gray-500 tracking-wider">Hover for description</p>
          </div>
          <div class="flex items-center gap-2">
            <button class="op-settings-btn p-2 hover:bg-[#404040] rounded-lg transition-all" title="Settings">
              <img src="../../../assets/settings.svg" alt="Settings" class="w-4 h-4 icon-grayscale" />
            </button>
            <button class="remove-btn p-2 hover:bg-[#404040] rounded-lg transition-all" title="Remove"><img src="../../../assets/delete.svg" alt="Delete" class="w-4 h-4 icon-grayscale" /></button>
          </div>
        </div>
      `;

        wrapper.addEventListener("dragstart", (e) =>
            callbacks.handleDragStart(e, item, index, pipeline),
        );
        wrapper.addEventListener("dragend", (e) =>
            callbacks.handleDragEnd(e, pipelineContainer, pipelinePlaceholder, pipeline),
        );

        addHoverListeners(wrapper, item.name, item.description);

        const removeBtn = wrapper.querySelector(".remove-btn");
        removeBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            callbacks.removeFromPipeline(item.instanceId);
        });

        const opSettingsBtn = wrapper.querySelector(".op-settings-btn");
        if (opSettingsBtn) {
            opSettingsBtn.addEventListener("click", (e) => {
                e.stopPropagation();
                callbacks.openOperationSettings(item);
            });
        }

        pipelineContainer.appendChild(wrapper);

        if (index < pipeline.length - 1) {
            const connector = document.createElement("div");
            connector.className = "flex justify-center py-1";
            connector.innerHTML = `<div class="w-0.5 h-6 bg-[#f9c845]"></div>`;
            pipelineContainer.appendChild(connector);
        }
    });

    callbacks.updateRunButton();
}
