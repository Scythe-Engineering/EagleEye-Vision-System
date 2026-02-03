/**
 * FlowchartNode - Node component with input/output ports based on config data
 */

import { escapeHtml } from "./utils.js";
import { BACKEND_BASE_URL } from "../config.js";

export class FlowchartNode {
    constructor(operationData, options = {}) {
        this.operationData = operationData;
        this.instanceId = operationData.instanceId;
        this.position = operationData.position || { x: 100, y: 100 };
        this.inputNodes = [];
        this.outputNodes = [];
        this.inputNodeConfig = new Map();
        this.element = null;
        this.inputPorts = new Map();
        this.outputPorts = new Map();

        this.onDragStart = options.onDragStart || null;
        this.onDragEnd = options.onDragEnd || null;
        this.onPositionChange = options.onPositionChange || null;
        this.onSettingsClick = options.onSettingsClick || null;
        this.onRemoveClick = options.onRemoveClick || null;
        this.onPortHover = options.onPortHover || null;
        this.onPortClick = options.onPortClick || null;

        this.isDragging = false;
        this.dragOffsetX = 0;
        this.dragOffsetY = 0;
        this.gridSpacing = options.gridSpacing || 20;

        this.configDataLoaded = false;
    }

    async loadConfigData() {
        if (this.configDataLoaded) return;

        try {
            const isSecondary = this.operationData.isSecondary || false;
            const response = await fetch(
                `${BACKEND_BASE_URL}/get-operation-config-data/${encodeURIComponent(this.operationData.id)}/${isSecondary ? 1 : 0}`,
            );

            if (response.ok) {
                const configData = await response.json();
                const rawInputNodes = configData.input_nodes || ["data"];
                this.inputNodes = rawInputNodes.map((node) => {
                    if (typeof node === "object" && node.name) {
                        this.inputNodeConfig.set(node.name, {
                            hasDefault: node.has_default ?? false,
                        });
                        return node.name;
                    }
                    this.inputNodeConfig.set(node, { hasDefault: false });
                    return node;
                });
                this.outputNodes = configData.output_nodes || ["data"];
                this.configDataLoaded = true;
            }
        } catch (error) {
            console.warn(
                `Failed to load config data for ${this.operationData.id}:`,
                error,
            );
            this.inputNodes = ["data"];
            this.outputNodes = ["data"];
        }
    }

    async createElement() {
        await this.loadConfigData();

        this.element = document.createElement("div");
        this.element.className = "flowchart-node";
        this.element.dataset.instanceId = this.instanceId;
        this.element.style.position = "absolute";
        this.element.style.left = `${this.position.x}px`;
        this.element.style.top = `${this.position.y}px`;
        this.element.style.minWidth = "200px";
        this.element.style.zIndex = "10";

        this.applyStyles();
        this.renderContent();
        this.setupDragListeners();

        return this.element;
    }

    applyStyles() {
        Object.assign(this.element.style, {
            backgroundColor: "#232323",
            border: "2px solid #404040",
            borderRadius: "12px",
            boxShadow: "4px 4px 12px rgba(0, 0, 0, 0.5)",
            cursor: "move",
            userSelect: "none",
            transition: "border-color 0.15s ease, box-shadow 0.15s ease",
            pointerEvents: "auto", // Ensure the node itself is interactable
        });

        this.element.addEventListener("mouseenter", () => {
            if (!this.isDragging) {
                this.element.style.borderColor = "#f9c845";
                this.element.style.boxShadow =
                    "4px 4px 16px rgba(0, 0, 0, 0.6), 0 0 8px rgba(249, 200, 69, 0.2)";
            }
        });

        this.element.addEventListener("mouseleave", () => {
            if (!this.isDragging) {
                this.element.style.borderColor = "#404040";
                this.element.style.boxShadow =
                    "4px 4px 12px rgba(0, 0, 0, 0.5)";
            }
        });
    }

    renderContent() {
        const categoryColors = {
            det: "#995e19",
            loc: "#196099",
            prep: "#199960",
            proc: "#601999",
            net: "#996019",
            out: "#199919",
        };

        const categoryColor =
            categoryColors[this.operationData.type?.toLowerCase()] || "#995e19";

        const maxPorts = Math.max(
            this.inputNodes.length,
            this.outputNodes.length,
        );

        const threadColor = this.getThreadColor(this.threadInfo?.thread);
        const timestep = this.threadInfo?.timestep ?? null;

        this.element.innerHTML = `
            <div class="node-header" style="
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 8px 12px;
                background: linear-gradient(180deg, #2a2a2a 0%, #232323 100%);
                border-bottom: 1px solid #404040;
                border-radius: 10px 10px 0 0;
                position: relative;
            ">
                <div class="thread-badge" style="
                    position: absolute;
                    top: -12px;
                    left: -12px;
                    width: 24px;
                    height: 24px;
                    background-color: ${threadColor};
                    border: 2px solid #404040;
                    border-radius: 6px;
                    display: ${timestep !== null ? "flex" : "none"};
                    align-items: center;
                    justify-content: center;
                    font-size: 12px;
                    font-weight: 600;
                    color: white;
                    z-index: 10;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
                ">${timestep !== null ? timestep : ""}</div>
                <div style="display: flex; align-items: center; gap: 8px; flex: 1; min-width: 0;">
                    <span style="
                        background-color: ${categoryColor};
                        color: white;
                        font-size: 10px;
                        font-weight: 600;
                        padding: 2px 6px;
                        border-radius: 4px;
                        text-transform: uppercase;
                        letter-spacing: 0.5px;
                        flex-shrink: 0;
                    ">${escapeHtml(this.operationData.type || "OP")}</span>
                    <span style="
                        color: white;
                        font-weight: 500;
                        font-size: 13px;
                        max-width: 140px;
                        overflow: hidden;
                        text-overflow: ellipsis;
                        white-space: nowrap;
                        flex: 1;
                        min-width: 0;
                    ">${escapeHtml(this.operationData.name)}</span>
                    <div class="node-error-icon" style="
                        display: none;
                        width: 18px;
                        height: 18px;
                        border-radius: 50%;
                        background-color: #ff5c5c;
                        color: #1a1a1a;
                        font-size: 12px;
                        font-weight: 700;
                        align-items: center;
                        justify-content: center;
                        flex-shrink: 0;
                    ">i</div>
                </div>
                <div style="display: flex; gap: 4px; flex-shrink: 0;">
                    <button class="node-settings-btn" title="Settings" style="
                        padding: 4px;
                        background: transparent;
                        border: none;
                        cursor: pointer;
                        border-radius: 4px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                    ">
                        <img src="../../../assets/settings.svg" alt="Settings" style="width: 14px; height: 14px; filter: grayscale(100%); transition: filter 0.15s;" />
                    </button>
                    <button class="node-remove-btn" title="Remove" style="
                        padding: 4px;
                        background: transparent;
                        border: none;
                        cursor: pointer;
                        border-radius: 4px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        ${this.operationData.id === "device_input.py" ? "display: none;" : ""}
                    ">
                        <img src="../../../assets/delete.svg" alt="Delete" style="width: 14px; height: 14px; filter: grayscale(100%); transition: filter 0.15s;" />
                    </button>
                </div>
            </div>
            <div class="node-ports" style="
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 0;
                padding: 8px 0;
                align-items: start;
                min-height: ${maxPorts > 0 ? maxPorts * 24 + "px" : "auto"};
            ">
                <div class="input-ports" style="
                    display: flex;
                    flex-direction: column;
                    gap: 0;
                ">
                    ${this.renderInputPorts()}
                </div>
                <div class="output-ports" style="
                    display: flex;
                    flex-direction: column;
                    gap: 0;
                ">
                    ${this.renderOutputPorts()}
                </div>
            </div>
        `;

        this.setupButtonListeners();
        this.cachePortElements();
    }

    renderInputPorts() {
        return this.inputNodes
            .map(
                (nodeName) => `
            <div class="port-row input-port-row" data-port-name="${escapeHtml(nodeName)}" data-port-type="input" style="
                display: flex;
                align-items: center;
                padding: 4px 12px;
                gap: 8px;
                height: 24px;
            ">
                <div class="port-connector input-connector" data-port-name="${escapeHtml(nodeName)}" data-port-type="input" style="
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                    background-color: #404040;
                    border: 2px solid #606060;
                    cursor: pointer;
                    transition: all 0.15s ease;
                    margin-left: -18px;
                    flex-shrink: 0;
                "></div>
                <span style="
                    color: #a0a0a0;
                    font-size: 11px;
                    font-weight: 500;
                    overflow: hidden;
                    text-overflow: ellipsis;
                    white-space: nowrap;
                ">${escapeHtml(nodeName)}</span>
            </div>
        `,
            )
            .join("");
    }

    renderOutputPorts() {
        return this.outputNodes
            .map(
                (nodeName) => `
            <div class="port-row output-port-row" data-port-name="${escapeHtml(nodeName)}" data-port-type="output" style="
                display: flex;
                align-items: center;
                justify-content: flex-end;
                padding: 4px 12px;
                gap: 8px;
                height: 24px;
            ">
                <span style="
                    color: #a0a0a0;
                    font-size: 11px;
                    font-weight: 500;
                    overflow: hidden;
                    text-overflow: ellipsis;
                    white-space: nowrap;
                ">${escapeHtml(nodeName)}</span>
                <div class="port-connector output-connector" data-port-name="${escapeHtml(nodeName)}" data-port-type="output" style="
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                    background-color: #404040;
                    border: 2px solid #606060;
                    cursor: pointer;
                    transition: all 0.15s ease;
                    margin-right: -18px;
                    flex-shrink: 0;
                "></div>
            </div>
        `,
            )
            .join("");
    }

    cachePortElements() {
        this.inputPorts.clear();
        this.outputPorts.clear();

        this.element.querySelectorAll(".input-connector").forEach((port) => {
            const portName = port.dataset.portName;
            this.inputPorts.set(portName, port);
            this.setupPortListeners(port, portName, "input");
        });

        this.element.querySelectorAll(".output-connector").forEach((port) => {
            const portName = port.dataset.portName;
            this.outputPorts.set(portName, port);
            this.setupPortListeners(port, portName, "output");
        });
    }

    setupPortListeners(portElement, portName, portType) {
        portElement.addEventListener("mouseenter", () => {
            portElement.style.backgroundColor = "#f9c845";
            portElement.style.borderColor = "#f9c845";
            portElement.style.transform = "scale(1.2)";
            if (this.onPortHover) {
                this.onPortHover(this, portName, portType, true);
            }
        });

        portElement.addEventListener("mouseleave", () => {
            portElement.style.backgroundColor = "#404040";
            portElement.style.borderColor = "#606060";
            portElement.style.transform = "scale(1)";
            if (this.onPortHover) {
                this.onPortHover(this, portName, portType, false);
            }
        });

        portElement.addEventListener("mousedown", (e) => {
            e.stopPropagation();
            if (this.onPortClick) {
                this.onPortClick(this, portName, portType, e);
            }
        });
    }

    setupButtonListeners() {
        const settingsBtn = this.element.querySelector(".node-settings-btn");
        const removeBtn = this.element.querySelector(".node-remove-btn");

        if (settingsBtn) {
            settingsBtn.addEventListener("mouseenter", () => {
                settingsBtn.style.backgroundColor = "#404040";
                const img = settingsBtn.querySelector("img");
                if (img) img.style.filter = "none";
            });
            settingsBtn.addEventListener("mouseleave", () => {
                settingsBtn.style.backgroundColor = "transparent";
                const img = settingsBtn.querySelector("img");
                if (img) img.style.filter = "grayscale(100%)";
            });
            settingsBtn.addEventListener("click", (e) => {
                e.stopPropagation();
                if (this.onSettingsClick) {
                    this.onSettingsClick(this.operationData);
                }
            });
        }

        if (removeBtn) {
            removeBtn.addEventListener("mouseenter", () => {
                removeBtn.style.backgroundColor = "#404040";
                const img = removeBtn.querySelector("img");
                if (img) img.style.filter = "none";
            });
            removeBtn.addEventListener("mouseleave", () => {
                removeBtn.style.backgroundColor = "transparent";
                const img = removeBtn.querySelector("img");
                if (img) img.style.filter = "grayscale(100%)";
            });
            removeBtn.addEventListener("click", (e) => {
                e.stopPropagation();
                if (this.onRemoveClick) {
                    this.onRemoveClick(this.instanceId);
                }
            });
        }
    }

    setupDragListeners() {
        this.element.addEventListener(
            "mousedown",
            this.handleDragStart.bind(this),
        );
    }

    handleDragStart(event) {
        // Only handle left mouse button
        if (event.button !== 0) return;

        this.isDragging = true;
        // Find the canvas to get scale and translate
        // We assume the canvas instance is available via some global or we can find it
        // For now we'll stick to DOM inspection but make it more robust

        const scale = this.getCanvasScale();
        const rect = this.element.getBoundingClientRect();

        this.dragOffsetX = (event.clientX - rect.left) / scale;
        this.dragOffsetY = (event.clientY - rect.top) / scale;

        this.element.style.zIndex = "100";
        this.element.style.cursor = "grabbing";
        this.element.style.borderColor = "#f9c845";
        this.element.style.boxShadow = "8px 8px 24px rgba(0, 0, 0, 0.7)";

        if (this.onDragStart) {
            this.onDragStart(this, event);
        }

        const handleDragMove = (e) => {
            if (!this.isDragging) return;

            const scale = this.getCanvasScale();
            const viewport = this.element.closest("#flowchartViewport");
            if (!viewport) return;

            const containerRect =
                viewport.parentElement.getBoundingClientRect();
            const translate = this.getCanvasTranslate();

            // Calculate position in world coordinates
            let worldX =
                (e.clientX - containerRect.left - translate.x) / scale -
                this.dragOffsetX;
            let worldY =
                (e.clientY - containerRect.top - translate.y) / scale -
                this.dragOffsetY;

            // Snap to grid
            const snappedX =
                Math.round(worldX / this.gridSpacing) * this.gridSpacing;
            const snappedY =
                Math.round(worldY / this.gridSpacing) * this.gridSpacing;

            // Visual feedback for snapping - if we moved enough to snap to a new position
            if (snappedX !== this.position.x || snappedY !== this.position.y) {
                this.position.x = snappedX;
                this.position.y = snappedY;
                this.element.style.left = `${snappedX}px`;
                this.element.style.top = `${snappedY}px`;

                if (this.onPositionChange) {
                    this.onPositionChange(this, { x: snappedX, y: snappedY });
                }
            }
        };

        const handleDragEnd = () => {
            if (!this.isDragging) return;

            this.isDragging = false;
            this.element.style.zIndex = "10";
            this.element.style.cursor = "move";
            this.element.style.borderColor = "#404040";
            this.element.style.boxShadow = "4px 4px 12px rgba(0, 0, 0, 0.5)";

            document.removeEventListener("mousemove", handleDragMove);
            document.removeEventListener("mouseup", handleDragEnd);

            if (this.onDragEnd) {
                this.onDragEnd(this, this.position);
            }
        };

        document.addEventListener("mousemove", handleDragMove);
        document.addEventListener("mouseup", handleDragEnd);

        event.preventDefault();
        event.stopPropagation();
    }

    getCanvasScale() {
        const viewport = this.element.closest("#flowchartViewport");
        if (!viewport) return 1;

        const transform = viewport.style.transform;
        const scaleMatch = transform.match(/scale\(([^)]+)\)/);
        return scaleMatch ? Number.parseFloat(scaleMatch[1]) : 1;
    }

    getCanvasTranslate() {
        const viewport = this.element.closest("#flowchartViewport");
        if (!viewport) return { x: 0, y: 0 };

        const transform = viewport.style.transform;
        const translateMatch = transform.match(
            /translate\(([^,]+)px,\s*([^)]+)px\)/,
        );
        if (translateMatch) {
            return {
                x: Number.parseFloat(translateMatch[1]),
                y: Number.parseFloat(translateMatch[2]),
            };
        }
        return { x: 0, y: 0 };
    }

    getInputPortPosition(portName) {
        const port = this.inputPorts.get(portName);
        if (!port) return null;

        const portRect = port.getBoundingClientRect();
        const nodeRect = this.element.getBoundingClientRect();
        const scale = this.getCanvasScale();

        return {
            x: this.position.x,
            y:
                this.position.y +
                (portRect.top - nodeRect.top + portRect.height / 2) / scale,
        };
    }

    getOutputPortPosition(portName) {
        const port = this.outputPorts.get(portName);
        if (!port) return null;

        const portRect = port.getBoundingClientRect();
        const nodeRect = this.element.getBoundingClientRect();
        const scale = this.getCanvasScale();

        return {
            x: this.position.x + this.element.offsetWidth,
            y:
                this.position.y +
                (portRect.top - nodeRect.top + portRect.height / 2) / scale,
        };
    }

    setPosition(x, y) {
        this.position.x = x;
        this.position.y = y;
        if (this.element) {
            this.element.style.left = `${x}px`;
            this.element.style.top = `${y}px`;
        }
    }

    getPosition() {
        return { ...this.position };
    }

    canInputPortBeDefault(portName) {
        const config = this.inputNodeConfig.get(portName);
        return config?.hasDefault ?? false;
    }

    getThreadColor(threadNumber) {
        if (!threadNumber || threadNumber <= 0) return null;

        const threadColors = [
            "#ff6b6b", // Red
            "#4ecdc4", // Teal
            "#45b7d1", // Blue
            "#96ceb4", // Light Green
            "#ffeaa7", // Light Yellow
            "#dfe6e9", // Light Gray
            "#fd79a8", // Pink
            "#a29bfe", // Light Purple
            "#00b894", // Green
            "#e17055", // Orange
        ];

        return threadColors[(threadNumber - 1) % threadColors.length];
    }

    updateThreadInfo(threadInfo) {
        this.threadInfo = threadInfo;
        const badge = this.element?.querySelector(".thread-badge");
        if (!badge) return;

        if (
            !threadInfo ||
            threadInfo.timestep === null ||
            threadInfo.timestep === undefined
        ) {
            badge.style.display = "none";
        } else {
            badge.style.display = "flex";
            badge.style.backgroundColor = this.getThreadColor(
                threadInfo.thread,
            );
            badge.textContent = threadInfo.timestep;
        }
    }

    hideThreadBadge() {
        this.threadInfo = null;
        const badge = this.element?.querySelector(".thread-badge");
        if (badge) {
            badge.style.display = "none";
        }
    }

    setErrorState(errorRecord, isDownstream) {
        if (!this.element) {
            return;
        }

        const hasError = Boolean(errorRecord);
        const nodeIcon = this.element.querySelector(".node-error-icon");

        this.element.classList.toggle("pipeline-error-node", hasError);
        this.element.classList.toggle(
            "pipeline-downstream-disabled",
            Boolean(isDownstream),
        );

        if (hasError) {
            this.element.style.borderColor = "#ff5c5c";
            this.element.style.boxShadow =
                "0 0 0 2px rgba(255,92,92,0.35), 4px 4px 12px rgba(0, 0, 0, 0.5)";
        } else if (!this.isDragging) {
            this.element.style.borderColor = "#404040";
            this.element.style.boxShadow =
                "4px 4px 12px rgba(0, 0, 0, 0.5)";
        }

        if (nodeIcon) {
            nodeIcon.style.display = hasError ? "inline-flex" : "none";
        }
    }

    destroy() {
        this.element?.remove();
        this.inputPorts.clear();
        this.outputPorts.clear();
    }
}
