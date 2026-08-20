/**
 * FlowchartNode renders and manages a pipeline node with configurable ports.
 */
// Responsible for node layout, port generation, interaction, and status badges.

import { escapeHtml } from "./utils.js";
import { BACKEND_BASE_URL } from "../config.js";
import {
    hasCachedConfig,
    getCachedConfig,
    setCachedConfig,
} from "./operationConfigCache.js";

export class FlowchartNode {
    /**
     * Creates a new flowchart node instance.
     * @param {object} operationData - Operation metadata and state.
     * @param {object} [options={}] - Optional event callbacks and layout settings.
     */
    constructor(operationData, options = {}) {
        this.operationData = operationData;
        this.instanceId = operationData.instanceId;
        this.position = operationData.position || { x: 100, y: 100 };
        this.inputNodes = [];
        this.outputNodes = [];
        this.staticInputNodes = [];
        this.staticOutputNodes = [];
        this.dynamicInputNodes = [];
        this.dynamicOutputNodes = [];
        this.dynamicGroup = null;
        this.inputNodeConfig = new Map();
        this.outputNodeConfig = new Map();
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
        this.isHovered = false;
        this.isSelected = false;
        this.dragOffsetX = 0;
        this.dragOffsetY = 0;
        this.gridSpacing = options.gridSpacing || 20;

        this.configDataLoaded = false;
        this.docking = null;
        this.isDocked = false;
        this.isDockInvalid = false;
        this.threadInfo = null;
        this.profilingInfo = null;
        this.dragContext = null;
        this.cachedElementWidth = 200;
    }

    /**
     * Builds a numbered port name from a base name.
     * @param {string} baseName - The base port name.
     * @param {number} index - 1-based port index.
     * @returns {string} Indexed port name.
     */
    buildIndexedPortName(baseName, index) {
        return `${baseName}_${index}`;
    }

    /**
     * Parses a dynamic port index from a port name.
     * @param {string} portName - The port name to inspect.
     * @param {string} baseName - The dynamic port base name.
     * @returns {number|null} Parsed 1-based index, or null if not dynamic.
     */
    parseDynamicPortIndex(portName, baseName) {
        if (!portName || !baseName) {
            return null;
        }

        if (portName === baseName) {
            return 1;
        }

        const prefix = `${baseName}_`;
        if (!portName.startsWith(prefix)) {
            return null;
        }

        const maybeIndex = Number.parseInt(portName.slice(prefix.length), 10);
        if (!Number.isInteger(maybeIndex) || maybeIndex < 1) {
            return null;
        }

        return maybeIndex;
    }

    /**
     * Normalizes raw dynamic group config into internal state.
     * @param {object|null} rawDynamicGroup - Raw dynamic group configuration.
     * @param {Array} rawInputNodes - Configured input node names.
     * @param {Array} rawOutputNodes - Configured output node names.
     * @returns {object|null} Normalized dynamic group state.
     */
    normalizeDynamicGroup(rawDynamicGroup, rawInputNodes, rawOutputNodes) {
        if (!rawDynamicGroup || typeof rawDynamicGroup !== "object") {
            return null;
        }

        const maxInputs = Math.max(
            1,
            Number.parseInt(rawDynamicGroup.max_inputs ?? 1, 10) || 1,
        );
        const maxOutputs = Math.max(
            1,
            Number.parseInt(rawDynamicGroup.max_outputs ?? maxInputs, 10) ||
                maxInputs,
        );

        const mirroredOutputGroup =
            rawDynamicGroup.mirrored_output_group === true ||
            String(rawDynamicGroup.mirrored_output_group).toLowerCase() ===
                "true";

        const outputDynamicGroup =
            rawDynamicGroup.output_dynamic_group === true ||
            String(rawDynamicGroup.output_dynamic_group).toLowerCase() ===
                "true";
        const inputDynamicGroupDisabled =
            rawDynamicGroup.input_dynamic_group === false ||
            String(rawDynamicGroup.input_dynamic_group).toLowerCase() ===
                "false";

        const hasDynamicInputGroup = !inputDynamicGroupDisabled;
        const hasDynamicOutputGroup = mirroredOutputGroup || outputDynamicGroup;

        const coupledGroups =
            rawDynamicGroup.coupled_groups === undefined
                ? mirroredOutputGroup
                : rawDynamicGroup.coupled_groups === true ||
                  String(rawDynamicGroup.coupled_groups).toLowerCase() ===
                      "true";

        const inputBaseName =
            rawDynamicGroup.input_base_name ||
            rawDynamicGroup.input_node ||
            (rawInputNodes.length > 0
                ? rawInputNodes[rawInputNodes.length - 1]
                : "data");

        const outputBaseName =
            rawDynamicGroup.output_base_name ||
            rawDynamicGroup.output_node ||
            (rawOutputNodes.length > 0
                ? rawOutputNodes[rawOutputNodes.length - 1]
                : inputBaseName);

        return {
            maxInputs,
            maxOutputs,
            mirroredOutputGroup,
            outputDynamicGroup,
            hasDynamicInputGroup,
            hasDynamicOutputGroup,
            coupledGroups:
                coupledGroups && hasDynamicInputGroup && hasDynamicOutputGroup,
            inputBaseName,
            outputBaseName,
            inputCount: hasDynamicInputGroup ? 1 : 0,
            outputCount: hasDynamicOutputGroup ? 1 : 0,
            connectedInputCount: 0,
            connectedOutputCount: 0,
        };
    }

    /**
     * Initializes input/output port state from config data.
     * @param {object} configData - Operation config payload.
     */
    initializePortsFromConfig(configData) {
        const docking = configData?.docking;
        this.docking =
            docking?.source_action &&
            docking?.source_port &&
            docking?.target_port
                ? docking
                : null;
        this.inputNodeConfig.clear();
        this.outputNodeConfig.clear();

        const rawInputNodes = configData.input_nodes || ["data"];
        const rawOutputNodes = configData.output_nodes || ["data"];

        const normalizedInputNodes = rawInputNodes.map((node) => {
            if (typeof node === "object" && node.name) {
                this.inputNodeConfig.set(node.name, {
                    hasDefault: node.has_default ?? false,
                });
                return node.name;
            }
            this.inputNodeConfig.set(node, { hasDefault: false });
            return node;
        });

        const normalizedOutputNodes = rawOutputNodes.map((node) => {
            if (typeof node === "object" && node.name) {
                this.outputNodeConfig.set(node.name, {
                    hasDefault: node.has_default ?? false,
                });
                return node.name;
            }
            this.outputNodeConfig.set(node, { hasDefault: false });
            return node;
        });

        this.dynamicGroup = this.normalizeDynamicGroup(
            configData.dynamic_group,
            normalizedInputNodes,
            normalizedOutputNodes,
        );

        if (!this.dynamicGroup) {
            this.staticInputNodes = [...normalizedInputNodes];
            this.staticOutputNodes = [...normalizedOutputNodes];
            this.dynamicInputNodes = [];
            this.dynamicOutputNodes = [];
            this.inputNodes = [...this.staticInputNodes];
            this.outputNodes = [...this.staticOutputNodes];
            return;
        }

        this.staticInputNodes = this.dynamicGroup.hasDynamicInputGroup
            ? normalizedInputNodes.filter(
                  (name) => name !== this.dynamicGroup.inputBaseName,
              )
            : [...normalizedInputNodes];

        this.staticOutputNodes = normalizedOutputNodes.filter((name) => {
            if (this.dynamicGroup.hasDynamicOutputGroup) {
                return name !== this.dynamicGroup.outputBaseName;
            }
            return true;
        });

        this.rebuildDynamicPortNames({
            inputCount: this.dynamicGroup.inputCount,
            outputCount: this.dynamicGroup.outputCount,
        });
    }

    /**
     * Rebuilds dynamic port names and clamps dynamic counts.
     * @param {object} [counts={}] - Desired and connected port counts.
     * @returns {void}
     */
    rebuildDynamicPortNames({
        inputCount = this.dynamicGroup?.inputCount ?? 0,
        outputCount = this.dynamicGroup?.outputCount ?? 0,
        connectedInputCount = this.dynamicGroup?.connectedInputCount ?? 0,
        connectedOutputCount = this.dynamicGroup?.connectedOutputCount ?? 0,
    } = {}) {
        if (!this.dynamicGroup) {
            this.inputNodes = [...this.staticInputNodes];
            this.outputNodes = [...this.staticOutputNodes];
            return;
        }

        const boundedInputCount = this.dynamicGroup.hasDynamicInputGroup
            ? Math.max(1, Math.min(this.dynamicGroup.maxInputs, inputCount))
            : 0;
        const boundedOutputCount = this.dynamicGroup.hasDynamicOutputGroup
            ? Math.max(1, Math.min(this.dynamicGroup.maxOutputs, outputCount))
            : 0;

        const boundedConnectedInputCount = this.dynamicGroup
            .hasDynamicInputGroup
            ? Math.max(0, Math.min(boundedInputCount, connectedInputCount))
            : 0;
        const boundedConnectedOutputCount = this.dynamicGroup
            .hasDynamicOutputGroup
            ? Math.max(0, Math.min(boundedOutputCount, connectedOutputCount))
            : 0;

        this.dynamicGroup.inputCount = boundedInputCount;
        this.dynamicGroup.outputCount = boundedOutputCount;
        this.dynamicGroup.connectedInputCount = boundedConnectedInputCount;
        this.dynamicGroup.connectedOutputCount = boundedConnectedOutputCount;

        this.dynamicInputNodes = this.dynamicGroup.hasDynamicInputGroup
            ? Array.from({ length: boundedInputCount }, (_, idx) =>
                  this.buildIndexedPortName(
                      this.dynamicGroup.inputBaseName,
                      idx + 1,
                  ),
              )
            : [];

        this.dynamicOutputNodes = this.dynamicGroup.hasDynamicOutputGroup
            ? Array.from({ length: boundedOutputCount }, (_, idx) =>
                  this.buildIndexedPortName(
                      this.dynamicGroup.outputBaseName,
                      idx + 1,
                  ),
              )
            : [];

        this.inputNodes = [...this.staticInputNodes, ...this.dynamicInputNodes];
        this.outputNodes = [
            ...this.staticOutputNodes,
            ...this.dynamicOutputNodes,
        ];

        this.dynamicInputNodes.forEach((portName) => {
            this.inputNodeConfig.set(portName, { hasDefault: false });
        });
    }

    /**
     * Synchronizes dynamic port counts against current connections.
     * @param {Array} connectionData - Connection records for the graph.
     * @returns {boolean} True when port state changed.
     */
    syncDynamicPorts(connectionData = []) {
        if (!this.dynamicGroup) {
            return false;
        }

        const connectedDynamicInputPorts = new Set();
        const connectedDynamicOutputPorts = new Set();
        connectionData.forEach((conn) => {
            if (
                this.dynamicGroup.hasDynamicInputGroup &&
                conn.toNodeId === this.instanceId &&
                this.parseDynamicPortIndex(
                    conn.toPortName,
                    this.dynamicGroup.inputBaseName,
                ) !== null
            ) {
                connectedDynamicInputPorts.add(conn.toPortName);
            }

            if (
                this.dynamicGroup.hasDynamicOutputGroup &&
                conn.fromNodeId === this.instanceId &&
                this.parseDynamicPortIndex(
                    conn.fromPortName,
                    this.dynamicGroup.outputBaseName,
                ) !== null
            ) {
                connectedDynamicOutputPorts.add(conn.fromPortName);
            }
        });

        const connectedDynamicInputCount = connectedDynamicInputPorts.size;
        const connectedDynamicOutputCount = connectedDynamicOutputPorts.size;

        const desiredInputCount = this.dynamicGroup.hasDynamicInputGroup
            ? Math.min(
                  this.dynamicGroup.maxInputs,
                  Math.max(1, connectedDynamicInputCount + 1),
              )
            : 0;
        const desiredOutputCount = this.dynamicGroup.hasDynamicOutputGroup
            ? Math.min(
                  this.dynamicGroup.maxOutputs,
                  Math.max(1, connectedDynamicOutputCount + 1),
              )
            : 0;
        const desiredConnectedInputCount = connectedDynamicInputCount;
        const desiredConnectedOutputCount = connectedDynamicOutputCount;

        const desiredState = this.dynamicGroup.coupledGroups
            ? (() => {
                  const sharedConnected = Math.max(
                      desiredConnectedInputCount,
                      desiredConnectedOutputCount,
                  );
                  const sharedCount = Math.min(
                      Math.min(
                          this.dynamicGroup.maxInputs,
                          this.dynamicGroup.maxOutputs,
                      ),
                      Math.max(1, sharedConnected + 1),
                  );
                  return {
                      inputCount: sharedCount,
                      outputCount: sharedCount,
                      connectedInputCount: desiredConnectedInputCount,
                      connectedOutputCount: desiredConnectedOutputCount,
                  };
              })()
            : {
                  inputCount: desiredInputCount,
                  outputCount: desiredOutputCount,
                  connectedInputCount: desiredConnectedInputCount,
                  connectedOutputCount: desiredConnectedOutputCount,
              };

        if (
            desiredState.inputCount === this.dynamicGroup.inputCount &&
            desiredState.outputCount === this.dynamicGroup.outputCount &&
            desiredState.connectedInputCount ===
                this.dynamicGroup.connectedInputCount &&
            desiredState.connectedOutputCount ===
                this.dynamicGroup.connectedOutputCount
        ) {
            return false;
        }

        this.rebuildDynamicPortNames(desiredState);
        if (this.element) {
            this.renderContent();
        }
        return true;
    }

    /**
     * Ensures a dynamic port exists for a connected port name.
     * @param {string} portName - Connected port name.
     * @param {"input"|"output"} [portType="input"] - Port side to expand.
     * @returns {boolean} True when ports were expanded.
     */
    ensureDynamicPortsForConnectionPort(portName, portType = "input") {
        if (!this.dynamicGroup) {
            return false;
        }

        const baseName =
            portType === "output"
                ? this.dynamicGroup.outputBaseName
                : this.dynamicGroup.inputBaseName;

        const parsedIndex = this.parseDynamicPortIndex(portName, baseName);
        if (parsedIndex === null) {
            return false;
        }

        const isOutputPort = portType === "output";
        const currentCount = isOutputPort
            ? this.dynamicGroup.outputCount
            : this.dynamicGroup.inputCount;
        const currentConnectedCount = isOutputPort
            ? this.dynamicGroup.connectedOutputCount
            : this.dynamicGroup.connectedInputCount;

        if (
            parsedIndex <= currentCount &&
            parsedIndex <= currentConnectedCount
        ) {
            return false;
        }

        const maxCount = isOutputPort
            ? this.dynamicGroup.maxOutputs
            : this.dynamicGroup.maxInputs;
        const desiredSideCount = Math.min(
            maxCount,
            Math.max(
                currentCount,
                parsedIndex < maxCount ? parsedIndex + 1 : parsedIndex,
            ),
        );

        if (this.dynamicGroup.coupledGroups) {
            const sharedMax = Math.min(
                this.dynamicGroup.maxInputs,
                this.dynamicGroup.maxOutputs,
            );
            const sharedCount = Math.min(sharedMax, desiredSideCount);

            this.rebuildDynamicPortNames({
                inputCount: sharedCount,
                outputCount: sharedCount,
                connectedInputCount: isOutputPort
                    ? this.dynamicGroup.connectedInputCount
                    : Math.max(
                          this.dynamicGroup.connectedInputCount,
                          parsedIndex,
                      ),
                connectedOutputCount: isOutputPort
                    ? Math.max(
                          this.dynamicGroup.connectedOutputCount,
                          parsedIndex,
                      )
                    : this.dynamicGroup.connectedOutputCount,
            });
        } else {
            this.rebuildDynamicPortNames({
                inputCount: isOutputPort
                    ? this.dynamicGroup.inputCount
                    : desiredSideCount,
                outputCount: isOutputPort
                    ? desiredSideCount
                    : this.dynamicGroup.outputCount,
                connectedInputCount: isOutputPort
                    ? this.dynamicGroup.connectedInputCount
                    : Math.max(
                          this.dynamicGroup.connectedInputCount,
                          parsedIndex,
                      ),
                connectedOutputCount: isOutputPort
                    ? Math.max(
                          this.dynamicGroup.connectedOutputCount,
                          parsedIndex,
                      )
                    : this.dynamicGroup.connectedOutputCount,
            });
        }

        if (this.element) {
            this.renderContent();
        }
        return true;
    }

    /**
     * Loads operation config data, using cache when available.
     * @returns {Promise<void>}
     */
    async loadConfigData() {
        if (this.configDataLoaded) return;

        const isSecondary = this.operationData.isSecondary || false;

        if (hasCachedConfig(this.operationData.id, isSecondary)) {
            this.initializePortsFromConfig(
                getCachedConfig(this.operationData.id, isSecondary),
            );
            this.configDataLoaded = true;
            return;
        }

        try {
            const response = await fetch(
                `${BACKEND_BASE_URL}/get-operation-config-data/${encodeURIComponent(this.operationData.id)}/${isSecondary ? 1 : 0}`,
            );

            if (response.ok) {
                const configData = await response.json();
                setCachedConfig(this.operationData.id, isSecondary, configData);
                this.initializePortsFromConfig(configData);
                this.configDataLoaded = true;
            } else {
                console.warn(
                    `Failed to load config data for ${this.operationData.id}: ${response.status}`,
                );
                this.inputNodes = ["data"];
                this.outputNodes = ["data"];
                this.staticInputNodes = ["data"];
                this.staticOutputNodes = ["data"];
                this.dynamicInputNodes = [];
                this.dynamicOutputNodes = [];
                this.dynamicGroup = null;
                this.configDataLoaded = true;
            }
        } catch (error) {
            console.warn(
                `Failed to load config data for ${this.operationData.id}:`,
                error,
            );
            this.inputNodes = ["data"];
            this.outputNodes = ["data"];
            this.staticInputNodes = ["data"];
            this.staticOutputNodes = ["data"];
            this.dynamicInputNodes = [];
            this.dynamicOutputNodes = [];
            this.dynamicGroup = null;
            this.configDataLoaded = true;
        }
    }

    /**
     * Creates the node DOM element and wires up listeners.
     * @returns {Promise<HTMLDivElement>} Node element.
     */
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
        this.cachedElementWidth = this.element.offsetWidth || 200;

        return this.element;
    }

    /**
     * Applies base node styling and hover chrome behavior.
     */
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

        /**
         * Restores the default node chrome styling.
         */
        const applyDefaultNodeChrome = () => {
            if (this.isSelected) {
                this.applySelectedNodeChrome();
                return;
            }
            this.element.style.borderColor = "#404040";
            this.element.style.boxShadow = "4px 4px 12px rgba(0, 0, 0, 0.5)";
        };

        /**
         * Applies the hovered node chrome styling.
         */
        const applyHoveredNodeChrome = () => {
            if (this.isSelected) {
                this.applySelectedNodeChrome();
                return;
            }
            this.element.style.borderColor = "#f9c845";
            this.element.style.boxShadow =
                "4px 4px 16px rgba(0, 0, 0, 0.6), 0 0 8px rgba(249, 200, 69, 0.2)";
        };

        this.element.addEventListener("mouseenter", () => {
            this.isHovered = true;
            if (!this.isDragging) {
                applyHoveredNodeChrome();
            }
        });

        this.element.addEventListener("mouseleave", (event) => {
            if (
                event.relatedTarget &&
                this.element.contains(event.relatedTarget)
            ) {
                return;
            }

            requestAnimationFrame(() => {
                if (this.element.matches(":hover")) {
                    this.isHovered = true;
                    if (!this.isDragging) {
                        applyHoveredNodeChrome();
                    }
                    return;
                }

                this.isHovered = false;
                if (!this.isDragging) {
                    applyDefaultNodeChrome();
                }
            });
        });
    }

    /**
     * Applies selected-node accent chrome.
     */
    applySelectedNodeChrome() {
        if (!this.element) {
            return;
        }
        this.element.style.borderColor = "#f9c845";
        this.element.style.boxShadow =
            "4px 4px 18px rgba(0, 0, 0, 0.65), 0 0 14px rgba(249, 200, 69, 0.4)";
    }

    /**
     * Updates the node chrome to match current hover/drag/selection state.
     */
    updateNodeHoverChrome() {
        if (!this.element || this.isDragging) {
            return;
        }
        if (this.isDocked) {
            this.element.style.borderColor = "#f9c845";
            return;
        }
        if (this.isDockInvalid) {
            this.element.style.borderColor = "#ff6b6b";
            this.element.style.borderStyle = "dashed";
            return;
        }

        if (this.isSelected) {
            this.applySelectedNodeChrome();
            return;
        }

        if (this.isHovered || this.element.matches(":hover")) {
            this.element.style.borderColor = "#f9c845";
            this.element.style.boxShadow =
                "4px 4px 16px rgba(0, 0, 0, 0.6), 0 0 8px rgba(249, 200, 69, 0.2)";
        } else {
            this.element.style.borderColor = "#404040";
            this.element.style.boxShadow = "4px 4px 12px rgba(0, 0, 0, 0.5)";
        }
    }

    /**
     * Sets whether this node is part of the canvas selection.
     * @param {boolean} selected
     */
    setSelected(selected) {
        this.isSelected = Boolean(selected);
        if (this.element) {
            this.element.classList.toggle("flowchart-node-selected", this.isSelected);
            this.updateNodeHoverChrome();
        }
    }

    /**
     * Resets cached hover chrome state from the DOM.
     */
    resetNodeChrome() {
        if (this.element) {
            this.isHovered = this.element.matches(":hover");
            if (!this.isDragging) {
                this.updateNodeHoverChrome();
            }
        }
    }

    /**
     * Renders the node header, badges, and port columns.
     */
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
        const hasTimestep = timestep !== null;

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
                    display: ${hasTimestep ? "flex" : "none"};
                    align-items: center;
                    justify-content: center;
                    font-size: 12px;
                    font-weight: 600;
                    color: white;
                    z-index: 10;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
                 ">${hasTimestep ? timestep : ""}</div>
                <div class="profiling-badge" style="
                    position: absolute;
                    top: -12px;
                    right: -12px;
                    min-width: 44px;
                    height: 24px;
                    padding: 0 8px;
                    background-color: #14532d;
                    border: 2px solid #404040;
                    border-radius: 8px;
                    display: none;
                    align-items: center;
                    justify-content: center;
                    font-size: 11px;
                    font-weight: 700;
                    color: #dcfce7;
                    z-index: 10;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
                "></div>
                <div style="display: flex; align-items: center; gap: 8px; flex: 1; min-width: 0;">
                    <span class="node-category-badge" style="
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
                    <span class="docking-status-badge" style="
                        display: ${this.isDockInvalid ? "inline-flex" : "none"};
                        align-items: center;
                        color: #ffb4b4;
                        font-size: 9px;
                        font-weight: 700;
                        letter-spacing: 0.4px;
                        flex-shrink: 0;
                    ">UNBOUND</span>
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
                <div style="display: flex; gap: 4px; margin-left: 8px; flex-shrink: 0;">
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
        this.applyIslandInactiveState();
        this.applyDockState();
        this.cachedElementWidth =
            this.element.offsetWidth || this.cachedElementWidth;
    }

    /**
     * Applies the visual and interaction state of a metadata-driven dock.
     *
     * @param {{docked:boolean, invalid:boolean}} state Dock state.
     */
    setDockState({ docked = false, invalid = false } = {}) {
        const normalizedDocked = Boolean(docked);
        const normalizedInvalid = Boolean(invalid);
        if (
            normalizedDocked === this.isDocked &&
            normalizedInvalid === this.isDockInvalid
        ) {
            return;
        }
        this.isDocked = normalizedDocked;
        this.isDockInvalid = normalizedInvalid;
        this.applyDockState();
    }

    /**
     * Updates dock state chrome without rebuilding the node.
     */
    applyDockState() {
        if (!this.element) return;
        const badge = this.element.querySelector(".docking-status-badge");
        if (badge) {
            badge.style.display = this.isDockInvalid ? "inline-flex" : "none";
        }
        this.element.classList.toggle("flowchart-node-docked", this.isDocked);
        this.element.classList.toggle(
            "flowchart-node-dock-invalid",
            this.isDockInvalid,
        );
        if (this.isDocked) {
            this.element.style.cursor = "default";
            this.element.style.borderColor = "#f9c845";
            this.element.style.borderStyle = "solid";
            this.element.title =
                "Docked to Device Input. Remove the dock connection to detach.";
        } else if (this.isDockInvalid) {
            this.element.style.cursor = "move";
            this.element.style.borderColor = "#ff6b6b";
            this.element.style.borderStyle = "dashed";
            this.element.title =
                "Unbound: connect Device Input frame to dock this detector.";
        } else {
            this.element.style.cursor = "move";
            this.element.style.borderStyle = "solid";
            this.element.removeAttribute("title");
            this.updateNodeHoverChrome();
        }
    }

    /**
     * Marks the node as inactive within an operation island.
     * @param {boolean} isInactive - Whether the node is inactive.
     */
    setIslandInactive(isInactive) {
        this.isIslandInactive = Boolean(isInactive);
        this.applyIslandInactiveState();
    }

    /**
     * Applies inactive-island visual treatment to the category badge.
     */
    applyIslandInactiveState() {
        if (!this.element) {
            return;
        }

        const categoryBadge = this.element.querySelector(
            ".node-category-badge",
        );
        if (!categoryBadge) {
            return;
        }

        if (this.isIslandInactive) {
            categoryBadge.style.filter = "grayscale(100%)";
            categoryBadge.style.opacity = "0.58";
            categoryBadge.title =
                "Operation Island: this operation will not execute in the current configuration.";
        } else {
            categoryBadge.style.filter = "";
            categoryBadge.style.opacity = "";
            categoryBadge.removeAttribute("title");
        }
    }

    /**
     * Renders input port markup for static and dynamic ports.
     * @returns {string} HTML markup for input ports.
     */
    renderInputPorts() {
        const staticPortsMarkup = this.staticInputNodes
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

        if (!this.dynamicGroup || !this.dynamicGroup.hasDynamicInputGroup) {
            return staticPortsMarkup;
        }

        const dynamicRows = this.dynamicInputNodes
            .map((nodeName, index) => {
                const displayLabel = `${this.dynamicGroup.inputBaseName}`;
                return `
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
                ">${escapeHtml(displayLabel)}</span>
            </div>
        `;
            })
            .join("");

        return `
            ${staticPortsMarkup}
            <div class="dynamic-port-group" style="
                margin: 4px 8px 0 8px;
                border: 2px solid #f9c845;
                border-radius: 8px;
                padding: 4px 0;
                display: inline-block;
                width: fit-content;
                max-width: calc(100% - 16px);
                box-shadow: 0 0 8px rgba(249, 200, 69, 0.25);
            ">
                <div style="
                    color: #f9c845;
                    font-size: 8px;
                    font-weight: 600;
                    letter-spacing: 0;
                    display: flex;
                    align-items: center;
                    height: 16px;
                    line-height: 16px;
                    padding: 0 8px 2px 8px;
                ">(${this.dynamicInputNodes.length}/${this.dynamicGroup.maxInputs})</div>
                ${dynamicRows}
            </div>
        `;
    }

    /**
     * Renders output port markup for static and dynamic ports.
     * @returns {string} HTML markup for output ports.
     */
    renderOutputPorts() {
        const staticPortsMarkup = this.staticOutputNodes
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

        if (!this.dynamicGroup || !this.dynamicGroup.hasDynamicOutputGroup) {
            return staticPortsMarkup;
        }

        const dynamicRows = this.dynamicOutputNodes
            .map((nodeName, index) => {
                const displayLabel = `${this.dynamicGroup.outputBaseName}`;
                return `
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
                ">${escapeHtml(displayLabel)}</span>
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
        `;
            })
            .join("");

        return `
            ${staticPortsMarkup}
            <div class="dynamic-port-group" style="
                margin: 4px 8px 0 8px;
                border: 2px solid #f9c845;
                border-radius: 8px;
                padding: 4px 0;
                display: inline-block;
                width: fit-content;
                max-width: calc(100% - 16px);
                margin-left: auto;
                box-shadow: 0 0 8px rgba(249, 200, 69, 0.25);
            ">
                <div style="
                    color: #f9c845;
                    font-size: 8px;
                    font-weight: 600;
                    letter-spacing: 0;
                    display: flex;
                    align-items: center;
                    justify-content: flex-end;
                    height: 16px;
                    line-height: 16px;
                    padding: 0 8px 2px 8px;
                    text-align: right;
                ">(${this.dynamicOutputNodes.length}/${this.dynamicGroup.maxOutputs})</div>
                ${dynamicRows}
            </div>
        `;
    }

    /**
     * Caches connector elements and attaches port listeners.
     */
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

    /**
     * Attaches hover and click handlers to a port connector.
     * @param {HTMLElement} portElement - Port connector element.
     * @param {string} portName - Port name.
     * @param {string} portType - Port side.
     */
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

    /**
     * Attaches hover and click handlers to node action buttons.
     */
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

    /**
     * Attaches drag-start handling to the node element.
     */
    setupDragListeners() {
        this.element.addEventListener(
            "mousedown",
            this.handleDragStart.bind(this),
        );
    }

    /**
     * Starts dragging the node and tracks pointer movement.
     * @param {MouseEvent} event - Mouse down event.
     */
    handleDragStart(event) {
        // Only handle left mouse button. Docked nodes follow their source and
        // can be detached only by removing their ordinary connection.
        if (event.button !== 0 || this.isDocked) {
            if (this.isDocked) {
                event.preventDefault();
                event.stopPropagation();
            }
            return;
        }

        this.isDragging = true;
        // Find the canvas to get scale and translate
        // We assume the canvas instance is available via some global or we can find it
        // For now we'll stick to DOM inspection but make it more robust

        const viewport = this.element.closest("#flowchartViewport");
        const scale = this.getCanvasScale(viewport);
        const rect = this.element.getBoundingClientRect();
        const containerRect = viewport?.parentElement?.getBoundingClientRect();
        const translate = this.getCanvasTranslate(viewport);

        this.dragContext = {
            viewport,
            scale,
            translate,
            containerLeft: containerRect?.left ?? 0,
            containerTop: containerRect?.top ?? 0,
        };

        this.dragOffsetX = (event.clientX - rect.left) / scale;
        this.dragOffsetY = (event.clientY - rect.top) / scale;

        this.element.style.zIndex = "100";
        this.element.style.cursor = "grabbing";
        this.element.style.borderColor = "#f9c845";
        this.element.style.boxShadow = "8px 8px 24px rgba(0, 0, 0, 0.7)";

        if (this.onDragStart) {
            this.onDragStart(this, event);
        }

        /**
         * Updates node position while dragging.
         * @param {MouseEvent} e - Mouse move event.
         */
        const handleDragMove = (e) => {
            if (!this.isDragging) return;

            const context = this.dragContext;
            if (!context?.viewport) return;

            // Calculate position in world coordinates
            let worldX =
                (e.clientX - context.containerLeft - context.translate.x) /
                    context.scale -
                this.dragOffsetX;
            let worldY =
                (e.clientY - context.containerTop - context.translate.y) /
                    context.scale -
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

        /**
         * Ends node dragging and restores interaction state.
         */
        const handleDragEnd = () => {
            if (!this.isDragging) return;

            this.isDragging = false;
            this.dragContext = null;
            this.element.style.zIndex = "10";
            this.element.style.cursor = "move";
            this.updateNodeHoverChrome();

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

    /**
     * Reads the current canvas scale from the viewport transform.
     * @param {HTMLElement|null} viewport - Optional viewport element.
     * @returns {number} Current scale factor.
     */
    getCanvasScale(viewport = null) {
        viewport = viewport || this.element.closest("#flowchartViewport");
        if (!viewport) return 1;

        const transform = viewport.style.transform;
        const scaleMatch = transform.match(/scale\(([^)]+)\)/);
        return scaleMatch ? Number.parseFloat(scaleMatch[1]) : 1;
    }

    /**
     * Reads the current canvas translation from the viewport transform.
     * @param {HTMLElement|null} viewport - Optional viewport element.
     * @returns {{x:number,y:number}} Current translation.
     */
    getCanvasTranslate(viewport = null) {
        viewport = viewport || this.element.closest("#flowchartViewport");
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

    /**
     * Gets the absolute flowchart-world position of a port connector center.
     * @param {HTMLElement} port - Port connector element.
     * @returns {{x:number,y:number}|null} Port center position or null.
     */
    getPortCenterPosition(port) {
        if (!port || !this.element) return null;

        const viewport = this.element.closest("#flowchartViewport");
        const scale = this.getCanvasScale(viewport);
        const portRect = port.getBoundingClientRect();
        const nodeRect = this.element.getBoundingClientRect();

        return {
            x:
                this.position.x +
                (portRect.left + portRect.width / 2 - nodeRect.left) / scale,
            y:
                this.position.y +
                (portRect.top + portRect.height / 2 - nodeRect.top) / scale,
        };
    }

    /**
     * Gets the absolute position of an input port.
     * @param {string} portName - Port name.
     * @returns {{x:number,y:number}|null} Port position or null.
     */
    getInputPortPosition(portName) {
        const port = this.inputPorts.get(portName);
        return this.getPortCenterPosition(port);
    }

    /**
     * Gets the absolute position of an output port.
     * @param {string} portName - Port name.
     * @returns {{x:number,y:number}|null} Port position or null.
     */
    getOutputPortPosition(portName) {
        const port = this.outputPorts.get(portName);
        return this.getPortCenterPosition(port);
    }

    /**
     * Updates the node position and element coordinates.
     * @param {number} x - X position.
     * @param {number} y - Y position.
     */
    setPosition(x, y) {
        this.position.x = x;
        this.position.y = y;
        if (this.element) {
            this.element.style.left = `${x}px`;
            this.element.style.top = `${y}px`;
        }
    }

    /**
     * Returns the current node position.
     * @returns {{x:number,y:number}} Current position.
     */
    getPosition() {
        return { ...this.position };
    }

    /**
     * Checks whether an input port supports a default value.
     * @param {string} portName - Port name.
     * @returns {boolean} True when the port has a default.
     */
    canInputPortBeDefault(portName) {
        const config = this.inputNodeConfig.get(portName);
        return config?.hasDefault ?? false;
    }

    /**
     * Maps a thread number to a display color.
     * @param {number} threadNumber - Thread identifier.
     * @returns {string|null} Color string or null.
     */
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

    /**
     * Updates the thread badge with execution thread info.
     * @param {object|null} threadInfo - Thread metadata.
     */
    updateThreadInfo(threadInfo) {
        this.threadInfo = threadInfo;
        const badge = this.element?.querySelector(".thread-badge");
        if (!badge) return;

        const timestep = threadInfo?.timestep ?? null;
        if (timestep === null) {
            badge.style.display = "none";
            return;
        }

        badge.style.display = "flex";
        badge.style.backgroundColor = this.getThreadColor(threadInfo?.thread);
        badge.textContent = timestep;
    }

    /**
     * Hides the thread badge and clears cached thread info.
     */
    hideThreadBadge() {
        this.threadInfo = null;
        const badge = this.element?.querySelector(".thread-badge");
        if (badge) {
            badge.style.display = "none";
        }
    }

    /**
     * Updates the profiling badge with execution timing info.
     * @param {object|null} profilingInfo - Profiling metadata.
     */
    updateProfilingInfo(profilingInfo) {
        this.profilingInfo = profilingInfo;
        const badge = this.element?.querySelector(".profiling-badge");
        if (!badge) {
            return;
        }

        const executionTimeMs = Number(profilingInfo?.execution_time_ms);
        if (!Number.isFinite(executionTimeMs) || executionTimeMs < 0) {
            badge.style.display = "none";
            return;
        }

        badge.style.display = "flex";
        badge.textContent = `${executionTimeMs.toFixed(2)}ms`;
    }

    /**
     * Hides the profiling badge and clears cached profiling info.
     */
    hideProfilingBadge() {
        this.profilingInfo = null;
        const badge = this.element?.querySelector(".profiling-badge");
        if (badge) {
            badge.style.display = "none";
        }
    }

    /**
     * Applies or clears error-state visuals for the node.
     * @param {object|null} errorRecord - Error data, if any.
     * @param {boolean} isDownstream - Whether the node is downstream-disabled.
     */
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
            this.updateNodeHoverChrome();
        }

        if (nodeIcon) {
            nodeIcon.style.display = hasError ? "inline-flex" : "none";
        }
    }

    /**
     * Removes the node element and clears cached port references.
     */
    destroy() {
        this.element?.remove();
        this.inputPorts.clear();
        this.outputPorts.clear();
    }
}
