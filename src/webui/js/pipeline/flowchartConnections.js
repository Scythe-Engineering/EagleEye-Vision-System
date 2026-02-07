/**
 * FlowchartConnections - SVG orthogonal connection management with data type labels
 */

import { ManualPathCreator } from "./manualPathCreator.js";

export class FlowchartConnections {
    constructor(svgLayer, options = {}) {
        this.svgLayer = svgLayer;
        this.connections = new Map();
        this.connectionColor = options.connectionColor || "#f9c845";
        this.connectionWidth = options.connectionWidth || 2;
        this.labelFontSize = options.labelFontSize || 10;
        this.onConnectionRemoved = options.onConnectionRemoved || (() => {});
        this.onConnectionChanged = options.onConnectionChanged || (() => {});
        this.onGetNode = options.onGetNode || (() => null);
        this.onCheckDefaultAllowed =
            options.onCheckDefaultAllowed || (() => true);
        this.canvas = options.canvas || null;

        this.manualPathCreator = new ManualPathCreator(svgLayer, {
            canvas: this.canvas,
            connectionColor: this.connectionColor,
            gridSpacing: this.canvas?.gridSpacing || 20,
            cornerRadius: 12,
            onComplete: this.handleManualPathComplete.bind(this),
            onCancel: this.handleManualPathCancel.bind(this),
        });

        this.setupDefs();
    }

    setupDefs() {
        let defs = this.svgLayer.querySelector("defs");
        if (!defs) {
            defs = document.createElementNS(
                "http://www.w3.org/2000/svg",
                "defs",
            );
            this.svgLayer.insertBefore(defs, this.svgLayer.firstChild);
        }

        const markerId = "flowchart-arrow";
        if (!defs.querySelector(`#${markerId}`)) {
            const marker = document.createElementNS(
                "http://www.w3.org/2000/svg",
                "marker",
            );
            marker.setAttribute("id", markerId);
            marker.setAttribute("viewBox", "0 0 10 10");
            marker.setAttribute("refX", "8");
            marker.setAttribute("refY", "5");
            marker.setAttribute("markerWidth", "6");
            marker.setAttribute("markerHeight", "6");
            marker.setAttribute("orient", "auto-start-reverse");

            const path = document.createElementNS(
                "http://www.w3.org/2000/svg",
                "path",
            );
            path.setAttribute("d", "M 0 0 L 10 5 L 0 10 z");
            path.setAttribute("fill", this.connectionColor);

            marker.appendChild(path);
            defs.appendChild(marker);
        }
    }

    createConnection(...args) {
        const options =
            args.length === 1 && typeof args[0] === "object" && args[0] !== null
                ? args[0]
                : {
                      connectionId: args[0],
                      fromNode: args[1],
                      fromPortName: args[2],
                      toNode: args[3],
                      toPortName: args[4],
                      dataType: args[5],
                      isDefault: args[6] ?? false,
                      customWaypoints: args[7] ?? null,
                  };
        const {
            connectionId,
            fromNode,
            fromPortName,
            toNode,
            toPortName,
            dataType,
            isDefault = false,
            customWaypoints = null,
        } = options;
        if (this.connections.has(connectionId)) {
            this.updateConnection(
                connectionId,
                fromNode,
                fromPortName,
                toNode,
                toPortName,
            );
            const existing = this.connections.get(connectionId);
            if (existing) {
                existing.isDefault = isDefault;
                if (isDefault) {
                    existing.path.setAttribute("stroke-dasharray", "5,5");
                } else {
                    existing.path.removeAttribute("stroke-dasharray");
                }
                if (customWaypoints) {
                    existing.customWaypoints = customWaypoints;
                    this.updateConnectionWithWaypoints(connectionId);
                }
            }
            return;
        }

        const group = document.createElementNS(
            "http://www.w3.org/2000/svg",
            "g",
        );
        group.dataset.connectionId = connectionId;
        group.setAttribute("pointer-events", "visibleStroke");

        const path = document.createElementNS(
            "http://www.w3.org/2000/svg",
            "path",
        );
        path.setAttribute("fill", "none");
        path.setAttribute("stroke", this.connectionColor);
        path.setAttribute("stroke-width", this.connectionWidth.toString());
        path.setAttribute("stroke-linecap", "round");
        path.setAttribute("marker-end", "url(#flowchart-arrow)");
        path.style.transition = "stroke 0.15s ease, stroke-width 0.15s ease";
        path.setAttribute("pointer-events", "none");

        const hitArea = document.createElementNS(
            "http://www.w3.org/2000/svg",
            "path",
        );
        hitArea.setAttribute("fill", "none");
        hitArea.setAttribute("stroke", "transparent");
        hitArea.setAttribute("stroke-width", "20");
        hitArea.style.cursor = "pointer";
        hitArea.setAttribute("pointer-events", "auto");

        const labelGroup = document.createElementNS(
            "http://www.w3.org/2000/svg",
            "g",
        );
        labelGroup.setAttribute("class", "connection-label");
        labelGroup.style.opacity = "0";
        labelGroup.style.transition = "opacity 0.3s ease";

        const labelBackground = document.createElementNS(
            "http://www.w3.org/2000/svg",
            "rect",
        );
        labelBackground.setAttribute("fill", "#1f1f1f");
        labelBackground.setAttribute("stroke", "#404040");
        labelBackground.setAttribute("stroke-width", "1");
        labelBackground.setAttribute("rx", "4");
        labelBackground.setAttribute("ry", "4");

        const labelText = document.createElementNS(
            "http://www.w3.org/2000/svg",
            "text",
        );
        labelText.setAttribute("fill", "#a0a0a0");
        labelText.setAttribute("font-size", this.labelFontSize.toString());
        labelText.setAttribute(
            "font-family",
            "system-ui, -apple-system, sans-serif",
        );
        labelText.setAttribute("text-anchor", "middle");
        labelText.setAttribute("dominant-baseline", "middle");
        labelText.textContent = dataType || fromPortName;

        labelGroup.appendChild(labelBackground);
        labelGroup.appendChild(labelText);

        group.appendChild(hitArea);
        group.appendChild(path);
        group.appendChild(labelGroup);

        this.setupHoverEffects(group, path);

        this.svgLayer.appendChild(group);

        this.connections.set(connectionId, {
            group,
            path,
            hitArea,
            labelGroup,
            labelBackground,
            labelText,
            fromNodeId: fromNode.instanceId,
            fromPortName,
            toNodeId: toNode.instanceId,
            toPortName,
            dataType,
            isDefault: isDefault || false,
            isAnimating: false,
            customWaypoints: customWaypoints ? [...customWaypoints] : null,
        });

        if (customWaypoints && customWaypoints.length >= 2) {
            this.updateConnectionWithWaypoints(connectionId);
        } else {
            this.updateConnection(
                connectionId,
                fromNode,
                fromPortName,
                toNode,
                toPortName,
            );
        }

        // Apply default visual style
        if (isDefault) {
            path.setAttribute("stroke-dasharray", "5,5");
        }

        // Fade in label after initial position is set
        requestAnimationFrame(() => {
            labelGroup.style.opacity = "1";
        });
    }

    setupHoverEffects(group, path) {
        path.style.transition =
            "stroke 0.15s ease, stroke-width 0.15s ease, filter 0.15s ease, opacity 0.15s ease, stroke-dasharray 0.15s ease";

        group.addEventListener("mouseenter", () => {
            const connectionId = group.dataset.connectionId;
            if (connectionId) {
                this.setHoverState(connectionId, true);
            }
        });

        group.addEventListener("mouseleave", () => {
            const connectionId = group.dataset.connectionId;
            if (connectionId) {
                this.setHoverState(connectionId, false);
            }
        });

        group.addEventListener("click", (e) => {
            e.stopPropagation();
            const connectionId = group.dataset.connectionId;
            if (connectionId) {
                // Fade out before removal
                path.style.opacity = "0";
                setTimeout(() => {
                    this.removeConnection(connectionId);
                    this.onConnectionRemoved(connectionId);
                }, 150);
            }
        });

        group.addEventListener("contextmenu", (e) => {
            e.preventDefault();
            e.stopPropagation();
            const connectionId = group.dataset.connectionId;
            if (connectionId) {
                this.showContextMenu(e.clientX, e.clientY, connectionId);
            }
        });
    }

    showContextMenu(x, y, connectionId) {
        // Remove existing context menu if any
        const existingMenu = document.getElementById("connection-context-menu");
        if (existingMenu) {
            existingMenu.remove();
        }

        const menu = document.createElement("div");
        menu.id = "connection-context-menu";
        menu.style.position = "fixed";
        menu.style.left = x + "px";
        menu.style.top = y + "px";
        menu.style.backgroundColor = "#2a2a2a";
        menu.style.border = "1px solid #404040";
        menu.style.borderRadius = "6px";
        menu.style.padding = "4px 0";
        menu.style.zIndex = "10000";
        menu.style.boxShadow = "0 4px 12px rgba(0,0,0,0.5)";
        menu.style.minWidth = "200px";

        const connection = this.connections.get(connectionId);
        const isDefault = connection?.isDefault || false;
        const defaultAllowed = this.onCheckDefaultAllowed(
            connection?.toNodeId,
            connection?.toPortName,
        );

        // Toggle Default option
        const toggleItem = document.createElement("div");
        if (!defaultAllowed && !isDefault) {
            toggleItem.textContent = "Cannot Set Default";
            toggleItem.style.cursor = "not-allowed";
            toggleItem.style.color = "#666666";
        } else {
            toggleItem.textContent = isDefault
                ? "Remove Default Status"
                : "Set as Default Connection";
            toggleItem.style.cursor = "pointer";
            toggleItem.style.color = "#f9c845";
        }
        toggleItem.style.padding = "8px 12px";
        toggleItem.style.fontSize = "13px";
        toggleItem.style.fontWeight = "500";
        toggleItem.style.transition = "background-color 0.15s ease";
        toggleItem.style.borderTop = "1px solid #404040";

        if (defaultAllowed || isDefault) {
            toggleItem.addEventListener("mouseenter", () => {
                toggleItem.style.backgroundColor = "#3a3a3a";
            });
            toggleItem.addEventListener("mouseleave", () => {
                toggleItem.style.backgroundColor = "transparent";
            });
            toggleItem.addEventListener("click", () => {
                this.toggleDefault(connectionId);
                menu.remove();
            });
        }

        // Remove option
        const removeItem = document.createElement("div");
        removeItem.textContent = "Remove Connection";
        removeItem.style.padding = "8px 12px";
        removeItem.style.cursor = "pointer";
        removeItem.style.color = "#ff6b6b";
        removeItem.style.fontSize = "13px";
        removeItem.style.fontWeight = "500";
        removeItem.style.transition = "background-color 0.15s ease";
        removeItem.style.borderTop = "1px solid #404040";

        removeItem.addEventListener("mouseenter", () => {
            removeItem.style.backgroundColor = "#3a3a3a";
        });
        removeItem.addEventListener("mouseleave", () => {
            removeItem.style.backgroundColor = "transparent";
        });
        removeItem.addEventListener("click", () => {
            // Fade out before removal
            const connection = this.connections.get(connectionId);
            if (connection) {
                connection.path.style.opacity = "0";
                setTimeout(() => {
                    this.removeConnection(connectionId);
                    this.onConnectionRemoved(connectionId);
                }, 150);
            }
            menu.remove();
        });

        const manualPathItem = document.createElement("div");
        const hasCustomPath = connection?.customWaypoints !== null;
        manualPathItem.textContent = hasCustomPath
            ? "Edit Custom Path"
            : "Manually Create Path";
        manualPathItem.style.padding = "8px 12px";
        manualPathItem.style.cursor = "pointer";
        manualPathItem.style.color = "#60a5fa";
        manualPathItem.style.fontSize = "13px";
        manualPathItem.style.fontWeight = "500";
        manualPathItem.style.transition = "background-color 0.15s ease";

        manualPathItem.addEventListener("mouseenter", () => {
            manualPathItem.style.backgroundColor = "#3a3a3a";
        });
        manualPathItem.addEventListener("mouseleave", () => {
            manualPathItem.style.backgroundColor = "transparent";
        });
        manualPathItem.addEventListener("click", () => {
            this.startManualPathCreation(connectionId);
            menu.remove();
        });

        const resetPathItem = document.createElement("div");
        resetPathItem.textContent = "Reset to Auto Path";
        resetPathItem.style.padding = "8px 12px";
        resetPathItem.style.cursor = hasCustomPath ? "pointer" : "not-allowed";
        resetPathItem.style.color = hasCustomPath ? "#a78bfa" : "#666666";
        resetPathItem.style.fontSize = "13px";
        resetPathItem.style.fontWeight = "500";
        resetPathItem.style.transition = "background-color 0.15s ease";

        if (hasCustomPath) {
            resetPathItem.addEventListener("mouseenter", () => {
                resetPathItem.style.backgroundColor = "#3a3a3a";
            });
            resetPathItem.addEventListener("mouseleave", () => {
                resetPathItem.style.backgroundColor = "transparent";
            });
            resetPathItem.addEventListener("click", () => {
                this.resetToAutoPath(connectionId);
                menu.remove();
            });
        }

        menu.appendChild(manualPathItem);
        menu.appendChild(resetPathItem);
        menu.appendChild(toggleItem);
        menu.appendChild(removeItem);
        document.body.appendChild(menu);

        // Close menu when clicking elsewhere
        const closeMenu = (e) => {
            if (!menu.contains(e.target)) {
                menu.remove();
                document.removeEventListener("click", closeMenu);
            }
        };

        setTimeout(() => {
            document.addEventListener("click", closeMenu);
        }, 10);
    }

    toggleDefault(connectionId) {
        const connection = this.connections.get(connectionId);
        if (!connection) return;

        connection.isDefault = !connection.isDefault;

        // Update visual style
        if (connection.isDefault) {
            connection.path.setAttribute("stroke-dasharray", "5,5");
        } else {
            connection.path.removeAttribute("stroke-dasharray");
        }

        // Reset cycle highlight if toggling default
        if (connection.isCycle) {
            this.setCycleHighlight(connectionId, false);
        }

        // Trigger callback for saving and cycle detection
        if (this.onConnectionChanged) {
            this.onConnectionChanged(connectionId);
        }
    }

    setCycleHighlight(connectionId, isCycle) {
        const connection = this.connections.get(connectionId);
        if (!connection) return;

        connection.isCycle = isCycle;

        if (isCycle) {
            connection.path.setAttribute("stroke", "#ff4444");
            connection.path.style.filter = "drop-shadow(0 0 6px #ff4444)";
        } else {
            connection.path.setAttribute("stroke", this.connectionColor);
            connection.path.style.filter = "none";
        }
    }

    updateConnection(
        connectionId,
        fromNode,
        fromPortName,
        toNode,
        toPortName,
        skipCache = false,
    ) {
        const connection = this.connections.get(connectionId);
        if (!connection) return;

        const fromPos = fromNode.getOutputPortPosition(fromPortName);
        const toPos = toNode.getInputPortPosition(toPortName);

        if (!fromPos || !toPos) return;

        if (!skipCache) {
            const posKey = `${fromPos.x},${fromPos.y},${toPos.x},${toPos.y}`;
            if (connection.lastPosKey === posKey) {
                return;
            }
            connection.lastPosKey = posKey;
        }

        let pathD;
        if (
            connection.customWaypoints &&
            connection.customWaypoints.length >= 2
        ) {
            const startOffset = 10;
            const endOffset = 10;
            const newStartPoint = { x: fromPos.x + startOffset, y: fromPos.y };
            const newEndPoint = { x: toPos.x - endOffset, y: toPos.y };

            const len = connection.customWaypoints.length;
            const updatedWaypoints = connection.customWaypoints.map(
                (wp, index) => {
                    if (index === 0) {
                        return { ...newStartPoint };
                    }
                    if (index === 1 && len > 2) {
                        return { x: wp.x, y: newStartPoint.y };
                    }
                    if (index === len - 2 && len > 2) {
                        return { x: wp.x, y: newEndPoint.y };
                    }
                    if (index === len - 1) {
                        return { ...newEndPoint };
                    }
                    return { ...wp };
                },
            );

            connection.customWaypoints = updatedWaypoints;
            pathD = this.buildPathFromWaypoints(updatedWaypoints);
        } else {
            pathD = this.calculateOrthogonalPath(fromPos, toPos);
        }

        connection.path.setAttribute("d", pathD);
        connection.hitArea.setAttribute("d", pathD);

        this.updateLabel(connection);
    }

    calculateOrthogonalPath(from, to) {
        const startOffset = 10;
        const endOffset = 10;
        const cornerRadius = 12;

        const startX = from.x + startOffset;
        const startY = from.y;
        const endX = to.x - endOffset;
        const endY = to.y;

        const midX = startX + (endX - startX) / 2;
        return this.buildSimpleOrthogonalPath(
            startX,
            startY,
            endX,
            endY,
            midX,
            cornerRadius,
        );
    }

    buildSimpleOrthogonalPath(startX, startY, endX, endY, midX, cornerRadius) {
        const segments = [];
        segments.push(`M ${startX} ${startY}`);

        // Determine the overall horizontal direction of the connection
        const horizontalDirection = Math.sign(endX - startX);

        this.addHorizontalSegment(segments, startX, startY, midX, cornerRadius);
        this.addVerticalSegment(
            segments,
            midX,
            startY,
            endY,
            cornerRadius,
            horizontalDirection,
        );

        // After vertical segment ends at endY, the corner should lead towards endX
        const lastCornerX =
            midX + (horizontalDirection > 0 ? cornerRadius : -cornerRadius);
        this.addHorizontalSegment(segments, lastCornerX, endY, endX, 0);

        return segments.join(" ");
    }

    addHorizontalSegment(segments, fromX, y, toX, radius) {
        if (Math.abs(toX - fromX) < radius) {
            segments.push(`L ${toX} ${y}`);
            return;
        }

        const direction = Math.sign(toX - fromX);
        segments.push(`L ${toX - radius * direction} ${y}`);
    }

    addVerticalSegment(
        segments,
        x,
        fromY,
        toY,
        radius,
        horizontalDirection = 1,
    ) {
        if (Math.abs(toY - fromY) < radius * 2) {
            segments.push(`L ${x} ${toY}`);
            return;
        }

        const direction = Math.sign(toY - fromY);
        const startCornerY = fromY + radius * direction;
        const endCornerY = toY - radius * direction;

        segments.push(`Q ${x} ${fromY}, ${x} ${startCornerY}`);

        if (Math.abs(endCornerY - startCornerY) > 1) {
            segments.push(`L ${x} ${endCornerY}`);
        }

        // Adjust the final corner based on horizontal direction
        // For rightward connections (positive), curve right; for leftward, curve left
        const horizontalOffset = radius * horizontalDirection;
        segments.push(`Q ${x} ${toY}, ${x + horizontalOffset} ${toY}`);
    }

    calculateMultiSegmentPath(from, to) {
        return this.calculateOrthogonalPath(from, to);
    }

    updateLabel(connection) {
        const path = connection.path;
        const labelGroup = connection.labelGroup;
        const text = connection.labelText;
        const background = connection.labelBackground;

        const pathLength = path.getTotalLength();
        const midPoint = path.getPointAtLength(pathLength / 2);

        labelGroup.setAttribute(
            "transform",
            `translate(${midPoint.x}, ${midPoint.y})`,
        );

        if (!connection.labelDimensions) {
            const textBBox = text.getBBox();
            const padding = 6;
            connection.labelDimensions = {
                width: textBBox.width + padding * 2,
                height: textBBox.height + padding,
            };
        }

        const { width, height } = connection.labelDimensions;
        background.setAttribute("x", (-width / 2).toString());
        background.setAttribute("y", (-height / 2).toString());
        background.setAttribute("width", width.toString());
        background.setAttribute("height", height.toString());

        text.setAttribute("x", "0");
        text.setAttribute("y", "0");
    }

    updateConnectionPath(connectionId) {
        const connection = this.connections.get(connectionId);
        if (!connection) return;

        const fromNode = this.onGetNode(connection.fromNodeId);
        const toNode = this.onGetNode(connection.toNodeId);

        if (!fromNode || !toNode) return;

        const fromPos = fromNode.getOutputPortPosition(connection.fromPortName);
        const toPos = toNode.getInputPortPosition(connection.toPortName);

        if (!fromPos || !toPos) return;

        const pathData = this.calculateMultiSegmentPath(fromPos, toPos);
        connection.path.setAttribute("d", pathData);
        connection.hitArea.setAttribute("d", pathData);
    }

    updateAllConnections(nodeMap, changedNodeIds = null, skipCache = false) {
        for (const [connectionId, connection] of this.connections) {
            if (
                changedNodeIds &&
                !changedNodeIds.has(connection.fromNodeId) &&
                !changedNodeIds.has(connection.toNodeId)
            ) {
                continue;
            }

            const fromNode = nodeMap.get(connection.fromNodeId);
            const toNode = nodeMap.get(connection.toNodeId);

            if (fromNode && toNode) {
                this.updateConnection(
                    connectionId,
                    fromNode,
                    connection.fromPortName,
                    toNode,
                    connection.toPortName,
                    skipCache,
                );
            }
        }
    }

    removeConnection(connectionId) {
        const connection = this.connections.get(connectionId);
        if (connection) {
            connection.group.remove();
            this.connections.delete(connectionId);
        }
    }

    removeConnectionsForNode(nodeInstanceId) {
        const toRemove = [];

        for (const [connectionId, connection] of this.connections) {
            if (
                connection.fromNodeId === nodeInstanceId ||
                connection.toNodeId === nodeInstanceId
            ) {
                toRemove.push(connectionId);
            }
        }

        toRemove.forEach((id) => this.removeConnection(id));
    }

    clearAllConnections() {
        for (const [connectionId] of this.connections) {
            this.removeConnection(connectionId);
        }
        this.connections.clear();
    }

    setHoverState(connectionId, isHovered) {
        const connection = this.connections.get(connectionId);
        if (!connection) return;

        if (isHovered) {
            const hoverColor = connection.isCycle
                ? "#ff4444"
                : this.connectionColor;
            connection.path.setAttribute(
                "stroke-width",
                (this.connectionWidth + 2).toString(),
            );
            connection.path.style.filter = `drop-shadow(0 0 4px ${hoverColor})`;
        } else if (connection.isCycle) {
            connection.path.setAttribute("stroke", "#ff4444");
            connection.path.setAttribute(
                "stroke-width",
                this.connectionWidth.toString(),
            );
            connection.path.style.filter = "drop-shadow(0 0 6px #ff4444)";
        } else {
            connection.path.setAttribute("stroke", this.connectionColor);
            connection.path.setAttribute(
                "stroke-width",
                this.connectionWidth.toString(),
            );
            connection.path.style.filter = "none";
        }
    }

    highlightConnection(connectionId, highlight = true) {
        this.setHoverState(connectionId, highlight);
    }

    getConnectionsForPort(nodeInstanceId, portName, portType) {
        const result = [];

        for (const [connectionId, connection] of this.connections) {
            const isOutputMatch =
                portType === "output" &&
                connection.fromNodeId === nodeInstanceId &&
                connection.fromPortName === portName;
            const isInputMatch =
                portType === "input" &&
                connection.toNodeId === nodeInstanceId &&
                connection.toPortName === portName;
            if (isOutputMatch || isInputMatch) {
                result.push(connectionId);
            }
        }

        return result;
    }

    createTemporaryConnection(startPos, options = {}) {
        const tempPath = document.createElementNS(
            "http://www.w3.org/2000/svg",
            "path",
        );
        tempPath.setAttribute("fill", "none");
        tempPath.setAttribute("stroke", this.connectionColor);

        // Check if we're morphing from a hovered connection
        const isFromHover = options.fromHover || false;

        if (isFromHover) {
            // Start with hovered state (thicker, solid line)
            tempPath.setAttribute(
                "stroke-width",
                (this.connectionWidth + 2).toString(),
            );
            tempPath.setAttribute("stroke-dasharray", "0");
            tempPath.setAttribute("opacity", "1");
            tempPath.style.filter =
                "drop-shadow(0 0 4px " + this.connectionColor + ")";
        } else {
            // Start invisible for new connections
            tempPath.setAttribute(
                "stroke-width",
                (this.connectionWidth + 2).toString(),
            );
            tempPath.setAttribute("stroke-dasharray", "5,5");
            tempPath.setAttribute("opacity", "0");
        }

        tempPath.id = "temp-connection";

        // Add transition for morphing effect
        tempPath.style.transition =
            "opacity 0.2s ease, stroke-width 0.2s ease, stroke-dasharray 0.2s ease, filter 0.2s ease";

        // Render immediately at start position
        const initialEndPos = { x: startPos.x + 1, y: startPos.y };
        const initialPathD = this.calculateOrthogonalPath(
            startPos,
            initialEndPos,
        );
        tempPath.setAttribute("d", initialPathD);

        this.svgLayer.appendChild(tempPath);

        // Trigger transition to dragging state
        requestAnimationFrame(() => {
            if (isFromHover) {
                // Morph from hover state to drag state
                tempPath.setAttribute("stroke-dasharray", "5,5");
                tempPath.setAttribute(
                    "stroke-width",
                    this.connectionWidth.toString(),
                );
                tempPath.setAttribute("opacity", "0.7");
                tempPath.style.filter = "none";
            } else {
                // Fade in for new connections
                tempPath.setAttribute("opacity", "0.7");
                tempPath.setAttribute(
                    "stroke-width",
                    this.connectionWidth.toString(),
                );
            }
        });

        return {
            update: (endPos) => {
                const pathD = this.calculateOrthogonalPath(startPos, endPos);
                tempPath.setAttribute("d", pathD);
            },
            remove: () => {
                tempPath.setAttribute("opacity", "0");
                setTimeout(() => tempPath.remove(), 200);
            },
        };
    }

    getConnectionData() {
        const data = [];
        for (const [connectionId, connection] of this.connections) {
            data.push({
                id: connectionId,
                fromNodeId: connection.fromNodeId,
                fromPortName: connection.fromPortName,
                toNodeId: connection.toNodeId,
                toPortName: connection.toPortName,
                dataType: connection.dataType,
                isDefault: connection.isDefault || false,
                customWaypoints: connection.customWaypoints || null,
            });
        }
        return data;
    }

    updateConnectionVisuals(connectionId, isDefault, isCycle) {
        const connection = this.connections.get(connectionId);
        if (!connection) return;

        // Update default state
        connection.isDefault = isDefault;
        if (isDefault) {
            connection.path.setAttribute("stroke-dasharray", "5,5");
        } else {
            connection.path.removeAttribute("stroke-dasharray");
        }

        // Update cycle state
        connection.isCycle = isCycle;
        if (isCycle) {
            connection.path.setAttribute("stroke", "#ff4444");
            connection.path.style.filter = "drop-shadow(0 0 6px #ff4444)";
        } else {
            connection.path.setAttribute("stroke", this.connectionColor);
            connection.path.style.filter = "none";
        }
    }

    startManualPathCreation(connectionId) {
        const connection = this.connections.get(connectionId);
        if (!connection) return;

        const fromNode = this.onGetNode(connection.fromNodeId);
        const toNode = this.onGetNode(connection.toNodeId);

        if (!fromNode || !toNode) return;

        const fromPos = fromNode.getOutputPortPosition(connection.fromPortName);
        const toPos = toNode.getInputPortPosition(connection.toPortName);

        if (!fromPos || !toPos) return;

        const startOffset = 10;
        const endOffset = 10;
        const startPoint = { x: fromPos.x + startOffset, y: fromPos.y };
        const endPoint = { x: toPos.x - endOffset, y: toPos.y };

        connection.path.style.opacity = "0.2";
        connection.hitArea.style.opacity = "0.2";

        this.manualPathCreator.start(connectionId, startPoint, endPoint);
    }

    handleManualPathComplete(connectionId, waypoints) {
        const connection = this.connections.get(connectionId);
        if (!connection) return;

        connection.customWaypoints = waypoints;
        connection.path.style.opacity = "1";
        connection.hitArea.style.opacity = "1";

        this.updateConnectionWithWaypoints(connectionId);

        if (this.onConnectionChanged) {
            this.onConnectionChanged(connectionId);
        }
    }

    handleManualPathCancel(connectionId) {
        const connection = this.connections.get(connectionId);
        if (!connection) return;

        connection.path.style.opacity = "1";
        connection.hitArea.style.opacity = "1";
    }

    resetToAutoPath(connectionId) {
        const connection = this.connections.get(connectionId);
        if (!connection) return;

        connection.customWaypoints = null;

        const fromNode = this.onGetNode(connection.fromNodeId);
        const toNode = this.onGetNode(connection.toNodeId);

        if (fromNode && toNode) {
            this.updateConnection(
                connectionId,
                fromNode,
                connection.fromPortName,
                toNode,
                connection.toPortName,
            );
        }

        if (this.onConnectionChanged) {
            this.onConnectionChanged(connectionId);
        }
    }

    updateConnectionWithWaypoints(connectionId) {
        const connection = this.connections.get(connectionId);
        const customWaypoints = connection?.customWaypoints;
        if (!customWaypoints) return;

        const pathD = this.buildPathFromWaypoints(customWaypoints);
        connection.path.setAttribute("d", pathD);
        connection.hitArea.setAttribute("d", pathD);

        this.updateLabel(connection);
    }

    buildPathFromWaypoints(waypoints) {
        if (!waypoints || waypoints.length < 2) return "";

        const cornerRadius = 12;
        const segments = [];
        segments.push(`M ${waypoints[0].x} ${waypoints[0].y}`);

        for (let i = 1; i < waypoints.length; i++) {
            const prev = waypoints[i - 1];
            const curr = waypoints[i];
            const next = waypoints[i + 1];

            if (next && i < waypoints.length - 1) {
                const maxRadiusPrev = Math.min(
                    Math.abs(curr.x - prev.x) / 2,
                    Math.abs(curr.y - prev.y) / 2,
                );
                const maxRadiusNext = Math.min(
                    Math.abs(next.x - curr.x) / 2,
                    Math.abs(next.y - curr.y) / 2,
                );
                const radius = Math.min(
                    cornerRadius,
                    Math.max(maxRadiusPrev, 1),
                    Math.max(maxRadiusNext, 1),
                );

                if (radius > 0) {
                    const isHorizontalToCurr =
                        Math.abs(curr.x - prev.x) > Math.abs(curr.y - prev.y);

                    if (isHorizontalToCurr) {
                        const dirX = Math.sign(curr.x - prev.x);
                        const dirY = Math.sign(next.y - curr.y);

                        segments.push(
                            `L ${curr.x - radius * dirX} ${curr.y}`,
                            `Q ${curr.x} ${curr.y}, ${curr.x} ${curr.y + radius * dirY}`,
                        );
                    } else {
                        const dirY = Math.sign(curr.y - prev.y);
                        const dirX = Math.sign(next.x - curr.x);

                        segments.push(
                            `L ${curr.x} ${curr.y - radius * dirY}`,
                            `Q ${curr.x} ${curr.y}, ${curr.x + radius * dirX} ${curr.y}`,
                        );
                    }
                } else {
                    segments.push(`L ${curr.x} ${curr.y}`);
                }
            } else {
                segments.push(`L ${curr.x} ${curr.y}`);
            }
        }

        return segments.join(" ");
    }

    setCustomWaypoints(connectionId, waypoints) {
        const connection = this.connections.get(connectionId);
        if (!connection) return;

        connection.customWaypoints = waypoints;
        if (waypoints) {
            this.updateConnectionWithWaypoints(connectionId);
        }
    }

    getCustomWaypoints(connectionId) {
        const connection = this.connections.get(connectionId);
        return connection?.customWaypoints || null;
    }

    destroy() {
        if (this.manualPathCreator) {
            this.manualPathCreator.destroy();
        }
        this.clearAllConnections();
    }
}
