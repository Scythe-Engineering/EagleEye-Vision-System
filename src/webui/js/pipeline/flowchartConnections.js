/**
 * FlowchartConnections - SVG bezier curve connection management with data type labels
 */

export class FlowchartConnections {
    constructor(svgLayer, options = {}) {
        this.svgLayer = svgLayer;
        this.connections = new Map();
        this.connectionColor = options.connectionColor || "#f9c845";
        this.connectionWidth = options.connectionWidth || 2;
        this.labelFontSize = options.labelFontSize || 10;
        this.curveTension = options.curveTension || 0.5;
        this.onConnectionRemoved = options.onConnectionRemoved || (() => {});

        this.setupDefs();
    }

    setupDefs() {
        let defs = this.svgLayer.querySelector("defs");
        if (!defs) {
            defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
            this.svgLayer.insertBefore(defs, this.svgLayer.firstChild);
        }
        
        const markerId = "flowchart-arrow";
        if (!defs.querySelector(`#${markerId}`)) {
            const marker = document.createElementNS("http://www.w3.org/2000/svg", "marker");
            marker.setAttribute("id", markerId);
            marker.setAttribute("viewBox", "0 0 10 10");
            marker.setAttribute("refX", "8");
            marker.setAttribute("refY", "5");
            marker.setAttribute("markerWidth", "6");
            marker.setAttribute("markerHeight", "6");
            marker.setAttribute("orient", "auto-start-reverse");
            
            const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
            path.setAttribute("d", "M 0 0 L 10 5 L 0 10 z");
            path.setAttribute("fill", this.connectionColor);
            
            marker.appendChild(path);
            defs.appendChild(marker);
        }
        
    }

    createConnection(connectionId, fromNode, fromPortName, toNode, toPortName, dataType) {
        if (this.connections.has(connectionId)) {
            this.updateConnection(connectionId, fromNode, fromPortName, toNode, toPortName);
            return;
        }
        
        const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
        group.setAttribute("data-connection-id", connectionId);
        group.setAttribute("pointer-events", "visibleStroke");
        
        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
        path.setAttribute("fill", "none");
        path.setAttribute("stroke", this.connectionColor);
        path.setAttribute("stroke-width", this.connectionWidth.toString());
        path.setAttribute("stroke-linecap", "round");
        path.setAttribute("marker-end", "url(#flowchart-arrow)");
        path.style.transition = "stroke 0.15s ease, stroke-width 0.15s ease, d 0.3s ease";
        path.setAttribute("pointer-events", "none");
        
        const hitArea = document.createElementNS("http://www.w3.org/2000/svg", "path");
        hitArea.setAttribute("fill", "none");
        hitArea.setAttribute("stroke", "transparent");
        hitArea.setAttribute("stroke-width", "20");
        hitArea.style.cursor = "pointer";
        hitArea.setAttribute("pointer-events", "auto");
        
        const labelGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
        labelGroup.setAttribute("class", "connection-label");
        labelGroup.style.opacity = "0";
        labelGroup.style.transition = "opacity 0.3s ease";
        
        const labelBackground = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        labelBackground.setAttribute("fill", "#1f1f1f");
        labelBackground.setAttribute("stroke", "#404040");
        labelBackground.setAttribute("stroke-width", "1");
        labelBackground.setAttribute("rx", "4");
        labelBackground.setAttribute("ry", "4");
        
        const labelText = document.createElementNS("http://www.w3.org/2000/svg", "text");
        labelText.setAttribute("fill", "#a0a0a0");
        labelText.setAttribute("font-size", this.labelFontSize.toString());
        labelText.setAttribute("font-family", "system-ui, -apple-system, sans-serif");
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
            isAnimating: false
        });
        
        this.updateConnection(connectionId, fromNode, fromPortName, toNode, toPortName);
        
        // Fade in label after initial position is set
        requestAnimationFrame(() => {
            labelGroup.style.opacity = "1";
        });
    }

    setupHoverEffects(group, path) {
        // Add smooth transition for all animatable properties
        path.style.transition = "stroke 0.15s ease, stroke-width 0.15s ease, d 0.3s ease, filter 0.15s ease, opacity 0.15s ease";
        
        group.addEventListener("mouseenter", () => {
            path.setAttribute("stroke-width", (this.connectionWidth + 2).toString());
            path.style.filter = "drop-shadow(0 0 4px " + this.connectionColor + ")";
        });

        group.addEventListener("mouseleave", () => {
            path.setAttribute("stroke-width", this.connectionWidth.toString());
            path.style.filter = "none";
        });

        group.addEventListener("click", (e) => {
            e.stopPropagation();
            const connectionId = group.getAttribute("data-connection-id");
            if (connectionId) {
                // Fade out before removal
                path.style.opacity = "0";
                setTimeout(() => {
                    this.removeConnection(connectionId);
                    this.onConnectionRemoved(connectionId);
                }, 150);
            }
        });
    }

    updateConnection(connectionId, fromNode, fromPortName, toNode, toPortName) {
        const connection = this.connections.get(connectionId);
        if (!connection) return;
        
        const fromPos = fromNode.getOutputPortPosition(fromPortName);
        const toPos = toNode.getInputPortPosition(toPortName);
        
        if (!fromPos || !toPos) return;
        
        const pathD = this.calculateBezierPath(fromPos, toPos);
        connection.path.setAttribute("d", pathD);
        connection.hitArea.setAttribute("d", pathD);
        
        // Start animation loop for label if not already running
        if (!connection.isAnimating) {
            connection.isAnimating = true;
            const startTime = performance.now();
            const duration = 300; // Match CSS transition duration
            
            const animate = (currentTime) => {
                const elapsed = currentTime - startTime;
                this.updateLabel(connection);
                
                if (elapsed < duration) {
                    requestAnimationFrame(animate);
                } else {
                    connection.isAnimating = false;
                    // Final update to ensure precision
                    this.updateLabel(connection);
                }
            };
            requestAnimationFrame(animate);
        }
    }

    calculateBezierPath(from, to) {
        const dx = to.x - from.x;
        const dy = to.y - from.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        const controlPointOffset = Math.max(50, Math.min(150, distance * this.curveTension));
        
        // Start slightly away from the beginning node and end slightly before the end node
        const startOffset = 10; // pixels to offset from start
        const endOffset = 10; // pixels to offset from end
        
        const startX = from.x + startOffset;
        const startY = from.y;
        const endX = to.x - endOffset;
        const endY = to.y;
        
        const cp1x = startX + controlPointOffset;
        const cp1y = startY;
        const cp2x = endX - controlPointOffset;
        const cp2y = endY;
        
        return `M ${startX} ${startY} C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${endX} ${endY}`;
    }

    updateLabel(connection) {
        const path = connection.path;
        const labelGroup = connection.labelGroup;
        const text = connection.labelText;
        const background = connection.labelBackground;
        
        // Get the point at the middle of the path (50% along the length)
        const pathLength = path.getTotalLength();
        const midPoint = path.getPointAtLength(pathLength / 2);
        
        const midX = midPoint.x;
        const midY = midPoint.y;
        
        const textBBox = text.getBBox();
        const padding = 6;
        
        const bgWidth = textBBox.width + padding * 2;
        const bgHeight = textBBox.height + padding;
        
        // Use transform on the group to move everything together
        labelGroup.setAttribute("transform", `translate(${midX}, ${midY})`);
        
        // Position background and text relative to the group center (0,0)
        background.setAttribute("x", (-bgWidth / 2).toString());
        background.setAttribute("y", (-bgHeight / 2).toString());
        background.setAttribute("width", bgWidth.toString());
        background.setAttribute("height", bgHeight.toString());
        
        text.setAttribute("x", "0");
        text.setAttribute("y", "0");
    }

    updateAllConnections(nodeMap) {
        for (const [connectionId, connection] of this.connections) {
            const fromNode = nodeMap.get(connection.fromNodeId);
            const toNode = nodeMap.get(connection.toNodeId);
            
            if (fromNode && toNode) {
                this.updateConnection(connectionId, fromNode, connection.fromPortName, toNode, connection.toPortName);
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
            if (connection.fromNodeId === nodeInstanceId || connection.toNodeId === nodeInstanceId) {
                toRemove.push(connectionId);
            }
        }
        
        toRemove.forEach(id => this.removeConnection(id));
    }

    clearAllConnections() {
        for (const [connectionId] of this.connections) {
            this.removeConnection(connectionId);
        }
        this.connections.clear();
    }

    highlightConnection(connectionId, highlight = true) {
        const connection = this.connections.get(connectionId);
        if (!connection) return;
        
        if (highlight) {
            connection.path.setAttribute("stroke", "#ffffff");
            connection.path.setAttribute("stroke-width", (this.connectionWidth + 2).toString());
            connection.path.style.filter = "drop-shadow(0 0 4px #ffffff)";
        } else {
            connection.path.setAttribute("stroke", this.connectionColor);
            connection.path.setAttribute("stroke-width", this.connectionWidth.toString());
            connection.path.style.filter = "none";
        }
    }

    getConnectionsForPort(nodeInstanceId, portName, portType) {
        const result = [];
        
        for (const [connectionId, connection] of this.connections) {
            if (portType === "output" && 
                connection.fromNodeId === nodeInstanceId && 
                connection.fromPortName === portName) {
                result.push(connectionId);
            } else if (portType === "input" && 
                       connection.toNodeId === nodeInstanceId && 
                       connection.toPortName === portName) {
                result.push(connectionId);
            }
        }
        
        return result;
    }

    createTemporaryConnection(startPos, options = {}) {
        const tempPath = document.createElementNS("http://www.w3.org/2000/svg", "path");
        tempPath.setAttribute("fill", "none");
        tempPath.setAttribute("stroke", this.connectionColor);
        
        // Check if we're morphing from a hovered connection
        const isFromHover = options.fromHover || false;
        
        if (isFromHover) {
            // Start with hovered state (thicker, solid line)
            tempPath.setAttribute("stroke-width", (this.connectionWidth + 2).toString());
            tempPath.setAttribute("stroke-dasharray", "0");
            tempPath.setAttribute("opacity", "1");
            tempPath.style.filter = "drop-shadow(0 0 4px " + this.connectionColor + ")";
        } else {
            // Start invisible for new connections
            tempPath.setAttribute("stroke-width", (this.connectionWidth + 2).toString());
            tempPath.setAttribute("stroke-dasharray", "5,5");
            tempPath.setAttribute("opacity", "0");
        }
        
        tempPath.id = "temp-connection";
        
        // Add transition for morphing effect
        tempPath.style.transition = "opacity 0.2s ease, stroke-width 0.2s ease, stroke-dasharray 0.2s ease, filter 0.2s ease";
        
        // Render immediately at start position
        const initialEndPos = { x: startPos.x + 1, y: startPos.y };
        const initialPathD = this.calculateBezierPath(startPos, initialEndPos);
        tempPath.setAttribute("d", initialPathD);
        
        this.svgLayer.appendChild(tempPath);
        
        // Trigger transition to dragging state
        requestAnimationFrame(() => {
            if (isFromHover) {
                // Morph from hover state to drag state
                tempPath.setAttribute("stroke-dasharray", "5,5");
                tempPath.setAttribute("stroke-width", this.connectionWidth.toString());
                tempPath.setAttribute("opacity", "0.7");
                tempPath.style.filter = "none";
            } else {
                // Fade in for new connections
                tempPath.setAttribute("opacity", "0.7");
                tempPath.setAttribute("stroke-width", this.connectionWidth.toString());
            }
        });
        
        return {
            update: (endPos) => {
                const pathD = this.calculateBezierPath(startPos, endPos);
                tempPath.setAttribute("d", pathD);
            },
            remove: () => {
                tempPath.setAttribute("opacity", "0");
                setTimeout(() => tempPath.remove(), 200);
            }
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
                dataType: connection.dataType
            });
        }
        return data;
    }

    destroy() {
        this.clearAllConnections();
    }
}
