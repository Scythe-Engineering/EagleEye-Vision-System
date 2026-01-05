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
        
        const filterId = "connection-glow";
        if (!defs.querySelector(`#${filterId}`)) {
            const filter = document.createElementNS("http://www.w3.org/2000/svg", "filter");
            filter.setAttribute("id", filterId);
            filter.setAttribute("x", "-50%");
            filter.setAttribute("y", "-50%");
            filter.setAttribute("width", "200%");
            filter.setAttribute("height", "200%");
            
            const feGaussianBlur = document.createElementNS("http://www.w3.org/2000/svg", "feGaussianBlur");
            feGaussianBlur.setAttribute("stdDeviation", "2");
            feGaussianBlur.setAttribute("result", "coloredBlur");
            
            const feMerge = document.createElementNS("http://www.w3.org/2000/svg", "feMerge");
            const feMergeNode1 = document.createElementNS("http://www.w3.org/2000/svg", "feMergeNode");
            feMergeNode1.setAttribute("in", "coloredBlur");
            const feMergeNode2 = document.createElementNS("http://www.w3.org/2000/svg", "feMergeNode");
            feMergeNode2.setAttribute("in", "SourceGraphic");
            feMerge.appendChild(feMergeNode1);
            feMerge.appendChild(feMergeNode2);
            
            filter.appendChild(feGaussianBlur);
            filter.appendChild(feMerge);
            defs.appendChild(filter);
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
        path.style.transition = "stroke 0.15s ease, stroke-width 0.15s ease";
        path.setAttribute("pointer-events", "none");
        
        const hitArea = document.createElementNS("http://www.w3.org/2000/svg", "path");
        hitArea.setAttribute("fill", "none");
        hitArea.setAttribute("stroke", "transparent");
        hitArea.setAttribute("stroke-width", "20");
        hitArea.style.cursor = "pointer";
        hitArea.setAttribute("pointer-events", "stroke");
        
        const labelGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
        labelGroup.setAttribute("class", "connection-label");
        
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
            dataType
        });
        
        this.updateConnection(connectionId, fromNode, fromPortName, toNode, toPortName);
    }

    setupHoverEffects(group, path) {
        group.addEventListener("mouseenter", () => {
            path.setAttribute("stroke-width", (this.connectionWidth + 2).toString());
            path.setAttribute("filter", "url(#connection-glow)");
        });

        group.addEventListener("mouseleave", () => {
            path.setAttribute("stroke-width", this.connectionWidth.toString());
            path.removeAttribute("filter");
        });

        group.addEventListener("click", (e) => {
            e.stopPropagation();
            const connectionId = group.getAttribute("data-connection-id");
            if (connectionId) {
                this.removeConnection(connectionId);
                this.onConnectionRemoved(connectionId);
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
        
        this.updateLabel(connection, fromPos, toPos);
    }

    calculateBezierPath(from, to) {
        const dx = to.x - from.x;
        const dy = to.y - from.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        const controlPointOffset = Math.max(50, Math.min(150, distance * this.curveTension));
        
        const cp1x = from.x + controlPointOffset;
        const cp1y = from.y;
        const cp2x = to.x - controlPointOffset;
        const cp2y = to.y;
        
        return `M ${from.x} ${from.y} C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${to.x} ${to.y}`;
    }

    updateLabel(connection, fromPos, toPos) {
        const midX = (fromPos.x + toPos.x) / 2;
        const midY = (fromPos.y + toPos.y) / 2;
        
        const text = connection.labelText;
        const textBBox = text.getBBox();
        const padding = 6;
        
        const bgWidth = textBBox.width + padding * 2;
        const bgHeight = textBBox.height + padding;
        
        connection.labelBackground.setAttribute("x", (midX - bgWidth / 2).toString());
        connection.labelBackground.setAttribute("y", (midY - bgHeight / 2).toString());
        connection.labelBackground.setAttribute("width", bgWidth.toString());
        connection.labelBackground.setAttribute("height", bgHeight.toString());
        
        text.setAttribute("x", midX.toString());
        text.setAttribute("y", midY.toString());
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
            connection.path.setAttribute("filter", "url(#connection-glow)");
        } else {
            connection.path.setAttribute("stroke", this.connectionColor);
            connection.path.setAttribute("stroke-width", this.connectionWidth.toString());
            connection.path.removeAttribute("filter");
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

    createTemporaryConnection(startPos) {
        const tempPath = document.createElementNS("http://www.w3.org/2000/svg", "path");
        tempPath.setAttribute("fill", "none");
        tempPath.setAttribute("stroke", this.connectionColor);
        tempPath.setAttribute("stroke-width", this.connectionWidth.toString());
        tempPath.setAttribute("stroke-dasharray", "5,5");
        tempPath.setAttribute("opacity", "0.7");
        tempPath.id = "temp-connection";
        
        // Render immediately at start position (short line to make it visible)
        const initialEndPos = { x: startPos.x + 50, y: startPos.y };
        const initialPathD = this.calculateBezierPath(startPos, initialEndPos);
        tempPath.setAttribute("d", initialPathD);
        
        this.svgLayer.appendChild(tempPath);
        
        return {
            update: (endPos) => {
                const pathD = this.calculateBezierPath(startPos, endPos);
                tempPath.setAttribute("d", pathD);
            },
            remove: () => {
                tempPath.remove();
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
