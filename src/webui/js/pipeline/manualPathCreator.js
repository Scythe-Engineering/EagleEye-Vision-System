export class ManualPathCreator {
    constructor(svgLayer, options = {}) {
        this.svgLayer = svgLayer;
        this.canvas = options.canvas;
        this.connectionColor = options.connectionColor || "#f9c845";
        this.previewColor = options.previewColor || "#f9c845";
        this.gridSpacing = options.gridSpacing || 20;
        this.cornerRadius = options.cornerRadius || 12;
        this.onComplete = options.onComplete || (() => {});
        this.onCancel = options.onCancel || (() => {});

        this.isActive = false;
        this.connectionId = null;
        this.startPoint = null;
        this.endPoint = null;
        this.waypoints = [];
        this.nextDirection = "horizontal";

        this.previewGroup = null;
        this.confirmedPath = null;
        this.previewLine = null;
        this.waypointMarkers = [];
        this.overlay = null;
        this.instructionTooltip = null;

        this.boundHandleMouseMove = this.handleMouseMove.bind(this);
        this.boundHandleClick = this.handleClick.bind(this);
        this.boundHandleKeyDown = this.handleKeyDown.bind(this);
        this.boundHandleContextMenu = this.handleContextMenu.bind(this);
        this.boundHandleWheel = this.handleWheel.bind(this);
        this.boundHandleMouseDown = this.handleMouseDown.bind(this);
        this.boundHandleMouseUp = this.handleMouseUp.bind(this);
        this.boundHandlePanMove = this.handlePanMove.bind(this);

        this.isPanning = false;
        this.panStartX = 0;
        this.panStartY = 0;
    }

    start(connectionId, startPoint, endPoint) {
        if (this.isActive) {
            this.cancel();
        }

        this.isActive = true;
        this.connectionId = connectionId;
        this.startPoint = { ...startPoint };
        this.endPoint = { ...endPoint };
        this.waypoints = [{ ...startPoint }];
        this.nextDirection = "horizontal";

        this.createOverlay();
        this.createPreviewElements();
        this.createInstructionTooltip();
        this.attachEventListeners();
        this.updatePreview(startPoint);
    }

    createOverlay() {
        this.overlay = document.createElement("div");
        this.overlay.id = "manual-path-overlay";
        Object.assign(this.overlay.style, {
            position: "fixed",
            top: "0",
            left: "0",
            width: "100%",
            height: "100%",
            zIndex: "9999",
            cursor: "crosshair",
            backgroundColor: "transparent",
        });
        document.body.appendChild(this.overlay);
    }

    createPreviewElements() {
        this.previewGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
        this.previewGroup.setAttribute("id", "manual-path-preview");
        this.previewGroup.style.pointerEvents = "none";

        this.confirmedPath = document.createElementNS("http://www.w3.org/2000/svg", "path");
        this.confirmedPath.setAttribute("fill", "none");
        this.confirmedPath.setAttribute("stroke", this.connectionColor);
        this.confirmedPath.setAttribute("stroke-width", "2");
        this.confirmedPath.setAttribute("stroke-linecap", "round");
        this.confirmedPath.style.filter = "drop-shadow(0 0 4px " + this.connectionColor + ")";

        this.previewLine = document.createElementNS("http://www.w3.org/2000/svg", "path");
        this.previewLine.setAttribute("fill", "none");
        this.previewLine.setAttribute("stroke", this.previewColor);
        this.previewLine.setAttribute("stroke-width", "2");
        this.previewLine.setAttribute("stroke-dasharray", "8,4");
        this.previewLine.setAttribute("stroke-linecap", "round");
        this.previewLine.style.opacity = "0.8";

        this.startMarker = this.createPointMarker(this.startPoint, "#4ade80");
        this.endMarker = this.createPointMarker(this.endPoint, "#f87171");

        this.previewGroup.appendChild(this.confirmedPath);
        this.previewGroup.appendChild(this.previewLine);
        this.previewGroup.appendChild(this.startMarker);
        this.previewGroup.appendChild(this.endMarker);

        this.svgLayer.appendChild(this.previewGroup);
    }

    createPointMarker(point, color) {
        const marker = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        marker.setAttribute("cx", point.x);
        marker.setAttribute("cy", point.y);
        marker.setAttribute("r", "6");
        marker.setAttribute("fill", color);
        marker.setAttribute("stroke", "#1a1a1a");
        marker.setAttribute("stroke-width", "2");
        marker.style.filter = `drop-shadow(0 0 4px ${color})`;
        return marker;
    }

    createWaypointMarker(point) {
        const marker = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        marker.setAttribute("cx", point.x);
        marker.setAttribute("cy", point.y);
        marker.setAttribute("r", "4");
        marker.setAttribute("fill", this.connectionColor);
        marker.setAttribute("stroke", "#1a1a1a");
        marker.setAttribute("stroke-width", "1.5");
        this.previewGroup.appendChild(marker);
        this.waypointMarkers.push(marker);
        return marker;
    }

    createInstructionTooltip() {
        this.instructionTooltip = document.createElement("div");
        this.instructionTooltip.id = "manual-path-tooltip";
        Object.assign(this.instructionTooltip.style, {
            position: "fixed",
            bottom: "20px",
            left: "50%",
            transform: "translateX(-50%)",
            backgroundColor: "#2a2a2a",
            border: "1px solid #404040",
            borderRadius: "8px",
            padding: "12px 20px",
            zIndex: "10001",
            boxShadow: "0 4px 12px rgba(0,0,0,0.5)",
            color: "#ffffff",
            fontSize: "14px",
            fontFamily: "system-ui, -apple-system, sans-serif",
            textAlign: "center",
            maxWidth: "500px",
        });
        this.updateTooltipText();
        document.body.appendChild(this.instructionTooltip);
    }

    updateTooltipText() {
        if (!this.instructionTooltip) return;

        const direction = this.nextDirection === "horizontal" ? "horizontal" : "vertical";
        const waypointCount = Math.max(0, this.waypoints.length - 1);

        this.instructionTooltip.innerHTML = `
            <div style="margin-bottom: 6px;">
                <span style="color: #f9c845; font-weight: 600;">Creating Manual Path</span>
            </div>
            <div style="color: #a0a0a0; font-size: 12px;">
                Click to add <span style="color: ${this.connectionColor}; font-weight: 500;">${direction}</span> corner point
                (${waypointCount} corner${waypointCount === 1 ? '' : 's'} added)
                <br>
                <span style="color: #888;">Click near the <span style="color: #f87171;">red endpoint</span> to finish • Press <kbd style="background: #404040; padding: 2px 6px; border-radius: 3px; margin: 0 2px;">Esc</kbd> to cancel</span>
                <br>
                <span style="color: #888;">Scroll to zoom • Middle mouse to pan</span>
            </div>
        `;
    }

    attachEventListeners() {
        this.overlay.addEventListener("mousemove", this.boundHandleMouseMove);
        this.overlay.addEventListener("click", this.boundHandleClick);
        this.overlay.addEventListener("contextmenu", this.boundHandleContextMenu);
        this.overlay.addEventListener("wheel", this.boundHandleWheel, { passive: false });
        this.overlay.addEventListener("mousedown", this.boundHandleMouseDown);
        globalThis.addEventListener("keydown", this.boundHandleKeyDown);
    }

    detachEventListeners() {
        if (this.overlay) {
            this.overlay.removeEventListener("mousemove", this.boundHandleMouseMove);
            this.overlay.removeEventListener("click", this.boundHandleClick);
            this.overlay.removeEventListener("contextmenu", this.boundHandleContextMenu);
            this.overlay.removeEventListener("wheel", this.boundHandleWheel);
            this.overlay.removeEventListener("mousedown", this.boundHandleMouseDown);
        }
        globalThis.removeEventListener("keydown", this.boundHandleKeyDown);
        globalThis.removeEventListener("mousemove", this.boundHandlePanMove);
        globalThis.removeEventListener("mouseup", this.boundHandleMouseUp);
    }

    handleMouseMove(e) {
        if (!this.isActive || !this.canvas || this.isPanning) return;

        const worldPos = this.canvas.screenToWorld(e.clientX, e.clientY);
        const snappedPos = this.canvas.snapPositionToGrid(worldPos.x, worldPos.y);
        this.updatePreview(snappedPos);
    }

    handleClick(e) {
        if (!this.isActive || !this.canvas) return;

        e.preventDefault();
        e.stopPropagation();

        const worldPos = this.canvas.screenToWorld(e.clientX, e.clientY);
        const snappedPos = this.canvas.snapPositionToGrid(worldPos.x, worldPos.y);

        const distToEnd = Math.sqrt(
            Math.pow(snappedPos.x - this.endPoint.x, 2) +
            Math.pow(snappedPos.y - this.endPoint.y, 2)
        );

        if (distToEnd < this.gridSpacing * 2) {
            this.complete();
            return;
        }

        this.addWaypoint(snappedPos);
    }

    handleKeyDown(e) {
        if (!this.isActive) return;

        if (e.key === "Escape") {
            e.preventDefault();
            this.cancel();
        } else if (e.key === "Enter") {
            e.preventDefault();
            this.complete();
        } else if (e.key === "z" && (e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            this.undoLastWaypoint();
        }
    }

    handleContextMenu(e) {
        e.preventDefault();
        e.stopPropagation();
        this.cancel();
    }

    handleWheel(e) {
        if (!this.canvas) return;

        e.preventDefault();
        e.stopPropagation();

        const canvasContainer = this.canvas.container;
        const wheelEvent = new WheelEvent("wheel", {
            deltaY: e.deltaY,
            deltaX: e.deltaX,
            deltaMode: e.deltaMode,
            clientX: e.clientX,
            clientY: e.clientY,
            bubbles: true,
            cancelable: true,
        });

        canvasContainer.dispatchEvent(wheelEvent);
    }

    handleMouseDown(e) {
        if (e.button === 1) {
            e.preventDefault();
            e.stopPropagation();
            this.isPanning = true;
            this.panStartX = e.clientX;
            this.panStartY = e.clientY;
            this.overlay.style.cursor = "grabbing";

            globalThis.addEventListener("mousemove", this.boundHandlePanMove);
            globalThis.addEventListener("mouseup", this.boundHandleMouseUp);
        }
    }

    handlePanMove(e) {
        if (!this.isPanning || !this.canvas) return;

        e.preventDefault();
        e.stopPropagation();

        const deltaX = e.clientX - this.panStartX;
        const deltaY = e.clientY - this.panStartY;

        this.canvas.translateX += deltaX;
        this.canvas.translateY += deltaY;
        this.canvas.updateTransform();

        this.panStartX = e.clientX;
        this.panStartY = e.clientY;
    }

    handleMouseUp(e) {
        if (e.button === 1 && this.isPanning) {
            e.preventDefault();
            e.stopPropagation();
            this.isPanning = false;
            this.overlay.style.cursor = "crosshair";

            globalThis.removeEventListener("mousemove", this.boundHandlePanMove);
            globalThis.removeEventListener("mouseup", this.boundHandleMouseUp);
        }
    }

    addWaypoint(snappedPos) {
        const lastPoint = this.waypoints.at(-1);
        let constrainedPoint;

        if (this.nextDirection === "horizontal") {
            constrainedPoint = { x: snappedPos.x, y: lastPoint.y };
        } else {
            constrainedPoint = { x: lastPoint.x, y: snappedPos.y };
        }

        if (constrainedPoint.x === lastPoint.x && constrainedPoint.y === lastPoint.y) {
            return;
        }

        this.waypoints.push(constrainedPoint);
        this.createWaypointMarker(constrainedPoint);

        this.nextDirection = this.nextDirection === "horizontal" ? "vertical" : "horizontal";

        this.updateConfirmedPath();
        this.updateTooltipText();
    }

    undoLastWaypoint() {
        if (this.waypoints.length <= 1) return;

        this.waypoints.pop();

        const lastMarker = this.waypointMarkers.pop();
        if (lastMarker) {
            lastMarker.remove();
        }

        this.nextDirection = this.nextDirection === "horizontal" ? "vertical" : "horizontal";

        this.updateConfirmedPath();
        this.updateTooltipText();
    }

    updatePreview(cursorPos) {
        if (!this.isActive) return;

        const lastPoint = this.waypoints.at(-1);
        let constrainedPos;

        if (this.nextDirection === "horizontal") {
            constrainedPos = { x: cursorPos.x, y: lastPoint.y };
        } else {
            constrainedPos = { x: lastPoint.x, y: cursorPos.y };
        }

        const pathD = `M ${lastPoint.x} ${lastPoint.y} L ${constrainedPos.x} ${constrainedPos.y}`;
        this.previewLine.setAttribute("d", pathD);
    }

    updateConfirmedPath() {
        if (this.waypoints.length < 2) {
            this.confirmedPath.setAttribute("d", "");
            return;
        }

        const pathD = this.buildPathFromWaypoints(this.waypoints, false);
        this.confirmedPath.setAttribute("d", pathD);
    }

    buildPathFromWaypoints(waypoints, includeEndPoint = true) {
        if (waypoints.length < 2) return "";

        const points = [...waypoints];
        if (includeEndPoint) {
            points.push(this.endPoint);
        }

        const segments = [];
        segments.push(`M ${points[0].x} ${points[0].y}`);

        for (let i = 1; i < points.length; i++) {
            const prev = points[i - 1];
            const curr = points[i];
            const next = points[i + 1];

            if (next && i < points.length - 1) {
                const radius = Math.min(
                    this.cornerRadius,
                    Math.abs(curr.x - prev.x) / 2,
                    Math.abs(curr.y - prev.y) / 2,
                    Math.abs(next.x - curr.x) / 2,
                    Math.abs(next.y - curr.y) / 2
                );

                if (radius > 0) {
                    const isHorizontalFirst = Math.abs(curr.x - prev.x) > Math.abs(curr.y - prev.y);

                    if (isHorizontalFirst) {
                        const dirX = Math.sign(curr.x - prev.x);
                        const dirY = Math.sign(next.y - curr.y);

                        segments.push(
                            `L ${curr.x - radius * dirX} ${curr.y}`,
                            `Q ${curr.x} ${curr.y}, ${curr.x} ${curr.y + radius * dirY}`
                        );
                    } else {
                        const dirY = Math.sign(curr.y - prev.y);
                        const dirX = Math.sign(next.x - curr.x);

                        segments.push(
                            `L ${curr.x} ${curr.y - radius * dirY}`,
                            `Q ${curr.x} ${curr.y}, ${curr.x + radius * dirX} ${curr.y}`
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

    complete() {
        if (!this.isActive) return;

        const finalWaypoints = this.buildFinalWaypoints();

        this.onComplete(this.connectionId, finalWaypoints);

        this.cleanup();
    }

    buildFinalWaypoints() {
        const points = [...this.waypoints];

        const lastWaypoint = points.at(-1);

        const needsHorizontal = lastWaypoint.y !== this.endPoint.y;
        const needsVertical = lastWaypoint.x !== this.endPoint.x;

        if (needsHorizontal && needsVertical) {
            if (this.nextDirection === "horizontal") {
                points.push({ x: this.endPoint.x, y: lastWaypoint.y });
            } else {
                points.push({ x: lastWaypoint.x, y: this.endPoint.y });
            }
        }

        points.push({ ...this.endPoint });

        return points;
    }

    cancel() {
        if (!this.isActive) return;

        this.onCancel(this.connectionId);
        this.cleanup();
    }

    cleanup() {
        this.isActive = false;
        this.isPanning = false;
        this.detachEventListeners();

        if (this.overlay) {
            this.overlay.remove();
            this.overlay = null;
        }

        if (this.previewGroup) {
            this.previewGroup.remove();
            this.previewGroup = null;
        }

        if (this.instructionTooltip) {
            this.instructionTooltip.remove();
            this.instructionTooltip = null;
        }

        this.confirmedPath = null;
        this.previewLine = null;
        this.startMarker = null;
        this.endMarker = null;
        this.waypointMarkers = [];
        this.waypoints = [];
        this.connectionId = null;
        this.startPoint = null;
        this.endPoint = null;
    }

    destroy() {
        this.cleanup();
    }
}
