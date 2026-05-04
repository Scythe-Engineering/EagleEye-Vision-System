/**
 * FlowchartMinimap - Navigation minimap component for flowchart canvas
 */

export class FlowchartMinimap {
    constructor(canvas, options = {}) {
        this.canvas = canvas;
        this.width = options.width || 180;
        this.height = options.height || 120;
        this.padding = options.padding || 10;
        this.backgroundColor = options.backgroundColor || "#1a1a1a";
        this.borderColor = options.borderColor || "#404040";
        this.nodeColor = options.nodeColor || "#f9c845";
        this.connectionColor = options.connectionColor || "#f9c845";
        this.viewportColor = options.viewportColor || "rgba(249, 200, 69, 0.3)";
        this.viewportBorderColor = options.viewportBorderColor || "#f9c845";

        this.element = null;
        this.canvasElement = null;
        this.ctx = null;
        this.viewportRect = null;
        this.closeButton = null;
        this.showButton = null;

        this.isDragging = false;
        this.nodes = [];
        this.connections = [];
        this.worldBounds = { minX: 0, minY: 0, maxX: 1000, maxY: 1000 };
        this.isVisible = true;

        // Cache minimap container rect
        this.minimapRect = { width: 0, height: 0, left: 0, top: 0 };

        this.viewWidth = this.width;
        this.viewHeight = this.height;

        this.init();
    }

    init() {
        this.element = document.createElement("div");
        this.element.id = "flowchartMinimap";
        this.element.style.cssText = `
            position: absolute;
            bottom: ${this.padding}px;
            right: ${this.padding}px;
            width: ${this.width}px;
            height: ${this.height}px;
            z-index: 100;
            transition: all 0.3s ease-in-out;
            transform: scale(1);
            opacity: 1;
            overflow: visible;
            pointer-events: none;
        `;

        // Inner container that handles clipping and background
        this.contentContainer = document.createElement("div");
        this.contentContainer.style.cssText = `
            width: 100%;
            height: 100%;
            background-color: ${this.backgroundColor};
            border: 2px solid ${this.borderColor};
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
            cursor: pointer;
            position: relative;
            pointer-events: auto;
        `;

        this.canvasElement = document.createElement("canvas");
        this.canvasElement.style.cssText = `
            display: block;
            width: 100%;
            height: 100%;
        `;

        this.ctx = this.canvasElement.getContext("2d");
        this.syncCanvasResolution();

        this.viewportRect = document.createElement("div");
        this.viewportRect.style.cssText = `
            position: absolute;
            background-color: ${this.viewportColor};
            border: 1px solid ${this.viewportBorderColor};
            border-radius: 4px;
            pointer-events: none;
        `;

        // Create close button
        this.closeButton = document.createElement("button");
        this.closeButton.innerHTML = "×";
        this.closeButton.style.cssText = `
            position: absolute;
            top: -12px;
            left: -12px;
            width: 26px;
            height: 26px;
            background-color: #f9c845;
            color: #1a1a1a;
            border: 2px solid #404040;
            border-radius: 50%;
            font-size: 20px;
            line-height: 1;
            font-weight: bold;
            cursor: pointer;
            z-index: 102;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s ease;
            box-shadow: 0 2px 8px rgba(0,0,0,0.5);
            padding: 0;
            margin: 0;
            pointer-events: auto;
        `;
        this.closeButton.title = "Hide minimap";

        // Create show button (initially hidden)
        this.showButton = document.createElement("button");
        this.showButton.innerHTML = `
            <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7"></path>
            </svg>
        `;
        this.showButton.style.cssText = `
            position: absolute;
            bottom: ${this.padding}px;
            right: ${this.padding}px;
            width: 36px;
            height: 36px;
            background-color: #f9c845;
            color: #1a1a1a;
            border: 2px solid #404040;
            border-radius: 50%;
            cursor: pointer;
            z-index: 100;
            display: none;
            align-items: center;
            justify-content: center;
            transition: all 0.3s ease-in-out;
            transform: scale(0);
            opacity: 0;
            flex-direction: column;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
            pointer-events: auto;
        `;
        this.showButton.title = "Show minimap";
        this.showButton.addEventListener("click", (e) => {
            e.stopPropagation();
            this.show();
        });

        this.contentContainer.appendChild(this.canvasElement);
        this.contentContainer.appendChild(this.viewportRect);

        this.element.appendChild(this.contentContainer);
        this.element.appendChild(this.closeButton);
        this.element.appendChild(this.showButton);

        this.setupEventListeners();
        this.updateMinimapRect();
    }

    updateMinimapRect() {
        const rect = this.canvasElement.getBoundingClientRect();
        this.minimapRect = {
            width: rect.width,
            height: rect.height,
            left: rect.left,
            top: rect.top
        };
    }

    syncCanvasResolution() {
        const canvasElement = this.canvasElement;
        const devicePixelRatio = window.devicePixelRatio || 1;
        const cssWidth = canvasElement.clientWidth || this.width;
        const cssHeight = canvasElement.clientHeight || this.height;
        const bufferWidth = Math.max(1, Math.round(cssWidth * devicePixelRatio));
        const bufferHeight = Math.max(1, Math.round(cssHeight * devicePixelRatio));

        canvasElement.width = bufferWidth;
        canvasElement.height = bufferHeight;
        this.ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);

        this.viewWidth = cssWidth;
        this.viewHeight = cssHeight;
    }

    setupEventListeners() {
        const resizeObserver = new ResizeObserver(() => {
            this.syncCanvasResolution();
            this.updateMinimapRect();
            this.render();
        });
        resizeObserver.observe(this.canvasElement);

        this.contentContainer.addEventListener(
            "mousedown",
            this.handleMouseDown.bind(this),
        );
        this.contentContainer.addEventListener(
            "mousemove",
            this.handleMouseMove.bind(this),
        );
        this.contentContainer.addEventListener(
            "mouseup",
            this.handleMouseUp.bind(this),
        );
        this.contentContainer.addEventListener(
            "mouseleave",
            this.handleMouseUp.bind(this),
        );

        // Close button logic
        this.closeButton.addEventListener("mousedown", (e) => {
            e.stopPropagation();
            e.preventDefault();
        });
        this.closeButton.addEventListener("click", (e) => {
            e.stopPropagation();
            e.preventDefault();
            this.hide();
        });
    }

    handleMouseDown(event) {
        this.isDragging = true;
        this.navigateToPosition(event);
        event.preventDefault();
        event.stopPropagation();
    }

    handleMouseMove(event) {
        if (!this.isDragging) return;
        this.navigateToPosition(event);
        event.preventDefault();
    }

    handleMouseUp() {
        this.isDragging = false;
    }

    navigateToPosition(event) {
        const rect = this.minimapRect;
        const clickX = event.clientX - rect.left;
        const clickY = event.clientY - rect.top;

        const { scale, offsetX, offsetY } = this.calculateTransform();
        const worldX =
            this.worldBounds.minX + (clickX - offsetX) / scale;
        const worldY =
            this.worldBounds.minY + (clickY - offsetY) / scale;

        const containerRect = this.canvas.containerRect;
        const viewportState = this.canvas.getViewportState();

        const newTranslateX =
            containerRect.width / 2 - worldX * viewportState.scale;
        const newTranslateY =
            containerRect.height / 2 - worldY * viewportState.scale;

        this.canvas.setViewportState({
            scale: viewportState.scale,
            translateX: newTranslateX,
            translateY: newTranslateY,
        });
    }

    calculateTransform() {
        const viewWidth = this.viewWidth;
        const viewHeight = this.viewHeight;
        const worldWidth = this.worldBounds.maxX - this.worldBounds.minX;
        const worldHeight = this.worldBounds.maxY - this.worldBounds.minY;

        const scaleX = viewWidth / worldWidth;
        const scaleY = viewHeight / worldHeight;
        const scale = Math.min(scaleX, scaleY);

        const drawnWidth = worldWidth * scale;
        const drawnHeight = worldHeight * scale;
        const offsetX = (viewWidth - drawnWidth) / 2;
        const offsetY = (viewHeight - drawnHeight) / 2;

        return { scale, offsetX, offsetY };
    }

    updateNodes(nodeDataList) {
        this.nodes = nodeDataList.map((node) => ({
            id: node.instanceId,
            x: node.position?.x || 0,
            y: node.position?.y || 0,
            width: node.width || 200,
            height: node.height || 80,
        }));

        this.calculateWorldBounds();
        this.render();
    }

    updateConnections(connectionsData) {
        this.connections = connectionsData;
        this.render();
    }

    calculateWorldBounds() {
        if (this.nodes.length === 0) {
            this.worldBounds = { minX: -500, minY: -400, maxX: 500, maxY: 400 };
            return;
        }

        const padding = 150;

        let minX = Infinity,
            minY = Infinity;
        let maxX = -Infinity,
            maxY = -Infinity;

        this.nodes.forEach((node) => {
            minX = Math.min(minX, node.x);
            minY = Math.min(minY, node.y);
            maxX = Math.max(maxX, node.x + node.width);
            maxY = Math.max(maxY, node.y + node.height);
        });

        // Calculate the center of all nodes
        const centerX = (minX + maxX) / 2;
        const centerY = (minY + maxY) / 2;

        // Calculate the size needed to contain all nodes with padding
        const width = Math.max(maxX - minX + padding * 2, 800);
        const height = Math.max(maxY - minY + padding * 2, 600);

        // Center the bounds on the nodes
        this.worldBounds = {
            minX: centerX - width / 2,
            minY: centerY - height / 2,
            maxX: centerX + width / 2,
            maxY: centerY + height / 2,
        };
    }

    render() {
        const vw = this.viewWidth;
        const vh = this.viewHeight;
        this.ctx.clearRect(0, 0, vw, vh);

        this.ctx.fillStyle = this.backgroundColor;
        this.ctx.fillRect(0, 0, vw, vh);

        this.renderGrid();
        this.renderConnections();
        this.renderNodes();
        this.updateViewportRect();
    }

    renderGrid() {
        const { scale, offsetX, offsetY } = this.calculateTransform();
        const gridSpacing = 100;

        this.ctx.strokeStyle = "rgba(64, 64, 64, 0.3)";
        this.ctx.lineWidth = 0.5;

        const startX =
            Math.floor(this.worldBounds.minX / gridSpacing) * gridSpacing;
        const startY =
            Math.floor(this.worldBounds.minY / gridSpacing) * gridSpacing;

        const vw = this.viewWidth;
        const vh = this.viewHeight;

        for (let x = startX; x <= this.worldBounds.maxX; x += gridSpacing) {
            const screenX =
                offsetX + (x - this.worldBounds.minX) * scale;
            this.ctx.beginPath();
            this.ctx.moveTo(screenX, 0);
            this.ctx.lineTo(screenX, vh);
            this.ctx.stroke();
        }

        for (let y = startY; y <= this.worldBounds.maxY; y += gridSpacing) {
            const screenY =
                offsetY + (y - this.worldBounds.minY) * scale;
            this.ctx.beginPath();
            this.ctx.moveTo(0, screenY);
            this.ctx.lineTo(vw, screenY);
            this.ctx.stroke();
        }
    }

    renderNodes() {
        const { scale, offsetX, offsetY } = this.calculateTransform();

        this.ctx.fillStyle = this.nodeColor;
        this.ctx.shadowColor = "rgba(249, 200, 69, 0.3)";
        this.ctx.shadowBlur = 2;

        this.nodes.forEach((node) => {
            const x =
                offsetX + (node.x - this.worldBounds.minX) * scale;
            const y =
                offsetY + (node.y - this.worldBounds.minY) * scale;
            const w = Math.max(4, node.width * scale);
            const h = Math.max(3, node.height * scale);

            this.ctx.beginPath();
            this.ctx.roundRect(x, y, w, h, 2);
            this.ctx.fill();
        });

        this.ctx.shadowColor = "transparent";
        this.ctx.shadowBlur = 0;
    }

    renderConnections() {
        const { scale, offsetX, offsetY } = this.calculateTransform();

        this.ctx.strokeStyle = this.connectionColor;
        this.ctx.lineWidth = 1.5;
        this.ctx.lineCap = "round";

        this.connections.forEach((conn) => {
            const fromNode = this.nodes.find((n) => n.id === conn.fromNodeId);
            const toNode = this.nodes.find((n) => n.id === conn.toNodeId);

            if (fromNode && toNode) {
                const fromX =
                    offsetX +
                    (fromNode.x +
                        fromNode.width / 2 -
                        this.worldBounds.minX) *
                        scale;
                const fromY =
                    offsetY +
                    (fromNode.y +
                        fromNode.height / 2 -
                        this.worldBounds.minY) *
                        scale;
                const toX =
                    offsetX +
                    (toNode.x + toNode.width / 2 - this.worldBounds.minX) *
                        scale;
                const toY =
                    offsetY +
                    (toNode.y + toNode.height / 2 - this.worldBounds.minY) *
                        scale;

                this.ctx.beginPath();
                this.ctx.moveTo(fromX, fromY);
                this.ctx.lineTo(toX, toY);
                this.ctx.stroke();
            }
        });
    }

    updateViewportRect() {
        const viewportState = this.canvas.getViewportState();
        const containerRect = this.canvas.containerRect;
        const { scale, offsetX, offsetY } = this.calculateTransform();

        const worldViewLeft = -viewportState.translateX / viewportState.scale;
        const worldViewTop = -viewportState.translateY / viewportState.scale;
        const worldViewWidth = containerRect.width / viewportState.scale;
        const worldViewHeight = containerRect.height / viewportState.scale;

        const rectX =
            offsetX + (worldViewLeft - this.worldBounds.minX) * scale;
        const rectY =
            offsetY + (worldViewTop - this.worldBounds.minY) * scale;
        const rectWidth = worldViewWidth * scale;
        const rectHeight = worldViewHeight * scale;

        const visualPadding = 5;
        const borderWidth = 1;
        const positionPadding = visualPadding + borderWidth;
        const sizePadding = visualPadding + borderWidth + 3;

        const vw = this.viewWidth;
        const vh = this.viewHeight;

        const clampedRectX = Math.max(positionPadding, rectX);
        const clampedRectY = Math.max(positionPadding, rectY);

        this.viewportRect.style.left = `${clampedRectX}px`;
        this.viewportRect.style.top = `${clampedRectY}px`;
        this.viewportRect.style.width = `${Math.min(vw - sizePadding - clampedRectX, rectWidth)}px`;
        this.viewportRect.style.height = `${Math.min(vh - sizePadding - clampedRectY, rectHeight)}px`;
    }

    onViewportChange(viewportState) {
        this.updateViewportRect();
    }

    attachTo(container) {
        container.appendChild(this.element);
        container.appendChild(this.showButton);
        this.syncCanvasResolution();
        this.updateMinimapRect();
        this.render();
    }

    show() {
        this.isVisible = true;
        this.contentContainer.style.display = "block";
        this.closeButton.style.display = "block";
        this.element.style.opacity = "1";

        // Hide show button with scale animation
        this.showButton.style.opacity = "0";
        this.showButton.style.transform = "scale(0)";
        setTimeout(() => {
            this.showButton.style.display = "none";
        }, 300);
    }

    hide() {
        this.isVisible = false;
        this.element.style.opacity = "0";

        // Show the show button after fade out completes
        setTimeout(() => {
            this.contentContainer.style.display = "none";
            this.closeButton.style.display = "none";
            this.showButton.style.display = "flex";
            this.showButton.style.transform = "scale(1)";
            // Trigger smooth fade-in by setting opacity after display change
            setTimeout(() => {
                this.showButton.style.opacity = "1";
            }, 10);
        }, 300);
    }

    toggle() {
        if (this.isVisible) {
            this.hide();
        } else {
            this.show();
        }
    }

    setSize(width, height) {
        this.width = width;
        this.height = height;
        this.element.style.width = `${width}px`;
        this.element.style.height = `${height}px`;
        this.syncCanvasResolution();
        this.render();
    }

    destroy() {
        this.element?.remove();
        this.showButton?.remove();
    }
}
