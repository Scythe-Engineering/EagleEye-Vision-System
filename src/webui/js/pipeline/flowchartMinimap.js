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
        this.viewportColor = options.viewportColor || "rgba(249, 200, 69, 0.3)";
        this.viewportBorderColor = options.viewportBorderColor || "#f9c845";
        
        this.element = null;
        this.canvasElement = null;
        this.ctx = null;
        this.viewportRect = null;
        
        this.isDragging = false;
        this.nodes = [];
        this.worldBounds = { minX: 0, minY: 0, maxX: 1000, maxY: 1000 };
        
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
            background-color: ${this.backgroundColor};
            border: 2px solid ${this.borderColor};
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
            z-index: 100;
            cursor: pointer;
        `;
        
        this.canvasElement = document.createElement("canvas");
        this.canvasElement.width = this.width;
        this.canvasElement.height = this.height;
        this.canvasElement.style.cssText = `
            width: 100%;
            height: 100%;
        `;
        
        this.ctx = this.canvasElement.getContext("2d");
        
        this.viewportRect = document.createElement("div");
        this.viewportRect.style.cssText = `
            position: absolute;
            background-color: ${this.viewportColor};
            border: 1px solid ${this.viewportBorderColor};
            pointer-events: none;
        `;
        
        this.element.appendChild(this.canvasElement);
        this.element.appendChild(this.viewportRect);
        
        this.setupEventListeners();
    }

    setupEventListeners() {
        this.element.addEventListener("mousedown", this.handleMouseDown.bind(this));
        this.element.addEventListener("mousemove", this.handleMouseMove.bind(this));
        this.element.addEventListener("mouseup", this.handleMouseUp.bind(this));
        this.element.addEventListener("mouseleave", this.handleMouseUp.bind(this));
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
        const rect = this.element.getBoundingClientRect();
        const clickX = event.clientX - rect.left;
        const clickY = event.clientY - rect.top;
        
        const scale = this.calculateScale();
        const worldX = this.worldBounds.minX + clickX / scale;
        const worldY = this.worldBounds.minY + clickY / scale;
        
        const containerRect = this.canvas.container.getBoundingClientRect();
        const viewportState = this.canvas.getViewportState();
        
        const newTranslateX = containerRect.width / 2 - worldX * viewportState.scale;
        const newTranslateY = containerRect.height / 2 - worldY * viewportState.scale;
        
        this.canvas.setViewportState({
            translateX: newTranslateX,
            translateY: newTranslateY
        });
    }

    calculateScale() {
        const worldWidth = this.worldBounds.maxX - this.worldBounds.minX;
        const worldHeight = this.worldBounds.maxY - this.worldBounds.minY;
        
        const scaleX = this.width / worldWidth;
        const scaleY = this.height / worldHeight;
        
        return Math.min(scaleX, scaleY);
    }

    updateNodes(nodeDataList) {
        this.nodes = nodeDataList.map(node => ({
            x: node.position?.x || 0,
            y: node.position?.y || 0,
            width: node.width || 200,
            height: node.height || 80
        }));
        
        this.calculateWorldBounds();
        this.render();
    }

    calculateWorldBounds() {
        if (this.nodes.length === 0) {
            this.worldBounds = { minX: 0, minY: 0, maxX: 1000, maxY: 800 };
            return;
        }
        
        const padding = 100;
        
        let minX = Infinity, minY = Infinity;
        let maxX = -Infinity, maxY = -Infinity;
        
        this.nodes.forEach(node => {
            minX = Math.min(minX, node.x);
            minY = Math.min(minY, node.y);
            maxX = Math.max(maxX, node.x + node.width);
            maxY = Math.max(maxY, node.y + node.height);
        });
        
        minX = Math.min(0, minX) - padding;
        minY = Math.min(0, minY) - padding;
        maxX = Math.max(1000, maxX) + padding;
        maxY = Math.max(800, maxY) + padding;
        
        this.worldBounds = { minX, minY, maxX, maxY };
    }

    render() {
        this.ctx.clearRect(0, 0, this.width, this.height);
        
        this.ctx.fillStyle = this.backgroundColor;
        this.ctx.fillRect(0, 0, this.width, this.height);
        
        this.renderGrid();
        this.renderNodes();
        this.updateViewportRect();
    }

    renderGrid() {
        const scale = this.calculateScale();
        const gridSpacing = 100;
        
        this.ctx.strokeStyle = "rgba(64, 64, 64, 0.3)";
        this.ctx.lineWidth = 0.5;
        
        const startX = Math.floor(this.worldBounds.minX / gridSpacing) * gridSpacing;
        const startY = Math.floor(this.worldBounds.minY / gridSpacing) * gridSpacing;
        
        for (let x = startX; x <= this.worldBounds.maxX; x += gridSpacing) {
            const screenX = (x - this.worldBounds.minX) * scale;
            this.ctx.beginPath();
            this.ctx.moveTo(screenX, 0);
            this.ctx.lineTo(screenX, this.height);
            this.ctx.stroke();
        }
        
        for (let y = startY; y <= this.worldBounds.maxY; y += gridSpacing) {
            const screenY = (y - this.worldBounds.minY) * scale;
            this.ctx.beginPath();
            this.ctx.moveTo(0, screenY);
            this.ctx.lineTo(this.width, screenY);
            this.ctx.stroke();
        }
    }

    renderNodes() {
        const scale = this.calculateScale();
        
        this.ctx.fillStyle = this.nodeColor;
        this.ctx.shadowColor = "rgba(249, 200, 69, 0.3)";
        this.ctx.shadowBlur = 2;
        
        this.nodes.forEach(node => {
            const x = (node.x - this.worldBounds.minX) * scale;
            const y = (node.y - this.worldBounds.minY) * scale;
            const w = Math.max(4, node.width * scale);
            const h = Math.max(3, node.height * scale);
            
            this.ctx.beginPath();
            this.ctx.roundRect(x, y, w, h, 2);
            this.ctx.fill();
        });
        
        this.ctx.shadowColor = "transparent";
        this.ctx.shadowBlur = 0;
    }

    updateViewportRect() {
        const viewportState = this.canvas.getViewportState();
        const containerRect = this.canvas.container.getBoundingClientRect();
        const scale = this.calculateScale();
        
        const worldViewLeft = -viewportState.translateX / viewportState.scale;
        const worldViewTop = -viewportState.translateY / viewportState.scale;
        const worldViewWidth = containerRect.width / viewportState.scale;
        const worldViewHeight = containerRect.height / viewportState.scale;
        
        const rectX = (worldViewLeft - this.worldBounds.minX) * scale;
        const rectY = (worldViewTop - this.worldBounds.minY) * scale;
        const rectWidth = worldViewWidth * scale;
        const rectHeight = worldViewHeight * scale;
        
        this.viewportRect.style.left = `${Math.max(0, rectX)}px`;
        this.viewportRect.style.top = `${Math.max(0, rectY)}px`;
        this.viewportRect.style.width = `${Math.min(this.width - rectX, rectWidth)}px`;
        this.viewportRect.style.height = `${Math.min(this.height - rectY, rectHeight)}px`;
    }

    onViewportChange(viewportState) {
        this.updateViewportRect();
    }

    attachTo(container) {
        container.appendChild(this.element);
    }

    show() {
        this.element.style.display = "block";
    }

    hide() {
        this.element.style.display = "none";
    }

    toggle() {
        if (this.element.style.display === "none") {
            this.show();
        } else {
            this.hide();
        }
    }

    setSize(width, height) {
        this.width = width;
        this.height = height;
        this.element.style.width = `${width}px`;
        this.element.style.height = `${height}px`;
        this.canvasElement.width = width;
        this.canvasElement.height = height;
        this.render();
    }

    destroy() {
        if (this.element && this.element.parentElement) {
            this.element.parentElement.removeChild(this.element);
        }
    }
}
