/**
 * FlowchartCanvas - Simplified canvas with just a grid for dropping nodes
 */

import { InteractiveGrid } from "./interactiveGrid.js";

export class FlowchartCanvas {
    constructor(containerElement, options = {}) {
        this.container = containerElement;
        this.gridSpacing = options.gridSpacing || 20;

        this.gridLayer = null;
        this.connectionsLayer = null;
        this.nodesLayer = null;

        this.onViewportChange = options.onViewportChange || (() => {});
        this.interactiveGrid = null;

        this.init();
    }

    init() {
        this.container.innerHTML = "";
        this.container.style.position = "relative";
        this.container.style.overflow = "hidden";
        this.container.style.width = "100%";
        this.container.style.height = "100%";
        this.container.style.backgroundColor = "#1a1a1a";
        this.container.style.cursor = "grab";
        this.container.style.borderRadius = "0 0 15px 15px";

        // Create interactive grid (canvas-based dynamic grid)
        this.interactiveGrid = new InteractiveGrid(this.container, {
            gridSpacing: this.gridSpacing,
        });

        // Create viewport for scaling and panning
        this.viewport = document.createElement("div");
        this.viewport.id = "flowchartViewport";
        this.viewport.style.position = "absolute";
        this.viewport.style.top = "0";
        this.viewport.style.left = "0";
        this.viewport.style.width = "100%";
        this.viewport.style.height = "100%";
        this.viewport.style.transformOrigin = "0 0";
        this.viewport.style.borderRadius = "inherit";
        this.viewport.style.pointerEvents = "auto";

        this.scale = 1;
        this.translateX = 0;
        this.translateY = 0;

        // Create connections layer (SVG)
        this.connectionsLayer = document.createElementNS(
            "http://www.w3.org/2000/svg",
            "svg",
        );
        this.connectionsLayer.id = "flowchartConnections";
        this.connectionsLayer.style.position = "absolute";
        this.connectionsLayer.style.top = "0";
        this.connectionsLayer.style.left = "0";
        this.connectionsLayer.style.width = "100%";
        this.connectionsLayer.style.height = "100%";
        this.connectionsLayer.style.overflow = "visible";
        this.connectionsLayer.style.pointerEvents = "none";
        this.connectionsLayer.style.zIndex = "1";

        // Create nodes layer
        // pointer-events: none on the layer allows clicks to pass through to connections below
        // Individual node elements have pointer-events: auto so they remain interactive
        this.nodesLayer = document.createElement("div");
        this.nodesLayer.id = "flowchartNodes";
        this.nodesLayer.style.position = "absolute";
        this.nodesLayer.style.top = "0";
        this.nodesLayer.style.left = "0";
        this.nodesLayer.style.width = "100%";
        this.nodesLayer.style.height = "100%";
        this.nodesLayer.style.overflow = "visible";
        this.nodesLayer.style.pointerEvents = "none";
        this.nodesLayer.style.zIndex = "2";

        this.viewport.appendChild(this.connectionsLayer);
        this.viewport.appendChild(this.nodesLayer);
        this.container.appendChild(this.viewport);

        this.updateTransform();
        this.setupPanZoom();
    }

    setupPanZoom() {
        let isPanning = false;
        let startX, startY;

        this.container.addEventListener("mousedown", (e) => {
            if (e.button !== 0) return;

            const isNode = e.target.closest(".flowchart-node");
            const isButton = e.target.closest("button");
            const isPort = e.target.closest(".port-connector");

            if (!isNode && !isButton && !isPort) {
                isPanning = true;
                this.container.style.cursor = "grabbing";
                startX = e.clientX - this.translateX;
                startY = e.clientY - this.translateY;
                e.preventDefault();
            }
        });

        window.addEventListener("mousemove", (e) => {
            if (!isPanning) return;

            this.translateX = e.clientX - startX;
            this.translateY = e.clientY - startY;
            this.updateTransform();
        });

        window.addEventListener("mouseup", () => {
            if (isPanning) {
                isPanning = false;
                this.container.style.cursor = "grab";
            }
        });

        this.container.addEventListener(
            "wheel",
            (e) => {
                const rect = this.container.getBoundingClientRect();
                if (
                    e.clientX < rect.left ||
                    e.clientX > rect.right ||
                    e.clientY < rect.top ||
                    e.clientY > rect.bottom
                ) {
                    return;
                }

                e.preventDefault();

                const delta = -e.deltaY;
                const zoomFactor = Math.pow(1.1, delta / 100);

                const mouseX = e.clientX - rect.left;
                const mouseY = e.clientY - rect.top;

                const worldX = (mouseX - this.translateX) / this.scale;
                const worldY = (mouseY - this.translateY) / this.scale;

                const newScale = Math.min(
                    Math.max(0.1, this.scale * zoomFactor),
                    3,
                );

                this.translateX = mouseX - worldX * newScale;
                this.translateY = mouseY - worldY * newScale;
                this.scale = newScale;

                this.updateTransform();
            },
            { passive: false },
        );

        this.container.addEventListener("dblclick", (e) => {
            const isNode = e.target.closest(".flowchart-node");
            if (!isNode) {
                this.resetView();
            }
        });
    }

    updateTransform() {
        this.viewport.style.transform = `translate(${this.translateX}px, ${this.translateY}px) scale(${this.scale})`;
        this.updateGrid();
        this.onViewportChange(this.getViewportState());
    }

    updateGrid() {
        // Update the interactive grid with current viewport state
        if (this.interactiveGrid) {
            this.interactiveGrid.updateViewport(
                this.translateX,
                this.translateY,
                this.scale
            );
        }
    }

    /**
     * Update operation positions on the grid for proximity effects
     */
    updateOperationPositions(nodes) {
        if (this.interactiveGrid) {
            this.interactiveGrid.updateOperationPositions(nodes);
        }
    }

    /**
     * Get the interactive grid instance
     */
    getInteractiveGrid() {
        return this.interactiveGrid;
    }

    screenToWorld(screenX, screenY) {
        const rect = this.container.getBoundingClientRect();

        const worldX = (screenX - rect.left - this.translateX) / this.scale;
        const worldY = (screenY - rect.top - this.translateY) / this.scale;

        return { x: worldX, y: worldY };
    }

    snapPositionToGrid(x, y) {
        return {
            x: Math.round(x / this.gridSpacing) * this.gridSpacing,
            y: Math.round(y / this.gridSpacing) * this.gridSpacing,
        };
    }

    snapToGrid(value) {
        return Math.round(value / this.gridSpacing) * this.gridSpacing;
    }

    getNodesLayer() {
        return this.nodesLayer;
    }

    getConnectionsLayer() {
        return this.connectionsLayer;
    }

    getViewportState() {
        return {
            scale: this.scale,
            translateX: this.translateX,
            translateY: this.translateY,
        };
    }

    setViewportState(state) {
        this.scale = state.scale;
        this.translateX = state.translateX;
        this.translateY = state.translateY;
        this.updateTransform();
    }

    fitToContent() {
        // Simple fit to content implementation
        this.translateX = 0;
        this.translateY = 0;
        this.scale = 1;
        this.updateTransform();
    }

    resetView() {
        this.translateX = 0;
        this.translateY = 0;
        this.scale = 1;
        this.updateTransform();
    }
}
