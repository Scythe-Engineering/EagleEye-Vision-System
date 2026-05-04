/**
 * Flowchart canvas viewport and interaction helpers.
 */

import { InteractiveGrid } from "./interactiveGrid.js";

export class FlowchartCanvas {
    /**
     * Create a new flowchart canvas bound to a container element.
     *
     * @param {HTMLElement} containerElement - Host container for the canvas.
     * @param {Object} [options={}] - Initialization options.
     */
    constructor(containerElement, options = {}) {
        this.container = containerElement;
        this.gridSpacing = options.gridSpacing || 20;

        this.gridLayer = null;
        this.islandLayer = null;
        this.connectionsLayer = null;
        this.nodesLayer = null;

        this.onViewportChange = options.onViewportChange || (() => {});
        this.interactiveGrid = null;

        // Cache container dimensions to avoid getBoundingClientRect during zoom/pan
        this.containerRect = { width: 0, height: 0, left: 0, top: 0 };

        this.init();
    }

    /**
     * Initialize the DOM layers and interaction handlers.
     */
    init() {
        this.container.innerHTML = "";
        this.container.style.position = "relative";
        this.container.style.overflow = "hidden";
        this.container.style.width = "100%";
        this.container.style.height = "100%";
        this.container.style.backgroundColor = "#1a1a1a";
        this.container.style.cursor = "grab";
        this.container.style.borderRadius = "0 0 15px 15px";

        this.updateContainerRect();

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

        // Create island background layer behind connections and nodes.
        this.islandLayer = document.createElement("div");
        this.islandLayer.id = "flowchartIslands";
        this.islandLayer.style.position = "absolute";
        this.islandLayer.style.top = "0";
        this.islandLayer.style.left = "0";
        this.islandLayer.style.width = "100%";
        this.islandLayer.style.height = "100%";
        this.islandLayer.style.overflow = "visible";
        this.islandLayer.style.pointerEvents = "none";
        this.islandLayer.style.zIndex = "0";

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

        this.viewport.appendChild(this.islandLayer);
        this.viewport.appendChild(this.connectionsLayer);
        this.viewport.appendChild(this.nodesLayer);
        this.container.appendChild(this.viewport);

        this.updateTransform();
        this.setupPanZoom();
    }

    /**
     * Refresh the cached container bounds.
     */
    updateContainerRect() {
        const rect = this.container.getBoundingClientRect();
        this.containerRect = {
            width: rect.width,
            height: rect.height,
            left: rect.left,
            top: rect.top,
        };
    }

    /**
     * Set up panning, zooming, and view reset interactions.
     */
    setupPanZoom() {
        let isPanning = false;
        let startX, startY;
        this.placeholderVisible = false;

        const resizeObserver = new ResizeObserver(() => {
            this.updateContainerRect();
        });
        resizeObserver.observe(this.container);

        this.container.addEventListener("mousedown", (e) => {
            if (e.button !== 0) return;
            if (this.placeholderVisible) return;

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

        globalThis.addEventListener("mousemove", (e) => {
            if (!isPanning) return;

            this.translateX = e.clientX - startX;
            this.translateY = e.clientY - startY;
            this.updateTransform();
        });

        globalThis.addEventListener("mouseup", () => {
            if (isPanning) {
                isPanning = false;
                this.container.style.cursor = "grab";
            }
        });

        this.container.addEventListener(
            "wheel",
            (e) => {
                if (this.placeholderVisible) return;

                const rect = this.containerRect;
                if (
                    e.clientX < rect.left ||
                    e.clientX > rect.left + rect.width ||
                    e.clientY < rect.top ||
                    e.clientY > rect.top + rect.height
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

    /**
     * Apply the current viewport transform and notify listeners.
     */
    updateTransform() {
        this.viewport.style.transform = `translate(${this.translateX}px, ${this.translateY}px) scale(${this.scale})`;
        this.updateGrid();
        this.onViewportChange(this.getViewportState());
    }

    /**
     * Synchronize the interactive grid with the current viewport.
     */
    updateGrid() {
        // Update the interactive grid with current viewport state
        if (this.interactiveGrid) {
            this.interactiveGrid.updateViewport(
                this.translateX,
                this.translateY,
                this.scale,
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

    /**
     * Convert screen coordinates to world coordinates.
     *
     * @param {number} screenX - Screen X coordinate.
     * @param {number} screenY - Screen Y coordinate.
     * @returns {{x: number, y: number}} World coordinates.
     */
    screenToWorld(screenX, screenY) {
        const rect = this.containerRect;

        const worldX = (screenX - rect.left - this.translateX) / this.scale;
        const worldY = (screenY - rect.top - this.translateY) / this.scale;

        return { x: worldX, y: worldY };
    }

    /**
     * Snap a 2D position to the grid.
     *
     * @param {number} x - X coordinate.
     * @param {number} y - Y coordinate.
     * @returns {{x: number, y: number}} Snapped coordinates.
     */
    snapPositionToGrid(x, y) {
        return {
            x: Math.round(x / this.gridSpacing) * this.gridSpacing,
            y: Math.round(y / this.gridSpacing) * this.gridSpacing,
        };
    }

    /**
     * Snap a scalar value to the grid.
     *
     * @param {number} value - Value to snap.
     * @returns {number} Snapped value.
     */
    snapToGrid(value) {
        return Math.round(value / this.gridSpacing) * this.gridSpacing;
    }

    /**
     * Get the nodes layer element.
     *
     * @returns {HTMLDivElement} Nodes layer element.
     */
    getNodesLayer() {
        return this.nodesLayer;
    }

    /**
     * Get the connections layer element.
     *
     * @returns {SVGSVGElement} Connections layer element.
     */
    getConnectionsLayer() {
        return this.connectionsLayer;
    }

    /**
     * Get the island layer element.
     *
     * @returns {HTMLDivElement} Island layer element.
     */
    getIslandLayer() {
        return this.islandLayer;
    }

    /**
     * Get the current viewport state.
     *
     * @returns {{scale: number, translateX: number, translateY: number}} Viewport state.
     */
    getViewportState() {
        return {
            scale: this.scale,
            translateX: this.translateX,
            translateY: this.translateY,
        };
    }

    /**
     * Set the viewport state.
     *
     * @param {{scale: number, translateX: number, translateY: number}} state - Viewport state.
     */
    setViewportState(state) {
        this.scale = state.scale;
        this.translateX = state.translateX;
        this.translateY = state.translateY;
        this.updateTransform();
    }

    /**
     * Toggle placeholder visibility for interaction gating.
     *
     * @param {boolean} isVisible - Whether the placeholder is visible.
     */
    setPlaceholderVisible(isVisible) {
        this.placeholderVisible = isVisible;
    }

    /**
     * Fit the viewport to the content.
     */
    fitToContent() {
        this.resetViewport();
    }

    /**
     * Reset the current view.
     */
    resetView() {
        this.resetViewport();
    }

    /**
     * Reset the viewport transform to its default state.
     */
    resetViewport() {
        this.translateX = 0;
        this.translateY = 0;
        this.scale = 1;
        this.updateTransform();
    }
}
