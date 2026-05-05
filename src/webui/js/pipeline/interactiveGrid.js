/**
 * InteractiveGrid - Dynamic grid that responds to cursor and operation proximity
 * Works with FlowchartCanvas transform-based viewport
 */
// Manages the interactive background grid rendering and proximity effects.
export class InteractiveGrid {
    /**
     * Create an interactive grid for a container element.
     * @param {HTMLElement} container
     * @param {Object} [options]
     */
    constructor(container, options = {}) {
        this.container = container;
        this.canvas = null;
        this.ctx = null;

        // Mouse position in screen space
        this.mouseScreenX = -1000;
        this.mouseScreenY = -1000;

        // Grid settings
        this.gridSpacing = options.gridSpacing || 20;
        this.baseDotSize = 1.5;
        this.maxDotSize = 4;
        this.cursorInfluenceRadius = 150;

        // Cross grid settings (for areas far from operations)
        this.baseCrossSpacing = 200;
        this.crossLineLength = 10; // Half-width of the cross arms
        this.crossLineWidth = 1;
        this.crossOpacity = 0.3;

        // Operation-related properties
        this.operationPositions = [];
        this.focusArea = null; // Alternative focal point (e.g., for placeholder)
        this.operationInfluenceRadius = 200;
        this.operationFadeDistance = 600;
        this.operationMaxDotSize = 3;
        this.focusAreaInfluenceRadius = 150; // Influence radius for placeholder
        this.focusAreaFadeDistance = 300; // Fade distance for placeholder

        // Caching for performance
        this.distanceCache = new Map();
        this.cacheResolution = 20; // Cache distance every 20 world units
        this.dotCache = new Map();
        this.containerRect = { width: 0, height: 0, left: 0, top: 0 };

        // Viewport transform state
        this.scale = 1;
        this.translateX = 0;
        this.translateY = 0;

        // Base opacity settings
        this.baseOpacity = 0.25;
        this.maxOpacity = 0.7;

        this.animationFrame = null;
        this.redrawTimer = null;
        this.lastRedrawRequestTime = 0;
        this.redrawThrottleMs = options.redrawThrottleMs || 33;
        this.needsRedraw = false;
        this.init();
    }

    /**
     * Set up the canvas and begin rendering.
     */
    init() {
        this.canvas = document.createElement("canvas");
        this.canvas.style.position = "absolute";
        this.canvas.style.top = "0";
        this.canvas.style.left = "0";
        this.canvas.style.width = "100%";
        this.canvas.style.height = "100%";
        this.canvas.style.pointerEvents = "none";
        this.canvas.style.zIndex = "0";

        this.container.insertBefore(this.canvas, this.container.firstChild);

        this.ctx = this.canvas.getContext("2d");

        this.updateContainerRect();
        this.resize();
        this.setupEventListeners();
        this.startAnimation();
    }

    /**
     * Refresh cached container bounds.
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
     * Resize the canvas to match the container.
     */
    resize() {
        const rect = this.containerRect;
        const dpr = window.devicePixelRatio || 1;

        this.canvas.width = rect.width * dpr;
        this.canvas.height = rect.height * dpr;
        this.canvas.style.width = `${rect.width}px`;
        this.canvas.style.height = `${rect.height}px`;

        this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        this.requestRedraw();
    }

    /**
     * Attach pointer and resize listeners.
     */
    setupEventListeners() {
        this.container.addEventListener("mousemove", (e) => {
            const rect = this.containerRect;
            this.mouseScreenX = e.clientX - rect.left;
            this.mouseScreenY = e.clientY - rect.top;
            this.requestRedraw();
        });

        this.container.addEventListener("mouseleave", () => {
            this.mouseScreenX = -1000;
            this.mouseScreenY = -1000;
            this.requestRedraw();
        });

        const resizeObserver = new ResizeObserver(() => {
            this.updateContainerRect();
            this.resize();
        });
        resizeObserver.observe(this.container);
    }

    /**
     * Update the viewport transform state from FlowchartCanvas
     */
    updateViewport(translateX, translateY, scale) {
        if (
            this.translateX !== translateX ||
            this.translateY !== translateY ||
            this.scale !== scale
        ) {
            this.translateX = translateX;
            this.translateY = translateY;
            this.scale = scale;
            this.requestRedraw();
        }
    }

    /**
     * Update operation positions (in world coordinates)
     */
    updateOperationPositions(nodes) {
        this.operationPositions = nodes.map((node) => ({
            x: node.position.x,
            y: node.position.y,
            width: 200,
            height: 80,
        }));
        this.distanceCache.clear();
        this.dotCache.clear();
        this.requestRedraw();
    }

    /**
     * Set a focus area (e.g., placeholder) for dot proximity effects
     */
    setFocusArea(area) {
        this.focusArea = area;
        this.distanceCache.clear();
        this.dotCache.clear();
        this.requestRedraw();
    }

    /**
     * Clear the focus area
     */
    clearFocusArea() {
        this.focusArea = null;
        this.distanceCache.clear();
        this.dotCache.clear();
        this.requestRedraw();
    }

    /**
     * Convert screen coordinates to world coordinates
     */
    screenToWorld(screenX, screenY) {
        return {
            x: (screenX - this.translateX) / this.scale,
            y: (screenY - this.translateY) / this.scale,
        };
    }

    /**
     * Convert world coordinates to screen coordinates
     */
    worldToScreen(worldX, worldY) {
        return {
            x: worldX * this.scale + this.translateX,
            y: worldY * this.scale + this.translateY,
        };
    }

    /**
     * Get distance from a world point to the nearest operation rectangle
     */
    getDistanceToNearestOperation(worldX, worldY) {
        if (this.operationPositions.length === 0) {
            return Infinity;
        }

        // Create cache key based on rounded coordinates
        const cacheKey = `${Math.round(worldX / this.cacheResolution)},${Math.round(worldY / this.cacheResolution)}`;

        if (this.distanceCache.has(cacheKey)) {
            return this.distanceCache.get(cacheKey);
        }

        let minDistance = Infinity;

        // Check distance to operations
        for (const op of this.operationPositions) {
            // Calculate distance to the rectangle's edge
            const closestX = Math.max(op.x, Math.min(worldX, op.x + op.width));
            const closestY = Math.max(op.y, Math.min(worldY, op.y + op.height));
            const dx = worldX - closestX;
            const dy = worldY - closestY;
            const distance = Math.hypot(dx, dy);
            minDistance = Math.min(minDistance, distance);
        }

        // Cache the result
        this.distanceCache.set(cacheKey, minDistance);
        return minDistance;
    }

    /**
     * Get distance from a world point to the focus area (e.g., placeholder)
     */
    getDistanceToFocusArea(worldX, worldY) {
        if (!this.focusArea) {
            return Infinity;
        }

        const { x, y, width, height } = this.focusArea;
        const closestX = Math.max(x, Math.min(worldX, x + width));
        const closestY = Math.max(y, Math.min(worldY, y + height));
        const dx = worldX - closestX;
        const dy = worldY - closestY;
        return Math.hypot(dx, dy);
    }

    /**
     * Calculate base dot properties (only operation-dependent)
     */
    calculateBaseDotProperties(worldX, worldY) {
        // Cache based on grid-aligned coordinates to avoid recalculating during zoom/pan
        const gridX = Math.floor(worldX / this.gridSpacing);
        const gridY = Math.floor(worldY / this.gridSpacing);
        const cacheKey = `${gridX},${gridY}`;

        if (this.dotCache.has(cacheKey)) {
            return this.dotCache.get(cacheKey);
        }

        // Calculate distances
        const distanceToOperation = this.getDistanceToNearestOperation(
            worldX,
            worldY,
        );
        const distanceToFocusArea = this.getDistanceToFocusArea(worldX, worldY);

        let size = this.baseDotSize;
        let opacity = this.baseOpacity;

        // Operation proximity influence
        if (this.operationPositions.length > 0) {
            if (distanceToOperation <= this.operationInfluenceRadius) {
                // Dots grow near operations
                const operationInfluence =
                    1 - distanceToOperation / this.operationInfluenceRadius;
                const easedOperationInfluence =
                    operationInfluence * operationInfluence;
                size = Math.max(
                    size,
                    this.baseDotSize +
                        (this.operationMaxDotSize - this.baseDotSize) *
                            easedOperationInfluence,
                );
                opacity = Math.max(
                    opacity,
                    this.baseOpacity +
                        (this.maxOpacity - this.baseOpacity) *
                            easedOperationInfluence *
                            0.3,
                );
            }

            // Fade dots based on distance from nearest operation
            if (distanceToOperation < this.operationFadeDistance) {
                const fadeInfluence =
                    1 - distanceToOperation / this.operationFadeDistance;
                const fadeFactor = Math.pow(fadeInfluence, 0.5); // Smooth falloff
                opacity *= fadeFactor;
            } else {
                // Far from any operation - fade to zero
                opacity = 0;
            }
        } else if (this.focusArea) {
            // No operations, but focus area exists - apply focus area effects only
            if (distanceToFocusArea <= this.focusAreaInfluenceRadius) {
                // Dots grow near focus area
                const focusInfluence =
                    1 - distanceToFocusArea / this.focusAreaInfluenceRadius;
                const easedFocusInfluence = focusInfluence * focusInfluence;
                size = Math.max(
                    size,
                    this.baseDotSize +
                        (this.operationMaxDotSize - this.baseDotSize) *
                            easedFocusInfluence,
                );
                opacity = Math.max(
                    opacity,
                    this.baseOpacity +
                        (this.maxOpacity - this.baseOpacity) *
                            easedFocusInfluence *
                            0.3,
                );
            }

            // Fade dots based on distance from focus area
            if (distanceToFocusArea < this.focusAreaFadeDistance) {
                const fadeInfluence =
                    1 - distanceToFocusArea / this.focusAreaFadeDistance;
                const fadeFactor = Math.pow(fadeInfluence, 0.5); // Smooth falloff
                opacity *= fadeFactor;
            } else {
                // Far from focus area - fade to zero
                opacity = 0;
            }
        }

        const result = { size, opacity };
        this.dotCache.set(cacheKey, result);
        return result;
    }

    /**
     * Calculate dot properties based on cursor and operation proximity
     */
    calculateDotProperties(worldX, worldY) {
        // Get base properties from cache or calculation
        const { size: baseSize, opacity: baseOpacity } =
            this.calculateBaseDotProperties(worldX, worldY);

        // If base opacity is 0 and mouse is far, we can skip
        const mouseWorld = this.screenToWorld(
            this.mouseScreenX,
            this.mouseScreenY,
        );
        const dx = worldX - mouseWorld.x;
        const dy = worldY - mouseWorld.y;
        const distanceToMouse = Math.hypot(dx, dy);

        if (baseOpacity === 0 && distanceToMouse > this.cursorInfluenceRadius) {
            return { size: 0, opacity: 0 };
        }

        let size = baseSize;
        let opacity = baseOpacity;

        // Cursor influence (in world coordinates)
        if (distanceToMouse <= this.cursorInfluenceRadius) {
            const influence = 1 - distanceToMouse / this.cursorInfluenceRadius;
            const easedInfluence = influence * influence;
            size += (this.maxDotSize - this.baseDotSize) * easedInfluence;
            opacity +=
                (this.maxOpacity - this.baseOpacity) * easedInfluence * 0.5;
        }

        return { size, opacity };
    }

    /**
     * Calculate cross grid spacing based on zoom level
     * Spacing doubles at thresholds: 1x, 0.5x, 0.25x, etc.
     */
    getCrossSpacing() {
        if (this.scale >= 1) {
            return this.baseCrossSpacing;
        } else if (this.scale >= 0.5) {
            return this.baseCrossSpacing * 2;
        } else if (this.scale >= 0.25) {
            return this.baseCrossSpacing * 4;
        } else if (this.scale >= 0.125) {
            return this.baseCrossSpacing * 8;
        } else {
            return this.baseCrossSpacing * 16;
        }
    }

    /**
     * Get cross size - constant screen size regardless of zoom
     */
    getCrossSize() {
        return 10;
    }

    /**
     * Calculate cross opacity based on distance from operations or focus area
     * Fades in gradually beyond the fade distance
     */
    calculateCrossOpacity(distanceToOperation, distanceToFocusArea) {
        // If we have operations, use operation-based fade
        if (this.operationPositions.length > 0) {
            const fadeStart = this.operationFadeDistance;
            const fadeEnd = this.operationFadeDistance + 200;

            if (distanceToOperation <= fadeStart) {
                return 0;
            } else if (distanceToOperation >= fadeEnd) {
                return this.crossOpacity;
            } else {
                const fadeProgress =
                    (distanceToOperation - fadeStart) / (fadeEnd - fadeStart);
                return this.crossOpacity * fadeProgress;
            }
        }

        // If we have a focus area but no operations, use focus area-based fade
        if (this.focusArea) {
            const fadeStart = this.focusAreaFadeDistance;
            const fadeEnd = this.focusAreaFadeDistance + 100; // Shorter fade range for focus area

            if (distanceToFocusArea <= fadeStart) {
                return 0;
            } else if (distanceToFocusArea >= fadeEnd) {
                return this.crossOpacity;
            } else {
                const fadeProgress =
                    (distanceToFocusArea - fadeStart) / (fadeEnd - fadeStart);
                return this.crossOpacity * fadeProgress;
            }
        }

        // No operations or focus area
        return this.crossOpacity;
    }

    /**
     * Draw a cross marker at the given screen coordinates
     */
    drawCross(screenX, screenY, size, opacity) {
        this.ctx.strokeStyle = `rgba(128, 128, 128, ${opacity})`;
        this.ctx.lineWidth = 1.5;

        this.ctx.beginPath();
        this.ctx.moveTo(screenX - size, screenY);
        this.ctx.lineTo(screenX + size, screenY);
        this.ctx.stroke();

        this.ctx.beginPath();
        this.ctx.moveTo(screenX, screenY - size);
        this.ctx.lineTo(screenX, screenY + size);
        this.ctx.stroke();
    }

    /**
     * Render the grid for the current viewport state.
     */
    draw() {
        if (!this.ctx) return;

        const rect = this.containerRect;
        const width = rect.width;
        const height = rect.height;

        this.ctx.clearRect(0, 0, width, height);

        // Calculate visible world bounds
        const topLeft = this.screenToWorld(0, 0);
        const bottomRight = this.screenToWorld(width, height);

        // Calculate grid range in world coordinates
        const startX =
            Math.floor(topLeft.x / this.gridSpacing) * this.gridSpacing;
        const startY =
            Math.floor(topLeft.y / this.gridSpacing) * this.gridSpacing;
        const endX =
            Math.ceil(bottomRight.x / this.gridSpacing) * this.gridSpacing;
        const endY =
            Math.ceil(bottomRight.y / this.gridSpacing) * this.gridSpacing;

        // Draw dots
        for (let worldX = startX; worldX <= endX; worldX += this.gridSpacing) {
            for (
                let worldY = startY;
                worldY <= endY;
                worldY += this.gridSpacing
            ) {
                const { size, opacity } = this.calculateDotProperties(
                    worldX,
                    worldY,
                );

                if (opacity > 0.01) {
                    // Convert world position to screen position for drawing
                    const screen = this.worldToScreen(worldX, worldY);

                    // Keep dot size more consistent across zoom levels
                    const scaledSize =
                        size * Math.max(0.8, Math.min(1.2, this.scale));

                    this.ctx.fillStyle = `rgba(128, 128, 128, ${opacity})`;
                    this.ctx.beginPath();
                    this.ctx.arc(
                        screen.x,
                        screen.y,
                        scaledSize,
                        0,
                        Math.PI * 2,
                    );
                    this.ctx.fill();
                }
            }
        }

        // Draw cross markers in areas far from operations
        const crossSpacing = this.getCrossSpacing();
        const crossSize = this.getCrossSize();

        const crossStartX = Math.floor(topLeft.x / crossSpacing) * crossSpacing;
        const crossStartY = Math.floor(topLeft.y / crossSpacing) * crossSpacing;
        const crossEndX =
            Math.ceil(bottomRight.x / crossSpacing) * crossSpacing;
        const crossEndY =
            Math.ceil(bottomRight.y / crossSpacing) * crossSpacing;

        for (
            let worldX = crossStartX;
            worldX <= crossEndX;
            worldX += crossSpacing
        ) {
            for (
                let worldY = crossStartY;
                worldY <= crossEndY;
                worldY += crossSpacing
            ) {
                const distanceToOperation = this.getDistanceToNearestOperation(
                    worldX,
                    worldY,
                );
                const distanceToFocusArea = this.getDistanceToFocusArea(
                    worldX,
                    worldY,
                );

                // Calculate cross opacity based on distance
                const crossOpacity = this.calculateCrossOpacity(
                    distanceToOperation,
                    distanceToFocusArea,
                );

                if (crossOpacity > 0.01) {
                    const screen = this.worldToScreen(worldX, worldY);
                    this.drawCross(screen.x, screen.y, crossSize, crossOpacity);
                }
            }
        }
    }

    /**
     * Kick off the redraw loop.
     */
    startAnimation() {
        this.requestRedraw();
    }

    /**
     * Schedule the next animation frame draw.
     */
    scheduleDraw() {
        if (this.animationFrame) {
            return;
        }

        this.animationFrame = requestAnimationFrame(() => {
            this.animationFrame = null;
            if (!this.needsRedraw) {
                return;
            }
            this.draw();
            this.needsRedraw = false;
        });
    }

    /**
     * Request a redraw with throttling.
     */
    requestRedraw() {
        const now = performance.now();
        const elapsed = now - this.lastRedrawRequestTime;

        if (elapsed >= this.redrawThrottleMs) {
            this.lastRedrawRequestTime = now;
            this.needsRedraw = true;
            this.scheduleDraw();
            return;
        }

        if (this.redrawTimer) {
            return;
        }

        this.redrawTimer = setTimeout(() => {
            this.redrawTimer = null;
            this.lastRedrawRequestTime = performance.now();
            this.needsRedraw = true;
            this.scheduleDraw();
        }, this.redrawThrottleMs - elapsed);
    }

    /**
     * Tear down timers and remove the canvas.
     */
    destroy() {
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
        }
        if (this.redrawTimer) {
            clearTimeout(this.redrawTimer);
            this.redrawTimer = null;
        }
        this.canvas?.remove();
    }
}

/**
 * Initialize interactive grid for the flowchart
 */
export function initializeInteractiveGrid(container, options = {}) {
    if (container) {
        return new InteractiveGrid(container, options);
    }
    return null;
}
