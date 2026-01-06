/**
 * InteractiveGrid - Dynamic grid that responds to cursor and operation proximity
 * Works with FlowchartCanvas transform-based viewport
 */
export class InteractiveGrid {
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

        // Operation-related properties
        this.operationPositions = [];
        this.operationInfluenceRadius = 200;
        this.operationFadeDistance = 600;
        this.operationMaxDotSize = 3;

        // Viewport transform state
        this.scale = 1;
        this.translateX = 0;
        this.translateY = 0;

        // Base opacity settings
        this.baseOpacity = 0.25;
        this.maxOpacity = 0.7;

        this.animationFrame = null;
        this.init();
    }

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

        this.resize();
        this.setupEventListeners();
        this.startAnimation();
    }

    resize() {
        const rect = this.container.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;

        this.canvas.width = rect.width * dpr;
        this.canvas.height = rect.height * dpr;
        this.canvas.style.width = `${rect.width}px`;
        this.canvas.style.height = `${rect.height}px`;

        this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    setupEventListeners() {
        this.container.addEventListener("mousemove", (e) => {
            const rect = this.container.getBoundingClientRect();
            this.mouseScreenX = e.clientX - rect.left;
            this.mouseScreenY = e.clientY - rect.top;
        });

        this.container.addEventListener("mouseleave", () => {
            this.mouseScreenX = -1000;
            this.mouseScreenY = -1000;
        });

        const resizeObserver = new ResizeObserver(() => {
            this.resize();
        });
        resizeObserver.observe(this.container);
    }

    /**
     * Update the viewport transform state from FlowchartCanvas
     */
    updateViewport(translateX, translateY, scale) {
        this.translateX = translateX;
        this.translateY = translateY;
        this.scale = scale;
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

        let minDistance = Infinity;
        for (const op of this.operationPositions) {
            // Calculate distance to the rectangle's edge
            const closestX = Math.max(op.x, Math.min(worldX, op.x + op.width));
            const closestY = Math.max(op.y, Math.min(worldY, op.y + op.height));
            const dx = worldX - closestX;
            const dy = worldY - closestY;
            const distance = Math.hypot(dx, dy);
            minDistance = Math.min(minDistance, distance);
        }
        return minDistance;
    }

    /**
     * Calculate dot properties based on cursor and operation proximity
     */
    calculateDotProperties(worldX, worldY) {
        // Convert mouse screen position to world coordinates
        const mouseWorld = this.screenToWorld(
            this.mouseScreenX,
            this.mouseScreenY,
        );

        // Calculate distance to mouse in world coordinates
        const dx = worldX - mouseWorld.x;
        const dy = worldY - mouseWorld.y;
        const distanceToMouse = Math.hypot(dx, dy);

        // Calculate distance to nearest operation
        const distanceToOperation = this.getDistanceToNearestOperation(
            worldX,
            worldY,
        );

        let size = this.baseDotSize;
        let opacity = this.baseOpacity;

        // Cursor influence (in world coordinates)
        if (distanceToMouse <= this.cursorInfluenceRadius) {
            const influence = 1 - distanceToMouse / this.cursorInfluenceRadius;
            const easedInfluence = influence * influence;
            size += (this.maxDotSize - this.baseDotSize) * easedInfluence;
            opacity +=
                (this.maxOpacity - this.baseOpacity) * easedInfluence * 0.5;
        }

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
        }

        return { size, opacity };
    }

    draw() {
        if (!this.ctx) return;

        const rect = this.container.getBoundingClientRect();
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

                    // Scale dot size with zoom level (but not too much)
                    const scaledSize = size * Math.pow(this.scale, 0.5);

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
    }

    startAnimation() {
        const animate = () => {
            this.draw();
            this.animationFrame = requestAnimationFrame(animate);
        };
        animate();
    }

    destroy() {
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
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
