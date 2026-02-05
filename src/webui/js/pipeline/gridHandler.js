export class InteractiveGridHandler {
    constructor(canvasElement, containerElement) {
        this.canvas = canvasElement;
        this.container = containerElement;
        this.ctx = this.canvas.getContext("2d");
        this.dotSize = 2;
        this.dotSpacing = 20;
        this.maxDotSize = 6;
        this.influenceRadius = 100;
        this.cursorX = 0;
        this.cursorY = 0;
        this.scrollX = 0;
        this.scrollY = 0;
        this.animationFrameId = null;
        this.handleResize = () => this.resizeCanvas();
        this.handleMouseMoveBound = (event) => this.handleMouseMove(event);
        this.handleMouseLeaveBound = () => this.handleMouseLeave();
        this.handleScrollBound = () => this.handleScroll();

        this.setupCanvas();
        this.setupEventListeners();
        this.startAnimation();
    }

    setupCanvas() {
        this.resizeCanvas();
        window.addEventListener("resize", this.handleResize);
    }

    resizeCanvas() {
        this.canvas.width = this.container.clientWidth;
        this.canvas.height = this.container.clientHeight;
        this.draw();
    }

    setupEventListeners() {
        this.container.addEventListener("mousemove", this.handleMouseMoveBound);
        this.container.addEventListener("mouseleave", this.handleMouseLeaveBound);
        this.container.addEventListener("scroll", this.handleScrollBound);
    }

    handleMouseMove(event) {
        const rect = this.canvas.getBoundingClientRect();
        const containerRect = this.container.getBoundingClientRect();

        this.cursorX =
            event.clientX - containerRect.left + this.container.scrollLeft;
        this.cursorY =
            event.clientY - containerRect.top + this.container.scrollTop;
    }

    handleMouseLeave() {
        this.cursorX = -this.influenceRadius * 2;
        this.cursorY = -this.influenceRadius * 2;
    }

    handleScroll() {
        this.scrollX = this.container.scrollLeft;
        this.scrollY = this.container.scrollTop;
    }

    calculateDotSize(distance) {
        if (distance > this.influenceRadius) {
            return this.dotSize;
        }

        const normalizedDistance = distance / this.influenceRadius;
        const sizeIncrease =
            (1 - normalizedDistance) * (this.maxDotSize - this.dotSize);
        return this.dotSize + sizeIncrease;
    }

    calculateOpacity(distance) {
        const baseOpacity = 0.2;
        if (distance > this.influenceRadius) {
            return baseOpacity;
        }

        const normalizedDistance = distance / this.influenceRadius;
        const opacityIncrease = (1 - normalizedDistance) * (0.6 - baseOpacity);
        return baseOpacity + opacityIncrease;
    }

    draw() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        for (let x = 0; x < this.canvas.width; x += this.dotSpacing) {
            for (let y = 0; y < this.canvas.height; y += this.dotSpacing) {
                const dx = x - this.cursorX;
                const dy = y - this.cursorY;
                const distance = Math.sqrt(dx * dx + dy * dy);

                const size = this.calculateDotSize(distance);
                const opacity = this.calculateOpacity(distance);

                this.ctx.fillStyle = `rgba(128, 128, 128, ${opacity})`;
                this.ctx.beginPath();
                this.ctx.arc(x, y, size, 0, Math.PI * 2);
                this.ctx.fill();
            }
        }
    }

    startAnimation() {
        const animate = () => {
            this.draw();
            this.animationFrameId = requestAnimationFrame(animate);
        };
        animate();
    }

    destroy() {
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
        }
        this.container.removeEventListener("mousemove", this.handleMouseMoveBound);
        this.container.removeEventListener("mouseleave", this.handleMouseLeaveBound);
        this.container.removeEventListener("scroll", this.handleScrollBound);
        window.removeEventListener("resize", this.handleResize);
    }
}
