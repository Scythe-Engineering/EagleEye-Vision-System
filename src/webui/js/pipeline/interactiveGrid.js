class InteractiveGrid {
    constructor(container, options = {}) {
        this.container = container;
        this.canvas = null;
        this.ctx = null;
        this.mouseX = -1000;
        this.mouseY = -1000;
        this.gridSpacing = 20;
        this.baseDotSize = 1;
        this.maxDotSize = 3;
        this.influenceRadius = 100;
        this.animationFrame = null;
        this.changeOpacity = 'changeOpacity' in options ? options.changeOpacity : false;
        this.baseOpacity = 0.2;
        this.maxOpacity = 0.7;
        
        this.init();
    }
    
    init() {
        this.canvas = document.createElement('canvas');
        this.canvas.style.position = 'absolute';
        this.canvas.style.top = '0';
        this.canvas.style.left = '0';
        this.canvas.style.pointerEvents = 'none';
        this.canvas.style.zIndex = '0';
        
        this.container.style.position = 'relative';
        this.container.insertBefore(this.canvas, this.container.firstChild);
        
        this.ctx = this.canvas.getContext('2d');
        
        this.resize();
        this.setupEventListeners();
        this.draw();
    }
    
    resize() {
        const rect = this.container.getBoundingClientRect();
        const scrollWidth = this.container.scrollWidth;
        const scrollHeight = this.container.scrollHeight;
        
        const dpr = window.devicePixelRatio || 1;
        
        const width = Math.max(rect.width, scrollWidth);
        const height = Math.max(rect.height, scrollHeight);
        
        this.canvas.width = width * dpr;
        this.canvas.height = height * dpr;
        this.canvas.style.width = `${width}px`;
        this.canvas.style.height = `${height}px`;
        
        this.ctx.scale(dpr, dpr);
    }
    
    setupEventListeners() {
        this.container.addEventListener('mousemove', (e) => {
            const rect = this.container.getBoundingClientRect();
            this.mouseX = e.clientX - rect.left + this.container.scrollLeft;
            this.mouseY = e.clientY - rect.top + this.container.scrollTop;
        });
        
        this.container.addEventListener('mouseleave', () => {
            this.mouseX = -1000;
            this.mouseY = -1000;
        });
        
        const resizeObserver = new ResizeObserver(() => {
            this.resize();
        });
        resizeObserver.observe(this.container);
        
        this.container.addEventListener('scroll', () => {
            this.draw();
        });
    }
    
    calculateDotProperties(dotX, dotY) {
        const dx = dotX - this.mouseX;
        const dy = dotY - this.mouseY;
        const distance = Math.hypot(dx, dy);
        
        if (distance > this.influenceRadius) {
            return {
                size: this.baseDotSize,
                opacity: this.baseOpacity
            };
        }
        
        const influence = 1 - (distance / this.influenceRadius);
        const easedInfluence = influence * influence;
        
        const size = this.baseDotSize + (this.maxDotSize - this.baseDotSize) * easedInfluence;
        const opacity = this.changeOpacity 
            ? this.baseOpacity + (this.maxOpacity - this.baseOpacity) * easedInfluence
            : this.baseOpacity;
        
        return { size, opacity };
    }
    
    draw() {
        if (!this.ctx) return;
        
        const dpr = window.devicePixelRatio || 1;
        this.ctx.clearRect(0, 0, this.canvas.width / dpr, this.canvas.height / dpr);
        
        const scrollX = this.container.scrollLeft;
        const scrollY = this.container.scrollTop;
        const viewWidth = this.container.clientWidth;
        const viewHeight = this.container.clientHeight;
        
        const startX = Math.floor(scrollX / this.gridSpacing) * this.gridSpacing;
        const startY = Math.floor(scrollY / this.gridSpacing) * this.gridSpacing;
        const endX = scrollX + viewWidth + this.gridSpacing;
        const endY = scrollY + viewHeight + this.gridSpacing;
        
        for (let x = startX; x <= endX; x += this.gridSpacing) {
            for (let y = startY; y <= endY; y += this.gridSpacing) {
                const { size, opacity } = this.calculateDotProperties(x, y);
                
                this.ctx.fillStyle = `rgba(128, 128, 128, ${opacity})`;
                this.ctx.beginPath();
                this.ctx.arc(x, y, size, 0, Math.PI * 2);
                this.ctx.fill();
            }
        }
        
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
        }
        this.animationFrame = requestAnimationFrame(() => this.draw());
    }
    
    destroy() {
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
        }
        this.canvas?.remove();
    }
}

export function initializeInteractiveGrid(options = {}) {
    const pipelineArea = document.getElementById('pipelineArea');
    if (pipelineArea) {
        return new InteractiveGrid(pipelineArea, options);
    }
    return null;
}
