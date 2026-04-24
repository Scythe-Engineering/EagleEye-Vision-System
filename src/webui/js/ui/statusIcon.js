const ICON_SIZE = 64;
const ICON_CENTER = ICON_SIZE / 2;
const ICON_RADIUS = 28;
const ICON_STROKE_WIDTH = 5;
const ANIMATION_FPS = 20;
const ANIMATION_FRAME_MS = 1000 / ANIMATION_FPS;
const RING_SWEEP_RADIANS = Math.PI * 1.35;
const RING_ROTATION_PERIOD_MS = 1800;
const CONNECTED_RING_HOLD_MS = 1000;
const CONNECTED_RING_COLLAPSE_MS = 450;
const CONNECTED_DOT_ANGLE = -Math.PI / 4;
const CONNECTED_DOT_RADIUS = 8;

const STATUS_COLORS = {
    connected: "#43d17a",
    partial: "#f9c845",
    disconnected: "#ff4d4d",
};

function createCanvasSurface() {
    const canvas = document.createElement("canvas");
    canvas.width = ICON_SIZE;
    canvas.height = ICON_SIZE;

    const context = canvas.getContext("2d");
    if (!context) {
        throw new Error("Unable to create 2D canvas context for favicon");
    }

    return { canvas, context };
}

function loadImage(url) {
    return new Promise((resolve, reject) => {
        const image = new Image();
        image.decoding = "async";
        image.onload = () => resolve(image);
        image.onerror = () => reject(new Error(`Failed to load favicon image: ${url}`));
        image.src = url;
    });
}

function drawArc(context, color, startAngle, endAngle) {
    context.strokeStyle = color;
    context.lineWidth = ICON_STROKE_WIDTH;
    context.lineCap = "round";
    context.beginPath();
    context.arc(ICON_CENTER, ICON_CENTER, ICON_RADIUS, startAngle, endAngle);
    context.stroke();
}

function drawConnectedDot(context, color) {
    context.fillStyle = color;
    context.beginPath();
    context.arc(
        ICON_CENTER + Math.cos(CONNECTED_DOT_ANGLE) * ICON_RADIUS,
        ICON_CENTER + Math.sin(CONNECTED_DOT_ANGLE) * ICON_RADIUS,
        CONNECTED_DOT_RADIUS,
        0,
        Math.PI * 2,
    );
    context.fill();
}

function drawConnectedIndicator(context, color, elapsedMs) {
    if (elapsedMs < CONNECTED_RING_HOLD_MS) {
        drawArc(context, color, 0, Math.PI * 2);
        return;
    }

    const collapseProgress = Math.min(
        1,
        (elapsedMs - CONNECTED_RING_HOLD_MS) / CONNECTED_RING_COLLAPSE_MS,
    );

    if (collapseProgress >= 1) {
        drawConnectedDot(context, color);
        return;
    }

    const sweep = Math.PI * 2 * (1 - collapseProgress);
    drawArc(
        context,
        color,
        CONNECTED_DOT_ANGLE - sweep / 2,
        CONNECTED_DOT_ANGLE + sweep / 2,
    );
}

function drawAnimatedRing(context, color, timestampMs) {
    const phase = (timestampMs % RING_ROTATION_PERIOD_MS) / RING_ROTATION_PERIOD_MS;
    const baseAngle = -Math.PI / 2 + phase * Math.PI * 2;
    drawArc(context, color, baseAngle, baseAngle + RING_SWEEP_RADIANS);
}

function renderFrame(context, image, status, timestampMs, connectedStartedAtMs) {
    context.clearRect(0, 0, ICON_SIZE, ICON_SIZE);

    if (image) {
        context.drawImage(image, 0, 0, ICON_SIZE, ICON_SIZE);
    }

    if (status === "connected") {
        drawConnectedIndicator(
            context,
            STATUS_COLORS.connected,
            timestampMs - connectedStartedAtMs,
        );
        return;
    }

    const ringColor =
        status === "partial"
            ? STATUS_COLORS.partial
            : STATUS_COLORS.disconnected;
    drawAnimatedRing(context, ringColor, timestampMs);
}

export function createStatusIconController({ targetLink, baseIconUrl }) {
    const { canvas, context } = createCanvasSurface();
    let destroyed = false;
    let animationFrameId = null;
    let lastRenderedFrameMs = 0;
    let currentStatus = "disconnected";
    let connectedStartedAtMs = 0;
    let baseImage = null;
    let baseImageReady = false;
    let targetHref = targetLink?.href ?? null;

    function isConnectedTransitionComplete(timestampMs) {
        return (
            currentStatus === "connected" &&
            timestampMs - connectedStartedAtMs >=
                CONNECTED_RING_HOLD_MS + CONNECTED_RING_COLLAPSE_MS
        );
    }

    function syncTargetLink() {
        if (!targetLink) {
            return;
        }

        const dataUrl = canvas.toDataURL("image/png");
        if (targetLink.href !== dataUrl) {
            targetLink.href = dataUrl;
        }
    }

    function render(timestampMs = performance.now()) {
        if (destroyed || !baseImageReady) {
            return;
        }

        if (timestampMs - lastRenderedFrameMs < ANIMATION_FRAME_MS) {
            scheduleAnimation();
            return;
        }

        lastRenderedFrameMs = timestampMs;
        renderFrame(
            context,
            baseImage,
            currentStatus,
            timestampMs,
            connectedStartedAtMs,
        );
        syncTargetLink();

        if (
            currentStatus !== "connected" ||
            !isConnectedTransitionComplete(timestampMs)
        ) {
            scheduleAnimation();
        }
    }

    function stopAnimation() {
        if (animationFrameId !== null) {
            cancelAnimationFrame(animationFrameId);
            animationFrameId = null;
        }
    }

    function scheduleAnimation() {
        if (destroyed || animationFrameId !== null) {
            return;
        }

        animationFrameId = requestAnimationFrame((timestampMs) => {
            animationFrameId = null;
            render(timestampMs);
        });
    }

    function resolveBaseImage() {
        if (!baseIconUrl) {
            baseImageReady = true;
            render();
            return;
        }

        loadImage(baseIconUrl)
            .then((image) => {
                if (destroyed) {
                    return;
                }

                baseImage = image;
                baseImageReady = true;
                render();
            })
            .catch((error) => {
                console.warn(error.message);
                baseImageReady = true;
                render();
            });
    }

    function setStatus(nextStatus) {
        if (
            nextStatus !== "connected" &&
            nextStatus !== "partial" &&
            nextStatus !== "disconnected"
        ) {
            return;
        }

        if (currentStatus === nextStatus) {
            return;
        }

        currentStatus = nextStatus;
        if (nextStatus === "connected") {
            connectedStartedAtMs = performance.now();
        }
        lastRenderedFrameMs = 0;
        stopAnimation();
        render();
    }

    function destroy() {
        destroyed = true;
        stopAnimation();

        if (targetLink && targetHref) {
            targetLink.href = targetHref;
        }

        baseImage = null;
        baseImageReady = false;
    }

    if (targetLink) {
        targetHref = targetLink.href;
    }

    resolveBaseImage();

    return {
        setStatus,
        destroy,
    };
}
