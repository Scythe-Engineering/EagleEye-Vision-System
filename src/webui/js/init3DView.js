/**
 * Initializes and manages the 3D web UI view, rendering models, detections, and camera markers.
 */
import { GLTFLoader } from "GLTFLoader";
import {
    PCFSoftShadowMap,
    WebGLRenderer,
    AmbientLight,
    AxesHelper,
    DirectionalLight,
    PerspectiveCamera,
    Scene,
    Color,
    Clock,
    Mesh,
    MeshStandardMaterial,
    PlaneGeometry,
    CylinderGeometry,
    TextureLoader,
    CanvasTexture,
    Box3,
    Matrix4,
    NearestFilter,
    Vector3,
    BufferGeometry,
    BufferAttribute,
    LineSegments,
    LineBasicMaterial,
    Group,
    Sprite,
    SpriteMaterial,
    MathUtils,
} from "three";
import { OrbitControls } from "OrbitControls";
import { DRACOLoader } from "DRACOLoader";
import { populateRobotDropdown } from "./dropdown/robotDropdown.js";
import { BACKEND_BASE_URL } from "./config.js";
import { position3DToFieldSpaceVector } from "./utils/fieldSpaceTransforms.js";

let renderer, scene, camera, directionalLight, controls;
let shadowsEnabled = true;
let gamePiecesVisible = true;
let statsDisplay;
let frameCount = 0;
let lastTime = performance.now();
let robotObject = null;
let fieldObject = null;
let robotAxes = null;
let animationStarted = false;
let animationFrameId = null;
let resizeHandler = null;
let detectedObjectsGroup = null;
let detectedObjectsFrameId = null;
let cameraMarkersGroup = null;
let pendingDetectedObjects = null;
const pendingCameraPoses = new Map();
const cameraMarkers = new Map();
let allCameraMarkersVisible = true;
let robotModelVisible = true;
let currentLoadToken = 0;
let gamePieces = [];
let gamePieceObjects = [];
let currentRobotLoader = null;
let robotFileSelectListenerAttached = false;
let currentRobotFile = null;
let currentRobotScaleFactor = 1;
let lastRobotTransformMatrix = null;
let currentFieldScaleFactor = 1;
let currentFieldRotationOffset = { x: 0, y: 0, z: 0 };
let currentFieldYear = null;
let currentFieldFilename = null;
let activeDracoLoader = null;

let maxFPS = 30;
let interval = 1 / maxFPS;

const robotBaseScale = 1000;
const robotScaleMatrix = new Matrix4().makeScale(
    robotBaseScale,
    robotBaseScale,
    robotBaseScale,
);
const robotFinalMatrix = new Matrix4();
const robotPoseRotationMatrix = new Matrix4();
const robotPoseTranslationMatrix = new Matrix4();
const robotPosePosition = new Vector3();
// The pose comes in robot-space, then gets converted into the Three.js visual
// basis. Keep the basis swap separate from the visual roll steps:
// - Y-axis swap: matches the robot object's axis convention
// - X rotation: the original fixed visual tilt
// - Z rotation: the extra 90° roll that would otherwise read like yaw here
const visualOrientationMatrix = new Matrix4().makeRotationX(-Math.PI / 2);
const extraVisualRollMatrix = new Matrix4().makeRotationZ(-Math.PI / 2);
// Robot-only pitch correction. Keep this separate from the shared visual basis
// so camera markers continue using the same transform chain.
const robotPitchUpMatrix = new Matrix4().makeRotationY(Math.PI / 2);
const robotActualPitchMatrix = new Matrix4().makeRotationX(-Math.PI / 2);
const robotPitchRollSwap = new Matrix4().makeRotationY(Math.PI / 2);
const robotPitchRollSwapInverse = new Matrix4().makeRotationY(-Math.PI / 2);
const robotCorrectedRotation = new Matrix4();
const cameraCorrectedRotation = new Matrix4();
const detectionCylinderRadius = 150;
const detectionCylinderHeight = 400;
const detectionLabelOffset = 250;
const cameraFrustumLength = 350;
const cameraFrustumHalfHeight = 130;
const cameraFrustumHalfWidth = 160;
const cameraLabelOffset = 230;
const cameraPoseStaleTimeoutMs = 2000;

/**
 * Returns whether a URL is absolute or uses a non-HTTP data/blob scheme.
 */
function isAbsoluteUrl(url) {
    return (
        /^[a-z][a-z\d+\-.]*:\/\//i.test(url) ||
        url.startsWith("data:") ||
        url.startsWith("blob:")
    );
}

/**
 * Resolves a backend asset path to a fully qualified URL when needed.
 */
function buildBackendAssetUrl(assetPath) {
    if (isAbsoluteUrl(assetPath)) {
        return assetPath;
    }
    const normalizedPath = assetPath.startsWith("./")
        ? assetPath.slice(1)
        : assetPath;
    if (normalizedPath.startsWith("/assets/")) {
        return `${BACKEND_BASE_URL}${normalizedPath}`;
    }
    if (normalizedPath.startsWith("/")) {
        return `${BACKEND_BASE_URL}${normalizedPath}`;
    }
    return assetPath;
}

/**
 * Normalizes an asset scale value to a positive finite number.
 */
function normalizeAssetScale(scale) {
    const numericScale = Number.parseFloat(scale);
    return Number.isFinite(numericScale) && numericScale > 0 ? numericScale : 1;
}

function normalizeRotationOffset(rotationOffset) {
    const rotation = { x: 0, y: 0, z: 0 };
    for (const axis of Object.keys(rotation)) {
        const value = Number.parseFloat(rotationOffset?.[axis]);
        rotation[axis] = Number.isFinite(value) ? value : 0;
    }
    return rotation;
}

function applyFieldTransform(model) {
    if (!model) {
        return;
    }
    model.rotation.set(
        Math.PI / 2 + MathUtils.degToRad(currentFieldRotationOffset.x),
        MathUtils.degToRad(currentFieldRotationOffset.y),
        MathUtils.degToRad(currentFieldRotationOffset.z),
    );
    model.scale.set(
        currentFieldScaleFactor,
        currentFieldScaleFactor,
        currentFieldScaleFactor,
    );
    model.updateMatrix();
    model.updateMatrixWorld(true);
}

function renderOnce() {
    if (renderer && scene && camera) {
        renderer.render(scene, camera);
    }
}

/**
 * Recomputes the robot scale matrix from the current robot scale factor.
 */
function updateRobotScaleMatrix() {
    const scale = robotBaseScale * currentRobotScaleFactor;
    robotScaleMatrix.makeScale(scale, scale, scale);
}

/**
 * Updates the robot model transform using the latest pose and scale state.
 */
function refreshRobotMatrix() {
    if (!robotObject) {
        return;
    }

    if (lastRobotTransformMatrix) {
        robotPoseRotationMatrix.extractRotation(lastRobotTransformMatrix);
        robotPosePosition.setFromMatrixPosition(lastRobotTransformMatrix);
        robotPoseTranslationMatrix.identity().setPosition(robotPosePosition);

        robotCorrectedRotation
            .copy(robotPitchRollSwap)
            .multiply(robotPoseRotationMatrix)
            .multiply(robotPitchRollSwapInverse);

        robotFinalMatrix
            .copy(robotPoseTranslationMatrix)
            .multiply(robotCorrectedRotation)
            .multiply(visualOrientationMatrix)
            .multiply(robotPitchUpMatrix)
            .multiply(robotActualPitchMatrix)
            .multiply(robotScaleMatrix);
    } else {
        robotFinalMatrix
            .copy(visualOrientationMatrix)
            .multiply(robotPitchUpMatrix)
            .multiply(robotActualPitchMatrix)
            .multiply(robotScaleMatrix);
    }

    robotObject.matrixAutoUpdate = false;
    robotObject.matrix.copy(robotFinalMatrix);
    robotObject.matrixWorldNeedsUpdate = true;
}

/**
 * Applies the shared visual-axis correction used for camera and detection transforms.
 */
function applySharedVisualAxisCorrection(matrix) {
    return matrix
        .multiply(robotPitchRollSwap)
        .multiply(visualOrientationMatrix)
        .multiply(extraVisualRollMatrix)
        .multiply(robotPitchRollSwapInverse);
}

/**
 * Applies a new robot scale factor and refreshes the robot transform.
 */
function applyRobotScaleFactor(scale) {
    currentRobotScaleFactor = normalizeAssetScale(scale);
    updateRobotScaleMatrix();
    refreshRobotMatrix();
}

/**
 * Applies a new field scale factor to the field and game-piece models.
 */
function applyFieldScaleFactor(scale) {
    currentFieldScaleFactor = normalizeAssetScale(scale);
    applyFieldTransform(fieldObject);

    for (const gamePieceObject of gamePieceObjects) {
        applyFieldTransform(gamePieceObject);
    }

    if (renderer?.shadowMap) {
        renderer.shadowMap.needsUpdate = true;
    }
}

/**
 * Applies a scale update to the currently loaded 3D asset when it matches the active asset.
 */

export function apply3DAssetScale(assetType, asset, scale, rotationOffset = null) {
    if (assetType === "robot" && asset?.filename === currentRobotFile) {
        applyRobotScaleFactor(scale);
        return;
    }

    if (
        assetType === "field" &&
        asset?.year === currentFieldYear &&
        asset?.filename === currentFieldFilename
    ) {
        currentFieldRotationOffset = normalizeRotationOffset(rotationOffset || currentFieldRotationOffset);
        applyFieldScaleFactor(scale);
        renderOnce();
    }
}

/**
 * Collects the DOM elements used by the 3D loading overlay.
 */
function getLoadingElements() {
    return {
        overlay: document.getElementById("threeDLoadingOverlay"),
        status: document.getElementById("threeDLoadingStatus"),
        progress: document.getElementById("threeDLoadingProgress"),
    };
}

/**
 * Hides and resets the 3D loading overlay.
 */
function hideLoadingOverlay() {
    const { overlay, progress } = getLoadingElements();
    if (overlay) {
        overlay.classList.add("hidden");
        overlay.setAttribute("aria-busy", "false");
    }
    if (progress) {
        progress.value = 0;
        progress.removeAttribute("aria-valuetext");
    }
}

/**
 * Creates a tracker for coordinating loading progress across multiple async tasks.
 */
function createLoadingTracker(token) {
    const pendingTasks = new Map();
    const taskProgress = new Map();
    let failedCount = 0;

    /**
     * Refreshes the loading overlay using the current task state.
     */
    function update() {
        if (token !== currentLoadToken) {
            return;
        }

        const { overlay, status, progress } = getLoadingElements();
        if (!overlay || !status || !progress) {
            return;
        }

        if (pendingTasks.size === 0) {
            status.textContent =
                failedCount > 0
                    ? "Loaded with missing assets. Check the browser console."
                    : "Ready.";
            progress.value = failedCount > 0 ? 0 : 100;
            progress.removeAttribute("aria-valuetext");
            overlay.classList.add("hidden");
            overlay.setAttribute("aria-busy", "false");
            return;
        }

        const taskList = Array.from(pendingTasks.values()).join(", ");
        status.textContent = `Loading ${taskList}...`;
        const knownProgress = Array.from(taskProgress.values()).filter(
            (value) => Number.isFinite(value),
        );
        if (knownProgress.length > 0) {
            const averageProgress =
                knownProgress.reduce((sum, value) => sum + value, 0) /
                knownProgress.length;
            progress.value = Math.round(averageProgress);
            progress.setAttribute(
                "aria-valuetext",
                `${Math.round(averageProgress)}% loaded`,
            );
        } else {
            progress.removeAttribute("value");
            progress.setAttribute("aria-valuetext", "Loading assets");
        }
        overlay.classList.remove("hidden");
        overlay.setAttribute("aria-busy", "true");
    }

    /**
     * Registers a new loading task.
     */
    function start(key, label) {
        pendingTasks.set(key, label);
        taskProgress.set(key, 0);
        update();
    }

    /**
     * Updates the progress for a loading task.
     */
    function progress(key, loaded, total) {
        if (!pendingTasks.has(key)) {
            return;
        }

        if (Number.isFinite(total) && total > 0) {
            taskProgress.set(key, Math.min(100, (loaded / total) * 100));
        } else {
            taskProgress.set(key, Number.NaN);
        }
        update();
    }

    /**
     * Marks a loading task as complete.
     */
    function finish(key) {
        pendingTasks.delete(key);
        taskProgress.delete(key);
        update();
    }

    /**
     * Marks a loading task as failed.
     */
    function fail(key, errorMessage) {
        failedCount += 1;
        pendingTasks.delete(key);
        taskProgress.delete(key);
        console.error(errorMessage);
        update();
    }

    return { start, progress, finish, fail };
}

/**
 * Updates the on-screen scene statistics display.
 */
function updateStats() {
    if (!scene || !statsDisplay) {
        return;
    }

    const currentTime = performance.now();
    frameCount++;
    if (currentTime - lastTime >= 1000) {
        const fps = frameCount;
        frameCount = 0;
        lastTime = currentTime;

        let numVerts = 0;
        scene.traverse((object) => {
            if (object.isMesh) {
                numVerts += object.geometry.attributes.position.count;
            }
        });

        statsDisplay.textContent = `Verts: ${numVerts} | FPS: ${fps}`;
    }
}

/**
 * Creates the axis helper used to visualize the robot coordinate frame.
 */
function createRobotAxes() {
    const axesGroup = new Group();
    const axisLength = 500; // Adjust length as needed

    // Create geometry for axes lines
    const positions = new Float32Array([
        // X-axis (red)
        0,
        0,
        0,
        axisLength,
        0,
        0,
        // Y-axis (green)
        0,
        0,
        0,
        0,
        axisLength,
        0,
        // Z-axis (blue)
        0,
        0,
        0,
        0,
        0,
        axisLength,
    ]);

    const colors = new Float32Array([
        // X-axis (red)
        1, 0, 0, 1, 0, 0,
        // Y-axis (green)
        0, 1, 0, 0, 1, 0,
        // Z-axis (blue)
        0, 0, 1, 0, 0, 1,
    ]);

    const geometry = new BufferGeometry();
    geometry.setAttribute("position", new BufferAttribute(positions, 3));
    geometry.setAttribute("color", new BufferAttribute(colors, 3));

    const material = new LineBasicMaterial({
        vertexColors: true,
        linewidth: 3,
    });

    const axes = new LineSegments(geometry, material);
    axesGroup.add(axes);

    return axesGroup;
}

/**
 * Disposes a Three.js object tree and any attached GPU resources.
 */
function disposeObject(object) {
    object.traverse((node) => {
        if (node.geometry) {
            node.geometry.dispose();
        }
        if (node.material) {
            const materials = Array.isArray(node.material)
                ? node.material
                : [node.material];
            for (const material of materials) {
                for (const key in material) {
                    const value = material[key];
                    if (value?.isTexture) {
                        value.dispose();
                    }
                }
                material.dispose();
            }
        }
    });
}

/**
 * Removes an object from its parent and disposes its resources.
 */
function removeAndDisposeObject(object) {
    if (!object) {
        return;
    }
    object.parent?.remove(object);
    disposeObject(object);
}

/**
 * Stops the active animation loop if one is running.
 */
function stopAnimationLoop() {
    animationStarted = false;
    if (animationFrameId !== null) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
}

/**
 * Cancels any queued detected-object render request.
 */
function cancelPendingDetectedObjectRender() {
    if (detectedObjectsFrameId !== null) {
        cancelAnimationFrame(detectedObjectsFrameId);
        detectedObjectsFrameId = null;
    }
}

/**
 * Disposes the WebGL renderer and removes its canvas from the DOM.
 */
function disposeRenderer() {
    if (!renderer) {
        return;
    }

    renderer.dispose();
    renderer.forceContextLoss?.();

    if (renderer.domElement?.parentNode) {
        renderer.domElement.parentNode.removeChild(renderer.domElement);
    }

    renderer = null;
}

/**
 * Disposes all objects currently attached to the scene.
 */
function disposeSceneObjects() {
    if (!scene) {
        return;
    }

    clearDetectedObjectsGroup();
    clearCameraMarkersGroup();

    while (scene.children.length > 0) {
        const child = scene.children[0];
        scene.remove(child);
        disposeObject(child);
    }

    scene.clear();
    scene = null;
}

/**
 * Cleans up the current 3D view state and optionally invalidates pending loads.
 */
function teardown3DView({ invalidateLoads = true } = {}) {
    if (invalidateLoads) {
        currentLoadToken += 1;
    }

    stopAnimationLoop();
    cancelPendingDetectedObjectRender();
    controls?.dispose();
    controls = null;
    activeDracoLoader?.dispose();
    activeDracoLoader = null;
    disposeSceneObjects();
    disposeRenderer();
    hideLoadingOverlay();

    camera = null;
    directionalLight = null;
    robotObject = null;
    fieldObject = null;
    robotAxes = null;
    detectedObjectsGroup = null;
    cameraMarkersGroup = null;
    gamePieceObjects = [];
    gamePieces = [];
    currentRobotLoader = null;
    currentRobotFile = null;
    lastRobotTransformMatrix = null;
}

/**
 * Publicly disposes the 3D view.
 */

export function dispose3DView() {
    teardown3DView();
}

/**
 * Removes and disposes all detected-object visuals.
 */
function clearDetectedObjectsGroup() {
    if (!detectedObjectsGroup) {
        return;
    }
    while (detectedObjectsGroup.children.length > 0) {
        const child = detectedObjectsGroup.children.pop();
        if (child) {
            detectedObjectsGroup.remove(child);
            disposeObject(child);
        }
    }
}

/**
 * Removes and disposes all camera marker visuals.
 */
function clearCameraMarkersGroup() {
    while (cameraMarkersGroup && cameraMarkersGroup.children.length > 0) {
        const child = cameraMarkersGroup.children.pop();
        if (child) {
            cameraMarkersGroup.remove(child);
            disposeObject(child);
        }
    }
    cameraMarkers.clear();
}

/**
 * Derives a stable hue value from a detection class identifier.
 */
function getHueFromClassIdentifier(classIdentifier) {
    if (
        typeof classIdentifier === "number" &&
        Number.isFinite(classIdentifier)
    ) {
        const hueDegrees = (classIdentifier * 137.5) % 360;
        return hueDegrees / 360;
    }
    const key = String(classIdentifier ?? "detection");
    let hash = 0;
    for (let index = 0; index < key.length; index += 1) {
        hash = (hash * 31 + key.charCodeAt(index)) % 360;
    }
    return (hash % 360) / 360;
}

/**
 * Clamps a confidence value to the inclusive range [0, 1].
 */
function clampConfidence(confidence) {
    if (typeof confidence !== "number" || !Number.isFinite(confidence)) {
        return null;
    }
    if (confidence < 0) {
        return 0;
    }
    if (confidence > 1) {
        return 1;
    }
    return confidence;
}

/**
 * Converts a detection position into field-space coordinates.
 */
function normalizeDetectionPosition(position) {
    return position3DToFieldSpaceVector(position);
}

/**
 * Creates the material used to render a detection cylinder.
 */
function createDetectionMaterial(classIdentifier, normalizedConfidence) {
    const hue = getHueFromClassIdentifier(classIdentifier);
    const saturation = 0.7;
    const baseLightness = 0.45;
    const lightnessRange = 0.2;
    const confidenceValue =
        normalizedConfidence === null ? 0.5 : normalizedConfidence;
    const materialColor = new Color();
    materialColor.setHSL(
        hue,
        saturation,
        baseLightness + lightnessRange * (confidenceValue - 0.5),
    );
    const opacityBase = 0.35;
    const opacityRange = 0.35;
    const materialOpacity = opacityBase + opacityRange * confidenceValue;
    return new MeshStandardMaterial({
        color: materialColor,
        transparent: true,
        opacity: materialOpacity,
        depthWrite: false,
    });
}

/**
 * Creates the cylinder mesh used to visualize a detection.
 */
function createDetectionCylinderMesh(classIdentifier, normalizedConfidence) {
    const geometry = new CylinderGeometry(
        detectionCylinderRadius,
        detectionCylinderRadius,
        detectionCylinderHeight,
        28,
    );
    const material = createDetectionMaterial(
        classIdentifier,
        normalizedConfidence,
    );
    const cylinder = new Mesh(geometry, material);
    cylinder.position.y = detectionCylinderHeight / 2;
    cylinder.castShadow = false;
    cylinder.receiveShadow = false;
    cylinder.excludeFromShadowToggle = true;
    return cylinder;
}

/**
 * Builds the label text for a detection marker.
 */
function buildDetectionLabelText(detection, normalizedConfidence) {
    const classIdentifier =
        detection.class_name ?? detection.class_id ?? "Detection";
    const classLabel = String(classIdentifier);
    if (normalizedConfidence === null) {
        return classLabel;
    }
    const confidencePercent = Math.round(normalizedConfidence * 100);
    return `${classLabel} ${confidencePercent}%`;
}

/**
 * Creates a text sprite for use as a scene label.
 */
function createLabelSprite(
    labelText,
    textColor = "#ffffff",
    backgroundColor = "rgba(20, 20, 20, 0.8)",
) {
    const canvas = document.createElement("canvas");
    canvas.width = 512;
    canvas.height = 128;
    const context = canvas.getContext("2d");
    if (!context) {
        return null;
    }
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = backgroundColor;
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = textColor;
    context.font = "bold 64px Arial";
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillText(labelText, canvas.width / 2, canvas.height / 2);
    const texture = new CanvasTexture(canvas);
    const material = new SpriteMaterial({ map: texture, transparent: true });
    const sprite = new Sprite(material);
    const scaleFactor = 0.4;
    sprite.scale.set(
        canvas.width * scaleFactor,
        canvas.height * scaleFactor,
        1,
    );
    sprite.center.set(0.5, 0);
    sprite.renderOrder = 1000;
    return sprite;
}

/**
 * Creates a Three.js group that represents a single detection.
 */
function createDetectionGroup(detection) {
    if (!detection || typeof detection !== "object") {
        return null;
    }
    const positionVector = normalizeDetectionPosition(detection.position_3d);
    if (!positionVector) {
        return null;
    }
    const normalizedConfidence = clampConfidence(detection.confidence);
    const classIdentifier =
        detection.class_id ?? detection.class_name ?? "Detection";
    const detectionGroup = new Group();
    detectionGroup.position.copy(positionVector);
    const cylinder = createDetectionCylinderMesh(
        classIdentifier,
        normalizedConfidence,
    );
    detectionGroup.add(cylinder);
    const labelText = buildDetectionLabelText(detection, normalizedConfidence);
    const labelSprite = createLabelSprite(labelText);
    if (labelSprite) {
        labelSprite.position.y = detectionCylinderHeight + detectionLabelOffset;
        detectionGroup.add(labelSprite);
    }
    return detectionGroup;
}

/**
 * Renders the current set of detected objects into the scene.
 */
function renderDetectedObjects(detections) {
    if (!detectedObjectsGroup) {
        return;
    }
    clearDetectedObjectsGroup();
    if (!Array.isArray(detections)) {
        return;
    }
    for (const detection of detections) {
        const detectionGroup = createDetectionGroup(detection);
        if (detectionGroup) {
            detectedObjectsGroup.add(detectionGroup);
        }
    }
}

/**
 * Queues a detected-object scene update for the next animation frame.
 */

export function updateDetectedObjects(detections) {
    pendingDetectedObjects = detections;
    if (detectedObjectsFrameId !== null) {
        return;
    }

    detectedObjectsFrameId = requestAnimationFrame(() => {
        detectedObjectsFrameId = null;
        renderDetectedObjects(pendingDetectedObjects);
    });
}

/**
 * Synchronizes the robot model visibility with the current toggle state.
 */
function applyRobotModelVisibility() {
    if (robotObject) {
        robotObject.visible = robotModelVisible;
    }
}

/**
 * Returns the display name for a camera pose.
 */
function getCameraDisplayName(cameraPose) {
    return String(cameraPose.cameraName || cameraPose.cameraBusId);
}

/**
 * Derives a stable hue value from a generic identifier.
 */
function getHueFromIdentifier(identifier) {
    const key = String(identifier ?? "camera");
    let hash = 0;
    for (let index = 0; index < key.length; index += 1) {
        hash = (hash * 31 + key.charCodeAt(index)) % 360;
    }
    return (hash % 360) / 360;
}

/**
 * Returns the accent color used for a camera marker.
 */
function getCameraAccentColor(cameraBusId) {
    const color = new Color();
    color.setHSL(getHueFromIdentifier(cameraBusId), 0.72, 0.58);
    return color;
}

/**
 * Creates the wireframe frustum used to visualize a camera.
 */
function createCameraFrustumMesh(cameraBusId) {
    const positions = new Float32Array([
        0,
        0,
        0,
        cameraFrustumLength,
        cameraFrustumHalfHeight,
        cameraFrustumHalfWidth,
        0,
        0,
        0,
        cameraFrustumLength,
        cameraFrustumHalfHeight,
        -cameraFrustumHalfWidth,
        0,
        0,
        0,
        cameraFrustumLength,
        -cameraFrustumHalfHeight,
        cameraFrustumHalfWidth,
        0,
        0,
        0,
        cameraFrustumLength,
        -cameraFrustumHalfHeight,
        -cameraFrustumHalfWidth,
        cameraFrustumLength,
        cameraFrustumHalfHeight,
        cameraFrustumHalfWidth,
        cameraFrustumLength,
        cameraFrustumHalfHeight,
        -cameraFrustumHalfWidth,
        cameraFrustumLength,
        cameraFrustumHalfHeight,
        -cameraFrustumHalfWidth,
        cameraFrustumLength,
        -cameraFrustumHalfHeight,
        -cameraFrustumHalfWidth,
        cameraFrustumLength,
        -cameraFrustumHalfHeight,
        -cameraFrustumHalfWidth,
        cameraFrustumLength,
        -cameraFrustumHalfHeight,
        cameraFrustumHalfWidth,
        cameraFrustumLength,
        -cameraFrustumHalfHeight,
        cameraFrustumHalfWidth,
        cameraFrustumLength,
        cameraFrustumHalfHeight,
        cameraFrustumHalfWidth,
    ]);

    const geometry = new BufferGeometry();
    geometry.setAttribute("position", new BufferAttribute(positions, 3));
    const material = new LineBasicMaterial({
        color: getCameraAccentColor(cameraBusId),
    });
    const frustum = new LineSegments(geometry, material);
    frustum.excludeFromShadowToggle = true;
    return frustum;
}

/**
 * Creates the scene object used to represent a camera pose.
 */
function createCameraMarker(cameraPose) {
    const markerGroup = new Group();
    markerGroup.matrixAutoUpdate = false;
    markerGroup.userData.cameraBusId = cameraPose.cameraBusId;
    markerGroup.userData.labelText = getCameraDisplayName(cameraPose);
    markerGroup.userData.lastUpdatedMs = cameraPose.timestampMs;

    const axes = new AxesHelper(220);
    axes.excludeFromShadowToggle = true;
    markerGroup.add(axes);

    markerGroup.add(createCameraFrustumMesh(cameraPose.cameraBusId));

    const label = createLabelSprite(
        markerGroup.userData.labelText,
        getCameraAccentColor(cameraPose.cameraBusId).getStyle(),
    );
    if (label) {
        label.position.set(0, cameraLabelOffset, 0);
        label.excludeFromShadowToggle = true;
        markerGroup.add(label);
    }

    return markerGroup;
}

/**
 * Removes and disposes the marker for a specific camera.
 */
function removeCameraMarker(cameraBusId) {
    const marker = cameraMarkers.get(cameraBusId);
    if (!marker) {
        return;
    }

    if (cameraMarkersGroup) {
        cameraMarkersGroup.remove(marker);
    }
    disposeObject(marker);
    cameraMarkers.delete(cameraBusId);
}

/**
 * Creates or updates the marker for a camera pose.
 */
function upsertCameraMarker(cameraPose) {
    if (!cameraMarkersGroup) {
        return;
    }

    const labelText = getCameraDisplayName(cameraPose);
    const existingMarker = cameraMarkers.get(cameraPose.cameraBusId);
    let marker = existingMarker;

    if (!marker || marker.userData.labelText !== labelText) {
        if (marker) {
            removeCameraMarker(cameraPose.cameraBusId);
        }
        marker = createCameraMarker(cameraPose);
        cameraMarkers.set(cameraPose.cameraBusId, marker);
        cameraMarkersGroup.add(marker);
    }

    marker.userData.lastUpdatedMs = cameraPose.timestampMs;
    cameraCorrectedRotation.copy(cameraPose.transformMatrix);
    applySharedVisualAxisCorrection(cameraCorrectedRotation);
    marker.matrix.copy(cameraCorrectedRotation);
    marker.matrixWorldNeedsUpdate = true;
    marker.visible = allCameraMarkersVisible;
}

/**
 * Removes camera poses that have exceeded the stale timeout.
 */
function pruneStaleCameraPoses(now = Date.now()) {
    for (const [cameraBusId, cameraPose] of pendingCameraPoses.entries()) {
        if (now - cameraPose.timestampMs <= cameraPoseStaleTimeoutMs) {
            continue;
        }
        pendingCameraPoses.delete(cameraBusId);
        removeCameraMarker(cameraBusId);
    }
}

/**
 * Synchronizes all camera markers from the pending pose map.
 */
function syncCameraMarkersFromPending() {
    pruneStaleCameraPoses();
    for (const cameraPose of pendingCameraPoses.values()) {
        upsertCameraMarker(cameraPose);
    }
}

/**
 * Updates the stored pose for a camera and refreshes its marker.
 */

export function updateCameraPose(cameraPoseUpdate) {
    if (
        !cameraPoseUpdate ||
        typeof cameraPoseUpdate.cameraBusId !== "string" ||
        !cameraPoseUpdate.transformMatrix
    ) {
        console.warn("Invalid camera pose update:", cameraPoseUpdate);
        return;
    }

    const normalizedUpdate = {
        cameraBusId: cameraPoseUpdate.cameraBusId,
        cameraName:
            typeof cameraPoseUpdate.cameraName === "string"
                ? cameraPoseUpdate.cameraName
                : cameraPoseUpdate.cameraBusId,
        transformMatrix: cameraPoseUpdate.transformMatrix.clone(),
        timestampMs: Number.isFinite(cameraPoseUpdate.timestampMs)
            ? cameraPoseUpdate.timestampMs
            : Date.now(),
    };

    pendingCameraPoses.set(normalizedUpdate.cameraBusId, normalizedUpdate);

    pruneStaleCameraPoses(normalizedUpdate.timestampMs);
    upsertCameraMarker(normalizedUpdate);
}

/**
 * Initializes the 3D scene, loads assets, and starts rendering.
 */

export async function init3DView(modelUrl, options = {}) {
    const loadToken = currentLoadToken + 1;
    currentLoadToken = loadToken;
    teardown3DView({ invalidateLoads: false });
    const loadingTracker = createLoadingTracker(loadToken);
    loadingTracker.start("setup", "3D controls");

    const container = document.getElementById("view-3d");
    const robotModelToggle = document.getElementById("toggleRobotModelBtn");
    if (robotModelToggle) {
        robotModelVisible = robotModelToggle.checked;
    }
    const camerasToggle = document.getElementById("toggleCamerasBtn");
    if (camerasToggle) {
        allCameraMarkersVisible = camerasToggle.checked;
    }
    statsDisplay = document.getElementById("statsDisplay");
    if (statsDisplay) {
        statsDisplay.style.position = "absolute";
        statsDisplay.style.bottom = "10px";
        statsDisplay.style.right = "10px";
        statsDisplay.style.color = "#f9c84a";
        statsDisplay.style.fontSize = "1rem";
        statsDisplay.style.zIndex = "10";
    }

    const scale = 40;
    currentFieldScaleFactor = normalizeAssetScale(options.fieldScale);
    currentFieldRotationOffset = normalizeRotationOffset(options.fieldRotationOffset);
    currentFieldYear = options.fieldYear || null;
    currentFieldFilename = options.fieldFilename || null;

    await populateRobotDropdown();
    if (loadToken !== currentLoadToken) {
        return;
    }
    loadingTracker.finish("setup");

    scene = new Scene();
    detectedObjectsGroup = new Group();
    detectedObjectsGroup.excludeFromShadowToggle = true;
    scene.add(detectedObjectsGroup);
    cameraMarkersGroup = new Group();
    cameraMarkersGroup.excludeFromShadowToggle = true;
    scene.add(cameraMarkersGroup);
    if (pendingDetectedObjects) {
        renderDetectedObjects(pendingDetectedObjects);
    }
    syncCameraMarkersFromPending();

    const dracoLoader = new DRACOLoader();
    dracoLoader.setDecoderPath(`${BACKEND_BASE_URL}/draco/gltf/`);
    activeDracoLoader = dracoLoader;
    const resolvedModelUrl = buildBackendAssetUrl(modelUrl);

    scene.background = new Color(0x222222);

    /**
     * Returns the scale for the currently selected robot model.
     */
    function selectedRobotScale() {
        const selectedOption = robotFileSelect.selectedOptions?.[0];
        return normalizeAssetScale(
            selectedOption?.dataset.scale || robotFileSelect.value,
        );
    }

    /**
     * Loads a robot model and applies its scale.
     */
    function loadRobot(robotFile, scaleFactor = selectedRobotScale()) {
        if (loadToken !== currentLoadToken) {
            return;
        }
        if (!robotFile) {
            loadingTracker.finish("robot");
            return;
        }

        currentRobotFile = robotFile;
        applyRobotScaleFactor(scaleFactor);
        console.log("Loading robot:", robotFile);
        loadingTracker.start("robot", "robot model");
        try {
            if (robotObject) {
                removeAndDisposeObject(robotObject);
                robotObject = null;
            }

            const robotLoader = new GLTFLoader();
            robotLoader.setDRACOLoader(dracoLoader);

            robotLoader.load(
                `${BACKEND_BASE_URL}/get-robot-file/${robotFile}`,
                (gltf) => {
                    if (loadToken !== currentLoadToken) {
                        disposeObject(gltf.scene);
                        return;
                    }
                    robotObject = gltf.scene;

                    robotObject.traverse((child) => {
                        if (child.isMesh) {
                            child.castShadow = false;
                            child.receiveShadow = false;
                            child.excludeFromShadowToggle = true;
                            child.geometry.computeVertexNormals();

                            // Remove reflective properties from materials
                            if (child.material) {
                                if (Array.isArray(child.material)) {
                                    for (const material of child.material) {
                                        material.metalness = 0;
                                        material.roughness = 1;
                                    }
                                } else {
                                    child.material.metalness = 0;
                                    child.material.roughness = 1;
                                }
                            }
                        }
                    });

                    scene.add(robotObject);
                    applyRobotModelVisibility();
                    refreshRobotMatrix();
                    console.log("Loaded robot:", robotFile);
                    loadingTracker.finish("robot");
                },
                (event) => {
                    loadingTracker.progress("robot", event.loaded, event.total);
                },
                (error) => {
                    loadingTracker.fail(
                        "robot",
                        `Error loading robot ${robotFile}: ${error}`,
                    );
                },
            );
        } catch (error) {
            loadingTracker.fail("robot", `Error loading robot: ${error}`);
        }
    }

    const robotFileSelect = document.getElementById("robotFileSelect");
    let selectedRobotFile = robotFileSelect.value;

    currentRobotLoader = loadRobot;
    loadRobot(selectedRobotFile);

    if (!robotFileSelectListenerAttached) {
        robotFileSelect.addEventListener("change", () => {
            selectedRobotFile = robotFileSelect.value;
            if (currentRobotLoader) {
                currentRobotLoader(selectedRobotFile, selectedRobotScale());
            }
        });
        robotFileSelectListenerAttached = true;
    }

    camera = new PerspectiveCamera(
        75,
        container.clientWidth / container.clientHeight,
        100,
        40000,
    );
    camera.position.set(100 * scale, 100 * scale, 100 * scale);

    renderer = new WebGLRenderer({
        antialias: true,
        powerPreference: "default",
    });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = PCFSoftShadowMap;
    renderer.domElement.style.width = "100%";
    renderer.domElement.style.height = "100%";
    renderer.domElement.style.display = "block";
    renderer.domElement.classList.add(
        "absolute",
        "top-0",
        "left-0",
        "w-full",
        "h-full",
        "rounded-inherit",
        "-z-10",
        "block",
    );
    container.appendChild(renderer.domElement);

    controls = new OrbitControls(camera, renderer.domElement);

    scene.add(new AmbientLight(0xffffff, 0.2));

    directionalLight = new DirectionalLight(0xffffff, 2);
    directionalLight.position.set(100 * scale, 200 * scale, 200 * scale);
    directionalLight.castShadow = true;
    directionalLight.shadow.bias = -0.0005;
    directionalLight.shadow.normalBias = -0.0005;
    directionalLight.shadow.mapSize.width = 2048;
    directionalLight.shadow.mapSize.height = 2048;
    directionalLight.shadow.camera.left = -300 * scale;
    directionalLight.shadow.camera.right = 300 * scale;
    directionalLight.shadow.camera.top = 150 * scale;
    directionalLight.shadow.camera.bottom = -150 * scale;
    directionalLight.shadow.camera.near = 100 * scale;
    directionalLight.shadow.camera.far = 500 * scale;
    scene.add(directionalLight);

    const fieldLoader = new GLTFLoader();
    fieldLoader.setDRACOLoader(dracoLoader);
    loadingTracker.start("field", "field model");
    fieldLoader.load(
        resolvedModelUrl,
        (gltf) => {
            if (loadToken !== currentLoadToken) {
                disposeObject(gltf.scene);
                return;
            }
            const model = gltf.scene;
            fieldObject = model;

            const modelBounds = new Box3().setFromObject(model);
            const modelSize = new Vector3();
            const modelCenter = new Vector3();
            modelBounds.getSize(modelSize);
            modelBounds.getCenter(modelCenter);
            console.log("FIELD MODEL BOUNDS", {
                size: modelSize,
                center: modelCenter,
            });

            applyFieldTransform(model);

            model.traverse((child) => {
                if (child.isMesh) {
                    child.castShadow = true;
                    child.receiveShadow = true;
                    child.geometry.computeVertexNormals();
                }
            });
            scene.add(model);

            // Disable shadow map auto updates after initial generation for performance
            renderer.shadowMap.autoUpdate = false;
            // Force initial shadow map generation
            renderer.shadowMap.needsUpdate = true;

            startAnimationLoop();
            loadingTracker.finish("field");
        },
        (event) => {
            loadingTracker.progress("field", event.loaded, event.total);
        },
        (error) => {
            console.error("Error loading the model:", error);
            startAnimationLoop();
            loadingTracker.fail("field", `Error loading field model: ${error}`);
        },
    );

    gamePieces = [];
    gamePieceObjects = [];
    const gamePieceUrls = Array.isArray(options.gamePieceUrls)
        ? options.gamePieceUrls
        : [
              resolvedModelUrl.split("/").slice(0, -2).join("/") +
                  "/game_pieces/" +
                  resolvedModelUrl.split("/").pop().slice(0, 7) +
                  "-GP.glb",
          ];

    if (gamePieceUrls.length > 0) {
        const gpLoader = new GLTFLoader();
        gpLoader.setDRACOLoader(dracoLoader);
        let pendingGamePieces = gamePieceUrls.length;
        const gamePieceProgress = new Map();

        loadingTracker.start("gamePieces", "game pieces");

        /**
         * Aggregates progress across all game-piece loads.
         */
        function updateGamePieceProgress(url, loaded, total) {
            gamePieceProgress.set(url, {
                loaded: Number.isFinite(loaded) ? loaded : 0,
                total: Number.isFinite(total) ? total : 0,
            });
            const progressValues = Array.from(gamePieceProgress.values());
            const loadedBytes = progressValues.reduce(
                (sum, progressValue) => sum + progressValue.loaded,
                0,
            );
            const totalBytes = progressValues.reduce(
                (sum, progressValue) => sum + progressValue.total,
                0,
            );
            loadingTracker.progress("gamePieces", loadedBytes, totalBytes);
        }

        /**
         * Finalizes a single game-piece load.
         */
        function finishGamePieceLoad() {
            pendingGamePieces -= 1;
            if (pendingGamePieces === 0) {
                loadingTracker.finish("gamePieces");
            }
        }

        gamePieceUrls.forEach((gamePieceUrl) => {
            const resolvedGamePieceUrl = buildBackendAssetUrl(gamePieceUrl);
            gpLoader.load(
                resolvedGamePieceUrl,
                (gltf) => {
                    if (loadToken !== currentLoadToken) {
                        disposeObject(gltf.scene);
                        finishGamePieceLoad();
                        return;
                    }
                    const model = gltf.scene;

                    gamePieceObjects.push(model);
                    applyFieldTransform(model);

                    model.traverse((child) => {
                        if (child.isMesh) {
                            child.castShadow = true;
                            child.receiveShadow = true;
                            child.geometry.computeVertexNormals();
                            child.visible = gamePiecesVisible;
                            gamePieces.push(child);
                        }
                    });
                    scene.add(model);
                    finishGamePieceLoad();
                },
                (event) => {
                    updateGamePieceProgress(
                        resolvedGamePieceUrl,
                        event.loaded,
                        event.total,
                    );
                },
                (error) => {
                    console.error(
                        `Error loading game piece ${resolvedGamePieceUrl}: ${error}`,
                    );
                    finishGamePieceLoad();
                },
            );
        });
    }

    if (!globalThis.__eev_gamePiecesToggleAttached) {
        document
            .getElementById("toggleGamePiecesBtn")
            .addEventListener("change", (event) => {
                gamePiecesVisible = event.target.checked;
                for (const gp of gamePieces) {
                    gp.visible = gamePiecesVisible;
                }
            });
        globalThis.__eev_gamePiecesToggleAttached = true;
    }

    if (!globalThis.__eev_robotModelToggleAttached) {
        const robotToggleEl = document.getElementById("toggleRobotModelBtn");
        if (robotToggleEl) {
            robotToggleEl.addEventListener("change", (event) => {
                robotModelVisible = event.target.checked;
                applyRobotModelVisibility();
            });
        }
        globalThis.__eev_robotModelToggleAttached = true;
    }

    if (!globalThis.__eev_camerasToggleAttached) {
        const camerasToggleEl = document.getElementById("toggleCamerasBtn");
        if (camerasToggleEl) {
            camerasToggleEl.addEventListener("change", (event) => {
                allCameraMarkersVisible = event.target.checked;
                for (const marker of cameraMarkers.values()) {
                    marker.visible = allCameraMarkersVisible;
                }
            });
        }
        globalThis.__eev_camerasToggleAttached = true;
    }

    const clock = new Clock();
    let delta = 0;

    /**
     * Starts the render loop when the view is visible and assets are ready.
     */
    function startAnimationLoop() {
        const container = document.getElementById("view-3d");
        const isViewVisible =
            container && !container.classList.contains("hidden");

        if (animationStarted && isViewVisible) return;

        if (
            isViewVisible &&
            loadToken === currentLoadToken &&
            renderer &&
            scene &&
            camera
        ) {
            animationStarted = true;
            animate();
        }
    }

    /**
     * Renders animation frames while the 3D view remains active.
     */
    function animate() {
        const container = document.getElementById("view-3d");
        const isViewVisible =
            container && !container.classList.contains("hidden");

        if (
            isViewVisible &&
            loadToken === currentLoadToken &&
            renderer &&
            scene &&
            camera
        ) {
            animationFrameId = requestAnimationFrame(animate);

            delta += clock.getDelta();

            if (delta >= interval) {
                pruneStaleCameraPoses();
                renderer.render(scene, camera);
                updateStats();
                delta = delta % interval;
            }
        } else {
            stopAnimationLoop();
        }
    }

    if (!resizeHandler) {
        resizeHandler = () => {
            const activeContainer = document.getElementById("view-3d");
            if (!activeContainer || !camera || !renderer) {
                return;
            }
            const width = activeContainer.clientWidth;
            const height = activeContainer.clientHeight;
            camera.aspect = width / height;
            camera.updateProjectionMatrix();
            renderer.setSize(width, height);
        };
        globalThis.addEventListener("resize", resizeHandler);
    }

    if (!globalThis.__eev_shadowToggleAttached) {
        document
            .getElementById("toggleShadowBtn")
            .addEventListener("change", (event) => {
                if (!scene || !directionalLight || !renderer) {
                    return;
                }
                shadowsEnabled = event.target.checked;
                scene.traverse((object) => {
                    if (object.isMesh && !object.excludeFromShadowToggle) {
                        object.castShadow = shadowsEnabled;
                        object.receiveShadow = shadowsEnabled;
                    }
                });
                directionalLight.castShadow = shadowsEnabled;
                renderer.shadowMap.enabled = shadowsEnabled;

                // Force shadow map update when shadows are re-enabled
                if (shadowsEnabled) {
                    renderer.shadowMap.needsUpdate = true;
                }
            });
        globalThis.__eev_shadowToggleAttached = true;
    }

    const aprilTagMapUrl = options.aprilTagMapUrl || "/frc2025r2.json";

    // Add AprilTag images as planes at fiducial transforms
    loadingTracker.start("apriltags", "AprilTags");
    fetch(buildBackendAssetUrl(aprilTagMapUrl))
        .then((response) => response.json())
        .then((json) => {
            if (loadToken !== currentLoadToken) {
                return;
            }
            const textureLoader = new TextureLoader();
            const fiducials = Array.isArray(json.fiducials)
                ? json.fiducials
                : [];
            if (fiducials.length === 0) {
                loadingTracker.finish("apriltags");
                return;
            }
            let remainingTags = fiducials.length;
            let loadedTags = 0;
            let aprilTagLoadFailed = false;
            const finishTag = () => {
                remainingTags -= 1;
                loadedTags += 1;
                loadingTracker.progress(
                    "apriltags",
                    loadedTags,
                    fiducials.length,
                );
                if (remainingTags === 0) {
                    if (aprilTagLoadFailed) {
                        loadingTracker.fail(
                            "apriltags",
                            "One or more AprilTag images failed to load.",
                        );
                    } else {
                        loadingTracker.finish("apriltags");
                    }
                }
            };
            for (const fiducial of fiducials) {
                const tagId = fiducial.id;
                const tagImageName = `tag36_11_${String(tagId).padStart(5, "0")}.webp`;
                const tagImagePath = `${BACKEND_BASE_URL}/src/webui/assets/apriltags/${tagImageName}`;
                textureLoader.load(
                    tagImagePath,
                    (texture) => {
                        if (loadToken !== currentLoadToken) {
                            texture.dispose();
                            finishTag();
                            return;
                        }
                        // Configure texture for crisp pixel art
                        texture.magFilter = NearestFilter;
                        texture.minFilter = NearestFilter;
                        texture.generateMipmaps = false;

                        const planeGeometry = new PlaneGeometry(
                            fiducial.size,
                            fiducial.size,
                        );
                        const planeMaterial = new MeshStandardMaterial({
                            map: texture,
                        });
                        const plane = new Mesh(planeGeometry, planeMaterial);
                        // Apply 4x4 transform from JSON
                        const t = fiducial.transform;
                        // Three.js uses column-major, so set matrix directly
                        const matrix = new Matrix4();
                        matrix.set(
                            t[0],
                            t[1],
                            t[2],
                            t[3] * 1000,
                            t[4],
                            t[5],
                            t[6],
                            t[7] * 1000,
                            t[8],
                            t[9],
                            t[10],
                            t[11] * 1000,
                            t[12],
                            t[13],
                            t[14],
                            t[15],
                        );

                        const rotationYMatrix = new Matrix4();
                        rotationYMatrix.makeRotationY(Math.PI / 2);
                        const rotationXMatrix = new Matrix4();
                        rotationXMatrix.makeRotationX(-Math.PI / 2);
                        matrix.premultiply(rotationXMatrix);
                        matrix.multiply(rotationYMatrix);

                        plane.applyMatrix4(matrix);

                        // Move plane 1 unit along its world normal
                        const normal = new Vector3();
                        matrix.extractBasis(
                            new Vector3(),
                            new Vector3(),
                            normal,
                        );
                        normal.normalize();
                        plane.position.add(normal);

                        plane.castShadow = false;
                        plane.receiveShadow = false;
                        plane.excludeFromShadowToggle = true;
                        scene.add(plane);
                        finishTag();
                    },
                    undefined,
                    (error) => {
                        aprilTagLoadFailed = true;
                        console.error(
                            `Error loading AprilTag ${tagId}:`,
                            error,
                        );
                        finishTag();
                    },
                );
            }
        })
        .catch((error) => {
            loadingTracker.fail(
                "apriltags",
                `Error loading AprilTag field data: ${error}`,
            );
        });
}

/**
 * Updates the stored robot transform and refreshes the robot model.
 */

export function updateRobotTransform(transformMatrix) {
    lastRobotTransformMatrix = transformMatrix.clone();
    if (robotObject) {
        refreshRobotMatrix();
    }
}
