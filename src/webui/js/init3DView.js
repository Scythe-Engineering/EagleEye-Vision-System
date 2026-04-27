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
} from "three";
import { OrbitControls } from "OrbitControls";
import { DRACOLoader } from "DRACOLoader";
import { populateRobotDropdown } from "./dropdown/robotDropdown.js";
import { BACKEND_BASE_URL } from "./config.js";
import { position3DToFieldSpaceVector } from "./utils/fieldSpaceTransforms.js";

let renderer, scene, camera, directionalLight;
let shadowsEnabled = true;
let gamePiecesVisible = true;
let statsDisplay;
let frameCount = 0;
let lastTime = performance.now();
let robotObject = null;
let fieldObject = null;
let robotAxes = null;
let animationStarted = false;
let detectedObjectsGroup = null;
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
let currentFieldYear = null;
let currentFieldFilename = null;

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

function isAbsoluteUrl(url) {
    return (
        /^[a-z][a-z\d+\-.]*:\/\//i.test(url) ||
        url.startsWith("data:") ||
        url.startsWith("blob:")
    );
}

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

function normalizeAssetScale(scale) {
    const numericScale = Number.parseFloat(scale);
    return Number.isFinite(numericScale) && numericScale > 0 ? numericScale : 1;
}

function updateRobotScaleMatrix() {
    const scale = robotBaseScale * currentRobotScaleFactor;
    robotScaleMatrix.makeScale(scale, scale, scale);
}

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

function applySharedVisualAxisCorrection(matrix) {
    return matrix
        .multiply(robotPitchRollSwap)
        .multiply(visualOrientationMatrix)
        .multiply(extraVisualRollMatrix)
        .multiply(robotPitchRollSwapInverse);
}

function applyRobotScaleFactor(scale) {
    currentRobotScaleFactor = normalizeAssetScale(scale);
    updateRobotScaleMatrix();
    refreshRobotMatrix();
}

function applyFieldScaleFactor(scale) {
    currentFieldScaleFactor = normalizeAssetScale(scale);
    if (fieldObject) {
        fieldObject.scale.set(
            currentFieldScaleFactor,
            currentFieldScaleFactor,
            currentFieldScaleFactor,
        );
    }

    for (const gamePieceObject of gamePieceObjects) {
        gamePieceObject.scale.set(
            currentFieldScaleFactor,
            currentFieldScaleFactor,
            currentFieldScaleFactor,
        );
    }

    if (renderer?.shadowMap) {
        renderer.shadowMap.needsUpdate = true;
    }
}

export function apply3DAssetScale(assetType, asset, scale) {
    if (assetType === "robot" && asset?.filename === currentRobotFile) {
        applyRobotScaleFactor(scale);
        return;
    }

    if (
        assetType === "field" &&
        asset?.year === currentFieldYear &&
        asset?.filename === currentFieldFilename
    ) {
        applyFieldScaleFactor(scale);
    }
}

function getLoadingElements() {
    return {
        overlay: document.getElementById("threeDLoadingOverlay"),
        status: document.getElementById("threeDLoadingStatus"),
        progress: document.getElementById("threeDLoadingProgress"),
    };
}

function createLoadingTracker(token) {
    const pendingTasks = new Map();
    const taskProgress = new Map();
    let failedCount = 0;

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

    function start(key, label) {
        pendingTasks.set(key, label);
        taskProgress.set(key, 0);
        update();
    }

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

    function finish(key) {
        pendingTasks.delete(key);
        taskProgress.delete(key);
        update();
    }

    function fail(key, errorMessage) {
        failedCount += 1;
        pendingTasks.delete(key);
        taskProgress.delete(key);
        console.error(errorMessage);
        update();
    }

    return { start, progress, finish, fail };
}

function updateStats() {
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

function removeAndDisposeObject(object) {
    if (!object) {
        return;
    }
    object.parent?.remove(object);
    disposeObject(object);
}

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

function normalizeDetectionPosition(position) {
    return position3DToFieldSpaceVector(position);
}

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

export function updateDetectedObjects(detections) {
    pendingDetectedObjects = detections;
    renderDetectedObjects(detections);
}

function getCameraListElements() {
    return {
        list: document.getElementById("cameraPoseList"),
        emptyState: document.getElementById("cameraPoseListEmpty"),
        allCamerasButton: document.getElementById("toggleAllCamerasBtn"),
    };
}

function applyRobotModelVisibility() {
    if (robotObject) {
        robotObject.visible = robotModelVisible;
    }
}

function updateAllCamerasToggleButton() {
    const { allCamerasButton } = getCameraListElements();
    if (!allCamerasButton) {
        return;
    }
    const hasCameras = pendingCameraPoses.size > 0;
    allCamerasButton.classList.toggle("hidden", !hasCameras);
    if (hasCameras) {
        allCamerasButton.textContent = allCameraMarkersVisible
            ? "Hide cameras"
            : "Show cameras";
    }
}

function getCameraDisplayName(cameraPose) {
    return String(cameraPose.cameraName || cameraPose.cameraBusId);
}

function getHueFromIdentifier(identifier) {
    const key = String(identifier ?? "camera");
    let hash = 0;
    for (let index = 0; index < key.length; index += 1) {
        hash = (hash * 31 + key.charCodeAt(index)) % 360;
    }
    return (hash % 360) / 360;
}

function getCameraAccentColor(cameraBusId) {
    const color = new Color();
    color.setHSL(getHueFromIdentifier(cameraBusId), 0.72, 0.58);
    return color;
}

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

function pruneStaleCameraPoses(now = Date.now()) {
    let removedAny = false;
    for (const [cameraBusId, cameraPose] of pendingCameraPoses.entries()) {
        if (now - cameraPose.timestampMs <= cameraPoseStaleTimeoutMs) {
            continue;
        }
        pendingCameraPoses.delete(cameraBusId);
        removeCameraMarker(cameraBusId);
        removedAny = true;
    }

    if (removedAny) {
        renderCameraVisibilityList();
    }
}

function renderCameraVisibilityList() {
    const { list, emptyState } = getCameraListElements();
    if (!list || !emptyState) {
        return;
    }

    list.replaceChildren();

    const cameraEntries = Array.from(pendingCameraPoses.values()).sort((left, right) =>
        getCameraDisplayName(left).localeCompare(getCameraDisplayName(right)),
    );

    emptyState.classList.toggle("hidden", cameraEntries.length > 0);
    list.classList.toggle("hidden", cameraEntries.length === 0);

    for (const cameraPose of cameraEntries) {
        const row = document.createElement("div");
        row.className = "text-xs text-[#e8e8e8] truncate";
        row.textContent = getCameraDisplayName(cameraPose);
        list.appendChild(row);
    }

    updateAllCamerasToggleButton();
}

function syncCameraMarkersFromPending() {
    pruneStaleCameraPoses();
    for (const cameraPose of pendingCameraPoses.values()) {
        upsertCameraMarker(cameraPose);
    }
    renderCameraVisibilityList();
}

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
    renderCameraVisibilityList();
}

export async function init3DView(modelUrl, options = {}) {
    const loadToken = currentLoadToken + 1;
    currentLoadToken = loadToken;
    const loadingTracker = createLoadingTracker(loadToken);
    loadingTracker.start("setup", "3D controls");

    const container = document.getElementById("view-3d");
    const robotModelToggle = document.getElementById("toggleRobotModelBtn");
    if (robotModelToggle) {
        robotModelVisible = robotModelToggle.checked;
    }
    statsDisplay = document.getElementById("statsDisplay");
    statsDisplay.style.position = "absolute";
    statsDisplay.style.bottom = "10px";
    statsDisplay.style.right = "10px";
    statsDisplay.style.color = "#f9c84a";
    statsDisplay.style.fontSize = "1rem";
    statsDisplay.style.zIndex = "10";

    const scale = 40;
    currentFieldScaleFactor = normalizeAssetScale(options.fieldScale);
    currentFieldYear = options.fieldYear || null;
    currentFieldFilename = options.fieldFilename || null;

    await populateRobotDropdown();
    loadingTracker.finish("setup");

    // Clear and destroy existing scene if it exists
    if (scene) {
        clearDetectedObjectsGroup();
        clearCameraMarkersGroup();
        // Remove all objects from the scene
        while (scene.children.length > 0) {
            const child = scene.children[0];
            scene.remove(child);
            disposeObject(child);
        }

        // Clear the scene
        scene.clear();
        scene = null;
        robotObject = null;
        fieldObject = null;
        detectedObjectsGroup = null;
        cameraMarkersGroup = null;
        gamePieceObjects = [];
        gamePieces = [];

        // Dispose and cleanup existing WebGLRenderer to prevent context leaks
        if (renderer) {
            // Force WebGL context loss if method exists
            if (renderer.forceContextLoss) {
                renderer.forceContextLoss();
            }

            // Dispose of the renderer
            renderer.dispose();

            // Remove canvas element from DOM
            if (renderer.domElement && renderer.domElement.parentNode) {
                renderer.domElement.parentNode.removeChild(renderer.domElement);
            }

            // Null out renderer reference
            renderer = null;
        }
    }

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
    const resolvedModelUrl = buildBackendAssetUrl(modelUrl);

    scene.background = new Color(0x222222);

    function selectedRobotScale() {
        const selectedOption = robotFileSelect.selectedOptions?.[0];
        return normalizeAssetScale(
            selectedOption?.dataset.scale || robotFileSelect.value,
        );
    }

    function loadRobot(robotFile, scaleFactor = selectedRobotScale()) {
        if (loadToken !== currentLoadToken) {
            return;
        }
        if (!robotFile) {
            loadingTracker.finish("robot");
            return;
        }

        currentRobotFile = robotFile;
        lastRobotTransformMatrix = null;
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
                    loadingTracker.progress(
                        "robot",
                        event.loaded,
                        event.total,
                    );
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
        powerPreference: "high-performance",
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

    const controls = new OrbitControls(camera, renderer.domElement);

    scene.add(new AmbientLight(0xffffff, 0.2));

    directionalLight = new DirectionalLight(0xffffff, 2);
    directionalLight.position.set(100 * scale, 200 * scale, 200 * scale);
    directionalLight.castShadow = true;
    directionalLight.shadow.bias = -0.0005;
    directionalLight.shadow.normalBias = -0.0005;
    directionalLight.shadow.mapSize.width = 1024 * 3;
    directionalLight.shadow.mapSize.height = 1024 * 3;
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

            model.rotation.x = Math.PI / 2;
            model.scale.set(
                currentFieldScaleFactor,
                currentFieldScaleFactor,
                currentFieldScaleFactor,
            );

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

                    model.rotation.x = Math.PI / 2;
                    gamePieceObjects.push(model);
                    model.scale.set(
                        currentFieldScaleFactor,
                        currentFieldScaleFactor,
                        currentFieldScaleFactor,
                    );

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

    if (!globalThis.__eev_allCamerasToggleAttached) {
        const allCamerasBtn = document.getElementById("toggleAllCamerasBtn");
        if (allCamerasBtn) {
            allCamerasBtn.addEventListener("click", () => {
                allCameraMarkersVisible = !allCameraMarkersVisible;
                for (const marker of cameraMarkers.values()) {
                    marker.visible = allCameraMarkersVisible;
                }
                updateAllCamerasToggleButton();
            });
        }
        globalThis.__eev_allCamerasToggleAttached = true;
    }

    let clock = new Clock();
    let delta = 0;

    function startAnimationLoop() {
        const container = document.getElementById("view-3d");
        const isViewVisible =
            container && !container.classList.contains("hidden");

        if (animationStarted && isViewVisible) return;

        if (isViewVisible) {
            animationStarted = true;
            animate();
        }
    }

    function animate() {
        const container = document.getElementById("view-3d");
        const isViewVisible =
            container && !container.classList.contains("hidden");

        if (isViewVisible) {
            requestAnimationFrame(animate);

            delta += clock.getDelta();

            if (delta >= interval) {
                pruneStaleCameraPoses();
                renderer.render(scene, camera);
                updateStats();
                delta = delta % interval;
            }
        } else {
            animationStarted = false;
        }
    }

    if (!globalThis.__eev_resizeAttached) {
        const onResize = () => {
            const width = container.clientWidth;
            const height = container.clientHeight;
            camera.aspect = width / height;
            camera.updateProjectionMatrix();
            renderer.setSize(width, height);
        };
        globalThis.addEventListener("resize", onResize);
        globalThis.__eev_resizeAttached = true;
    }

    if (!globalThis.__eev_shadowToggleAttached) {
        document
            .getElementById("toggleShadowBtn")
            .addEventListener("change", (event) => {
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

export function updateRobotTransform(transformMatrix) {
    if (robotObject) {
        lastRobotTransformMatrix = transformMatrix.clone();
        refreshRobotMatrix();
    } else {
        console.warn("Robot not initialized yet");
    }
}
