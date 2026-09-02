// Utilities for camera configuration UI, including intrinsics, extrinsics, and calibration flows.
import { BACKEND_BASE_URL } from "../config.js";
import {
    showDanger,
    showSuccess,
    showUploadToast,
    showWarning,
} from "../ui/notificationSystem.js";
import { uploadWithProgress } from "../ui/uploadWithProgress.js";
import * as THREE from "three";
import { OrbitControls } from "OrbitControls";

const EXTRINSICS_KEYS = [
    "pitch",
    "yaw",
    "roll",
    "x_offset",
    "y_offset",
    "z_offset",
];

const INPUT_ID_BY_KEY = {
    pitch: "utils-pitch",
    yaw: "utils-yaw",
    roll: "utils-roll",
    x_offset: "utils-x-offset",
    y_offset: "utils-y-offset",
    z_offset: "utils-z-offset",
};

let initialized = false;
let currentCameraBusId = "";
let currentCalibrationStreamName = "";
let poseVizState = null;
let poseVizRenderFrame = null;
let calibrationModalOpen = false;
let distortionModalOpen = false;
let selectedCalibrationFrameIndex = null;
let calibrationHistoryFrames = [];

const DEG_TO_RAD = Math.PI / 180;
const POSE_VIZ_MAX_FPS = 30;
const POSE_VIZ_FRAME_MS = 1000 / POSE_VIZ_MAX_FPS;

/**
 * Returns a DOM element by id.
 * @param {string} id - Element id.
 * @returns {HTMLElement | null}
 */
function getElement(id) {
    return document.getElementById(id);
}

/**
 * Creates a text sprite label for the 3D view.
 * @param {string} text - Label text.
 * @param {string} [color="#f9c845"] - Text color.
 * @returns {THREE.Sprite | null}
 */
function createLabelSprite(text, color = "#f9c845") {
    const canvas = document.createElement("canvas");
    canvas.width = 256;
    canvas.height = 80;
    const context = canvas.getContext("2d");
    if (!context) {
        return null;
    }

    context.clearRect(0, 0, canvas.width, canvas.height);
    context.font = "600 30px sans-serif";
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillStyle = color;
    context.fillText(text, canvas.width / 2, canvas.height / 2);

    const texture = new THREE.CanvasTexture(canvas);
    texture.needsUpdate = true;
    const material = new THREE.SpriteMaterial({
        map: texture,
        transparent: true,
        depthTest: false,
    });
    const sprite = new THREE.Sprite(material);
    sprite.scale.set(0.45, 0.14, 1);
    return sprite;
}

/**
 * Initializes the 3D camera pose visualization.
 */
function initCameraPoseVisualization() {
    if (poseVizState) {
        startCameraPoseVisualizationLoop();
        return;
    }

    const container = getElement("utilsCameraPoseViz");
    if (!container) {
        return;
    }

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x232323);

    const width = Math.max(container.clientWidth, 1);
    const height = Math.max(container.clientHeight, 1);
    const camera = new THREE.PerspectiveCamera(50, width / height, 0.01, 100);
    camera.position.set(1.2, 1.0, 1.2);

    const renderer = new THREE.WebGLRenderer({
        antialias: true,
        powerPreference: "default",
    });
    renderer.setPixelRatio(Math.min(globalThis.devicePixelRatio || 1, 1.5));
    renderer.setSize(width, height);
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 0.4;
    controls.maxDistance = 4;
    controls.target.set(0, 0.1, 0);

    scene.add(new THREE.AmbientLight(0xffffff, 0.65));
    const directional = new THREE.DirectionalLight(0xffffff, 0.8);
    directional.position.set(2, 3, 2);
    scene.add(directional);

    const grid = new THREE.GridHelper(2, 20, 0x666666, 0x3a3a3a);
    grid.position.y = -0.001;
    scene.add(grid);

    const robotGroup = new THREE.Group();
    const chassis = new THREE.Mesh(
        new THREE.BoxGeometry(0.72, 0.12, 0.72),
        new THREE.MeshStandardMaterial({
            color: 0x4a5568,
            metalness: 0.2,
            roughness: 0.7,
        }),
    );
    chassis.position.y = 0.06;
    robotGroup.add(chassis);

    const bumperFront = new THREE.Mesh(
        new THREE.BoxGeometry(0.03, 0.14, 0.72),
        new THREE.MeshStandardMaterial({ color: 0xf9c845 }),
    );
    bumperFront.position.set(0.375, 0.07, 0);
    robotGroup.add(bumperFront);

    const frontLabel = createLabelSprite("FRONT");
    if (frontLabel) {
        frontLabel.position.set(0.48, 0.22, 0);
        robotGroup.add(frontLabel);
    }

    const wheelGeometry = new THREE.CylinderGeometry(0.05, 0.05, 0.04, 16);
    const wheelMaterial = new THREE.MeshStandardMaterial({ color: 0x1f1f1f });
    const wheelOffsets = [
        [0.27, 0.05, 0.32],
        [0.27, 0.05, -0.32],
        [-0.27, 0.05, 0.32],
        [-0.27, 0.05, -0.32],
    ];
    wheelOffsets.forEach((offset) => {
        const wheel = new THREE.Mesh(wheelGeometry, wheelMaterial);
        wheel.rotation.z = Math.PI / 2;
        wheel.position.set(offset[0], offset[1], offset[2]);
        robotGroup.add(wheel);
    });

    scene.add(robotGroup);

    const cameraMarkerGroup = new THREE.Group();
    const cameraBody = new THREE.Mesh(
        new THREE.BoxGeometry(0.08, 0.05, 0.05),
        new THREE.MeshStandardMaterial({ color: 0x38bdf8 }),
    );
    cameraMarkerGroup.add(cameraBody);

    const lens = new THREE.Mesh(
        new THREE.ConeGeometry(0.018, 0.05, 16),
        new THREE.MeshStandardMaterial({ color: 0x0ea5e9 }),
    );
    lens.rotation.z = -Math.PI / 2;
    lens.position.set(0.06, 0, 0);
    cameraMarkerGroup.add(lens);

    const cameraAxes = new THREE.AxesHelper(0.2);
    cameraMarkerGroup.add(cameraAxes);

    const cameraLabel = createLabelSprite("CAMERA", "#7dd3fc");
    if (cameraLabel) {
        cameraLabel.position.set(0, 0.15, 0);
        cameraMarkerGroup.add(cameraLabel);
    }

    scene.add(cameraMarkerGroup);

    poseVizState = {
        container,
        scene,
        camera,
        renderer,
        controls,
        cameraMarkerGroup,
        frameHandle: 0,
        lastFrameMs: 0,
    };

    poseVizRenderFrame = (timestampMs = performance.now()) => {
        if (!poseVizState) {
            return;
        }

        const utilsView = getElement("view-utils");
        if (utilsView?.classList.contains("hidden")) {
            poseVizState.frameHandle = 0;
            return;
        }

        if (timestampMs - poseVizState.lastFrameMs < POSE_VIZ_FRAME_MS) {
            poseVizState.frameHandle =
                globalThis.requestAnimationFrame(poseVizRenderFrame);
            return;
        }

        poseVizState.lastFrameMs = timestampMs;
        poseVizState.controls.update();
        poseVizState.renderer.render(poseVizState.scene, poseVizState.camera);
        poseVizState.frameHandle =
            globalThis.requestAnimationFrame(poseVizRenderFrame);
    };

    startCameraPoseVisualizationLoop();
}

/**
 * Starts the pose visualization animation loop if needed.
 */
function startCameraPoseVisualizationLoop() {
    if (!poseVizState || poseVizState.frameHandle || !poseVizRenderFrame) {
        return;
    }

    poseVizState.frameHandle =
        globalThis.requestAnimationFrame(poseVizRenderFrame);
}

/**
 * Resizes the pose visualization renderer to match its container.
 */
function resizeCameraPoseVisualization() {
    if (!poseVizState) {
        return;
    }

    const width = Math.max(poseVizState.container.clientWidth, 1);
    const height = Math.max(poseVizState.container.clientHeight, 1);
    poseVizState.camera.aspect = width / height;
    poseVizState.camera.updateProjectionMatrix();
    poseVizState.renderer.setSize(width, height);
}

/**
 * Updates the pose visualization from the current extrinsics inputs.
 */
function updateCameraPoseVisualization() {
    if (!poseVizState) {
        return;
    }

    const xOffset = Number.parseFloat(
        getElement("utils-x-offset")?.value ?? "0",
    );
    const yOffset = Number.parseFloat(
        getElement("utils-y-offset")?.value ?? "0",
    );
    const zOffset = Number.parseFloat(
        getElement("utils-z-offset")?.value ?? "0",
    );
    const yawDegrees = Number.parseFloat(getElement("utils-yaw")?.value ?? "0");
    const pitchDegrees = Number.parseFloat(
        getElement("utils-pitch")?.value ?? "0",
    );
    const rollDegrees = Number.parseFloat(
        getElement("utils-roll")?.value ?? "0",
    );

    const x = Number.isFinite(xOffset) ? xOffset : 0;
    const y = Number.isFinite(yOffset) ? yOffset : 0;
    const z = Number.isFinite(zOffset) ? zOffset : 0;
    const yaw = (Number.isFinite(yawDegrees) ? yawDegrees : 0) * DEG_TO_RAD;
    const pitch =
        (Number.isFinite(pitchDegrees) ? pitchDegrees : 0) * DEG_TO_RAD;
    const roll = (Number.isFinite(rollDegrees) ? rollDegrees : 0) * DEG_TO_RAD;

    // Map robot convention (x forward, y left, z up) -> Three.js (x right, y up, z depth).
    poseVizState.cameraMarkerGroup.position.set(x, z, y);

    const qRoll = new THREE.Quaternion().setFromAxisAngle(
        new THREE.Vector3(1, 0, 0),
        roll,
    );
    const qPitch = new THREE.Quaternion().setFromAxisAngle(
        new THREE.Vector3(0, 0, 1),
        pitch,
    );
    const qYaw = new THREE.Quaternion().setFromAxisAngle(
        new THREE.Vector3(0, 1, 0),
        yaw,
    );
    const orientation = new THREE.Quaternion();
    orientation.multiplyQuaternions(qYaw, qPitch);
    orientation.multiply(qRoll);
    poseVizState.cameraMarkerGroup.quaternion.copy(orientation);
}

/**
 * Builds the extrinsics payload from the current form inputs.
 * @returns {{pitch: number, yaw: number, roll: number, x_offset: number, y_offset: number, z_offset: number}}
 */
function getExtrinsicsPayload() {
    const payload = {};
    EXTRINSICS_KEYS.forEach((key) => {
        const input = getElement(INPUT_ID_BY_KEY[key]);
        const numericValue = Number.parseFloat(input?.value ?? "0");
        payload[key] = Number.isFinite(numericValue) ? numericValue : 0;
    });
    return payload;
}

/**
 * Populates extrinsics inputs from a config object.
 * @param {Object} extrinsics - Extrinsics values.
 */
function setExtrinsicsInputs(extrinsics = {}) {
    EXTRINSICS_KEYS.forEach((key) => {
        const input = getElement(INPUT_ID_BY_KEY[key]);
        if (!input) {
            return;
        }
        const value = extrinsics[key];
        input.value = Number.isFinite(value) ? String(value) : "0";
    });
    updateCameraPoseVisualization();
}

/**
 * Updates the intrinsics status text.
 * @param {string} text - Status message.
 */
function setIntrinsicsStatus(text) {
    const status = getElement("utilsIntrinsicsStatus");
    if (status) {
        status.textContent = text;
    }
}

/**
 * Updates the selected camera metadata display.
 * @param {{name: string, bus_id: string} | null} camera - Selected camera.
 */
function setCameraMeta(camera) {
    const meta = getElement("utilsCameraMeta");
    if (!meta) {
        return;
    }

    if (!camera) {
        meta.textContent = "";
        return;
    }

    meta.textContent = `Selected: ${camera.name} (bus_id: ${camera.bus_id})`;
}

/**
 * Fetches JSON from the backend and throws on HTTP errors.
 * @param {string} path - Backend path.
 * @param {RequestInit} [options={}] - Fetch options.
 * @returns {Promise<any>}
 */
async function fetchJson(path, options = {}) {
    const response = await fetch(`${BACKEND_BASE_URL}${path}`, options);
    let data = null;
    try {
        data = await response.json();
    } catch {
        data = null;
    }

    if (!response.ok) {
        const errorMessage =
            data?.error ||
            data?.message ||
            `Request failed: ${response.status}`;
        throw new Error(errorMessage);
    }

    return data;
}

/**
 * Loads the available cameras into the selector.
 */
async function loadCameraList() {
    const select = getElement("utilsCameraSelect");
    if (!select) {
        return;
    }

    try {
        const payload = await fetchJson("/camera-config/cameras");
        const cameras = Array.isArray(payload?.cameras) ? payload.cameras : [];

        select.innerHTML = "";
        if (cameras.length === 0) {
            const option = document.createElement("option");
            option.value = "";
            option.textContent = "No active cameras";
            select.appendChild(option);
            currentCameraBusId = "";
            setExtrinsicsInputs({});
            setIntrinsicsStatus("No camera selected.");
            setCameraMeta(null);
            return;
        }

        cameras.forEach((camera) => {
            const option = document.createElement("option");
            option.value = String(camera.bus_id);
            option.dataset.streamName = String(
                camera.stream_name || camera.name || "",
            );
            option.textContent = `${camera.name} (${camera.bus_id})`;
            select.appendChild(option);
        });

        if (
            !currentCameraBusId ||
            !cameras.some(
                (camera) => String(camera.bus_id) === currentCameraBusId,
            )
        ) {
            currentCameraBusId = String(cameras[0].bus_id);
        }

        select.value = currentCameraBusId;
        const selected = cameras.find(
            (camera) => String(camera.bus_id) === currentCameraBusId,
        );
        currentCalibrationStreamName = String(
            selected?.stream_name || selected?.name || "",
        );
        setCameraMeta(selected || null);
        await loadCameraConfig(currentCameraBusId);
    } catch (error) {
        showDanger(`Failed to load camera list: ${error.message}`);
    }
}

/**
 * Loads the config for a specific camera bus id.
 * @param {string} cameraBusId - Camera bus id.
 */
async function loadCameraConfig(cameraBusId) {
    if (!cameraBusId) {
        return;
    }

    try {
        const payload = await fetchJson(
            `/camera-config/${encodeURIComponent(cameraBusId)}`,
        );
        setExtrinsicsInputs(payload?.extrinsics || {});
        if (payload?.intrinsics_exists) {
            setIntrinsicsStatus(
                `Current intrinsics: ${payload.intrinsics_path || "intrinsics.json"}`,
            );
        } else {
            setIntrinsicsStatus("No intrinsics file currently set.");
        }
    } catch (error) {
        showDanger(`Failed to load camera config: ${error.message}`);
    }
}

/**
 * Saves the current extrinsics for the selected camera.
 *
 * @returns {Promise<boolean>} Whether the save succeeded.
 */
export async function saveExtrinsics() {
    if (!currentCameraBusId) {
        showWarning("Select a camera first");
        return false;
    }

    try {
        await fetchJson(
            `/camera-config/${encodeURIComponent(currentCameraBusId)}/extrinsics`,
            {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(getExtrinsicsPayload()),
            },
        );
        showSuccess("Camera extrinsics saved");
        await loadCameraConfig(currentCameraBusId);
        return true;
    } catch (error) {
        showDanger(`Failed to save extrinsics: ${error.message}`);
        return false;
    }
}

/**
 * Uploads an intrinsics JSON file for the selected camera.
 * @param {File} file - Intrinsics file.
 */
async function uploadIntrinsicsFile(file) {
    if (!currentCameraBusId) {
        showWarning("Select a camera first");
        return;
    }

    if (!file) {
        showWarning("Select a file first");
        return;
    }

    if (!file.name.toLowerCase().endsWith(".json")) {
        showWarning("Only .json files are supported for intrinsics");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);
    const uploadToast = showUploadToast({
        label: `Uploading ${file.name}...`,
    });

    try {
        await uploadWithProgress({
            url: `/camera-config/${encodeURIComponent(currentCameraBusId)}/intrinsics`,
            formData,
            onProgress: uploadToast.setProgress,
        });
        uploadToast.complete("Intrinsics file uploaded");
        await loadCameraConfig(currentCameraBusId);
    } catch (error) {
        uploadToast.fail(`Failed to upload intrinsics: ${error.message}`);
    }
}

/**
 * Builds the calibration payload from the calibration form inputs.
 * @returns {{squares_x: number, squares_y: number, square_size: number, marker_size: number}}
 */
function calibrationPayload() {
    return {
        squares_x: Number.parseInt(
            getElement("utilsCalibrationSquaresX")?.value || "13",
            10,
        ),
        squares_y: Number.parseInt(
            getElement("utilsCalibrationSquaresY")?.value || "10",
            10,
        ),
        square_size: Number.parseFloat(
            getElement("utilsCalibrationSquareSize")?.value || "0.020",
        ),
        marker_size: Number.parseFloat(
            getElement("utilsCalibrationMarkerSize")?.value || "0.015",
        ),
    };
}

/**
 * Toggles calibration UI busy state.
 * @param {boolean} isBusy - Whether calibration is in progress.
 */
function setCalibrationBusy(isBusy) {
    getElement("utilsCalibrationProgress")?.classList.toggle("hidden", !isBusy);
    [
        "utilsCalibrationCaptureBtn",
        "utilsCalibrationResetBtn",
        "utilsCalibrationRunBtn",
    ].forEach((id) => {
        const button = getElement(id);
        if (button) button.disabled = isBusy;
    });
}

/**
 * Converts the live resolution selector into backend query params.
 * @returns {Object}
 */
function calibrationLiveResolutionParams() {
    const value =
        getElement("utilsCalibrationLiveResolution")?.value || "1280x720";
    if (value === "full") {
        return {};
    }
    const [width, height] = value
        .split("x")
        .map((part) => Number.parseInt(part, 10));
    if (!Number.isFinite(width) || !Number.isFinite(height)) {
        return {};
    }
    return { live_width: String(width), live_height: String(height) };
}

/**
 * Refreshes the calibration feed image URL.
 */
function updateCalibrationFeed() {
    const img = getElement("utilsCalibrationFeed");
    if (!img || !currentCameraBusId || !calibrationModalOpen) return;
    const calibrationCameraId =
        currentCalibrationStreamName || currentCameraBusId;
    const payload = calibrationPayload();
    const params = new URLSearchParams({
        squares_x: String(payload.squares_x),
        squares_y: String(payload.squares_y),
        square_size: String(payload.square_size),
        marker_size: String(payload.marker_size),
        ...calibrationLiveResolutionParams(),
        t: String(Date.now()),
    });
    img.src = `${BACKEND_BASE_URL}/camera-config/${encodeURIComponent(calibrationCameraId)}/calibration/feed?${params}`;
}

/**
 * Renders the captured calibration corners history canvas.
 */
function drawCalibrationHistoryCanvas() {
    const canvas = getElement("utilsCalibrationHistoryCanvas");
    const empty = getElement("utilsCalibrationHistoryEmpty");
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(1, Math.floor(rect.width));
    const height = Math.max(1, Math.floor(rect.height));
    if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, width, height);

    const frames = calibrationHistoryFrames.filter(
        (frame) => frame?.image_size && Array.isArray(frame.corners),
    );
    const hasCorners = frames.some((frame) => frame.corners.length > 0);
    if (empty) empty.classList.toggle("hidden", hasCorners);
    if (!hasCorners) return;

    for (const frame of frames) {
        const sourceWidth = Math.max(1, Number(frame.image_size.width));
        const sourceHeight = Math.max(1, Number(frame.image_size.height));
        const scale = Math.min(width / sourceWidth, height / sourceHeight);
        const offsetX = (width - sourceWidth * scale) / 2;
        const offsetY = (height - sourceHeight * scale) / 2;
        ctx.fillStyle =
            frame.index === selectedCalibrationFrameIndex
                ? "#f9c845"
                : "rgba(0, 255, 160, 0.75)";
        for (const corner of frame.corners) {
            const x = offsetX + Number(corner[0]) * scale;
            const y = offsetY + Number(corner[1]) * scale;
            ctx.beginPath();
            ctx.arc(
                x,
                y,
                frame.index === selectedCalibrationFrameIndex ? 3.5 : 2.5,
                0,
                Math.PI * 2,
            );
            ctx.fill();
        }
    }
}

/**
 * Selects a calibration frame for display.
 * @param {number | null} index - Frame index.
 */
function showCalibrationFrame(index) {
    selectedCalibrationFrameIndex = index;
    drawCalibrationHistoryCanvas();
}

/**
 * Refreshes the captured calibration frames list and canvas.
 * @param {number | null} [preferredIndex=selectedCalibrationFrameIndex] - Preferred frame index.
 */
async function refreshCalibrationFrames(
    preferredIndex = selectedCalibrationFrameIndex,
) {
    if (!currentCameraBusId) return;
    const payload = await fetchJson(
        `/camera-config/${encodeURIComponent(currentCameraBusId)}/calibration/frames`,
    );
    const frames = payload?.frames || [];
    calibrationHistoryFrames = frames;
    const count = payload?.frame_count || 0;
    if (
        preferredIndex !== null &&
        preferredIndex !== undefined &&
        preferredIndex >= count
    )
        preferredIndex = count - 1;
    const status = getElement("utilsCalibrationStatus");
    if (status)
        status.textContent = `${count} frames captured. 10 recommended.`;
    const list = getElement("utilsCalibrationFrames");
    if (list) {
        list.innerHTML = "";
        frames.forEach((frame) => {
            const tile = document.createElement("div");
            tile.className = `relative rounded border ${frame.index === preferredIndex ? "border-[#f9c845]" : "border-[#414141]"} bg-[#232323] overflow-hidden`;
            const thumb = document.createElement("button");
            thumb.type = "button";
            thumb.className = "block w-full";
            thumb.innerHTML = `<img class="w-full h-16 object-cover bg-black" alt="Calibration frame #${frame.index + 1}" src="${BACKEND_BASE_URL}/camera-config/${encodeURIComponent(currentCameraBusId)}/calibration/frames/${frame.index}?t=${Date.now()}" /><span class="block py-1">#${frame.index + 1}</span>`;
            thumb.addEventListener("click", () =>
                showCalibrationFrame(frame.index),
            );
            const del = document.createElement("button");
            del.type = "button";
            del.className =
                "absolute right-1 top-1 px-1 rounded bg-black/70 text-white";
            del.textContent = "×";
            del.addEventListener("click", async (event) => {
                event.stopPropagation();
                await fetchJson(
                    `/camera-config/${encodeURIComponent(currentCameraBusId)}/calibration/frames/${frame.index}`,
                    { method: "DELETE" },
                );
                await refreshCalibrationFrames(
                    frame.index === selectedCalibrationFrameIndex
                        ? null
                        : preferredIndex,
                );
                updateCalibrationFeed();
            });
            tile.append(thumb, del);
            list.appendChild(tile);
        });
    }
    showCalibrationFrame(count > 0 ? preferredIndex : null);
    drawCalibrationHistoryCanvas();
}

/**
 * Opens live corrected and distortion-grid views for the selected camera.
 */
function openDistortionModal() {
    const selectedBusId = getElement("utilsCameraSelect")?.value;
    if (!selectedBusId) {
        showWarning("Select a camera first");
        return;
    }

    currentCameraBusId = selectedBusId;
    distortionModalOpen = true;
    getElement("utilsDistortionModal")?.classList.remove("hidden");
    const timestamp = Date.now();
    const basePath = `${BACKEND_BASE_URL}/camera-config/${encodeURIComponent(currentCameraBusId)}/distortion/feed`;
    const undistortedFeed = getElement("utilsUndistortedFeed");
    const distortedFeed = getElement("utilsDistortedFeed");
    if (undistortedFeed)
        undistortedFeed.src = `${basePath}?view=undistorted&t=${timestamp}`;
    if (distortedFeed)
        distortedFeed.src = `${basePath}?view=distorted&t=${timestamp}`;
}

/**
 * Closes the distortion modal and stops both live streams.
 */
function closeDistortionModal() {
    distortionModalOpen = false;
    getElement("utilsDistortionModal")?.classList.add("hidden");
    ["utilsUndistortedFeed", "utilsDistortedFeed"].forEach((id) => {
        const image = getElement(id);
        if (image) image.src = "";
    });
}

/**
 * Opens the calibration modal for the selected camera.
 */
async function openCalibrationModal() {
    const selectedBusId = getElement("utilsCameraSelect")?.value;
    if (!selectedBusId) {
        showWarning("Select a camera first");
        return;
    }

    // Ensure we are operating on the currently selected camera in the utils tab.
    currentCameraBusId = selectedBusId;
    const selectedOption =
        getElement("utilsCameraSelect")?.selectedOptions?.[0];
    currentCalibrationStreamName = selectedOption?.dataset?.streamName || "";

    calibrationModalOpen = true;
    selectedCalibrationFrameIndex = null;
    getElement("utilsCalibrationModal")?.classList.remove("hidden");
    updateCalibrationFeed();
    await refreshCalibrationFrames();
}

/**
 * Closes the calibration modal and clears its state.
 */
function closeCalibrationModal() {
    calibrationModalOpen = false;
    calibrationHistoryFrames = [];
    selectedCalibrationFrameIndex = null;
    drawCalibrationHistoryCanvas();
    getElement("utilsCalibrationModal")?.classList.add("hidden");
    const img = getElement("utilsCalibrationFeed");
    if (img) img.src = "";
}

/**
 * Captures a calibration frame from the backend.
 */
async function captureCalibrationFrame() {
    if (!currentCameraBusId || !calibrationModalOpen) return;
    try {
        const result = await fetchJson(
            `/camera-config/${encodeURIComponent(currentCameraBusId)}/calibration/capture`,
            {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(calibrationPayload()),
            },
        );
        if (result?.squares_x && result?.squares_y) {
            getElement("utilsCalibrationSquaresX").value = String(
                result.squares_x,
            );
            getElement("utilsCalibrationSquaresY").value = String(
                result.squares_y,
            );
        }
        showSuccess("Calibration frame captured");
        await refreshCalibrationFrames(result?.frame_index ?? null);
        updateCalibrationFeed();
    } catch (error) {
        showDanger(`Failed to capture frame: ${error.message}`);
    }
}

/**
 * Resets stored calibration frames on the backend.
 */
async function resetCalibrationFrames() {
    if (!currentCameraBusId) return;
    await fetchJson(
        `/camera-config/${encodeURIComponent(currentCameraBusId)}/calibration/reset`,
        { method: "POST" },
    );
    selectedCalibrationFrameIndex = null;
    await refreshCalibrationFrames(null);
    updateCalibrationFeed();
}

/**
 * Runs calibration and saves the resulting intrinsics.
 */
async function runCalibration() {
    if (!currentCameraBusId) return;
    setCalibrationBusy(true);
    try {
        const result = await fetchJson(
            `/camera-config/${encodeURIComponent(currentCameraBusId)}/calibration/run`,
            {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(calibrationPayload()),
            },
        );
        showSuccess(
            `Calibration saved using ${result.frame_count} of ${result.captured_frame_count ?? result.frame_count} frames. Reprojection error: ${result.reprojection_error?.toFixed?.(4) ?? result.reprojection_error}`,
        );
        for (const warning of result.warnings || []) showWarning(warning);
        closeCalibrationModal();
        await loadCameraConfig(currentCameraBusId);
    } catch (error) {
        showDanger(`Calibration failed: ${error.message}`);
    } finally {
        setCalibrationBusy(false);
    }
}

/**
 * Deletes the selected camera's intrinsics file.
 */
async function deleteIntrinsics() {
    if (!currentCameraBusId) {
        showWarning("Select a camera first");
        return;
    }

    try {
        await fetchJson(
            `/camera-config/${encodeURIComponent(currentCameraBusId)}/intrinsics`,
            {
                method: "DELETE",
            },
        );
        showSuccess("Intrinsics file deleted");
        await loadCameraConfig(currentCameraBusId);
    } catch (error) {
        showDanger(`Failed to delete intrinsics: ${error.message}`);
    }
}

/**
 * Sets up drag-and-drop handling for intrinsics upload.
 */
function setupDropzone() {
    const dropzone = getElement("utilsIntrinsicsDropzone");
    if (!dropzone) {
        return;
    }

    const setHover = (isHover) => {
        dropzone.classList.toggle("border-[#f9c845]", isHover);
        dropzone.classList.toggle("text-[#f9c845]", isHover);
    };

    dropzone.addEventListener("dragover", (event) => {
        event.preventDefault();
        setHover(true);
    });

    dropzone.addEventListener("dragleave", () => {
        setHover(false);
    });

    dropzone.addEventListener("drop", (event) => {
        event.preventDefault();
        setHover(false);
        const file = event.dataTransfer?.files?.[0];
        if (file) {
            void uploadIntrinsicsFile(file);
        }
    });
}

/**
 * Select a camera in Camera Config Utils without opening a nested modal.
 *
 * @param {string} cameraBusId - Stable camera bus identifier.
 * @returns {Promise<void>}
 */
export async function selectCameraConfig(cameraBusId) {
    const requestedCameraBusId = String(cameraBusId);
    initCameraConfigUtils();
    currentCameraBusId = requestedCameraBusId;
    await loadCameraList();
    const select = getElement("utilsCameraSelect");
    if (
        !select?.querySelector(
            `option[value="${CSS.escape(requestedCameraBusId)}"]`,
        )
    ) {
        throw new Error("The selected camera is no longer active.");
    }
}

/**
 * Open the existing intrinsics calibration flow for a selected camera.
 *
 * @param {string} cameraBusId - Stable camera bus identifier.
 * @returns {Promise<void>}
 */
export async function openCameraCalibration(cameraBusId) {
    await selectCameraConfig(cameraBusId);
    await openCalibrationModal();
}

/**
 * Initializes the camera configuration utilities UI.
 */
export function initCameraConfigUtils() {
    if (initialized) {
        startCameraPoseVisualizationLoop();
        return;
    }

    const cameraSelect = getElement("utilsCameraSelect");
    const saveButton = getElement("utilsSaveExtrinsicsBtn");
    const refreshButton = getElement("utilsRefreshConfigBtn");
    const uploadButton = getElement("utilsUploadIntrinsicsBtn");
    const calibrateButton = getElement("utilsCalibrateIntrinsicsBtn");
    const distortionButton = getElement("utilsViewDistortionBtn");
    const deleteButton = getElement("utilsDeleteIntrinsicsBtn");
    const fileInput = getElement("utilsIntrinsicsFileInput");

    if (!cameraSelect) {
        return;
    }

    initCameraPoseVisualization();
    resizeCameraPoseVisualization();
    updateCameraPoseVisualization();

    cameraSelect.addEventListener("change", () => {
        currentCameraBusId = cameraSelect.value;
        currentCalibrationStreamName =
            cameraSelect.selectedOptions?.[0]?.dataset?.streamName || "";
        void loadCameraConfig(currentCameraBusId);
        const selectedText =
            cameraSelect.options[cameraSelect.selectedIndex]?.text || "";
        setCameraMeta(
            currentCameraBusId
                ? {
                      name: selectedText.split(" (")[0] || selectedText,
                      bus_id: currentCameraBusId,
                  }
                : null,
        );
    });

    saveButton?.addEventListener("click", () => {
        void saveExtrinsics();
    });

    refreshButton?.addEventListener("click", () => {
        void loadCameraList();
    });

    uploadButton?.addEventListener("click", () => {
        fileInput?.click();
    });

    fileInput?.addEventListener("change", () => {
        const file = fileInput.files?.[0];
        if (file) {
            void uploadIntrinsicsFile(file);
            fileInput.value = "";
        }
    });

    calibrateButton?.addEventListener("click", () => {
        void openCalibrationModal();
    });

    distortionButton?.addEventListener("click", openDistortionModal);

    deleteButton?.addEventListener("click", () => {
        void deleteIntrinsics();
    });

    document.addEventListener("backend-disconnected", () => {
        closeCalibrationModal();
        closeDistortionModal();
    });
    getElement("utilsDistortionCloseBtn")?.addEventListener(
        "click",
        closeDistortionModal,
    );
    getElement("utilsCalibrationCloseBtn")?.addEventListener(
        "click",
        closeCalibrationModal,
    );
    getElement("utilsCalibrationCaptureBtn")?.addEventListener(
        "click",
        () => void captureCalibrationFrame(),
    );
    getElement("utilsCalibrationResetBtn")?.addEventListener(
        "click",
        () => void resetCalibrationFrames(),
    );
    getElement("utilsCalibrationRunBtn")?.addEventListener(
        "click",
        () => void runCalibration(),
    );
    window.addEventListener("resize", drawCalibrationHistoryCanvas);
    [
        "utilsCalibrationSquaresX",
        "utilsCalibrationSquaresY",
        "utilsCalibrationSquareSize",
        "utilsCalibrationMarkerSize",
        "utilsCalibrationLiveResolution",
    ].forEach((id) => {
        getElement(id)?.addEventListener("change", updateCalibrationFeed);
    });
    document.addEventListener("keydown", (event) => {
        if (distortionModalOpen && event.key === "Escape") {
            closeDistortionModal();
            return;
        }
        if (!calibrationModalOpen) return;
        if (event.key === "Escape") closeCalibrationModal();
        if (event.key === " " || event.key.toLowerCase() === "c") {
            event.preventDefault();
            void captureCalibrationFrame();
        }
    });

    EXTRINSICS_KEYS.forEach((key) => {
        const input = getElement(INPUT_ID_BY_KEY[key]);
        input?.addEventListener("input", () => {
            updateCameraPoseVisualization();
        });
    });

    globalThis.addEventListener("resize", () => {
        resizeCameraPoseVisualization();
    });

    setupDropzone();
    void loadCameraList();
    initialized = true;
}

/**
 * Refreshes the camera configuration utilities UI state.
 */
export function refreshCameraConfigUtils() {
    resizeCameraPoseVisualization();
    void loadCameraList();
}
