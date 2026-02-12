import { BACKEND_BASE_URL } from "../config.js";
import { showDanger, showSuccess, showWarning } from "../ui/notificationSystem.js";
import * as THREE from "three";
import { OrbitControls } from "OrbitControls";

const EXTRINSICS_KEYS = [
    "horizontal_fov",
    "vertical_fov",
    "pitch",
    "yaw",
    "roll",
    "x_offset",
    "y_offset",
    "z_offset",
];

const INPUT_ID_BY_KEY = {
    horizontal_fov: "utils-horizontal-fov",
    vertical_fov: "utils-vertical-fov",
    pitch: "utils-pitch",
    yaw: "utils-yaw",
    roll: "utils-roll",
    x_offset: "utils-x-offset",
    y_offset: "utils-y-offset",
    z_offset: "utils-z-offset",
};

let initialized = false;
let currentCameraBusId = "";
let poseVizState = null;

const DEG_TO_RAD = Math.PI / 180;

function getElement(id) {
    return document.getElementById(id);
}

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

function initCameraPoseVisualization() {
    if (poseVizState) {
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

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(globalThis.devicePixelRatio || 1);
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
        new THREE.MeshStandardMaterial({ color: 0x4a5568, metalness: 0.2, roughness: 0.7 }),
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
    };

    const renderFrame = () => {
        if (!poseVizState) {
            return;
        }
        poseVizState.controls.update();
        poseVizState.renderer.render(poseVizState.scene, poseVizState.camera);
        poseVizState.frameHandle = globalThis.requestAnimationFrame(renderFrame);
    };

    renderFrame();
}

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

function updateCameraPoseVisualization() {
    if (!poseVizState) {
        return;
    }

    const xOffset = Number.parseFloat(getElement("utils-x-offset")?.value ?? "0");
    const yOffset = Number.parseFloat(getElement("utils-y-offset")?.value ?? "0");
    const zOffset = Number.parseFloat(getElement("utils-z-offset")?.value ?? "0");
    const yawDegrees = Number.parseFloat(getElement("utils-yaw")?.value ?? "0");
    const pitchDegrees = Number.parseFloat(getElement("utils-pitch")?.value ?? "0");
    const rollDegrees = Number.parseFloat(getElement("utils-roll")?.value ?? "0");

    const x = Number.isFinite(xOffset) ? xOffset : 0;
    const y = Number.isFinite(yOffset) ? yOffset : 0;
    const z = Number.isFinite(zOffset) ? zOffset : 0;
    const yaw = (Number.isFinite(yawDegrees) ? yawDegrees : 0) * DEG_TO_RAD;
    const pitch = (Number.isFinite(pitchDegrees) ? pitchDegrees : 0) * DEG_TO_RAD;
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

function getExtrinsicsPayload() {
    const payload = {};
    EXTRINSICS_KEYS.forEach((key) => {
        const input = getElement(INPUT_ID_BY_KEY[key]);
        const numericValue = Number.parseFloat(input?.value ?? "0");
        payload[key] = Number.isFinite(numericValue) ? numericValue : 0;
    });
    return payload;
}

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

function setIntrinsicsStatus(text) {
    const status = getElement("utilsIntrinsicsStatus");
    if (status) {
        status.textContent = text;
    }
}

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

async function fetchJson(path, options = {}) {
    const response = await fetch(`${BACKEND_BASE_URL}${path}`, options);
    let data = null;
    try {
        data = await response.json();
    } catch {
        data = null;
    }

    if (!response.ok) {
        const errorMessage = data?.error || data?.message || `Request failed: ${response.status}`;
        throw new Error(errorMessage);
    }

    return data;
}

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
            option.textContent = `${camera.name} (${camera.bus_id})`;
            select.appendChild(option);
        });

        if (!currentCameraBusId || !cameras.some((camera) => String(camera.bus_id) === currentCameraBusId)) {
            currentCameraBusId = String(cameras[0].bus_id);
        }

        select.value = currentCameraBusId;
        const selected = cameras.find((camera) => String(camera.bus_id) === currentCameraBusId);
        setCameraMeta(selected || null);
        await loadCameraConfig(currentCameraBusId);
    } catch (error) {
        showDanger(`Failed to load camera list: ${error.message}`);
    }
}

async function loadCameraConfig(cameraBusId) {
    if (!cameraBusId) {
        return;
    }

    try {
        const payload = await fetchJson(`/camera-config/${encodeURIComponent(cameraBusId)}`);
        setExtrinsicsInputs(payload?.extrinsics || {});
        if (payload?.intrinsics_exists) {
            setIntrinsicsStatus(`Current intrinsics: ${payload.intrinsics_path || "intrinsics.json"}`);
        } else {
            setIntrinsicsStatus("No intrinsics file currently set.");
        }
    } catch (error) {
        showDanger(`Failed to load camera config: ${error.message}`);
    }
}

async function saveExtrinsics() {
    if (!currentCameraBusId) {
        showWarning("Select a camera first");
        return;
    }

    try {
        await fetchJson(`/camera-config/${encodeURIComponent(currentCameraBusId)}/extrinsics`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(getExtrinsicsPayload()),
        });
        showSuccess("Camera extrinsics saved");
        await loadCameraConfig(currentCameraBusId);
    } catch (error) {
        showDanger(`Failed to save extrinsics: ${error.message}`);
    }
}

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

    try {
        const formData = new FormData();
        formData.append("file", file);
        await fetchJson(`/camera-config/${encodeURIComponent(currentCameraBusId)}/intrinsics`, {
            method: "POST",
            body: formData,
        });
        showSuccess("Intrinsics file uploaded");
        await loadCameraConfig(currentCameraBusId);
    } catch (error) {
        showDanger(`Failed to upload intrinsics: ${error.message}`);
    }
}

async function deleteIntrinsics() {
    if (!currentCameraBusId) {
        showWarning("Select a camera first");
        return;
    }

    try {
        await fetchJson(`/camera-config/${encodeURIComponent(currentCameraBusId)}/intrinsics`, {
            method: "DELETE",
        });
        showSuccess("Intrinsics file deleted");
        await loadCameraConfig(currentCameraBusId);
    } catch (error) {
        showDanger(`Failed to delete intrinsics: ${error.message}`);
    }
}

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

export function initCameraConfigUtils() {
    if (initialized) {
        return;
    }

    const cameraSelect = getElement("utilsCameraSelect");
    const saveButton = getElement("utilsSaveExtrinsicsBtn");
    const refreshButton = getElement("utilsRefreshConfigBtn");
    const uploadButton = getElement("utilsUploadIntrinsicsBtn");
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
        void loadCameraConfig(currentCameraBusId);
        const selectedText = cameraSelect.options[cameraSelect.selectedIndex]?.text || "";
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

    deleteButton?.addEventListener("click", () => {
        void deleteIntrinsics();
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

export function refreshCameraConfigUtils() {
    resizeCameraPoseVisualization();
    void loadCameraList();
}
