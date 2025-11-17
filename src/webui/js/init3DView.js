import { GLTFLoader } from "GLTFLoader";
import {
    PCFSoftShadowMap,
    WebGLRenderer,
    AmbientLight,
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

let renderer, scene, camera, directionalLight;
let shadowsEnabled = true;
let gamePiecesVisible = true;
let statsDisplay;
let frameCount = 0;
let lastTime = performance.now();
let robotObject = null;
let robotAxes = null;
let animationStarted = false;
let detectedObjectsGroup = null;
let pendingDetectedObjects = null;

let maxFPS = 30;
let interval = 1 / maxFPS;

const robotScaleMatrix = new Matrix4().makeScale(1000, 1000, 1000);
const robotFinalMatrix = new Matrix4();
const detectionCylinderRadius = 150;
const detectionCylinderHeight = 400;
const detectionLabelOffset = 250;
const detectionScaleFactor = 1000;

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
                if (material.map) {
                    material.map.dispose();
                }
                material.dispose();
            }
        }
    });
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
    if (!Array.isArray(position) || position.length !== 3) {
        return null;
    }
    const numericPosition = position.map((value) => Number(value));
    if (numericPosition.some((value) => !Number.isFinite(value))) {
        return null;
    }
    const fieldCenterX = 8.774125;
    const fieldCenterZ = 4.025901;
    return new Vector3(
        (numericPosition[0] - fieldCenterX) * detectionScaleFactor,
        numericPosition[2] * detectionScaleFactor,
        (-numericPosition[1] + fieldCenterZ) * detectionScaleFactor,
    );
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

function createDetectionLabelSprite(labelText) {
    const canvas = document.createElement("canvas");
    canvas.width = 512;
    canvas.height = 128;
    const context = canvas.getContext("2d");
    if (!context) {
        return null;
    }
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = "rgba(20, 20, 20, 0.8)";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = "#ffffff";
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
    const labelSprite = createDetectionLabelSprite(labelText);
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

export async function init3DView(modelUrl) {
    const container = document.getElementById("view-3d");
    statsDisplay = document.getElementById("statsDisplay");
    statsDisplay.style.position = "absolute";
    statsDisplay.style.bottom = "10px";
    statsDisplay.style.right = "10px";
    statsDisplay.style.color = "#f9c84a";
    statsDisplay.style.fontSize = "1rem";
    statsDisplay.style.zIndex = "10";

    const scale = 40;

    await populateRobotDropdown();

    // Clear and destroy existing scene if it exists
    if (scene) {
        clearDetectedObjectsGroup();
        // Remove all objects from the scene
        while (scene.children.length > 0) {
            const child = scene.children[0];
            scene.remove(child);
            child.traverse((node) => {
                // dispose geometry
                if (node.geometry) {
                    node.geometry.dispose();
                }
                // dispose material(s) and any bound textures
                if (node.material) {
                    const materials = Array.isArray(node.material)
                        ? node.material
                        : [node.material];
                    for (const m of materials) {
                        for (const key in m) {
                            const val = m[key];
                            if (val && val.isTexture) {
                                val.dispose();
                            }
                        }
                        m.dispose();
                    }
                }
            });
        }

        // Clear the scene
        scene.clear();
        scene = null;
        detectedObjectsGroup = null;

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
    if (pendingDetectedObjects) {
        renderDetectedObjects(pendingDetectedObjects);
    }

    const dracoLoader = new DRACOLoader();
    dracoLoader.setDecoderPath(`${BACKEND_BASE_URL}/draco/`);

    scene.background = new Color(0x222222);

    function loadRobot(robotFile) {
        console.log("Loading robot:", robotFile);
        try {
            if (robotObject) {
                scene.remove(robotObject);
            }

            const robotLoader = new GLTFLoader();
            robotLoader.setDRACOLoader(dracoLoader);

            robotLoader.load(
                `${BACKEND_BASE_URL}/get-robot-file/${robotFile}`,
                (gltf) => {
                    robotObject = gltf.scene;
                    robotObject.scale.set(1000, 1000, 1000);

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
                },
            );
        } catch (error) {
            console.error("Error loading robot:", error);
        }
        console.log("Loaded robot:", robotFile);
    }

    const robotFileSelect = document.getElementById("robotFileSelect");
    let selectedRobotFile = robotFileSelect.value;

    loadRobot(selectedRobotFile);

    robotFileSelect.addEventListener("change", () => {
        selectedRobotFile = robotFileSelect.value;
        loadRobot(selectedRobotFile);
    });

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
    fieldLoader.load(
        modelUrl,
        (gltf) => {
            const model = gltf.scene;

            model.rotation.x = Math.PI / 2;

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
        },
        undefined,
        (error) => {
            console.error("Error loading the model:", error);
            startAnimationLoop();
        },
    );

    const gamePiecePath =
        modelUrl.split("/").slice(0, -2).join("/") +
        "/game_pieces/" +
        modelUrl.split("/").pop().slice(0, 7) +
        "-GP.glb";

    const gamePieces = [];

    const gpLoader = new GLTFLoader();
    gpLoader.load(
        gamePiecePath,
        (gltf) => {
            const model = gltf.scene;

            model.rotation.x = Math.PI / 2;

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
        },
        undefined,
        (error) => {
            console.error("Error loading the model:", error);
        },
    );

    if (!globalThis.__eev_gamePiecesToggleAttached) {
        document
            .getElementById("toggleGamePiecesBtn")
            .addEventListener("click", () => {
                gamePiecesVisible = !gamePiecesVisible;
                for (const gp of gamePieces) {
                    gp.visible = gamePiecesVisible;
                }
            });
        globalThis.__eev_gamePiecesToggleAttached = true;
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
            .addEventListener("click", () => {
                shadowsEnabled = !shadowsEnabled;
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

    // Add AprilTag PNGs as planes at fiducial transforms
    fetch(`${BACKEND_BASE_URL}/frc2025r2.json`)
        .then((response) => response.json())
        .then((json) => {
            const textureLoader = new TextureLoader();
            for (const fiducial of json.fiducials) {
                const tagId = fiducial.id;
                const pngName = `tag36_11_${String(tagId).padStart(5, "0")}.png`;
                const pngPath = `${BACKEND_BASE_URL}/src/webui/assets/apriltags/${pngName}`;
                textureLoader.load(pngPath, (texture) => {
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
                    matrix.extractBasis(new Vector3(), new Vector3(), normal);
                    normal.normalize();
                    plane.position.add(normal);

                    plane.castShadow = false;
                    plane.receiveShadow = false;
                    plane.excludeFromShadowToggle = true;
                    scene.add(plane);
                });
            }
        });
}

export function updateRobotTransform(transformMatrix) {
    if (robotObject) {
        robotFinalMatrix.multiplyMatrices(transformMatrix, robotScaleMatrix);

        robotObject.matrixAutoUpdate = false;
        robotObject.matrix.copy(robotFinalMatrix);
        robotObject.matrixWorldNeedsUpdate = true;
    } else {
        console.warn("Robot not initialized yet");
    }
}
