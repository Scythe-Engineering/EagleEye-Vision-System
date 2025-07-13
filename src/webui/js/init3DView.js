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
    TextureLoader,
    Matrix4,
    NearestFilter,
    Vector3,
    BufferGeometry,
    BufferAttribute,
    LineSegments,
    LineBasicMaterial,
    Group,
} from "three";
import { OrbitControls } from "OrbitControls";
import { populateRobotDropdown } from "./dropdown/robotDropdown.js";

let renderer, scene, camera, directionalLight;
let shadowsEnabled = true;
let gamePiecesVisible = true;
let statsDisplay;
let frameCount = 0;
let lastTime = performance.now();
let robotObject = null;
let robotAxes = null;

let maxFPS = 60;
let interval = 1 / maxFPS;

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
    // TODO: Create coordinate axes for robot visualization
    const axesGroup = new Group();
    const axisLength = 500; // Adjust length as needed
    
    // Create geometry for axes lines
    const positions = new Float32Array([
        // X-axis (red)
        0, 0, 0,  axisLength, 0, 0,
        // Y-axis (green)
        0, 0, 0,  0, axisLength, 0,
        // Z-axis (blue)
        0, 0, 0,  0, 0, axisLength
    ]);
    
    const colors = new Float32Array([
        // X-axis (red)
        1, 0, 0,  1, 0, 0,
        // Y-axis (green)
        0, 1, 0,  0, 1, 0,
        // Z-axis (blue)
        0, 0, 1,  0, 0, 1
    ]);
    
    const geometry = new BufferGeometry();
    geometry.setAttribute('position', new BufferAttribute(positions, 3));
    geometry.setAttribute('color', new BufferAttribute(colors, 3));
    
    const material = new LineBasicMaterial({ 
        vertexColors: true,
        linewidth: 3
    });
    
    const axes = new LineSegments(geometry, material);
    axesGroup.add(axes);
    
    return axesGroup;
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
        // Remove all objects from the scene
        while (scene.children.length > 0) {
            const child = scene.children[0];
            scene.remove(child);
            
            // Dispose of geometries and materials to free memory
            if (child.geometry) {
                child.geometry.dispose();
            }
            if (child.material) {
                if (Array.isArray(child.material)) {
                    child.material.forEach(material => material.dispose());
                } else {
                    child.material.dispose();
                }
            }
        }
        
        // Clear the scene
        scene.clear();
        scene = null;
    }

    scene = new Scene();
    scene.background = new Color(0x222222);

    function loadRobot(robotFile) {
        console.log("Loading robot:", robotFile);
        try {
            if (robotObject) {
                scene.remove(robotObject);
            }
            if (robotAxes) {
                scene.remove(robotAxes);
                robotAxes = null;
            }
            
            const robotLoader = new GLTFLoader();
            robotLoader.load("/get-robot-file/" + robotFile, (gltf) => {
                robotObject = gltf.scene;
                robotObject.scale.set(1000, 1000, 1000);

                robotObject.traverse((child) => {
                    if (child.isMesh) {
                        child.castShadow = true;
                        child.receiveShadow = true;
                        child.geometry.computeVertexNormals();
                        
                        // Remove reflective properties from materials
                        if (child.material) {
                            if (Array.isArray(child.material)) {
                                child.material.forEach(material => {
                                    material.metalness = 0;
                                    material.roughness = 1;
                                });
                            } else {
                                child.material.metalness = 0;
                                child.material.roughness = 1;
                            }
                        }
                    }
                });

                scene.add(robotObject);
                
                robotAxes = createRobotAxes();
                scene.add(robotAxes);
            });
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

    renderer = new WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = PCFSoftShadowMap;
    renderer.domElement.style.width = "100%";
    renderer.domElement.style.height = "100%";
    renderer.domElement.style.display = "block";
    renderer.domElement.classList.add('absolute', 'top-0', 'left-0', 'w-full', 'h-full', 'rounded-inherit', '-z-10', 'block');
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);

    scene.add(new AmbientLight(0xffffff, 0.2));

    directionalLight = new DirectionalLight(0xffffff, 2);
    directionalLight.position.set(100 * scale, 200 * scale, 200 * scale);
    directionalLight.castShadow = true;
    directionalLight.shadow.bias = -0.0005;
    directionalLight.shadow.normalBias = -0.0005;
    directionalLight.shadow.mapSize.width = 1024 * 5;
    directionalLight.shadow.mapSize.height = 1024 * 5;
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
            animate();
        },
        undefined,
        (error) => {
            console.error("Error loading the model:", error);
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
                    gamePieces.push(child);
                }
            });
            scene.add(model);
            animate();
        },
        undefined,
        (error) => {
            console.error("Error loading the model:", error);
        },
    );

    document
        .getElementById("toggleGamePiecesBtn")
        .addEventListener("click", () => {
            gamePiecesVisible = !gamePiecesVisible;
            gamePieces.forEach((gp) => {
                gp.visible = gamePiecesVisible;
            });
        });

    let clock = new Clock();
    let delta = 0;

    function animate() {
        requestAnimationFrame(animate);
        delta += clock.getDelta();

        if (delta > interval) {
            renderer.render(scene, camera);
            updateStats();
            delta = delta % interval;
        }
    }

    window.addEventListener("resize", () => {
        const width = container.clientWidth;
        const height = container.clientHeight;
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
    });

    document.getElementById("toggleShadowBtn").addEventListener("click", () => {
        shadowsEnabled = !shadowsEnabled;
        scene.traverse((object) => {
            if (object.isMesh) {
                object.castShadow = shadowsEnabled;
                object.receiveShadow = shadowsEnabled;
            }
        });
        directionalLight.castShadow = shadowsEnabled;
        renderer.shadowMap.enabled = shadowsEnabled;
    });

    // Add AprilTag PNGs as planes at fiducial transforms
    fetch("/frc2025r2.json")
        .then((response) => response.json())
        .then((json) => {
            const textureLoader = new TextureLoader();
            json.fiducials.forEach((fiducial) => {
                const tagId = fiducial.id;
                const pngName = `tag36_11_${String(tagId).padStart(5, "0")}.png`;
                const pngPath = `/src/webui/assets/apriltags/${pngName}`;
                textureLoader.load(pngPath, (texture) => {
                    // Configure texture for crisp pixel art
                    texture.magFilter = NearestFilter;
                    texture.minFilter = NearestFilter;
                    texture.generateMipmaps = false;
                    
                    const planeGeometry = new PlaneGeometry(fiducial.size, fiducial.size);
                    const planeMaterial = new MeshStandardMaterial({ 
                        map: texture,
                    });
                    const plane = new Mesh(planeGeometry, planeMaterial);
                    // Apply 4x4 transform from JSON
                    const t = fiducial.transform;
                    // Three.js uses column-major, so set matrix directly
                    const matrix = new Matrix4();
                    matrix.set(
                        t[0], t[1], t[2], t[3] * 1000,
                        t[4], t[5], t[6], t[7] * 1000,
                        t[8], t[9], t[10], t[11] * 1000,
                        t[12], t[13], t[14], t[15]
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
                    
                    plane.castShadow = true;
                    plane.receiveShadow = true;
                    scene.add(plane);
                });
            });
        });
}

export function updateRobotTransform(transformMatrix) {
    if (robotObject) {
        // Create a scale matrix to preserve the robot's scale (1000)
        const scaleMatrix = new Matrix4();
        scaleMatrix.makeScale(1000, 1000, 1000);
        
        // Combine the input transformation with the scale
        const finalMatrix = new Matrix4();
        finalMatrix.multiplyMatrices(transformMatrix, scaleMatrix);
        
        robotObject.matrixAutoUpdate = false;
        robotObject.matrix.copy(finalMatrix);
        robotObject.matrixWorldNeedsUpdate = true;

        // Update robot axes to match robot transformation
        if (robotAxes) {
            robotAxes.matrixAutoUpdate = false;
            robotAxes.matrix.copy(finalMatrix);
            robotAxes.matrixWorldNeedsUpdate = true;
        }
        
        // Force immediate re-render when transformation updates
        if (renderer && scene && camera) {
            renderer.render(scene, camera);
        }
    } else {
        console.warn("Robot not initialized yet");
    }
}
