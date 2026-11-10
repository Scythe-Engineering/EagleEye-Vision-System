// Three.js scene for the camera position preview: a robot, the camera on it,
// and dimension lines for the camera's X/Y/Z offsets from the robot origin.
import * as THREE from "three";
import { OrbitControls } from "OrbitControls";
import { mountingPoseToViewMatrix } from "./fieldSpaceTransforms.js";

const BACKGROUND = 0x1c1c1c;
const ACCENT = "#f9c845";
const CAMERA_COLOR = 0x22d3ee;
const AXIS_COLORS = { x: "#f87171", y: "#4ade80", z: "#60a5fa" };

const CHASSIS = { length: 0.72, width: 0.72, height: 0.1, clearance: 0.03 };
const BUMPER = { thickness: 0.06, height: 0.12 };
const ARROW = { length: 0.03, radius: 0.011 };
// Offsets smaller than this get no dimension line.
const MIN_DIMENSION_M = 0.005;
// Drawn after the robot so they stay visible through it. Overlay materials
// are marked transparent so they sort with the cutaway sprite by this order.
const OVERLAY_ORDER = { cutaway: 10, camera: 11, dimension: 12 };

/**
 * Draws text in a rounded pill on a canvas and wraps it in a sprite that
 * keeps the same on-screen size at any zoom.
 * @param {string} color - Text color.
 * @returns {{sprite: THREE.Sprite, setText: (text: string) => void}}
 */
function createLabel(color) {
    const canvas = document.createElement("canvas");
    const context = canvas.getContext("2d");
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    const sprite = new THREE.Sprite(
        new THREE.SpriteMaterial({
            map: texture,
            transparent: true,
            depthTest: false,
            sizeAttenuation: false,
        }),
    );
    sprite.renderOrder = OVERLAY_ORDER.dimension + 1;
    let current = null;

    const setText = (text) => {
        if (text === current) {
            return;
        }
        current = text;
        const fontPx = 22;
        context.font = `600 ${fontPx}px ui-sans-serif, system-ui, sans-serif`;
        const width = Math.ceil(context.measureText(text).width) + 20;
        const height = fontPx + 12;
        canvas.width = width;
        canvas.height = height;
        context.font = `600 ${fontPx}px ui-sans-serif, system-ui, sans-serif`;
        context.fillStyle = "rgba(20, 20, 20, 0.85)";
        context.beginPath();
        context.roundRect(1, 1, width - 2, height - 2, height / 2);
        context.fill();
        context.strokeStyle = color;
        context.globalAlpha = 0.5;
        context.lineWidth = 2;
        context.stroke();
        context.globalAlpha = 1;
        context.fillStyle = color;
        context.textAlign = "center";
        context.textBaseline = "middle";
        context.fillText(text, width / 2, height / 2 + 1);
        texture.needsUpdate = true;
        const screenHeight = 0.055;
        sprite.scale.set((screenHeight * width) / height, screenHeight, 1);
    };

    return { sprite, setText };
}

/**
 * Makes a material that draws over the robot.
 * @param {number | string} color - Material color.
 * @returns {THREE.MeshBasicMaterial}
 */
function overlayMaterial(color) {
    return new THREE.MeshBasicMaterial({ color, depthTest: false, transparent: true });
}

/**
 * Builds a dimension line: a line with an arrowhead at each end and a label
 * at its middle, all drawn over the robot.
 * @param {string} color - Line and label color.
 * @param {[number, number]} labelAnchor - Which point of the label sits on
 *     the line's middle, as a sprite center (0..1 across, 0..1 up).
 * @returns {{group: THREE.Group, update: (start: THREE.Vector3, end: THREE.Vector3, text: string) => void}}
 */
function createDimensionLine(color, labelAnchor) {
    const group = new THREE.Group();
    const lineGeometry = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(),
        new THREE.Vector3(),
    ]);
    const line = new THREE.Line(
        lineGeometry,
        new THREE.LineBasicMaterial({ color, depthTest: false, transparent: true }),
    );
    line.renderOrder = OVERLAY_ORDER.dimension;
    group.add(line);

    const arrowGeometry = new THREE.ConeGeometry(ARROW.radius, ARROW.length, 12);
    arrowGeometry.translate(0, -ARROW.length / 2, 0);
    const arrowMaterial = overlayMaterial(color);
    const arrows = [0, 1].map(() => {
        const arrow = new THREE.Mesh(arrowGeometry, arrowMaterial);
        arrow.renderOrder = OVERLAY_ORDER.dimension;
        group.add(arrow);
        return arrow;
    });

    const label = createLabel(color);
    label.sprite.center.set(...labelAnchor);
    group.add(label.sprite);

    const direction = new THREE.Vector3();
    const up = new THREE.Vector3(0, 1, 0);
    const update = (start, end, text) => {
        direction.subVectors(end, start);
        const length = direction.length();
        group.visible = length >= MIN_DIMENSION_M;
        if (!group.visible) {
            return;
        }
        direction.divideScalar(length);
        lineGeometry.setFromPoints([start, end]);
        // Each cone's tip sits on an end point, pointing outward.
        arrows[0].position.copy(start);
        arrows[0].quaternion.setFromUnitVectors(up, direction.clone().negate());
        arrows[1].position.copy(end);
        arrows[1].quaternion.setFromUnitVectors(up, direction);
        const showArrows = length > ARROW.length * 2.5;
        arrows.forEach((arrow) => {
            arrow.visible = showArrows;
        });
        label.sprite.position.lerpVectors(start, end, 0.5);
        label.setText(text);
    };

    return { group, update };
}

/**
 * Builds the robot: chassis, bumpers with the front one highlighted and a
 * forward arrow on top.
 * @returns {THREE.Group}
 */
function createRobot() {
    const robot = new THREE.Group();
    const { length, width, height, clearance } = CHASSIS;

    const chassis = new THREE.Mesh(
        new THREE.BoxGeometry(length, height, width),
        new THREE.MeshStandardMaterial({ color: 0x3b4252, roughness: 0.8 }),
    );
    chassis.position.y = clearance + height / 2;
    chassis.add(
        new THREE.LineSegments(
            new THREE.EdgesGeometry(chassis.geometry),
            new THREE.LineBasicMaterial({ color: 0x5b6475 }),
        ),
    );
    robot.add(chassis);

    const bumperY = clearance + BUMPER.height / 2 - 0.01;
    const sideMaterial = new THREE.MeshStandardMaterial({ color: 0x2b2f38, roughness: 0.9 });
    const frontMaterial = new THREE.MeshStandardMaterial({ color: ACCENT, roughness: 0.6 });
    const half = BUMPER.thickness / 2;
    const bumpers = [
        { size: [BUMPER.thickness, width + BUMPER.thickness * 2], at: [length / 2 + half, 0], front: true },
        { size: [BUMPER.thickness, width + BUMPER.thickness * 2], at: [-length / 2 - half, 0] },
        { size: [length, BUMPER.thickness], at: [0, width / 2 + half] },
        { size: [length, BUMPER.thickness], at: [0, -width / 2 - half] },
    ];
    bumpers.forEach(({ size, at, front }) => {
        const bumper = new THREE.Mesh(
            new THREE.BoxGeometry(size[0], BUMPER.height, size[1]),
            front ? frontMaterial : sideMaterial,
        );
        bumper.position.set(at[0], bumperY, at[1]);
        robot.add(bumper);
    });

    // A flat chevron on the deck pointing forward (+X).
    const chevron = new THREE.Shape();
    chevron.moveTo(0.12, 0);
    chevron.lineTo(-0.04, 0.1);
    chevron.lineTo(-0.04, 0.05);
    chevron.lineTo(0.04, 0);
    chevron.lineTo(-0.04, -0.05);
    chevron.lineTo(-0.04, -0.1);
    chevron.closePath();
    const arrow = new THREE.Mesh(
        new THREE.ShapeGeometry(chevron),
        new THREE.MeshBasicMaterial({ color: ACCENT, transparent: true, opacity: 0.55 }),
    );
    arrow.rotation.x = -Math.PI / 2;
    arrow.position.set(0.18, clearance + height + 0.001, 0);
    robot.add(arrow);

    return robot;
}

/**
 * Builds the camera marker: a body, a lens and a wireframe view cone along
 * the camera's forward axis (+X in its own frame).
 * @returns {{group: THREE.Group, setOverlay: (overlay: boolean) => void}}
 */
function createCameraMarker() {
    const group = new THREE.Group();
    const parts = new THREE.Group();
    parts.scale.setScalar(1.4);
    group.add(parts);
    const bodyMaterial = new THREE.MeshStandardMaterial({
        color: 0xe5e7eb,
        roughness: 0.5,
        transparent: true,
    });
    const lensMaterial = new THREE.MeshStandardMaterial({
        color: CAMERA_COLOR,
        roughness: 0.3,
        transparent: true,
    });

    const body = new THREE.Mesh(new THREE.BoxGeometry(0.035, 0.04, 0.05), bodyMaterial);
    parts.add(body);
    const lens = new THREE.Mesh(new THREE.CylinderGeometry(0.013, 0.013, 0.02, 20), lensMaterial);
    lens.rotation.z = -Math.PI / 2;
    lens.position.x = 0.027;
    parts.add(lens);

    const reach = 0.16;
    const halfHeight = 0.05;
    const halfWidth = 0.07;
    const tip = new THREE.Vector3(0.037, 0, 0);
    const corners = [
        [reach, halfHeight, halfWidth],
        [reach, halfHeight, -halfWidth],
        [reach, -halfHeight, -halfWidth],
        [reach, -halfHeight, halfWidth],
    ].map(([x, y, z]) => new THREE.Vector3(x, y, z));
    const points = [];
    corners.forEach((corner, index) => {
        points.push(tip, corner, corner, corners[(index + 1) % corners.length]);
    });
    const frustumMaterial = new THREE.LineBasicMaterial({
        color: CAMERA_COLOR,
        transparent: true,
        opacity: 0.8,
    });
    const frustum = new THREE.LineSegments(
        new THREE.BufferGeometry().setFromPoints(points),
        frustumMaterial,
    );
    parts.add(frustum);

    const materials = [bodyMaterial, lensMaterial, frustumMaterial];
    const setOverlay = (overlay) => {
        materials.forEach((material) => {
            material.depthTest = !overlay;
        });
        parts.children.forEach((part) => {
            part.renderOrder = overlay ? OVERLAY_ORDER.camera : 0;
        });
    };

    return { group, setOverlay };
}

/**
 * A screen-facing disc in the background color with a dashed rim, drawn over
 * the robot so a camera inside it shows through, like a cutaway.
 * @returns {THREE.Sprite}
 */
function createCutaway() {
    const size = 256;
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    const context = canvas.getContext("2d");
    const center = size / 2;
    const radius = center - 6;
    context.fillStyle = `#${new THREE.Color(BACKGROUND).getHexString()}`;
    context.beginPath();
    context.arc(center, center, radius, 0, Math.PI * 2);
    context.fill();
    context.strokeStyle = "#22d3ee";
    context.lineWidth = 5;
    context.setLineDash([16, 10]);
    context.stroke();

    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    const sprite = new THREE.Sprite(
        new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false }),
    );
    sprite.scale.set(0.16, 0.16, 1);
    sprite.renderOrder = OVERLAY_ORDER.cutaway;
    return sprite;
}

/**
 * Creates the camera position preview inside ``container``.
 * @param {HTMLElement} container - Element the canvas is added to.
 * @returns {{render: () => void, resize: () => void, setPose: (extrinsics: Object) => void}}
 */
export function createCameraPoseScene(container) {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(BACKGROUND);

    const width = Math.max(container.clientWidth, 1);
    const height = Math.max(container.clientHeight, 1);
    const view = new THREE.PerspectiveCamera(45, width / height, 0.01, 100);
    view.position.set(0.95, 0.75, 1.1);

    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "default" });
    renderer.setPixelRatio(Math.min(globalThis.devicePixelRatio || 1, 2));
    renderer.setSize(width, height);
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(view, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 0.4;
    controls.maxDistance = 4;
    controls.target.set(0, 0.08, 0);

    scene.add(new THREE.HemisphereLight(0xffffff, 0x303030, 1.6));
    const sun = new THREE.DirectionalLight(0xffffff, 1.4);
    sun.position.set(1.5, 3, 1);
    scene.add(sun);

    const grid = new THREE.GridHelper(3, 30, 0x3a3a3a, 0x2a2a2a);
    grid.position.y = -0.001;
    scene.add(grid);

    const robot = createRobot();
    scene.add(robot);
    // The footprint from the floor up, so a camera at zero height counts too.
    const robotBounds = new THREE.Box3().setFromObject(robot);
    robotBounds.min.y = 0;

    const frontLabel = createLabel(ACCENT);
    frontLabel.setText("FRONT");
    frontLabel.sprite.position.set(CHASSIS.length / 2 + BUMPER.thickness + 0.16, 0, 0);
    frontLabel.sprite.renderOrder = 0;
    scene.add(frontLabel.sprite);

    const marker = createCameraMarker();
    scene.add(marker.group);
    const cutaway = createCutaway();
    scene.add(cutaway);

    const dimensions = {
        x: createDimensionLine(AXIS_COLORS.x, [0.5, 1.3]),
        y: createDimensionLine(AXIS_COLORS.y, [-0.1, 0.5]),
        z: createDimensionLine(AXIS_COLORS.z, [1.15, 0.5]),
    };
    Object.values(dimensions).forEach(({ group }) => scene.add(group));

    const origin = new THREE.Mesh(new THREE.SphereGeometry(0.008, 12, 8), overlayMaterial(0xffffff));
    origin.renderOrder = OVERLAY_ORDER.dimension;
    scene.add(origin);

    const atX = new THREE.Vector3();
    const atXY = new THREE.Vector3();
    const setPose = (extrinsics) => {
        mountingPoseToViewMatrix(extrinsics).decompose(
            marker.group.position,
            marker.group.quaternion,
            marker.group.scale,
        );
        const position = marker.group.position;

        // Walk from the origin along X, then Y on the floor, then up Z.
        atX.set(position.x, 0, 0);
        atXY.set(position.x, 0, position.z);
        const format = (axis, value) => `${axis} ${value.toFixed(3)} m`;
        dimensions.x.update(origin.position, atX, format("X", extrinsics.x_offset ?? 0));
        dimensions.y.update(atX, atXY, format("Y", extrinsics.y_offset ?? 0));
        dimensions.z.update(atXY, position, format("Z", extrinsics.z_offset ?? 0));

        const inside = robotBounds.containsPoint(position);
        cutaway.visible = inside;
        cutaway.position.copy(position);
        marker.setOverlay(inside);
    };

    return {
        render: () => {
            controls.update();
            renderer.render(scene, view);
        },
        resize: () => {
            const newWidth = Math.max(container.clientWidth, 1);
            const newHeight = Math.max(container.clientHeight, 1);
            view.aspect = newWidth / newHeight;
            view.updateProjectionMatrix();
            renderer.setSize(newWidth, newHeight);
        },
        setPose,
    };
}
