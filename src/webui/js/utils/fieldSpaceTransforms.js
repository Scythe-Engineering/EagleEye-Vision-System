import { Matrix4, Vector3 } from "three";

// Utilities for converting camera and position data into field-space units.
export const FIELD_CENTER_X_METERS = 8.774125;
export const FIELD_CENTER_Y_METERS = 4.025901;
export const METERS_TO_VIEW_UNITS = 1000;

/**
 * Converts a 4x4 camera pose matrix into field-space coordinates.
 *
 * @param {number[][]} transformMatrix - 4x4 transform matrix in source space.
 * @returns {Matrix4|null} The converted field-space matrix, or null for invalid input.
 */
export function cameraPoseToFieldSpaceMatrix(transformMatrix) {
    if (
        !Array.isArray(transformMatrix) ||
        transformMatrix.length !== 4 ||
        !transformMatrix.every(
            (row) =>
                Array.isArray(row) &&
                row.length === 4 &&
                row.every(
                    (value) =>
                        typeof value === "number" && Number.isFinite(value),
                ),
        )
    ) {
        return null;
    }

    // Field NWU -> view X/right, Y/up, Z/back: A = [X, Z, -Y].
    // The incoming pose has EDN local axes. Convert them to NWU, then
    // express both local and field axes in the same Three.js basis.
    // R_view = A * R_field_from_EDN * B^T * A^T. This is a proper
    // rotation (det +1); swapping columns alone introduces a reflection.
    const r = transformMatrix;
    const resultMatrix = new Matrix4();
    resultMatrix.set(
        r[0][2], -r[0][1], r[0][0],
        (r[0][3] - FIELD_CENTER_X_METERS) * METERS_TO_VIEW_UNITS,
        r[2][2], -r[2][1], r[2][0],
        r[2][3] * METERS_TO_VIEW_UNITS,
        -r[1][2], r[1][1], -r[1][0],
        (-r[1][3] + FIELD_CENTER_Y_METERS) * METERS_TO_VIEW_UNITS,
        0, 0, 0, 1,
    );
    return resultMatrix;
}

/**
 * Converts a 3D position into a field-space vector.
 *
 * @param {number[]} position3D - [x, y, z] position in meters.
 * @returns {Vector3|null} The converted field-space vector, or null for invalid input.
 */
export function position3DToFieldSpaceVector(position3D) {
    if (
        !Array.isArray(position3D) ||
        position3D.length !== 3 ||
        position3D.some(
            (value) => typeof value !== "number" || !Number.isFinite(value),
        )
    ) {
        return null;
    }

    return new Vector3(
        (position3D[0] - FIELD_CENTER_X_METERS) * METERS_TO_VIEW_UNITS,
        position3D[2] * METERS_TO_VIEW_UNITS,
        (-position3D[1] + FIELD_CENTER_Y_METERS) * METERS_TO_VIEW_UNITS,
    );
}

/** Mount in meters/degrees -> Three.js camera marker pose. Positive pitch is down. */
export function mountingPoseToViewMatrix({x_offset = 0, y_offset = 0, z_offset = 0,
    yaw = 0, pitch = 0, roll = 0} = {}) {
    const toRad = Math.PI / 180;
    return new Matrix4().makeTranslation(x_offset, z_offset, -y_offset)
        .multiply(new Matrix4().makeRotationY(yaw * toRad))
        .multiply(new Matrix4().makeRotationZ(-pitch * toRad))
        .multiply(new Matrix4().makeRotationX(roll * toRad));
}

// Bundled glTF robots use +Y up and +Z forward; the view pose uses +Y up,
// +X forward. This changes the asset axes only, not the measured robot pose.
const robotAssetToView = new Matrix4().makeRotationY(Math.PI / 2);

/** Places a Y-up, +Z-forward robot GLB using the converted view pose. */
export function robotViewPoseToModelMatrix(viewPose, scaleMatrix, target = new Matrix4()) {
    if (viewPose) {
        target.extractRotation(viewPose);
        target.copyPosition(viewPose);
    } else {
        target.identity();
    }
    return target.multiply(robotAssetToView).multiply(scaleMatrix);
}
