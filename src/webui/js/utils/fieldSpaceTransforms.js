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

    const resultMatrix = new Matrix4();
    resultMatrix.set(
        transformMatrix[0][0],
        transformMatrix[0][2],
        transformMatrix[0][1],
        (transformMatrix[0][3] - FIELD_CENTER_X_METERS) * METERS_TO_VIEW_UNITS,
        transformMatrix[2][0],
        transformMatrix[2][2],
        transformMatrix[2][1],
        transformMatrix[2][3] * METERS_TO_VIEW_UNITS,
        -transformMatrix[1][0],
        -transformMatrix[1][2],
        -transformMatrix[1][1],
        (-transformMatrix[1][3] + FIELD_CENTER_Y_METERS) * METERS_TO_VIEW_UNITS,
        transformMatrix[3][0],
        transformMatrix[3][1],
        transformMatrix[3][2],
        transformMatrix[3][3],
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
