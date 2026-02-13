import numpy as np
from typing import List


def rotation_matrix_to_quaternion(rotation_matrix: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion [w, x, y, z].

    Args:
        rotation_matrix: 3x3 rotation matrix.

    Returns:
        Quaternion as [w, x, y, z].
    """
    trace = np.trace(rotation_matrix)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) * s
        y = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) * s
        z = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) * s
    elif rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2])
        w = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
        x = 0.25 * s
        y = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
        z = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
    elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2])
        w = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
        x = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
        y = 0.25 * s
        z = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1])
        w = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
        x = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
        y = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
        z = 0.25 * s

    return np.array([w, x, y, z])


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to rotation matrix.

    Args:
        q: Quaternion as [w, x, y, z].

    Returns:
        3x3 rotation matrix.
    """
    q = q / np.linalg.norm(q)
    w, x, y, z = q

    rotation_matrix = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])

    return rotation_matrix


def quaternion_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    """Compute angular distance between two quaternions.

    Args:
        q1: First quaternion [w, x, y, z].
        q2: Second quaternion [w, x, y, z].

    Returns:
        Angular distance in radians.
    """
    dot_product = np.abs(np.dot(q1, q2))
    dot_product = np.clip(dot_product, -1.0, 1.0)
    return 2.0 * np.arccos(dot_product)


def average_quaternions(quaternions: List[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """Compute weighted average of quaternions.

    Args:
        quaternions: List of quaternions [w, x, y, z].
        weights: Weight for each quaternion.

    Returns:
        Averaged quaternion [w, x, y, z].
    """
    quaternion_array = np.array(quaternions)

    if quaternion_array[0, 0] < 0:
        quaternion_array[0] = -quaternion_array[0]

    for i in range(1, len(quaternion_array)):
        if np.dot(quaternion_array[i], quaternion_array[0]) < 0:
            quaternion_array[i] = -quaternion_array[i]

    weighted_sum = np.sum(quaternion_array * weights[:, np.newaxis], axis=0)

    norm = np.linalg.norm(weighted_sum)
    if norm < 1e-12:
        avg_quat = quaternion_array[0] / np.linalg.norm(quaternion_array[0])
    else:
        avg_quat = weighted_sum / norm

    return avg_quat
