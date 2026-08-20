import numpy as np
from typing import List, Optional, Any
from src.main_operations.definitions.base.base_class import OperationInstance
from src.webui.web_server import EagleEyeInterface
from src.utils.quaternion_utils import (
    rotation_matrix_to_quaternion,
    quaternion_to_rotation_matrix,
    quaternion_distance,
    average_quaternions,
)


class PoseFusion(OperationInstance):
    """Fuse multiple pose estimates into a single consensus pose.

    Uses weighted averaging with outlier rejection to combine pose estimates
    from multiple sources. When 4+ inputs are available, poses significantly
    distant from the cluster center are rejected before averaging.
    """

    def __init__(
        self,
        web_interface: EagleEyeInterface,
        outlier_threshold: float = 1.0,
        rotation_weight: float = 0.5,
    ) -> None:
        """Initialize pose fusion operation.

        Args:
            web_interface: Web interface for runtime updates.
            outlier_threshold: Distance threshold for outlier rejection.
            rotation_weight: Weight factor for rotation distance relative to translation.
        """
        self.web_interface = web_interface
        self.outlier_threshold = outlier_threshold
        self.rotation_weight = rotation_weight

    def update_config(self, json_config: dict) -> None:
        """Update operation parameters from JSON configuration.

        Args:
            json_config: Configuration dictionary with updated parameters.
        """
        if "outlier_threshold" in json_config:
            self.outlier_threshold = float(json_config["outlier_threshold"])
        if "rotation_weight" in json_config:
            self.rotation_weight = float(json_config["rotation_weight"])

    def _validate_pose(self, pose: Any) -> Optional[np.ndarray]:
        """Validate and convert pose to numpy array.

        Args:
            pose: Input pose, expected to be 4x4 transformation matrix.

        Returns:
            Valid 4x4 numpy array or None if invalid.
        """
        if pose is None:
            return None

        try:
            pose_array = np.asarray(pose, dtype=float)
        except Exception:
            return None

        if pose_array.shape != (4, 4):
            return None

        if not np.all(np.isfinite(pose_array)):
            return None

        return pose_array

    def _compute_pose_distance(
        self, pose1: np.ndarray, pose2: np.ndarray
    ) -> float:
        """Compute distance between two poses.

        Distance is computed as weighted combination of translation and rotation distance.

        Args:
            pose1: First 4x4 transformation matrix.
            pose2: Second 4x4 transformation matrix.

        Returns:
            Combined distance metric.
        """
        trans1 = pose1[:3, 3]
        trans2 = pose2[:3, 3]
        translation_dist = np.linalg.norm(trans1 - trans2)

        q1 = rotation_matrix_to_quaternion(pose1[:3, :3])
        q2 = rotation_matrix_to_quaternion(pose2[:3, :3])
        rotation_dist = quaternion_distance(q1, q2) * self.rotation_weight

        return translation_dist + rotation_dist

    def _compute_cluster_center(self, poses: List[np.ndarray]) -> np.ndarray:
        """Compute rough cluster center by averaging poses.

        Args:
            poses: List of 4x4 transformation matrices.

        Returns:
            Average pose (4x4 transformation matrix).
        """
        translations = np.array([pose[:3, 3] for pose in poses])
        avg_translation = np.mean(translations, axis=0)

        quaternions = [rotation_matrix_to_quaternion(pose[:3, :3]) for pose in poses]
        weights = np.ones(len(quaternions)) / len(quaternions)
        avg_quaternion = average_quaternions(quaternions, weights)

        center_pose = np.eye(4)
        center_pose[:3, :3] = quaternion_to_rotation_matrix(avg_quaternion)
        center_pose[:3, 3] = avg_translation

        return center_pose

    def _reject_outliers(self, poses: List[np.ndarray]) -> List[np.ndarray]:
        """Reject poses that are outliers from the cluster center.

        Only performs rejection when 4+ poses are available.
        Uses median-based approach to avoid outliers affecting the center.

        Args:
            poses: List of 4x4 transformation matrices.

        Returns:
            Filtered list with outliers removed.
        """
        if len(poses) < 4:
            return poses

        translations = np.array([pose[:3, 3] for pose in poses])
        median_translation = np.median(translations, axis=0)

        quaternions = [rotation_matrix_to_quaternion(pose[:3, :3]) for pose in poses]
        quaternion_array = np.array(quaternions)
        if quaternion_array[0, 0] < 0:
            quaternion_array[0] = -quaternion_array[0]
        for i in range(1, len(quaternion_array)):
            if np.dot(quaternion_array[i], quaternion_array[0]) < 0:
                quaternion_array[i] = -quaternion_array[i]
        median_quaternion = np.median(quaternion_array, axis=0)
        median_quaternion = median_quaternion / np.linalg.norm(median_quaternion)

        median_pose = np.eye(4)
        median_pose[:3, :3] = quaternion_to_rotation_matrix(median_quaternion)
        median_pose[:3, 3] = median_translation

        distances = [
            self._compute_pose_distance(pose, median_pose) for pose in poses
        ]

        inliers = [
            pose for pose, dist in zip(poses, distances)
            if dist <= self.outlier_threshold
        ]

        return inliers if len(inliers) > 0 else poses

    def _compute_weights(
        self, poses: List[np.ndarray], center: np.ndarray
    ) -> np.ndarray:
        """Compute weights for each pose based on distance from center.

        Weights decrease as distance from center increases.

        Args:
            poses: List of 4x4 transformation matrices.
            center: Cluster center pose.

        Returns:
            Array of normalized weights.
        """
        distances = np.array([
            self._compute_pose_distance(pose, center) for pose in poses
        ])

        epsilon = 1e-6
        weights = 1.0 / (distances + epsilon)

        weights = weights / np.sum(weights)

        return weights

    def _weighted_average_poses(self, poses: List[np.ndarray]) -> np.ndarray:
        """Compute weighted average of poses.

        Args:
            poses: List of 4x4 transformation matrices.

        Returns:
            Weighted average pose (4x4 transformation matrix).
        """
        if len(poses) == 1:
            return poses[0]

        cluster_center = self._compute_cluster_center(poses)
        weights = self._compute_weights(poses, cluster_center)

        translations = np.array([pose[:3, 3] for pose in poses])
        avg_translation = np.average(translations, axis=0, weights=weights)

        quaternions = [rotation_matrix_to_quaternion(pose[:3, :3]) for pose in poses]
        avg_quaternion = average_quaternions(quaternions, weights)

        result_pose = np.eye(4)
        result_pose[:3, :3] = quaternion_to_rotation_matrix(avg_quaternion)
        result_pose[:3, 3] = avg_translation

        return result_pose

    def run(self, input_data: Any) -> Optional[np.ndarray]:
        """Fuse multiple pose estimates into a single consensus pose.

        Args:
            input_data: Either a single pose (4x4 array) or dict with multiple
                       pose inputs (e.g., {"pose_0": pose0, "pose_1": pose1, ...}).

        Returns:
            Fused 4x4 transformation matrix, or None if no valid poses.
        """
        valid_poses: List[np.ndarray] = []

        if isinstance(input_data, dict):
            for key in sorted(input_data.keys()):
                if key.startswith("pose"):
                    validated = self._validate_pose(input_data[key])
                    if validated is not None:
                        valid_poses.append(validated)
        else:
            validated = self._validate_pose(input_data)
            if validated is not None:
                valid_poses.append(validated)

        if len(valid_poses) == 0:
            return None

        if len(valid_poses) == 1:
            return valid_poses[0]

        inlier_poses = self._reject_outliers(valid_poses)

        fused_pose = self._weighted_average_poses(inlier_poses)

        return fused_pose
