import numpy as np
from src.secondary_operations.base_class import SecondaryOperation


class FlattenPose(SecondaryOperation):
    """Flatten a pose matrix to have no z position component and preserve only y-axis rotation."""

    def run(self, pose: np.ndarray) -> np.ndarray:
        """Flatten the pose matrix to 2D by removing z position and x/y rotations.

        This method sets z position to 0 and removes x-axis (roll) and
        y-axis (pitch) rotations while preserving z-axis (yaw) rotation.

        Args:
            pose: 4x4 transformation matrix representing a 3D pose.

        Returns:
            4x4 pose matrix flattened to 2D with only z-axis rotation preserved.
        """
        flattened_pose = pose.copy()
        flattened_pose[2, 3] = 0.0

        yaw_angle = np.arctan2(flattened_pose[1, 0], flattened_pose[0, 0])
        cos_yaw = np.cos(yaw_angle)
        sin_yaw = np.sin(yaw_angle)

        flattened_pose[0, 0] = cos_yaw
        flattened_pose[0, 1] = -sin_yaw
        flattened_pose[0, 2] = 0.0
        flattened_pose[1, 0] = sin_yaw
        flattened_pose[1, 1] = cos_yaw
        flattened_pose[1, 2] = 0.0
        flattened_pose[2, 0] = 0.0
        flattened_pose[2, 1] = 0.0
        flattened_pose[2, 2] = 1.0

        return flattened_pose
