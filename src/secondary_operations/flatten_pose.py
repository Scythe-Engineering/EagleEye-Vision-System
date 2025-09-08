from line_profiler import profile
import numpy as np


class FlattenPose:
    """Flatten a pose matrix to have no z position component and preserve only y-axis rotation."""

    @profile
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

    def visualize(self, frame: np.ndarray) -> None:
        """Visualize the flatten pose outputs.

        This operation returns pose transformation data only,
        so no frame visualization is available.

        Args:
            frame: Input frame (unused).

        Returns:
            None - no visualization available for transform-only operations.
        """
        return None

    def update_config(self, _: dict) -> None:
        """Update the configuration of the flatten pose operation. No live-updatable parameters available.

        Args:
            json_config: JSON configuration for the flatten pose operation (ignored).
        """
        pass
