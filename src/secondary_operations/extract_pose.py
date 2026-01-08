import numpy as np
from typing import Optional, Dict
from src.main_operations.definitions.base.base_class import OperationInstance


class ExtractPose(OperationInstance):
    """Extract 2D pose data (position and rotation) from a 4x4 transformation matrix."""

    def __init__(self) -> None:
        """Initialize the extract pose operation."""
        pass

    def run(self, pose: Optional[np.ndarray]) -> Optional[Dict[str, float]]:
        """Extract the 2D pose (x, y, rotation) from a 4x4 transformation matrix.

        Args:
            pose: 4x4 transformation matrix, or None if pose estimation failed.

        Returns:
            Dictionary containing 'x', 'y', and 'rotation' keys with numeric values,
            or None if input pose is None.
        """
        if pose is None:
            return None

        if pose.shape != (4, 4):
            raise ValueError("Input pose must be a 4x4 transformation matrix")

        # Extract translation (x, y)
        x = float(pose[0, 3])
        y = float(pose[1, 3])

        # Extract rotation angle from the 2D rotation matrix (yaw around Z-axis)
        rotation_matrix = pose[:2, :2]
        rotation = float(np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))

        return {"x": x, "y": y, "rotation": rotation}
