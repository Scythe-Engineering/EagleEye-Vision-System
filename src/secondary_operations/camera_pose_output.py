from __future__ import annotations

import numpy as np

from src.main_operations.definitions.base.base_class import OperationInstance
from src.webui.web_server import EagleEyeInterface


class CameraPoseOutput(OperationInstance):
    """Publish camera poses to the WebUI for 3D visualization."""

    def __init__(
        self, camera_bus_id: str, web_interface: EagleEyeInterface
    ) -> None:
        """Initialize the camera pose output operation.

        Args:
            camera_bus_id: Stable identifier for the camera represented by this op.
            web_interface: Web interface responsible for pushing updates.
        """
        self.camera_bus_id = str(camera_bus_id)
        self.web_interface = web_interface
        self._last_sent_pose: np.ndarray | None = None

    def run(self, camera_pose: np.ndarray | None) -> np.ndarray | None:
        """Publish a camera pose update and pass the pose through.

        Args:
            camera_pose: 4x4 camera pose transform, or None.

        Returns:
            The original pose for downstream chaining, or None when invalid.
        """
        if camera_pose is None:
            return None

        camera_pose_matrix = np.asarray(camera_pose, dtype=float)
        if camera_pose_matrix.shape != (4, 4) or not np.all(
            np.isfinite(camera_pose_matrix)
        ):
            return None

        if self._last_sent_pose is not None and np.array_equal(
            self._last_sent_pose, camera_pose_matrix
        ):
            return camera_pose

        self.web_interface.update_camera_pose(self.camera_bus_id, camera_pose_matrix)
        self._last_sent_pose = camera_pose_matrix.copy()
        return camera_pose
