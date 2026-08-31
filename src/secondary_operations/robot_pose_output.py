from time import monotonic

import numpy as np

from src.main_operations.definitions.base.base_class import OperationInstance
from src.webui.web_server import EagleEyeInterface


class RobotPoseOutput(OperationInstance):
    def __init__(self, web_interface: EagleEyeInterface) -> None:
        """Output the robot pose to the web interface.

        Args:
            web_interface: Web interface responsible for pushing updates.
        """
        self.web_interface = web_interface
        self._last_sent_pose: np.ndarray | None = None
        self._last_sent_at = 0.0

    def run(self, pose: np.ndarray) -> np.ndarray | None:
        """Output changed poses and periodic snapshots to the web interface.

        Args:
            pose: Robot pose transformation matrix.
        """
        now = monotonic()
        if (
            self._last_sent_pose is not None
            and np.array_equal(self._last_sent_pose, pose)
            and now - self._last_sent_at < 1.0
        ):
            return None
        self.web_interface.update_robot_position(pose)
        self._last_sent_pose = pose.copy()
        self._last_sent_at = now
        return pose
