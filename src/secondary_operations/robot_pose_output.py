import numpy as np

from src.webui.web_server import EagleEyeInterface
from src.secondary_operations.base_class import SecondaryOperation


class RobotPoseOutput(SecondaryOperation):
    def __init__(self, web_interface: EagleEyeInterface) -> None:
        """Output the robot pose to the web interface.

        Args:
            web_interface: Web interface responsible for pushing updates.
            pose_change_epsilon: Absolute tolerance for considering two poses identical.
        """
        self.web_interface = web_interface
        self._last_sent_pose: np.ndarray | None = None

    def run(self, pose: np.ndarray) -> np.ndarray | None:
        """Output the robot pose to the web interface."""
        if self._last_sent_pose is not None and np.array_equal(
            self._last_sent_pose, pose
        ):
            return None
        self.web_interface.update_robot_position(pose)
        self._last_sent_pose = pose.copy()
        return pose
