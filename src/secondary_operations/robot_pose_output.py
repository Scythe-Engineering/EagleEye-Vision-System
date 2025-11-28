import numpy as np

from src.webui.web_server import EagleEyeInterface


class RobotPoseOutput:
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

    def visualize(self, frame: np.ndarray) -> None:
        """Visualize the robot pose output.

        This operation outputs pose data to the web interface only,
        so no frame visualization is available.

        Args:
            frame: Input frame (unused).

        Returns:
            None - no visualization available for output-only operations.
        """
        return None

    def update_config(self, _: dict) -> None:
        """Update the configuration of the robot pose output. No live-updatable parameters available.

        Args:
            json_config: JSON configuration for the robot pose output (ignored).
        """
        pass
