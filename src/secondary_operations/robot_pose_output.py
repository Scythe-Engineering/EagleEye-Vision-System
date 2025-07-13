from src.webui.web_server import EagleEyeInterface
import numpy as np


class RobotPoseOutput:
    def __init__(self, web_interface: EagleEyeInterface):
        """Output the robot pose to the web interface."""
        self.web_interface = web_interface

    def run(self, pose: np.ndarray) -> None:
        """Output the robot pose to the web interface.

        Returns:
            np.ndarray: 4x4 pose matrix with no z position component.
        """
        self.web_interface.update_robot_position(pose)