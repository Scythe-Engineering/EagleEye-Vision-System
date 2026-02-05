import time
from typing import Any
import numpy as np

from src.webui.web_server import EagleEyeInterface
from src.main_operations.definitions.base.base_class import OperationInstance


class FpsLimiter(OperationInstance):
    """Limits the frame rate by sleeping to maintain a target FPS.

    This operation records the time between runs and sleeps for the remaining
    time needed to achieve the target frames per second. The input frame is
    passed through unchanged.
    """

    def __init__(self, web_interface: EagleEyeInterface, fps: float) -> None:
        """Initialize the FPS limiter.

        Args:
            web_interface: Web interface for potential future visualization.
            fps: Target frames per second to maintain.
        """
        self.web_interface = web_interface
        self.target_fps = fps
        self.target_interval_seconds = 1.0 / fps
        self.last_run_time: float | None = None

    def run(self, frame: np.ndarray) -> np.ndarray:
        """Limit FPS by sleeping if necessary, then return the frame unchanged.

        Args:
            frame: Input frame to process.

        Returns:
            The input frame unchanged after potential sleep delay.
        """
        current_time = time.time()

        if self.last_run_time is not None:
            elapsed_time = current_time - self.last_run_time
            sleep_time = self.target_interval_seconds - elapsed_time

            if sleep_time > 0:
                time.sleep(sleep_time)

        self.last_run_time = time.time()
        return frame

    def update_config(self, json_config: dict[str, Any]) -> None:
        """Update configuration at runtime.

        Args:
            json_config: Mapping of parameter names to new values.
        """
        target_fps_updated = False

        for key, value in json_config.items():
            if key == "fps":
                self.target_fps = value
                target_fps_updated = True
                continue
            if hasattr(self, key):
                setattr(self, key, value)
                if key == "target_fps":
                    target_fps_updated = True

        if target_fps_updated:
            if not isinstance(self.target_fps, (int, float)) or self.target_fps <= 0:
                raise ValueError("target_fps must be a positive number")
            self.target_interval_seconds = 1.0 / self.target_fps
