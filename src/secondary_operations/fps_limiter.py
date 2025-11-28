import time
import numpy as np

from src.webui.web_server import EagleEyeInterface


class FpsLimiter:
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

    def visualize(self, frame: np.ndarray) -> np.ndarray:
        """Visualize the pose outlier filter outputs.

        This operation returns pose estimation data (transform) only,
        so no frame visualization is available.

        Args:
            frame: Input frame (unused).

        Returns:
            None - no visualization available for transform-only operations.
        """
        return frame
