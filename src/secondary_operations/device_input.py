from typing import Any, TYPE_CHECKING
import numpy as np
from src.main_operations.definitions.base.base_class import OperationInstance

if TYPE_CHECKING:
    from src.utils.camera_utils.camera_thread_manager import CameraThreadManager


class DeviceInput(OperationInstance):
    """Device input operation that fetches frames from a specified camera."""

    def __init__(self, camera_manager: "CameraThreadManager", camera_name: str) -> None:
        """Initialize the device input operation.

        Args:
            camera_manager: Camera thread manager to fetch frames from.
            camera_name: Name of the camera to read frames from.
        """
        self.camera_manager = camera_manager
        self.camera_name = camera_name

    def run(self, input_data: Any) -> np.ndarray | None:
        """Fetch the current frame from the configured camera.

        Args:
            input_data: Unused (data source operations don't use input).

        Returns:
            Current camera frame as numpy array, or None if camera unavailable.
        """
        frame_result = self.camera_manager.get_current_frame(self.camera_name)
        if frame_result is not None:
            frame, _ = frame_result
            return frame
        return None
