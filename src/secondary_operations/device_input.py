from typing import Any, Mapping, TYPE_CHECKING
import numpy as np
from src.main_operations.definitions.base.base_class import OperationInstance
from src.utils.device_management_utils.compute_pool import ComputePool
from src.webui.web_server import EagleEyeInterface

if TYPE_CHECKING:
    from src.utils.camera_utils.camera_thread_manager import CameraThreadManager


class DeviceInput(OperationInstance):
    """Device input operation that fetches frames from a specified camera."""

    def __init__(
        self,
        web_interface: EagleEyeInterface,
        compute_pool: ComputePool,
        camera_manager: "CameraThreadManager",
        camera_name: str,
    ) -> None:
        """Initialize the device input operation.

        Args:
            web_interface: Web interface for runtime updates.
            compute_pool: Compute pool available for device operations.
            camera_manager: Camera thread manager to fetch frames from.
            camera_name: Name of the camera to read frames from.
        """
        self.web_interface = web_interface
        self.compute_pool = compute_pool
        self.camera_manager = camera_manager
        self.camera_name = camera_name

    def run(self, _input_data: Any) -> np.ndarray | None:
        """Fetch the current frame from the configured camera.

        Args:
            _input_data: Unused (data source operations don't use input).

        Returns:
            Current camera frame as numpy array, or None if camera unavailable.
        """
        frame_result = self.camera_manager.get_current_frame(self.camera_name)
        if frame_result is not None:
            frame, _ = frame_result
            return frame
        return None

    def update_config(self, config: Mapping[str, Any]) -> None:
        """Update runtime-configurable settings for the operation.

        Args:
            config: Runtime configuration overrides.
        """
        self.camera_name = config.get("camera_name", self.camera_name)
