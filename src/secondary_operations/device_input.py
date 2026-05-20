from typing import Any, TYPE_CHECKING

import cv2
import numpy as np

from src.main_operations.definitions.base.base_class import OperationInstance
from src.utils.device_management_utils.compute_pool import ComputePool
from src.utils.timing import FramePacket, TimedValue
from src.webui.web_server import EagleEyeInterface

if TYPE_CHECKING:
    from src.utils.camera_utils.camera_thread_manager import CameraThreadManager


class DeviceInput(OperationInstance):
    """Device input operation that fetches frames from a specified camera.

    This operation handles frame rotation, which is configured at the operation
    level rather than the camera level. This allows different pipelines to
    apply different rotations to the same camera source.

    The run() method signature:

        run(self, input) -> FramePacket | None

    Input:
        _input_data (Any): Unused parameter. Data source operations don't consume
            input from previous pipeline stages. The camera source is determined
            by the `bus_id` constructor parameter.

    Output:
        FramePacket | None: The current camera frame wrapped with capture timing
            metadata. The frame value is a numpy array in BGR format with rotation
            applied if configured. Returns None if the camera is unavailable or
            no frame has been captured yet.
    """

    VALID_ROTATIONS = {0, 90, 180, 270}

    def __init__(
        self,
        web_interface: EagleEyeInterface,
        compute_pool: ComputePool,
        camera_manager: "CameraThreadManager",
        bus_id: str,
        frame_rotation: int = 0,
    ) -> None:
        """Initialize the object.
        
        Args:
            web_interface (EagleEyeInterface): Web interface.
            compute_pool (ComputePool): Compute pool.
            camera_manager ('CameraThreadManager'): Camera manager.
            bus_id (str): Bus id.
            frame_rotation (int): Frame rotation."""
        self.web_interface = web_interface
        self.compute_pool = compute_pool
        self.camera_manager = camera_manager
        self.bus_id = bus_id
        self.frame_rotation = self._normalize_rotation(frame_rotation)

    def _normalize_rotation(self, rotation: int) -> int:
        """Normalize rotation.
        
        Args:
            rotation (int): Rotation.
        
        Returns:
            int: Result of normalize rotation."""
        normalized = ((rotation % 360) + 360) % 360
        if normalized not in self.VALID_ROTATIONS:
            raise ValueError(
                f"Invalid frame_rotation {rotation}. Must be one of {sorted(self.VALID_ROTATIONS)}"
            )
        return normalized

    def _apply_rotation(self, frame: np.ndarray) -> np.ndarray:
        """Apply rotation.
        
        Args:
            frame (np.ndarray): Frame.
        
        Returns:
            np.ndarray: Result of apply rotation."""
        if self.frame_rotation == 0:
            return frame
        elif self.frame_rotation == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.frame_rotation == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif self.frame_rotation == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def run(self, _input_data: Any) -> FramePacket | None:
        """Run.
        
        Args:
            _input_data (Any):  input data.
        
        Returns:
            FramePacket | None: Result of run."""
        packet = self.camera_manager.get_current_packet_by_bus_id(self.bus_id)
        if packet is None:
            return None
        return TimedValue(self._apply_rotation(packet.value), packet.timing)

    def update_config(self, json_config: dict[str, Any]) -> None:
        """Update config.
        
        Args:
            json_config (dict[str, Any]): Json config."""
        self.bus_id = json_config.get("bus_id", self.bus_id)

        if "frame_rotation" in json_config:
            try:
                new_rotation = self._normalize_rotation(json_config["frame_rotation"])
                if new_rotation != self.frame_rotation:
                    self.frame_rotation = new_rotation
            except ValueError:
                pass
