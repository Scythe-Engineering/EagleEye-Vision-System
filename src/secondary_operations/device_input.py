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

        run(self, input) -> np.ndarray | None

    Input:
        _input_data (Any): Unused parameter. Data source operations don't consume
            input from previous pipeline stages. The camera source is determined
            by the `bus_id` constructor parameter.

    Output:
        np.ndarray | None: The current camera frame as a numpy array in BGR format
            with rotation applied if configured. Returns None if the camera is
            unavailable or no frame has been captured yet. The array shape is
            (height, width, 3) for color images.

    Example:
        >>> device_input = DeviceInput(web_interface, compute_pool, camera_manager, "1", 90)
        >>> frame = device_input.run(None)
        >>> if frame is not None:
        ...     print(frame.shape)  # e.g., (720, 1280, 3)
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
        """Initialize the device input operation.

        Args:
            web_interface: Web interface for runtime updates.
            compute_pool: Compute pool available for device operations.
            camera_manager: Camera thread manager to fetch frames from.
            bus_id: USB bus identifier for the camera to read frames from.
            frame_rotation: Rotation angle in degrees (0, 90, 180, or 270). Defaults to 0.
        """
        self.web_interface = web_interface
        self.compute_pool = compute_pool
        self.camera_manager = camera_manager
        self.bus_id = bus_id
        self.frame_rotation = self._normalize_rotation(frame_rotation)

    def _normalize_rotation(self, rotation: int) -> int:
        """Normalize and validate rotation value.

        Args:
            rotation: Raw rotation value from config.

        Returns:
            Normalized rotation in {0, 90, 180, 270}.

        Raises:
            ValueError: If rotation cannot be normalized to a valid value.
        """
        normalized = ((rotation % 360) + 360) % 360
        if normalized not in self.VALID_ROTATIONS:
            raise ValueError(
                f"Invalid frame_rotation {rotation}. Must be one of {sorted(self.VALID_ROTATIONS)}"
            )
        return normalized

    def _apply_rotation(self, frame: np.ndarray) -> np.ndarray:
        """Apply configured rotation to a frame.

        Args:
            frame: Input frame to rotate.

        Returns:
            Rotated frame (or original if rotation is 0).
        """
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
        """Fetch the current timestamped frame from the configured camera.

        Args:
            _input_data: Unused (data source operations don't use input).

        Returns:
            Timestamped frame packet with rotation applied, or None if unavailable.
        """
        get_packet = getattr(self.camera_manager, "get_current_packet_by_bus_id", None)
        if callable(get_packet):
            packet = get_packet(self.bus_id)
            if packet is None:
                return None
            return TimedValue(self._apply_rotation(packet.value), packet.timing)

        frame_result = self.camera_manager.get_current_frame_by_bus_id(self.bus_id)
        if frame_result is not None:
            frame, _ = frame_result
            return self._apply_rotation(frame)
        return None

    def update_config(self, json_config: dict[str, Any]) -> None:
        """Update runtime-configurable settings for the operation.

        Args:
            json_config: Runtime configuration overrides. Supports:
                - bus_id: Changes the camera source (requires restart).
                - frame_rotation: Changes rotation angle (applied immediately).
        """
        self.bus_id = json_config.get("bus_id", self.bus_id)

        if "frame_rotation" in json_config:
            try:
                new_rotation = self._normalize_rotation(json_config["frame_rotation"])
                if new_rotation != self.frame_rotation:
                    self.frame_rotation = new_rotation
            except ValueError:
                pass
