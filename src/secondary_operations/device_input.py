from typing import Any, Callable, TYPE_CHECKING

import cv2
import numpy as np

from src.main_operations.definitions.base.base_class import OperationInstance
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
            by the `camera_bus_id` constructor parameter.

    Output:
        np.ndarray | None: The current camera frame as a numpy array in BGR format
            with rotation applied if configured. Returns None if the camera is
            unavailable or no frame has been captured yet. The array shape is
            (height, width, 3) for color images.

    Example:
        >>> device_input = DeviceInput(web_interface, camera_manager, "1", 90)
        >>> frame = device_input.run(None)
        >>> if frame is not None:
        ...     print(frame.shape)  # e.g., (720, 1280, 3)
    """

    VALID_ROTATIONS = {0, 90, 180, 270}

    def __init__(
        self,
        web_interface: EagleEyeInterface,
        camera_manager: "CameraThreadManager",
        camera_bus_id: str,
        frame_rotation: int = 0,
    ) -> None:
        """Initialize the device input operation.

        Args:
            web_interface: Web interface for runtime updates.
            camera_manager: Camera thread manager to fetch frames from.
            camera_bus_id: USB bus identifier for the camera to read frames from.
            frame_rotation: Rotation angle in degrees (0, 90, 180, or 270). Defaults to 0.
        """
        self.web_interface = web_interface
        self.camera_manager = camera_manager
        self.camera_bus_id = camera_bus_id
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
            packet = get_packet(self.camera_bus_id)
            if packet is None:
                return None
            return TimedValue(self._apply_rotation(packet.value), packet.timing)

        frame_result = self.camera_manager.get_current_frame_by_bus_id(
            self.camera_bus_id
        )
        if frame_result is not None:
            frame, _ = frame_result
            return self._apply_rotation(frame)
        return None

    def _require_running_worker(self) -> None:
        """Fail fast when the configured camera's worker is gone or stopped.

        Raises:
            RuntimeError: If the camera, its worker, or its thread has stopped.
        """
        camera_name = self.camera_manager.get_camera_name_by_bus_id(self.camera_bus_id)
        if camera_name is None:
            raise RuntimeError(f"Unknown camera bus ID {self.camera_bus_id!r}")
        worker = self.camera_manager.cameras.get(camera_name)
        if worker is None:
            raise RuntimeError(f"Unknown camera worker {camera_name!r}")
        # A stopped worker never publishes another frame, so waiting forever
        # would silently starve the docked asynchronous consumer.
        thread = getattr(worker, "thread", None)
        if not getattr(worker, "running", True) or (
            thread is not None and not thread.is_alive()
        ):
            raise RuntimeError(f"Camera worker {camera_name!r} has stopped")

    def latest_frame_seq(self) -> int | None:
        """Return the newest captured frame sequence for this camera."""
        get_timing = getattr(self.camera_manager, "get_current_timing_by_bus_id", None)
        if not callable(get_timing):
            return None
        timing = get_timing(self.camera_bus_id)
        return timing.frame_seq if timing is not None else None

    def wait_for_next_packet(
        self,
        after_frame_seq: int,
        should_continue: Callable[[], bool],
    ) -> FramePacket | None:
        """Wait for and return the newest unique transformed camera packet.

        Args:
            after_frame_seq: Last consumed frame sequence number.
            should_continue: Callback that remains true while waiting is allowed.

        Returns:
            The next transformed packet, or ``None`` when waiting stops or the
            manager cannot provide timestamped packets.

        Raises:
            RuntimeError: If the configured camera or its worker is unknown.
        """
        while should_continue():
            frame_available = self.camera_manager.wait_for_new_frame_by_bus_id(
                self.camera_bus_id,
                after_frame_seq,
                timeout_s=0.05,
            )
            if not frame_available:
                self._require_running_worker()
                continue
            if not should_continue():
                return None
            get_packet = getattr(
                self.camera_manager, "get_current_packet_by_bus_id", None
            )
            if not callable(get_packet):
                return None
            packet = get_packet(self.camera_bus_id)
            if (
                packet is not None
                and packet.timing.frame_seq is not None
                and packet.timing.frame_seq > after_frame_seq
            ):
                return TimedValue(
                    self._apply_rotation(packet.value),
                    packet.timing,
                )
        return None

    def update_config(self, json_config: dict[str, Any]) -> None:
        """Update runtime-configurable settings for the operation.

        Args:
            json_config: Runtime configuration overrides. Supports:
                - camera_bus_id: Changes the camera source (requires restart).
                - frame_rotation: Changes rotation angle (applied immediately).
        """
        camera_bus_id = json_config.get("camera_bus_id")
        if camera_bus_id is not None:
            self.camera_bus_id = camera_bus_id

        if json_config.get("frame_rotation") is not None:
            try:
                new_rotation = self._normalize_rotation(json_config["frame_rotation"])
                if new_rotation != self.frame_rotation:
                    self.frame_rotation = new_rotation
            except ValueError:
                pass
