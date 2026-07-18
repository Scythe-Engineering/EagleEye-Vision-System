from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.config.utils.operation import SKIP_PIPELINE_CYCLE
from src.main_operations.definitions.base.base_class import OperationInstance
from src.utils.timing import TimedValue

if TYPE_CHECKING:
    from src.config.utils.pipeline import Pipeline
    from src.utils.camera_utils.camera_thread_manager import CameraThreadManager


class NewFrameGate(OperationInstance):
    """Pass one camera frame per capture and wait instead of processing duplicates.

    Input and output are timestamped camera frame packets. The first available frame
    passes immediately. Later calls block efficiently on the camera worker's condition
    variable until its frame sequence advances.
    """

    uses_timed_inputs = True
    allows_indefinite_wait = True
    WAIT_TIMEOUT_S = 0.1

    def __init__(
        self,
        camera_manager: CameraThreadManager,
        bus_id: str,
        pipeline: Pipeline | None = None,
    ) -> None:
        """Initialize a gate for the camera identified by ``bus_id``.

        Args:
            camera_manager: Camera manager that publishes captured frames.
            bus_id: USB bus identifier matching the upstream device input.
            pipeline: Injected pipeline used to cancel waits during shutdown.
        """
        self.camera_manager = camera_manager
        self.bus_id = bus_id
        self.pipeline = pipeline
        self._last_frame_seq: int | None = None

    def run(self, input_data: Any) -> Any:
        """Return the input once per frame, waiting for a new packet when repeated.

        Args:
            input_data: Current output from the upstream device input.

        Returns:
            The current input when it belongs to a new frame. A pipeline-control
            sentinel aborts stale cycles after waiting or during pipeline shutdown.

        Raises:
            ValueError: If no camera is registered for the configured bus ID.
        """
        self._validate_input_camera(input_data)
        input_seq = self._get_frame_seq(input_data)
        if self._last_frame_seq is None and input_data is not None:
            self._last_frame_seq = input_seq if input_seq is not None else 0
            return input_data
        if self._last_frame_seq is None:
            self._last_frame_seq = 0

        if input_seq is not None and input_seq > self._last_frame_seq:
            self._last_frame_seq = input_seq
            return input_data

        if self.camera_manager.get_camera_name_by_bus_id(self.bus_id) is None:
            raise ValueError(f"No camera is registered for bus ID {self.bus_id!r}")

        while self._pipeline_is_running():
            frame_available = self.camera_manager.wait_for_new_frame_by_bus_id(
                self.bus_id,
                self._last_frame_seq,
                timeout_s=self.WAIT_TIMEOUT_S,
            )
            if frame_available:
                return SKIP_PIPELINE_CYCLE

        return SKIP_PIPELINE_CYCLE

    def _validate_input_camera(self, input_data: Any) -> None:
        """Reject timestamped input from a camera other than the configured one."""
        if not isinstance(input_data, TimedValue):
            return
        input_camera_name = input_data.timing.camera_name
        if input_camera_name is None:
            return
        configured_camera_name = self.camera_manager.get_camera_name_by_bus_id(
            self.bus_id
        )
        if configured_camera_name != input_camera_name:
            raise ValueError(
                f"Input camera {input_camera_name!r} does not match bus ID "
                f"{self.bus_id!r} ({configured_camera_name!r})"
            )

    @staticmethod
    def _get_frame_seq(input_data: Any) -> int | None:
        """Extract a valid frame sequence from a timestamped input."""
        if not isinstance(input_data, TimedValue):
            return None
        return input_data.timing.frame_seq

    def _pipeline_is_running(self) -> bool:
        """Return whether waiting should continue during pipeline shutdown."""
        return self.pipeline is None or self.pipeline.thread_running
