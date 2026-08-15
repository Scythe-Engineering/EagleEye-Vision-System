"""Asynchronous MX3 YOLO object-detection pipeline operation."""

from __future__ import annotations

from threading import Lock
from typing import Any, Callable, cast

import numpy as np

from src.config.utils.async_docking import AsyncDockedOperation
from src.config.utils.operation import SKIP_PIPELINE_CYCLE
from src.main_operations.definitions.base.base_class import OperationInstance
from src.main_operations.modules.object_detection.utils.detection_visualization import (
    draw_detections,
)
from src.utils.device_registry import DeviceRegistry
from src.utils.model_library import ModelLibrary
from src.utils.mx3_runtime import (
    Mx3Profile,
    Mx3ResultPacket,
    Mx3RuntimeCoordinator,
    Mx3StreamBinding,
    TransformedFrameSource,
)


class Mx3AsyncObjectDetectionDefinition(AsyncDockedOperation[Mx3ResultPacket]):
    """Expose matched frame and detection outputs from one asynchronous MX3 stream."""

    dock_source_action = "device_input"
    dock_source_port = "frame"
    dock_target_port = "frame"
    allows_indefinite_wait = True

    def __init__(
        self,
        model_id: str,
        device_id: str,
        device_registry: DeviceRegistry,
        model_library: ModelLibrary,
        mx3_coordinator: Mx3RuntimeCoordinator | None,
        confidence_threshold: float = 0.25,
        max_detections: int = 100,
    ) -> None:
        """Resolve the managed DFP and retain stream settings until binding.

        Args:
            model_id: Managed model identifier.
            device_id: Canonical MX3 device identifier.
            device_registry: Injected startup device inventory.
            model_library: Injected managed model library.
            mx3_coordinator: Shared MX3 runtime coordinator.
            confidence_threshold: Minimum detection confidence.
            max_detections: Maximum detections returned per frame.

        Raises:
            RuntimeError: If the shared MX3 runtime coordinator is unavailable.
            ValueError: If the device or decoder settings are invalid.
        """
        if mx3_coordinator is None:
            raise RuntimeError("MX3 runtime coordinator is unavailable")

        descriptor = device_registry.get(device_id)
        if descriptor.device_type != "mx3" or descriptor.physical_index is None:
            raise ValueError("MX3 Async Object Detection requires an mx3:N device ID")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if max_detections < 1:
            raise ValueError("max_detections must be positive")

        self.model_id = model_id
        self.device_id = device_id
        self.physical_index = descriptor.physical_index
        self.artifact = model_library.resolve_artifact(model_id, device_id)
        self.profile = Mx3Profile.from_metadata(self.artifact.mx3_profile)
        self.class_names = model_library.get_model(model_id).class_names
        self.coordinator = mx3_coordinator
        self.confidence_threshold = float(confidence_threshold)
        self.max_detections = int(max_detections)
        self.binding: Mx3StreamBinding | None = None
        self._last_result: Mx3ResultPacket | None = None
        self._result_lock = Lock()
        self.class_colors: dict[int, tuple[int, int, int]] = {}

    def bind(
        self,
        source: OperationInstance,
        should_remain_active: Callable[[], bool],
    ) -> None:
        """Bind the MX3 callback directly to Device Input's frame-source API."""
        if self.binding is not None:
            raise RuntimeError("MX3 operation is already bound")
        frame_source = cast(TransformedFrameSource, source)
        self.binding = self.coordinator.register_stream(
            self.physical_index,
            self.artifact,
            frame_source,
            self.class_names,
            self.confidence_threshold,
            self.max_detections,
            should_remain_active,
        )

    def _required_binding(self) -> Mx3StreamBinding:
        """Return the initialized stream binding or fail clearly."""
        if self.binding is None:
            raise RuntimeError("MX3 operation is not docked to Device Input")
        return self.binding

    def activate(self) -> None:
        """Resume this operation's accelerator stream."""
        self._required_binding().activate()

    def wait_for_next(self) -> Mx3ResultPacket | None:
        """Wait for the newest completed inference packet."""
        return self._required_binding().wait_for_next()

    def deactivate(self) -> None:
        """Pause this operation's accelerator stream."""
        if self.binding is not None:
            self.binding.deactivate()

    @property
    def terminal_error(self) -> BaseException | None:
        """Return the stream's persistent backend failure, if present."""
        return self.binding.terminal_error if self.binding is not None else None

    def close(self) -> None:
        """Close this operation's accelerator stream."""
        if self.binding is not None:
            self.binding.close()

    def update_config(self, json_config: dict[str, Any]) -> None:
        """Apply changed decoder controls to an initialized stream binding.

        Empty, unchanged, and pre-docking updates are no-ops.
        """
        raw_confidence = json_config.get("confidence_threshold")
        raw_maximum = json_config.get("max_detections")
        confidence = float(raw_confidence) if raw_confidence is not None else None
        maximum = int(raw_maximum) if raw_maximum is not None else None
        if confidence == self.confidence_threshold:
            confidence = None
        if maximum == self.max_detections:
            maximum = None
        if (confidence is None and maximum is None) or self.binding is None:
            return
        self.binding.update_live_settings(confidence, maximum)
        if confidence is not None:
            self.confidence_threshold = confidence
        if maximum is not None:
            self.max_detections = maximum

    def run(self, _input_data: Any) -> Any:
        """Wait for an async result and expose its timing-matched outputs.

        Args:
            _input_data: Unused graph input; frames arrive through the docked source.

        Returns:
            Matched frame and detections, or ``SKIP_PIPELINE_CYCLE`` when stopped.

        Raises:
            RuntimeError: If the operation has not been docked.
        """
        result = self.wait_for_next()
        if result is None:
            return SKIP_PIPELINE_CYCLE
        with self._result_lock:
            self._last_result = result
        return {"frame": result.frame, "detections": result.detections}

    def visualize(self, frame: np.ndarray) -> np.ndarray | None:
        """Draw detections on the exact transformed frame used for inference.

        Args:
            frame: Pipeline visualization frame; the matched inference frame is used.

        Returns:
            Annotated inference frame, or ``None`` before the first result.
        """
        with self._result_lock:
            result = self._last_result
        if result is None:
            return None
        return draw_detections(
            result.frame.value.copy(),
            result.detections.value,
            self.class_colors,
        )
