from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from src.config.utils.port_validation import validate_pipeline_connections
from src.main_operations.definitions.mx3_async_object_detection import (
    Mx3AsyncObjectDetectionDefinition,
)
from src.utils.model_library import ResolvedArtifact
from src.utils.mx3_runtime import (
    Mx3ResultPacket,
    Mx3RuntimeCoordinator,
    Mx3RuntimeError,
)
from src.utils.timing import TimedValue, TimingMetadata
from src.webui.web_server_utils.operation_config_mixin import OperationConfigMixin


PROFILE = {
    "input_width": 320,
    "input_height": 320,
    "color_order": "rgb",
    "layout": "hwzc",
    "normalization": "zero_to_one",
    "use_model_shape": [False, True],
    "decoder": "yolo_nms_xyxy",
    "adjustable_controls": {"confidence": True, "max_detections": True},
    "max_inflight": 2,
}


class FakeSource:
    """Supply deterministic timestamped frames to a fake MX3 callback."""

    def __init__(self, frame_sequences: list[int]) -> None:
        self.packets = deque(
            TimedValue(
                np.full((320, 320, 3), frame_seq, dtype=np.uint8),
                TimingMetadata(
                    capture_nt_us=frame_seq * 100,
                    capture_monotonic_ns=frame_seq * 1000,
                    frame_seq=frame_seq,
                    camera_name="camera",
                    bus_id="1",
                ),
            )
            for frame_seq in frame_sequences
        )

    def wait_for_next_packet(
        self,
        after_frame_seq: int,
        should_continue: Callable[[], bool],
    ) -> TimedValue[np.ndarray] | None:
        """Return the next frame, yielding briefly while the stream remains open."""
        while should_continue():
            if self.packets:
                packet = self.packets.popleft()
                assert packet.timing.frame_seq is not None
                assert packet.timing.frame_seq > after_frame_seq
                return packet
            time.sleep(0.001)
        return None


class FakeMxAccl:
    """Minimal blocking MemryX accelerator fake used by coordinator tests."""

    def __init__(self, dfp_path: str, **kwargs: object) -> None:
        self.dfp_path = dfp_path
        self.kwargs = kwargs
        self.callbacks = {}
        self.post_model = None
        self.started = False
        self.stopped = threading.Event()

    def connect_post_model(self, path: str, model_id: int = 0) -> None:
        self.post_model = (path, model_id)

    def connect_stream(
        self,
        input_callback: Callable[..., object],
        output_callback: Callable[..., object],
        stream_id: int,
        model_id: int = 0,
    ) -> None:
        self.callbacks[stream_id] = (input_callback, output_callback, model_id)

    def start(self) -> None:
        self.started = True

    def wait(self) -> None:
        self.stopped.wait()

    def stop(self) -> None:
        self.stopped.set()


def _artifact(tmp_path: Path, name: str = "model.dfp") -> ResolvedArtifact:
    path = tmp_path / name
    path.write_bytes(b"dfp")
    return ResolvedArtifact(
        model_id="model",
        device_id="mx3:0",
        slot="mx3_dfp",
        path=path,
        mx3_profile=PROFILE,
    )


def test_fake_mx3_streams_preserve_frame_correlation_and_shutdown(
    tmp_path: Path,
) -> None:
    accelerators: list[FakeMxAccl] = []

    def factory(path: str, **kwargs: object) -> FakeMxAccl:
        accelerator = FakeMxAccl(path, **kwargs)
        accelerators.append(accelerator)
        return accelerator

    coordinator = Mx3RuntimeCoordinator(accelerator_factory=factory)
    artifact = _artifact(tmp_path)
    active = True
    first = coordinator.register_stream(
        0,
        artifact,
        FakeSource([1, 2, 3, 4]),
        ("note",),
        0.25,
        10,
        lambda: active,
    )
    second = coordinator.register_stream(
        0,
        artifact,
        FakeSource([10]),
        ("note",),
        0.25,
        10,
        lambda: active,
    )

    coordinator.start()
    accelerator = accelerators[0]
    assert accelerator.started
    assert accelerator.kwargs == {
        "device_ids_to_use": [0],
        "use_model_shape": [False, True],
        "local_mode": True,
    }
    assert set(accelerator.callbacks) == {0, 1}

    first.activate()
    second.activate()
    first_input, first_output, _ = accelerator.callbacks[0]
    second_input, second_output, _ = accelerator.callbacks[1]
    assert first_input(0)[0].shape == (320, 320, 1, 3)
    assert second_input(1)[0].shape == (320, 320, 1, 3)
    first_output([np.array([[[32, 64, 160, 192, 0.9, 0]]])], 0)
    second_output([np.array([[[0, 0, 320, 320, 0.8, 0]]])], 1)

    first_result = first.wait_for_next()
    second_result = second.wait_for_next()
    assert first_result.frame.timing.frame_seq == 1
    assert first_result.detections.timing is first_result.frame.timing
    assert first_result.detections.value == [
        {
            "bbox": [0.1, 0.2, 0.5, 0.6],
            "confidence": 0.9,
            "class_id": 0,
            "class_name": "note",
        }
    ]
    assert second_result.frame.timing.frame_seq == 10

    # A result submitted before pause is discarded after resume.
    first_input(0)
    first.deactivate()
    first_output([np.array([[[0, 0, 10, 10, 0.8, 0]]])], 0)
    assert first.wait_for_next() is None
    first.activate()

    # Completed results are latest-only while their exact in-flight packets stay FIFO.
    first_input(0)
    first_output([np.array([[[0, 0, 10, 10, 0.8, 0]]])], 0)
    first_input(0)
    first_output([np.array([[[0, 0, 20, 20, 0.7, 0]]])], 0)
    assert first.wait_for_next().frame.timing.frame_seq == 4

    first.fail("hardware disconnected")
    with pytest.raises(Mx3RuntimeError, match="hardware disconnected"):
        first.wait_for_next()
    coordinator.stop()
    assert accelerator.stopped.is_set()


def test_mx3_operation_is_advertised_as_visualizable() -> None:
    mixin = OperationConfigMixin()

    assert mixin._operation_has_visualization(
        "mx3_async_object_detection.py", is_secondary=False
    )


def test_mx3_operation_visualizes_detections_on_the_inference_frame() -> None:
    timing = TimingMetadata(capture_nt_us=1, capture_monotonic_ns=2)
    inference_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = Mx3ResultPacket(
        frame=TimedValue(inference_frame, timing),
        detections=TimedValue(
            [
                {
                    "bbox": [0.1, 0.2, 0.5, 0.6],
                    "confidence": 0.9,
                    "class_id": 0,
                    "class_name": "note",
                }
            ],
            timing,
        ),
    )
    operation = Mx3AsyncObjectDetectionDefinition.__new__(
        Mx3AsyncObjectDetectionDefinition
    )
    operation._result_lock = threading.Lock()
    operation._last_result = result
    operation.class_colors = {}

    visualization = operation.visualize(np.full_like(inference_frame, 255))

    assert visualization is not None
    assert np.any(visualization != inference_frame)
    assert np.all(inference_frame == 0)
    assert np.all(visualization[99, 99] == 0)


def test_coordinator_rejects_different_dfps_on_one_physical_mx3(
    tmp_path: Path,
) -> None:
    coordinator = Mx3RuntimeCoordinator(accelerator_factory=FakeMxAccl)
    coordinator.register_stream(
        0, _artifact(tmp_path), FakeSource([]), None, 0.25, 10, lambda: True
    )
    with pytest.raises(Mx3RuntimeError, match="different DFP"):
        coordinator.register_stream(
            0,
            _artifact(tmp_path, "other.dfp"),
            FakeSource([]),
            None,
            0.25,
            10,
            lambda: True,
        )


def _device_input(uuid: str = "camera") -> dict[str, object]:
    return {
        "uuid": uuid,
        "action_name": "device_input",
        "action_params": {},
        "connections": [],
    }


def _mx3(uuid: str = "mx3") -> dict[str, object]:
    return {
        "uuid": uuid,
        "action_name": "mx3_async_object_detection",
        "action_params": {},
        "connections": [],
    }


def _dock(from_uuid: str, to_uuid: str) -> dict[str, str]:
    return {
        "from_uuid": from_uuid,
        "from_port": "frame",
        "to_uuid": to_uuid,
        "to_port": "frame",
        "data_type": "frame",
    }


def test_docking_validation_requires_direct_exclusive_device_input() -> None:
    camera = _device_input()
    detector = _mx3()
    with pytest.raises(ValueError, match="must bind directly"):
        validate_pipeline_connections([camera, detector])

    camera["connections"] = [_dock("camera", "mx3")]
    validate_pipeline_connections([camera, detector])

    second = _mx3("mx3-2")
    camera["connections"].append(_dock("camera", "mx3-2"))
    with pytest.raises(ValueError, match="already has a docked"):
        validate_pipeline_connections([camera, detector, second])


def test_connection_validation_requires_data_type() -> None:
    camera = _device_input()
    detector = _mx3()
    connection = _dock("camera", "mx3")
    del connection["data_type"]
    camera["connections"] = [connection]
    with pytest.raises(ValueError, match="Malformed connection"):
        validate_pipeline_connections([camera, detector])


def test_connection_validation_rejects_path_qualified_action_names() -> None:
    camera = _device_input()
    camera["action_name"] = "definitions/device_input"
    with pytest.raises(ValueError, match="must not contain a path"):
        validate_pipeline_connections([camera])


def test_resumed_stream_discards_output_submitted_before_the_pause(
    tmp_path: Path,
) -> None:
    """A pre-pause output must never be paired with a post-resume frame."""
    accelerators: list[FakeMxAccl] = []

    def factory(path: str, **kwargs: object) -> FakeMxAccl:
        accelerator = FakeMxAccl(path, **kwargs)
        accelerators.append(accelerator)
        return accelerator

    coordinator = Mx3RuntimeCoordinator(accelerator_factory=factory)
    binding = coordinator.register_stream(
        0,
        _artifact(tmp_path),
        FakeSource([1, 2, 3]),
        ("note",),
        0.25,
        10,
        lambda: True,
    )
    coordinator.start()
    stream_input, stream_output, _ = accelerators[0].callbacks[0]

    binding.activate()
    stream_input(0)
    binding.deactivate()
    binding.activate()
    stream_input(0)

    # The accelerator finally emits the output for the pre-pause submission,
    # which must be discarded rather than paired with the post-resume frame.
    stream_output([np.array([[[0, 0, 32, 32, 0.9, 0]]])], 0)
    stream_output([np.array([[[0, 0, 64, 64, 0.8, 0]]])], 0)
    assert binding.terminal_error is None
    result = binding.wait_for_next()
    assert result is not None
    assert result.frame.timing.frame_seq == 2
    assert result.detections.value[0]["confidence"] == pytest.approx(0.8)
    coordinator.stop()


def test_unregistered_stream_frees_its_physical_mx3_reservation(
    tmp_path: Path,
) -> None:
    """A pipeline that fails to initialize must not reserve a conflicting DFP."""
    coordinator = Mx3RuntimeCoordinator(accelerator_factory=FakeMxAccl)
    binding = coordinator.register_stream(
        0, _artifact(tmp_path), FakeSource([]), None, 0.25, 10, lambda: True
    )
    coordinator.unregister_stream(binding)
    coordinator.register_stream(
        0,
        _artifact(tmp_path, "other.dfp"),
        FakeSource([]),
        None,
        0.25,
        10,
        lambda: True,
    )


def test_failed_coordinator_startup_is_reported_to_every_caller(
    tmp_path: Path,
) -> None:
    """A partial startup must not leave the coordinator silently inert."""

    def factory(path: str, **kwargs: object) -> FakeMxAccl:
        raise RuntimeError("accelerator unavailable")

    coordinator = Mx3RuntimeCoordinator(accelerator_factory=factory)
    coordinator.register_stream(
        0, _artifact(tmp_path), FakeSource([]), None, 0.25, 10, lambda: True
    )
    with pytest.raises(Mx3RuntimeError, match="accelerator unavailable"):
        coordinator.start()
    with pytest.raises(Mx3RuntimeError, match="accelerator unavailable"):
        coordinator.start()
