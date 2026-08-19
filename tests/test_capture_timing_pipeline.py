from __future__ import annotations

import threading
import time
from collections import deque
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from src.config.utils.operation import Operation
from src.config.utils.pipeline import Pipeline
from src.main_operations.definitions.base.base_class import OperationInstance
from src.secondary_operations.device_input import DeviceInput
from src.utils.timing import TimedValue, TimingMetadata
from tests.utils.dummy_data import dummy_frame
from tests.utils.dummy_dependencies import (
    FakeCameraThreadManager,
    FakeEagleEyeInterface,
)


class ShapeOperation(OperationInstance):
    def run(self, input_data):
        return input_data.shape


class TimedAwareOperation(OperationInstance):
    uses_timed_inputs = True

    def run(self, input_data):
        return input_data.timing.capture_nt_us


def test_device_input_returns_frame_packet_with_capture_timing() -> None:
    manager = FakeCameraThreadManager(default_frame=dummy_frame())
    manager.add_camera("camera_1")
    manager.register_bus_id("1", "camera_1")

    packet = DeviceInput(
        web_interface=FakeEagleEyeInterface(),
        camera_manager=manager,
        camera_bus_id="1",
    ).run(None)

    assert isinstance(packet, TimedValue)
    assert isinstance(packet.value, np.ndarray)
    assert packet.timing.capture_nt_us > 0


def test_wait_for_next_packet_stops_when_the_camera_worker_stops() -> None:
    """A stopped worker must fail the wait instead of blocking the feeder."""
    manager = FakeCameraThreadManager(default_frame=dummy_frame())
    manager.add_camera("camera_1")
    manager.register_bus_id("1", "camera_1")
    device_input = DeviceInput(
        web_interface=FakeEagleEyeInterface(),
        camera_manager=manager,
        camera_bus_id="1",
    )
    manager.cameras["camera_1"].running = False

    with pytest.raises(RuntimeError, match="has stopped"):
        device_input.wait_for_next_packet(1_000_000, lambda: True)


def test_operation_unwraps_inputs_and_reattaches_timing() -> None:
    timing = TimingMetadata(capture_nt_us=42, capture_monotonic_ns=100)
    input_packet = TimedValue(np.zeros((2, 3), dtype=np.uint8), timing)
    op = Operation(ShapeOperation(), uuid="shape", name="shape")

    output = op.run(input_packet)

    assert isinstance(output, TimedValue)
    assert output.value == (2, 3)
    assert output.timing is timing


def test_operation_can_opt_in_to_timed_inputs() -> None:
    timing = TimingMetadata(capture_nt_us=42, capture_monotonic_ns=100)
    op = Operation(TimedAwareOperation(), uuid="timed", name="timed")

    output = op.run(TimedValue("value", timing))

    assert isinstance(output, TimedValue)
    assert output.value == 42
    assert output.timing is timing


def test_visualization_device_frame_is_unwrapped() -> None:
    """Visualization receives the image rather than its timing wrapper."""
    timing = TimingMetadata(capture_nt_us=42, capture_monotonic_ns=100)
    frame = np.zeros((2, 3), dtype=np.uint8)
    device_input = Operation(
        ShapeOperation(), uuid="camera", name="device_input", is_data_source=True
    )
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.operations = {device_input.uuid: device_input}
    pipeline.flow_manager = cast(
        Any,
        SimpleNamespace(
            operation_outputs={device_input.uuid: TimedValue(frame, timing)}
        ),
    )

    result = pipeline._get_device_input_frame(device_input.uuid)

    assert result is frame
    assert result.copy().shape == (2, 3)


def _latency_pipeline(*packets: TimedValue) -> Pipeline:
    """Build a bare pipeline with completed timed outputs.

    Args:
        packets: Timed values to expose as operation outputs.

    Returns:
        A pipeline ready for capture-latency measurement.
    """
    operations = {
        f"camera_{index}": Operation(
            ShapeOperation(),
            uuid=f"camera_{index}",
            name="device_input",
            is_data_source=True,
        )
        for index in range(len(packets))
    }
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.operations = operations
    pipeline.device_input_uuids = tuple(operations)
    pipeline.flow_manager = cast(
        Any,
        SimpleNamespace(
            operation_outputs={
                uuid: packet for uuid, packet in zip(operations, packets, strict=True)
            }
        ),
    )
    return pipeline


def test_capture_latency_measures_frame_age_on_the_monotonic_clock() -> None:
    """Measure capture age using the monotonic clock."""
    captured_ns = time.monotonic_ns() - 40_000_000
    pipeline = _latency_pipeline(
        TimedValue(
            np.zeros((2, 2), dtype=np.uint8),
            TimingMetadata(capture_nt_us=1, capture_monotonic_ns=captured_ns),
        )
    )

    latency_ms = pipeline._capture_latency_ms()

    assert latency_ms is not None
    assert 40.0 <= latency_ms < 60.0


def test_capture_latency_reports_the_stalest_camera() -> None:
    """Report the oldest timed output produced by a completed cycle."""
    now_ns = time.monotonic_ns()
    pipeline = _latency_pipeline(
        TimedValue(
            np.zeros((2, 2), dtype=np.uint8),
            TimingMetadata(capture_nt_us=1, capture_monotonic_ns=now_ns - 10_000_000),
        ),
        TimedValue(
            np.zeros((2, 2), dtype=np.uint8),
            TimingMetadata(capture_nt_us=2, capture_monotonic_ns=now_ns - 90_000_000),
        ),
    )

    latency_ms = pipeline._capture_latency_ms()

    assert latency_ms is not None
    assert latency_ms >= 90.0


def test_capture_latency_uses_the_timing_matched_async_result() -> None:
    """Include an older async result rather than only the latest camera packet."""
    now_ns = time.monotonic_ns()
    pipeline = _latency_pipeline(
        TimedValue(
            np.zeros((2, 2), dtype=np.uint8),
            TimingMetadata(capture_nt_us=1, capture_monotonic_ns=now_ns - 5_000_000),
        )
    )
    pipeline.flow_manager.operation_outputs["async"] = {
        "frame": TimedValue(
            np.zeros((2, 2), dtype=np.uint8),
            TimingMetadata(
                capture_nt_us=2,
                capture_monotonic_ns=now_ns - 100_000_000,
            ),
        )
    }

    latency_ms = pipeline._capture_latency_ms()

    assert latency_ms is not None
    assert latency_ms >= 100.0


def test_capture_latency_is_absent_without_timed_outputs() -> None:
    """Return no latency when a cycle produced no timed values."""
    pipeline = _latency_pipeline()

    assert pipeline._capture_latency_ms() is None


def test_skipped_cycle_does_not_modify_the_previous_profile() -> None:
    """Do not attach current timing fields when no new snapshot was recorded."""

    class SkippedFlowManager:
        """Flow manager stand-in for an aborted pipeline cycle."""

        def __init__(self) -> None:
            self.operation_outputs: dict[str, Any] = {}

        def run_flow(self) -> bool:
            """Report that the flow aborted before profiling."""
            return False

        def set_latest_profile_cycle_time(self, _cycle_time_ms: float) -> None:
            """Fail if a skipped cycle attempts to mutate the prior snapshot."""
            raise AssertionError("stale profile was modified")

    pipeline = Pipeline.__new__(Pipeline)
    pipeline.limit_frames_to_camera_capture_speed = False
    pipeline.flow_manager = SkippedFlowManager()
    pipeline.total_time_history = deque(maxlen=1)
    pipeline.total_time_history_lock = threading.Lock()

    pipeline.run()


def test_fake_timing_lookup_does_not_advance_the_frame_sequence() -> None:
    """Keep timing inspection observational like the production manager."""
    manager = FakeCameraThreadManager(default_frame=dummy_frame)
    manager.add_camera("camera")
    manager.register_bus_id("bus", "camera")
    worker = manager.cameras["camera"]
    worker.get_current_packet()
    frame_seq = worker.frame_seq

    manager.get_current_timing_by_bus_id("bus")
    manager.get_current_timing_by_bus_id("bus")

    assert worker.frame_seq == frame_seq
