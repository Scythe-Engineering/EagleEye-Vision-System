from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np

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
