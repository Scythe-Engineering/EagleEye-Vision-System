from __future__ import annotations

import numpy as np

from src.config.utils.operation import Operation
from src.main_operations.definitions.base.base_class import OperationInstance
from src.secondary_operations.device_input import DeviceInput
from src.utils.timing import TimedValue, TimingMetadata
from tests.utils.dummy_data import dummy_frame
from tests.utils.dummy_dependencies import (
    DummyComputePool,
    FakeCameraThreadManager,
    FakeEagleEyeInterface,
)


class ShapeOperation(OperationInstance):
    def run(self, input_data):
        """Run.
        
        Args:
            input_data: Input data."""
        return input_data.shape


class PassthroughOperation(OperationInstance):
    def run(self, input_data):
        """Run.
        
        Args:
            input_data: Input data."""
        return input_data


class TimedAwareOperation(OperationInstance):
    uses_timed_inputs = True

    def run(self, input_data):
        """Run.
        
        Args:
            input_data: Input data."""
        return input_data.timing.capture_nt_us


def test_device_input_returns_frame_packet_with_capture_timing() -> None:
    """Verify device input returns frame packet with capture timing."""
    manager = FakeCameraThreadManager(default_frame=dummy_frame())
    manager.add_camera("camera_1")
    manager.register_bus_id("1", "camera_1")

    packet = DeviceInput(
        web_interface=FakeEagleEyeInterface(),
        compute_pool=DummyComputePool(),
        camera_manager=manager,
        bus_id="1",
    ).run(None)

    assert isinstance(packet, TimedValue)
    assert isinstance(packet.value, np.ndarray)
    assert packet.timing.capture_nt_us > 0


def test_operation_unwraps_inputs_and_reattaches_timing() -> None:
    """Verify operation unwraps inputs and reattaches timing."""
    timing = TimingMetadata(capture_nt_us=42)
    input_packet = TimedValue(np.zeros((2, 3), dtype=np.uint8), timing)
    op = Operation(ShapeOperation(), uuid="shape", name="shape")

    output = op.run(input_packet)

    assert isinstance(output, TimedValue)
    assert output.value == (2, 3)
    assert output.timing is timing


def test_operation_reattaches_oldest_input_timing() -> None:
    """Verify operation uses oldest timing when multiple inputs are present."""
    newest = TimingMetadata(capture_nt_us=200)
    oldest = TimingMetadata(capture_nt_us=100)
    op = Operation(PassthroughOperation(), uuid="passthrough", name="passthrough")

    output = op.run(
        {
            "newest": TimedValue(np.zeros((2, 3), dtype=np.uint8), newest),
            "oldest": TimedValue(np.zeros((4, 5), dtype=np.uint8), oldest),
        }
    )

    assert isinstance(output, TimedValue)
    assert output.timing is oldest


def test_operation_can_opt_in_to_timed_inputs() -> None:
    """Verify operation can opt in to timed inputs."""
    timing = TimingMetadata(capture_nt_us=42)
    op = Operation(TimedAwareOperation(), uuid="timed", name="timed")

    output = op.run(TimedValue("value", timing))

    assert isinstance(output, TimedValue)
    assert output.value == 42
    assert output.timing is timing
