from __future__ import annotations

import threading
from collections import deque
from types import SimpleNamespace
from typing import Any

import numpy as np

from src.config.utils.operation import Connection, Operation
from src.config.utils.pipeline import Pipeline
from src.main_operations.definitions.base.base_class import OperationInstance
from src.utils.timing import TimedValue, TimingMetadata


class _DeviceSource(OperationInstance):
    def __init__(self, camera_bus_id: str) -> None:
        self.camera_bus_id = camera_bus_id

    def run(self, _input_data: Any) -> None:
        return None


class _Join(OperationInstance):
    def run(self, input_data: Any) -> Any:
        return input_data


class _CountingSource(OperationInstance):
    def run(self, _input_data: Any) -> int:
        return 1


class _PacketManager:
    def __init__(self) -> None:
        self.packets: dict[str, TimedValue[np.ndarray]] = {}
        self.wait_requests: list[tuple[str, int, float | None]] = []
        self.on_wait: Any = None

    def publish(
        self,
        bus_id: str,
        frame_seq: int,
        camera_name: str | None = None,
    ) -> TimedValue[np.ndarray]:
        packet = TimedValue(
            np.full((1, 1), frame_seq, dtype=np.uint8),
            TimingMetadata(
                capture_nt_us=frame_seq,
                capture_monotonic_ns=frame_seq,
                frame_seq=frame_seq,
                camera_name=camera_name or f"camera_{bus_id}",
                bus_id=bus_id,
            ),
        )
        self.packets[bus_id] = packet
        return packet

    def get_current_packet_by_bus_id(
        self, bus_id: str
    ) -> TimedValue[np.ndarray] | None:
        return self.packets.get(bus_id)

    def get_current_timing_by_bus_id(
        self, bus_id: str
    ) -> TimingMetadata | None:
        packet = self.packets.get(bus_id)
        return packet.timing if packet is not None else None

    def wait_for_new_frame_by_bus_id(
        self,
        bus_id: str,
        after_frame_seq: int,
        timeout_s: float | None = None,
    ) -> bool:
        self.wait_requests.append((bus_id, after_frame_seq, timeout_s))
        if self.on_wait is not None:
            self.on_wait(bus_id)
        timing = self.get_current_timing_by_bus_id(bus_id)
        return (
            timing is not None
            and timing.frame_seq is not None
            and timing.frame_seq > after_frame_seq
        )


def _two_camera_pipeline() -> tuple[Pipeline, _PacketManager]:
    source_a = Operation(
        _DeviceSource("a"),
        "source-a",
        "device_input",
        is_data_source=True,
        output_ports=("frame",),
    )
    source_b = Operation(
        _DeviceSource("b"),
        "source-b",
        "device_input",
        is_data_source=True,
        output_ports=("frame",),
    )
    join = Operation(_Join(), "join", "join", input_ports=("a", "b"))
    Connection(source_a, "frame", join, "a", "frame")
    Connection(source_b, "frame", join, "b", "frame")

    manager = _PacketManager()
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.operations = {
        source_a.uuid: source_a,
        source_b.uuid: source_b,
        join.uuid: join,
    }
    pipeline.camera_manager = manager
    pipeline.device_input_uuids = ("source-a", "source-b")
    pipeline._last_device_input_tokens = {}
    pipeline.flow_manager = SimpleNamespace(
        operation_outputs={}, previous_operation_outputs={}
    )
    return pipeline, manager


def test_pipeline_requires_every_device_input_to_advance() -> None:
    pipeline, manager = _two_camera_pipeline()
    packet_a = manager.publish("a", 1)
    packet_b = manager.publish("b", 1)

    assert pipeline._all_device_inputs_are_fresh()

    pipeline.flow_manager.operation_outputs = {
        "source-a": packet_a,
        "source-b": packet_b,
    }
    pipeline._record_device_input_tokens()

    manager.publish("a", 2)
    assert not pipeline._all_device_inputs_are_fresh()

    manager.publish("b", 2)
    assert pipeline._all_device_inputs_are_fresh()


def test_fresh_input_gate_waits_for_camera_notification() -> None:
    pipeline, manager = _two_camera_pipeline()
    pipeline.flow_manager.operation_outputs = {
        "source-a": manager.publish("a", 1),
        "source-b": manager.publish("b", 1),
    }
    pipeline._record_device_input_tokens()
    manager.publish("a", 2)
    manager.on_wait = lambda bus_id: manager.publish(bus_id, 2)
    pipeline.limit_frames_to_camera_capture_speed = True
    pipeline.thread = object()
    pipeline.thread_running = True
    pipeline.pipeline_name = "test"

    assert pipeline._wait_for_fresh_device_inputs()
    assert manager.wait_requests == [("b", 1, 0.05)]


def test_camera_identity_change_is_treated_as_a_new_stream() -> None:
    pipeline, manager = _two_camera_pipeline()
    pipeline.flow_manager.operation_outputs = {
        "source-a": manager.publish("a", 5, camera_name="old-camera"),
        "source-b": manager.publish("b", 5),
    }
    pipeline._record_device_input_tokens()

    manager.publish("a", 1, camera_name="replacement-camera")
    manager.publish("b", 6)

    assert pipeline._all_device_inputs_are_fresh()


def test_whole_pipeline_runs_only_when_all_camera_inputs_are_fresh() -> None:
    pipeline, manager = _two_camera_pipeline()
    continuous = Operation(
        _CountingSource(), "continuous", "network_source", is_data_source=True
    )
    pipeline.operations[continuous.uuid] = continuous
    pipeline.limit_frames_to_camera_capture_speed = True
    pipeline.total_time_history = deque(maxlen=5)
    pipeline.total_time_history_lock = threading.Lock()
    pipeline.thread = None
    pipeline.thread_running = False

    class RecordingFlowManager:
        def __init__(self) -> None:
            self.operation_outputs: dict[str, Any] = {}
            self.previous_operation_outputs: dict[str, Any] = {}
            self.call_count = 0

        def run_flow(self) -> None:
            self.call_count += 1
            for operation_uuid in pipeline.device_input_uuids:
                operation = pipeline.operations[operation_uuid]
                self.operation_outputs[operation_uuid] = (
                    manager.get_current_packet_by_bus_id(
                        operation.instance.camera_bus_id
                    )
                )

        def set_latest_profile_cycle_time(self, cycle_time_ms: float) -> None:
            self.cycle_time_ms = cycle_time_ms

        def set_latest_profile_capture_latency(self, latency_ms: float) -> None:
            self.capture_latency_ms = latency_ms

    flow_manager = RecordingFlowManager()
    pipeline.flow_manager = flow_manager
    manager.publish("a", 1)
    manager.publish("b", 1)

    pipeline.run()
    pipeline.run()
    manager.publish("a", 2)
    pipeline.run()
    manager.publish("b", 2)
    pipeline.run()

    assert flow_manager.call_count == 2


def test_pipeline_without_device_inputs_continues_running() -> None:
    operation = Operation(
        _CountingSource(), "continuous", "network_source", is_data_source=True
    )
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.operations = {operation.uuid: operation}
    pipeline.device_input_uuids = ()
    pipeline._last_device_input_tokens = {}
    pipeline.limit_frames_to_camera_capture_speed = True
    pipeline.total_time_history = deque(maxlen=5)
    pipeline.total_time_history_lock = threading.Lock()
    pipeline.thread = None
    pipeline.thread_running = False
    pipeline.flow_manager = SimpleNamespace(
        call_count=0,
        run_flow=lambda: setattr(
            pipeline.flow_manager,
            "call_count",
            pipeline.flow_manager.call_count + 1,
        ),
        set_latest_profile_cycle_time=lambda cycle_time_ms: setattr(
            pipeline.flow_manager, "cycle_time_ms", cycle_time_ms
        ),
    )

    pipeline.run()
    pipeline.run()

    assert pipeline.flow_manager.call_count == 2
