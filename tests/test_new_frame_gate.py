from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from src.config.utils.flow_manager import FlowManager
from src.config.utils.operation import SKIP_PIPELINE_CYCLE, Operation
from src.config.utils.thread_object import ThreadObject
from src.main_operations.definitions.base.base_class import OperationInstance
from src.secondary_operations.new_frame_gate import NewFrameGate
from src.utils.camera_utils.camera_thread_manager import CameraThreadManager, CameraWorker
from src.utils.timing import TimedValue, TimingMetadata


def _packet(frame_seq: int) -> TimedValue[np.ndarray]:
    return TimedValue(
        np.full((2, 2), frame_seq, dtype=np.uint8),
        TimingMetadata(
            capture_nt_us=frame_seq,
            capture_monotonic_ns=frame_seq,
            frame_seq=frame_seq,
            camera_name="camera_1",
            bus_id="1",
            source="test",
        ),
    )


def _manager_with_worker() -> tuple[CameraThreadManager, CameraWorker]:
    worker = CameraWorker("camera_1", cast(Any, object()))
    manager = CameraThreadManager.__new__(CameraThreadManager)
    manager.cameras = {"camera_1": worker}
    manager.bus_id_to_name = {"1": "camera_1"}
    return manager, worker


def test_new_frame_gate_waits_for_the_camera_sequence_to_advance() -> None:
    manager, worker = _manager_with_worker()
    pipeline = SimpleNamespace(thread_running=True)
    first_packet = _packet(1)
    worker.set_current_packet(first_packet)
    gate = NewFrameGate(manager, "1", cast(Any, pipeline))

    assert gate.run(first_packet) is first_packet

    with ThreadPoolExecutor(max_workers=1) as executor:
        waiting_run = executor.submit(gate.run, first_packet)
        time.sleep(0.03)
        assert not waiting_run.done()

        second_packet = _packet(2)
        worker.set_current_packet(second_packet)
        result = waiting_run.result(timeout=1.0)

    assert result is SKIP_PIPELINE_CYCLE
    assert gate.run(second_packet) is second_packet


def test_skip_signal_aborts_the_remaining_pipeline_cycle() -> None:
    class SkipOperation(OperationInstance):
        def run(self, _input_data: Any) -> Any:
            return SKIP_PIPELINE_CYCLE

    class DownstreamOperation(OperationInstance):
        def __init__(self) -> None:
            self.called = False

        def run(self, _input_data: Any) -> Any:
            self.called = True
            return None

    downstream = DownstreamOperation()
    flow_manager = FlowManager.__new__(FlowManager)
    flow_manager.previous_operation_outputs = {}
    flow_manager.operation_outputs = {}
    flow_manager.execution_time_groups = [
        [Operation(SkipOperation(), "gate", "new_frame_gate", is_data_source=True)],
        [Operation(downstream, "downstream", "downstream", is_data_source=True)],
    ]
    flow_manager.on_operation_success = None
    flow_manager.on_operation_error = None

    flow_manager._run_flow_direct()

    assert not downstream.called


def test_threaded_skip_waits_for_concurrent_operations_before_reuse() -> None:
    class SkipOperation(OperationInstance):
        def run(self, _input_data: Any) -> Any:
            return SKIP_PIPELINE_CYCLE

    class SlowOperation(OperationInstance):
        def __init__(self) -> None:
            self.call_count = 0

        def run(self, _input_data: Any) -> None:
            time.sleep(0.03)
            self.call_count += 1

    skip = Operation(SkipOperation(), "gate", "new_frame_gate", is_data_source=True)
    slow_instance = SlowOperation()
    slow = Operation(slow_instance, "slow", "slow", is_data_source=True)
    thread_objects = [ThreadObject(1), ThreadObject(1)]
    for operation, thread_obj in zip((skip, slow), thread_objects):
        operation.execution_timestep = 0
        operation.finish_timestep = 0
        thread_obj.occupy(operation)
        operation.set_thread_object(thread_obj)

    flow_manager = FlowManager.__new__(FlowManager)
    flow_manager.previous_operation_outputs = {}
    flow_manager.operation_outputs = {}
    flow_manager.execution_time_groups = [[skip, slow]]
    flow_manager.operations_by_finish_timestep = {0: [skip, slow]}
    flow_manager.thread_objects = thread_objects
    flow_manager.on_operation_success = None
    flow_manager.on_operation_error = None

    flow_manager._run_flow_threaded()
    flow_manager._run_flow_threaded()

    assert slow_instance.call_count == 2
    assert all(thread_obj.state == "idle" for thread_obj in thread_objects)


def test_new_frame_gate_cancels_wait_when_pipeline_stops() -> None:
    manager, worker = _manager_with_worker()
    pipeline = SimpleNamespace(thread_running=True)
    first_packet = _packet(1)
    worker.set_current_packet(first_packet)
    gate = NewFrameGate(manager, "1", cast(Any, pipeline))
    gate.run(first_packet)

    with ThreadPoolExecutor(max_workers=1) as executor:
        waiting_run = executor.submit(gate.run, first_packet)
        time.sleep(0.03)
        pipeline.thread_running = False

        assert waiting_run.result(timeout=1.0) is SKIP_PIPELINE_CYCLE
