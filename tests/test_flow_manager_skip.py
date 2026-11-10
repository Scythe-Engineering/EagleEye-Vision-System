from __future__ import annotations

import threading
import time
from typing import Any

from src.config.utils.flow_manager import FlowManager
from src.config.utils.operation import SKIP_PIPELINE_CYCLE, Operation
from src.config.utils.thread_object import ThreadObject
from src.main_operations.definitions.base.base_class import OperationInstance
from src.utils.logging.logger import Logger


def _add_profiling_state(flow_manager: FlowManager) -> None:
    """Give a bare FlowManager the profiling state __init__ would create."""
    flow_manager.pipeline_name = "Skip"
    flow_manager.logger = Logger(log_directory="logs/test")
    flow_manager._profile_lock = threading.Lock()
    flow_manager._profile_seq = 0
    flow_manager._last_frame_timings = None


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
    skip_operation = Operation(SkipOperation(), "skip", "skip", is_data_source=True)
    downstream_operation = Operation(
        downstream, "downstream", "downstream", is_data_source=True
    )
    flow_manager = FlowManager.__new__(FlowManager)
    flow_manager.operations = {
        "skip": skip_operation,
        "downstream": downstream_operation,
    }
    flow_manager.previous_operation_outputs = {}
    flow_manager.operation_outputs = {}
    flow_manager.execution_time_groups = [
        [skip_operation],
        [downstream_operation],
    ]
    flow_manager.on_operation_success = None
    flow_manager.on_operation_error = None
    flow_manager.thread_objects = []
    _add_profiling_state(flow_manager)
    skip_operation.execution_timestep = 0
    downstream_operation.execution_timestep = 1

    assert flow_manager._run_flow_direct() is False

    assert not downstream.called
    # The skipped cycle is still profiled: the skipping op and its timestep,
    # nothing that never ran.
    snapshot = flow_manager.get_latest_profile_snapshot()
    assert snapshot is not None
    assert snapshot["skipped"] is True
    assert snapshot["frame_seq"] == 1
    assert set(snapshot["operations"]) == {"skip"}
    assert snapshot["operations"]["skip"]["execution_time_ms"] >= 0.0
    assert [row["timestep"] for row in snapshot["timesteps"]] == [0]
    assert snapshot["timesteps"][0]["max_operation_uuid"] == "skip"
    assert snapshot["frame_time_ms"] >= snapshot["timesteps"][0]["total_time_ms"]


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

    skip = Operation(SkipOperation(), "skip", "skip", is_data_source=True)
    slow_instance = SlowOperation()
    slow = Operation(slow_instance, "slow", "slow", is_data_source=True)
    thread_objects = [ThreadObject(1), ThreadObject(1)]
    for operation, thread_obj in zip((skip, slow), thread_objects):
        operation.execution_timestep = 0
        operation.finish_timestep = 0
        thread_obj.occupy(operation)
        operation.set_thread_object(thread_obj)

    flow_manager = FlowManager.__new__(FlowManager)
    flow_manager.operations = {"skip": skip, "slow": slow}
    flow_manager.previous_operation_outputs = {}
    flow_manager.operation_outputs = {}
    flow_manager.execution_time_groups = [[skip, slow]]
    flow_manager.operations_by_finish_timestep = {0: [skip, slow]}
    flow_manager.thread_objects = thread_objects
    flow_manager.on_operation_success = None
    flow_manager.on_operation_error = None
    _add_profiling_state(flow_manager)

    assert flow_manager._run_flow_threaded() is False
    assert flow_manager._run_flow_threaded() is False

    assert slow_instance.call_count == 2
    assert all(thread_obj.state == "idle" for thread_obj in thread_objects)
    snapshot = flow_manager.get_latest_profile_snapshot()
    assert snapshot is not None
    assert snapshot["skipped"] is True
    assert snapshot["frame_seq"] == 2
    assert "skip" in snapshot["operations"]


def test_pipeline_skip_is_profiled_but_not_treated_as_completed() -> None:
    """Skipped cycles show in profiling; flush and cycle time stay completion-only."""
    from collections import deque
    from types import SimpleNamespace

    from src.config.utils.operation import Connection
    from src.config.utils.pipeline import Pipeline

    class Source(OperationInstance):
        def run(self, *_args: Any) -> int:
            return 1

    class Gate(OperationInstance):
        skip = True

        def run(self, value: int) -> Any:
            return SKIP_PIPELINE_CYCLE if self.skip else value

    class Sink(OperationInstance):
        def run(self, value: int) -> int:
            return value

    gate = Gate()
    source_op = Operation(
        Source(), "src", "source", is_data_source=True, output_ports=("data",)
    )
    gate_op = Operation(
        gate, "gate", "gate", input_ports=("data",), output_ports=("data",)
    )
    sink_op = Operation(Sink(), "sink", "sink", input_ports=("data",))
    Connection(source_op, "data", gate_op, "data", "any")
    Connection(gate_op, "data", sink_op, "data", "any")
    operations = {op.uuid: op for op in (source_op, gate_op, sink_op)}

    flushes: list[str] = []
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.limit_frames_to_camera_capture_speed = False
    pipeline.has_networktables_publisher = True
    pipeline.network_table = SimpleNamespace(
        getInstance=lambda: SimpleNamespace(flush=lambda: flushes.append("flush"))
    )
    pipeline.total_time_history = deque(maxlen=5)
    pipeline.total_time_history_lock = threading.Lock()
    pipeline.operations = operations
    pipeline.flow_manager = FlowManager(
        operations, Logger(log_directory="logs/test"), pipeline_name="Gate"
    )

    pipeline.run()
    skipped = pipeline.get_latest_profile_snapshot()
    assert flushes == []
    assert skipped is not None and skipped["skipped"] is True
    assert skipped["frame_seq"] == 1
    assert set(skipped["operations"]) == {"src", "gate"}
    assert "cycle_time_ms" not in skipped
    assert len(skipped["timesteps"]) == 2

    gate.skip = False
    pipeline.run()
    completed = pipeline.get_latest_profile_snapshot()
    assert flushes == ["flush"]
    assert completed is not None and "skipped" not in completed
    assert completed["frame_seq"] == 2
    assert set(completed["operations"]) == {"src", "gate", "sink"}
    assert completed["cycle_time_ms"] >= 0.0
    assert len(completed["timesteps"]) == 3
