from __future__ import annotations

import time
from typing import Any

from src.config.utils.flow_manager import FlowManager
from src.config.utils.operation import SKIP_PIPELINE_CYCLE, Operation
from src.config.utils.thread_object import ThreadObject
from src.main_operations.definitions.base.base_class import OperationInstance


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

    flow_manager._run_flow_threaded()
    flow_manager._run_flow_threaded()

    assert slow_instance.call_count == 2
    assert all(thread_obj.state == "idle" for thread_obj in thread_objects)
