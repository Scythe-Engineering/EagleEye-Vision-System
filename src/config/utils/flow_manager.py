from __future__ import annotations
import traceback
from time import sleep
from typing import Any
from line_profiler import profile

from src.config.utils.operation import Operation
from src.config.utils.thread_object import ThreadObject
from src.utils.logging.logger import Logger


def recursive_forward_flow_register(
    operations: list[Operation],
    time_step: int,
    time_step_groups: list[list[Operation]] | None = None,
) -> list[list[Operation]]:
    """Recursively register the forward flow of operations.

    Args:
        operations (dict[str, Operation]): The dictionary of operations to look at.
        time_step (int): The current time step.

    Returns:
        list[list[str]]: The forward flow of operations.
    """
    if time_step_groups is None:
        time_step_groups = []

    valid_group: list[Operation] = []
    for operation in operations:
        if operation.all_inputs_solved():
            valid_group.append(operation)

    if len(valid_group) == 0:
        return time_step_groups

    # once valid groups are found set time_step for all operations in the group
    for operation in valid_group:
        operation.execution_timestep = time_step

    time_step_groups.append(valid_group)

    next_operations: list[Operation] = []

    for operation in valid_group:
        output_connections = operation.get_output_connections()
        non_default_connections = [
            conn for conn in output_connections if not conn.is_default
        ]
        for connection in non_default_connections:
            if connection.to_operation not in next_operations:
                next_operations.append(connection.to_operation)

    time_step_groups = recursive_forward_flow_register(
        next_operations, time_step + 1, time_step_groups
    )

    return time_step_groups


def _validate_operation_timestep(operation: Operation) -> None:
    """Validate that an operation has an execution timestep assigned.

    Args:
        operation: The operation to validate.

    Raises:
        ValueError: If operation has no execution timestep.
    """
    if operation.execution_timestep is None:
        raise ValueError(
            f"Operation {operation.name} has no execution timestep assigned"
        )


def _get_downstream_timesteps(output_connections: list) -> list[int]:
    """Extract execution timesteps from downstream operations.

    Skips default connections since they're temporal dependencies.

    Args:
        output_connections: List of output connections from an operation.

    Returns:
        list[int]: List of execution timesteps from downstream operations.

    Raises:
        ValueError: If any downstream operation has no execution timestep.
    """
    downstream_timesteps: list[int] = []
    for conn in output_connections:
        if conn.is_default:
            continue
        _validate_operation_timestep(conn.to_operation)
        downstream_timesteps.append(conn.to_operation.execution_timestep)
    return downstream_timesteps


def _calculate_completion_timestep(operation: Operation) -> int:
    """Calculate the completion timestep for a single operation.

    Args:
        operation: The operation to calculate completion timestep for.

    Returns:
        int: The completion timestep for the operation.

    Raises:
        ValueError: If operation or downstream operation has no execution timestep.
    """
    _validate_operation_timestep(operation)

    output_connections = operation.get_output_connections()
    if not output_connections:
        assert operation.execution_timestep is not None
        return operation.execution_timestep

    downstream_timesteps = _get_downstream_timesteps(output_connections)
    return min(downstream_timesteps) - 1


def backward_flow_register(
    time_step_groups: list[list[Operation]],
) -> None:
    """Calculate and set finish timestep for operations based on forward flow.

    Uses the forward flow timesteps to determine the latest timestep by which
    each operation must be complete. This is the minimum timestep of all
    downstream operations minus 1.

    Args:
        time_step_groups: The forward flow time step groups from forward pass.
    """
    for group in time_step_groups:
        for operation in group:
            finish_timestep = _calculate_completion_timestep(operation)
            operation.finish_timestep = finish_timestep


class FlowManager:
    def __init__(self, operations: dict[str, Operation], logger: Logger):
        self.operations: dict[str, Operation] = operations
        self.logger = logger
        self.start_operation: Operation = self._find_start_operation()

        self.start_operation.execution_timestep = 0

        self.execution_time_groups: list[list[Operation]] = (
            self.forward_pass_operation_order()
        )

        # set the finish timestep for each operation
        backward_flow_register(self.execution_time_groups)

        self.num_threads = self._calculate_required_threads()

        self.logger.log(f"Number of threads: {self.num_threads}")

        # Pre-compute operations by finish timestep for faster lookup
        self.operations_by_finish_timestep: dict[int, list[Operation]] = {}
        for op in self.operations.values():
            if op.finish_timestep is not None:
                if op.finish_timestep not in self.operations_by_finish_timestep:
                    self.operations_by_finish_timestep[op.finish_timestep] = []
                self.operations_by_finish_timestep[op.finish_timestep].append(op)

        self.thread_objects: list[ThreadObject] = [
            ThreadObject(len(self.execution_time_groups))
            for _ in range(self.num_threads)
        ]

        for operation_group in self.execution_time_groups:
            for operation in operation_group:
                if operation.execution_timestep is None:
                    raise ValueError(
                        f"Operation {operation.name} has no execution timestep, thread occupy failed"
                    )
                if operation.finish_timestep is None:
                    raise ValueError(
                        f"Operation {operation.name} has no finish timestep, thread occupy failed"
                    )

                sorted_threads = sorted(
                    self.thread_objects,
                    key=lambda thread: thread.number_of_occupied_timesteps,
                )

                # remove threads that are occupied
                available_threads = [
                    thread
                    for thread in sorted_threads
                    if not thread.is_occupied(operation.execution_timestep)
                ]

                available_threads[0].occupy(operation)
                operation.set_thread_object(available_threads[0])

        self.operation_outputs: dict[str, Any] = {}
        self.previous_operation_outputs: dict[str, Any] = {}

    @profile
    def run_flow(self, input_data: Any) -> None:
        """Run the flow of operations using timestep-based execution.

        Automatically uses direct execution for single-threaded pipelines,
        or threaded execution for multi-threaded pipelines.

        Args:
            input_data: The input data to pass to the flow.
        """
        if self.num_threads == 1:
            self._run_flow_direct(input_data)
        else:
            self._run_flow_threaded(input_data)

    @profile
    def _run_flow_direct(self, input_data: Any) -> None:
        """Direct execution without thread signaling for linear pipelines.

        Args:
            input_data: The input data to pass to the flow.
        """
        self.previous_operation_outputs = self.operation_outputs.copy()
        self.operation_outputs.clear()
        self.operation_outputs[self.start_operation.uuid] = input_data

        for operation_group in self.execution_time_groups:
            for operation in operation_group:
                input_for_op = self._gather_operation_inputs(operation)

                try:
                    output = operation.instance.run(input_for_op)
                    self.operation_outputs[operation.uuid] = output
                except Exception as e:
                    raise ValueError(
                        f"Operation {operation.name} had an error: {traceback.format_exc()}"
                    ) from e

    @profile
    def _run_flow_threaded(self, input_data: Any) -> None:
        """Threaded execution for parallel pipelines.

        Args:
            input_data: The input data to pass to the flow.
        """
        self.previous_operation_outputs = self.operation_outputs.copy()
        self.operation_outputs.clear()
        self.operation_outputs[self.start_operation.uuid] = input_data

        max_timestep = len(self.execution_time_groups)

        for current_timestep in range(max_timestep):
            operation_group = self.execution_time_groups[current_timestep]

            for operation in operation_group:
                input_for_op = self._gather_operation_inputs(operation)

                thread_obj = operation.get_thread_object()
                if thread_obj is None:
                    raise ValueError(f"Operation {operation.name} has no thread object")
                try:
                    thread_obj.set_needs_processing(input_for_op, current_timestep)
                except Exception as _:
                    sleep(1)  # wait a bit before trying again
                    raise ValueError(
                        f"Operation {operation.name} had an error setting needs_processing: {traceback.format_exc()}"
                    )

            # Use pre-computed finish timestep lookup
            for operation in self.operations_by_finish_timestep.get(
                current_timestep, []
            ):
                thread_obj = operation.get_thread_object()
                if thread_obj is None:
                    raise ValueError(f"Operation {operation.name} has no thread object")

                not_timed_out = thread_obj.wait_done_processing()
                if not not_timed_out:
                    # Reset thread state on timeout before raising error
                    thread_obj.reset_state()
                    sleep(1)  # wait a bit before trying again
                    raise ValueError(
                        f"Operation {operation.name} timed out after 5 seconds"
                    )

                if thread_obj.had_error:
                    error_msg = thread_obj.error
                    # Reset thread state after error before raising exception
                    thread_obj.reset_state()
                    raise ValueError(
                        f"Operation {operation.name} had an error: {error_msg}"
                    )

                output_data = thread_obj.get_output_data()
                self.operation_outputs[operation.uuid] = output_data

    def _gather_operation_inputs(self, operation: Operation) -> Any:
        """Gather input data for an operation from upstream operations.

        Default connections use previous frame outputs, non-default use current frame.
        First frame: default inputs are skipped (None or missing dict key).

        Args:
            operation: The operation that needs inputs.

        Returns:
            Input data for the operation (single value or dict of inputs).

        Raises:
            ValueError: If operation has no input connections.
        """
        input_connections = operation.get_input_connections()

        if len(input_connections) == 0:
            raise ValueError(f"Operation {operation.name} has no input connections")

        if len(input_connections) == 1:
            conn = input_connections[0]
            if conn.is_default:
                from_uuid = conn.from_operation.uuid
                if from_uuid in self.previous_operation_outputs:
                    return self.previous_operation_outputs[from_uuid]
                else:
                    return None
            else:
                return self.operation_outputs[conn.from_operation.uuid]
        else:
            inputs: dict[str, Any] = {}

            for conn in input_connections:
                if not conn.is_default:
                    inputs[conn.to_port] = self.operation_outputs[conn.from_operation.uuid]

            for conn in input_connections:
                if conn.is_default:
                    from_uuid = conn.from_operation.uuid
                    if from_uuid in self.previous_operation_outputs:
                        inputs[conn.to_port] = self.previous_operation_outputs[from_uuid]

            return inputs

    def _calculate_required_threads(self) -> int:
        """Calculate the number of threads needed to run all operations concurrently.

        Analyzes the execution timeline to determine how many operations can run
        simultaneously at any given timestep, accounting for operations that span
        multiple timesteps.

        Returns:
            int: The number of threads required based on maximum concurrent operations.
        """
        num_operations_active_at_timestep: dict[int, int] = {}

        for operation in self.operations.values():
            start_time = operation.execution_timestep
            finish_time = operation.finish_timestep

            if start_time is None or finish_time is None:
                continue

            for time_step in range(start_time, finish_time + 1):
                if time_step not in num_operations_active_at_timestep:
                    num_operations_active_at_timestep[time_step] = 0
                num_operations_active_at_timestep[time_step] += 1

        return (
            max(num_operations_active_at_timestep.values())
            if num_operations_active_at_timestep
            else 0
        )

    def forward_pass_operation_order(self) -> list[list[Operation]]:
        """Returns the starting execution time of each operation in the flow. Returned as execution groups."""
        first_operations: list[Operation] = [
            connection.to_operation
            for connection in self.start_operation.get_output_connections()
            if not connection.is_default
        ]

        execution_time_groups: list[list[Operation]] = recursive_forward_flow_register(
            first_operations, 0, []
        )
        return execution_time_groups

    def _find_start_operation(self) -> Operation:
        """Finds the starting operation in the flow, always is the device_input operation name."""
        for uuid, operation_data in self.operations.items():
            if operation_data.name == "device_input":
                return self.operations[uuid]
        raise ValueError("No starting operation (device_input) found in operations.")
