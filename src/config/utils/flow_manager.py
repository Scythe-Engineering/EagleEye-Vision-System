from __future__ import annotations
import traceback
from time import sleep
from typing import Any, Callable
from line_profiler import profile

from src.config.utils.operation import Operation
from src.config.utils.thread_object import ThreadObject
from src.utils.colors import Colors
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
    if not downstream_timesteps:
        assert operation.execution_timestep is not None
        return operation.execution_timestep
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
    def __init__(
        self,
        operations: dict[str, Operation],
        logger: Logger,
        on_operation_error: Callable[[Operation, str], None] | None = None,
        on_operation_success: Callable[[Operation], None] | None = None,
        pipeline_name: str | None = None,
    ) -> None:
        """Initialize the flow manager that schedules operations at runtime.

        Args:
            operations: All operations configured in the flow.
            logger: Shared logger instance for the system.
            on_operation_error: Optional callback when an operation errors.
            on_operation_success: Optional callback when an operation succeeds.
            pipeline_name: Optional name of the pipeline or flow for logging.
        """
        self.operations: dict[str, Operation] = operations
        self.logger = logger
        self.on_operation_error = on_operation_error
        self.on_operation_success = on_operation_success
        self.pipeline_name = pipeline_name or "unknown"

        self.execution_time_groups: list[list[Operation]] = (
            self.forward_pass_operation_order()
        )

        # set the finish timestep for each operation
        backward_flow_register(self.execution_time_groups)

        self.num_threads = self._calculate_required_threads()

        self.logger.log(
            f"{Colors.GREEN}Number of threads required: {self.num_threads} for flow: "
            f"{self.pipeline_name}{Colors.RESET}"
        )

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
                available_threads = []
                for thread in sorted_threads:
                    is_available = True
                    for timestep in range(
                        operation.execution_timestep,
                        operation.finish_timestep + 1,
                    ):
                        if thread.is_occupied(timestep):
                            is_available = False
                            break
                    if is_available:
                        available_threads.append(thread)

                if not available_threads:
                    raise ValueError(
                        "No available threads for operation "
                        f"{operation.name} at timestep "
                        f"{operation.execution_timestep}"
                    )

                available_thread = available_threads[0]
                available_thread.occupy(operation)
                operation.set_thread_object(available_thread)

        self.operation_outputs: dict[str, Any] = {}
        self.previous_operation_outputs: dict[str, Any] = {}

    @profile
    def run_flow(self) -> None:
        """Run the flow of operations using timestep-based execution.

        Automatically uses direct execution for single-threaded pipelines,
        or threaded execution for multi-threaded pipelines.
        """
        if self.num_threads == 1:
            self._run_flow_direct()
        else:
            self._run_flow_threaded()

    @profile
    def _run_flow_direct(self) -> None:
        """Direct execution without thread signaling for linear pipelines."""
        self.previous_operation_outputs = self.operation_outputs.copy()
        self.operation_outputs.clear()

        for operation_group in self.execution_time_groups:
            for operation in operation_group:
                input_for_op = self._gather_operation_inputs(operation)

                try:
                    output = operation.instance.run(input_for_op)
                    if self.on_operation_success is not None:
                        self.on_operation_success(operation)
                    self.operation_outputs[operation.uuid] = output
                except TypeError as e:
                    if "None" in str(e):
                        # Skip entire frame when operation can't handle None input
                        return
                    if self.on_operation_error is not None:
                        self.on_operation_error(operation, traceback.format_exc())
                    raise ValueError(
                        f"Operation {operation.name} had an error: {traceback.format_exc()}"
                    )
                except Exception as e:
                    if self.on_operation_error is not None:
                        self.on_operation_error(operation, traceback.format_exc())
                    raise ValueError(
                        f"Operation {operation.name} had an error: {traceback.format_exc()}"
                    ) from e

    @profile
    def _run_flow_threaded(self) -> None:
        """Threaded execution for parallel pipelines."""
        self.previous_operation_outputs = self.operation_outputs.copy()
        self.operation_outputs.clear()

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
                    # Skip entire frame if error is None-related TypeError
                    if (
                        error_msg is not None
                        and "None" in error_msg
                        and "TypeError" in error_msg
                    ):
                        return
                    if self.on_operation_error is not None:
                        self.on_operation_error(operation, error_msg or "")
                    raise ValueError(
                        f"Operation {operation.name} had an error: {error_msg}"
                    )

                output_data = thread_obj.get_output_data()
                self.operation_outputs[operation.uuid] = output_data
                if self.on_operation_success is not None:
                    self.on_operation_success(operation)

    def _gather_operation_inputs(self, operation: Operation) -> Any:
        """Gather input data for an operation from upstream operations.

        Default connections use previous frame outputs, non-default use current frame.
        First frame: default inputs are skipped (None or missing dict key).
        Data source operations return None (they generate their own data).

        Args:
            operation: The operation that needs inputs.

        Returns:
            Input data for the operation (single value or dict of inputs), or None for data sources.

        Raises:
            ValueError: If operation has no input connections and is not a data source.
        """
        # Data source operations generate their own data - no input gathering needed
        if operation.is_data_source:
            return None

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
                    inputs[conn.to_port] = self.operation_outputs[
                        conn.from_operation.uuid
                    ]

            for conn in input_connections:
                if conn.is_default:
                    from_uuid = conn.from_operation.uuid
                    if from_uuid in self.previous_operation_outputs:
                        inputs[conn.to_port] = self.previous_operation_outputs[
                            from_uuid
                        ]

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
        """Returns the starting execution time of each operation in the flow.

        Data sources execute one timestep before their data is needed to get the most
        up-to-date value possible. Uses a two-pass approach:
        1. First pass: Include data sources at timestep 0 so dependents can get timesteps
        2. Second pass: Move data sources to min_dependent_timestep - 1 for fresh data

        Returns:
            list[list[Operation]]: Operations grouped by execution timestep.
        """
        # Start with all data sources as first operations
        # Data sources have no inputs, so all_inputs_solved() returns True
        data_sources = self._find_data_source_operations()
        first_operations: list[Operation] = data_sources.copy()

        # First pass: assign timesteps to all operations
        # Data sources get timestep 0 initially (no inputs)
        execution_time_groups: list[list[Operation]] = recursive_forward_flow_register(
            first_operations, 0, []
        )

        # Second pass: move data sources to one timestep before they're needed
        # This ensures the most up-to-date data is used
        for data_source in data_sources:
            min_dependent_timestep = self._find_min_dependent_timestep(data_source)

            if min_dependent_timestep is None:
                continue  # No dependents, keep at current timestep

            old_timestep = data_source.execution_timestep
            new_timestep = max(0, min_dependent_timestep - 1)

            # Only move if new timestep is later (more up-to-date data)
            if old_timestep is not None and new_timestep > old_timestep:
                # Remove from current group
                if data_source in execution_time_groups[old_timestep]:
                    execution_time_groups[old_timestep].remove(data_source)

                # Update timestep
                data_source.execution_timestep = new_timestep

                # Ensure group exists
                while new_timestep >= len(execution_time_groups):
                    execution_time_groups.append([])

                # Add to new group
                execution_time_groups[new_timestep].append(data_source)

        return execution_time_groups

    def _find_min_dependent_timestep(self, operation: Operation) -> int | None:
        """Find the minimum timestep among operations that depend on this operation.

        Args:
            operation: The operation to find dependent timesteps for.

        Returns:
            The minimum timestep of dependent operations, or None if no dependents.
        """
        dependent_timesteps: list[int] = []

        # Check direct downstream operations
        for connection in operation.get_output_connections():
            if connection.is_default:
                continue

            downstream_op = connection.to_operation
            if downstream_op.execution_timestep is not None:
                dependent_timesteps.append(downstream_op.execution_timestep)

        return min(dependent_timesteps) if dependent_timesteps else None

    def _find_data_source_operations(self) -> list[Operation]:
        """Find all operations marked as data sources.

        Returns:
            list[Operation]: List of data source operations.
        """
        return [op for op in self.operations.values() if op.is_data_source]

    def get_thread_and_timestep_info(self) -> dict[str, dict[str, int]]:
        """Get thread number and execution timestep for each operation.

        Returns:
            dict[str, dict[str, int]]: Dictionary mapping operation UUID to a dict with
                'thread' (1-indexed thread number) and 'timestep' (execution timestep).
        """
        result: dict[str, dict[str, int]] = {}

        for uuid, operation in self.operations.items():
            thread_number = 1
            thread_obj = operation.assigned_thread_object

            if thread_obj is not None:
                for idx, thread in enumerate(self.thread_objects):
                    if thread is thread_obj:
                        thread_number = idx + 1
                        break

            result[uuid] = {
                "thread": thread_number,
                "timestep": operation.execution_timestep
                if operation.execution_timestep is not None
                else -1,
            }

        return result
