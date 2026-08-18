from __future__ import annotations
import threading
import traceback
from time import perf_counter, perf_counter_ns, sleep, time
from typing import Any, Callable
from line_profiler import profile

from src.config.utils.operation import SKIP_PIPELINE_CYCLE, Operation
from src.config.utils.thread_object import ThreadObject
from src.utils.colors import Colors
from src.utils.logging.logger import Logger
from src.utils.timing import unwrap_timed_deep


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
        self._profile_lock = threading.Lock()
        self._last_frame_profile: dict[str, Any] | None = None
        self._profile_seq = 0

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
        frame_start = perf_counter()
        operation_time_by_uuid_ms: dict[str, float] = {}
        timestep_total_ms: dict[int, float] = {}

        for timestep, operation_group in enumerate(self.execution_time_groups):
            timestep_start = perf_counter()
            for operation in operation_group:
                input_for_op = self._gather_operation_inputs(operation)

                try:
                    operation_start = perf_counter()
                    output = operation.run(input_for_op)
                    operation_end = perf_counter()
                    if output is SKIP_PIPELINE_CYCLE:
                        return
                    operation_time_by_uuid_ms[operation.uuid] = max(
                        (operation_end - operation_start) * 1000.0,
                        0.0,
                    )
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

            timestep_end = perf_counter()
            timestep_total_ms[timestep] = max(
                (timestep_end - timestep_start) * 1000.0,
                0.0,
            )

        frame_end = perf_counter()
        frame_time_ms = max((frame_end - frame_start) * 1000.0, 0.0)
        self._record_profile_snapshot(
            frame_time_ms=frame_time_ms,
            operation_time_by_uuid_ms=operation_time_by_uuid_ms,
            timestep_total_ms=timestep_total_ms,
        )

    @profile
    def _run_flow_threaded(self) -> None:
        """Threaded execution for parallel pipelines."""
        self.previous_operation_outputs = self.operation_outputs.copy()
        self.operation_outputs.clear()
        frame_start = perf_counter()
        operation_time_by_uuid_ms: dict[str, float] = {}
        timestep_total_ms: dict[int, float] = {}
        cycle_id = perf_counter_ns()
        active_thread_objects: set[ThreadObject] = set()

        max_timestep = len(self.execution_time_groups)

        for current_timestep in range(max_timestep):
            timestep_start = perf_counter()
            operation_group = self.execution_time_groups[current_timestep]

            for operation in operation_group:
                input_for_op = self._gather_operation_inputs(operation)

                thread_obj = operation.get_thread_object()
                if thread_obj is None:
                    raise ValueError(f"Operation {operation.name} has no thread object")
                try:
                    thread_obj.set_needs_processing(
                        input_for_op, current_timestep, cycle_id
                    )
                    active_thread_objects.add(thread_obj)
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

                wait_timeout_s = (
                    None
                    if getattr(operation.instance, "allows_indefinite_wait", False)
                    else 5.0
                )
                not_timed_out = thread_obj.wait_done_processing(wait_timeout_s)
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
                active_thread_objects.discard(thread_obj)
                if output_data is SKIP_PIPELINE_CYCLE:
                    self._discard_active_threaded_operations(active_thread_objects)
                    return
                self.operation_outputs[operation.uuid] = output_data
                timing_uuid, execution_time_ms = thread_obj.get_last_cycle_timing(
                    cycle_id
                )
                if timing_uuid == operation.uuid and execution_time_ms is not None:
                    operation_time_by_uuid_ms[operation.uuid] = max(
                        execution_time_ms, 0.0
                    )
                if self.on_operation_success is not None:
                    self.on_operation_success(operation)

            timestep_end = perf_counter()
            timestep_total_ms[current_timestep] = max(
                (timestep_end - timestep_start) * 1000.0,
                0.0,
            )

        frame_end = perf_counter()
        frame_time_ms = max((frame_end - frame_start) * 1000.0, 0.0)
        self._record_profile_snapshot(
            frame_time_ms=frame_time_ms,
            operation_time_by_uuid_ms=operation_time_by_uuid_ms,
            timestep_total_ms=timestep_total_ms,
        )

    @staticmethod
    def _discard_active_threaded_operations(
        active_thread_objects: set[ThreadObject],
    ) -> None:
        """Wait for concurrent stale work and reset its threads before the next cycle."""
        for thread_obj in active_thread_objects:
            if not thread_obj.wait_done_processing():
                thread_obj.reset_state()
                raise ValueError(
                    "Operation did not finish while discarding a skipped pipeline cycle"
                )
            thread_obj.get_output_data()

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
                    selected = conn.from_operation.resolve_output_port(
                        self.previous_operation_outputs[from_uuid],
                        conn.from_port,
                    )
                    return unwrap_timed_deep(selected)
                else:
                    return None
            else:
                return conn.from_operation.resolve_output_port(
                    self.operation_outputs[conn.from_operation.uuid],
                    conn.from_port,
                )
        else:
            inputs: dict[str, Any] = {}

            for conn in input_connections:
                if not conn.is_default:
                    inputs[conn.to_port] = conn.from_operation.resolve_output_port(
                        self.operation_outputs[conn.from_operation.uuid],
                        conn.from_port,
                    )

            for conn in input_connections:
                if conn.is_default:
                    from_uuid = conn.from_operation.uuid
                    if from_uuid in self.previous_operation_outputs:
                        selected = conn.from_operation.resolve_output_port(
                            self.previous_operation_outputs[from_uuid],
                            conn.from_port,
                        )
                        inputs[conn.to_port] = unwrap_timed_deep(selected)

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

    def _get_operation_thread_number(self, operation: Operation) -> int:
        """Get 1-indexed thread number for an operation.

        Args:
            operation: Operation to resolve.

        Returns:
            Thread number, or 1 when unassigned.
        """
        thread_obj = operation.assigned_thread_object
        if thread_obj is None:
            return 1

        for index, thread in enumerate(self.thread_objects):
            if thread is thread_obj:
                return index + 1
        return 1

    def _build_timestep_rows(
        self,
        operation_time_by_uuid_ms: dict[str, float],
        timestep_total_ms: dict[int, float],
    ) -> list[dict[str, Any]]:
        """Build timestep profiling rows from runtime metrics.

        Args:
            operation_time_by_uuid_ms: Operation runtime map for current frame.
            timestep_total_ms: Timestep wall-clock duration map.

        Returns:
            Timestep rows for profile payload.
        """
        rows: list[dict[str, Any]] = []
        for timestep, operations in enumerate(self.execution_time_groups):
            candidate_rows: list[tuple[Operation, float]] = []
            for operation in operations:
                operation_time_ms = operation_time_by_uuid_ms.get(operation.uuid)
                if operation_time_ms is None:
                    continue
                candidate_rows.append((operation, operation_time_ms))

            max_operation = (
                max(candidate_rows, key=lambda item: item[1])
                if candidate_rows
                else None
            )
            rows.append(
                {
                    "timestep": timestep,
                    "total_time_ms": float(timestep_total_ms.get(timestep, 0.0)),
                    "max_operation_uuid": max_operation[0].uuid
                    if max_operation
                    else None,
                    "max_operation_name": max_operation[0].name
                    if max_operation
                    else None,
                    "max_operation_time_ms": float(max_operation[1])
                    if max_operation
                    else 0.0,
                    "operation_count": len(candidate_rows),
                }
            )
        return rows

    def _record_profile_snapshot(
        self,
        frame_time_ms: float,
        operation_time_by_uuid_ms: dict[str, float],
        timestep_total_ms: dict[int, float],
    ) -> None:
        """Record a lock-safe profiling snapshot for the current frame.

        Args:
            frame_time_ms: Frame wall-clock runtime.
            operation_time_by_uuid_ms: Per-operation runtime map.
            timestep_total_ms: Per-timestep wall-clock runtime map.
        """
        try:
            operations_payload: dict[str, dict[str, Any]] = {}
            for operation in self.operations.values():
                execution_time_ms = operation_time_by_uuid_ms.get(operation.uuid)
                if execution_time_ms is None:
                    continue
                operations_payload[operation.uuid] = {
                    "name": operation.name,
                    "timestep": operation.execution_timestep
                    if operation.execution_timestep is not None
                    else -1,
                    "thread": self._get_operation_thread_number(operation),
                    "execution_time_ms": float(execution_time_ms),
                }

            timestep_rows = self._build_timestep_rows(
                operation_time_by_uuid_ms,
                timestep_total_ms,
            )

            with self._profile_lock:
                self._profile_seq += 1
                snapshot = {
                    "pipeline_name": self.pipeline_name,
                    "frame_seq": self._profile_seq,
                    "frame_time_ms": float(frame_time_ms),
                    "timestamp_ms": int(time() * 1000),
                    "operations": operations_payload,
                    "timesteps": timestep_rows,
                }
                self._last_frame_profile = snapshot
        except Exception as error:
            self.logger.log(
                f"{Colors.YELLOW}Profiling snapshot failed for pipeline "
                f"{self.pipeline_name}: {error}{Colors.RESET}"
            )

    def set_latest_profile_cycle_time(self, cycle_time_ms: float) -> None:
        """Attach the full pipeline cycle time to the latest profile snapshot.

        The flow runtime is measured inside this manager, while camera-input gating
        happens in ``Pipeline`` before the flow starts. Keeping both measurements
        lets the UI show operation runtime and an FPS that includes input wait time.

        Args:
            cycle_time_ms: Elapsed time from the start of input gating through the
                completed flow execution.
        """
        with self._profile_lock:
            if self._last_frame_profile is not None:
                self._last_frame_profile["cycle_time_ms"] = float(
                    max(cycle_time_ms, 0.0)
                )

    def get_latest_profile_snapshot(self) -> dict[str, Any] | None:
        """Get a copy of the latest profiling snapshot.

        Returns:
            Latest profiling snapshot or None when unavailable.
        """
        with self._profile_lock:
            if self._last_frame_profile is None:
                return None

            operations = {
                operation_uuid: row.copy()
                for operation_uuid, row in self._last_frame_profile.get(
                    "operations", {}
                ).items()
            }
            timesteps = [
                row.copy() for row in self._last_frame_profile.get("timesteps", [])
            ]
            snapshot = self._last_frame_profile.copy()
            snapshot["operations"] = operations
            snapshot["timesteps"] = timesteps
            return snapshot
