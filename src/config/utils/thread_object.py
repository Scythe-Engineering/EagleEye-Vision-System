from __future__ import annotations

import threading
import traceback
from time import perf_counter
from typing import TYPE_CHECKING, Any
from line_profiler import profile

if TYPE_CHECKING:
    from src.config.utils.operation import Operation


class ThreadObject:
    def __init__(self, number_of_timesteps: int) -> None:
        self.operation_obligations: list[Operation | bool] = [
            False
        ] * number_of_timesteps

        self.condition = threading.Condition()
        self.state: str = "idle"  # "idle", "processing", "done", "error"
        self.had_error: bool = False

        self.input_data: Any = None
        self.output_data: Any = None
        self.error: str | None = None

        self.current_timestep: int = 0
        self.current_cycle_id: int = 0

        self.number_of_occupied_timesteps: int = 0
        self.last_execution_time_ms: float | None = None
        self.last_operation_uuid: str | None = None
        self.last_cycle_id: int = 0

        self.processing_thread_object: threading.Thread = threading.Thread(
            target=self.processing_thread, daemon=True
        )
        self.processing_thread_object.start()

    def _set_error(self, error: str, print_error: bool = False) -> None:
        """
        Set the error message for the thread.

        Args:
            error (str): The error message to set.
            print_error (bool): Whether to print the error to console.
        """
        if print_error:
            print(error)
        self.error = error

    @profile
    def processing_thread(self) -> None:
        """The processing thread for the thread object.

        This thread is responsible for processing the input data and setting the output data.
        """
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.state == "processing")

                self.had_error = False
                self.error = None

                time_step = self.current_timestep
                input_data = self.input_data
                obligation = self.operation_obligations[time_step]

                if obligation is True:
                    self._set_error(
                        f"Thread should already be occupied at time step {time_step}, Thread obligations: {self.operation_obligations}",
                        print_error=True
                    )
                    self.output_data = None
                    self.had_error = True
                    self.state = "error"
                elif obligation is False:
                    self._set_error(
                        f"Thread should not be occupied at time step {time_step}, Thread obligations: {self.operation_obligations}",
                        print_error=True
                    )
                    self.output_data = None
                    self.had_error = True
                    self.state = "error"
                else:
                    # Release condition lock during operation execution
                    self.condition.release()
                    try:
                        operation_start = perf_counter()
                        output_data = obligation.run(input_data)
                        operation_end = perf_counter()
                        # Re-acquire lock to update state
                        self.condition.acquire()
                        self.output_data = output_data
                        self.last_execution_time_ms = max(
                            (operation_end - operation_start) * 1000.0,
                            0.0,
                        )
                        self.last_operation_uuid = obligation.uuid
                        self.last_cycle_id = self.current_cycle_id
                        self.state = "done"
                    except Exception as _:
                        self.condition.acquire()
                        self.output_data = None
                        self.last_execution_time_ms = None
                        self.last_operation_uuid = None
                        self.last_cycle_id = self.current_cycle_id
                        self._set_error(
                            f"Error in operation {obligation.name}: {traceback.format_exc()}"
                        )
                        self.had_error = True
                        self.state = "error"

                self.condition.notify()

    @profile
    def set_needs_processing(
        self, input_data: Any, time_step: int, cycle_id: int = 0
    ) -> None:
        """Set the needs_processing flag and input data for the thread.

        Args:
            input_data (Any): The input data to pass to the thread.
            time_step (int): The current time step.

        Raises:
            ValueError: If the thread is already processing.
        """
        with self.condition:
            if self.state == "processing":
                raise ValueError(
                    "Thread already processing. Something is very verrryyy wrong."
                )

            if time_step < 0 or time_step >= len(self.operation_obligations):
                raise ValueError(
                    "time_step must be within the obligations range for this thread."
                )

            self.current_timestep = time_step
            self.current_cycle_id = cycle_id
            self.input_data = input_data
            self.state = "processing"
            self.condition.notify()

    def get_last_cycle_timing(self, cycle_id: int) -> tuple[str | None, float | None]:
        """Get timing data for the requested cycle id.

        Args:
            cycle_id (int): Cycle identifier for stale-data protection.

        Returns:
            Tuple of operation UUID and execution time in milliseconds.
            Returns (None, None) when no timing is available for the cycle.
        """
        with self.condition:
            if self.last_cycle_id != cycle_id:
                return None, None
            return self.last_operation_uuid, self.last_execution_time_ms

    @profile
    def is_done_processing(self) -> bool:
        """Check if the thread is done processing.

        Returns:
            bool: True if the thread is done processing, False otherwise.
        """
        with self.condition:
            return self.state in ("done", "error")

    @profile
    def wait_done_processing(self, timeout_s: float | None = 5.0) -> bool:
        """Wait for the thread to finish processing.

        Args:
            timeout_s: Maximum wait in seconds, or ``None`` to wait indefinitely.

        Returns:
            ``False`` if the thread timed out, otherwise ``True``.
        """
        with self.condition:
            return self.condition.wait_for(
                lambda: self.state in ("done", "error"), timeout=timeout_s
            )

    @profile
    def get_output_data(self) -> Any:
        """Retrieve the output data after processing is complete.

        This method clears the done processing flag and returns the output data.

        Returns:
            Any: The processed output data.

        Raises:
            ValueError: If the thread is not done processing.
        """
        with self.condition:
            if self.state not in ("done", "error"):
                raise ValueError(
                    "Thread not done processing operation. Make sure to call is_done_processing() first."
                )

            output_data = self.output_data
            self.state = "idle"
            return output_data

    def reset_state(self) -> None:
        """Reset the thread state to allow for reprocessing after an error.

        This method clears error flags and resets processing state to allow
        the thread to be reused after a failure.
        """
        with self.condition:
            self.had_error = False
            self.error = None
            self.output_data = None
            self.last_execution_time_ms = None
            self.last_operation_uuid = None
            self.state = "idle"

    # functions for initializing the flow, not runtime execution
    def is_occupied(self, time_step: int) -> bool:
        """Check if the thread is occupied at the specified time step.

        Args:
            time_step (int): The time step to check for occupation.

        Returns:
            bool: True if the thread is occupied, False otherwise.
        """
        return self.operation_obligations[time_step] is not False

    def occupy(self, operation: Operation) -> None:
        """Occupy the thread for the duration of the operation.

        Marks the thread as occupied from the operation's execution timestep

        to its finish timestep, inclusive.

        Args:
            operation (Operation): The operation occupying the thread.

        Raises:
            ValueError: If the operation lacks execution or finish timesteps.
        """
        if operation.execution_timestep is None:
            raise ValueError(
                f"Operation {operation.name} has no execution timestep, thread occupy failed"
            )
        if operation.finish_timestep is None:
            raise ValueError(
                f"Operation {operation.name} has no finish timestep, thread occupy failed"
            )

        self.operation_obligations[operation.execution_timestep] = operation
        self.number_of_occupied_timesteps += 1

        if operation.execution_timestep != operation.finish_timestep:
            for time in range(
                operation.execution_timestep + 1, operation.finish_timestep + 1
            ):
                self.operation_obligations[time] = True
                self.number_of_occupied_timesteps += 1
