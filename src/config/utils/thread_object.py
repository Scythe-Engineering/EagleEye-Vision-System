from __future__ import annotations

import threading
import traceback
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

        self.number_of_occupied_timesteps: int = 0

        self.processing_thread_object: threading.Thread = threading.Thread(
            target=self.processing_thread, daemon=True
        )
        self.processing_thread_object.start()

    def _set_error(self, error: str) -> None:
        """
        Set the error message for the thread.

        Args:
            error (str): The error message to set.
        """
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
                        f"Thread should already be occupied at time step {time_step}, Thread obligations: {self.operation_obligations}"
                    )
                    self.output_data = None
                    self.had_error = True
                    self.state = "error"
                elif obligation is False:
                    self._set_error(
                        f"Thread should not be occupied at time step {time_step}, Thread obligations: {self.operation_obligations}"
                    )
                    self.output_data = None
                    self.had_error = True
                    self.state = "error"
                else:
                    # Release condition lock during operation execution
                    self.condition.release()
                    try:
                        output_data = obligation.run(input_data)
                        # Re-acquire lock to update state
                        self.condition.acquire()
                        self.output_data = output_data
                        self.state = "done"
                    except Exception as _:
                        self.condition.acquire()
                        self.output_data = None
                        self._set_error(
                            f"Error in operation {obligation.name}: {traceback.format_exc()}"
                        )
                        self.had_error = True
                        self.state = "error"

                self.condition.notify()

    @profile
    def set_needs_processing(self, input_data: Any, time_step: int) -> None:
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

            self.current_timestep = time_step
            self.input_data = input_data
            self.state = "processing"
            self.condition.notify()

    @profile
    def is_done_processing(self) -> bool:
        """Check if the thread is done processing.

        Returns:
            bool: True if the thread is done processing, False otherwise.
        """
        with self.condition:
            return self.state in ("done", "error")

    @profile
    def wait_done_processing(self) -> bool:
        """
        Wait for the thread to finish processing.

        Returns:
            bool: False if the thread timed out, True otherwise.
        """
        with self.condition:
            return self.condition.wait_for(
                lambda: self.state in ("done", "error"), timeout=5
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
