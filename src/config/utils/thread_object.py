from __future__ import annotations
from typing import Any
from src.config.utils.operation import Operation
import threading
import traceback


class ThreadObject:
    def __init__(self, number_of_timesteps: int) -> None:
        self.operation_obligations: list[Operation | bool] = [
            False
        ] * number_of_timesteps

        self.needs_processing: threading.Event = threading.Event()
        self.done_processing: threading.Event = threading.Event()

        self.data_lock = (
            threading.Lock()
        )  # not strictly necessary, but provides extra thread safety

        self.input_data: Any = None
        self.output_data: Any = None
        self.error: Exception | None = None

        self.current_timestep: int = 0

        self.number_of_occupied_timesteps: int = 0

        self.processing_thread_object: threading.Thread = threading.Thread(
            target=self.processing_thread, daemon=True
        )
        self.processing_thread_object.start()

    def processing_thread(self) -> None:
        """The processing thread for the thread object.

        This thread is responsible for processing the input data and setting the output data.
        """
        while True:
            self.needs_processing.wait()

            with self.data_lock:
                time_step = self.current_timestep
                input_data = self.input_data
                self.error = None

                obligation = self.operation_obligations[time_step]
                if obligation is True:
                    self.error = ValueError(
                        f"Thread should already be occupied at time step {time_step}"
                    )
                    self.output_data = None
                elif obligation is False:
                    self.error = ValueError(
                        f"Thread should not be occupied at time step {time_step}"
                    )
                    self.output_data = None
                else:
                    try:
                        self.output_data = obligation.run(input_data)
                    except Exception as e:
                        self.error = e
                        self.output_data = None
                        print(
                            f"Error in operation {obligation.name}: {traceback.format_exc()}"
                        )

            self.needs_processing.clear()
            self.done_processing.set()

    def set_needs_processing(self, input_data: Any, time_step: int) -> None:
        """Set the needs_processing flag and input data for the thread.

        Args:
            input_data (Any): The input data to pass to the thread.
            time_step (int): The current time step.

        Raises:
            ValueError: If the thread is already processing.
        """
        if self.needs_processing.is_set():
            raise ValueError(
                "Thread already processing. Something is very verrryyy wrong."
            )

        with self.data_lock:
            self.current_timestep = time_step
            self.input_data = input_data
        self.needs_processing.set()

    def is_done_processing(self) -> bool:
        """Check if the thread is done processing.

        Returns:
            bool: True if the thread is done processing, False otherwise.
        """
        return self.done_processing.is_set()

    def wait_done_processing(self) -> None:
        """
        Wait for the thread to finish processing.
        """
        if not self.is_done_processing():
            self.done_processing.wait()

    def get_output_data(self) -> Any:
        """Retrieve the output data after processing is complete.

        This method clears the done processing flag and returns the output data.

        Returns:
            Any: The processed output data.

        Raises:
            ValueError: If the thread is not done processing.
        """
        if not self.done_processing.is_set():
            raise ValueError(
                "Thread not done processing. Make sure to call is_done_processing() first."
            )

        with self.data_lock:
            self.done_processing.clear()
            return self.output_data

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
