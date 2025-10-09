from src.utils.device_management_utils.compute_device import ComputeDevice
from src.utils.colors import Colors
import numpy as np
from torch import Tensor
import time

print(f"{Colors.YELLOW}Initializing MX3 Library{Colors.RESET}")
from memryx import MultiStreamAsyncAccl  # type: ignore  # noqa: E402

print(f"{Colors.GREEN}MX3 Library initialized{Colors.RESET}")


class MX3ModelIO:
    def __init__(
        self, model_object: MultiStreamAsyncAccl, input_data_shape: tuple[int, int]
    ):
        """
        Initializes the MX3 model IO object.

        Args:
            model_object (MultiStreamAsyncAccl): Model object.
        """
        self.stop_signal: bool = False
        self.model_object: MultiStreamAsyncAccl = model_object
        self.input_data_shape: np.ndarray = np.array(
            [1, 1, input_data_shape[0], input_data_shape[1]]
        )
        self.zero_object = np.zeros(self.input_data_shape, dtype=np.float32)

        self.model_most_recent_inputs: dict[int, np.ndarray] = {}
        self.model_most_recent_outputs: dict[int, np.ndarray] = {}

    def model_input_generator(self, stream_idx: int) -> np.ndarray | None:
        """
        Generator for the model input.

        Args:
            stream_idx (int): Index of the stream to be run.

        Returns:
            np.ndarray | None: Input data for the model, or None if stopped.
        """
        if stream_idx in self.model_most_recent_inputs and not self.stop_signal:
            return self.model_most_recent_inputs[stream_idx]
        elif not self.stop_signal:
            return np.zeros(self.input_data_shape, dtype=np.float32)
        else:
            return None

    def model_output_processor(self, stream_idx: int, *outputs) -> None:
        """
        Processor for the model output.

        Args:
            stream_idx (int): Index of the stream to be processed.
            *outputs: Outputs from the model.
        """
        self.model_most_recent_outputs[stream_idx] = [outputs, time.time()]

    def connect_streams(self, stream_count: int) -> None:
        """
        Connect the streams to the model.

        Args:
            stream_count (int): The count of the streams to be connected.
        """
        self.model_object.connect_streams(
            self.model_input_generator, self.model_output_processor, stream_count
        )
        print(
            f"{Colors.GREEN}Connected {stream_count} streams to the model.{Colors.RESET}"
        )

    def sequential_run(self, stream_idx: int, data_array: np.ndarray) -> np.ndarray:
        """
        Run a model on the MX3 accelerator. (sequential)
        """
        start_output_time = self.model_most_recent_outputs.get(stream_idx, None)

        if start_output_time is None:
            start_output_time = 0
        else:
            start_output_time = start_output_time[1]

        self.model_most_recent_inputs[stream_idx] = data_array

        while stream_idx not in self.model_most_recent_outputs:
            time.sleep(0.001)

        while start_output_time == self.model_most_recent_outputs[stream_idx][1]:
            time.sleep(0.001)

        return self.model_most_recent_outputs[stream_idx][0][0]

    def stop(self) -> None:
        """
        Stop the MX3 model IO object.
        """
        self.stop_signal = True


class MX3Accelerator(ComputeDevice):
    def __init__(self, device_id: str = "MX3_001"):
        """
        Initializes the MX3 accelerator.

        Args:
            device_id (str): A unique identifier for the MX3 accelerator.
        """
        super().__init__(device_id=device_id, device_type="MX3")

        self.models = {}
        self.model_io_objects = {}

        self.thread_access_count = 0

    def load_model(self, model_path: str, input_data_shape: tuple[int, int]) -> None:
        """
        Load a model into the MX3 accelerator.

        Args:
            model_path (str): Path to the model to be loaded.
        """

        model_name = model_path.split("/")[-1].split(".")[0]
        if model_name in self.models:
            print(
                f"{Colors.YELLOW}Model {model_name} already loaded, skipping...{Colors.RESET}"
            )
            return

        try:
            self.models[model_name] = MultiStreamAsyncAccl(model_path)
            self.model_io_objects[model_name] = MX3ModelIO(
                model_object=self.models[model_name], input_data_shape=input_data_shape
            )
        except Exception as e:
            print(f"{Colors.RED}Error loading model {model_path}: {e}{Colors.RESET}")
            raise e

    def run(
        self,
        model_name: str,
        input_tensor: Tensor,
        input_data_shape: tuple[int, int],
        stream_idx: int,
    ) -> np.ndarray:
        """
        Run a model on the MX3 accelerator. (synchronous)

        Args:
            model_name (str): Name of the model to be run.
            input_tensor (torch.Tensor): Input tensor to be processed.
            input_data_shape (tuple[int, int]): Shape of the input data.

        Returns:
            np.ndarray: Processed output data.
        """
        data_array = input_tensor.cpu().numpy()

        data_array = data_array.reshape(1, 1, input_data_shape[0], input_data_shape[1])
        data_array = data_array.astype(np.float32)

        return_data = self.model_io_objects[model_name].sequential_run(
            stream_idx, data_array
        )

        return return_data

    def register_thread_access(self) -> int:
        """
        Register a thread access to the MX3 accelerator.
        """
        self.thread_access_count += 1
        return self.thread_access_count - 1

    def connect_streams(self, stream_count: int) -> None:
        """
        Connect the streams to the MX3 accelerator.
        """
        for model_io_object in self.model_io_objects.values():
            model_io_object.connect_streams(stream_count)

    def stop(self) -> None:
        """
        Stop the MX3 accelerator.
        """
        for model_io_object in self.model_io_objects.values():
            model_io_object.stop()
        for model_object in self.models.values():
            model_object.stop()
        print(f"{Colors.CYAN}MX3 accelerator stopped{Colors.RESET}")
