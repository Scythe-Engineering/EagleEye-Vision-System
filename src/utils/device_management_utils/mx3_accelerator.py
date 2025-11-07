from src.utils.device_management_utils.compute_device import ComputeDevice
from src.utils.colors import Colors
import numpy as np
from torch import Tensor
import time
import onnxruntime as ort
from threading import Condition

print(f"{Colors.YELLOW}Initializing MX3 Library{Colors.RESET}")
from memryx import MultiStreamAsyncAccl  # type: ignore  # noqa: E402

print(f"{Colors.GREEN}MX3 Library initialized{Colors.RESET}")

POLL_INTERVAL_S = 0.001

EPOCH_MAX = 2**31 - 1


class MX3ModelIO:
    def __init__(
        self,
        model_object: MultiStreamAsyncAccl,
        input_data_shape: tuple[int, int],
        is_grayscale: bool = False,
    ):
        """
        Initializes the MX3 model IO object.

        Args:
            model_object (MultiStreamAsyncAccl): Model object.
        """
        self.stop_signal: bool = False
        self.model_object: MultiStreamAsyncAccl = model_object
        self.input_data_shape: np.ndarray = np.array(
            [1, 1 if is_grayscale else 3, input_data_shape[0], input_data_shape[1]]
        )
        self.zero_object = np.zeros(self.input_data_shape, dtype=np.float32)

        self.model_most_recent_inputs: dict[int, np.ndarray] = {}
        self.model_most_recent_outputs: dict[int, np.ndarray] = {}

        self.input_conditions: dict[int, Condition] = {}
        self.output_conditions: dict[int, Condition] = {}
        self.pending_inputs: dict[int, list[tuple[int, np.ndarray]]] = {}
        self.inflight_epochs: dict[int, list[int]] = {}
        self.input_epoch_counter: dict[int, int] = {}
        self.latest_output_epoch: dict[int, int] = {}

    def model_input_generator(self, stream_idx: int) -> np.ndarray | None:
        """
        Generator for the model input.

        Args:
            stream_idx (int): Index of the stream to be run.

        Returns:
            np.ndarray | None: Input data for the model, or None if stopped.
        """
        cond = self.input_conditions.get(stream_idx, None)
        if cond is None:
            return None
        with cond:
            while not self.stop_signal and len(self.pending_inputs[stream_idx]) == 0:
                cond.wait(timeout=POLL_INTERVAL_S)
            if self.stop_signal:
                return None
            epoch, data_array = self.pending_inputs[stream_idx].pop(0)
            self.inflight_epochs[stream_idx].append(epoch)
            return data_array

    def model_output_processor(self, stream_idx: int, *outputs) -> None:
        """
        Processor for the model output.

        Args:
            stream_idx (int): Index of the stream to be processed.
            *outputs: Outputs from the model.
        """
        epoch = None
        if self.inflight_epochs.get(stream_idx):
            epoch = self.inflight_epochs[stream_idx].pop(0)
        now = time.time()
        self.model_most_recent_outputs[stream_idx] = [outputs, now, epoch]
        if epoch is not None:
            self.latest_output_epoch[stream_idx] = epoch
        out_cond = self.output_conditions.get(stream_idx, None)
        if out_cond is not None:
            with out_cond:
                out_cond.notify_all()

    def connect_streams(self, stream_count: int) -> None:
        """
        Connect the streams to the model.

        Args:
            stream_count (int): The count of the streams to be connected.
        """
        for s in range(stream_count):
            if s not in self.input_conditions:
                self.input_conditions[s] = Condition()
            if s not in self.output_conditions:
                self.output_conditions[s] = Condition()
            self.pending_inputs.setdefault(s, [])
            self.inflight_epochs.setdefault(s, [])
            self.input_epoch_counter.setdefault(s, 0)
            self.latest_output_epoch.setdefault(s, -1)
        self.model_object.connect_streams(
            self.model_input_generator, self.model_output_processor, stream_count
        )
        print(
            f"{Colors.GREEN}Connected {stream_count} streams to the model.{Colors.RESET}"
        )

    def sequential_run(
        self, stream_idx: int, data_array: np.ndarray, timeout_s: float = 5.0
    ) -> np.ndarray:
        """
        Run a model on the MX3 accelerator. (sequential)

        Args:
            stream_idx (int): Index of the stream to be run.
            data_array (np.ndarray): Input data array.
            timeout_s (float): Timeout in seconds for waiting for output.

        Returns:
            np.ndarray: Processed output data.
        """
        if stream_idx not in self.input_conditions:
            raise RuntimeError(
                "Streams not connected. Call connect_streams(stream_count) before running."
            )

        with self.input_conditions[stream_idx]:
            next_epoch = (self.input_epoch_counter[stream_idx] + 1) % EPOCH_MAX
            self.input_epoch_counter[stream_idx] = next_epoch
            self.pending_inputs[stream_idx].append((next_epoch, data_array))
            self.input_conditions[stream_idx].notify()

        deadline = time.time() + timeout_s
        with self.output_conditions[stream_idx]:
            while True:
                rec = self.model_most_recent_outputs.get(stream_idx)
                if rec is not None and len(rec) == 3 and rec[2] == next_epoch:
                    return rec[0]
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Timed out waiting for output on stream {stream_idx} for epoch {next_epoch}"
                    )
                self.output_conditions[stream_idx].wait(
                    timeout=min(POLL_INTERVAL_S, remaining)
                )

    def stop(self) -> None:
        """
        Stop the MX3 model IO object.
        """
        self.stop_signal = True
        for cond in self.input_conditions.values():
            with cond:
                cond.notify_all()
        for cond in self.output_conditions.values():
            with cond:
                cond.notify_all()


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

        self.post_processing_models: dict[str, ort.InferenceSession] = {}
        self.post_processing_input_names: dict[str, list[str]] = {}
        self.post_processing_output_names: dict[str, str] = {}

        self.thread_access_count = 0

    def load_model(
        self,
        model_path: str,
        input_data_shape: tuple[int, int],
        post_processing_model_path: str | None = None,
        is_grayscale: bool = False,
    ) -> None:
        """
        Load a model into the MX3 accelerator.

        Args:
            model_path (str): Path to the model to be loaded.
            input_data_shape (tuple[int, int]): Shape of the input data.
            post_processing_model_path (str | None): Optional path to ONNX post-processing model.
            is_grayscale (bool): Whether the model expects grayscale input (single channel) instead of RGB.
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
                model_object=self.models[model_name],
                input_data_shape=input_data_shape,
                is_grayscale=is_grayscale,
            )

            if post_processing_model_path is not None:
                session_options = ort.SessionOptions()
                session_options.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                )

                providers = ["CPUExecutionProvider"]

                session = ort.InferenceSession(
                    post_processing_model_path, session_options, providers=providers
                )

                input_names = [inp.name for inp in session.get_inputs()]
                output_name = session.get_outputs()[0].name

                self.post_processing_models[model_name] = session
                self.post_processing_input_names[model_name] = input_names
                self.post_processing_output_names[model_name] = output_name

                print(
                    f"{Colors.GREEN}Loaded post-processing model for {model_name}{Colors.RESET}"
                )

        except Exception as e:
            print(f"{Colors.RED}Error loading model {model_path}: {e}{Colors.RESET}")
            raise e

    def run(
        self,
        model_name: str,
        input_data: Tensor | np.ndarray,
        input_data_shape: tuple[int, int],
        stream_idx: int,
    ) -> np.ndarray:
        """
        Run a model on the MX3 accelerator. (synchronous)

        Args:
            model_name (str): Name of the model to be run.
            input_data (Tensor | np.ndarray): Input data to be processed.
            input_data_shape (tuple[int, int]): Shape of the input data.

        Returns:
            np.ndarray: Processed output data.
        """
        if isinstance(input_data, Tensor):
            input_data = input_data.cpu().numpy()

        expected_channels = input_data.shape[1]
        expected_shape = (
            1,
            expected_channels,
            input_data_shape[0],
            input_data_shape[1],
        )
        if input_data.shape != expected_shape:
            input_data = input_data.reshape(expected_shape)

        if input_data.dtype != np.float32:
            input_data = input_data.astype(np.float32)

        return_data = self.model_io_objects[model_name].sequential_run(
            stream_idx, input_data
        )

        if model_name in self.post_processing_models:
            session = self.post_processing_models[model_name]
            input_names = self.post_processing_input_names[model_name]
            output_name = self.post_processing_output_names[model_name]

            if isinstance(return_data, (list, tuple)):
                outputs_list = list(return_data)
            else:
                outputs_list = [return_data]

            if len(input_names) != len(outputs_list):
                raise ValueError(
                    f"Post-processing model for {model_name} expects {len(input_names)} inputs but received {len(outputs_list)} from accelerator."
                )

            input_feed = {
                name: tensor for name, tensor in zip(input_names, outputs_list)
            }
            post_processed_outputs = np.array(session.run([output_name], input_feed))

            return post_processed_outputs[0]

        if isinstance(return_data, (list, tuple)):
            return return_data[0]

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
        # Cancel the auto-connect timer if it's still running
        if hasattr(self, "_connect_timer") and self._connect_timer.is_alive():
            self._connect_timer.cancel()

        for model_io_object in self.model_io_objects.values():
            model_io_object.stop()
        for model_object in self.models.values():
            model_object.stop()

        self.post_processing_models.clear()
        self.post_processing_input_names.clear()
        self.post_processing_output_names.clear()

        print(f"{Colors.CYAN}MX3 accelerator stopped{Colors.RESET}")
