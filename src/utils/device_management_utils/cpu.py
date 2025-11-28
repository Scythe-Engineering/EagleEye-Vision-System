from src.utils.device_management_utils.compute_device import ComputeDevice
import numpy as np
import torch
import onnxruntime as ort
import os


class CPU(ComputeDevice):
    def __init__(self):
        super().__init__(device_id="CPU_001", device_type="CPU")
        self.models = {}
        self.model_input_names = {}
        self.model_output_names = {}

    def load_model(self, model_path: str, input_data_shape: tuple[int, int], post_processing_model_path: str | None = None, is_grayscale: bool = False) -> None:
        """
        Loads an ONNX model from the specified file path.

        Args:
            model_path: The file path to the ONNX model.
            input_data_shape: The shape of the input data. (unused)
            post_processing_model_path: The file path to the post-processing model. (unused)
            is_grayscale: Whether the model is grayscale. (unused)
        """
        try:
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )

            providers = ["CPUExecutionProvider"]

            session = ort.InferenceSession(
                model_path, session_options, providers=providers
            )

            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name

            model_key = os.path.splitext(os.path.basename(model_path))[0]

            self.models[model_key] = session
            self.model_input_names[model_key] = input_name
            self.model_output_names[model_key] = output_name

        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model from {model_path}: {e}")

    def run(self, model_path: str, input_data: np.ndarray | torch.Tensor, input_data_shape: tuple[int, int], stream_idx: int) -> np.ndarray:
        """
        Runs inference on the specified ONNX model.

        Args:
            model_name: The key of the loaded model (derived from file name).
            input_tensor: The input data as a torch.Tensor. It will be
                          converted to a NumPy array for ONNX Runtime.
            input_data_shape: The shape of the input data. (unused)
            stream_idx: The index of the stream to be run. (unused)
        Returns:
            The output of the ONNX model as a NumPy array.
        """
        model_key = os.path.splitext(os.path.basename(model_path))[0]

        if model_key not in self.models:
            raise ValueError(f"Model '{model_key}' not loaded.")

        session = self.models[model_key]
        input_name = self.model_input_names[model_key]
        output_name = self.model_output_names[model_key]
        
        if isinstance(input_data, np.ndarray):
            input_data = torch.from_numpy(input_data)

        # .detach() is important to remove it from the computation graph to avoid memory leaks
        input_data = input_data.detach().cpu().numpy()

        input_feed = {input_name: input_data}

        outputs = np.array(session.run([output_name], input_feed))
        
        return outputs.reshape(outputs.shape[2], outputs.shape[3]) # outputs is 1, 1, x, y before this for some reason ¯\_(ツ)_/¯ double np array or something?
