import numpy as np
import torch

from src.utils.device_management_utils.compute_device import ComputeDevice


class GPU(ComputeDevice):
    def __init__(self, device_id: str = "GPU_001"):
        """
        Initializes the GPU compute device.

        Args:
            device_id (str): A unique identifier for the GPU device.

        Raises:
            RuntimeError: If CUDA is not available or no CUDA GPUs are detected.
        """
        # Check if CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. GPU device requires CUDA support."
            )

        # Check if there are actual CUDA GPUs
        if torch.cuda.device_count() == 0:
            raise RuntimeError(
                "No CUDA GPUs detected. GPU device requires at least one CUDA-compatible GPU."
            )

        self.cuda_available = True
        device_type = "GPU_CUDA"

        super().__init__(device_id=device_id, device_type=device_type)

        # Set the torch device
        self.device = torch.device("cuda")
        self.models = {}

    def load_model(self, model_path: str) -> None:
        """
        Load a PyTorch model from the specified file path.

        Args:
            model_path (str): Path to the PyTorch model (.pt or .pth file).
        """
        try:
            # Load the model
            model = torch.load(model_path, map_location=self.device)
            model.to(self.device)
            model.eval()

            # Extract model name from path
            model_name = model_path.split("/")[-1].split(".")[0]
            self.models[model_name] = model

        except Exception as e:
            raise RuntimeError(f"Failed to load PyTorch model from {model_path}: {e}")

    def run(
        self,
        model_path: str,
        input_data: np.ndarray,
        input_data_shape: tuple[int, int],
        stream_idx: int,
    ) -> np.ndarray:
        """
        Run inference on the specified PyTorch model.

        Args:
            model_path (str): Path to the model.
            input_data (np.ndarray): Input data.
            input_data_shape (tuple[int, int]): Shape of the input data (unused for GPU inference).
            stream_idx (int): Stream index (unused for GPU inference).

        Returns:
            np.ndarray: Model output as numpy array.
        """
        # Extract model name from path (same logic as load_model)
        model_name = model_path.split("/")[-1].split(".")[0]

        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not loaded.")

        model = self.models[model_name]

        # Convert input data to torch tensor and ensure it's on the correct device
        input_tensor = torch.from_numpy(input_data).to(self.device)

        # Run inference
        with torch.no_grad():
            output = model(input_tensor)

        # Convert to numpy and return
        return output.cpu().numpy()

    def stop(self) -> None:
        """
        Stop the GPU device and clear loaded models.
        """
        self.models.clear()
        # PyTorch handles GPU memory cleanup automatically when tensors go out of scope
