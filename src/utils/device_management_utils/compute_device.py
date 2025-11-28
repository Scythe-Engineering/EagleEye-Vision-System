from abc import abstractmethod
import numpy as np
import torch


class ComputeDevice:
    def __init__(self, device_id: str, device_type: str):
        """
        Initializes the ComputeDevice.

        Args:
            device_id (str): A unique identifier for the compute device.
            device_type (str): The type of the compute device (e.g., 'CPU', 'GPU', 'MX3', 'CORAL').
        """
        self.device_id = device_id
        self.device_type = device_type

    @abstractmethod
    def load_model(self, model_path: str, input_data_shape: tuple[int, int], post_processing_model_path: str | None = None, is_grayscale: bool = False) -> None:
        """
        Load a model into the compute device.

        Args:
            model_path (str): Path to the model.
            input_data_shape (tuple[int, int]): Shape of the input data.
            post_processing_model_path (str | None): Path to the post-processing model.
            is_grayscale (bool): Whether the model is grayscale.
        """
        pass

    @abstractmethod
    def run(
        self, model_path: str, input_data: np.ndarray | torch.Tensor, input_data_shape: tuple[int, int], stream_idx: int
    ) -> np.ndarray:
        """
        Run a model on the compute device.

        Args:
            model_path (str): Path to the model.
            input_data (np.ndarray | torch.Tensor): Input data.
            input_data_shape (tuple[int, int]): Shape of the input data.
            stream_idx (int): Index of the stream to be run.

        Returns:
            np.ndarray: Output data.
        """
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """
        Stop the compute device.
        """
        pass
    
    @abstractmethod
    def connect_streams(self, num_streams: int) -> None:
        """
        Connect the compute device to the number of streams.
        """
        pass
