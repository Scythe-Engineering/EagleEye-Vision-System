from src.utils.device_management_utils.compute_device import ComputeDevice
import numpy as np
from memryx import SyncAccl # type: ignore


class MX3Accelerator(ComputeDevice):
    def __init__(self, device_id: str = "MX3_001"):
        """
        Initializes the MX3 accelerator.

        Args:
            device_id (str): A unique identifier for the MX3 accelerator.
        """
        super().__init__(
            device_id=device_id,
            device_type="MX3"
        )
        
        self.models = {}
        
    def load_model(self, model_path: str) -> None:
        """
        Load a model into the MX3 accelerator.

        Args:
            model_path (str): Path to the model to be loaded.
        """
        
        model_name = model_path.split("/")[-1].split(".")[0]
        if model_name in self.models:
            print(f"Model {model_name} already loaded, skipping...")
            return
        
        try:
            self.models[model_name] = SyncAccl(model_path)
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            raise e

    def run(self, model_name: str, input_data: np.ndarray) -> np.ndarray:
        """
        Run a model on the MX3 accelerator. (synchronous)

        Args:
            model_name (str): Name of the model to be run.
            input_data (np.ndarray): Input data to be processed.

        Returns:
            np.ndarray: Processed output data.
        """
        return self.models[model_name].run(input_data)
