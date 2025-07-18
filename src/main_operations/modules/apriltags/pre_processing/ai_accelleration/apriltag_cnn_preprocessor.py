import numpy as np
from typing import Optional

from src.main_operations.modules.apriltags.pre_processing.ai_accelleration.utils import (
    TARGET_WIDTH,
    TARGET_HEIGHT,
    CONF_THRESHOLD,
    LetterboxTransform,
    letterbox_image,
    calculate_crop_regions_from_grid,
    GRID_WIDTH,
    GRID_HEIGHT,
)
from utils.device_management_utils.compute_device import ComputeDevice


class ApriltagCnnPreprocessor:
    """A class to handle AprilTag CNN preprocessing and inference.

    This class loads a trained model and performs inference on video frames to detect
    potential AprilTag locations using a grid-based prediction approach.

    Attributes:
        model_path: Path to the trained model weights.
        device: The computation device (CPU/CUDA).
        model: The loaded GridPredictor model.
        transform: Image transformation pipeline.
        conf_threshold: Confidence threshold for predictions.
    """

    def __init__(
        self, model_path: str, device: ComputeDevice, conf_threshold: float = CONF_THRESHOLD
    ) -> None:
        """Initialize the AprilTag CNN preprocessor.

        Args:
            model_path: Path to the trained model weights file.
            device: The computation device (CPU/CUDA/MX3/CORAL).
            conf_threshold: Confidence threshold for predictions.
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.device = device
        self._load_model()
        self.transform = LetterboxTransform((TARGET_WIDTH, TARGET_HEIGHT))
        
        self.last_probs: Optional[np.ndarray] = None

    def _load_model(self) -> None:
        """Load and prepare the model for inference.

        Returns:
            None

        Raises:
            FileNotFoundError: If the model file is not found.
            RuntimeError: If there's an error loading the model.
        """
        try:
            self.device.load_model(self.model_path)
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")

    def process_frame(
        self, frame: np.ndarray, output_size: Optional[tuple[int, int]] = None
    ) -> list[tuple[int, int, int, int]]:
        """Process a single frame through the model.

        Args:
            frame: Input frame to process.
            output_size: Optional output size for the regions.

        Returns:
            A tuple containing:
                - List of crop regions as (left, top, right, bottom)
        """
        scaled_frame = letterbox_image(frame, (TARGET_WIDTH, TARGET_HEIGHT))
        transformed_image_np = self.transform(scaled_frame)
        
        probs = self.device.run(self.model_path, transformed_image_np)
        self.last_probs = probs

        h, w = scaled_frame.shape[:2]
        cell_w = w // GRID_WIDTH
        cell_h = h // GRID_HEIGHT
        conf_grid_mask = probs >= self.conf_threshold
        
        crop_regions = calculate_crop_regions_from_grid(
            conf_grid_mask, cell_w, cell_h
        )
            
        if output_size is not None:
            crop_regions = [
                (
                    int(region[0] * output_size[1] / TARGET_WIDTH),
                    int(region[1] * output_size[0] / TARGET_HEIGHT),
                    int(region[2] * output_size[1] / TARGET_WIDTH),
                    int(region[3] * output_size[0] / TARGET_HEIGHT),
                )
                for region in crop_regions
            ]
        return crop_regions
