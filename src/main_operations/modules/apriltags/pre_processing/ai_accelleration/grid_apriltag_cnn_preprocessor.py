import json
import numpy as np
from typing import Optional
import traceback
import torch

from src.main_operations.modules.apriltags.pre_processing.ai_accelleration.utils import (
    calculate_crop_regions_from_grid,
    letterbox_image,
)
from src.utils.device_management_utils.compute_device import ComputeDevice


class GridApriltagCnnPreprocessor:
    """A class to handle AprilTag CNN preprocessing and inference.

    This class loads a trained model and performs inference on video frames to detect
    potential AprilTag locations using a grid-based prediction approach.
    """

    def __init__(
        self, model_path: str, device: ComputeDevice, conf_threshold: float = 0.15
    ) -> None:
        """Initialize the AprilTag CNN preprocessor.

        Args:
            model_path: Path to the trained model weights file.
            device: The computation device (CPU/CUDA/MX3/CORAL).
            conf_threshold: Confidence threshold for predictions.
        """
        self.model_path: str = model_path

        data_path: str = model_path.split(".")[0] + ".json"
        try:
            with open(data_path, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {data_path}")

        self.target_width: int = data["target_width"]
        self.target_height: int = data["target_height"]
        self.grid_width: int = data["grid_width"]
        self.grid_height: int = data["grid_height"]

        self.model_name: str = model_path.split("/")[-1].split(".")[0]
        self.conf_threshold: float = conf_threshold
        self.device: ComputeDevice = device

        self.scaled_frame_buffer: np.ndarray = np.zeros(
            (self.target_height, self.target_width, 3), dtype=np.uint8
        )
        self.scaled_frame_tensor_buffer: torch.Tensor = torch.zeros(
            (1, 3, self.target_height, self.target_width), dtype=torch.float32
        )

        self.stream_idx: int = self.device.register_thread_access()
        print(f"Assigned stream index: {self.stream_idx}")

        self._load_model()

    def _load_model(self) -> None:
        """Load and prepare the model for inference.

        Returns:
            None

        Raises:
            FileNotFoundError: If the model file is not found.
            RuntimeError: If there's an error loading the model.
        """
        try:
            self.device.load_model(
                self.model_path, (self.target_height, self.target_width)
            )
        except Exception as _:
            raise RuntimeError(f"Error loading model: {traceback.format_exc()}")

    def get_grid_probs(self, frame: np.ndarray) -> np.ndarray:
        """Get the grid probabilities for a frame.

        Args:
            frame: Input frame to process.
        """
        self.scaled_frame_buffer = letterbox_image(
            frame, (self.target_width, self.target_height)
        )
        self.scaled_frame_tensor_buffer = (
            torch.from_numpy(self.scaled_frame_buffer.astype(np.float32)).unsqueeze(0)
            / 255.0
        )

        logits = self.device.run(
            self.model_name,
            self.scaled_frame_tensor_buffer,
            (self.target_height, self.target_width),
            self.stream_idx,
        )

        logits = logits.reshape(logits.shape[1], logits.shape[2])

        logits = np.rot90(logits, k=1)
        logits = np.flip(logits, axis=0)

        return 1.0 / (1.0 + np.exp(-logits))

    def generate_cropped_images(
        self, frame: np.ndarray, crop_regions: list[tuple[int, int, int, int]]
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate cropped images from a frame and crop regions.

        Args:
            frame: Input frame to process.
            crop_regions: List of crop regions as (left, top, right, bottom).

        Returns:
            List of cropped images and their offsets from the top-left corner.
        """
        cropped_images = []
        offsets = []
        for region in crop_regions:
            cropped_images.append(frame[region[1] : region[3], region[0] : region[2]])
            offsets.append((region[0], region[1]))
        return list(zip(cropped_images, offsets))

    def process_frame(
        self, frame: np.ndarray, output_size: Optional[tuple[int, int]] = None
    ) -> tuple[list[tuple[int, int, int, int]], list[tuple[np.ndarray, np.ndarray]]]:
        """Process a single frame through the model.

        Args:
            frame: Input frame to process.
            output_size: Optional output size for the regions.

        Returns:
            Tuple of crop regions and list of cropped images and their offsets from the top-left corner.
        """
        probs = self.get_grid_probs(frame)

        output_frame_height, output_frame_width = frame.shape[:2]
        cell_w = output_frame_width // self.grid_width
        cell_h = output_frame_height // self.grid_height

        conf_grid_mask = probs >= self.conf_threshold

        crop_regions = calculate_crop_regions_from_grid(conf_grid_mask, cell_w, cell_h)

        if output_size is not None:
            width_scale = output_size[1] / self.target_width
            height_scale = output_size[0] / self.target_height

            crop_regions = [
                (
                    int(region[0] * width_scale),
                    int(region[1] * height_scale),
                    int(region[2] * width_scale),
                    int(region[3] * height_scale),
                )
                for region in crop_regions
            ]

        return self.generate_cropped_images(frame, crop_regions), crop_regions

    def change_conf_threshold(self, conf_threshold: float) -> None:
        """Change the confidence threshold.

        Args:
            conf_threshold: Confidence threshold for predictions.
        """
        self.conf_threshold = conf_threshold
