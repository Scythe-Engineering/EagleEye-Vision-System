import cv2
import numpy as np
from typing import Optional, Tuple
from threading import Lock

from ..modules.apriltags.pre_processing.ai_acceleration.position_apriltag_preprocessor import (
    PositionApriltagPreprocessor,
)
from src.utils.device_management_utils.compute_pool import ComputePool


class PositionApriltagPreprocessorDefinition:
    """Definition for position-based AprilTag preprocessing operations."""

    def __init__(
        self,
        model_path: str,
        device_id: str,
        compute_pool: ComputePool,
        conf_threshold: float = 0.5,
        padding_factor: float = 0.3,
    ) -> None:
        """Initialize the position-based AprilTag preprocessor definition.

        Args:
            model_path: Path to the trained model weights file.
            device_id: The id of the computation device (CPU/CUDA/MX3/CORAL).
            compute_pool: The compute pool to use for the pipelines.
            conf_threshold: Confidence threshold for predictions (0-1).
            padding_factor: Factor to pad around detected positions.
        """
        self.preprocessor = PositionApriltagPreprocessor(
            model_path=model_path,
            device=compute_pool.get_compute_device(device_id),
            conf_threshold=conf_threshold,
            padding_factor=padding_factor,
        )

        self.last_crop_regions: list[tuple[int, int, int, int]] = []
        self.last_crop_regions_lock: Lock = Lock()

    def run(
        self, frame: np.ndarray, output_size: Optional[Tuple[int, int]] = None
    ) -> list[tuple[np.ndarray, tuple[int, int]]]:
        """Process a frame through the position-based preprocessor.

        Args:
            frame: Input frame to process.
            output_size: Optional output size for scaling the regions.

        Returns:
            List of tuples containing processed regions and their coordinates, or None if no outputs.
        """
        with self.last_crop_regions_lock:
            outputs, self.last_crop_regions = self.preprocessor.process_frame(
                frame, output_size
            )
        if outputs is None:
            return None
        return outputs

    def update_config(self, json_config: dict) -> None:
        """Update the configuration of the position preprocessor.

        Args:
            json_config: JSON configuration for the position preprocessor.
        """
        if "conf_threshold" in json_config:
            self.preprocessor.change_conf_threshold(json_config["conf_threshold"])
        if "padding_factor" in json_config:
            self.preprocessor.change_padding_factor(json_config["padding_factor"])

    def visualize(self, frame: np.ndarray) -> np.ndarray:
        """Visualize the position preprocessor outputs by blacking out non-detected areas.

        Args:
            frame: Input frame to process.

        Returns:
            Frame with non-detected areas blackened.
        """
        with self.last_crop_regions_lock:
            crop_regions = self.last_crop_regions

        # Start with a copy of the frame at low brightness so that the detected areas are more visible
        visualization_frame = cv2.convertScaleAbs(frame, alpha=0.3, beta=0)

        # Copy the crop regions (which are the detected areas) to the black frame
        for region in crop_regions:
            left, top, right, bottom = region
            # Ensure coordinates are within frame bounds
            left = max(0, left)
            top = max(0, top)
            right = min(frame.shape[1], right)
            bottom = min(frame.shape[0], bottom)

            if right > left and bottom > top:
                visualization_frame[top:bottom, left:right] = frame[
                    top:bottom, left:right
                ]

        return visualization_frame
