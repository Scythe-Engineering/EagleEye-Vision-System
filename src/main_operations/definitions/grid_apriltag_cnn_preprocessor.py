import numpy as np
from typing import Optional, Tuple
from threading import Lock

from ..modules.apriltags.pre_processing.ai_acceleration.grid_apriltag_cnn_preprocessor import (
    GridApriltagCnnPreprocessor,
)
from src.utils.device_management_utils.compute_pool import ComputePool

BoundingBox = tuple[int, int, int, int]


class GridApriltagCnnPreprocessorDefinition:
    """Definition for AprilTag CNN preprocessing operations."""

    def __init__(
        self,
        model_path: str,
        device_id: str,
        compute_pool: ComputePool,
        conf_threshold: float = 0.15,
    ) -> None:
        """Initialize the AprilTag CNN preprocessor definition.

        Args:
            model_path: Path to the trained model weights file.
            device_id: The id of the computation device (CPU/CUDA/MX3/CORAL).
            compute_pool: The compute pool to use for the pipelines.
            conf_threshold: Confidence threshold for predictions.
        """
        self.preprocessor = GridApriltagCnnPreprocessor(
            model_path=model_path,
            device=compute_pool.get_compute_device(device_id),
            conf_threshold=conf_threshold,
        )

        self.last_crop_regions: list[BoundingBox] = []
        self.last_crop_regions_lock: Lock = Lock()

    def run(
        self, frame: np.ndarray, output_size: Optional[Tuple[int, int]] = None
    ) -> Optional[np.ndarray]:
        """Process a frame through the CNN preprocessor.

        Args:
            frame: Input frame to process.
            output_size: Optional output size for scaling the regions.

        Returns:
            Processed frame with non-ROI regions replaced with black pixels,
            or None when processing fails.
        """
        with self.last_crop_regions_lock:
            outputs, self.last_crop_regions = self.preprocessor.process_frame(
                frame, output_size
            )
        if outputs is None:
            return None
        return outputs

    def update_config(self, json_config: dict) -> None:
        """Update the configuration of the CNN preprocessor. Only conf threshold is updated (restart is required for other changes).

        Args:
            json_config: JSON configuration for the CNN preprocessor.
        """
        if "conf_threshold" in json_config:
            self.preprocessor.change_conf_threshold(json_config["conf_threshold"])

    def visualize(self, frame: np.ndarray) -> np.ndarray:
        """Visualize the CNN preprocessor outputs by blacking out eliminated grid squares.

        Args:
            frame: Input frame to process.

        Returns:
            Frame with eliminated grid squares blackened.
        """
        with self.last_crop_regions_lock:
            crop_regions = self.last_crop_regions

        # Start with a black frame
        visualization_frame = np.zeros_like(frame)

        # Copy the crop regions (which are the non-eliminated areas) to the black frame
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
