from typing import Any, Dict, List, Tuple
from threading import Lock

import cv2
import numpy as np

from src.main_operations.modules.apriltags.pre_processing.temportal_acceleration.temporal_acceleration import (
    TemporalAcceleration,
)
from src.main_operations.modules.apriltags.utils.fmap_parser import load_fmap_file
from src.utils.camera_utils.load_camera_parameters import load_camera_parameters


class TemporalAccelerationPreprocessorDefinition:
    """Definition for temporal acceleration-based ROI generation.

    This operation consumes back-propagated poses and predicts ROIs for
    accelerating the AprilTag detector in the next run. The ROI outputs follow
    the same format as `PositionApriltagPreprocessor.process_frame`.
    """

    def __init__(
        self,
        camera_parameters_path: str,
        apriltag_map_path: str,
        padding_factor: float = 0.65,
        max_regions: int = 10,
        min_region_size_px: int = 16,
    ) -> None:
        """Initialize the temporal acceleration definition.

        Args:
            camera_parameters_path: Path to camera intrinsics JSON.
            apriltag_map_path: Path to fmap apriltag map JSON.
            padding_factor: Fractional padding applied to ROI size.
            max_regions: Maximum number of ROIs to return.
            min_region_size_px: Minimum side length for ROI squares.
        """
        camera_matrix, distortion_coefficients = load_camera_parameters(
            camera_parameters_path
        )
        apriltag_map = load_fmap_file(apriltag_map_path)

        self.impl = TemporalAcceleration(
            camera_matrix=camera_matrix,
            distortion_coefficients=distortion_coefficients,
            apriltag_map=apriltag_map,
            padding_factor=padding_factor,
            max_regions=max_regions,
            min_region_size_px=min_region_size_px,
        )

        self._last_regions: List[Tuple[int, int, int, int]] = []
        self._last_regions_lock: Lock = Lock()

    def back_propagate_input(self, input_data: Any) -> None:
        """Receive back-propagated input (camera pose) from the pipeline.

        Args:
            input_data: Expected to be a 4x4 camera-to-world transform (np.ndarray).
        """
        if isinstance(input_data, np.ndarray) and input_data.shape == (4, 4):
            self.impl.back_propagate_input(input_data)

    def run(self, frame: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate predicted ROIs for the current frame.

        Args:
            frame: Input frame (BGR) for which to generate ROIs.

        Returns:
            List of (cropped_image, (offset_x, offset_y)) tuples for detector input.
        """
        cropped, crop_regions = self.impl.process_frame(frame)
        with self._last_regions_lock:
            self._last_regions = crop_regions
        return (cropped, frame)

    def update_config(self, json_config: Dict[str, Any]) -> None:
        """Update live configuration for the temporal acceleration.

        Args:
            json_config: Parameters to update. Supported keys:
                - padding_factor
                - max_regions
                - min_region_size_px
        """
        if "padding_factor" in json_config:
            self.impl.padding_factor = float(json_config["padding_factor"])
        if "max_regions" in json_config:
            self.impl.max_regions = int(json_config["max_regions"])
        if "min_region_size_px" in json_config:
            self.impl.min_region_size_px = int(json_config["min_region_size_px"])

    def visualize(self, frame: np.ndarray) -> np.ndarray:
        """Visualize the temporal acceleration outputs by darkening non-predicted areas.

        Args:
            frame: Input frame to process.

        Returns:
            Frame with non-predicted areas darkened.
        """
        with self._last_regions_lock:
            crop_regions = self._last_regions

        # Start with a copy of the frame at low brightness so that the predicted areas are more visible
        visualization_frame = cv2.convertScaleAbs(frame, alpha=0.3, beta=0)

        # Copy the crop regions (which are the predicted areas) to the darkened frame
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
