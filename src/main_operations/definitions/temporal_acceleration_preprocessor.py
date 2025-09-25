from threading import Lock
from typing import List, Optional, Tuple

import numpy as np
from imutils.convenience import cv2

from src.utils.device_management_utils.compute_pool import ComputePool

from ..modules.apriltags.pre_processing.temportal_acceleration.temporal_acceleration import (
    TemporalAccelerationPreprocessor,
)


class TemporalAccelerationPreprocessorDefinition:
    """Definition for temporal ROI acceleration using back-propagated detections.

    Input: np.ndarray frame (BGR)
    Output: list of (cropped_image, (offset_x, offset_y)) tuples
    """

    def __init__(
        self,
        compute_pool: ComputePool,
        padding_factor: float = 0.3,
        max_missed_updates: int = 2,
        velocity_smoothing: float = 0.5,
        match_distance_px: float = 80.0,
        max_tracks: int = 16,
    ) -> None:
        """Initialize the temporal acceleration definition.

        Args:
            compute_pool: Injected compute pool (unused, kept for consistency).
            padding_factor: Fractional padding around each ROI when cropping.
            max_missed_updates: Number of consecutive updates without a match before a track is removed.
            velocity_smoothing: Exponential smoothing factor in [0, 1] for velocity updates.
            match_distance_px: Maximum center distance in pixels to associate detections to existing tracks.
            max_tracks: Maximum number of simultaneous tracks to maintain.
        """
        self.preprocessor = TemporalAccelerationPreprocessor(
            padding_factor=padding_factor,
            max_missed_updates=max_missed_updates,
            velocity_smoothing=velocity_smoothing,
            match_distance_px=match_distance_px,
            max_tracks=max_tracks,
        )
        self.last_crop_regions: list[tuple[int, int, int, int]] = []
        self.last_crop_regions_lock: Lock = Lock()

    def run(
        self, frame: np.ndarray, output_size: Optional[Tuple[int, int]] = None
    ) -> List[tuple[np.ndarray, tuple[int, int]]]:
        """Process a frame to generate temporal ROIs.

        Args:
            frame: Input frame to process.
            output_size: Optional output size for scaling the regions.

        Returns:
            List of (cropped_image, (offset_x, offset_y)) tuples.
        """
        outputs, crop_regions = self.preprocessor.process_frame(frame, output_size)
        with self.last_crop_regions_lock:
            self.last_crop_regions = crop_regions
        return outputs

    def update_config(self, json_config: dict) -> None:
        """Update live configuration values.

        Args:
            json_config: JSON configuration for the temporal preprocessor.
        """
        if "padding_factor" in json_config:
            self.preprocessor.change_padding_factor(json_config["padding_factor"])

    def back_propagate_input(self, input_data) -> None:
        """Receive back-propagated detections to update ROI tracks.

        Args:
            input_data: Detections in camera space.
        """
        self.preprocessor.back_propagate_input(input_data)

    def visualize(self, frame: np.ndarray) -> np.ndarray:
        """Visualize the current ROIs by highlighting predicted regions.

        Args:
            frame: Input frame to visualize over.

        Returns:
            Visualization frame with predicted ROIs overlaid.
        """
        with self.last_crop_regions_lock:
            crop_regions = list(self.last_crop_regions)

        visualization_frame = cv2.convertScaleAbs(frame, alpha=0.3, beta=0)
        for left, top, right, bottom in crop_regions:
            left = max(0, left)
            top = max(0, top)
            right = min(frame.shape[1], right)
            bottom = min(frame.shape[0], bottom)
            if right > left and bottom > top:
                visualization_frame[top:bottom, left:right] = frame[
                    top:bottom, left:right
                ]
        return visualization_frame
