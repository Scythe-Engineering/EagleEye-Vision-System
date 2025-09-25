from threading import Lock
from typing import List, Optional, Tuple

import numpy as np
from imutils.convenience import cv2
from temporal_acceleration import TemporalAcceleration as RustTemporalAcceleration

from src.utils.device_management_utils.compute_pool import ComputePool


class TemporalAccelerationPreprocessorRustDefinition:
    """Rust-backed temporal ROI acceleration using back-propagated detections.

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
        """Initialize the Rust-based temporal acceleration definition.

        Args:
            compute_pool: Injected compute pool (unused, kept for consistency).
            padding_factor: Fractional padding around each ROI when cropping.
            max_missed_updates: Number of consecutive updates without a match before a track is removed.
            velocity_smoothing: Exponential smoothing factor in [0, 1] for velocity updates.
            match_distance_px: Maximum center distance in pixels to associate detections to existing tracks.
            max_tracks: Maximum number of simultaneous tracks to maintain.
        """
        if RustTemporalAcceleration is None:
            raise ImportError(
                "Rust temporal_acceleration module not available. Please build the Rust extension first."
            )

        self.preprocessor = RustTemporalAcceleration(
            padding_factor=float(padding_factor),
            max_missed_updates=int(max_missed_updates),
            velocity_smoothing=float(velocity_smoothing),
            match_distance_px=float(match_distance_px),
            max_tracks=int(max_tracks),
        )
        self.last_crop_regions: list[tuple[int, int, int, int]] = []
        self.last_crop_regions_lock: Lock = Lock()

    def run(
        self, frame: np.ndarray, output_size: Optional[Tuple[int, int]] = None
    ) -> List[tuple[np.ndarray, tuple[int, int]]]:
        """Process a frame to generate temporal ROIs using the Rust backend.

        Args:
            frame: Input frame to process.
            output_size: Optional output size for scaling the regions.

        Returns:
            List of (cropped_image, (offset_x, offset_y)) tuples.
        """
        height, width = frame.shape[:2]
        regions = self.preprocessor.process(int(width), int(height))

        crop_regions: list[tuple[int, int, int, int]] = []
        cropped_images_with_offsets: list[tuple[np.ndarray, tuple[int, int]]] = []

        for left, top, right, bottom in regions:
            if right <= left or bottom <= top:
                continue
            left_i = max(0, int(left))
            top_i = max(0, int(top))
            right_i = min(width, int(right))
            bottom_i = min(height, int(bottom))
            cropped = frame[top_i:bottom_i, left_i:right_i]
            crop_regions.append((left_i, top_i, right_i, bottom_i))
            cropped_images_with_offsets.append((cropped, (left_i, top_i)))

        if not crop_regions:
            crop_regions = [(0, 0, width, height)]
            cropped_images_with_offsets = [(frame, (0, 0))]

        with self.last_crop_regions_lock:
            self.last_crop_regions = crop_regions

        return cropped_images_with_offsets

    def update_config(self, json_config: dict) -> None:
        """Update live configuration values for the Rust backend.

        Args:
            json_config: JSON configuration for the temporal preprocessor.
        """
        self.preprocessor.update_config(json_config)

    def back_propagate_input(self, input_data) -> None:
        """Receive back-propagated detections to update ROI tracks in Rust backend.

        Args:
            input_data: Detections in camera space.
        """
        detections: list[tuple[float, float, float]] = []
        if input_data is None:
            self.preprocessor.back_propagate_input(None)
            return
        if isinstance(input_data, (list, tuple)):
            for det in input_data:
                if isinstance(det, (list, tuple)) and len(det) >= 3:
                    detections.append((float(det[0]), float(det[1]), float(det[2])))
        else:
            detections.append(
                (
                    float(getattr(input_data, 0, 0.0)),
                    float(getattr(input_data, 1, 0.0)),
                    float(getattr(input_data, 2, 1.0)),
                )
            )
        self.preprocessor.back_propagate_input(detections)

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
