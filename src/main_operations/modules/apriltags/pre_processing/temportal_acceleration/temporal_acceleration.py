from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class RoiTrack:
    center_x_px: float
    center_y_px: float
    size_px: float
    velocity_x_px: float
    velocity_y_px: float
    velocity_size_px: float
    missed_updates: int


class TemporalAccelerationPreprocessor:
    """Temporal ROI preprocessor driven by back-propagated AprilTag detections.

    This preprocessor maintains tracked regions of interest (ROIs) in camera space.
    Tracks are updated from back-propagated detections, and between updates their
    positions and sizes are predicted using a simple constant-velocity model.

    If an ROI has no matching detection for ``max_missed_updates`` consecutive
    updates, it is removed. If no ROIs remain when processing a frame, the full
    frame is returned as a single ROI.
    """

    def __init__(
        self,
        padding_factor: float = 0.3,
        max_missed_updates: int = 2,
        velocity_smoothing: float = 0.5,
        match_distance_px: float = 80.0,
        max_tracks: int = 16,
    ) -> None:
        """Initialize the temporal preprocessor.

        Args:
            padding_factor: Fractional padding around each ROI when cropping.
            max_missed_updates: Number of consecutive updates without a match before a track is removed.
            velocity_smoothing: Exponential smoothing factor in [0, 1] for velocity updates.
            match_distance_px: Maximum center distance in pixels to associate detections to existing tracks.
            max_tracks: Maximum number of simultaneous tracks to maintain.
        """
        self.padding_factor: float = padding_factor
        self.max_missed_updates: int = int(max_missed_updates)
        self.velocity_smoothing: float = float(velocity_smoothing)
        self.match_distance_px: float = float(match_distance_px)
        self.max_tracks: int = int(max_tracks)

        self.tracks: List[RoiTrack] = []

    def change_padding_factor(self, padding_factor: float) -> None:
        """Change the padding factor used for crop generation.

        Args:
            padding_factor: Fractional padding around each ROI when cropping.
        """
        self.padding_factor = padding_factor

    def _compute_center_and_size_from_corners(
        self, corners: np.ndarray
    ) -> Tuple[float, float, float]:
        """Compute center and approximate square size from quadrilateral corners.

        Args:
            corners: Array of shape (4, 2) in pixel coordinates.

        Returns:
            Tuple of (center_x_px, center_y_px, size_px).
        """
        center_x_px = float(np.mean(corners[:, 0]))
        center_y_px = float(np.mean(corners[:, 1]))
        dx = float(np.max(corners[:, 0]) - np.min(corners[:, 0]))
        dy = float(np.max(corners[:, 1]) - np.min(corners[:, 1]))
        size_px = float(max(dx, dy))
        return center_x_px, center_y_px, max(1.0, size_px)

    def _parse_detection(
        self, detection: object
    ) -> Optional[Tuple[float, float, float]]:
        """Parse a single detection into (cx, cy, size) in pixels if possible.

        Supports `pupil_apriltags.Detection` (via `.corners`) and tuple-like inputs
        of the form (cx, cy, size) or (cx, cy, w, h).

        Args:
            detection: Detection object or tuple-like.

        Returns:
            Parsed (cx, cy, size) tuple, or None if unsupported.
        """
        if detection is None:
            return None

        if hasattr(detection, "corners"):
            corners = np.asarray(getattr(detection, "corners"), dtype=np.float32)
            if corners.shape == (4, 2):
                return self._compute_center_and_size_from_corners(corners)

        if isinstance(detection, (list, tuple)) and len(detection) >= 3:
            cx = float(detection[0])
            cy = float(detection[1])
            if len(detection) >= 4:
                w = float(detection[2])
                h = float(detection[3])
                return cx, cy, max(1.0, max(w, h))
            else:
                size = float(detection[2])
                return cx, cy, max(1.0, size)

        return None

    def _associate_detections(
        self,
        detections: List[Tuple[float, float, float]],
    ) -> Tuple[List[Optional[int]], List[bool]]:
        """Associate detections to existing tracks by nearest-neighbor matching.

        Args:
            detections: List of (cx, cy, size) tuples.

        Returns:
            Tuple of (assigned_track_index_per_detection, track_matched_flags).
        """
        if not self.tracks or not detections:
            return [None] * len(detections), [False] * len(self.tracks)

        assigned_track_indices: List[Optional[int]] = [None] * len(detections)
        track_matched: List[bool] = [False] * len(self.tracks)

        for det_idx, (cx, cy, _) in enumerate(detections):
            best_track_index: Optional[int] = None
            best_distance: float = float("inf")
            for track_index, track in enumerate(self.tracks):
                if track_matched[track_index]:
                    continue
                dx = cx - track.center_x_px
                dy = cy - track.center_y_px
                distance = float(np.hypot(dx, dy))
                if distance < best_distance and distance <= self.match_distance_px:
                    best_distance = distance
                    best_track_index = track_index
            assigned_track_indices[det_idx] = best_track_index
            if best_track_index is not None:
                track_matched[best_track_index] = True

        return assigned_track_indices, track_matched

    def back_propagate_input(self, input_data: object) -> None:
        """Update tracks using back-propagated detections in camera space.

        Args:
            input_data: Iterable of detection objects or tuples in pixel units.
        """
        if input_data is None:
            return

        parsed_detections: List[Tuple[float, float, float]] = []
        if isinstance(input_data, np.ndarray):
            input_iterable: Iterable = list(input_data)
        elif isinstance(input_data, (list, tuple)):
            input_iterable = input_data
        else:
            input_iterable = [input_data]

        for det in input_iterable:
            parsed = self._parse_detection(det)
            if parsed is not None:
                parsed_detections.append(parsed)

        if not parsed_detections and not self.tracks:
            return

        assigned, track_matched = self._associate_detections(parsed_detections)

        new_tracks: List[RoiTrack] = []

        for track_index, track in enumerate(self.tracks):
            if track_index < len(track_matched) and track_matched[track_index]:
                matched_det_index = None
                for det_idx, assigned_index in enumerate(assigned):
                    if assigned_index == track_index:
                        matched_det_index = det_idx
                        break
                if matched_det_index is None:
                    track.missed_updates += 1
                    new_tracks.append(track)
                    continue

                det_cx, det_cy, det_size = parsed_detections[matched_det_index]
                new_velocity_x = det_cx - track.center_x_px
                new_velocity_y = det_cy - track.center_y_px
                new_velocity_size = det_size - track.size_px

                alpha = self.velocity_smoothing
                track.velocity_x_px = (
                    1.0 - alpha
                ) * track.velocity_x_px + alpha * new_velocity_x
                track.velocity_y_px = (
                    1.0 - alpha
                ) * track.velocity_y_px + alpha * new_velocity_y
                track.velocity_size_px = (
                    1.0 - alpha
                ) * track.velocity_size_px + alpha * new_velocity_size

                track.center_x_px = det_cx
                track.center_y_px = det_cy
                track.size_px = det_size
                track.missed_updates = 0
                new_tracks.append(track)
            else:
                track.missed_updates += 1
                if track.missed_updates <= self.max_missed_updates:
                    new_tracks.append(track)

        for det_idx, assigned_index in enumerate(assigned):
            if assigned_index is None and len(new_tracks) < self.max_tracks:
                det_cx, det_cy, det_size = parsed_detections[det_idx]
                new_tracks.append(
                    RoiTrack(
                        center_x_px=det_cx,
                        center_y_px=det_cy,
                        size_px=det_size,
                        velocity_x_px=0.0,
                        velocity_y_px=0.0,
                        velocity_size_px=0.0,
                        missed_updates=0,
                    )
                )

        self.tracks = new_tracks

    def _predict_tracks(self) -> None:
        """Predict track state forward by one step using current velocities."""
        for track in self.tracks:
            track.center_x_px += track.velocity_x_px
            track.center_y_px += track.velocity_y_px
            track.size_px = max(1.0, track.size_px + track.velocity_size_px)

    def _create_crop_region(
        self,
        center_x_px: float,
        center_y_px: float,
        size_px: float,
        frame_width: int,
        frame_height: int,
    ) -> Tuple[int, int, int, int]:
        """Create an integer crop region with padding and clamped to frame bounds.

        Args:
            center_x_px: Center x coordinate of the ROI in pixels.
            center_y_px: Center y coordinate of the ROI in pixels.
            size_px: Side length of the ROI square in pixels.
            frame_width: Frame width in pixels.
            frame_height: Frame height in pixels.

        Returns:
            Crop region as (left, top, right, bottom) in pixels.
        """
        half_size = (size_px * (1.0 + self.padding_factor)) * 0.5
        left = max(0, int(center_x_px - half_size))
        top = max(0, int(center_y_px - half_size))
        right = min(frame_width, int(center_x_px + half_size))
        bottom = min(frame_height, int(center_y_px + half_size))
        return left, top, right, bottom

    def process_frame(
        self, frame: np.ndarray, _: Optional[Tuple[int, int]] = None
    ) -> Tuple[
        List[Tuple[np.ndarray, Tuple[int, int]]], List[Tuple[int, int, int, int]]
    ]:
        """Produce cropped ROIs for the current frame using tracked predictions.

        Args:
            frame: Current camera frame in BGR format.
            output_size: Unused placeholder for interface consistency.

        Returns:
            Tuple of (cropped_images_with_offsets, crop_regions).
        """
        if not self.tracks:
            frame_height, frame_width = frame.shape[:2]
            entire_region = (0, 0, frame_width, frame_height)
            return [(frame, (0, 0))], [entire_region]

        self._predict_tracks()

        frame_height, frame_width = frame.shape[:2]
        crop_regions: List[Tuple[int, int, int, int]] = []
        cropped_images_with_offsets: List[Tuple[np.ndarray, Tuple[int, int]]] = []

        for track in self.tracks:
            left, top, right, bottom = self._create_crop_region(
                track.center_x_px,
                track.center_y_px,
                track.size_px,
                frame_width,
                frame_height,
            )
            if right <= left or bottom <= top:
                continue
            cropped = frame[top:bottom, left:right]
            crop_regions.append((left, top, right, bottom))
            cropped_images_with_offsets.append((cropped, (left, top)))

        if not crop_regions:
            frame_height, frame_width = frame.shape[:2]
            entire_region = (0, 0, frame_width, frame_height)
            return [(frame, (0, 0))], [entire_region]

        return cropped_images_with_offsets, crop_regions
