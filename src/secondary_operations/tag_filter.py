from typing import List, Union, Optional
import numpy as np
import cv2
from threading import Lock
from pupil_apriltags import Detection

from src.main_operations.modules.apriltags.apriltag_detector import CustomDetection


class TagFilter:
    """Filter AprilTag detections based on whitelist or blacklist of tag IDs."""

    def __init__(
        self,
        filter_mode: str = "whitelist",
        tag_ids: List[int] = None,
    ) -> None:
        """Initialize the tag filter.

        Args:
            filter_mode: Either "whitelist" or "blacklist". In whitelist mode,
                        only tags with IDs in the tag_ids list are kept. In blacklist mode,
                        tags with IDs in the tag_ids list are removed.
            tag_ids: List of tag IDs to filter by. Behavior depends on filter_mode.
        """
        if tag_ids is None:
            tag_ids = []

        self.filter_mode = filter_mode
        self.tag_ids = set(tag_ids)  # Use set for O(1) lookup

        if self.filter_mode not in ["whitelist", "blacklist"]:
            raise ValueError(f"filter_mode must be 'whitelist' or 'blacklist', got '{filter_mode}'")

        self.last_input_detections: Optional[Union[List[Detection], List[CustomDetection]]] = None
        self.last_input_detections_lock: Lock = Lock()

    def run(self, detections: Union[List[Detection], List[CustomDetection], None]) -> Union[List[Detection], List[CustomDetection], None]:
        """Filter the detections based on the configured whitelist/blacklist.

        Args:
            detections: List of Detection or CustomDetection objects, or None.

        Returns:
            Filtered list of detections, or None if input is None.
        """
        # Store the input detections for visualization
        with self.last_input_detections_lock:
            self.last_input_detections = detections

        if detections is None:
            return None

        filtered_detections = []

        for detection in detections:
            tag_id = detection.tag_id

            if self.filter_mode == "whitelist":
                # Keep only tags in the whitelist
                if tag_id in self.tag_ids:
                    filtered_detections.append(detection)
            elif self.filter_mode == "blacklist":
                # Remove tags in the blacklist
                if tag_id not in self.tag_ids:
                    filtered_detections.append(detection)
            else:
                # If no tag_ids provided, all IDs pass
                filtered_detections.append(detection)

        return filtered_detections

    def update_config(self, json_config: dict) -> None:
        """Update the configuration of the tag filter.

        Args:
            json_config: JSON configuration for the tag filter.
        """
        if "filter_mode" in json_config:
            new_mode = json_config["filter_mode"]
            if new_mode not in ["whitelist", "blacklist"]:
                raise ValueError(f"filter_mode must be 'whitelist' or 'blacklist', got '{new_mode}'")
            self.filter_mode = new_mode

        if "tag_ids" in json_config:
            self.tag_ids = set(json_config["tag_ids"])

    def visualize(self, frame: np.ndarray) -> np.ndarray:
        """Visualize the tag filter by drawing all input detections with color coding.

        Green boxes indicate tags that are kept by the filter.
        Red boxes indicate tags that are excluded by the filter.

        Args:
            frame: Input frame to draw detections on.

        Returns:
            Frame with filtered AprilTag detections drawn on it.
        """
        visualization_frame = frame.copy()

        with self.last_input_detections_lock:
            detections = self.last_input_detections

        if detections is not None:
            for detection in detections:
                # Determine if this detection would be kept or excluded
                tag_id = detection.tag_id
                is_kept = False

                if self.filter_mode == "whitelist":
                    is_kept = tag_id in self.tag_ids
                elif self.filter_mode == "blacklist":
                    is_kept = tag_id not in self.tag_ids
                else:
                    # If no tag_ids provided, all IDs are kept
                    is_kept = True

                # Choose color based on whether the tag is kept or excluded
                color = (0, 255, 0) if is_kept else (0, 0, 255)  # Green for kept, Red for excluded

                # Draw the bounding box
                corners = detection.corners.astype(int)
                cv2.polylines(visualization_frame, [corners], True, color, 2)

                # Draw the tag ID at the center
                center_x = int(corners[:, 0].mean())
                center_y = int(corners[:, 1].mean())
                cv2.putText(
                    visualization_frame,
                    f"ID: {detection.tag_id}",
                    (center_x, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

        return visualization_frame
