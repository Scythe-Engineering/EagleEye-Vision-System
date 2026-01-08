import cv2
import numpy as np
from threading import Lock

from ..modules.apriltags.apriltag_detector import AprilTagDetector
from pupil_apriltags import Detection
from ..modules.apriltags.apriltag_detector import CustomDetection
from typing import List, Optional
from src.main_operations.definitions.base.base_class import OperationInstance


class DetectApriltagsDefinition(OperationInstance):
    """Definition for AprilTag detection operations."""

    def __init__(
        self,
        families: str = "tag36h11",
        nthreads: int = 1,
        quad_decimate: float = 2.0,
        quad_sigma: float = 0.0,
        refine_edges: int = 1,
        decode_sharpening: float = 0.25,
    ) -> None:
        """Initialize the AprilTag detection definition.

        Args:
            families: AprilTag family to detect (e.g., "tag16h5", "tag25h9", "tag36h11").
            nthreads: Number of threads to use for detection.
            quad_decimate: Detection of quads can be done on a lower-resolution image,
                          improving speed at a cost of pose accuracy and a slight
                          decrease in detection rate.
            quad_sigma: What Gaussian blur should be applied to the segmented image
                       (used for quad detection). Parameter is the standard deviation
                       in pixels.
            refine_edges: When non-zero, the edges of the each quad are adjusted to
                         "snap to" strong gradients nearby.
            decode_sharpening: How much sharpening should be done to decoded images?
            tag_size: Physical size of tags in meters for pose estimation.
        """
        self.detector = AprilTagDetector(
            families=families,
            nthreads=nthreads,
            quad_decimate=quad_decimate,
            quad_sigma=quad_sigma,
            refine_edges=refine_edges,
            decode_sharpening=decode_sharpening,
        )

        self.last_detections: Optional[List[Detection] | List[CustomDetection]] = None
        self.last_detections_lock: Lock = Lock()

    def run(self, input_data) -> List[Detection] | List[CustomDetection] | None:
        """Detect AprilTags in the given image or image segments.

        Args:
            input_data: Either a single image array (np.ndarray) or a tuple of
                       (segments, full_frame) where segments is a list of (image, offset) tuples.

        Returns:
            List of Detection objects containing detected AprilTag information.
            None if no detections are found.
        """
        # Handle input from temporal acceleration preprocessor (tuple) or direct image input
        if isinstance(input_data, tuple) and len(input_data) == 2:
            segments, full_frame = input_data
            detections = self.detector.detect(segments, full_frame)
        else:
            detections = self.detector.detect(input_data)

        with self.last_detections_lock:
            self.last_detections = detections

        if detections is None or (
            isinstance(detections, list) and len(detections) == 0
        ):
            return None

        # Ensure we return a list even if it's a single detection
        if isinstance(detections, list):
            return detections
        else:
            return [detections] if detections is not None else []

    def update_config(self, json_config: dict) -> None:
        """Update the configuration of the AprilTag detector. Only live-updatable parameters are changed.

        Args:
            json_config: JSON configuration for the AprilTag detector.
        """
        update_params = {}
        if "families" in json_config:
            update_params["families"] = json_config["families"]
        if "nthreads" in json_config:
            update_params["nthreads"] = json_config["nthreads"]
        if "quad_decimate" in json_config:
            update_params["quad_decimate"] = json_config["quad_decimate"]
        if "quad_sigma" in json_config:
            update_params["quad_sigma"] = json_config["quad_sigma"]
        if "refine_edges" in json_config:
            update_params["refine_edges"] = json_config["refine_edges"]
        if "decode_sharpening" in json_config:
            update_params["decode_sharpening"] = json_config["decode_sharpening"]

        if update_params:
            self.detector.update_parameters(**update_params)

    def visualize(self, frame: np.ndarray) -> np.ndarray:
        """Visualize the AprilTag detections by drawing them on the frame.

        Args:
            frame: Input frame to draw detections on.

        Returns:
            Frame with detected AprilTags drawn on it.
        """
        visualization_frame = frame.copy()

        with self.last_detections_lock:
            detections = self.last_detections

        if detections is not None:
            for detection in detections:
                # Draw the bounding box
                corners = detection.corners.astype(int)
                cv2.polylines(visualization_frame, [corners], True, (0, 255, 0), 2)

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
