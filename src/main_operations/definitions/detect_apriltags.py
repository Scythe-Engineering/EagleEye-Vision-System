import numpy as np

from ..modules.apriltags.apriltag_detector import AprilTagDetector
from pupil_apriltags import Detection
from typing import List


class DetectApriltagsDefinition:
    """Definition for AprilTag detection operations."""
    def __init__(
        self,
        families: str = "tag36h11",
        nthreads: int = 1,
        quad_decimate: float = 2.0,
        quad_sigma: float = 0.0,
        refine_edges: int = 1,
        decode_sharpening: float = 0.25
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
            decode_sharpening=decode_sharpening
        )

    def run(self, image: np.ndarray) -> List[Detection] | None:
        """Detect AprilTags in the given image.

        Args:
            image: Input image array (grayscale or BGR format).

        Returns:
            List of Detection objects containing detected AprilTag information.
            None if no detections are found.
        """
        detections = self.detector.detect(image)
        
        if detections is None:
            return None
        
        # Ensure we return a list even if it's a single detection
        if isinstance(detections, list):
            return detections
        else:
            return [detections] if detections is not None else []
