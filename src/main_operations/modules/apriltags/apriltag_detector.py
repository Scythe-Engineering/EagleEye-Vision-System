from typing import Optional

import cv2
import numpy as np
from pupil_apriltags import Detector, Detection


class AprilTagDetector:
    """A configurable AprilTag detector that exposes all detector parameters.
    
    This class provides a clean interface for AprilTag detection with full control
    over all detector parameters. It can be used independently from pose estimation.
    """

    def __init__(
        self,
        families: str = "tag36h11",
        nthreads: int = 1,
        quad_decimate: float = 2.0,
        quad_sigma: float = 0.0,
        refine_edges: int = 1,
        decode_sharpening: float = 0.25
    ) -> None:
        """Initialize the AprilTag detector with configurable parameters.

        Args:
            families: AprilTag family to detect (e.g., "tag16h5", "tag25h9", "tag36h11").
            nthreads: Number of threads to use for detection.
            quad_decimate: Detection of quads can be done on a lower-resolution image,
                          improving speed at a cost of pose accuracy and a slight
                          decrease in detection rate. Decoding the binary payload is
                          still done at full resolution.
            quad_sigma: What Gaussian blur should be applied to the segmented image
                       (used for quad detection). Parameter is the standard deviation
                       in pixels. Very noisy images benefit from non-zero values
                       (e.g., 0.8).
            refine_edges: When non-zero, the edges of the each quad are adjusted to
                         "snap to" strong gradients nearby. This is useful when
                         decimation is used, as it can increase the quality of the
                         initial quad estimate substantially. Generally recommended
                         to be on (1). Very computationally inexpensive. Option is
                         ignored if quad_decimate = 1.
            decode_sharpening: How much sharpening should be done to decoded images?
                              This can help decode small tags but may or may not help
                              in odd lighting conditions or low light conditions.
        """
        self.families = families
        self.nthreads = nthreads
        self.quad_decimate = quad_decimate
        self.quad_sigma = quad_sigma
        self.refine_edges = refine_edges
        self.decode_sharpening = decode_sharpening
        
        self.detector = Detector(
            families=self.families,
            nthreads=self.nthreads,
            quad_decimate=self.quad_decimate,
            quad_sigma=self.quad_sigma,
            refine_edges=self.refine_edges,
            decode_sharpening=self.decode_sharpening,
        )


    def update_parameters(
        self,
        families: Optional[str] = None,
        nthreads: Optional[int] = None,
        quad_decimate: Optional[float] = None,
        quad_sigma: Optional[float] = None,
        refine_edges: Optional[int] = None,
        decode_sharpening: Optional[float] = None,
    ) -> None:
        """Update detector parameters and recreate the detector.

        Args:
            families: AprilTag family to detect.
            nthreads: Number of threads to use for detection.
            quad_decimate: Quad detection decimation factor.
            quad_sigma: Gaussian blur standard deviation for quad detection.
            refine_edges: Whether to refine quad edges.
            decode_sharpening: Sharpening amount for decoded images.
        """
        if families is not None:
            self.families = families
        if nthreads is not None:
            self.nthreads = nthreads
        if quad_decimate is not None:
            self.quad_decimate = quad_decimate
        if quad_sigma is not None:
            self.quad_sigma = quad_sigma
        if refine_edges is not None:
            self.refine_edges = refine_edges
        if decode_sharpening is not None:
            self.decode_sharpening = decode_sharpening

        self.detector = Detector(
            families=self.families,
            nthreads=self.nthreads,
            quad_decimate=self.quad_decimate,
            quad_sigma=self.quad_sigma,
            refine_edges=self.refine_edges,
            decode_sharpening=self.decode_sharpening,
        )

    def detect(
        self,
        image: np.ndarray,
    ) -> Detection:
        """Detect AprilTags in an image.
        Note:
        - Input image is always converted to grayscale.

        Args:
            image: Input image (grayscale or BGR).

        Returns:
            List of Detection objects containing tag information.
        """
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image.copy()

        gray_image = gray_image.astype(np.uint8)

        return self.detector.detect(
            gray_image
        )
 