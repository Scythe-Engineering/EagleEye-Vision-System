from dataclasses import dataclass
import logging
from threading import Lock
from typing import Optional, cast

import cv2
import numpy as np
from pupil_apriltags import Detector, Detection


logger = logging.getLogger(__name__)


@dataclass
class CustomDetection:
    """A detection of an AprilTag.

    Attributes:
        tag_id: The ID of the detected AprilTag.
        corners: The corners of the detected AprilTag.
    """

    tag_id: int
    corners: np.ndarray


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
        decode_sharpening: float = 0.25,
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

        self.ready = False
        self._detect_lock: Lock = Lock()

        self.detector = self._create_detector(
            self.families,
            self.nthreads,
            self.quad_decimate,
            self.quad_sigma,
            self.refine_edges,
            self.decode_sharpening,
        )
        self.ready = True

    def _create_detector(
        self,
        families: str,
        nthreads: int,
        quad_decimate: float,
        quad_sigma: float,
        refine_edges: int,
        decode_sharpening: float,
    ) -> Detector:
        """Create the native AprilTag detector with normalized parameters."""
        return Detector(
            families=families,
            nthreads=max(1, int(nthreads)),
            quad_decimate=max(1.0, float(quad_decimate)),
            quad_sigma=max(0.0, float(quad_sigma)),
            refine_edges=int(refine_edges),
            decode_sharpening=float(decode_sharpening),
        )

    def _disable_native_destructor(self, detector: Detector) -> None:
        """Prevent a known unsafe pupil_apriltags destructor path during reconfigure.

        pupil_apriltags.Detector.__del__ releases C pointers. On macOS this has
        been observed to segfault while replacing a detector from the WebUI config
        path. The old detector is no longer used after the lock-protected swap, so
        clearing these attributes lets Python drop the wrapper without entering
        the crashing native destroy path.
        """
        try:
            if hasattr(detector, "tag_detector_ptr"):
                detector.tag_detector_ptr = None
            if hasattr(detector, "tag_families"):
                detector.tag_families = {}
        except Exception as exc:
            logger.warning("Failed to disable old AprilTag detector destructor: %s", exc)

    def _preprocess_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Preprocess image to grayscale uint8 format.

        Args:
            image: Input image (grayscale or BGR).

        Returns:
            Preprocessed grayscale image or None if invalid.
        """
        if image is None or image.size == 0:
            return None

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray_image = np.empty(image.shape[:2], dtype=np.uint8)
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY, dst=gray_image)
        else:
            if image.dtype != np.uint8:
                gray_image = np.empty(image.shape, dtype=np.uint8)
                cv2.convertScaleAbs(image, dst=gray_image)
            else:
                gray_image = image

        # Validate dimensions
        if gray_image is None or gray_image.size == 0:
            return None
        if gray_image.ndim != 2:
            return None
        if gray_image.shape[0] < 8 or gray_image.shape[1] < 8:
            return None

        # Ensure writable C-contiguous uint8 buffer
        gray_image = np.require(
            gray_image, dtype=np.uint8, requirements=["C", "W"]
        )  # type: ignore

        # Check decimated size
        if (
            gray_image.shape[0] / max(self.quad_decimate, 1.0) < 4
            or gray_image.shape[1] / max(self.quad_decimate, 1.0) < 4
        ):
            return None

        return gray_image

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
        next_families = self.families if families is None else families
        next_nthreads = self.nthreads if nthreads is None else nthreads
        next_quad_decimate = (
            self.quad_decimate if quad_decimate is None else quad_decimate
        )
        next_quad_sigma = self.quad_sigma if quad_sigma is None else quad_sigma
        next_refine_edges = self.refine_edges if refine_edges is None else refine_edges
        next_decode_sharpening = (
            self.decode_sharpening
            if decode_sharpening is None
            else decode_sharpening
        )

        try:
            new_detector = self._create_detector(
                next_families,
                next_nthreads,
                next_quad_decimate,
                next_quad_sigma,
                next_refine_edges,
                next_decode_sharpening,
            )
        except Exception as exc:
            logger.exception("Failed to create AprilTag detector with updated parameters")
            raise ValueError(f"Invalid AprilTag detector configuration: {exc}") from exc

        with self._detect_lock:
            self.ready = False
            old_detector = self.detector
            self.families = next_families
            self.nthreads = max(1, int(next_nthreads))
            self.quad_decimate = max(1.0, float(next_quad_decimate))
            self.quad_sigma = max(0.0, float(next_quad_sigma))
            self.refine_edges = int(next_refine_edges)
            self.decode_sharpening = float(next_decode_sharpening)
            self.detector = new_detector
            self.ready = True
            self._disable_native_destructor(old_detector)

    def run_detection(
        self, images: list[tuple[np.ndarray, np.ndarray]] | np.ndarray
    ) -> Optional[list[Detection] | list[CustomDetection]]:
        """Run detection on a single image."""
        # prevents issues with detector settings being changed mid-frame / mid-run
        if not self.ready:
            return None

        if isinstance(images, np.ndarray):
            gray_image = self._preprocess_image(images)
            if gray_image is None:
                return None
            with self._detect_lock:
                try:
                    detections = cast(list[Detection], self.detector.detect(gray_image))
                except Exception as exc:
                    logger.exception("AprilTag detection failed: %s", exc)
                    return None
                return detections
        else:
            detections = []
            for image, offset in images:
                gray_image = self._preprocess_image(image)
                if gray_image is None:
                    continue
                with self._detect_lock:
                    try:
                        detected_tags = self.detector.detect(gray_image)
                    except Exception as exc:
                        logger.exception("AprilTag detection failed: %s", exc)
                        continue
                if isinstance(detected_tags, Detection):
                    detections.append(
                        CustomDetection(
                            tag_id=detected_tags.tag_id,
                            corners=(detected_tags.corners + offset),
                        )
                    )
                elif isinstance(detected_tags, list):
                    for detection in detected_tags:
                        detections.append(
                            CustomDetection(
                                tag_id=detection.tag_id,
                                corners=(detection.corners + offset),
                            )
                        )
            return detections

    def detect(
        self,
        images: list[tuple[np.ndarray, np.ndarray]] | np.ndarray,
        full_frame: Optional[np.ndarray] = None,
    ) -> Optional[list[Detection] | list[CustomDetection]]:
        """Detect AprilTags in an image.
        Note:
        - Input image is always converted to grayscale.

        Args:
            images: Input image / list of images (grayscale or BGR). If list of images, each image is a list of two items (image segment and image segment offset from origonal center) (np.ndarray, np.ndarray).
            full_frame: Optional full frame image for if no tags are detected.
        Returns:
            List of Detection objects containing tag information. If list of images, returns list of CustomDetection objects. Returns None if no detections found and no fallback available.
        """
        detections = self.run_detection(images)
        if detections is None or (len(detections) == 0 and full_frame is not None):
            if full_frame is not None:
                full_frame_detections = self.run_detection(full_frame)
                if full_frame_detections is not None and len(full_frame_detections) > 0:
                    return full_frame_detections
        return detections
