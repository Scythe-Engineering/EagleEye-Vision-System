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
        self._last_search_regions: list[np.ndarray] = []

        # The camera pipeline is continuous, so frame shape/dtype/channel layout is
        # expected to stay stable. Cache the chosen preprocessing path and reusable
        # grayscale buffers to avoid repeated introspection/allocation every frame.
        self._preprocess_signature: Optional[
            tuple[tuple[int, ...], np.dtype, bool]
        ] = None
        self._preprocess_mode: Optional[str] = None
        self._gray_buffer: Optional[np.ndarray] = None
        self._segment_gray_buffers: dict[tuple[int, int], np.ndarray] = {}
        self._min_input_dimension = int(np.ceil(4 * max(1.0, float(self.quad_decimate))))

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

    def _get_gray_buffer(self, shape: tuple[int, int]) -> np.ndarray:
        """Return a reusable grayscale conversion buffer for a frame shape."""
        if self._gray_buffer is None or self._gray_buffer.shape != shape:
            self._gray_buffer = np.empty(shape, dtype=np.uint8)
        return self._gray_buffer

    def _get_segment_gray_buffer(self, shape: tuple[int, int]) -> np.ndarray:
        """Return a reusable grayscale conversion buffer for a segment shape."""
        buffer = self._segment_gray_buffers.get(shape)
        if buffer is None:
            buffer = np.empty(shape, dtype=np.uint8)
            self._segment_gray_buffers[shape] = buffer
        return buffer

    def _is_valid_image(self, image: np.ndarray) -> bool:
        """Return whether an image is large enough for AprilTag processing."""
        return (
            image is not None
            and image.size != 0
            and image.ndim >= 2
            and image.shape[0] >= self._min_input_dimension
            and image.shape[1] >= self._min_input_dimension
        )

    def _preprocess_image(
        self, image: np.ndarray, *, segment: bool = False
    ) -> Optional[np.ndarray]:
        """Preprocess image to grayscale uint8 format.

        The first valid frame establishes the common preprocessing path. If later
        frames match that signature, the hot path skips repeated shape/dtype/color
        checks and reuses conversion buffers.
        """
        if not self._is_valid_image(image):
            return None

        shape = image.shape

        c_contiguous = image.flags.c_contiguous
        signature = (shape, image.dtype, c_contiguous)
        mode = self._preprocess_mode if signature == self._preprocess_signature else None

        if mode is None:
            if len(shape) == 3:
                mode = "bgr"
            elif len(shape) == 2 and image.dtype == np.uint8 and c_contiguous and image.flags.writeable:
                mode = "gray_passthrough"
            elif len(shape) == 2:
                mode = "gray_convert" if image.dtype != np.uint8 else "gray_require"
            else:
                return None
            self._preprocess_signature = signature
            self._preprocess_mode = mode

        if mode == "bgr":
            gray_image = (
                self._get_segment_gray_buffer(shape[:2])
                if segment
                else self._get_gray_buffer(shape[:2])
            )
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY, dst=gray_image)
            return gray_image

        if mode == "gray_passthrough":
            return image

        if mode == "gray_convert":
            gray_image = (
                self._get_segment_gray_buffer(shape[:2])
                if segment
                else self._get_gray_buffer(shape[:2])
            )
            cv2.convertScaleAbs(image, dst=gray_image)
            return gray_image

        return np.require(image, dtype=np.uint8, requirements=["C", "W"])  # type: ignore

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
            self._min_input_dimension = int(np.ceil(4 * self.quad_decimate))
            self._preprocess_signature = None
            self._preprocess_mode = None
            self._gray_buffer = None
            self._segment_gray_buffers.clear()
            self._disable_native_destructor(old_detector)

    @staticmethod
    def _map_segment_corners(
        corners: np.ndarray, full_frame_mapping: np.ndarray
    ) -> np.ndarray:
        """Map detected segment corners into full-frame image coordinates.

        Args:
            corners: Detected corner coordinates within an image segment.
            full_frame_mapping: Shape-(2,) XY offset or shape-(3, 3) perspective
                transform from segment coordinates to full-frame coordinates.

        Returns:
            Detected corners in full-frame image coordinates.
        """
        mapping = np.asarray(full_frame_mapping)
        if mapping.shape == (2,):
            return corners + mapping
        if mapping.shape == (3, 3):
            points = corners.astype(np.float32, copy=False).reshape(-1, 1, 2)
            return cv2.perspectiveTransform(points, mapping).reshape(-1, 2)
        raise ValueError(
            f"Expected a 2D offset or 3x3 perspective transform, got {mapping.shape}"
        )

    @staticmethod
    def _segment_search_region(
        image: np.ndarray, full_frame_mapping: np.ndarray
    ) -> np.ndarray:
        """Return an image segment's oriented boundary in full-frame coordinates.

        Args:
            image: Image segment sent to the detector.
            full_frame_mapping: Segment offset or perspective transform.

        Returns:
            Four perimeter-ordered full-frame corners.
        """
        height, width = image.shape[:2]
        corners = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )
        return AprilTagDetector._map_segment_corners(corners, full_frame_mapping)

    def get_last_search_regions(self) -> list[np.ndarray]:
        """Return copies of the full-frame regions searched by the last detection."""
        with self._detect_lock:
            return [region.copy() for region in self._last_search_regions]

    def run_detection(
        self, images: list[tuple[np.ndarray, np.ndarray]] | np.ndarray
    ) -> Optional[list[Detection] | list[CustomDetection]]:
        """Run detection on a single image or a list of mapped image segments."""
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
            with self._detect_lock:
                for image, full_frame_mapping in images:
                    gray_image = self._preprocess_image(image, segment=True)
                    if gray_image is None:
                        continue
                    try:
                        detected_tags = self.detector.detect(gray_image)
                    except Exception as exc:
                        logger.exception("AprilTag detection failed: %s", exc)
                        continue
                    if isinstance(detected_tags, Detection):
                        detections.append(
                            CustomDetection(
                                tag_id=detected_tags.tag_id,
                                corners=self._map_segment_corners(
                                    detected_tags.corners, full_frame_mapping
                                ),
                            )
                        )
                    elif isinstance(detected_tags, list):
                        for detection in detected_tags:
                            detections.append(
                                CustomDetection(
                                    tag_id=detection.tag_id,
                                    corners=self._map_segment_corners(
                                        detection.corners, full_frame_mapping
                                    ),
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
            images: Input image or image segments paired with either an XY offset
                or a 3x3 transform from segment coordinates to the full frame.
            full_frame: Optional fallback used when the first search finds no tags.

        Returns:
            Detected tags from the supplied image or regions. If no tag is found,
            the full frame is searched once before returning no detections.
        """
        temporal_segments = not isinstance(images, np.ndarray)
        search_regions = (
            [
                self._segment_search_region(image, mapping)
                for image, mapping in images
                if self._is_valid_image(image)
            ]
            if temporal_segments
            else []
        )

        detections = self.run_detection(images)
        if (
            full_frame is not None
            and self._is_valid_image(full_frame)
            and (detections is None or len(detections) == 0)
        ):
            full_frame_detections = self.run_detection(full_frame)
            if temporal_segments:
                search_regions = [
                    np.array(
                        [
                            [0, 0],
                            [full_frame.shape[1] - 1, 0],
                            [full_frame.shape[1] - 1, full_frame.shape[0] - 1],
                            [0, full_frame.shape[0] - 1],
                        ],
                        dtype=np.float32,
                    )
                ]
            if full_frame_detections:
                detections = full_frame_detections

        with self._detect_lock:
            self._last_search_regions = search_regions
        return detections
