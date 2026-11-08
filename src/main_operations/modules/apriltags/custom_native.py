"""pupil_apriltags-compatible adapter for the built Rust AprilTag detector."""

from __future__ import annotations

import numpy as np
from pupil_apriltags import Detection

from .lazy_segment import LazySegment


class CustomNativeDetector:
    """Expose the Rust detector through pupil_apriltags' constructor and results."""

    def __init__(
        self,
        families: str = "tag36h11",
        nthreads: int = 1,
        quad_decimate: float = 2.0,
        quad_sigma: float = 0.0,
        refine_edges: int = 1,
        decode_sharpening: float = 0.25,
    ) -> None:
        """Create the Rust detector.

        Args:
            families: Comma or whitespace separated AprilTag family names.
            nthreads: Worker threads for large images.
            quad_decimate: Decimation for quad detection.
            quad_sigma: Gaussian blur (positive) or sharpening (negative) sigma.
            refine_edges: When 1, quad edges snap to nearby image gradients.
            decode_sharpening: Sharpening applied to payload samples.

        Raises:
            ImportError: The Rust module has not been built.
            ValueError: The Rust detector rejects the configuration.
        """
        try:
            from custom_apriltag_detector import CustomAprilTagDetector
        except ModuleNotFoundError as exc:
            raise ImportError(
                "Custom Rust detector is not built. Run: "
                "uv run python src/rust_implementations/build.py custom_apriltag_detector"
            ) from exc
        self._family_names = [
            name.encode("ascii") for name in families.replace(",", " ").split()
        ]
        self._detector = CustomAprilTagDetector(
            families=families,
            nthreads=nthreads,
            quad_decimate=quad_decimate,
            quad_sigma=quad_sigma,
            refine_edges=refine_edges,
            decode_sharpening=decode_sharpening,
        )

    def detect(
        self,
        gray: np.ndarray,
        *,
        source_map: np.ndarray | None = None,
        source_shape: tuple[int, int] | None = None,
    ) -> list[Detection]:
        """Detect tags, ignoring pixels that map outside the physical source frame.

        Args:
            gray: 2D uint8 grayscale image. Pixels must not change during the call.
            source_map: XY offset or 3x3 transform from image pixel centers to the
                physical source frame. None treats every pixel as observed.
            source_shape: Physical source (height, width); required with source_map.

        Returns:
            Detections in pupil_apriltags' result type.
        """
        mapping = None
        if source_map is not None:
            mapping = np.asarray(source_map, dtype=np.float64)
            if mapping.shape == (2,):
                mapping = np.array(
                    [[1.0, 0.0, mapping[0]], [0.0, 1.0, mapping[1]], [0.0, 0.0, 1.0]]
                )
            mapping = mapping.ravel().tolist()
        if not gray.flags["C_CONTIGUOUS"]:
            gray = np.ascontiguousarray(gray)
        detections = []
        for native in self._detector.run(
            gray, source_map=mapping, source_shape=source_shape
        ):
            detection = Detection()
            detection.tag_family = self._family_names[native.family_index]
            detection.tag_id = native.tag_id
            detection.hamming = native.hamming
            detection.decision_margin = native.decision_margin
            detection.corners = np.array(native.corners).reshape(4, 2)
            detection.center = np.array(native.center)
            detection.homography = np.array(native.homography).reshape(3, 3)
            detections.append(detection)
        return detections

    def detect_segment(self, segment: LazySegment) -> list[tuple[int, np.ndarray]]:
        """Detect tags in a lazy segment, resampling it natively from its frame.

        The segment's crop is never warped in Python: the detector bilinearly samples
        the grayscale frame through the segment's mapping. Pixels mapping outside the
        frame are ignored.

        Args:
            segment: Square segment of a 2D grayscale or packed BGR uint8 frame.

        Returns:
            Tag IDs with shape-(4, 2) full-frame OpenCV pixel-centre corners.
        """
        frame = segment.frame
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)
        found = self._detector.run_warped(
            frame, segment.shape[0], segment.mapping.ravel().tolist()
        )
        return [(tag_id, np.array(corners).reshape(4, 2)) for tag_id, corners in found]

    def track(
        self, gray: np.ndarray, corners: list[float], tag_id: int
    ) -> np.ndarray | None:
        """Confirm a tag at predicted corners and return its grid-fitted corners.

        Args:
            gray: 2D uint8 grayscale image around the prediction.
            corners: Eight predicted values in pupil_apriltags corner order and
                pixel-edge coordinates of ``gray``.
            tag_id: Tag expected at the prediction.

        Returns:
            Shape-(4, 2) pixel-edge corners, or None when the tag is not confirmed.
        """
        native = self._detector.track(gray, corners, tag_id)
        return None if native is None else np.array(native.corners).reshape(4, 2)

    def track_many(
        self, frame: np.ndarray, predictions: list[np.ndarray], tag_ids: list[int]
    ) -> list[np.ndarray | None]:
        """Confirm several tags around predicted full-frame corners in one call.

        Each tag is searched in a grayscale window of ``frame`` that covers its
        white quiet zone and the edge-snap retry; BGR is converted only there.

        Args:
            frame: Full 2D grayscale or packed BGR uint8 frame.
            predictions: Shape-(4, 2) predicted corners per tag, in pupil_apriltags
                order and OpenCV pixel-centre coordinates.
            tag_ids: Tag expected at each prediction.

        Returns:
            Per tag, shape-(4, 2) grid-fitted pixel-centre corners, or None when the
            tag is not confirmed.
        """
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)
        flat = np.asarray(predictions, dtype=np.float64).ravel().tolist()
        return [
            None if corners is None else np.array(corners).reshape(4, 2)
            for corners in self._detector.track_many(frame, flat, tag_ids)
        ]
