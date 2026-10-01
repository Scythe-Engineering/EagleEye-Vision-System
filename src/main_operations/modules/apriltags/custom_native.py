"""pupil_apriltags-compatible adapter for the built Rust AprilTag detector."""

from __future__ import annotations

import numpy as np
from pupil_apriltags import Detection


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
