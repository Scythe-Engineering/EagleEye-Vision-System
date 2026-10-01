"""AprilTag detection through the built Rust operation, with shared ROI handling."""

from src.main_operations.definitions.detect_apriltags import DetectApriltagsDefinition


class DetectApriltagsRustDefinition(DetectApriltagsDefinition):
    """Use Rust detection with the existing outputs, visualization and live settings."""

    def __init__(
        self,
        families: str = "tag36h11",
        nthreads: int = 1,
        quad_decimate: float = 2.0,
        quad_sigma: float = 0.0,
        refine_edges: int = 1,
        decode_sharpening: float = 0.25,
        large_roi_decimate: float = 3.0,
        large_roi_min_px: int = 96,
        full_frame_nthreads: int = 2,
        small_roi_max_px: int = 0,
    ) -> None:
        """Select the PyO3 detector independently of benchmark environment overrides.

        Defaults retain the validated ROI-one/full-frame-two thread selection.
        A zero small-ROI threshold keeps the validated decimation behavior.
        Other parameters have the same meanings as the existing detector operation.
        """
        super().__init__(
            families=families,
            nthreads=nthreads,
            quad_decimate=quad_decimate,
            quad_sigma=quad_sigma,
            refine_edges=refine_edges,
            decode_sharpening=decode_sharpening,
            large_roi_decimate=large_roi_decimate,
            large_roi_min_px=large_roi_min_px,
            full_frame_nthreads=full_frame_nthreads,
            small_roi_max_px=small_roi_max_px,
            backend="rust",
        )
