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
        """Initialize the Rust AprilTag detection definition.

        Args:
            families: AprilTag family to detect.
            nthreads: Detector threads for temporal ROIs.
            quad_decimate: Decimation for quad detection; decoding uses full resolution.
            quad_sigma: Gaussian blur standard deviation in pixels for quad detection.
            refine_edges: When non-zero, quad edges snap to nearby image gradients.
            decode_sharpening: Sharpening applied during decoding.
            large_roi_decimate: Decimation for large temporal ROIs; zero disables it.
            large_roi_min_px: Minimum ROI side for large_roi_decimate; zero disables it.
            full_frame_nthreads: Thread count for full-frame searches; zero uses nthreads.
            small_roi_max_px: ROIs smaller than this use decimate one; zero disables it.
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
            rust_backend=True,
        )
