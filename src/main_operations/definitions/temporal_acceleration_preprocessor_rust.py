from typing import Any, Dict, List, Tuple
from threading import Lock

import cv2
import numpy as np

# Import the Rust module (built automatically)
try:
    from temporal_acceleration import TemporalAcceleration  # type: ignore
except ImportError:
    TemporalAcceleration = None

from src.main_operations.modules.apriltags.utils.fmap_parser import load_fmap_file
from src.utils.camera_utils.camera_config_manager import CameraConfigRegistry
from src.utils.camera_utils.load_camera_parameters import load_camera_parameters
from src.main_operations.definitions.base.base_class import OperationInstance
from src.webui.web_server import EagleEyeInterface


class TemporalAccelerationPreprocessorRustDefinition(OperationInstance):
    """Definition for temporal acceleration-based ROI generation using Rust implementation.

    This operation consumes back-propagated poses and predicts ROIs for
    accelerating the AprilTag detector in the next run using a high-performance
    Rust implementation. The ROI outputs follow the same format as
    `PositionApriltagPreprocessor.process_frame`.
    """

    def __init__(
        self,
        camera_bus_id: str,
        apriltag_map_path: str,
        padding_factor: float = 0.65,
        max_regions: int = 10,
        min_region_size_px: int = 16,
        max_detection_distance_m: float = 0.0,
        camera_config_registry: CameraConfigRegistry | None = None,
        web_interface: EagleEyeInterface | None = None,
    ) -> None:
        """Initialize the temporal acceleration definition with Rust backend.

        Args:
            camera_bus_id: Camera bus ID used to resolve calibration files.
            apriltag_map_path: Path to fmap apriltag map JSON.
            padding_factor: Fractional padding applied to ROI size.
            max_regions: Maximum number of ROIs to return.
            min_region_size_px: Minimum side length for ROI squares.
            max_detection_distance_m: Maximum 3D distance in meters from the
                camera to the tag center for ROI generation. Tags farther than
                this are skipped. Zero disables the limit.
            camera_config_registry: Injected shared camera config registry.
        """
        if TemporalAcceleration is None:
            raise ImportError(
                "Rust temporal_acceleration module not available. "
                "Please build the Rust extension first."
            )

        self.web_interface = web_interface

        intrinsics_path: str
        if camera_config_registry is not None:
            camera_config = camera_config_registry.get_config(camera_bus_id)
            if camera_config.intrinsics_path is None:
                raise ValueError(
                    f"No intrinsics path found for camera bus ID '{camera_bus_id}'"
                )
            intrinsics_path = camera_config.intrinsics_path
        else:
            intrinsics_path = (
                f"src/utils/camera_utils/camera_calibrations/"
                f"{camera_bus_id}/intrinsics.json"
            )

        camera_matrix, distortion_coefficients = load_camera_parameters(
            intrinsics_path
        )
        apriltag_map = load_fmap_file(apriltag_map_path)

        # Convert data for Rust consumption
        camera_matrix_flat: List[float] = (
            camera_matrix.astype(np.float32).flatten().tolist()
        )
        distortion_coefficients_flat: List[float] = (
            distortion_coefficients.astype(np.float32).flatten().tolist()
        )

        # Build AprilTag geometry buffers
        apriltag_ids: List[int] = []
        apriltag_corners_flat: List[
            float
        ] = []  # 12 floats per tag (4 corners x 3 coords)
        apriltag_centers_flat: List[float] = []  # 3 floats per tag
        for tag_id, tag in apriltag_map.items():
            apriltag_ids.append(int(tag_id))
            # Ensure float32 and correct shape
            corners: np.ndarray = np.asarray(
                tag.global_corners, dtype=np.float32
            ).reshape(4, 3)
            centers: np.ndarray = np.asarray(
                tag.global_center, dtype=np.float32
            ).reshape(3)
            apriltag_corners_flat.extend(corners.flatten().tolist())
            apriltag_centers_flat.extend(centers.flatten().tolist())

        self._rust_impl = TemporalAcceleration(
            camera_matrix=camera_matrix_flat,
            distortion_coefficients=distortion_coefficients_flat,
            apriltag_ids=apriltag_ids,
            apriltag_corners=apriltag_corners_flat,
            apriltag_centers=apriltag_centers_flat,
            padding_factor=padding_factor,
            max_regions=max_regions,
            min_region_size_px=min_region_size_px,
            max_detection_distance_m=max_detection_distance_m,
        )

        self._last_regions: List[Tuple[int, int, int, int]] = []
        self._last_regions_lock: Lock = Lock()

    def run(
        self, input_data: Any
    ) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
        """Generate predicted ROIs for the current frame using Rust implementation.

        Args:
            input_data: Input data - dict with 'frame' and optionally 'camera_pose' keys.

        Returns:
            Tuple of (list of (cropped_image, (offset_x, offset_y)) tuples, original frame).
        """
        if isinstance(input_data, dict):
            frame = input_data.get("frame")
            camera_pose = input_data.get("camera_pose")
        else:
            frame = input_data
            camera_pose = None

        if frame is None:
            raise ValueError("Frame input is required")

        if camera_pose is not None:
            if isinstance(camera_pose, np.ndarray) and camera_pose.shape == (4, 4):
                transform_flat: List[float] = (
                    camera_pose.astype(np.float32).flatten().tolist()
                )
                self._rust_impl.back_propagate_input(transform_flat)
            else:
                raise ValueError(
                    f"Expected 4x4 numpy array for camera_pose, got {type(camera_pose)} "
                    f"with shape {getattr(camera_pose, 'shape', 'N/A')}"
                )

        height, width = frame.shape[:2]
        crop_quads, crop_regions = self._rust_impl.process_frame(width, height)

        cropped_images: List[Tuple[np.ndarray, np.ndarray]] = []
        regions: List[Tuple[int, int, int, int]] = []
        use_perspective_crops = len(crop_quads) == len(crop_regions)
        for index, region in enumerate(crop_regions):
            left, top, right, bottom = region
            left = max(0, int(left))
            top = max(0, int(top))
            right = min(int(width), int(right))
            bottom = min(int(height), int(bottom))
            if right <= left or bottom <= top:
                continue

            if use_perspective_crops:
                crop, full_frame_from_crop = self._perspective_crop(
                    frame, np.asarray(crop_quads[index], dtype=np.float32)
                )
                cropped_images.append((crop, full_frame_from_crop))
            else:
                crop = frame[top:bottom, left:right]
                cropped_images.append((crop, np.array([left, top])))
            regions.append((left, top, right, bottom))

        with self._last_regions_lock:
            self._last_regions = regions

        return (cropped_images, frame)

    @staticmethod
    def _perspective_crop(
        frame: np.ndarray, flattened_quad: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Rectify a projected tag region and return its full-frame mapping."""
        source = flattened_quad.reshape(4, 2)
        edge_lengths = np.linalg.norm(source - np.roll(source, -1, axis=0), axis=1)
        side = min(int(np.ceil(float(edge_lengths.max()))), max(frame.shape[:2]))
        side = max(side, 2)
        destination = np.array(
            [[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]],
            dtype=np.float32,
        )
        crop_from_full_frame = cv2.getPerspectiveTransform(source, destination)
        full_frame_from_crop = cv2.getPerspectiveTransform(destination, source)
        crop = cv2.warpPerspective(frame, crop_from_full_frame, (side, side))
        return crop, full_frame_from_crop

    def update_config(self, json_config: Dict[str, Any]) -> None:
        """Update live configuration for the temporal acceleration.

        Args:
            json_config: Parameters to update. Supported keys:
                - padding_factor
                - max_regions
                - min_region_size_px
                - max_detection_distance_m
        """
        self._rust_impl.update_config(json_config)

    def visualize(self, frame: np.ndarray) -> np.ndarray:
        """Visualize the temporal acceleration outputs by darkening non-predicted areas.

        Args:
            frame: Input frame to process.

        Returns:
            Frame with non-predicted areas darkened.
        """
        with self._last_regions_lock:
            crop_regions = self._last_regions

        visualization_frame = cv2.convertScaleAbs(frame, alpha=0.3, beta=0)
        for region in crop_regions:
            left, top, right, bottom = region
            left = max(0, left)
            top = max(0, top)
            right = min(frame.shape[1], right)
            bottom = min(frame.shape[0], bottom)
            if right > left and bottom > top:
                visualization_frame[top:bottom, left:right] = frame[
                    top:bottom, left:right
                ]

        return visualization_frame
