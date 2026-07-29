import cv2
import numpy as np
from threading import Lock
from typing import List, Dict, Any, Optional

from src.main_operations.modules.object_detection.color_threshold_detection.implementation import (
    ColorThresholdDetectionImplementation,
)
from src.utils.camera_utils.camera_config_manager import CameraConfigRegistry
from src.utils.camera_utils.load_camera_parameters import load_camera_parameters
from src.main_operations.definitions.base.base_class import OperationInstance
from src.webui.web_server import EagleEyeInterface


class ColorThresholdDetectionDefinition(OperationInstance):
    """Color-based object detection with preprocessing and multi-object support.

    This operation performs:
    1. Downscaling and letterboxing to 320x320 (configurable)
    2. Color thresholding in HSV space for multiple color ranges
    3. Contour extraction and bounding box calculation
    4. Returns bounding boxes as 0-1 percentages of resized content area

    Input: np.ndarray (BGR image)
    Output: List[Dict[str, Any]] with detection information

    Each detection contains:
        - bbox: [x1, y1, x2, y2] as percentages (0-1) of resized content area dimensions
        - class_id: Integer class identifier for the color
        - color_name: String name of detected color
        - area: Contour area in letterboxed coordinates
    """

    def __init__(
        self,
        camera_bus_id: str,
        target_size: int = 320,
        color_ranges: List[Dict[str, Any]] | None = None,
        min_area: int = 100,
        max_area: int = 50000,
        blur_kernel_size: int = 5,
        morphology_kernel_size: int = 5,
        morphology_iterations: int = 2,
        camera_config_registry: CameraConfigRegistry | None = None,
        web_interface: EagleEyeInterface | None = None,
    ) -> None:
        """Initialize color threshold detection operation.

        Args:
            camera_bus_id: Camera bus ID used to resolve calibration files.
            target_size: Target size for square letterboxed image
            color_ranges: List of color range dictionaries with format:
                {
                    "name": "red",
                    "class_id": 0,
                    "lower_hsv": [0, 100, 100],
                    "upper_hsv": [10, 255, 255]
                }
                HSV values: H (0-179), S (0-255), V (0-255)
                If None, defaults to single red color range
            min_area: Minimum contour area to consider as detection
            max_area: Maximum contour area to consider as detection
            blur_kernel_size: Gaussian blur kernel size (0 to disable)
            morphology_kernel_size: Kernel size for morphological operations
            morphology_iterations: Number of morphological operation iterations
            camera_config_registry: Injected shared camera config registry.
        """
        self.camera_bus_id = camera_bus_id
        self.camera_config_registry = camera_config_registry
        self.web_interface = web_interface

        self.camera_matrix: Optional[np.ndarray] = None
        self.distortion_coefficients: Optional[np.ndarray] = None
        self._load_camera_parameters()

        if color_ranges is None:
            raise ValueError("Color ranges are required")
        if self.camera_matrix is None:
            raise ValueError("Camera matrix is required")
        if self.distortion_coefficients is None:
            raise ValueError("Distortion coefficients are required")
        self.delegate = ColorThresholdDetectionImplementation(
            target_size=target_size,
            color_ranges=color_ranges,
            min_area=min_area,
            max_area=max_area,
            blur_kernel_size=blur_kernel_size,
            morphology_kernel_size=morphology_kernel_size,
            morphology_iterations=morphology_iterations,
            camera_matrix=self.camera_matrix,
            distortion_coefficients=self.distortion_coefficients,
        )

        self.last_detections: Optional[List[Dict[str, Any]]] = None
        self.last_thresholded_frame: Optional[np.ndarray] = None
        self.last_detections_lock: Lock = Lock()
        self.color_map: Dict[str, tuple] = {}

    def _load_camera_parameters(self) -> None:
        """Load camera intrinsics from the camera config registry."""
        intrinsics_path: str
        if self.camera_config_registry is not None:
            camera_config = self.camera_config_registry.get_config(self.camera_bus_id)
            if camera_config.intrinsics_path is None:
                raise ValueError(
                    f"No intrinsics path found for camera bus ID '{self.camera_bus_id}'"
                )
            intrinsics_path = camera_config.intrinsics_path
        else:
            intrinsics_path = (
                f"src/utils/camera_utils/camera_calibrations/"
                f"{self.camera_bus_id}/intrinsics.json"
            )

        try:
            self.camera_matrix, self.distortion_coefficients = load_camera_parameters(
                intrinsics_path
            )
        except Exception as e:
            raise ValueError(
                f"Failed to load camera parameters from {intrinsics_path}: {e}"
            )

        if self.camera_matrix is None:
            raise ValueError("Camera matrix not loaded")

    def _undistort_point(self, point: np.ndarray) -> np.ndarray:
        """Undistort a single 2D point using camera distortion coefficients.

        Args:
            point: [x, y] point in image coordinates

        Returns:
            Undistorted [x, y] point
        """
        if self.distortion_coefficients is None:
            return point

        point_reshaped = point.reshape(1, 1, 2).astype(np.float32)
        undistorted = cv2.undistortPoints(  # type: ignore
            point_reshaped,
            self.camera_matrix,
            self.distortion_coefficients,
            P=self.camera_matrix,
        )
        return undistorted.reshape(2)

    def run(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Process frame and detect colored objects.

        Args:
            frame: Input BGR image

        Returns:
            List of detection dictionaries
        """
        detections, thresholded_frame = self.delegate.run(frame)

        with self.last_detections_lock:
            self.last_detections = detections
            self.last_thresholded_frame = thresholded_frame

        return detections

    def visualize(self, frame: np.ndarray) -> np.ndarray:
        """Visualize color threshold detections with split screen showing detections and thresholded mask.

        Args:
            frame: Input frame to draw detections on

        Returns:
            Split screen frame with detections on left and thresholded mask on right
        """
        with self.last_detections_lock:
            detections = self.last_detections
            thresholded_frame = self.last_thresholded_frame

        visualization_frame = frame.copy()
        height, width = frame.shape[:2]

        if detections is not None and len(detections) > 0:
            for detection in detections:
                x1_pct, y1_pct, x2_pct, y2_pct = detection["bbox"]
                color_name = detection["color_name"]
                class_id = detection["class_id"]

                # Convert percentages to pixel coordinates
                x1 = int(x1_pct * width)
                y1 = int(y1_pct * height)
                x2 = int(x2_pct * width)
                y2 = int(y2_pct * height)

                if color_name not in self.color_map:
                    color_name_to_bgr = {
                        "red": (0, 0, 255),
                        "blue": (255, 0, 0),
                        "green": (0, 255, 0),
                        "yellow": (0, 255, 255),
                        "orange": (0, 165, 255),
                        "purple": (255, 0, 255),
                        "cyan": (255, 255, 0),
                        "pink": (203, 192, 255),
                        "white": (255, 255, 255),
                        "black": (0, 0, 0),
                    }
                    self.color_map[color_name] = color_name_to_bgr.get(
                        color_name.lower(), (255, 255, 255)
                    )

                color = self.color_map[color_name]

                cv2.rectangle(visualization_frame, (x1, y1), (x2, y2), color, 3)

                label = f"{color_name} (ID: {class_id})"

                font_scale = 0.6
                font_thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                )

                text_x = x1
                text_y = max(y1 - 10, text_height + baseline + 5)

                cv2.rectangle(
                    visualization_frame,
                    (text_x - 5, text_y - text_height - baseline - 5),
                    (text_x + text_width + 5, text_y + baseline + 5),
                    color,
                    -1,
                )

                cv2.putText(
                    visualization_frame,
                    label,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 0, 0),
                    font_thickness,
                )

        if thresholded_frame is not None:
            thresholded_resized = cv2.resize(
                thresholded_frame, (width, height), interpolation=cv2.INTER_LINEAR
            )

            split_frame = np.hstack([visualization_frame, thresholded_resized])

            divider_x = width
            cv2.line(
                split_frame, (divider_x, 0), (divider_x, height), (255, 255, 255), 2
            )

            font_scale = 0.8
            font_thickness = 2
            cv2.putText(
                split_frame,
                "Detections",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                font_thickness,
            )
            cv2.putText(
                split_frame,
                "Threshold Mask",
                (width + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                font_thickness,
            )

            return split_frame

        return visualization_frame
