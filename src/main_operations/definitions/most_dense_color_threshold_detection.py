import cv2
import numpy as np
from threading import Lock
from typing import List, Dict, Any, Optional

from src.main_operations.modules.object_detection.color_threshold_detection.implementation import (
    ColorThresholdDetectionImplementation,
)
from src.utils.device_management_utils.compute_pool import ComputePool
from src.utils.camera_utils.camera_config_manager import CameraConfigRegistry
from src.utils.camera_utils.load_camera_parameters import load_camera_parameters
from src.main_operations.definitions.base.base_class import OperationInstance
from src.webui.web_server import EagleEyeInterface


class MostDenseColorThresholdDetectionDefinition(OperationInstance):
    """Color-based detection that returns a single region selected by density (contour area).

    Performs the same thresholding pipeline as ColorThresholdDetectionDefinition but
    returns exactly one detection — either the most dense or least dense region —
    rather than all candidates.

    Input: np.ndarray (BGR image)
    Output: List[Dict[str, Any]] with at most one detection

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
        selection_mode: str = "most_dense",
        camera_config_registry: CameraConfigRegistry | None = None,
        web_interface: EagleEyeInterface | None = None,
        compute_pool: ComputePool | None = None,
    ) -> None:
        """Initialize most dense color threshold detection operation.

        Args:
            camera_bus_id: Camera bus ID used to resolve calibration files.
            target_size: Target size for square letterboxed image.
            color_ranges: List of color range dictionaries with format:
                {
                    "name": "red",
                    "class_id": 0,
                    "lower_hsv": [0, 100, 100],
                    "upper_hsv": [10, 255, 255]
                }
            min_area: Minimum contour area to consider as detection.
            max_area: Maximum contour area to consider as detection.
            blur_kernel_size: Gaussian blur kernel size (0 to disable).
            morphology_kernel_size: Kernel size for morphological operations.
            morphology_iterations: Number of morphological operation iterations.
            selection_mode: Either "most_dense" (largest area) or "least_dense" (smallest area).
            camera_config_registry: Injected shared camera config registry.
            web_interface: Injected web interface reference.
            compute_pool: Injected compute pool reference.
        """
        if selection_mode not in ("most_dense", "least_dense"):
            raise ValueError(
                f"selection_mode must be 'most_dense' or 'least_dense', got '{selection_mode}'"
            )

        if blur_kernel_size != 0 and blur_kernel_size % 2 == 0:
            raise ValueError(
                f"blur_kernel_size must be 0 (disabled) or an odd positive integer, got {blur_kernel_size}"
            )
        if morphology_kernel_size != 0 and morphology_kernel_size % 2 == 0:
            raise ValueError(
                f"morphology_kernel_size must be 0 (disabled) or an odd positive integer, got {morphology_kernel_size}"
            )

        self.camera_bus_id = camera_bus_id
        self.camera_config_registry = camera_config_registry
        self.web_interface = web_interface
        self.compute_pool = compute_pool
        self.selection_mode = selection_mode

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

        self.last_detection: Optional[Dict[str, Any]] = None
        self.last_thresholded_frame: Optional[np.ndarray] = None
        self.last_detection_lock: Lock = Lock()
        self.color_map: Dict[str, tuple] = {}

    def _load_camera_parameters(self) -> None:
        """Load camera intrinsics from the camera config registry."""
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

    def run(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Process frame and return the single most or least dense color detection.

        Args:
            frame: Input BGR image.

        Returns:
            Single-element list containing the selected detection, or empty list if
            no detections were found.
        """
        detections, thresholded_frame = self.delegate.run(frame)

        selected: Optional[Dict[str, Any]] = None
        if detections:
            reverse = self.selection_mode == "most_dense"
            selected = sorted(detections, key=lambda d: d["area"], reverse=reverse)[0]

        with self.last_detection_lock:
            self.last_detection = selected
            self.last_thresholded_frame = thresholded_frame

        return [selected] if selected is not None else []

    def update_config(self, json_config: Dict[str, Any]) -> None:
        """Update runtime-configurable parameters without restarting.

        Args:
            json_config: Dictionary of parameter keys and new values.
        """
        if "selection_mode" in json_config:
            new_mode = json_config["selection_mode"]
            if new_mode not in ("most_dense", "least_dense"):
                raise ValueError(
                    f"selection_mode must be 'most_dense' or 'least_dense', got '{new_mode}'"
                )
            self.selection_mode = new_mode

    def visualize(self, frame: np.ndarray) -> np.ndarray:
        """Visualize the selected detection alongside the thresholded mask.

        Args:
            frame: Input frame to draw detections on.

        Returns:
            Split-screen frame with the selected detection on the left and the
            thresholded mask on the right, or the plain frame if no mask is available.
        """
        with self.last_detection_lock:
            detection = self.last_detection
            thresholded_frame = self.last_thresholded_frame

        visualization_frame = frame.copy()
        height, width = frame.shape[:2]

        if detection is not None:
            x1_pct, y1_pct, x2_pct, y2_pct = detection["bbox"]
            color_name = detection["color_name"]
            class_id = detection["class_id"]
            area = detection["area"]

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

            label = f"{color_name} (ID: {class_id}) area:{int(area)}"
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
                f"Selection ({self.selection_mode})",
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
