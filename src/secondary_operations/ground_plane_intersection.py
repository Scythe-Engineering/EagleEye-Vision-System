import numpy as np
from typing import List, Dict, Any, Optional
from threading import Lock
from src.secondary_operations.base_class import SecondaryOperation


class GroundPlaneIntersection(SecondaryOperation):
    """Ground plane intersection for 3D position estimation.

    This operation calculates the 3D intersection points of detection bounding boxes
    with the ground plane using camera pose and calibration parameters.

    Input: List[Dict[str, Any]] with detection information
    Output: List[Dict[str, Any]] with 3D position information
    """

    def __init__(
        self,
        camera_height: float = 1.0,
        camera_pitch: float = 0.0,
        fov_horizontal: float = 60.0,
        fov_vertical: float = 45.0,
        pipeline: Any = None,
    ) -> None:
        """Initialize ground plane intersection operation.

        Args:
            camera_height: Height of camera above ground plane in meters
            camera_pitch: Pitch angle of camera in radians (positive = looking down)
            fov_horizontal: Horizontal field of view in degrees
            fov_vertical: Vertical field of view in degrees
            pipeline: Injected pipeline reference for accessing camera information
        """
        self.camera_height = float(camera_height)
        self.camera_pitch = float(camera_pitch)
        self.fov_horizontal = float(fov_horizontal)
        self.fov_vertical = float(fov_vertical)
        self.pipeline = pipeline

        self.last_detections: Optional[List[Dict[str, Any]]] = None
        self.last_detections_lock: Lock = Lock()

    def run(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process detections for ground plane intersection.

        Args:
            detections: List of detection dictionaries (already undistorted)

        Returns:
            List of detection dictionaries with ground plane intersection information
        """
        output_detections = []

        for detection in detections:
            if not isinstance(detection, dict):
                continue
            bbox = detection.get("bbox")
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            x_center = (x1 + x2) / 2
            y_bottom = max(y1, y2)

            # Compute angles using a pinhole model rather than linear scaling, and include camera pitch
            hfov_rad = np.deg2rad(self.fov_horizontal)
            vfov_rad = np.deg2rad(self.fov_vertical)

            x_norm_centered = np.clip(2.0 * (x_center - 0.5), -1.0, 1.0)
            y_norm_centered = np.clip(2.0 * (y_bottom - 0.5), -1.0, 1.0)

            horizontal_angle_rad = np.arctan(x_norm_centered * np.tan(hfov_rad / 2.0))
            vertical_angle_from_optical_rad = np.arctan(
                y_norm_centered * np.tan(vfov_rad / 2.0)
            )
            total_vertical_angle_rad = (
                vertical_angle_from_optical_rad + self.camera_pitch
            )

            min_vertical_angle_rad = np.deg2rad(3.0)
            if total_vertical_angle_rad <= min_vertical_angle_rad:
                continue

            distance = self.camera_height / np.tan(total_vertical_angle_rad)
            if not np.isfinite(distance) or distance <= 0:
                continue

            x_position = distance * np.cos(horizontal_angle_rad)
            y_position = distance * np.sin(horizontal_angle_rad)

            z_position = 0.0

            position_3d = np.array([y_position, z_position, x_position])

            updated_detection = detection.copy()
            updated_detection["position_3d"] = position_3d.tolist()
            output_detections.append(updated_detection)

        with self.last_detections_lock:
            self.last_detections = output_detections

        return output_detections

    def update_config(self, json_config: dict) -> None:
        """Update configuration parameters.

        Args:
            json_config: Dictionary with parameter names and new values
        """
        if "camera_height" in json_config:
            self.camera_height = float(json_config["camera_height"])
        if "camera_pitch" in json_config:
            self.camera_pitch = float(json_config["camera_pitch"])
        if "fov_horizontal" in json_config:
            self.fov_horizontal = float(json_config["fov_horizontal"])
        if "fov_vertical" in json_config:
            self.fov_vertical = float(json_config["fov_vertical"])
