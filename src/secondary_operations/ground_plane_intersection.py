import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple
from threading import Lock

from src.utils.camera_utils.load_camera_parameters import load_camera_parameters


class GroundPlaneIntersection:
    """Perform ground plane intersection with bounding box detections to estimate 3D pose.

    This operation takes bounding box detections (e.g., from color threshold detection)
    and performs ray-ground plane intersection using the bottom middle point of each
    bounding box to estimate the 3D position of detected objects relative to the camera.

    Input: List[Dict[str, Any]] with detection information containing 'bbox' key
    Output: List[Dict[str, Any]] with detection information plus 'pose' and 'position_3d' keys

    Each input detection should contain:
        - bbox: [x1_pct, y1_pct, x2_pct, y2_pct] as percentages (0-1) of frame dimensions
        - class_id: Integer class identifier
        - Additional fields are preserved in output

    Each output detection contains all input fields plus:
        - pose: 4x4 transformation matrix representing object pose (position only, no rotation)
        - position_3d: [x, y, z] 3D position in camera coordinate system
    """

    def __init__(
        self,
        camera_intrinsics_path: str,
        camera_height: float = 1.0,
        camera_pitch: float = 0.0,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
        pipeline: Any = None,
    ) -> None:
        """Initialize ground plane intersection operation.

        Args:
            camera_intrinsics_path: Path to camera intrinsics JSON file, or camera bus ID
                (e.g., "0", "0-1") to auto-resolve path
            camera_height: Height of camera above ground plane in meters
            camera_pitch: Pitch angle of camera in radians (positive = looking down)
            frame_width: Width of input frames (if None, will try to get from intrinsics)
            frame_height: Height of input frames (if None, will try to get from intrinsics)
            pipeline: Injected pipeline reference for accessing camera information
        """
        self.camera_intrinsics_path = camera_intrinsics_path
        self.camera_height = float(camera_height)
        self.camera_pitch = float(camera_pitch)
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.pipeline = pipeline

        self.camera_matrix: Optional[np.ndarray] = None
        self.distortion_coefficients: Optional[np.ndarray] = None
        self._load_camera_parameters()

        self.last_detections: Optional[List[Dict[str, Any]]] = None
        self.last_detections_lock: Lock = Lock()

    def _load_camera_parameters(self) -> None:
        """Load camera intrinsics from file or resolve from camera bus ID."""
        intrinsics_path = self.camera_intrinsics_path

        if self.pipeline is not None:
            camera_bus_id = getattr(self.pipeline, "camera_bus_id", None)
            if camera_bus_id is not None and not intrinsics_path.endswith(".json"):
                intrinsics_path = f"src/utils/camera_utils/camera_calibrations/{camera_bus_id}/intrinsics.json"

        try:
            self.camera_matrix, self.distortion_coefficients = load_camera_parameters(
                intrinsics_path
            )

            if self.frame_width is None or self.frame_height is None:
                import json
                import os

                if os.path.exists(intrinsics_path):
                    with open(intrinsics_path, "r") as f:
                        data = json.load(f)
                        if "img_size" in data:
                            self.frame_width = data["img_size"][0]
                            self.frame_height = data["img_size"][1]
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
        undistorted = cv2.undistortPoints(
            point_reshaped,
            self.camera_matrix,
            self.distortion_coefficients,
            P=self.camera_matrix,
        )
        return undistorted.reshape(2)

    def _pixel_to_ray(
        self, pixel_x: float, pixel_y: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert pixel coordinates to a 3D ray in camera coordinate system.

        Args:
            pixel_x: X coordinate in image pixels
            pixel_y: Y coordinate in image pixels

        Returns:
            Tuple of (ray_origin, ray_direction) in camera coordinate system
        """
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        x_normalized = (pixel_x - cx) / fx
        y_normalized = (pixel_y - cy) / fy

        ray_direction = np.array([x_normalized, y_normalized, 1.0])
        ray_direction = ray_direction / np.linalg.norm(ray_direction)

        cos_pitch = np.cos(self.camera_pitch)
        sin_pitch = np.sin(self.camera_pitch)

        rotation_matrix = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, cos_pitch, -sin_pitch],
                [0.0, sin_pitch, cos_pitch],
            ]
        )

        ray_direction = rotation_matrix @ ray_direction

        ray_origin = np.array([0.0, 0.0, self.camera_height])

        return ray_origin, ray_direction

    def _ray_plane_intersection(
        self, ray_origin: np.ndarray, ray_direction: np.ndarray, plane_z: float
    ) -> Optional[np.ndarray]:
        """Compute intersection point of ray with horizontal plane.

        Args:
            ray_origin: 3D point on ray
            ray_direction: Normalized 3D direction vector
            plane_z: Z coordinate of horizontal plane

        Returns:
            3D intersection point [x, y, z] or None if no intersection
        """
        if abs(ray_direction[2]) < 1e-6:
            return None

        t = (plane_z - ray_origin[2]) / ray_direction[2]

        if t < 0:
            return None

        intersection_point = ray_origin + t * ray_direction
        return intersection_point

    def run(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process detections and compute ground plane intersections.

        Args:
            detections: List of detection dictionaries with 'bbox' key containing
                [x1_pct, y1_pct, x2_pct, y2_pct] as percentages (0-1)

        Returns:
            List of detection dictionaries with added 'pose' and 'position_3d' keys
        """
        if self.frame_width is None or self.frame_height is None:
            raise ValueError(
                "Frame dimensions not set. Provide frame_width and frame_height or ensure intrinsics file contains img_size"
            )

        output_detections = []

        for detection in detections:
            if "bbox" not in detection:
                continue

            bbox = detection["bbox"]
            x1_pct, y1_pct, x2_pct, y2_pct = bbox

            x1_pixel = x1_pct * self.frame_width
            x2_pixel = x2_pct * self.frame_width
            y1_pixel = y1_pct * self.frame_height
            y2_pixel = y2_pct * self.frame_height

            bottom_middle_x = (x1_pixel + x2_pixel) / 2.0
            bottom_middle_y = max(y1_pixel, y2_pixel)

            ray_origin, ray_direction = self._pixel_to_ray(
                bottom_middle_x, bottom_middle_y
            )

            intersection_point = self._ray_plane_intersection(
                ray_origin, ray_direction, 0.0
            )

            output_detection = detection.copy()

            if intersection_point is not None:
                position_3d = intersection_point.tolist()

                pose_matrix = np.eye(4)
                pose_matrix[0, 3] = position_3d[0]
                pose_matrix[1, 3] = position_3d[1]
                pose_matrix[2, 3] = position_3d[2]

                output_detection["position_3d"] = position_3d
                output_detection["pose"] = pose_matrix.tolist()
            else:
                output_detection["position_3d"] = None
                output_detection["pose"] = None

            output_detections.append(output_detection)

        with self.last_detections_lock:
            self.last_detections = output_detections

        return output_detections

    def visualize(self, frame: np.ndarray) -> np.ndarray:
        """Visualize ground plane intersections.

        Args:
            frame: Input frame (unused)

        Returns:
            Frame (unused, visualized in webui)
        """
        return frame

    def update_config(self, json_config: dict) -> None:
        """Update configuration parameters.

        Args:
            json_config: Dictionary with parameter names and new values
        """
        if "camera_height" in json_config:
            self.camera_height = float(json_config["camera_height"])
        if "camera_pitch" in json_config:
            self.camera_pitch = float(json_config["camera_pitch"])
        if "camera_intrinsics_path" in json_config:
            self.camera_intrinsics_path = json_config["camera_intrinsics_path"]
            self._load_camera_parameters()
