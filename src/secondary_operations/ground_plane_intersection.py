from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

import numpy as np

from src.main_operations.definitions.base.base_class import OperationInstance
from src.utils.camera_utils.camera_config_manager import CameraConfigRegistry


class GroundPlaneIntersection(OperationInstance):
    """Ground plane intersection for 3D position estimation.

    This operation calculates the 3D intersection points of detection bounding boxes
    with the ground plane using camera pose and calibration parameters.

    Input: List[Dict[str, Any]] with detection information
    Output: List[Dict[str, Any]] with 3D position information
    """

    def __init__(
        self,
        camera_bus_id: str | None = None,
        camera_height: float = 1.0,
        camera_pitch: float = 0.0,
        camera_config_registry: CameraConfigRegistry | None = None,
    ) -> None:
        """Initialize ground plane intersection operation.

        Args:
            camera_bus_id: Camera bus ID used to resolve intrinsics and extrinsics.
            camera_height: Legacy fallback height used when extrinsics are unavailable.
            camera_pitch: Legacy fallback pitch used when extrinsics are unavailable.
            camera_config_registry: Injected shared camera config registry.
        """
        self.camera_bus_id = str(camera_bus_id) if camera_bus_id is not None else None
        self._fallback_camera_height = float(camera_height)
        self._fallback_camera_pitch = float(camera_pitch)
        self.camera_config_registry = camera_config_registry
        self._intrinsics_cache: dict[str, Any] | None = None

        self.last_detections: Optional[List[Dict[str, Any]]] = None
        self.last_detections_lock: Lock = Lock()

    def _resolve_camera_bus_id(self) -> str:
        """Resolve the configured camera bus ID.

        Returns:
            Camera bus ID from this operation's configuration.

        Raises:
            ValueError: If no camera bus ID is configured.
        """
        if self.camera_bus_id:
            return self.camera_bus_id

        raise ValueError(
            "camera_bus_id is required; set it in the operation settings to select a camera."
        )

    def _default_intrinsics_path(self, camera_bus_id: str) -> Path:
        """Return the conventional intrinsics path for a camera bus ID."""
        src_path = Path(__file__).resolve().parents[1]
        return (
            src_path
            / "utils"
            / "camera_utils"
            / "camera_calibrations"
            / camera_bus_id
            / "intrinsics.json"
        )

    def _resolve_intrinsics_path(self, camera_bus_id: str) -> Path:
        """Resolve the intrinsics JSON path for a camera bus ID."""
        if self.camera_config_registry is not None:
            camera_config = self.camera_config_registry.get_config(camera_bus_id)
            if camera_config.intrinsics_path:
                return Path(camera_config.intrinsics_path)

        return self._default_intrinsics_path(camera_bus_id)

    @staticmethod
    def _read_intrinsics_image_size(
        intrinsics_data: dict[str, Any], camera_matrix: np.ndarray
    ) -> tuple[float, float]:
        """Read image size metadata from an intrinsics payload.

        Intrinsics files in this project currently use both ``img_size`` and
        ``image_width``/``image_height`` shapes, so support both.
        """
        img_size = intrinsics_data.get("img_size")
        if isinstance(img_size, list) and len(img_size) >= 2:
            return float(img_size[0]), float(img_size[1])

        image_width = intrinsics_data.get("image_width")
        image_height = intrinsics_data.get("image_height")
        if image_width is not None and image_height is not None:
            return float(image_width), float(image_height)

        frame_width = intrinsics_data.get("frame_width")
        frame_height = intrinsics_data.get("frame_height")
        if frame_width is not None and frame_height is not None:
            return float(frame_width), float(frame_height)

        cx = float(camera_matrix[0, 2])
        cy = float(camera_matrix[1, 2])
        if cx > 0.0 and cy > 0.0:
            return cx * 2.0, cy * 2.0

        raise ValueError("Intrinsics file does not include image size metadata")

    def _load_intrinsics(self) -> dict[str, Any]:
        """Load selected camera intrinsics for ray projection."""
        camera_bus_id = self._resolve_camera_bus_id()
        intrinsics_path = self._resolve_intrinsics_path(camera_bus_id)

        if not intrinsics_path.exists():
            raise FileNotFoundError(
                f"Camera intrinsics file not found for bus ID '{camera_bus_id}': "
                f"{intrinsics_path}"
            )

        intrinsics_stat = intrinsics_path.stat()
        cache_key = (
            f"{intrinsics_path}:{intrinsics_stat.st_mtime_ns}:"
            f"{intrinsics_stat.st_size}"
        )

        if (
            self._intrinsics_cache is not None
            and self._intrinsics_cache.get("cache_key") == cache_key
        ):
            return self._intrinsics_cache

        with intrinsics_path.open("r", encoding="utf-8") as handle:
            intrinsics_data = json.load(handle)

        camera_matrix = np.array(intrinsics_data["camera_matrix"], dtype=float)
        if camera_matrix.shape != (3, 3):
            raise ValueError(f"Invalid camera_matrix shape in {intrinsics_path}")

        fx = float(camera_matrix[0, 0])
        fy = float(camera_matrix[1, 1])
        if not np.isfinite(fx) or not np.isfinite(fy) or fx == 0.0 or fy == 0.0:
            raise ValueError(f"Invalid focal length in {intrinsics_path}")

        image_width, image_height = self._read_intrinsics_image_size(
            intrinsics_data, camera_matrix
        )

        self._intrinsics_cache = {
            "cache_key": cache_key,
            "camera_matrix": camera_matrix,
            "image_width": image_width,
            "image_height": image_height,
        }
        return self._intrinsics_cache

    def _load_camera_pose_params(self) -> tuple[float, float]:
        """Load camera height and pitch from selected camera extrinsics.

        Returns:
            Tuple of camera height in meters and pitch in radians. The current
            extrinsics editor stores height as ``z_offset`` and pitch in degrees.
        """
        if self.camera_config_registry is None:
            return self._fallback_camera_height, self._fallback_camera_pitch

        camera_bus_id = self._resolve_camera_bus_id()
        camera_config = self.camera_config_registry.get_config(camera_bus_id)
        extrinsics = camera_config.extrinsics

        camera_height = float(extrinsics.z_offset)
        camera_pitch = float(np.deg2rad(extrinsics.pitch))

        if not np.isfinite(camera_height) or not np.isfinite(camera_pitch):
            return self._fallback_camera_height, self._fallback_camera_pitch

        return camera_height, camera_pitch

    def run(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process detections for ground plane intersection.

        Args:
            detections: List of detection dictionaries (already undistorted)

        Returns:
            List of detection dictionaries with ground plane intersection information
        """
        output_detections = []
        intrinsics = self._load_intrinsics()
        camera_matrix = intrinsics["camera_matrix"]
        image_width = intrinsics["image_width"]
        image_height = intrinsics["image_height"]
        fx = float(camera_matrix[0, 0])
        fy = float(camera_matrix[1, 1])
        cx = float(camera_matrix[0, 2])
        cy = float(camera_matrix[1, 2])
        camera_height, camera_pitch = self._load_camera_pose_params()

        for detection in detections:
            if not isinstance(detection, dict):
                continue
            bbox = detection.get("bbox")
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            x_center = (x1 + x2) / 2
            y_bottom = max(y1, y2)

            x_pixel = np.clip(float(x_center), 0.0, 1.0) * image_width
            y_pixel = np.clip(float(y_bottom), 0.0, 1.0) * image_height

            x_ray = (x_pixel - cx) / fx
            y_ray = (y_pixel - cy) / fy

            horizontal_angle_rad = np.arctan(x_ray)
            vertical_angle_from_optical_rad = np.arctan(y_ray)
            total_vertical_angle_rad = (
                vertical_angle_from_optical_rad + camera_pitch
            )

            min_vertical_angle_rad = np.deg2rad(3.0)
            if total_vertical_angle_rad <= min_vertical_angle_rad:
                continue

            distance = camera_height / np.tan(total_vertical_angle_rad)
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
        if "camera_bus_id" in json_config:
            next_camera_bus_id = str(json_config["camera_bus_id"])
            if next_camera_bus_id != self.camera_bus_id:
                self.camera_bus_id = next_camera_bus_id
                self._intrinsics_cache = None
