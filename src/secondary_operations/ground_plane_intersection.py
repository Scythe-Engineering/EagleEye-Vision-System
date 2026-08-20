from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

import numpy as np

from src.main_operations.definitions.base.base_class import OperationInstance
from src.utils.camera_utils.camera_config_manager import CameraConfigRegistry


MINIMUM_DOWNWARD_ANGLE_RADIANS = np.deg2rad(3.0)


class GroundPlaneIntersection(OperationInstance):
    """Ground plane intersection for 3D position estimation.

    This operation projects detection bounding boxes through calibrated camera
    intrinsics, then intersects those rays with the field ground plane using the
    supplied field-from-camera pose.

    Input: Detection dictionaries and a 4x4 field-from-camera pose
    Output: Detection dictionaries with field-relative 3D positions
    """

    def __init__(
        self,
        camera_bus_id: str | None = None,
        ground_level: float = 0.0,
        camera_config_registry: CameraConfigRegistry | None = None,
    ) -> None:
        """Initialize ground plane intersection operation.

        Args:
            camera_bus_id: Camera bus ID used to resolve intrinsics.
            ground_level: Ground-plane height in field coordinates, in meters.
            camera_config_registry: Injected shared camera config registry.
        """
        self.camera_bus_id = str(camera_bus_id) if camera_bus_id is not None else None
        self.ground_level = float(ground_level)
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
            f"{intrinsics_path}:{intrinsics_stat.st_mtime_ns}:{intrinsics_stat.st_size}"
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

    def run(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Intersect detection rays with the field ground plane.

        Args:
            input_data: Dictionary containing undistorted ``detections`` and a
                4x4 ``camera_pose`` mapping camera coordinates into field coordinates.

        Returns:
            Detection dictionaries with field-relative ``position_3d`` values.
        """
        detections = input_data.get("detections")
        camera_pose = input_data.get("camera_pose")
        if detections is None or camera_pose is None:
            return []

        field_from_camera = np.asarray(camera_pose, dtype=float)
        if field_from_camera.shape != (4, 4) or not np.all(
            np.isfinite(field_from_camera)
        ):
            raise ValueError("Camera pose must be a finite 4x4 matrix.")

        intrinsics = self._load_intrinsics()
        camera_matrix = intrinsics["camera_matrix"]
        image_width = intrinsics["image_width"]
        image_height = intrinsics["image_height"]
        fx = float(camera_matrix[0, 0])
        fy = float(camera_matrix[1, 1])
        cx = float(camera_matrix[0, 2])
        cy = float(camera_matrix[1, 2])
        camera_origin = field_from_camera[:3, 3]
        camera_height_from_ground = camera_origin[2] - self.ground_level
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

            x_pixel = np.clip(float(x_center), 0.0, 1.0) * image_width
            y_pixel = np.clip(float(y_bottom), 0.0, 1.0) * image_height

            camera_ray = np.array(
                [(x_pixel - cx) / fx, (y_pixel - cy) / fy, 1.0],
                dtype=float,
            )
            field_ray = field_from_camera[:3, :3] @ camera_ray
            horizontal_ray_length = float(np.linalg.norm(field_ray[:2]))
            downward_angle = np.arctan2(-field_ray[2], horizontal_ray_length)

            if (
                camera_height_from_ground <= 0.0
                or downward_angle <= MINIMUM_DOWNWARD_ANGLE_RADIANS
            ):
                continue

            ray_scale = -camera_height_from_ground / field_ray[2]
            if not np.isfinite(ray_scale) or ray_scale <= 0.0:
                continue

            position_3d = camera_origin + field_ray * ray_scale

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
        if json_config.get("ground_level") is not None:
            self.ground_level = float(json_config["ground_level"])
