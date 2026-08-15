from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

import numpy as np

from src.main_operations.definitions.base.base_class import OperationInstance
from src.utils.camera_utils.camera_config_manager import (
    CameraConfigRegistry,
    CameraExtrinsics,
)
from src.utils.camera_utils.camera_coordinate_transforms import (
    build_robot_from_camera_transform,
)


MINIMUM_DOWNWARD_ANGLE_RADIANS = np.deg2rad(3.0)


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
        ground_level: float = 0.0,
        camera_config_registry: CameraConfigRegistry | None = None,
    ) -> None:
        """Initialize ground plane intersection operation.

        Args:
            camera_bus_id: Camera bus ID used to resolve intrinsics and extrinsics.
            camera_height: Legacy fallback height used when extrinsics are unavailable.
            camera_pitch: Legacy fallback pitch in radians, used when extrinsics are unavailable.
            ground_level: Ground-plane height in robot coordinates, in meters.
            camera_config_registry: Injected shared camera config registry.
        """
        self.camera_bus_id = str(camera_bus_id) if camera_bus_id is not None else None
        self._fallback_camera_height = float(camera_height)
        self._fallback_camera_pitch = float(camera_pitch)
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

    def _load_robot_from_camera_transform(self) -> np.ndarray:
        """Load the selected camera mounting transform."""
        if self.camera_config_registry is None:
            extrinsics = CameraExtrinsics(
                pitch=float(np.rad2deg(self._fallback_camera_pitch)),
                z_offset=self._fallback_camera_height,
            )
        else:
            camera_bus_id = self._resolve_camera_bus_id()
            extrinsics = self.camera_config_registry.get_config(
                camera_bus_id
            ).extrinsics

        transform = build_robot_from_camera_transform(extrinsics)
        if not np.all(np.isfinite(transform)):
            raise ValueError("Camera extrinsics must contain finite values.")
        return transform

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
        robot_from_camera = self._load_robot_from_camera_transform()

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
            robot_ray = robot_from_camera[:3, :3] @ camera_ray
            camera_height_from_ground = robot_from_camera[2, 3] - self.ground_level
            horizontal_ray_length = float(np.linalg.norm(robot_ray[:2]))
            downward_angle = np.arctan2(-robot_ray[2], horizontal_ray_length)

            if (
                camera_height_from_ground <= 0.0
                or downward_angle <= MINIMUM_DOWNWARD_ANGLE_RADIANS
            ):
                continue

            ray_scale = -camera_height_from_ground / robot_ray[2]
            if not np.isfinite(ray_scale) or ray_scale <= 0.0:
                continue

            position_3d = camera_ray * ray_scale

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
        if json_config.get("camera_height") is not None:
            self._fallback_camera_height = float(json_config["camera_height"])
        if json_config.get("camera_pitch") is not None:
            # The operation-level fallback contract is radians; camera registry
            # extrinsics remain degree-based and are converted when constructed.
            self._fallback_camera_pitch = float(json_config["camera_pitch"])
        if json_config.get("ground_level") is not None:
            self.ground_level = float(json_config["ground_level"])
