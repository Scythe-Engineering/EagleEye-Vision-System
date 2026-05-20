from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    from robust_2d_solve_pnp import Robust2dSolvePnp  # type: ignore
except ImportError:
    Robust2dSolvePnp = None

from src.main_operations.definitions.base.base_class import OperationInstance
from src.main_operations.modules.apriltags.utils.fmap_parser import load_fmap_file
from src.utils.camera_utils.camera_config_manager import CameraConfigRegistry
from src.utils.camera_utils.load_camera_parameters import load_camera_parameters
from src.utils.device_management_utils.compute_pool import ComputePool
from src.utils.quaternion_utils import euler_to_rotation_matrix
from src.webui.web_server import EagleEyeInterface


class Robust2dSolvePnpDefinition(OperationInstance):
    """Rust-backed 2D AprilTag localization that outputs camera pose matrices.

    Input: AprilTag detections, or a dict with ``detections`` and optional
    ``gyro_yaw``. Output: 4x4 ``np.ndarray`` camera pose in field coordinates,
    or ``None`` when no stable pose can be estimated.
    """

    def __init__(
        self,
        camera_bus_id: str,
        apriltag_map_path: str,
        jump_threshold: float = 2.0,
        gyro_prior_weight: float = 1_000_000.0,
        gyro_yaw_units: str = "radians",
        max_iterations: int = 20,
        camera_config_registry: CameraConfigRegistry | None = None,
        web_interface: EagleEyeInterface | None = None,
        compute_pool: ComputePool | None = None,
    ) -> None:
        """Initialize the robust 2D solvePnP operation.

        Args:
            camera_bus_id: Camera bus ID used to resolve calibration metadata.
            apriltag_map_path: Path to the AprilTag fmap file.
            jump_threshold: Maximum camera pose jump before resetting the seed.
            gyro_prior_weight: Positive values hard-constrain yaw to gyro input.
            gyro_yaw_units: Unit for optional gyro yaw values, radians or degrees.
            max_iterations: Maximum Levenberg-Marquardt iterations per frame.
            camera_config_registry: Injected shared camera config registry.
            web_interface: Injected WebUI interface.
            compute_pool: Injected compute pool.
        """
        if Robust2dSolvePnp is None:
            raise ImportError(
                "Rust robust_2d_solve_pnp module not available. "
                "Please build the Rust extension first."
            )

        self.web_interface = web_interface
        self.compute_pool = compute_pool
        self.gyro_yaw_units = gyro_yaw_units

        intrinsics_path, robot_from_camera = self._load_camera_metadata(
            camera_bus_id,
            camera_config_registry,
        )
        camera_matrix, distortion_coefficients = load_camera_parameters(intrinsics_path)
        apriltag_map = load_fmap_file(apriltag_map_path)

        apriltag_ids: list[int] = []
        apriltag_corners_flat: list[float] = []
        for tag_id, apriltag in apriltag_map.items():
            apriltag_ids.append(int(tag_id))
            apriltag_corners = np.asarray(apriltag.global_corners, dtype=np.float64)
            apriltag_corners_flat.extend(apriltag_corners.reshape(4, 3).flatten().tolist())

        self._rust_impl = Robust2dSolvePnp(
            camera_matrix=np.asarray(camera_matrix, dtype=np.float64).flatten().tolist(),
            distortion_coefficients=np.asarray(
                distortion_coefficients,
                dtype=np.float64,
            )
            .flatten()
            .tolist(),
            robot_from_camera=robot_from_camera.astype(np.float64).flatten().tolist(),
            apriltag_ids=apriltag_ids,
            apriltag_corners=apriltag_corners_flat,
            jump_threshold=float(jump_threshold),
            gyro_prior_weight=float(gyro_prior_weight),
            max_iterations=int(max_iterations),
        )

    @staticmethod
    def _load_camera_metadata(
        camera_bus_id: str,
        camera_config_registry: CameraConfigRegistry | None,
    ) -> tuple[str, np.ndarray]:
        """Load camera intrinsics path and robot-from-camera extrinsics.

        Args:
            camera_bus_id: Camera bus ID used to resolve calibration metadata.
            camera_config_registry: Shared camera configuration registry.

        Returns:
            Intrinsics path and 4x4 robot-from-camera transform.
        """
        if camera_config_registry is None:
            intrinsics_path = str(
                Path(__file__).resolve().parents[2]
                / "utils"
                / "camera_utils"
                / "camera_calibrations"
                / camera_bus_id
                / "intrinsics.json"
            )
            return intrinsics_path, np.eye(4, dtype=float)

        camera_config = camera_config_registry.get_config(camera_bus_id)
        if camera_config.intrinsics_path is None:
            raise ValueError(f"No intrinsics path found for camera bus ID '{camera_bus_id}'")

        robot_from_camera = Robust2dSolvePnpDefinition._robot_from_camera_transform(
            camera_config.extrinsics
        )
        return camera_config.intrinsics_path, robot_from_camera

    @staticmethod
    def _robot_from_camera_transform(extrinsics: Any) -> np.ndarray:
        """Build a robot-from-camera transform from camera extrinsics.

        Args:
            extrinsics: Camera extrinsics record from the registry.

        Returns:
            4x4 transform mapping camera-frame points into robot frame.
        """
        transform = euler_to_rotation_matrix(
            pitch=float(-extrinsics.yaw),
            yaw=float(-extrinsics.pitch),
            roll=float(extrinsics.roll),
        )
        transform[:3, 3] = np.array(
            [
                float(extrinsics.y_offset),
                float(-extrinsics.z_offset),
                float(extrinsics.x_offset),
            ],
            dtype=float,
        )
        return transform

    def run(self, input_data: Any) -> np.ndarray | None:
        """Estimate camera pose from AprilTag detections and optional gyro yaw.

        Args:
            input_data: Detection list, or dict with ``detections`` and optional
                ``gyro_yaw`` from ``GetNetworktablesValue``.

        Returns:
            4x4 camera pose in field coordinates, or ``None`` if solving fails.
        """
        detections, gyro_yaw = self._split_inputs(input_data)
        if not detections:
            return None

        tag_ids: list[int] = []
        image_corners: list[float] = []
        decision_margins: list[float] = []

        for detection in detections:
            corners = np.asarray(getattr(detection, "corners", None), dtype=np.float64)
            if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
                continue
            tag_ids.append(int(detection.tag_id))
            image_corners.extend(corners.flatten().tolist())
            decision_margins.append(float(getattr(detection, "decision_margin", 50.0)))

        if not tag_ids:
            return None

        pose_flat = self._rust_impl.estimate_pose(
            tag_ids,
            image_corners,
            decision_margins,
            self._normalize_gyro_yaw(gyro_yaw),
        )
        if pose_flat is None:
            return None

        pose = np.asarray(pose_flat, dtype=np.float32).reshape(4, 4)
        if not np.all(np.isfinite(pose)):
            return None
        return pose

    @staticmethod
    def _split_inputs(input_data: Any) -> tuple[Any, Any]:
        """Split pipeline input into detections and optional gyro yaw.

        Args:
            input_data: Pipeline input object.

        Returns:
            Detection input and gyro yaw input.
        """
        if isinstance(input_data, dict):
            return input_data.get("detections"), input_data.get("gyro_yaw")
        return input_data, None

    def _normalize_gyro_yaw(self, gyro_yaw: Any) -> float | None:
        """Convert optional gyro yaw into radians.

        Args:
            gyro_yaw: Raw gyro yaw value from the pipeline.

        Returns:
            Finite yaw in radians, or ``None`` if unavailable.
        """
        if gyro_yaw is None:
            return None

        gyro_yaw_float = float(gyro_yaw)
        if not np.isfinite(gyro_yaw_float):
            return None
        if self.gyro_yaw_units == "degrees":
            return float(np.deg2rad(gyro_yaw_float))
        return gyro_yaw_float

    def update_config(self, json_config: dict[str, Any]) -> None:
        """Update runtime-tunable solver parameters.

        Args:
            json_config: Updated operation configuration values.
        """
        if "gyro_yaw_units" in json_config:
            self.gyro_yaw_units = str(json_config["gyro_yaw_units"])
        self._rust_impl.update_config(json_config)
