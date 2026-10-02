"""Level-robot XY localization with capture-aligned NWU gyro yaw."""

from typing import Any

import numpy as np

from src.main_operations.definitions.base.base_class import OperationInstance
from src.main_operations.modules.apriltags.utils.fmap_parser import load_fmap_file
from src.utils.camera_utils.camera_config_manager import CameraConfigRegistry
from src.utils.camera_utils.camera_coordinate_transforms import (
    build_robot_from_camera_transform,
)
from src.utils.camera_utils.load_camera_parameters import load_camera_parameters
from src.utils.timing import get_timing, unwrap_timed
from src.webui.web_server import EagleEyeInterface

try:
    from pnp_localization_2d import (  # type: ignore[import-not-found, import-untyped]
        PnpLocalization2D,
    )
except ImportError:
    PnpLocalization2D = None


class PnpCameraLocalization2DDefinition(OperationInstance):
    """Emit field-from-camera EDN pose; robot is level at field Z=0.

    Gyro values are finite NWU yaw radians CCW about field +Z. Camera mounting
    is fetched every solve, so registry calibration edits take effect immediately.
    """

    def __init__(
        self,
        camera_bus_id: str,
        apriltag_map_path: str,
        camera_config_registry: CameraConfigRegistry | None = None,
        web_interface: EagleEyeInterface | None = None,
        refinement_iterations: int = 10,
        gyro_max_gap_ms: float = 100.0,
        gyro_nearest_ms: float = 20.0,
    ) -> None:
        """Load shared calibration/map contracts and set capture alignment limits.

        Args:
            camera_bus_id: Camera identifier in the configuration registry.
            apriltag_map_path: Field map containing global tag corners.
            camera_config_registry: Required source of current camera calibration.
            web_interface: Unused interface accepted for pipeline compatibility.
            refinement_iterations: Maximum pixel-space refinement steps (0 to 100).
            gyro_max_gap_ms: Maximum interpolation bracket width (0 to 10000 ms).
            gyro_nearest_ms: Maximum nearest-sample distance (0 to 10000 ms).

        Raises:
            OSError: A calibration or field map file cannot be read.
            ValueError: Registry or intrinsics are missing, settings are invalid,
                or a calibration or field map file contains invalid JSON.
            TypeError: A numeric setting cannot be converted to a number.
            ImportError: The required native extension has not been built.
        """
        self.uses_timed_inputs = True
        self.camera_bus_id = str(camera_bus_id)
        if camera_config_registry is None:
            raise ValueError("2D PnP requires the current camera config registry")
        self.camera_config_registry = camera_config_registry
        config = camera_config_registry.get_config(self.camera_bus_id)
        if not config.intrinsics_path:
            raise ValueError("2D PnP requires a camera intrinsics calibration file")
        matrix, distortion = load_camera_parameters(config.intrinsics_path)
        if PnpLocalization2D is None:
            raise ImportError(
                "Rust pnp_localization_2d module not available. "
                "Please build the Rust extension first with "
                "uv run python src/rust_implementations/build.py pnp_localization_2d."
            )
        if matrix.shape != (3, 3):
            raise ValueError("camera_matrix must be a 3x3 matrix")
        if distortion.ndim not in (1, 2) or (
            distortion.ndim == 2 and 1 not in distortion.shape
        ):
            raise ValueError("distortion_coefficients must be a vector")
        tags = load_fmap_file(apriltag_map_path)
        corners = []
        for tag in tags.values():
            points = np.asarray(tag.global_corners, dtype=np.float64)
            if points.shape != (4, 3):
                raise ValueError("AprilTag corners must be 4x3 arrays")
            corners.extend(points.reshape(-1).tolist())
        self.native_solver = PnpLocalization2D(
            matrix.reshape(-1).tolist(),
            distortion.reshape(-1).tolist(),
            list(tags),
            corners,
        )
        self.refinement_iterations: int
        self.gyro_max_gap_ms: float
        self.gyro_nearest_ms: float
        self.update_config(
            {
                "refinement_iterations": refinement_iterations,
                "gyro_max_gap_ms": gyro_max_gap_ms,
                "gyro_nearest_ms": gyro_nearest_ms,
            }
        )

    def update_config(self, json_config: dict[str, Any]) -> None:
        """Validate all supplied settings before applying any live changes.

        Args:
            json_config: Refinement or gyro alignment settings to update.

        Raises:
            ValueError: A setting is nonfinite, out of bounds, or fractional
                when an integer refinement count is required.
            TypeError: A setting cannot be converted to a number.
        """
        converted: dict[str, float | int] = {}
        for name, maximum in (
            ("refinement_iterations", 100),
            ("gyro_max_gap_ms", 10000),
            ("gyro_nearest_ms", 10000),
        ):
            if name in json_config:
                value = float(json_config[name])
                if not np.isfinite(value) or not 0 <= value <= maximum:
                    raise ValueError(
                        f"{name} must be finite and between 0 and {maximum}"
                    )
                if name == "refinement_iterations":
                    if value != int(value):
                        raise ValueError("refinement_iterations must be an integer")
                    value = int(value)
                converted[name] = value
        for name, value in converted.items():
            setattr(self, name, value)

    def run(self, input_data: Any) -> dict[str, Any]:
        """Consume {'detections': TimedValue, 'gyro_samples': plain sample list}.

        A sole connected detection input may also arrive as a bare TimedValue.
        Always emit exactly camera_pose, pose_meta, and diagnostics. Failures have
        None pose/meta; no unconstrained solve, pose holding, or smoothing occurs.

        Args:
            input_data: Timed detections and plain gyro history, or bare detections.

        Returns:
            Camera pose, quality metadata, and diagnostics; rejected solves have
            None pose and metadata with an explicit diagnostic reason.
        """
        if not isinstance(input_data, dict):
            # FlowManager sends the sole connected detection input bare.
            input_data = {"detections": input_data}
        detections = input_data.get("detections")
        timing = get_timing(detections)
        if (
            timing is None
            or type(timing.capture_nt_us) is not int
            or not 1 < timing.capture_nt_us <= 2**63 - 1
        ):
            return {
                "camera_pose": None,
                "pose_meta": None,
                "diagnostics": {"reason": "missing_capture"},
            }
        try:
            mount = build_robot_from_camera_transform(
                self.camera_config_registry.get_config(self.camera_bus_id).extrinsics
            )
            mounting_transform = mount.reshape(-1).tolist()
        except (TypeError, ValueError, AttributeError, OverflowError):
            mounting_transform = None
        # Native validation owns rejection precedence, including invalid mounting.
        result = self.native_solver.solve(
            capture_us=timing.capture_nt_us,
            gyro_samples=input_data.get("gyro_samples"),
            detections=unwrap_timed(detections),
            mounting_transform=mounting_transform,
            refinement_iterations=self.refinement_iterations,
            gyro_max_gap_us=round(self.gyro_max_gap_ms * 1000),
            gyro_nearest_us=round(self.gyro_nearest_ms * 1000),
        )
        if result["camera_pose"] is not None:
            result["camera_pose"] = np.asarray(
                result["camera_pose"], dtype=np.float64
            ).reshape(4, 4)
        return result


# The pipeline factory capitalizes each snake-case segment ("2d" -> "2d").
PnpCameraLocalization2dDefinition = PnpCameraLocalization2DDefinition
