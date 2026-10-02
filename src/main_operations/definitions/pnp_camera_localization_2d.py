"""Level-robot XY localization with capture-aligned NWU gyro yaw."""

from typing import Any

import cv2
import numpy as np

from src.main_operations.definitions.base.base_class import OperationInstance
from src.main_operations.modules.apriltags.pnp_localization import PnpLocalization
from src.main_operations.modules.apriltags.utils.fmap_parser import load_fmap_file
from src.utils.camera_utils.camera_config_manager import CameraConfigRegistry
from src.utils.camera_utils.camera_coordinate_transforms import (
    build_robot_from_camera_transform,
)
from src.utils.camera_utils.load_camera_parameters import load_camera_parameters
from src.utils.timestamped_samples import align_heading
from src.utils.timing import get_timing, unwrap_timed
from src.webui.web_server import EagleEyeInterface


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
        """Load shared calibration/map contracts and set capture alignment limits."""
        self.uses_timed_inputs = True
        self.camera_bus_id = str(camera_bus_id)
        if camera_config_registry is None:
            raise ValueError("2D PnP requires the current camera config registry")
        self.camera_config_registry = camera_config_registry
        config = camera_config_registry.get_config(self.camera_bus_id)
        if not config.intrinsics_path:
            raise ValueError("2D PnP requires a camera intrinsics calibration file")
        matrix, distortion = load_camera_parameters(config.intrinsics_path)
        self.pose_estimator = PnpLocalization(
            matrix, distortion, load_fmap_file(apriltag_map_path)
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
        """Validate and apply bounded refinement/alignment settings live."""
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
                setattr(self, name, value)

    def run(self, input_data: Any) -> dict[str, Any]:
        """Consume {'detections': TimedValue, 'gyro_samples': plain sample list}.

        A sole connected detection input may also arrive as a bare TimedValue.
        Always emit exactly camera_pose, pose_meta, and diagnostics. Failures have
        None pose/meta; no unconstrained solve, pose holding, or smoothing occurs.
        """
        diagnostics: dict[str, Any] = {}

        def failure(reason: str) -> dict[str, Any]:
            """Emit explicit failure without a held or unconstrained pose."""
            return {
                "camera_pose": None,
                "pose_meta": None,
                "diagnostics": {**diagnostics, "reason": reason},
            }

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
            return failure("missing_capture")
        diagnostics["capture_nt_us"] = timing.capture_nt_us
        try:
            yaw, alignment = align_heading(
                input_data.get("gyro_samples"),
                timing.capture_nt_us,
                round(self.gyro_max_gap_ms * 1000),
                round(self.gyro_nearest_ms * 1000),
            )
        except ValueError as exc:
            return failure(str(exc))
        diagnostics.update(alignment, yaw_rad=yaw)
        estimator = self.pose_estimator
        objects, images, seen = [], [], set()
        raw = unwrap_timed(detections)
        if not isinstance(raw, (list, tuple)):
            return failure("invalid_detections")
        for detection in raw:
            tag_id = getattr(detection, "tag_id", None)
            if tag_id not in estimator.apriltag_map or tag_id in seen:
                continue
            try:
                image = np.asarray(detection.corners, dtype=float)
                obj = np.asarray(
                    estimator.apriltag_map[tag_id].global_corners, dtype=float
                )
            except (TypeError, ValueError, AttributeError):
                return failure("invalid_points")
            if (
                image.shape != (4, 2)
                or obj.shape != (4, 3)
                or not np.isfinite(image).all()
                or not np.isfinite(obj).all()
            ):
                return failure("invalid_points")
            images.append(image)
            objects.append(obj)
            seen.add(tag_id)
        if not seen:
            return failure("no_mapped_tags")
        image, obj = np.vstack(images), np.vstack(objects)
        if (
            np.linalg.matrix_rank(image - image.mean(axis=0)) < 2
            or np.linalg.matrix_rank(obj - obj.mean(axis=0)) < 2
        ):
            return failure("degenerate_geometry")
        try:
            try:
                mount = build_robot_from_camera_transform(
                    self.camera_config_registry.get_config(
                        self.camera_bus_id
                    ).extrinsics
                )
            except (TypeError, ValueError, AttributeError, OverflowError):
                return failure("invalid_mounting")
            if not np.isfinite(mount).all():
                return failure("invalid_mounting")
            cosine, sine = np.cos(yaw), np.sin(yaw)
            robot_rotation = np.array(
                [[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]]
            )
            rotation = robot_rotation @ mount[:3, :3]
            offset = robot_rotation @ mount[:3, 3]
            camera_rotation = rotation.T
            rotated = (obj - offset) @ camera_rotation.T
            rays = cv2.undistortPoints(
                image.reshape(-1, 1, 2),
                estimator.camera_matrix,
                estimator.distortion_coefficients,
            ).reshape(-1, 2)
            # Fixed heading/height make normalized projection equations linear in XY.
            a = (
                camera_rotation[:2, :2][None, :, :]
                - rays[:, :, None] * camera_rotation[2, :2]
            ).reshape(-1, 2)
            b = (rotated[:, :2] - rays * rotated[:, 2, None]).reshape(-1)
            xy, _, rank, singular = np.linalg.lstsq(a, b, rcond=None)
            condition = singular[0] / singular[-1] if singular[-1] > 0 else float("inf")
            diagnostics["condition_number"] = float(condition)
            if rank != 2 or not np.isfinite(condition) or condition > 1e8:
                return failure("ill_conditioned_geometry")
            rvec = cv2.Rodrigues(camera_rotation)[0]

            def project(
                position: np.ndarray,
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                """Return distorted pixel residual, XY Jacobian, and camera translation."""
                translation = -camera_rotation @ (offset + np.r_[position, 0.0])
                depth = (obj @ camera_rotation.T + translation)[:, 2]
                if not np.isfinite(depth).all() or np.any(depth <= 1e-6):
                    raise ValueError("behind_camera")
                pixels, jacobian = cv2.projectPoints(
                    obj,
                    rvec,
                    translation,
                    estimator.camera_matrix,
                    estimator.distortion_coefficients,
                )
                residual = (pixels.reshape(-1, 2) - image).reshape(-1)
                if not np.isfinite(residual).all():
                    raise ValueError("nonfinite_solution")
                return (
                    residual,
                    jacobian[:, 3:6] @ (-camera_rotation[:, :2]),
                    translation,
                )

            residual, jacobian, translation = project(xy)
            # Bounded Gauss-Newton and backtracking optimize actual distorted pixels.
            for _ in range(self.refinement_iterations):
                step = np.linalg.lstsq(jacobian, -residual, rcond=None)[0]
                if not np.isfinite(step).all():
                    return failure("nonfinite_solution")
                # ponytail: 1 m step ceiling; expose tuning only if benchmarks need it.
                step /= max(1.0, np.linalg.norm(step))
                improved = False
                for scale in (1, 0.5, 0.25, 0.125, 0.0625, 0.03125):
                    try:
                        candidate = project(xy + scale * step)
                    except ValueError:
                        continue
                    if candidate[0] @ candidate[0] < residual @ residual:
                        xy += scale * step
                        residual, jacobian, translation = candidate
                        improved = True
                        break
                if not improved or np.linalg.norm(step) < 1e-9:
                    break
            if not np.isfinite(xy).all() or np.linalg.cond(jacobian) > 1e8:
                return failure("ill_conditioned_geometry")
            pose = np.eye(4)
            pose[:3, :3] = rotation
            pose[:3, 3] = offset + np.r_[xy, 0.0]
            meta = estimator._solution_quality(
                obj, image, rvec, camera_rotation, translation, len(seen)
            )
            if not np.isfinite(meta).all():
                return failure("nonfinite_solution")
            diagnostics.update(reason="ok", robot_xy_m=xy.tolist())
            return {"camera_pose": pose, "pose_meta": meta, "diagnostics": diagnostics}
        except (cv2.error, np.linalg.LinAlgError, ValueError) as exc:
            return failure(
                str(exc)
                if str(exc) in {"behind_camera", "nonfinite_solution"}
                else "invalid_geometry"
            )


# The pipeline factory capitalizes each snake-case segment ("2d" -> "2d").
PnpCameraLocalization2dDefinition = PnpCameraLocalization2DDefinition
