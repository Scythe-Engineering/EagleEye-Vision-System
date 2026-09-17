from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from pupil_apriltags import Detection

from src.main_operations.definitions.base.base_class import OperationInstance
from src.utils.camera_utils.camera_config_manager import CameraConfigRegistry
from src.utils.camera_utils.load_camera_parameters import load_camera_parameters
from src.utils.timing import TimedValue, get_timing, unwrap_timed
from src.webui.web_server import EagleEyeInterface

from ..modules.apriltags.pnp_localization import PnpLocalization
from ..modules.apriltags.utils.fmap_parser import load_fmap_file


class PnpCameraLocalizationDefinition(OperationInstance):
    """Definition for camera localization operations using AprilTags."""

    def __init__(
        self,
        camera_bus_id: str,
        apriltag_map_path: str,
        camera_config_registry: CameraConfigRegistry | None = None,
        web_interface: EagleEyeInterface | None = None,
        refinement_iterations: int = 10,
        use_pose_continuity: bool = True,
    ) -> None:
        """Initialize the camera localization definition.

        Args:
            camera_bus_id: Camera bus ID used to resolve calibration files.
            apriltag_map_path: Path to the apriltag map file.
            camera_config_registry: Injected shared camera config registry.
            web_interface: Optional frontend interface.
            refinement_iterations: Maximum LM iterations, zero disables refinement.
            use_pose_continuity: Resolve single-tag ties using recent capture-timed
                poses and reject implausible jumps without holding or smoothing output.
        """
        self.web_interface = web_interface
        self.uses_timed_inputs = True
        self.use_pose_continuity = bool(use_pose_continuity)
        self._previous_pose: np.ndarray | None = None
        self._previous_capture_ns: int | None = None
        self._last_capture_ns: int | None = None

        intrinsics_path: str
        if camera_config_registry is not None:
            camera_config = camera_config_registry.get_config(camera_bus_id)
            if camera_config.intrinsics_path is None:
                raise ValueError(
                    f"No intrinsics path found for camera bus ID '{camera_bus_id}'"
                )
            intrinsics_path = camera_config.intrinsics_path
        else:
            intrinsics_path = str(
                Path(__file__).resolve().parents[2]
                / "utils"
                / "camera_utils"
                / "camera_calibrations"
                / camera_bus_id
                / "intrinsics.json"
            )

        camera_matrix, distortion_coefficients = load_camera_parameters(intrinsics_path)
        apriltag_map = load_fmap_file(apriltag_map_path)

        self.pose_estimator = PnpLocalization(
            camera_matrix=camera_matrix,
            distortion_coefficients=distortion_coefficients,
            apriltag_map=apriltag_map,
            refinement_iterations=refinement_iterations,
        )

    def update_config(self, json_config: dict) -> None:
        """Apply live solver settings and reset history when continuity changes.

        Args:
            json_config: Operation configuration containing optional refinement and
                continuity settings.
        """
        if "refinement_iterations" in json_config:
            self.pose_estimator.set_refinement_iterations(
                json_config["refinement_iterations"]
            )
        if "use_pose_continuity" in json_config:
            self.use_pose_continuity = bool(json_config["use_pose_continuity"])
            self._reset_pose_history()

    def _reset_pose_history(self) -> None:
        """Forget history after a setting change or a discontinuous capture clock."""
        self._previous_pose = None
        self._previous_capture_ns = None
        self._last_capture_ns = None

    @staticmethod
    def _pose_delta(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
        """Return field translation and rotation separation between two poses."""
        distance = float(np.linalg.norm(first[:3, 3] - second[:3, 3]))
        cosine = np.clip(
            (np.trace(first[:3, :3].T @ second[:3, :3]) - 1) / 2, -1.0, 1.0
        )
        return distance, float(np.arccos(cosine))

    def run(
        self, detections: List[Detection] | TimedValue[List[Detection]]
    ) -> Dict[str, Any]:
        """Estimate camera pose from AprilTag detections.

        Args:
            detections: AprilTag detections, optionally carrying capture timing.
                Missing, repeated, or backward timestamps clear continuity history;
                only a pose captured within 250 ms is eligible as a reference.

        Returns:
            Mapping of ``camera_pose`` to a 4x4 transform in global coordinates and
            ``pose_meta`` to ``[tag_count, mean_tag_distance_m, reprojection_error_px]``.
            Both ports carry None when pose estimation fails or a recent pose
            exposes an implausible single-tag jump. No previous pose is emitted.
        """
        timing = get_timing(detections)
        raw_detections = unwrap_timed(detections)
        previous = None
        capture_ns = timing.capture_monotonic_ns if timing is not None else None
        if (
            not self.use_pose_continuity
            or capture_ns is None
            or (
                self._last_capture_ns is not None
                and capture_ns <= self._last_capture_ns
            )
        ):
            self._reset_pose_history()
            capture_ns = None
        elif (
            self._previous_capture_ns is not None
            and capture_ns - self._previous_capture_ns <= 250_000_000
        ):
            previous = self._previous_pose
        else:
            self._previous_pose = None
            self._previous_capture_ns = None

        solution = self.pose_estimator.estimate_pose_from_detections(
            raw_detections, previous_pose=previous
        )
        if solution is None:
            if capture_ns is not None and (
                self._last_capture_ns is None or capture_ns > self._last_capture_ns
            ):
                self._last_capture_ns = capture_ns
            return {"camera_pose": None, "pose_meta": None}
        camera_pose, pose_meta = solution
        if capture_ns is not None and (
            self._last_capture_ns is None or capture_ns > self._last_capture_ns
        ):
            if previous is not None and pose_meta[0] == 1.0:
                translation, rotation = self._pose_delta(camera_pose, previous)
                if translation > 2.0 or rotation > 0.75:
                    # No plausible current-image solution: publish nothing, not the
                    # previous pose. Keep its original age so rejection cannot lock
                    # out reacquisition beyond the 250 ms history window.
                    self._last_capture_ns = capture_ns
                    return {"camera_pose": None, "pose_meta": None}
            self._previous_pose = camera_pose.copy()
            self._previous_capture_ns = capture_ns
            self._last_capture_ns = capture_ns
        return {"camera_pose": camera_pose, "pose_meta": pose_meta}
