from typing import Any, Dict, Optional
import platform
import time
import numpy as np
import cv2
import subprocess
from src.utils.colors import Colors
from src.main_operations.definitions.base.base_class import OperationInstance


class CameraAdjust(OperationInstance):
    def __init__(
        self,
        camera_bus_id: str | None = None,
        brightness: float = 0.0,
        contrast: float = 0.5,
        saturation: float = 0.406,
        gain: float = 0.0,
        exposure: float = 0.5,
        camera_manager: Any | None = None,
        logger: Any | None = None,
    ) -> None:
        """Initialize hardware-accelerated camera adjustment operation.

        All adjustments are attempted via hardware device controls using v4l2-ctl.

        Args:
            camera_bus_id: Deterministic camera bus ID (matches device_input and
                camera config registry) for resolving the v4l2 device path.
            brightness: Brightness offset normalized in range [-1, 1], mapped to v4l2 range [-64, 64].
            contrast: Contrast multiplier normalized in range [0, 1], mapped to v4l2 range [0, 64].
            saturation: Saturation multiplier normalized in range [-1, 1], mapped to v4l2 range [0, 128].
            gain: Gain control normalized in range [0, 1], mapped to v4l2 range [0, 100].
            exposure: Exposure time normalized in range [0, 1], mapped to v4l2 range [1, 5000]. Disables auto exposure when set.
            camera_manager: Injected camera manager reference.
            logger: Project logger injected by the pipeline.
        """
        self.camera_bus_id = (
            str(camera_bus_id) if camera_bus_id is not None else None
        )
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self.gain = float(gain)
        self.exposure = float(exposure)

        self.camera_manager = camera_manager
        self.logger = logger

        self._last_applied: Dict[str, Any] = {}
        self._apriltag_detections: Optional[Any] = None
        self._controls_cache: Dict[str, Dict[str, int]] = {}
        self._controls_cache_device: str | None = None
        self._last_enforce_time = 0.0
        # The camera may not be open yet during construction; run() will keep
        # enforcing manual controls after OpenCV finishes applying format/FPS.
        self._apply_device_controls(force=True)

    def _log(self, message: str) -> None:
        """Log through the project logger when available."""
        if self.logger is not None:
            self.logger.log(message)

    def run(self, input_data: Any) -> np.ndarray:
        """Return the frame, hardware adjustments already applied.

        Args:
            input_data: Input data - dict with 'frame' and optionally 'detections' keys.

        Returns:
            Frame, already adjusted by hardware.
        """
        if isinstance(input_data, dict):
            frame = input_data.get("frame")
            detections = input_data.get("detections")
        else:
            frame = input_data
            detections = None

        if frame is None:
            raise ValueError("Frame input is required")

        if detections is not None:
            self._apriltag_detections = detections

        # Some UVC drivers/OpenCV renegotiate controls shortly after stream start
        # and can re-enable auto exposure/gain. Keep the Linux camera in manual
        # mode and re-apply values periodically instead of only on config change.
        now = time.monotonic()
        if now - self._last_enforce_time >= 1.0:
            self._apply_device_controls(force=True)
            self._last_enforce_time = now

        return frame

    def update_config(self, json_config: Dict[str, Any]) -> None:
        """Update configuration and apply to hardware device controls.

        Args:
            json_config: Mapping of parameter names to new values.
        """
        prior_bus = self.camera_bus_id
        for key, value in json_config.items():
            if hasattr(self, key):
                setattr(self, key, value)
        if "camera_bus_id" in json_config and self.camera_bus_id != prior_bus:
            self._controls_cache = {}
            self._controls_cache_device = None
        self._apply_device_controls(force=True)

    def visualize(self, frame: np.ndarray) -> np.ndarray:
        """Return a visualization frame with AprilTag detections drawn.

        Args:
            frame: Input frame.

        Returns:
            Visualization frame with AprilTag detections overlaid.
        """
        vis_frame = frame.copy()

        if self._apriltag_detections is not None:
            for detection in self._apriltag_detections:
                if detection is None:
                    continue

                # Get corners and tag_id
                corners = detection.corners
                tag_id = detection.tag_id

                if corners is not None and len(corners) == 4:
                    # Convert corners to integer coordinates
                    corners_int = np.round(corners).astype(int)

                    # Draw the tag outline
                    cv2.polylines(vis_frame, [corners_int], True, (0, 255, 0), 2)

                    # Calculate center point for tag ID
                    center_x = int(np.mean(corners[:, 0]))
                    center_y = int(np.mean(corners[:, 1]))

                    # Draw tag ID
                    cv2.putText(
                        vis_frame,
                        str(tag_id),
                        (center_x, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

        return vis_frame

    def _get_device_path(self) -> str | None:
        """Resolve the v4l2 device path for the current camera if available.

        Returns:
            Device path like "/dev/video0" or None if not available.
        """
        if self.camera_manager is None:
            return None
        bus_id = self.camera_bus_id
        if bus_id is None:
            return None
        camera_name = self.camera_manager.get_camera_name_by_bus_id(bus_id)
        if camera_name is None:
            return None
        worker = self.camera_manager.cameras.get(camera_name)
        if worker is None:
            return None
        camera_index = getattr(worker.camera, "camera_index", None)
        if camera_index is None:
            return None
        return f"/dev/video{int(camera_index)}"

    def _run_v4l2(self, args: list[str]) -> subprocess.CompletedProcess[str] | None:
        device_path = self._get_device_path()
        if device_path is None or platform.system() != "Linux":
            return None
        try:
            return subprocess.run(
                ["v4l2-ctl", "-d", device_path, *args],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception as e:
            self._log(f"{Colors.RED}Error running v4l2-ctl for {device_path}: {e}{Colors.RESET}")
            return None

    def _get_v4l2_controls(self) -> Dict[str, Dict[str, int]]:
        """Return available integer/menu controls and their ranges for the camera."""
        device_path = self._get_device_path()
        if device_path is None:
            return {}
        if self._controls_cache_device == device_path and self._controls_cache:
            return self._controls_cache

        result = self._run_v4l2(["--list-ctrls"])
        if result is None or result.returncode != 0:
            return {}

        controls: Dict[str, Dict[str, int]] = {}
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line or "(" not in line:
                continue
            name = line.split()[0]
            values: Dict[str, int] = {}
            for token in line.replace(":", " ").split():
                if "=" not in token:
                    continue
                key, raw_value = token.split("=", 1)
                try:
                    values[key] = int(raw_value)
                except ValueError:
                    continue
            controls[name] = values

        self._controls_cache = controls
        self._controls_cache_device = device_path
        return controls

    def _scale_control(self, control_name: str, normalized: float, default_min: int, default_max: int) -> int:
        controls = self._get_v4l2_controls()
        ctrl = controls.get(control_name, {})
        min_value = ctrl.get("min", default_min)
        max_value = ctrl.get("max", default_max)
        value = int(round(min_value + normalized * (max_value - min_value)))
        return max(min_value, min(max_value, value))

    def _set_v4l2_control(self, control_name: str, value: int) -> bool:
        """Set a v4l2 control using v4l2-ctl command."""
        result = self._run_v4l2(["-c", f"{control_name}={value}"])
        if result is None:
            return False
        if result.returncode == 0:
            return True
        device_path = self._get_device_path() or "/dev/video0"
        self._log(
            f"{Colors.RED}Failed command: v4l2-ctl -d {device_path} -c {control_name}={value}\n"
            f"stderr: {result.stderr.strip()}{Colors.RESET}"
        )
        return False

    def _set_first_available_control(self, control_names: list[str], value: int) -> bool:
        controls = self._get_v4l2_controls()
        for control_name in control_names:
            if control_name in controls and self._set_v4l2_control(control_name, value):
                return True
        return False

    def _apply_device_controls(self, force: bool = False) -> None:
        """Apply adjustments via Linux hardware device controls using v4l2-ctl."""
        if platform.system() != "Linux":
            return
        controls = self._get_v4l2_controls()
        if not controls:
            return

        # Disable automatic controls first. UVC auto_exposure manual mode is 1;
        # exposure_auto manual mode is also commonly 1 on older drivers.
        self._set_first_available_control(["auto_exposure", "exposure_auto"], 1)
        self._set_first_available_control(["gain_automatic", "auto_gain"], 0)
        self._set_first_available_control(["white_balance_automatic", "white_balance_temperature_auto"], 0)
        self._set_first_available_control(["focus_automatic_continuous", "focus_auto"], 0)

        settings = {
            "brightness": self._scale_control("brightness", (self.brightness + 1) / 2, -64, 64),
            "contrast": self._scale_control("contrast", self.contrast, 0, 64),
            "saturation": self._scale_control("saturation", (self.saturation + 1) / 2, 0, 128),
            "gain": self._scale_control("gain", self.gain, 0, 100),
            "exposure": self._scale_control("exposure_time_absolute", self.exposure, 1, 5000),
        }

        control_names = {
            "brightness": ["brightness"],
            "contrast": ["contrast"],
            "saturation": ["saturation"],
            "gain": ["gain"],
            "exposure": ["exposure_time_absolute", "exposure_absolute"],
        }

        for setting_name, value in settings.items():
            if not force and self._last_applied.get(setting_name) == getattr(self, setting_name):
                continue
            if self._set_first_available_control(control_names[setting_name], value):
                self._last_applied[setting_name] = getattr(self, setting_name)
