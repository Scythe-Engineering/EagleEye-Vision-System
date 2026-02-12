from typing import Any, Dict, Optional
import numpy as np
import cv2
import subprocess
from src.utils.colors import Colors
from src.main_operations.definitions.base.base_class import OperationInstance


class CameraAdjust(OperationInstance):
    def __init__(
        self,
        brightness: float = 0.0,
        contrast: float = 0.5,
        saturation: float = 0.406,
        gain: float = 0.0,
        exposure: float = 0.5,
        camera_manager: Any | None = None,
        pipeline: Any | None = None,
    ) -> None:
        """Initialize hardware-accelerated camera adjustment operation.

        All adjustments are attempted via hardware device controls using v4l2-ctl.

        Args:
            brightness: Brightness offset normalized in range [-1, 1], mapped to v4l2 range [-64, 64].
            contrast: Contrast multiplier normalized in range [0, 1], mapped to v4l2 range [0, 64].
            saturation: Saturation multiplier normalized in range [-1, 1], mapped to v4l2 range [0, 128].
            gain: Gain control normalized in range [0, 1], mapped to v4l2 range [0, 100].
            exposure: Exposure time normalized in range [0, 1], mapped to v4l2 range [1, 5000]. Disables auto exposure when set.
            camera_manager: Injected camera manager reference.
            pipeline: Injected pipeline reference.
        """
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self.gain = float(gain)
        self.exposure = float(exposure)

        self.camera_manager = camera_manager
        self.pipeline = pipeline

        self._last_applied: Dict[str, Any] = {}
        self._apriltag_detections: Optional[Any] = None
        self._apply_device_controls()

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

        return frame

    def update_config(self, json_config: Dict[str, Any]) -> None:
        """Update configuration and apply to hardware device controls.

        Args:
            json_config: Mapping of parameter names to new values.
        """
        for key, value in json_config.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self._apply_device_controls()

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
        if self.camera_manager is None or self.pipeline is None:
            return None
        bus_id = getattr(self.pipeline, "camera_bus_id", None)
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

    def _set_v4l2_control(self, control_name: str, value: int) -> bool:
        """Set a v4l2 control using v4l2-ctl command.

        Args:
            control_name: Name of the control (e.g., 'brightness', 'contrast').
            value: Value to set.

        Returns:
            True if setting succeeded, else False.
        """
        device_path = self._get_device_path()
        if device_path is None:
            return False

        try:
            cmd = ["v4l2-ctl", "-d", device_path, "-c", f"{control_name}={value}"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except Exception as e:
            cmd_str = f"v4l2-ctl -d {device_path} -c {control_name}={value}"
            print(f"{Colors.RED}Error running command '{cmd_str}': {e}{Colors.RESET}")
            return False

    def _apply_device_controls(self) -> None:
        """Apply adjustments via hardware device controls using v4l2-ctl."""
        # Apply brightness (map -1 to 1 -> -64 to 64)
        brightness_v4l2 = int(self.brightness * 64)
        if self._last_applied.get("brightness") != self.brightness:
            if self._set_v4l2_control("brightness", brightness_v4l2):
                self._last_applied["brightness"] = self.brightness
            else:
                device_path = self._get_device_path() or "/dev/video0"
                cmd_str = f"v4l2-ctl -d {device_path} -c brightness={brightness_v4l2}"
                print(
                    f"{Colors.RED}Failed to set brightness {self.brightness} using command: {cmd_str}{Colors.RESET}"
                )

        # Apply contrast (map 0-1 to 0-64)
        contrast_v4l2 = int(self.contrast * 64)
        if self._last_applied.get("contrast") != self.contrast:
            if self._set_v4l2_control("contrast", contrast_v4l2):
                self._last_applied["contrast"] = self.contrast
            else:
                device_path = self._get_device_path() or "/dev/video0"
                cmd_str = f"v4l2-ctl -d {device_path} -c contrast={contrast_v4l2}"
                print(
                    f"{Colors.RED}Failed to set contrast {self.contrast} using command: {cmd_str}{Colors.RESET}"
                )

        # Apply saturation (map -1 to 1 -> 0 to 128)
        saturation_v4l2 = int((self.saturation + 1) * 64)
        if self._last_applied.get("saturation") != self.saturation:
            if self._set_v4l2_control("saturation", saturation_v4l2):
                self._last_applied["saturation"] = self.saturation
            else:
                device_path = self._get_device_path() or "/dev/video0"
                cmd_str = f"v4l2-ctl -d {device_path} -c saturation={saturation_v4l2}"
                print(
                    f"{Colors.RED}Failed to set saturation {self.saturation} using command: {cmd_str}{Colors.RESET}"
                )

        # Apply gain (map 0-1 to 0-100)
        gain_v4l2 = int(self.gain * 100)
        if self._last_applied.get("gain") != self.gain:
            if self._set_v4l2_control("gain", gain_v4l2):
                self._last_applied["gain"] = self.gain
            else:
                device_path = self._get_device_path() or "/dev/video0"
                cmd_str = f"v4l2-ctl -d {device_path} -c gain={gain_v4l2}"
                print(
                    f"{Colors.RED}Failed to set gain {self.gain} using command: {cmd_str}{Colors.RESET}"
                )

        # Apply exposure (map 0-1 to 1-5000)
        exposure_v4l2 = int(1 + (self.exposure * (5000 - 1)))
        if self._last_applied.get("exposure") != self.exposure:
            # Disable auto exposure before setting manual exposure time
            if not self._set_v4l2_control("auto_exposure", 1):
                device_path = self._get_device_path() or "/dev/video0"
                cmd_str = f"v4l2-ctl -d {device_path} -c auto_exposure=1"
                print(
                    f"{Colors.RED}Failed to disable auto exposure using command: {cmd_str}{Colors.RESET}"
                )

            # Set manual exposure time
            if self._set_v4l2_control("exposure_time_absolute", exposure_v4l2):
                self._last_applied["exposure"] = self.exposure
            else:
                device_path = self._get_device_path() or "/dev/video0"
                cmd_str = f"v4l2-ctl -d {device_path} -c exposure_time_absolute={exposure_v4l2}"
                print(
                    f"{Colors.RED}Failed to set exposure {self.exposure} using command: {cmd_str}{Colors.RESET}"
                )
