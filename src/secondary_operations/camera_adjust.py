import cv2
from typing import Any, Dict
import numpy as np
from src.utils.colors import Colors


class CameraAdjust:
    def __init__(
        self,
        brightness: float = 0.0,
        contrast: float = 1.0,
        exposure: float = 0.0,
        saturation: float = 1.0,
        hue: int = 0,
        camera_manager: Any | None = None,
        pipeline: Any | None = None,
    ) -> None:
        """Initialize hardware-accelerated camera adjustment operation.

        All adjustments are attempted via hardware device controls. No software fallbacks.

        Args:
            brightness: Brightness offset in range [-100, 100]. Hardware accelerated only.
            contrast: Contrast multiplier in range [0.5, 3.0]. Hardware accelerated only.
            exposure: Exposure control mapped to gamma (-2.0..2.0). Hardware accelerated only.
            saturation: Saturation multiplier [0.0, 2.0]. Hardware accelerated only.
            hue: Hue shift in degrees [-180, 180]. Hardware accelerated only.
            camera_manager: Injected camera manager reference.
            pipeline: Injected pipeline reference.
        """
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.exposure = float(exposure)
        self.saturation = float(saturation)
        self.hue = int(hue)

        self.camera_manager = camera_manager
        self.pipeline = pipeline

        self._last_applied: Dict[str, Any] = {}
        self._apply_device_controls()

    def run(self, frame: np.ndarray) -> np.ndarray:
        """Return the frame, hardware adjustments already applied.

        Args:
            frame: Input BGR frame as numpy array (uint8).

        Returns:
            Frame, already adjusted by hardware.
        """
        return frame

    def update_config(self, params: Dict[str, Any]) -> None:
        """Update configuration and apply to hardware device controls.

        Args:
            params: Mapping of parameter names to new values.
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self._apply_device_controls()

    def visualize(self, frame: np.ndarray) -> np.ndarray:
        """Return a visualization frame.

        Args:
            frame: Input frame.

        Returns:
            Visualization frame.
        """
        return frame

    def _get_device_capture(self) -> Any | None:
        """Get the OpenCV VideoCapture for the current pipeline camera if available.

        Returns:
            The VideoCapture handle or None if not available.
        """
        if self.camera_manager is None or self.pipeline is None:
            return None
        camera_name = getattr(self.pipeline, "camera_bus_id", None)
        if camera_name is None:
            return None
        camera_obj = self.camera_manager.camera_objects.get(camera_name)
        if camera_obj is None:
            return None
        cap = getattr(camera_obj, "cap", None)
        if cap is None:
            return None
        try:
            opened = cap.isOpened()
        except Exception as e:
            print(
                f"{Colors.RED}Error checking camera capture status: {e}{Colors.RESET}"
            )
            opened = False
        return cap if opened else None

    def _apply_device_controls(self) -> None:
        """Apply adjustments via hardware device controls."""
        cap = self._get_device_capture()
        if cap is None:
            return

        # Brightness
        if self._last_applied.get("brightness") != self.brightness:
            try:
                if cap.set(cv2.CAP_PROP_BRIGHTNESS, float(self.brightness)):
                    self._last_applied["brightness"] = self.brightness
                else:
                    print(
                        f"{Colors.RED}Failed to set brightness {self.brightness}: hardware not supported{Colors.RESET}"
                    )
            except Exception as e:
                print(
                    f"{Colors.RED}Error setting brightness {self.brightness}: {e}{Colors.RESET}"
                )

        # Contrast
        if self._last_applied.get("contrast") != self.contrast:
            try:
                if cap.set(cv2.CAP_PROP_CONTRAST, float(self.contrast)):
                    self._last_applied["contrast"] = self.contrast
                else:
                    print(
                        f"{Colors.RED}Failed to set contrast {self.contrast}: hardware not supported{Colors.RESET}"
                    )
            except Exception as e:
                print(
                    f"{Colors.RED}Error setting contrast {self.contrast}: {e}{Colors.RESET}"
                )

        # Saturation
        if self._last_applied.get("saturation") != self.saturation:
            try:
                if cap.set(cv2.CAP_PROP_SATURATION, float(self.saturation)):
                    self._last_applied["saturation"] = self.saturation
                else:
                    print(
                        f"{Colors.RED}Failed to set saturation {self.saturation}: hardware not supported{Colors.RESET}"
                    )
            except Exception as e:
                print(
                    f"{Colors.RED}Error setting saturation {self.saturation}: {e}{Colors.RESET}"
                )

        # Hue
        if self._last_applied.get("hue") != self.hue:
            try:
                if cap.set(cv2.CAP_PROP_HUE, float(self.hue)):
                    self._last_applied["hue"] = self.hue
                else:
                    print(
                        f"{Colors.RED}Failed to set hue {self.hue}: hardware not supported{Colors.RESET}"
                    )
            except Exception as e:
                print(f"{Colors.RED}Error setting hue {self.hue}: {e}{Colors.RESET}")

        # Exposure and gamma (best effort)
        if self._last_applied.get("exposure") != self.exposure:
            try:
                # Try manual exposure first if supported
                try:
                    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0)
                except Exception as e:
                    print(
                        f"{Colors.RED}Warning: Could not disable auto exposure: {e}{Colors.RESET}"
                    )
                if cap.set(cv2.CAP_PROP_EXPOSURE, float(self.exposure)):
                    self._last_applied["exposure"] = self.exposure
                else:
                    # Fallback to gamma mapping when exposure not supported
                    exposure_clamped = max(-2.0, min(2.0, float(self.exposure)))
                    gamma = 2.0 ** (-exposure_clamped)
                    if cap.set(cv2.CAP_PROP_GAMMA, float(gamma)):
                        self._last_applied["exposure"] = self.exposure
                    else:
                        print(
                            f"{Colors.RED}Failed to set exposure {self.exposure} or gamma fallback: hardware not supported{Colors.RESET}"
                        )

            except Exception as e:
                print(
                    f"{Colors.RED}Error setting exposure {self.exposure}: {e}{Colors.RESET}"
                )
