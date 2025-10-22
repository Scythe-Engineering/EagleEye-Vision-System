import cv2
import numpy as np
from typing import Any, Dict


class CameraAdjust:
    def __init__(
        self,
        brightness: float = 0.0,
        contrast: float = 1.0,
        exposure: float = 0.0,
        saturation: float = 1.0,
        hue: int = 0,
        enable_clahe: bool = False,
        clahe_clip_limit: float = 2.0,
        camera_manager: Any | None = None,
        pipeline: Any | None = None,
    ) -> None:
        """Initialize adjustable image operation.

        Hardware-accelerated parameters (if device supports):
            brightness, contrast, saturation, hue, exposure
            These are attempted on device first; fallback to CPU if unsupported.

        CPU-based (always software):
            enable_clahe - CLAHE is always computed on CPU regardless of device.
            clahe_clip_limit - Only relevant when enable_clahe=True.

        Args:
            brightness: Brightness offset in range [-100, 100]. Attempts hardware first.
            contrast: Contrast multiplier in range [0.5, 3.0]. Attempts hardware first.
            exposure: Exposure control mapped to gamma (-2.0..2.0). Attempts hardware first.
            saturation: Saturation multiplier [0.0, 2.0]. Attempts hardware first.
            hue: Hue shift in degrees [-180, 180]. Attempts hardware first.
            enable_clahe: Enable CLAHE on luminance channel. Always CPU-based (causes overhead).
            clahe_clip_limit: CLAHE clip limit if enabled. Only relevant when enable_clahe=True.
            camera_manager: Injected camera manager reference.
            pipeline: Injected pipeline reference.
        """
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.exposure = float(exposure)
        self.saturation = float(saturation)
        self.hue = int(hue)
        self.enable_clahe = bool(enable_clahe)
        self.clahe_clip_limit = float(clahe_clip_limit)

        self.camera_manager = camera_manager
        self.pipeline = pipeline

        self._last_applied: Dict[str, Any] = {}
        self._hardware_applied: bool = False
        self._apply_device_controls()

    def _apply_brightness_contrast(self, frame: np.ndarray) -> np.ndarray:
        alpha = max(0.0, float(self.contrast))
        beta = float(self.brightness)
        adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        return adjusted

    def _apply_exposure(self, frame: np.ndarray) -> np.ndarray:
        # Map exposure [-2,2] to gamma (0.25..4.0), exposure>0 -> brighter (gamma<1)
        exposure_clamped = max(-2.0, min(2.0, float(self.exposure)))
        gamma = 2.0 ** (-exposure_clamped)
        if abs(gamma - 1.0) < 1e-3:
            return frame
        inv_gamma = 1.0 / gamma
        # Use lookup table for speed
        table = (
            np.clip(((np.arange(256) / 255.0) ** inv_gamma) * 255.0, 0, 255)
        ).astype(np.uint8)
        return cv2.LUT(frame, table)

    def _apply_hue_saturation(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        # Hue shift in OpenCV hue space [0,180]
        hue_shift = int(self.hue // 2)  # degrees to OpenCV units
        h = (h.astype(np.int16) + hue_shift) % 180
        h = h.astype(np.uint8)
        s = np.clip(s.astype(np.float32) * float(self.saturation), 0, 255).astype(
            np.uint8
        )
        hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        if not self.enable_clahe:
            return frame
        clip_limit = max(0.1, float(self.clahe_clip_limit))
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        luminance_channel, a_channel, b_channel = cv2.split(lab)
        luminance_channel = clahe.apply(luminance_channel)
        lab = cv2.merge([luminance_channel, a_channel, b_channel])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def run(self, frame: np.ndarray) -> np.ndarray:
        """Return the frame adjusted by hardware when possible, else fallback to software.

        Args:
            frame: Input BGR frame as numpy array (uint8).

        Returns:
            Frame after applying adjustments. Returns input unchanged if hardware handled it.
        """
        if frame is None:
            return frame

        if self._hardware_applied and not self.enable_clahe:
            return frame

        adjusted = frame
        if not self._hardware_applied:
            adjusted = self._apply_brightness_contrast(adjusted)
            adjusted = self._apply_exposure(adjusted)
            adjusted = self._apply_hue_saturation(adjusted)
        adjusted = self._apply_clahe(adjusted)
        return adjusted

    def update_config(self, params: Dict[str, Any]) -> None:
        """Update configuration and push to device when supported.

        Args:
            params: Mapping of parameter names to new values.
        """
        for key, value in params.items():
            if not hasattr(self, key):
                continue
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
        except Exception:
            opened = False
        return cap if opened else None

    def _apply_device_controls(self) -> None:
        """Attempt to apply adjustments via device properties to offload computation."""
        cap = self._get_device_capture()
        self._hardware_applied = False
        if cap is None:
            return

        any_success = False

        # Brightness
        if self._last_applied.get("brightness") != self.brightness:
            try:
                if cap.set(cv2.CAP_PROP_BRIGHTNESS, float(self.brightness)):
                    self._last_applied["brightness"] = self.brightness
                    any_success = True
            except Exception:
                pass

        # Contrast
        if self._last_applied.get("contrast") != self.contrast:
            try:
                if cap.set(cv2.CAP_PROP_CONTRAST, float(self.contrast)):
                    self._last_applied["contrast"] = self.contrast
                    any_success = True
            except Exception:
                pass

        # Saturation
        if self._last_applied.get("saturation") != self.saturation:
            try:
                if cap.set(cv2.CAP_PROP_SATURATION, float(self.saturation)):
                    self._last_applied["saturation"] = self.saturation
                    any_success = True
            except Exception:
                pass

        # Hue
        if self._last_applied.get("hue") != self.hue:
            try:
                if cap.set(cv2.CAP_PROP_HUE, float(self.hue)):
                    self._last_applied["hue"] = self.hue
                    any_success = True
            except Exception:
                pass

        # Exposure and gamma (best effort)
        if self._last_applied.get("exposure") != self.exposure:
            try:
                # Try manual exposure first if supported
                try:
                    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0)
                except Exception:
                    pass
                if cap.set(cv2.CAP_PROP_EXPOSURE, float(self.exposure)):
                    self._last_applied["exposure"] = self.exposure
                    any_success = True
                else:
                    # Fallback to gamma mapping when exposure not supported
                    exposure_clamped = max(-2.0, min(2.0, float(self.exposure)))
                    gamma = 2.0 ** (-exposure_clamped)
                    if cap.set(cv2.CAP_PROP_GAMMA, float(gamma)):
                        self._last_applied["exposure"] = self.exposure
                        any_success = True
            except Exception:
                pass

        self._hardware_applied = any_success and not self.enable_clahe
