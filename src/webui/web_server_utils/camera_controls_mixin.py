"""Live camera hardware controls and an AprilTag detection preview for the Utils tab."""

from __future__ import annotations

import threading
import time
from collections.abc import Generator
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from flask import Response, request

from src.main_operations.modules.apriltags.utils.visualization import (
    draw_apriltag_overlay,
)
from src.webui.web_server_utils.constants import VIEW_STREAM_FPS, no_image_jpeg_bytes

if TYPE_CHECKING:
    from src.main_operations.modules.apriltags.apriltag_detector import (
        AprilTagDetector,
    )
    from src.utils.camera_utils.cameras.physical_camera import PhysicalCamera

# The preview runs beside the pipeline, so it uses fewer threads than the live
# detector and stops when the browser closes the stream.
PREVIEW_DETECTOR_THREADS = 2
PREVIEW_QUAD_DECIMATE = 2.0


class CameraControlsMixin:
    """Read, change and save a camera's V4L2 controls; preview AprilTag detections."""

    _preview_detector: Any = None
    _preview_detector_lock = threading.Lock()
    # The detector reuses one grayscale buffer, so previews detect one at a time.
    _preview_detect_lock = threading.Lock()

    def _running_camera(self, camera_bus_id: str) -> PhysicalCamera | None:
        """Return the running camera for a bus ID, or None."""
        if self.camera_manager is None:
            return None
        return self.camera_manager.get_camera_by_bus_id(str(camera_bus_id))

    def _controls_payload(self, camera_bus_id: str, camera: PhysicalCamera) -> dict:
        """Describe the camera's controls, with the saved values next to them."""
        config = self._resolve_camera_config(camera_bus_id)
        return {
            "camera_bus_id": str(camera_bus_id),
            "controls": [asdict(control) for control in camera.list_controls()],
            "saved": config.camera_controls if config is not None else {},
        }

    def get_camera_controls(self, camera_bus_id: str) -> tuple[dict, int]:
        """Return the camera's controls and their current and saved values."""
        camera = self._running_camera(camera_bus_id)
        if camera is None:
            return {"error": "Camera is not running"}, 404
        return self._controls_payload(camera_bus_id, camera), 200

    def set_camera_controls(self, camera_bus_id: str) -> tuple[dict, int]:
        """Apply control values live without saving them.

        Expects a JSON object of control key to integer value. The response
        lists the refreshed controls, since changing an auto mode makes other
        controls active or inactive.
        """
        camera = self._running_camera(camera_bus_id)
        if camera is None:
            return {"error": "Camera is not running"}, 404
        values = request.get_json(silent=True)
        if not isinstance(values, dict) or not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in values.values()
        ):
            return {"error": "Expected a JSON object of integer control values"}, 400
        refused = camera.set_controls(values)
        payload = self._controls_payload(camera_bus_id, camera)
        payload["refused"] = refused
        return payload, 200

    def save_camera_controls(self, camera_bus_id: str) -> tuple[dict, int]:
        """Save the camera's current writable control values as its startup settings."""
        camera = self._running_camera(camera_bus_id)
        if camera is None:
            return {"error": "Camera is not running"}, 404
        config = self._resolve_camera_config(camera_bus_id)
        if config is None:
            return {"error": "Camera config registry unavailable"}, 503
        try:
            values = {
                control.key: control.value
                for control in camera.list_controls()
                if not control.read_only and control.value is not None
            }
            if not values:
                # An unplugged camera lists no controls; saving would erase its settings.
                return {"error": "Camera settings are unavailable; nothing saved"}, 409
            config.save_camera_controls(values)
        except OSError as error:
            self.log(f"Failed saving camera controls for {camera_bus_id}: {error}")
            return {"error": "Failed to save camera settings"}, 500
        return self._controls_payload(camera_bus_id, camera), 200

    def _apriltag_preview_detector(self) -> AprilTagDetector:
        """Return the shared custom-detector instance used by previews."""
        with self._preview_detector_lock:
            if CameraControlsMixin._preview_detector is None:
                from src.main_operations.modules.apriltags.apriltag_detector import (
                    AprilTagDetector,
                )

                CameraControlsMixin._preview_detector = AprilTagDetector(
                    nthreads=PREVIEW_DETECTOR_THREADS,
                    quad_decimate=PREVIEW_QUAD_DECIMATE,
                    rust_backend=True,
                    track_tags=False,
                )
            return CameraControlsMixin._preview_detector

    @staticmethod
    def _draw_apriltags(
        frame: np.ndarray, detections: list[Any], detect_ms: float
    ) -> np.ndarray:
        """Draw detected tag outlines, IDs and the detection time on a copy of the frame."""
        color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame
        output = draw_apriltag_overlay(
            color, [(d.tag_id, d.corners) for d in detections], []
        )
        cv2.putText(
            output,
            f"Tags: {len(detections)} | detect {detect_ms:.1f} ms",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if detections else (0, 0, 255),
            2,
        )
        return output

    def _apriltag_feed_generator(self, camera_bus_id: str) -> Generator[bytes, Any, Any]:
        """Stream the camera with the custom detector's detections drawn on it."""
        detector = self._apriltag_preview_detector()
        while True:
            start = time.time()
            frame = self._latest_camera_frame(camera_bus_id)
            jpeg = no_image_jpeg_bytes
            if frame is not None:
                with self._preview_detect_lock:
                    detect_start = time.perf_counter()
                    detections = detector.run_detection(frame) or []
                    detect_ms = (time.perf_counter() - detect_start) * 1e3
                output = self._resize_view_stream_frame(
                    self._draw_apriltags(frame, detections, detect_ms)
                )
                ok, encoded = cv2.imencode(".jpg", output, self._view_stream_jpeg_params())
                if ok:
                    jpeg = encoded.tobytes()
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            time.sleep(max((1 / VIEW_STREAM_FPS) - (time.time() - start), 0))

    def apriltag_preview_feed(self, camera_bus_id: str) -> Response | tuple[dict, int]:
        """Serve the AprilTag detection preview as an MJPEG stream."""
        try:
            self._apriltag_preview_detector()
        except ImportError as error:
            return {"error": f"Custom AprilTag detector unavailable: {error}"}, 503
        return Response(
            self._apriltag_feed_generator(camera_bus_id),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )
