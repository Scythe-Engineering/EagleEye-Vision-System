"""Live camera hardware controls and an AprilTag detection preview for the Utils tab."""

from __future__ import annotations

import threading
import time
from collections.abc import Generator
from typing import Any

import cv2
import numpy as np
from flask import Response, request

from src.webui.web_server_utils.constants import VIEW_STREAM_FPS, no_image_jpeg_bytes

# The preview runs beside the pipeline, so it uses fewer threads than the live
# detector and stops when the browser closes the stream.
PREVIEW_DETECTOR_THREADS = 2
PREVIEW_QUAD_DECIMATE = 2.0


class CameraControlsMixin:
    """Read, change and save a camera's V4L2 controls; preview AprilTag detections."""

    _preview_detector: Any = None
    _preview_detector_lock = threading.Lock()

    def _running_camera(self, camera_bus_id: str):
        """Return the running camera for a bus ID, or None."""
        if self.camera_manager is None:
            return None
        return self.camera_manager.get_camera_by_bus_id(str(camera_bus_id))

    def _controls_payload(self, camera_bus_id: str, camera) -> dict:
        """Describe the camera's controls, with the saved values next to them."""
        config = self._resolve_camera_config(camera_bus_id)
        return {
            "camera_bus_id": str(camera_bus_id),
            "controls": [control.to_dict() for control in camera.list_controls()],
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
        values = {
            control.key: control.value
            for control in camera.list_controls()
            if not control.read_only and control.value is not None
        }
        try:
            config.save_camera_controls(values)
        except OSError as error:
            self.log(f"Failed saving camera controls for {camera_bus_id}: {error}")
            return {"error": "Failed to save camera settings"}, 500
        return self._controls_payload(camera_bus_id, camera), 200

    def _apriltag_preview_detector(self):
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
    def _draw_apriltags(frame: np.ndarray, detections, detect_ms: float) -> np.ndarray:
        """Draw detected tag outlines, IDs and the detection time on a copy of the frame."""
        output = (
            cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame.copy()
        )
        for detection in detections:
            corners = np.rint(np.asarray(detection.corners)).astype(np.int32)
            cv2.polylines(output, [corners], True, (0, 255, 0), 2)
            # The first corner marks the tag's orientation.
            cv2.circle(output, tuple(int(v) for v in corners[0]), 5, (0, 0, 255), -1)
            center = corners.mean(axis=0).astype(int)
            cv2.putText(
                output,
                str(detection.tag_id),
                (int(center[0]) - 10, int(center[1]) + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
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
