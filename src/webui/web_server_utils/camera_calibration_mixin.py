from __future__ import annotations

import os
import threading
import time
from typing import Any, Generator

import cv2
import numpy as np
from flask import Response, request

from src.webui.web_server_utils.constants import (
    VIEW_STREAM_FPS,
    no_image_jpeg_bytes,
)


class CameraCalibrationMixin:
    """Simple in-memory checkerboard camera calibration support."""

    def _calibration_sessions(self) -> dict[str, dict[str, Any]]:
        if not hasattr(self, "_camera_calibration_sessions"):
            self._camera_calibration_sessions = {}
            self._camera_calibration_lock = threading.Lock()
        return self._camera_calibration_sessions

    def _calibration_lock(self) -> threading.Lock:
        self._calibration_sessions()
        return self._camera_calibration_lock

    def _camera_name_for_bus_id(self, camera_bus_id: str) -> str | None:
        requested_bus_id = str(camera_bus_id)
        for camera_name, info in self.available_cameras.items():
            if camera_name == requested_bus_id:
                return camera_name

            if isinstance(info, dict):
                candidate_stream_name = info.get("name")
                if (
                    candidate_stream_name is not None
                    and str(candidate_stream_name) == requested_bus_id
                ):
                    return camera_name

                # Do not use `or` here: numeric camera IDs/bus IDs can be 0,
                # and 0 is a valid camera identifier.
                candidate_bus_id = info.get("bus_id")
                if (
                    candidate_bus_id is not None
                    and str(candidate_bus_id) == requested_bus_id
                ):
                    return camera_name

                candidate_id = info.get("id")
                if candidate_id is not None and str(candidate_id) == requested_bus_id:
                    return camera_name
            elif str(info) == requested_bus_id:
                return camera_name
        return None

    def _latest_camera_frame(self, camera_bus_id: str) -> np.ndarray | None:
        camera_name = self._camera_name_for_bus_id(camera_bus_id)
        if camera_name is None:
            return None
        lock = self.frame_locks.get(camera_name)
        if lock is None:
            return None
        with lock:
            frame = self.frame_list.get(camera_name)
            return None if frame is None else frame.copy()

    def _checkerboard_params_from_request(self) -> tuple[int, int, float, bool]:
        data = request.get_json(silent=True) if request.method == "POST" else None
        source = data if isinstance(data, dict) else request.args
        cols = int(source.get("cols", 9))
        rows = int(source.get("rows", 6))
        square_size = float(source.get("square_size", 0.025))
        if cols <= 2 or rows <= 2:
            raise ValueError("Checkerboard rows and columns must be greater than 2")
        if square_size <= 0:
            raise ValueError("Square size must be greater than 0")
        return cols, rows, square_size, False

    def _live_view_resolution_from_request(self) -> tuple[int | None, int | None]:
        width = int(request.args.get("live_width", 0) or 0)
        height = int(request.args.get("live_height", 0) or 0)
        if width <= 0 or height <= 0:
            return None, None
        return max(160, min(width, 4096)), max(120, min(height, 2160))

    def _resize_for_live_view(
        self, frame: np.ndarray, live_width: int | None, live_height: int | None
    ) -> np.ndarray:
        if live_width is None or live_height is None:
            return frame
        height, width = frame.shape[:2]
        if width <= 0 or height <= 0:
            return frame
        scale = min(live_width / width, live_height / height)
        if scale >= 0.999:
            return frame
        target_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

    def _find_checkerboard_sb_gray(self, gray: np.ndarray, cols: int, rows: int):
        if cols <= 2 or rows <= 2 or not hasattr(cv2, "findChessboardCornersSB"):
            return False, None
        try:
            return cv2.findChessboardCornersSB(
                gray, (cols, rows), flags=cv2.CALIB_CB_NORMALIZE_IMAGE
            )
        except cv2.error:
            return False, None

    def _find_checkerboard(self, frame: np.ndarray, cols: int, rows: int):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = self._find_checkerboard_sb_gray(gray, cols, rows)
        if not found:
            found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
            if found:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return found, corners, gray.shape[::-1]

    def _draw_detection(
        self,
        frame: np.ndarray,
        cols: int,
        rows: int,
        count: int,
    ) -> tuple[np.ndarray, tuple[int, int] | None]:
        output = frame.copy()
        found, corners, _, = self._find_checkerboard(frame, cols, rows)
        draw_size = (cols, rows)
        if found:
            cv2.drawChessboardCorners(output, draw_size, corners, found)

        detection_text = f"Checkerboard {draw_size[0]}x{draw_size[1]} detected" if found else "No checkerboard"
        text = f"{detection_text} | Frames: {count}/10"
        cv2.putText(
            output,
            text,
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if found else (0, 0, 255),
            2,
        )
        return output, draw_size if found else None

    def _calibration_feed_generator(
        self,
        camera_bus_id: str,
        cols: int,
        rows: int,
        live_width: int | None,
        live_height: int | None,
    ) -> Generator[bytes, Any, Any]:
        resolved_camera_name = self._camera_name_for_bus_id(camera_bus_id)
        while True:
            start = time.time()
            with self._calibration_lock():
                session = self._calibration_sessions().setdefault(str(camera_bus_id), {"frames": []})
                count = len(session.get("frames", []))
            if resolved_camera_name is None:
                jpeg = no_image_jpeg_bytes
            else:
                lock = self.frame_locks.get(resolved_camera_name)
                if lock is None:
                    jpeg = no_image_jpeg_bytes
                else:
                    with lock:
                        source_frame = self.frame_list.get(resolved_camera_name)
                        frame = None if source_frame is None else source_frame.copy()

                    if frame is None:
                        jpeg = no_image_jpeg_bytes
                    else:
                        frame = self._resize_for_live_view(frame, live_width, live_height)
                        frame, _ = self._draw_detection(frame, cols, rows, count)
                        ok, enc = cv2.imencode(
                            ".jpg",
                            frame,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 95],
                        )
                        jpeg = enc.tobytes() if ok else no_image_jpeg_bytes
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            time.sleep(max((1 / VIEW_STREAM_FPS) - (time.time() - start), 0))

    def calibration_feed(self, camera_bus_id: str) -> Response:
        try:
            cols, rows, _, _ = self._checkerboard_params_from_request()
            live_width, live_height = self._live_view_resolution_from_request()
        except ValueError:
            cols, rows = 9, 6
            live_width, live_height = None, None
        return Response(
            self._calibration_feed_generator(
                camera_bus_id, cols, rows, live_width, live_height
            ),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    def capture_calibration_frame(self, camera_bus_id: str) -> tuple[dict, int]:
        try:
            cols, rows, square_size, _ = self._checkerboard_params_from_request()
        except ValueError as error:
            return {"error": str(error)}, 400
        frame = self._latest_camera_frame(camera_bus_id)
        if frame is None:
            return {"error": "No frame available for selected camera"}, 404
        found, corners, _ = self._find_checkerboard(frame, cols, rows)
        if not found or corners is None:
            return {"error": "Checkerboard was not detected"}, 400
        with self._calibration_lock():
            session = self._calibration_sessions().setdefault(
                str(camera_bus_id), {"frames": []}
            )
            session.update({"cols": cols, "rows": rows, "square_size": square_size})
            session["frames"].append({"frame": frame, "corners": corners, "cols": cols, "rows": rows})
            count = len(session["frames"])
        return {"success": True, "frame_count": count, "frame_index": count - 1, "cols": cols, "rows": rows, "recommended_count": 10}, 200

    def get_calibration_frames(self, camera_bus_id: str) -> tuple[dict, int]:
        with self._calibration_lock():
            frames = list(
                self._calibration_sessions()
                .get(str(camera_bus_id), {})
                .get("frames", [])
            )
        response_frames = []
        for i, item in enumerate(frames):
            frame = item.get("frame")
            image_size = None
            if frame is not None:
                image_size = {"width": int(frame.shape[1]), "height": int(frame.shape[0])}
            corners = item.get("corners")
            response_frames.append(
                {
                    "index": i,
                    "image_size": image_size,
                    "corners": corners.reshape(-1, 2).astype(float).tolist() if corners is not None else [],
                }
            )
        count = len(response_frames)
        return {
            "frames": response_frames,
            "frame_count": count,
            "recommended_count": 10,
        }, 200

    def get_calibration_frame_image(
        self, camera_bus_id: str, frame_index: int
    ) -> Response | tuple[dict, int]:
        try:
            cols, rows, _, _ = self._checkerboard_params_from_request()
        except ValueError:
            cols, rows = 9, 6
        with self._calibration_lock():
            frames = (
                self._calibration_sessions()
                .get(str(camera_bus_id), {})
                .get("frames", [])
            )
            if frame_index < 0 or frame_index >= len(frames):
                return {"error": "Frame index out of range"}, 404
            frame = frames[frame_index]["frame"].copy()
        output, _ = self._draw_detection(frame, cols, rows, len(frames))
        ok, enc = cv2.imencode(".jpg", output, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return Response(enc.tobytes() if ok else no_image_jpeg_bytes, mimetype="image/jpeg")

    def delete_calibration_frame(
        self, camera_bus_id: str, frame_index: int
    ) -> tuple[dict, int]:
        with self._calibration_lock():
            frames = (
                self._calibration_sessions()
                .get(str(camera_bus_id), {})
                .get("frames", [])
            )
            if frame_index < 0 or frame_index >= len(frames):
                return {"error": "Frame index out of range"}, 404
            del frames[frame_index]
            count = len(frames)
        return {"success": True, "frame_count": count}, 200

    def reset_calibration_frames(self, camera_bus_id: str) -> tuple[dict, int]:
        with self._calibration_lock():
            self._calibration_sessions().pop(str(camera_bus_id), None)
        return {"success": True, "frame_count": 0}, 200

    def run_camera_calibration(self, camera_bus_id: str) -> tuple[dict, int]:
        try:
            cols, rows, square_size, _ = self._checkerboard_params_from_request()
        except ValueError as error:
            return {"error": str(error)}, 400
        config = self._resolve_camera_config(camera_bus_id)
        if config is None:
            return {"error": "Camera config registry unavailable"}, 503
        with self._calibration_lock():
            frames = list(
                self._calibration_sessions()
                .get(str(camera_bus_id), {})
                .get("frames", [])
            )
        if len(frames) < 3:
            return {"error": "At least 3 captured frames are required"}, 400

        objp = np.zeros((rows * cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size
        objpoints, imgpoints, image_size = [], [], None
        for item in frames:
            frame = item["frame"] if isinstance(item, dict) else item
            if isinstance(item, dict) and (item.get("cols") != cols or item.get("rows") != rows):
                continue
            found, corners, size = self._find_checkerboard(frame, cols, rows)
            if found:
                objpoints.append(objp)
                imgpoints.append(corners)
                image_size = size
        if len(objpoints) < 3 or image_size is None:
            return {"error": "At least 3 valid checkerboard frames are required"}, 400
        rms, camera_matrix, dist, _, _ = cv2.calibrateCamera(
            objpoints, imgpoints, image_size, None, None
        )
        target_path = config.intrinsics_path or self._default_intrinsics_path(
            camera_bus_id
        )
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        import json

        with open(target_path, "w") as f:
            json.dump(
                {
                    "camera_matrix": camera_matrix.tolist(),
                    "distortion_coefficients": dist.reshape(-1).tolist(),
                    "reprojection_error": float(rms),
                    "image_size": list(image_size),
                    "checkerboard": {
                        "cols": cols,
                        "rows": rows,
                        "square_size": square_size,
                    },
                },
                f,
                indent=4,
            )
        config.intrinsics_path = target_path
        return {
            "success": True,
            "frame_count": len(objpoints),
            "reprojection_error": float(rms),
            "intrinsics_path": target_path,
        }, 200
