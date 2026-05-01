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
        auto_detect = str(source.get("auto_detect", "false")).lower() == "true"
        if cols <= 2 or rows <= 2:
            raise ValueError("Checkerboard rows and columns must be greater than 2")
        if square_size <= 0:
            raise ValueError("Square size must be greater than 0")
        return cols, rows, square_size, auto_detect

    def _find_adaptive_checkerboard(
        self,
        frame: np.ndarray,
        cols: int,
        rows: int,
        auto_detect: bool,
        previous_size: tuple[int, int] | None = None,
    ):
        if not auto_detect:
            found, corners, image_size = self._find_checkerboard(frame, cols, rows)
            return found, corners, image_size, cols, rows

        found, corners, image_size, detected_cols, detected_rows = self._find_checkerboard_auto(
            frame, previous_size
        )
        if found:
            return found, corners, image_size, detected_cols, detected_rows
        return False, None, frame.shape[1::-1], cols, rows

    def _find_checkerboard_size_candidates_geometric(
        self, gray: np.ndarray, min_size: int = 5, max_size: int = 20
    ) -> list[tuple[int, int]]:
        detector_width = 640
        scale = min(1.0, detector_width / max(gray.shape[1], 1))
        small = cv2.resize(gray, None, fx=scale, fy=scale) if scale < 1.0 else gray
        small = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(small)
        corners = cv2.goodFeaturesToTrack(
            small,
            maxCorners=800,
            qualityLevel=0.01,
            minDistance=4,
            blockSize=5,
            useHarrisDetector=True,
            k=0.04,
        )
        if corners is None or len(corners) < min_size * min_size:
            return []

        points = corners.reshape(-1, 2).astype(np.float32)
        center = points.mean(axis=0)
        _, _, vt = np.linalg.svd(points - center, full_matrices=False)
        axes = vt[:2]
        projected = (points - center) @ axes.T

        def cluster_count(values: np.ndarray) -> int | None:
            values = np.sort(values)
            gaps = np.diff(values)
            positive_gaps = gaps[gaps > 2]
            if len(positive_gaps) == 0:
                return None
            step = float(np.median(positive_gaps))
            if step <= 2:
                return None
            bins: dict[int, int] = {}
            origin = float(values[0])
            for value in values:
                index = int(round((float(value) - origin) / step))
                if abs((origin + index * step) - float(value)) <= step * 0.35:
                    bins[index] = bins.get(index, 0) + 1
            occupied = [index for index, count in bins.items() if count >= 2]
            if not occupied:
                return None
            return max(occupied) - min(occupied) + 1

        a = cluster_count(projected[:, 0])
        b = cluster_count(projected[:, 1])
        if a is None or b is None:
            return []

        candidates: list[tuple[int, int]] = []
        for cols, rows in ((a, b), (b, a)):
            if min_size <= cols <= max_size and min_size <= rows <= max_size:
                candidates.append((int(cols), int(rows)))
                # Include one-off neighbors because Harris clustering can include
                # or miss an edge line; exact OpenCV verification filters them.
                for dc, dr in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    candidate = (int(cols + dc), int(rows + dr))
                    if (
                        min_size <= candidate[0] <= max_size
                        and min_size <= candidate[1] <= max_size
                        and candidate not in candidates
                    ):
                        candidates.append(candidate)
        return sorted(candidates, key=lambda size: size[0] * size[1], reverse=True)

    def _find_checkerboard_size_candidates_sb(
        self, gray: np.ndarray, max_size: int = 20
    ) -> list[tuple[int, int]]:
        if not hasattr(cv2, "findChessboardCornersSBWithMeta"):
            return []
        flags = (
            cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_LARGER
            | cv2.CALIB_CB_EXHAUSTIVE
        )
        candidates: set[tuple[int, int]] = set()
        # CALIB_CB_LARGER can under-report from a tiny 3x3 seed on some images.
        # Probe a few progressively larger seeds, then verify only the resulting
        # likely sizes with the exact SB detector.
        seed_sizes = [(3, 3), (5, 5), (7, 7), (9, 6), (6, 9), (7, 10), (10, 7)]
        for seed_size in seed_sizes:
            try:
                ok, _, meta = cv2.findChessboardCornersSBWithMeta(
                    gray, seed_size, flags
                )
            except cv2.error:
                continue
            if not ok or meta is None:
                continue
            rows, cols = meta.shape[:2]
            if 2 < cols <= max_size and 2 < rows <= max_size:
                candidates.add((cols, rows))
        return sorted(candidates, key=lambda size: size[0] * size[1], reverse=True)

    def _find_checkerboard_auto(
        self, frame: np.ndarray, previous_size: tuple[int, int] | None = None
    ):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]
        candidates = self._find_checkerboard_size_candidates_geometric(gray)
        for candidate in self._find_checkerboard_size_candidates_sb(gray):
            if candidate not in candidates:
                candidates.append(candidate)

        expanded_candidates: list[tuple[int, int]] = []
        for candidate in sorted(candidates, key=lambda size: size[0] * size[1], reverse=True):
            if candidate not in expanded_candidates:
                expanded_candidates.append(candidate)
            swapped = (candidate[1], candidate[0])
            if swapped not in expanded_candidates:
                expanded_candidates.append(swapped)
        if previous_size:
            previous = (int(previous_size[0]), int(previous_size[1]))
            if previous not in expanded_candidates:
                expanded_candidates.append(previous)

        for candidate_cols, candidate_rows in expanded_candidates[:12]:
            found, corners = self._find_checkerboard_sb_gray(gray, candidate_cols, candidate_rows)
            if found:
                return True, corners, image_size, candidate_cols, candidate_rows
        return False, None, image_size, None, None

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
        auto_detect: bool = False,
        previous_size: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, tuple[int, int] | None]:
        output = frame.copy()
        found, corners, _, detected_cols, detected_rows = self._find_adaptive_checkerboard(
            frame, cols, rows, auto_detect, previous_size
        )
        draw_size = (detected_cols, detected_rows)
        checking_size = draw_size
        if found:
            cv2.drawChessboardCorners(output, draw_size, corners, found)

        if auto_detect:
            detection_text = (
                f"Checkerboard {draw_size[0]}x{draw_size[1]} detected"
                if found
                else f"Auto checking {checking_size[0]}x{checking_size[1]} | No checkerboard"
            )
        else:
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
        self, camera_bus_id: str, cols: int, rows: int, auto_detect: bool
    ) -> Generator[bytes, Any, Any]:
        resolved_camera_name = self._camera_name_for_bus_id(camera_bus_id)
        while True:
            start = time.time()
            with self._calibration_lock():
                session = self._calibration_sessions().setdefault(str(camera_bus_id), {"frames": []})
                count = len(session.get("frames", []))
                previous_size = session.get("auto_detect_size")
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
                        frame, detected_size = self._draw_detection(
                            frame, cols, rows, count, auto_detect, previous_size
                        )
                        if auto_detect and detected_size is not None:
                            with self._calibration_lock():
                                session = self._calibration_sessions().setdefault(
                                    str(camera_bus_id), {"frames": []}
                                )
                                session["auto_detect_size"] = detected_size
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
            cols, rows, _, auto_detect = self._checkerboard_params_from_request()
        except ValueError:
            cols, rows, auto_detect = 9, 6, False
        return Response(
            self._calibration_feed_generator(camera_bus_id, cols, rows, auto_detect),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    def auto_detect_calibration_size(self, camera_bus_id: str) -> tuple[dict, int]:
        data = request.get_json(silent=True) if request.method == "POST" else {}
        target_count = int(data.get("frames", 5)) if isinstance(data, dict) else 5
        target_count = max(1, min(target_count, 10))
        last_size = None
        stable_count = 0
        attempts = 0
        max_attempts = max(target_count * 20, 60)
        while attempts < max_attempts:
            attempts += 1
            frame = self._latest_camera_frame(camera_bus_id)
            if frame is None:
                return {"error": "No frame available for selected camera"}, 404
            found, _, _, detected_cols, detected_rows = self._find_checkerboard_auto(frame)
            if found and detected_cols and detected_rows:
                detected_size = (int(detected_cols), int(detected_rows))
                if detected_size == last_size:
                    stable_count += 1
                else:
                    last_size = detected_size
                    stable_count = 1
                if stable_count >= target_count:
                    with self._calibration_lock():
                        session = self._calibration_sessions().setdefault(
                            str(camera_bus_id), {"frames": []}
                        )
                        session["auto_detect_size"] = detected_size
                    return {
                        "success": True,
                        "cols": detected_size[0],
                        "rows": detected_size[1],
                        "stable_count": stable_count,
                        "attempts": attempts,
                    }, 200
            else:
                last_size = None
                stable_count = 0
            time.sleep(max(1 / VIEW_STREAM_FPS, 0.02))
        return {"error": "Could not detect a stable checkerboard size"}, 400

    def capture_calibration_frame(self, camera_bus_id: str) -> tuple[dict, int]:
        try:
            cols, rows, square_size, auto_detect = self._checkerboard_params_from_request()
        except ValueError as error:
            return {"error": str(error)}, 400
        frame = self._latest_camera_frame(camera_bus_id)
        if frame is None:
            return {"error": "No frame available for selected camera"}, 404
        previous_size = None
        if auto_detect:
            with self._calibration_lock():
                previous_size = (
                    self._calibration_sessions()
                    .get(str(camera_bus_id), {})
                    .get("auto_detect_size")
                )
        found, corners, _, detected_cols, detected_rows = self._find_adaptive_checkerboard(
            frame, cols, rows, auto_detect, previous_size
        )
        if auto_detect and found:
            cols, rows = detected_cols, detected_rows
        if not found or corners is None:
            return {"error": "Checkerboard was not detected"}, 400
        with self._calibration_lock():
            session = self._calibration_sessions().setdefault(
                str(camera_bus_id), {"frames": []}
            )
            session.update({"cols": cols, "rows": rows, "square_size": square_size})
            if auto_detect:
                session["auto_detect_size"] = (cols, rows)
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
            cols, rows, _, auto_detect = self._checkerboard_params_from_request()
        except ValueError:
            cols, rows, auto_detect = 9, 6, False
        with self._calibration_lock():
            frames = (
                self._calibration_sessions()
                .get(str(camera_bus_id), {})
                .get("frames", [])
            )
            if frame_index < 0 or frame_index >= len(frames):
                return {"error": "Frame index out of range"}, 404
            frame = frames[frame_index]["frame"].copy()
        output, _ = self._draw_detection(frame, cols, rows, len(frames), auto_detect=auto_detect)
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
            cols, rows, square_size, auto_detect = self._checkerboard_params_from_request()
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

        if auto_detect:
            first = frames[0]
            if isinstance(first, dict) and first.get("cols") and first.get("rows"):
                cols, rows = int(first["cols"]), int(first["rows"])

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
