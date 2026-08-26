from __future__ import annotations

import json
import os
import threading
import time
from collections import Counter
from typing import Any, Generator

import cv2
import numpy as np
from flask import Response, request

from src.webui.web_server_utils.constants import (
    VIEW_STREAM_FPS,
    no_image_jpeg_bytes,
)


class CameraCalibrationMixin:
    """Simple in-memory ChArUco camera calibration support."""

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

    def _charuco_params_from_request(self) -> tuple[int, int, float, float, int]:
        data = request.get_json(silent=True) if request.method == "POST" else None
        source = data if isinstance(data, dict) else request.args
        squares_x = int(source.get("squares_x", source.get("cols", 11)))
        squares_y = int(source.get("squares_y", source.get("rows", 8)))
        square_size = float(source.get("square_size", 0.015))
        marker_size = float(source.get("marker_size", 0.011))
        dictionary_id = int(source.get("dictionary_id", cv2.aruco.DICT_4X4_50))
        if squares_x <= 1 or squares_y <= 1:
            raise ValueError("ChArUco square counts must be greater than 1")
        if square_size <= 0 or marker_size <= 0:
            raise ValueError("Square and marker sizes must be greater than 0")
        if marker_size >= square_size:
            raise ValueError("Marker size must be smaller than square size")
        return squares_x, squares_y, square_size, marker_size, dictionary_id

    def _charuco_dictionary(self, dictionary_id: int):
        if not hasattr(cv2, "aruco"):
            raise ValueError(
                "OpenCV ArUco support is unavailable; install opencv-contrib-python"
            )
        return cv2.aruco.getPredefinedDictionary(dictionary_id)

    def _charuco_board(
        self,
        squares_x: int,
        squares_y: int,
        square_size: float,
        marker_size: float,
        dictionary_id: int,
        legacy_pattern: bool = False,
    ):
        dictionary = self._charuco_dictionary(dictionary_id)
        board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y), square_size, marker_size, dictionary
        )
        if hasattr(board, "setLegacyPattern"):
            board.setLegacyPattern(legacy_pattern)
        return board

    def _charuco_board_candidates(self, params: tuple[int, int, float, float, int]):
        squares_x, squares_y, square_size, marker_size, dictionary_id = params
        candidates = []
        for candidate_x, candidate_y in (
            (squares_x, squares_y),
            (squares_y, squares_x),
        ):
            for legacy_pattern in (False, True):
                board = self._charuco_board(
                    candidate_x,
                    candidate_y,
                    square_size,
                    marker_size,
                    dictionary_id,
                    legacy_pattern,
                )
                candidate_params = (
                    candidate_x,
                    candidate_y,
                    square_size,
                    marker_size,
                    dictionary_id,
                    legacy_pattern,
                )
                if candidate_params not in [existing[0] for existing in candidates]:
                    candidates.append((candidate_params, board))
        return candidates

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

    def _grayscale_frame(self, frame: np.ndarray) -> np.ndarray:
        """Return a single-channel image from gray, BGR, or BGRA camera data."""
        if frame.ndim == 2:
            return frame
        if frame.ndim != 3:
            raise ValueError(f"Unsupported camera frame shape: {frame.shape}")

        channel_count = frame.shape[2]
        if channel_count == 1:
            return frame[:, :, 0]
        if channel_count == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if channel_count == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        raise ValueError(f"Unsupported camera frame shape: {frame.shape}")

    def _detector_parameters(self):
        """Build detector parameters tuned for calibration coverage.

        Two defaults work against a calibration workflow. ``minDistanceToBorder``
        discards markers near the image edge, which is exactly where corner
        coverage is needed to constrain the distortion model, and
        ``cornerRefinementMethod`` is off, so marker corners land on contour
        pixels and every ChArUco corner interpolated from them inherits that
        error.
        """
        parameters = cv2.aruco.DetectorParameters()
        parameters.minDistanceToBorder = 0
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        return parameters

    def _find_charuco(self, frame: np.ndarray, board: Any):
        gray = self._grayscale_frame(frame)
        dictionary = board.getDictionary()
        parameters = self._detector_parameters()
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, dictionary, parameters=parameters
        )
        if ids is None or len(ids) == 0:
            return False, None, None, corners, ids, gray.shape[::-1]
        refined_corners, refined_ids = corners, ids
        try:
            refined_corners, refined_ids, _, _ = cv2.aruco.refineDetectedMarkers(
                gray, board, corners, ids, rejected, parameters=parameters
            )
        except cv2.error:
            # Some third-party/generated boards have marker placement that does
            # not match OpenCV's board object exactly. Keep raw marker detections
            # and still try ChArUco interpolation below.
            pass
        # ``minMarkers=1`` keeps corners that only one decoded marker borders.
        # The default of 2 silently drops the outer ring of board corners and
        # any corner whose neighbour was clipped at the frame edge.
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            refined_corners, refined_ids, gray, board, minMarkers=1
        )
        found = (
            charuco_ids is not None
            and charuco_corners is not None
            and len(charuco_ids) >= 4
        )
        return (
            found,
            charuco_corners,
            charuco_ids,
            refined_corners,
            refined_ids,
            gray.shape[::-1],
        )

    # A correct mapping fits the board plane to well under a pixel plus lens
    # distortion; a wrong one misses by hundreds. Anything between is unusable.
    _MAX_PLANAR_FIT_ERROR_PX = 5.0

    def _planar_fit_error(self, board: Any, charuco_corners, charuco_ids) -> float:
        """Return the RMS residual of the best homography onto a board layout.

        A ChArUco board is planar, so a correct corner-to-object-point mapping is
        explained by a single homography to within lens distortion, while a
        scrambled mapping leaves a residual of hundreds of pixels. It needs no
        intrinsics, which is the point: it screens board layouts before any
        calibration has run.

        It cannot separate the legacy from the current marker layout. That
        mismatch shifts every corner by a whole square in the same direction, and
        a homography absorbs a uniform shift as translation, so both layouts fit
        equally well. Corner count is what separates those two; this is a
        validity check, not the tiebreak.
        """
        if charuco_ids is None or charuco_corners is None or len(charuco_ids) < 4:
            return float("inf")
        object_points = board.getChessboardCorners()[charuco_ids.reshape(-1)]
        image_points = charuco_corners.reshape(-1, 2).astype(np.float64)
        homography, _ = cv2.findHomography(object_points[:, :2], image_points)
        if homography is None:
            return float("inf")
        projected = cv2.perspectiveTransform(
            object_points[:, :2].reshape(-1, 1, 2), homography
        ).reshape(-1, 2)
        return float(np.sqrt(np.mean(np.sum((projected - image_points) ** 2, axis=1))))

    def _find_best_charuco(
        self, frame: np.ndarray, params: tuple[int, int, float, float, int]
    ):
        candidates = []
        for candidate_params, board in self._charuco_board_candidates(params):
            result = self._find_charuco(frame, board)
            charuco_ids = result[2]
            charuco_count = 0 if charuco_ids is None else len(charuco_ids)
            error = self._planar_fit_error(board, result[1], charuco_ids)
            candidates.append((candidate_params, board, result, charuco_count, error))
        if not candidates:
            board = self._charuco_board(*params)
            return ((*params, False), board, self._find_charuco(frame, board))

        # Drop layouts the board plane cannot explain at all before comparing
        # corner counts, so a layout that happens to interpolate many corners
        # from a geometrically inconsistent mapping cannot win on count alone.
        consistent = [
            candidate
            for candidate in candidates
            if candidate[4] <= self._MAX_PLANAR_FIT_ERROR_PX
        ] or candidates
        best = max(consistent, key=lambda candidate: (candidate[3], -candidate[4]))
        return best[0], best[1], best[2]

    def _draw_detection(
        self, frame: np.ndarray, params: tuple[int, int, float, float, int], count: int
    ) -> tuple[np.ndarray, bool]:
        gray = self._grayscale_frame(frame)
        output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        best_params, _, result = self._find_best_charuco(frame, params)
        found, charuco_corners, charuco_ids, marker_corners, marker_ids, _ = result
        if marker_ids is not None and len(marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(output, marker_corners, marker_ids)
        if (
            charuco_ids is not None
            and charuco_corners is not None
            and len(charuco_ids) > 0
        ):
            cv2.aruco.drawDetectedCornersCharuco(output, charuco_corners, charuco_ids)
        corner_count = 0 if charuco_ids is None else len(charuco_ids)
        legacy_suffix = " legacy" if best_params[5] else ""
        text = f"ChArUco corners: {corner_count} ({best_params[0]}x{best_params[1]}{legacy_suffix}) | Frames: {count}/10"
        cv2.putText(
            output,
            text,
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if found else (0, 0, 255),
            2,
        )
        return output, found

    def _calibration_feed_generator(
        self,
        camera_bus_id: str,
        params: tuple[int, int, float, float, int],
        live_width: int | None,
        live_height: int | None,
    ) -> Generator[bytes, Any, Any]:
        resolved_camera_name = self._camera_name_for_bus_id(camera_bus_id)
        while True:
            start = time.time()
            with self._calibration_lock():
                session = self._calibration_sessions().setdefault(
                    str(camera_bus_id), {"frames": []}
                )
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
                        frame = self._resize_for_live_view(
                            frame, live_width, live_height
                        )
                        frame, _ = self._draw_detection(frame, params, count)
                        ok, enc = cv2.imencode(
                            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95]
                        )
                        jpeg = enc.tobytes() if ok else no_image_jpeg_bytes
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            time.sleep(max((1 / VIEW_STREAM_FPS) - (time.time() - start), 0))

    def calibration_feed(self, camera_bus_id: str) -> Response:
        try:
            params = self._charuco_params_from_request()
            live_width, live_height = self._live_view_resolution_from_request()
        except ValueError:
            params = (11, 8, 0.015, 0.011, cv2.aruco.DICT_4X4_50)
            live_width, live_height = None, None
        return Response(
            self._calibration_feed_generator(
                camera_bus_id, params, live_width, live_height
            ),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    def _load_distortion_intrinsics(
        self, camera_bus_id: str
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, int] | None]:
        """Load the selected camera's matrix, coefficients, and calibration size."""
        config = self._resolve_camera_config(camera_bus_id)
        intrinsics_path = None if config is None else config.intrinsics_path
        if not intrinsics_path or not os.path.exists(intrinsics_path):
            raise ValueError("No intrinsics file is available for the selected camera")

        try:
            with open(intrinsics_path, encoding="utf-8") as intrinsics_file:
                payload = json.load(intrinsics_file)
            camera_matrix = np.asarray(payload["camera_matrix"], dtype=np.float64)
            coefficients = np.asarray(
                payload["distortion_coefficients"], dtype=np.float64
            ).reshape(-1)
            raw_size = payload.get("image_size", payload.get("img_size"))
            image_size = (
                (int(raw_size[0]), int(raw_size[1]))
                if isinstance(raw_size, list) and len(raw_size) >= 2
                else None
            )
            if camera_matrix.shape != (3, 3) or coefficients.size < 4:
                raise ValueError
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                "The selected camera's intrinsics file is invalid"
            ) from error
        return camera_matrix, coefficients, image_size

    def _scaled_camera_matrix(
        self,
        camera_matrix: np.ndarray,
        calibration_size: tuple[int, int] | None,
        frame_size: tuple[int, int],
    ) -> np.ndarray:
        """Scale a calibration matrix when the live frame uses another size."""
        if not calibration_size or calibration_size == frame_size:
            return camera_matrix
        scale_x = frame_size[0] / calibration_size[0]
        scale_y = frame_size[1] / calibration_size[1]
        scaled = camera_matrix.copy()
        scaled[0, 0] *= scale_x
        scaled[0, 2] *= scale_x
        scaled[1, 1] *= scale_y
        scaled[1, 2] *= scale_y
        return scaled

    def _draw_distortion_grid(
        self, frame: np.ndarray, camera_matrix: np.ndarray, coefficients: np.ndarray
    ) -> np.ndarray:
        """Overlay an ideal grid warped by the current distortion model."""
        height, width = frame.shape[:2]
        output = frame.copy()
        grid_steps = 10
        sample_count = 80

        def distort_points(points: np.ndarray) -> np.ndarray:
            normalized = np.empty((len(points), 3), dtype=np.float64)
            normalized[:, 0] = (points[:, 0] - camera_matrix[0, 2]) / camera_matrix[
                0, 0
            ]
            normalized[:, 1] = (points[:, 1] - camera_matrix[1, 2]) / camera_matrix[
                1, 1
            ]
            normalized[:, 2] = 1.0
            projected, _ = cv2.projectPoints(
                normalized,
                np.zeros(3),
                np.zeros(3),
                camera_matrix,
                coefficients,
            )
            return np.rint(projected.reshape(-1, 2)).astype(np.int32)

        for index in range(1, grid_steps):
            x = width * index / grid_steps
            vertical = np.column_stack(
                (np.full(sample_count, x), np.linspace(0, height - 1, sample_count))
            )
            y = height * index / grid_steps
            horizontal = np.column_stack(
                (np.linspace(0, width - 1, sample_count), np.full(sample_count, y))
            )
            cv2.polylines(
                output, [distort_points(vertical)], False, (0, 255, 255), 1, cv2.LINE_AA
            )
            cv2.polylines(
                output,
                [distort_points(horizontal)],
                False,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return output

    def _draw_straight_grid(self, frame: np.ndarray) -> np.ndarray:
        """Overlay a straight reference grid on an undistorted frame."""
        height, width = frame.shape[:2]
        output = frame.copy()
        for index in range(1, 10):
            x = round(width * index / 10)
            y = round(height * index / 10)
            cv2.line(output, (x, 0), (x, height - 1), (0, 255, 255), 1, cv2.LINE_AA)
            cv2.line(output, (0, y), (width - 1, y), (0, 255, 255), 1, cv2.LINE_AA)
        return output

    def _undistort_frame(
        self, frame: np.ndarray, camera_matrix: np.ndarray, coefficients: np.ndarray
    ) -> np.ndarray:
        """Undistort a frame while retaining the camera's full field of view."""
        height, width = frame.shape[:2]
        output_matrix, _ = cv2.getOptimalNewCameraMatrix(
            camera_matrix,
            coefficients,
            (width, height),
            1.0,
            (width, height),
        )
        map_x, map_y = cv2.initUndistortRectifyMap(
            camera_matrix,
            coefficients,
            None,
            output_matrix,
            (width, height),
            cv2.CV_32FC1,
        )
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)

    def _distortion_feed_generator(
        self, camera_bus_id: str, view: str
    ) -> Generator[bytes, Any, Any]:
        """Stream either a rectified view or a distortion-grid diagnostic."""
        try:
            matrix, coefficients, calibration_size = self._load_distortion_intrinsics(
                camera_bus_id
            )
        except ValueError:
            matrix = coefficients = calibration_size = None

        while True:
            start = time.time()
            frame = self._latest_camera_frame(camera_bus_id)
            if frame is None or matrix is None or coefficients is None:
                jpeg = no_image_jpeg_bytes
            else:
                height, width = frame.shape[:2]
                active_matrix = self._scaled_camera_matrix(
                    matrix, calibration_size, (width, height)
                )
                if view == "undistorted":
                    output = self._undistort_frame(frame, active_matrix, coefficients)
                    output = self._draw_straight_grid(output)
                else:
                    output = self._draw_distortion_grid(
                        frame, active_matrix, coefficients
                    )
                ok, encoded = cv2.imencode(
                    ".jpg", output, [int(cv2.IMWRITE_JPEG_QUALITY), 95]
                )
                jpeg = encoded.tobytes() if ok else no_image_jpeg_bytes
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            time.sleep(max((1 / VIEW_STREAM_FPS) - (time.time() - start), 0))

    def distortion_feed(self, camera_bus_id: str) -> Response | tuple[dict, int]:
        """Return a live distortion diagnostic stream for one camera."""
        view = request.args.get("view", "distorted")
        if view not in {"distorted", "undistorted"}:
            return {"error": "View must be 'distorted' or 'undistorted'"}, 400
        try:
            self._load_distortion_intrinsics(camera_bus_id)
        except ValueError as error:
            return {"error": str(error)}, 404
        return Response(
            self._distortion_feed_generator(camera_bus_id, view),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    def capture_calibration_frame(self, camera_bus_id: str) -> tuple[dict, int]:
        try:
            params = self._charuco_params_from_request()
        except ValueError as error:
            return {"error": str(error)}, 400
        frame = self._latest_camera_frame(camera_bus_id)
        if frame is None:
            return {"error": "No frame available for selected camera"}, 404
        best_params, _, result = self._find_best_charuco(frame, params)
        found, charuco_corners, charuco_ids, _, marker_ids, _ = result
        if not found:
            marker_count = 0 if marker_ids is None else len(marker_ids)
            return {
                "error": f"ChArUco corners were not detected ({marker_count} markers found)"
            }, 400
        with self._calibration_lock():
            session = self._calibration_sessions().setdefault(
                str(camera_bus_id), {"frames": []}
            )
            session.update(
                {
                    "squares_x": best_params[0],
                    "squares_y": best_params[1],
                    "square_size": best_params[2],
                    "marker_size": best_params[3],
                    "dictionary_id": best_params[4],
                    "legacy_pattern": best_params[5],
                }
            )
            session["frames"].append(
                {
                    "frame": frame,
                    "charuco_corners": charuco_corners,
                    "charuco_ids": charuco_ids,
                    "params": best_params,
                }
            )
            count = len(session["frames"])
        return {
            "success": True,
            "frame_count": count,
            "frame_index": count - 1,
            "squares_x": best_params[0],
            "squares_y": best_params[1],
            "recommended_count": 10,
        }, 200

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
            image_size = (
                {"width": int(frame.shape[1]), "height": int(frame.shape[0])}
                if frame is not None
                else None
            )
            corners = item.get("charuco_corners")
            if corners is None:
                corners = item.get("corners")
            response_frames.append(
                {
                    "index": i,
                    "image_size": image_size,
                    "corners": corners.reshape(-1, 2).astype(float).tolist()
                    if corners is not None
                    else [],
                }
            )
        return {
            "frames": response_frames,
            "frame_count": len(response_frames),
            "recommended_count": 10,
        }, 200

    def get_calibration_frame_image(
        self, camera_bus_id: str, frame_index: int
    ) -> Response | tuple[dict, int]:
        try:
            params = self._charuco_params_from_request()
        except ValueError:
            params = (11, 8, 0.015, 0.011, cv2.aruco.DICT_4X4_50)
        with self._calibration_lock():
            frames = (
                self._calibration_sessions()
                .get(str(camera_bus_id), {})
                .get("frames", [])
            )
            if frame_index < 0 or frame_index >= len(frames):
                return {"error": "Frame index out of range"}, 404
            item = frames[frame_index]
            frame = item["frame"].copy()
            stored_params = item.get("params")
            if stored_params is not None:
                params = tuple(stored_params[:5])
        output, _ = self._draw_detection(frame, params, len(frames))
        ok, enc = cv2.imencode(".jpg", output, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return Response(
            enc.tobytes() if ok else no_image_jpeg_bytes, mimetype="image/jpeg"
        )

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
            request_params = self._charuco_params_from_request()
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

        # Take the layout the most frames agree on rather than whichever frame
        # happens to be first: a single frame that resolved to a different
        # layout should not disqualify every other capture.
        stored_counts = Counter(
            tuple(item["params"])
            for item in frames
            if isinstance(item, dict) and item.get("params") is not None
        )
        active_params = (
            stored_counts.most_common(1)[0][0]
            if stored_counts
            else (*request_params, False)
        )
        board = self._charuco_board(*active_params)

        all_corners, all_ids, image_size = [], [], None
        mismatched_indices, undetected_indices = [], []
        for index, item in enumerate(frames):
            frame = item["frame"] if isinstance(item, dict) else item
            if isinstance(item, dict) and item.get("params") != active_params:
                mismatched_indices.append(index)
                continue
            found, charuco_corners, charuco_ids, _, _, size = self._find_charuco(
                frame, board
            )
            if not found:
                undetected_indices.append(index)
                continue
            all_corners.append(charuco_corners)
            all_ids.append(charuco_ids)
            image_size = size

        warnings = []
        if mismatched_indices:
            warnings.append(
                f"Skipped {len(mismatched_indices)} frame(s) captured with a "
                f"different board layout than "
                f"{active_params[0]}x{active_params[1]}"
                f"{' legacy' if active_params[5] else ''}: "
                f"indices {mismatched_indices}"
            )
        if undetected_indices:
            warnings.append(
                f"Skipped {len(undetected_indices)} frame(s) with too few "
                f"ChArUco corners to use: indices {undetected_indices}"
            )
        for warning in warnings:
            self.log(f"Warning: camera {camera_bus_id} calibration: {warning}")

        if len(all_corners) < 3 or image_size is None:
            return {
                "error": "At least 3 valid ChArUco frames are required",
                "warnings": warnings,
            }, 400
        rms, camera_matrix, dist, _, _ = cv2.aruco.calibrateCameraCharuco(
            all_corners, all_ids, board, image_size, None, None
        )
        target_path = config.intrinsics_path or self._default_intrinsics_path(
            camera_bus_id
        )
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, "w") as f:
            json.dump(
                {
                    "camera_matrix": camera_matrix.tolist(),
                    "distortion_coefficients": dist.reshape(-1).tolist(),
                    "reprojection_error": float(rms),
                    "image_size": list(image_size),
                    "charuco": {
                        "squares_x": active_params[0],
                        "squares_y": active_params[1],
                        "square_size": active_params[2],
                        "marker_size": active_params[3],
                        "dictionary": "DICT_4X4_50",
                        "dictionary_id": active_params[4],
                        "legacy_pattern": active_params[5],
                    },
                },
                f,
                indent=4,
            )
        config.intrinsics_path = target_path
        return {
            "success": True,
            "frame_count": len(all_corners),
            "captured_frame_count": len(frames),
            "reprojection_error": float(rms),
            "intrinsics_path": target_path,
            "warnings": warnings,
        }, 200
