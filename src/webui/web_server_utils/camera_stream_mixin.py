from __future__ import annotations

import threading
import time
from typing import Any, Generator

import cv2
import numpy as np
from flask import Response, request

from src.webui.web_server_utils.constants import (
    VIEW_STREAM_FPS,
    VIEW_STREAM_JPEG_QUALITY,
    no_image,
    no_image_jpeg_bytes,
)


class CameraStreamMixin:
    def add_camera(
        self,
        camera_name: str,
        camera_id: int | str | None = None,
        camera_bus_id: str | None = None,
    ) -> None:
        """
        Add a camera to the available cameras list.

        Args:
            camera_name (str): The name of the camera.
            camera_id (int | str | None, optional): The camera ID used by UI
                for display/debugging. If None, uses the camera name.
            camera_bus_id (str | None, optional): Deterministic bus_id used by
                pipeline device_input selection. If None, falls back to
                string(camera_id).
        """
        if camera_id is None:
            camera_id = camera_name

        with self.frame_list_structure_lock:
            self.cameras[camera_name] = camera_id
            if camera_name not in self.frame_list:
                self.frame_list[camera_name] = no_image
                self.frame_locks[camera_name] = threading.Lock()

            url_safe_name = camera_name.replace(" ", "_")
            self.available_cameras[camera_name] = {
                "name": url_safe_name,
                "id": camera_id,
                "bus_id": camera_bus_id
                if camera_bus_id is not None
                else str(camera_id),
            }

        self.log(f"Added camera: {camera_name} with ID: {camera_id}")

    def remove_camera(self, camera_name: str) -> None:
        """
        Remove a camera from the available cameras list.

        Args:
            camera_name (str): The name of the camera to remove.
        """
        with self.frame_list_structure_lock:
            if camera_name in self.cameras:
                del self.cameras[camera_name]

                if camera_name in self.frame_list:
                    del self.frame_list[camera_name]

                if camera_name in self.frame_locks:
                    del self.frame_locks[camera_name]

                if camera_name in self.available_cameras:
                    del self.available_cameras[camera_name]

                self.log(f"Removed camera: {camera_name}")

    def get_available_cameras(self) -> dict:
        """Return active cameras with stable identifiers and optional display names.

        Display names are copied from ``CameraConfig`` metadata only for UI
        rendering. Dictionary keys, URL-safe stream names, bus IDs, and all
        runtime camera identities remain unchanged.

        Returns:
            dict: Camera records keyed by their technical camera names.
        """
        with self.frame_list_structure_lock:
            cameras = {
                camera_name: dict(camera_info)
                if isinstance(camera_info, dict)
                else camera_info
                for camera_name, camera_info in self.available_cameras.items()
            }

        registry = self.camera_config_registry
        if registry is None:
            return cameras
        for camera_info in cameras.values():
            if not isinstance(camera_info, dict):
                continue
            bus_id = camera_info.get("bus_id") or camera_info.get("id")
            if bus_id is None:
                continue
            camera_info["display_name"] = registry.get_config(
                str(bus_id)
            ).display_name
        return cameras

    def update_camera_frame(self, camera_name: str, frame: np.ndarray) -> None:
        """
        Update the camera frame.

        Args:
            camera_name (str): The ID of the camera.
            frame: The frame to update as a numpy array.
        """
        lock = self.frame_locks.get(camera_name)
        if lock:
            with lock:
                self.frame_list[camera_name] = frame

    def _frame_generator(self, camera_name: str) -> Generator[bytes, Any, Any]:
        """
        Generate frames for the camera feed.

        Args:
            camera_name (str): The ID of the camera.

        Yields:
            Generator: The camera feed.
        """
        while True:
            time_start = time.time()

            lock = self.frame_locks.get(camera_name)
            if not lock:
                time.sleep(0.01)
                continue

            with lock:
                frame = self.frame_list[camera_name]

            if frame is not None:
                resized_frame = self._resize_view_stream_frame(frame)
                success, encoded_frame = cv2.imencode(
                    ".jpg",
                    resized_frame,
                    self._view_stream_jpeg_params(),
                )
                frame = encoded_frame.tobytes() if success else no_image_jpeg_bytes

            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"

            time.sleep(max((1 / VIEW_STREAM_FPS) - (time.time() - time_start), 0))

    def _resize_view_stream_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize a frame for the Views tab stream only."""
        with self._general_conf_lock:
            downscale = self.view_stream_downscale

        if downscale >= 1.0:
            return frame

        return cv2.resize(
            frame,
            None,
            fx=downscale,
            fy=downscale,
            interpolation=cv2.INTER_AREA,
        )

    def _view_stream_jpeg_params(self) -> list[int]:
        """Return JPEG encoding params for compressed Views tab streams."""
        params = [int(cv2.IMWRITE_JPEG_QUALITY), VIEW_STREAM_JPEG_QUALITY]
        optimize_flag = getattr(cv2, "IMWRITE_JPEG_OPTIMIZE", None)
        if optimize_flag is not None:
            params.extend([int(optimize_flag), 1])
        return params

    def _frame_generator_no_image(self) -> Generator[bytes, Any, Any]:
        """
        Generate no image frames when camera is not found.

        Yields:
            Generator: The no image feed.
        """
        success, encoded_no_image = cv2.imencode(".jpg", no_image)
        if success:
            no_image_frame_bytes = encoded_no_image.tobytes()
        else:
            no_image_frame_bytes = b""

        while True:
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                + no_image_frame_bytes
                + b"\r\n"
            )
            time.sleep(1 / 30)

    def serve_camera_feed_route(self, camera_name: str) -> Response:
        """
        Serve the camera feed.

        Args:
            camera_name (str): The URL-safe camera name.

        Returns:
            Response: The camera feed.
        """
        original_camera_name = camera_name.replace("_", " ")

        if original_camera_name not in self.cameras:
            for orig_name, cam_info in self.available_cameras.items():
                if isinstance(cam_info, dict) and cam_info.get("name") == camera_name:
                    original_camera_name = orig_name
                    break
            else:
                original_camera_name = ""

        if request.args.get("snapshot") == "1":
            # Finite thumbnails leave browser connections free for setup requests.
            jpeg = no_image_jpeg_bytes
            lock = self.frame_locks.get(original_camera_name)
            if lock is not None:
                with lock:
                    source = self.frame_list.get(original_camera_name)
                    frame = None if source is None else source.copy()
                if frame is not None:
                    height, width = frame.shape[:2]
                    if width > 320:
                        frame = cv2.resize(
                            frame, (320, max(1, round(height * 320 / width)))
                        )
                    success, encoded = cv2.imencode(
                        ".jpg", frame, self._view_stream_jpeg_params()
                    )
                    if success:
                        jpeg = encoded.tobytes()
            return Response(
                jpeg, mimetype="image/jpeg", headers={"Cache-Control": "no-store"}
            )

        return Response(
            self._frame_generator(original_camera_name)
            if original_camera_name
            else self._frame_generator_no_image(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )
