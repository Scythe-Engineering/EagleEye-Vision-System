from __future__ import annotations
import json
import logging
import os
import queue
import threading
import time
import traceback
from pathlib import Path
from threading import Thread
from typing import TYPE_CHECKING, Any, Callable, Generator, List, Optional

import cv2
import numpy as np
from flask import Flask, Response, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO
from werkzeug.serving import make_server

from src.utils.colors import Colors
from src.utils.logging.logger import Logger
from src.utils.camera_utils.camera_config_manager import (
    CameraConfig,
    CameraConfigRegistry,
)

if TYPE_CHECKING:
    from src.config.utils.pipeline import Pipeline
from src.webui.web_server_utils.serve_static_files import (
    serve_css,
    serve_index,
    serve_js,
)
from src.main_operations.definitions.base.base_class import OperationInstance

current_path = os.path.dirname(__file__)
src_path = os.path.abspath(os.path.join(current_path, os.pardir))

with open(os.path.join(current_path, "assets", "no_image.png"), "rb") as f:
    no_image_bytes = f.read()

no_image = cv2.imdecode(np.frombuffer(no_image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
success, _noimg_jpeg = cv2.imencode(".jpg", no_image)
no_image_jpeg_bytes: bytes = _noimg_jpeg.tobytes() if success else b""

CORS_ALLOWED_ORIGINS = "*"
PIPELINE_NOT_FOUND_MESSAGE = "Pipeline not found"
TEXT_PLAIN_MIMETYPE = "text/plain"
VISUALIZATION_STREAM_FPS = 12
PROFILING_PUBLISH_INTERVAL_SECONDS = 0.3
PIPELINE_ERROR_PUBLISH_FRAME_INTERVAL = 10
PIPELINE_ERROR_FALLBACK_PUBLISH_INTERVAL_SECONDS = 1.0
SSE_SERIALIZATION_WARN_INTERVAL_SECONDS = 5.0
WEB_SERVER_HOST = "0.0.0.0"
WEB_SERVER_PORT = 5001


class EagleEyeInterface:
    def __init__(
        self,
        restart_callback: Callable[[], None],
        pipeline_objects_callback: Callable[[], dict[str, Pipeline]],
        dev_mode: bool = False,
        logger: Logger | None = None,
    ):
        """
        Initialize the EagleEyeInterface.

        Starts a Flask server in a separate thread.

        Args:
            settings_object (Constants | None): Optional settings object.
            dev_mode (bool): Whether to run in development mode.
            logger: Logger instance for logging.
        """
        self.logger = logger
        self.log = self.logger.log if self.logger is not None else print

        def colored_log(*messages: object) -> None:
            """Log function with automatic color coding based on message content."""
            message = " ".join(str(m) for m in messages)
            if any(
                word in message.lower() for word in ["error", "failed", "exception"]
            ):
                self.logger.log(f"{Colors.RED}{message}{Colors.RESET}")
            elif any(
                word in message.lower()
                for word in ["success", "added", "updated", "started"]
            ):
                self.logger.log(f"{Colors.GREEN}{message}{Colors.RESET}")
            elif any(
                word in message.lower()
                for word in ["warning", "skipping", "queue full"]
            ):
                self.logger.log(f"{Colors.YELLOW}{message}{Colors.RESET}")
            elif any(
                word in message.lower()
                for word in [
                    "connected",
                    "disconnected",
                    "initialized",
                    "set",
                    "removed",
                ]
            ):
                self.logger.log(f"{Colors.CYAN}{message}{Colors.RESET}")
            else:
                self.logger.log(message)

            self.log = colored_log

        self.restart_callback = restart_callback
        self.pipeline_objects_callback = pipeline_objects_callback

        self.restart_required_for_config = False
        self.last_log_message_count = 0
        self._system_status_interval = 1.5
        self._system_status_error_logged = False

        self.app = Flask(
            __name__,
            static_folder=current_path,
            static_url_path="",
        )
        self.app.json.sort_keys = False
        CORS(
            self.app,
            resources={r"/*": {"origins": CORS_ALLOWED_ORIGINS}},
        )

        self.socketio = SocketIO(
            self.app,
            cors_allowed_origins=CORS_ALLOWED_ORIGINS,
        )
        self.app_thread: Thread | None = None
        self._http_server = None

        # Disable Werkzeug access logging (HTTP request logs)
        logging.getLogger("werkzeug").setLevel(logging.WARNING)
        # Simplified single-client SSE: one queue and a lock to guard it.
        self._sse_queue: queue.Queue | None = None
        self._sse_queue_lock = threading.Lock()
        self._pipeline_error_lock = threading.Lock()
        self._pipeline_error_cache: dict[str, dict[str, Any]] = {}
        self._pipeline_error_dirty_pipelines: set[str] = set()
        self._pipeline_error_last_seq_sent: dict[str, int] = {}
        self._pipeline_error_last_publish_ts: dict[str, float] = {}
        self._pipeline_profile_last_seq_sent: dict[str, int] = {}
        self._profiling_publish_interval = PROFILING_PUBLISH_INTERVAL_SECONDS
        self._last_profiling_publish_ts = 0.0
        self._last_sse_serialization_warning_ts = 0.0

        self.cameras = {}
        self.log(f"Initialized with cameras: {self.cameras}")
        self.frame_list = {}
        self.available_cameras = {}
        self.camera_config_registry: CameraConfigRegistry | None = None

        self.frame_locks = {}
        self.frame_list_structure_lock = threading.Lock()

        self._register_routes()

        if dev_mode:
            self.run()
        else:
            self._start_background_server()

        # Start heartbeat publisher thread for connection tracking
        self._heartbeat_interval = 5.0
        Thread(target=self._sse_heartbeat_loop, daemon=True).start()

        # Start log monitoring thread for real-time log updates
        Thread(target=self._log_monitor_loop, daemon=True).start()

        # Start system status monitoring thread for resource updates
        Thread(target=self._system_status_loop, daemon=True).start()

        @self.app.errorhandler(Exception)
        def _log_and_raise(_):
            self.log(f"Error: {traceback.format_exc()}")
            return {"message": "Internal server error"}, 500

    def _register_routes(self) -> None:
        """
        Register all Flask endpoints.
        """
        self.app.add_url_rule("/", "index", lambda: serve_index())
        self.app.add_url_rule("/js/main.js", "script", lambda: serve_js())
        self.app.add_url_rule("/style.css", "style", lambda: serve_css())

        self.app.add_url_rule(
            "/background.png",
            "background",
            lambda: send_from_directory("./static", "background.png"),
        )
        self.app.add_url_rule(
            "/favicon.ico",
            "favicon",
            lambda: send_from_directory(
                os.path.join(current_path, "assets"), "favicon.ico"
            ),
        )
        self.app.add_url_rule(
            "/get-available-cameras",
            "get_available_cameras",
            self.get_available_cameras,
            methods=["GET"],
        )
        self.app.add_url_rule(
            "/camera-config/cameras",
            "get_camera_config_cameras",
            self.get_camera_config_cameras,
            methods=["GET"],
        )
        self.app.add_url_rule(
            "/camera-config/<string:camera_bus_id>",
            "get_camera_config",
            self.get_camera_config,
            methods=["GET"],
        )
        self.app.add_url_rule(
            "/camera-config/<string:camera_bus_id>/extrinsics",
            "save_camera_extrinsics",
            self.save_camera_extrinsics,
            methods=["POST"],
        )
        self.app.add_url_rule(
            "/camera-config/<string:camera_bus_id>/intrinsics",
            "upload_camera_intrinsics",
            self.upload_camera_intrinsics,
            methods=["POST"],
        )
        self.app.add_url_rule(
            "/camera-config/<string:camera_bus_id>/intrinsics",
            "delete_camera_intrinsics",
            self.delete_camera_intrinsics,
            methods=["DELETE"],
        )
        self.app.add_url_rule(
            "/feed/<string:camera_name>",
            "camera_feed",
            self.serve_camera_feed_route,
            methods=["GET"],
        )
        self.app.add_url_rule(
            "/frc2025r2.json",
            "frc2025r2",
            lambda: send_from_directory(
                os.path.join("../", "utils", "field_data"), "frc2025r2.json"
            ),
        )
        self.app.add_url_rule(
            "/src/webui/assets/apriltags/<path:filename>",
            "apriltags_png",
            lambda filename: send_from_directory(
                os.path.join(current_path, "assets", "apriltags"), filename
            ),
        )
        self.app.add_url_rule(
            "/get-available-robots",
            "get_available_robots",
            self.get_available_robots,
            methods=["GET"],
        )
        self.app.add_url_rule(
            "/get-robot-file/<path:filename>",
            "get_robot_file",
            lambda filename: send_from_directory(
                os.path.join(current_path, "assets", "robots"), filename
            ),
        )
        self.app.add_url_rule(
            "/draco/<path:filename>",
            "draco",
            lambda filename: send_from_directory(
                os.path.join(current_path, "web_server_utils", "drako_loader"), filename
            ),
        )
        self.app.add_url_rule(
            "/get-available-operations",
            "get_available_operations",
            self.get_available_operations,
            methods=["GET"],
        )
        self.app.add_url_rule(
            "/get-operation-config-data/<path:operation_name>/<int:is_secondary>",
            "get_operation_config_data",
            self.get_operation_config_data,
            methods=["GET"],
        )
        self.app.add_url_rule(
            "/get-operation-files/<path:operation_name>/<path:parameter_name>",
            "get_operation_files",
            self.get_operation_files,
            methods=["GET"],
        )
        self.app.add_url_rule(
            "/upload-operation-file/<path:operation_name>/<path:parameter_name>",
            "upload_operation_file",
            self.upload_operation_file,
            methods=["POST"],
        )
        self.app.add_url_rule(
            "/delete-operation-file/<path:operation_name>/<path:parameter_name>/<path:filename>",
            "delete_operation_file",
            self.delete_operation_file,
            methods=["DELETE"],
        )
        self.app.add_url_rule(
            "/get-pipeline-config/<string:pipeline_name>",
            "get_pipeline_config_by_name",
            self.get_pipeline_config_by_name,
            methods=["GET"],
        )
        self.app.add_url_rule(
            "/get-pipeline-names",
            "get_pipeline_names",
            self.get_pipeline_names,
            methods=["GET"],
        )
        self.app.add_url_rule(
            "/save-pipeline-config/<string:pipeline_name>",
            "save_pipeline_config_by_name",
            self.save_pipeline_config_by_name,
            methods=["POST"],
        )
        self.app.add_url_rule(
            "/delete-pipeline/<string:pipeline_name>",
            "delete_pipeline_by_name",
            self.delete_pipeline_by_name,
            methods=["DELETE"],
        )
        self.app.add_url_rule(
            "/restart-backend",
            "restart_backend",
            self.restart_backend,
            methods=["POST"],
        )
        self.app.add_url_rule(
            "/start-visualize/<string:pipeline_name>/<string:operation_uuid>",
            "start_visualize",
            self.start_visualize,
            methods=["POST"],
        )
        self.app.add_url_rule(
            "/stop-visualize/<string:pipeline_name>",
            "stop_visualize",
            self.stop_visualize,
            methods=["POST"],
        )
        self.app.add_url_rule(
            "/visualize/<string:pipeline_name>",
            "visualize",
            self.visualize,
            methods=["GET"],
        )
        self.app.add_url_rule(
            "/visualize/stream/<string:pipeline_name>",
            "visualize_stream",
            self.visualize_stream,
            methods=["GET"],
        )
        self.app.add_url_rule(
            "/shutdown",
            "shutdown",
            self.shutdown,
            methods=["POST"],
        )
        self.app.add_url_rule(
            "/set_restart_required",
            "set_restart_required",
            self.set_restart_required,
            methods=["POST"],
        )
        self.app.add_url_rule(
            "/get_restart_required",
            "get_restart_required",
            self.get_restart_required,
            methods=["GET"],
        )
        self.app.add_url_rule(
            "/get-log-messages",
            "get_log_messages",
            self.get_log_messages,
            methods=["GET"],
        )
        self.app.add_url_rule(
            "/download-log-file",
            "download_log_file",
            self.download_log_file,
            methods=["GET"],
        )
        self.app.add_url_rule(
            "/get-general-conf",
            "get_general_conf",
            self.get_general_conf,
            methods=["GET"],
        )
        self.app.add_url_rule(
            "/save-general-conf",
            "save_general_conf",
            self.save_general_conf,
            methods=["POST"],
        )
        self.app.add_url_rule(
            "/get-pipeline-thread-info/<string:pipeline_name>",
            "get_pipeline_thread_info",
            self.get_pipeline_thread_info,
            methods=["GET"],
        )
        self.app.add_url_rule(
            "/get-pipeline-active/<string:pipeline_name>",
            "get_pipeline_active",
            self.get_pipeline_active,
            methods=["GET"],
        )
        self.app.add_url_rule(
            "/get-system-status",
            "get_system_status",
            self.get_system_status,
            methods=["GET"],
        )

        # SSE stream for frontend (named events)
        self.app.add_url_rule(
            "/sse/stream",
            "sse_stream",
            lambda: Response(
                self._sse_stream(),
                mimetype="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Cache-Control",
                },
            ),
        )

    def shutdown(self) -> tuple[dict, int]:
        """
        Shutdown the web interface.

        Returns:
            tuple[dict, int]: A success or failure message.
        """
        try:
            os._exit(0)
        except Exception as e:
            self.log(f"Error during shutdown: {e}")
            return {"message": "Failed to shutdown server"}, 500

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
        """
        Get a dict of available cameras.

        Returns:
            dict: A dict where keys are camera names and values are dicts with:
                - name (str): URL-safe camera name (spaces replaced with underscores)
                - id (int | str): The camera identifier
                - bus_id (str): The camera bus identifier, or string
                  representation of id if bus_id is not available
        """
        return self.available_cameras

    def get_camera_config_cameras(self) -> tuple[dict, int]:
        """Return active camera entries for the camera config UI.

        Returns:
            tuple[dict, int]: Camera list payload with HTTP status.
        """
        cameras: list[dict[str, str]] = []
        for camera_name, camera_info in self.available_cameras.items():
            if isinstance(camera_info, dict):
                bus_id = str(camera_info.get("bus_id") or camera_info.get("id") or "")
            else:
                bus_id = str(camera_info)

            if not bus_id:
                continue

            cameras.append({"name": str(camera_name), "bus_id": bus_id})

        cameras.sort(key=lambda camera: camera["name"])
        return {"cameras": cameras}, 200

    def _resolve_camera_config(self, camera_bus_id: str) -> Optional[CameraConfig]:
        """Resolve the camera config for a bus ID.

        Args:
            camera_bus_id: Deterministic bus ID for the camera.

        Returns:
            CameraConfig instance or None if unavailable.
        """
        if self.camera_config_registry is None:
            return None
        return self.camera_config_registry.get_config(str(camera_bus_id))

    def get_camera_config(self, camera_bus_id: str) -> tuple[dict, int]:
        """Get camera extrinsics and intrinsics metadata for a bus ID.

        Args:
            camera_bus_id: Deterministic camera bus ID.

        Returns:
            tuple[dict, int]: Camera config payload with HTTP status.
        """
        config = self._resolve_camera_config(camera_bus_id)
        if config is None:
            return {"error": "Camera config registry unavailable"}, 503

        intrinsics_path = config.intrinsics_path
        return {
            "camera_bus_id": str(config.camera_id),
            "extrinsics": config.extrinsics.to_dict(),
            "intrinsics_path": intrinsics_path,
            "intrinsics_exists": bool(
                intrinsics_path is not None and os.path.exists(intrinsics_path)
            ),
        }, 200

    def save_camera_extrinsics(self, camera_bus_id: str) -> tuple[dict, int]:
        """Save camera extrinsics for a bus ID.

        Args:
            camera_bus_id: Deterministic camera bus ID.

        Returns:
            tuple[dict, int]: Save result payload with HTTP status.
        """
        config = self._resolve_camera_config(camera_bus_id)
        if config is None:
            return {"error": "Camera config registry unavailable"}, 503

        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return {"error": "Expected JSON object payload"}, 400

        try:
            config.update_extrinsics_live(payload)
        except ValueError as error:
            return {"error": str(error)}, 400
        except Exception as error:
            self.log(f"Failed saving camera extrinsics for {camera_bus_id}: {error}")
            return {"error": "Failed to save camera extrinsics"}, 500

        return {
            "success": True,
            "camera_bus_id": str(config.camera_id),
            "extrinsics": config.extrinsics.to_dict(),
        }, 200

    def _default_intrinsics_path(self, camera_bus_id: str) -> str:
        """Return canonical intrinsics path for a camera bus ID.

        Args:
            camera_bus_id: Deterministic camera bus ID.

        Returns:
            str: Expected intrinsics JSON path.
        """
        return os.path.join(
            src_path,
            "utils",
            "camera_utils",
            "camera_calibrations",
            str(camera_bus_id),
            "intrinsics.json",
        )

    def upload_camera_intrinsics(self, camera_bus_id: str) -> tuple[dict, int]:
        """Upload and set camera intrinsics JSON for a bus ID.

        Args:
            camera_bus_id: Deterministic camera bus ID.

        Returns:
            tuple[dict, int]: Upload result payload with HTTP status.
        """
        config = self._resolve_camera_config(camera_bus_id)
        if config is None:
            return {"error": "Camera config registry unavailable"}, 503

        if "file" not in request.files:
            return {"error": "No file provided"}, 400

        upload = request.files["file"]
        if upload.filename is None or upload.filename.strip() == "":
            return {"error": "No file selected"}, 400

        if not upload.filename.lower().endswith(".json"):
            return {"error": "Only .json intrinsics files are supported"}, 400

        target_path = config.intrinsics_path or self._default_intrinsics_path(camera_bus_id)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        try:
            upload.save(target_path)
            config.intrinsics_path = target_path
        except Exception as error:
            self.log(f"Failed uploading intrinsics for {camera_bus_id}: {error}")
            return {"error": "Failed to upload intrinsics file"}, 500

        return {
            "success": True,
            "camera_bus_id": str(config.camera_id),
            "intrinsics_path": config.intrinsics_path,
            "intrinsics_exists": True,
        }, 200

    def delete_camera_intrinsics(self, camera_bus_id: str) -> tuple[dict, int]:
        """Delete the current intrinsics file for a camera bus ID.

        Args:
            camera_bus_id: Deterministic camera bus ID.

        Returns:
            tuple[dict, int]: Delete result payload with HTTP status.
        """
        config = self._resolve_camera_config(camera_bus_id)
        if config is None:
            return {"error": "Camera config registry unavailable"}, 503

        intrinsics_path = config.intrinsics_path
        if intrinsics_path is None:
            return {"error": "No intrinsics file configured"}, 404

        if not os.path.exists(intrinsics_path):
            config.intrinsics_path = None
            return {"error": "Intrinsics file not found"}, 404

        try:
            os.remove(intrinsics_path)
            config.intrinsics_path = None
        except Exception as error:
            self.log(f"Failed deleting intrinsics for {camera_bus_id}: {error}")
            return {"error": "Failed to delete intrinsics file"}, 500

        return {
            "success": True,
            "camera_bus_id": str(config.camera_id),
            "intrinsics_path": None,
            "intrinsics_exists": False,
        }, 200

    def run(self) -> None:
        """
        Run the development Flask application with SocketIO.
        """
        self.socketio.run(
            self.app,
            host=WEB_SERVER_HOST,
            port=WEB_SERVER_PORT,
            debug=False,
            allow_unsafe_werkzeug=True,
            extra_files=["./static/bundle.js", "./style.css", "./index.html"],
        )

    def _start_background_server(self) -> None:
        """Start the WebUI server in a background thread."""
        if self._requires_threaded_wsgi_fallback():
            self.log(
                "Starting WebUI with threaded WSGI fallback because no "
                "production Socket.IO async backend is installed. "
                "Install gevent or eventlet to enable production websocket support."
            )
            self.app_thread = Thread(
                target=self._serve_threaded_wsgi,
                daemon=True,
            )
        else:
            self.app_thread = Thread(
                target=self.socketio.run,
                args=(self.app,),
                kwargs={
                    "host": WEB_SERVER_HOST,
                    "port": WEB_SERVER_PORT,
                    "debug": False,
                },
                daemon=True,
            )

        self.app_thread.start()

    def _requires_threaded_wsgi_fallback(self) -> bool:
        """Return whether production startup should avoid socketio.run()."""
        return getattr(self.socketio, "async_mode", "threading") == "threading"

    def _serve_threaded_wsgi(self) -> None:
        """Serve the wrapped Flask app with a threaded WSGI server."""
        self._http_server = make_server(
            WEB_SERVER_HOST,
            WEB_SERVER_PORT,
            self.app,
            threaded=True,
        )
        self._http_server.serve_forever()

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
                resized_frame = cv2.resize(
                    frame,
                    None,
                    fx=0.5,
                    fy=0.5,
                    interpolation=cv2.INTER_AREA,
                )
                success, encoded_frame = cv2.imencode(".jpg", resized_frame)
                frame = encoded_frame.tobytes() if success else no_image_jpeg_bytes

            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"

            time.sleep(max((1 / 120) - (time.time() - time_start), 0))

    def _frame_generator_no_image(self) -> Generator[bytes, Any, Any]:
        """
        Generate no image frames when camera is not found.

        Yields:
            Generator: The no image feed.
        """
        success, encoded_no_image = cv2.imencode(".jpg", no_image)
        if success:
            no_image_bytes = encoded_no_image.tobytes()
        else:
            no_image_bytes = b""

        while True:
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                + no_image_bytes
                + b"\r\n"
            )
            time.sleep(1 / 30)

    def _format_sse(self, event: str, data: str) -> bytes:
        """
        Format an SSE message with a named event and JSON data payload.
        """
        return f"event: {event}\ndata: {data}\n\n".encode()

    def _sse_stream(self) -> Generator[bytes, Any, Any]:
        """
        Generator that yields SSE messages for a single client using a queue.
        """
        import queue

        q: queue.Queue = queue.Queue(maxsize=100)
        # assume single client: set queue, replacing any existing queue
        with self._sse_queue_lock:
            self._sse_queue = q

        with self._pipeline_error_lock:
            self._pipeline_error_dirty_pipelines.update(self._pipeline_error_cache.keys())

        self._publish_cached_pipeline_errors()

        try:
            while True:
                try:
                    # Use timeout to allow checking for disconnection
                    msg = q.get(timeout=1.0)
                    yield msg
                except queue.Empty:
                    # Check if client is still connected by yielding a comment
                    yield b": keepalive\n\n"
                    continue
        except GeneratorExit:
            # Client disconnected
            pass
        finally:
            # clear the single-client queue on disconnect
            with self._sse_queue_lock:
                if self._sse_queue is q:
                    self._sse_queue = None

    def _publish_event(self, event_name: str, data: object) -> None:
        """
        Publish a named SSE event (JSON-serialized) to all connected subscribers.
        """
        try:
            payload = json.dumps(data, allow_nan=False)
        except Exception as error:
            now = time.time()
            if (
                now - self._last_sse_serialization_warning_ts
                >= SSE_SERIALIZATION_WARN_INTERVAL_SECONDS
            ):
                self._last_sse_serialization_warning_ts = now
                self.log(f"Failed to serialize SSE event {event_name}: {error}")
            return
        msg = self._format_sse(event_name, payload)
        # publish only to the single client's queue if present
        with self._sse_queue_lock:
            q = self._sse_queue
        if q is not None:
            try:
                # Use put_nowait to avoid blocking, and catch Full exception
                q.put_nowait(msg)
            except queue.Full:
                # Queue is full, drop the oldest item and retry
                try:
                    q.get_nowait()  # Remove oldest item
                    q.put_nowait(msg)  # Try again with new message
                    self.log(
                        f"SSE queue full, dropped oldest event to add {event_name}"
                    )
                except queue.Empty:
                    # Shouldn't happen, but log if queue became empty
                    self.log(
                        f"SSE queue unexpectedly empty when trying to drop oldest for {event_name}"
                    )
                except queue.Full:
                    # Still full after dropping oldest, skip this event
                    self.log(
                        f"SSE queue still full after dropping oldest, dropping {event_name} event"
                    )
            except Exception as e:
                # Other error, client likely disconnected
                self.log(f"SSE publish error for {event_name}: {e}")

    def _publish_cached_pipeline_errors(self) -> None:
        """Publish cached pipeline operation errors to the active SSE client.

        Error snapshots are published in batches (all operation errors in one
        payload per pipeline) and throttled to reduce event spam.
        """
        with self._pipeline_error_lock:
            cached_payloads = {
                pipeline_name: payload.copy()
                for pipeline_name, payload in self._pipeline_error_cache.items()
            }
            dirty_pipelines = set(self._pipeline_error_dirty_pipelines)
            last_seq_sent = self._pipeline_error_last_seq_sent.copy()
            last_publish_ts = self._pipeline_error_last_publish_ts.copy()

        if not cached_payloads:
            return

        now = time.time()
        frame_seq_by_pipeline: dict[str, int] = {}
        try:
            pipelines = self.pipeline_objects_callback() or {}
            for pipeline_name, pipeline in pipelines.items():
                snapshot = pipeline.get_latest_profile_snapshot()
                if not snapshot:
                    continue
                frame_seq = int(snapshot.get("frame_seq", 0))
                if frame_seq > 0:
                    frame_seq_by_pipeline[pipeline_name] = frame_seq
        except Exception as error:
            self.log(f"Failed to read pipelines for error batching: {error}")

        for pipeline_name, payload in cached_payloads.items():
            if pipeline_name not in dirty_pipelines:
                continue

            current_frame_seq = frame_seq_by_pipeline.get(pipeline_name, 0)
            previously_sent_seq = last_seq_sent.get(pipeline_name, 0)
            previously_sent_ts = last_publish_ts.get(pipeline_name, 0.0)

            frame_gate_open = (
                current_frame_seq > 0
                and current_frame_seq - previously_sent_seq
                >= PIPELINE_ERROR_PUBLISH_FRAME_INTERVAL
            )
            fallback_gate_open = (
                now - previously_sent_ts
                >= PIPELINE_ERROR_FALLBACK_PUBLISH_INTERVAL_SECONDS
            )

            if not frame_gate_open and not fallback_gate_open:
                continue

            errors = payload.get("errors")
            normalized_payload = {
                "pipeline_name": pipeline_name,
                "errors": errors if isinstance(errors, list) else [],
            }
            try:
                self._publish_event("pipeline_operation_errors", normalized_payload)
                with self._pipeline_error_lock:
                    self._pipeline_error_dirty_pipelines.discard(pipeline_name)
                    self._pipeline_error_last_publish_ts[pipeline_name] = now
                    if current_frame_seq > 0:
                        self._pipeline_error_last_seq_sent[pipeline_name] = (
                            current_frame_seq
                        )
            except Exception:
                continue

    def _sse_heartbeat_loop(self) -> None:
        """
        Periodically publish a heartbeat event for connection tracking.
        """
        last_heartbeat_sent = 0.0
        while True:
            try:
                self._publish_cached_pipeline_errors()
                self._publish_profiling_updates()
                now = time.time()
                if now - last_heartbeat_sent >= self._heartbeat_interval:
                    self._publish_event("heartbeat", {"ts": now})
                    last_heartbeat_sent = now
            except Exception as e:
                self.log(f"Error sending heartbeat: {e}")
            time.sleep(min(self._profiling_publish_interval, 0.1))

    def _publish_profiling_updates(self) -> None:
        """Publish latest pipeline profiling snapshots over SSE."""
        now = time.time()
        if now - self._last_profiling_publish_ts < self._profiling_publish_interval:
            return
        self._last_profiling_publish_ts = now

        try:
            pipelines = self.pipeline_objects_callback() or {}
        except Exception as error:
            self.log(f"Failed to get pipelines for profiling SSE: {error}")
            return

        for pipeline_name, pipeline in pipelines.items():
            try:
                snapshot = pipeline.get_latest_profile_snapshot()
                if not snapshot:
                    continue
                frame_seq = int(snapshot.get("frame_seq", 0))
                if frame_seq <= 0:
                    continue

                last_sent_seq = self._pipeline_profile_last_seq_sent.get(
                    pipeline_name,
                    0,
                )
                if frame_seq <= last_sent_seq:
                    continue

                self._publish_event("profiling_update", snapshot)
                self._pipeline_profile_last_seq_sent[pipeline_name] = frame_seq
            except Exception as error:
                self.log(
                    f"Failed to publish profiling_update for {pipeline_name}: {error}"
                )

    def serve_camera_feed_route(self, camera_name: str) -> Response:
        """
        Serve the camera feed.

        Args:
            camera_name (str): The URL-safe camera name.

        Returns:
            Response: The camera feed.
        """
        # Convert URL-safe name back to original camera name
        original_camera_name = camera_name.replace("_", " ")

        # Check if camera exists in our available cameras
        if original_camera_name not in self.cameras:
            # Try to find camera by URL-safe name in reverse mapping
            for orig_name, cam_info in self.available_cameras.items():
                if isinstance(cam_info, dict) and cam_info.get("name") == camera_name:
                    original_camera_name = orig_name
                    break
            else:
                # Return no image if camera not found
                return Response(
                    self._frame_generator_no_image(),
                    mimetype="multipart/x-mixed-replace; boundary=frame",
                )

        return Response(
            self._frame_generator(original_camera_name),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    def update_robot_position(self, transformation_matrix: np.ndarray) -> None:
        """
        Push the tracked robot's transformation matrix to the frontend via websocket.

        Args:
            transformation_matrix (np.ndarray): The new transformation matrix as a 4x4 numpy array.
        """
        if transformation_matrix.shape != (4, 4):
            raise ValueError("Transformation matrix must be a 4x4 numpy array.")

        # Skip publishing if any value is non-finite to avoid invalid JSON or bad transforms
        if not np.all(np.isfinite(transformation_matrix)):
            self.log("Skipping publish of robot transform due to non-finite values")
            return

        # Convert matrix to list for JSON serialization and publish via SSE
        matrix_list = transformation_matrix.tolist()
        try:
            self._publish_event(
                "update_robot_transform", {"transform_matrix": matrix_list}
            )
        except Exception:
            # fallback: log if publish fails
            self.log("Failed to publish update_robot_transform via SSE")

    def update_detected_objects(self, detections: list[dict[str, Any]]) -> None:
        """
        Publish detected objects for 3D visualization.

        Args:
            detections (list[dict[str, Any]]): Detected objects with 3D positions and metadata.

        Returns:
            None: This method does not return a value.
        """
        if not isinstance(detections, list):
            return

        validated_detections: list[dict[str, Any]] = []
        for detection in detections:
            if not isinstance(detection, dict):
                continue

            position = detection.get("position_3d")
            if not (
                isinstance(position, (list, tuple))
                and len(position) == 3
                and all(isinstance(coord, (int, float)) for coord in position)
            ):
                continue

            position_values = [float(coord) for coord in position]
            if not np.all(np.isfinite(position_values)):
                continue

            detection_payload: dict[str, Any] = {"position_3d": position_values}

            class_id = detection.get("class_id")
            if isinstance(class_id, (int, float, str)):
                detection_payload["class_id"] = class_id

            confidence = detection.get("confidence")
            if isinstance(confidence, (int, float)) and np.isfinite(confidence):
                detection_payload["confidence"] = float(confidence)

            class_name = detection.get("class_name")
            if class_name is not None:
                detection_payload["class_name"] = str(class_name)

            validated_detections.append(detection_payload)

        try:
            self._publish_event(
                "update_detected_objects", {"detections": validated_detections}
            )
        except Exception:
            self.log("Failed to publish update_detected_objects via SSE")

    def get_available_robots(self) -> dict:
        """
        Get a dict of available robots.

        Returns:
            dict:
                robots: list of dicts with the name and path of the robot file.
                    name: the name of the robot file.
                    path: the path of the robot file.
        """

        return {
            "robots": [
                os.path.basename(file)
                for file in os.listdir(os.path.join(current_path, "assets", "robots"))
                if file.endswith(".glb") and not file.startswith("_")
            ]
        }

    def get_available_operations(self) -> dict:
        """
        Get a dict of available operations.

        Returns:
            dict:
                operations: list of dicts with the name and path of the operation file.
        """
        NO_DESCRIPTION_AVAILABLE_MESSAGE = "No description available"
        main_operations = []

        for file in os.listdir(
            os.path.join(src_path, "main_operations", "definitions")
        ):
            if file.endswith(".py") and not file.startswith("_"):
                config_data_path = os.path.join(
                    src_path,
                    "main_operations",
                    "definitions",
                    "config_data",
                    file.rstrip(".py") + "_config_def.json",
                )
                try:
                    with open(config_data_path, "r") as f:
                        config_data = json.load(f)
                    description = config_data.get(
                        "description", NO_DESCRIPTION_AVAILABLE_MESSAGE
                    )
                    category = config_data.get("category", "Uncategorized")
                except (FileNotFoundError, json.JSONDecodeError, KeyError):
                    description = NO_DESCRIPTION_AVAILABLE_MESSAGE
                    category = "Uncategorized"

                main_operations.append(
                    {
                        "name": os.path.basename(file),
                        "path": os.path.join(
                            src_path, "main_operations", "definitions", file
                        ),
                        "config_data_path": config_data_path,
                        "description": description,
                        "category": category,
                        "is_secondary": False,
                        "has_visualization": self._operation_has_visualization(
                            file,
                            is_secondary=False,
                        ),
                    }
                )

        secondary_operations = []

        for file in os.listdir(os.path.join(src_path, "secondary_operations")):
            if file.endswith(".py") and not file.startswith("_"):
                config_data_path = os.path.join(
                    src_path,
                    "secondary_operations",
                    "config_data",
                    file.rstrip(".py") + "_config_def.json",
                )
                try:
                    with open(config_data_path, "r") as f:
                        config_data = json.load(f)
                    description = config_data.get(
                        "description", NO_DESCRIPTION_AVAILABLE_MESSAGE
                    )
                    category = config_data.get("category", "Uncategorized")
                except (FileNotFoundError, json.JSONDecodeError, KeyError):
                    description = NO_DESCRIPTION_AVAILABLE_MESSAGE
                    category = "Uncategorized"

                secondary_operations.append(
                    {
                        "name": os.path.basename(file),
                        "path": os.path.join(src_path, "secondary_operations", file),
                        "config_data_path": config_data_path,
                        "description": description,
                        "category": category,
                        "is_secondary": True,
                        "has_visualization": self._operation_has_visualization(
                            file,
                            is_secondary=True,
                        ),
                    }
                )

        return {
            "operations": main_operations + secondary_operations,
        }

    def _operation_has_visualization(self, filename: str, is_secondary: bool) -> bool:
        """Check if an operation overrides the base visualize method."""
        module_path = (
            f"src.secondary_operations.{filename[:-3]}"
            if is_secondary
            else f"src.main_operations.definitions.{filename[:-3]}"
        )
        try:
            module = __import__(module_path, fromlist=["*"])
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, OperationInstance)
                    and attr is not OperationInstance
                ):
                    return attr.visualize is not OperationInstance.visualize
        except Exception as e:
            self.log(f"Warning: Could not detect visualization for {filename}: {e}")
        return False

    def get_operation_config_data(
        self, operation_name: str, is_secondary: bool = False
    ) -> dict:
        """
        Get the config data for an operation.

        Args:
            operation_name (str): The name of the operation.
            is_secondary (bool): Whether the operation is a secondary operation.

        Returns:
            dict: The config data for the operation.
        """
        config_file_name = (
            operation_name.lower().replace(" ", "_").replace(".py", "")
            + "_config_def.json"
        )

        if is_secondary:
            config_path = os.path.join(
                src_path,
                "secondary_operations",
                "config_data",
                config_file_name,
            )
        else:
            config_path = os.path.join(
                src_path,
                "main_operations",
                "definitions",
                "config_data",
                config_file_name,
            )

        try:
            with open(config_path, "r") as f:
                config_data = json.load(f, object_pairs_hook=dict)
                return self._normalize_dynamic_group_config(config_data)
        except FileNotFoundError:
            # Don't log errors for missing configs when trying both locations
            return {}
        except json.JSONDecodeError as e:
            self.log(f"Error loading config for operation {operation_name}: {e}")
            return {}

    def _normalize_dynamic_group_config(self, config_data: dict[str, Any]) -> dict[str, Any]:
        """Normalize optional dynamic group metadata in operation config.

        Args:
            config_data (dict[str, Any]): Raw operation config definition JSON,
                including optional `dynamic_group` metadata that is normalized
                for downstream port handling.

        Returns:
            dict[str, Any]: Config data with normalized `dynamic_group`
            metadata values (for example max counts, boolean flags, and base
            node names).
        """
        if not isinstance(config_data, dict):
            return {}

        dynamic_group = config_data.get("dynamic_group")
        if not isinstance(dynamic_group, dict):
            return config_data

        normalized_group = dict(dynamic_group)
        try:
            normalized_group["max_inputs"] = max(
                1,
                int(dynamic_group.get("max_inputs", 1)),
            )
        except (TypeError, ValueError):
            normalized_group["max_inputs"] = 1

        try:
            normalized_group["max_outputs"] = max(
                1,
                int(dynamic_group.get("max_outputs", normalized_group["max_inputs"])),
            )
        except (TypeError, ValueError):
            normalized_group["max_outputs"] = normalized_group["max_inputs"]

        mirrored_output_group = dynamic_group.get("mirrored_output_group", False)
        if isinstance(mirrored_output_group, str):
            mirrored_output_group = mirrored_output_group.lower() == "true"
        normalized_group["mirrored_output_group"] = bool(mirrored_output_group)

        output_dynamic_group = dynamic_group.get("output_dynamic_group", False)
        if isinstance(output_dynamic_group, str):
            output_dynamic_group = output_dynamic_group.lower() == "true"
        normalized_group["output_dynamic_group"] = bool(output_dynamic_group)

        input_dynamic_group = dynamic_group.get("input_dynamic_group", True)
        if isinstance(input_dynamic_group, str):
            input_dynamic_group = input_dynamic_group.lower() == "true"
        normalized_group["input_dynamic_group"] = bool(input_dynamic_group)

        coupled_groups = dynamic_group.get(
            "coupled_groups",
            normalized_group["mirrored_output_group"],
        )
        if isinstance(coupled_groups, str):
            coupled_groups = coupled_groups.lower() == "true"
        normalized_group["coupled_groups"] = bool(coupled_groups)

        input_nodes = config_data.get("input_nodes") or []
        output_nodes = config_data.get("output_nodes") or []

        input_base_name = normalized_group.get("input_base_name") or normalized_group.get(
            "input_node"
        )
        output_base_name = normalized_group.get("output_base_name") or normalized_group.get(
            "output_node"
        )

        if not input_base_name:
            if input_nodes:
                candidate = input_nodes[-1]
                if isinstance(candidate, dict):
                    input_base_name = candidate.get("name")
                elif isinstance(candidate, str):
                    input_base_name = candidate
            if not input_base_name:
                input_base_name = "data"
        normalized_group["input_base_name"] = input_base_name

        if not output_base_name:
            if output_nodes:
                candidate = output_nodes[-1]
                if isinstance(candidate, dict):
                    output_base_name = candidate.get("name")
                elif isinstance(candidate, str):
                    output_base_name = candidate
            if not output_base_name:
                output_base_name = input_base_name
        normalized_group["output_base_name"] = output_base_name

        config_data["dynamic_group"] = normalized_group
        return config_data

    def _reorder_operation_params(
        self, operation_name: str, action_params: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Reorder operation parameters according to their config definition.

        Args:
            operation_name (str): The name of the operation.
            action_params (dict[str, Any]): The current action parameters.

        Returns:
            dict[str, Any]: The reordered action parameters.
        """
        try:
            # Try secondary operations first (most secondary ops don't have main equivalents)
            config_def = self.get_operation_config_data(operation_name, True)
            if not config_def or "parameters" not in config_def:
                # Try main operations
                config_def = self.get_operation_config_data(operation_name, False)

            if config_def and "parameters" in config_def:
                param_order = list(config_def["parameters"].keys())
                # Reorder action_params according to config definition
                reordered_params = {}
                for param in param_order:
                    if param in action_params:
                        reordered_params[param] = action_params[param]
                # Add any parameters not in config definition (for backward compatibility)
                for param, value in action_params.items():
                    if param not in reordered_params:
                        reordered_params[param] = value
                return reordered_params
        except Exception as e:
            # If reordering fails, keep original order
            self.log(f"Warning: Could not reorder parameters for {operation_name}: {e}")

        # Return original params if reordering failed
        return action_params

    def _get_parameter_file_extensions(self, parameter_name: str) -> List[str]:
        """
        Get allowed file extensions for a parameter.

        Args:
            parameter_name: Name of the parameter.

        Returns:
            List of allowed file extensions (with dots).
        """
        extension_map = {
            "camera_parameters_path": [".json"],
            "apriltag_map_path": [".fmap", ".json"],
            "model_path": [".onnx", ".dfp", ".pt"],
            "post_processing_model_path": [".onnx"],
        }
        return extension_map.get(parameter_name, [])

    def _ensure_parameter_directory(self, parameter_name: str) -> Path:
        """
        Ensure the parameter-specific file directory exists.

        Args:
            parameter_name: Name of the parameter.

        Returns:
            Path to the parameter-specific directory.
        """
        files_base_dir = Path(src_path).parent / "files"
        parameter_dir = files_base_dir / parameter_name
        parameter_dir.mkdir(parents=True, exist_ok=True)
        return parameter_dir

    def get_operation_files(
        self, operation_name: str, parameter_name: str
    ) -> tuple[dict, int]:
        """
        Get list of available files for an operation parameter.

        Args:
            operation_name: Name of the operation (for UI context only).
            parameter_name: Name of the parameter.

        Returns:
            Tuple of (response dict, status code).
        """
        try:
            parameter_dir = self._ensure_parameter_directory(parameter_name)
            allowed_extensions = self._get_parameter_file_extensions(parameter_name)

            if not allowed_extensions:
                return {
                    "error": f"No file extensions defined for parameter {parameter_name}"
                }, 400

            files = []
            if parameter_dir.exists():
                for file_path in parameter_dir.iterdir():
                    if (
                        file_path.is_file()
                        and file_path.suffix.lower() in allowed_extensions
                    ):
                        file_stat = file_path.stat()
                        files.append(
                            {
                                "filename": file_path.name,
                                "size": file_stat.st_size,
                                "modified": file_stat.st_mtime,
                            }
                        )

            files.sort(key=lambda x: x["modified"], reverse=True)

            relative_path = parameter_dir.relative_to(Path(src_path).parent)
            base_path = f"{{project_root}}/{relative_path}"
            return {
                "files": [f["filename"] for f in files],
                "file_details": files,
                "base_path": str(base_path),
            }, 200
        except Exception as e:
            self.log(f"Error getting operation files: {e}")
            return {"error": str(e)}, 500

    def upload_operation_file(
        self, operation_name: str, parameter_name: str
    ) -> tuple[dict, int]:
        """
        Upload a file for an operation parameter.

        Args:
            operation_name: Name of the operation (for UI context only).
            parameter_name: Name of the parameter.

        Returns:
            Tuple of (response dict, status code).
        """
        try:
            if "file" not in request.files:
                return {"error": "No file provided"}, 400

            file = request.files["file"]
            if file.filename == "":
                return {"error": "No file selected"}, 400

            allowed_extensions = self._get_parameter_file_extensions(parameter_name)
            if not allowed_extensions:
                return {
                    "error": f"No file extensions defined for parameter {parameter_name}"
                }, 400

            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in allowed_extensions:
                return {
                    "error": f"Invalid file extension. Allowed: {', '.join(allowed_extensions)}"
                }, 400

            parameter_dir = self._ensure_parameter_directory(parameter_name)
            file_path = parameter_dir / file.filename

            file.save(str(file_path))
            self.log(
                f"Uploaded file {file.filename} for {operation_name}/{parameter_name}"
            )

            relative_path = parameter_dir.relative_to(Path(src_path).parent)
            full_path = f"{{project_root}}/{relative_path}/{file.filename}"
            return {
                "success": True,
                "filename": file.filename,
                "path": full_path,
            }, 200
        except Exception as e:
            self.log(f"Error uploading operation file: {e}")
            return {"error": str(e)}, 500

    def delete_operation_file(
        self, operation_name: str, parameter_name: str, filename: str
    ) -> tuple[dict, int]:
        """
        Delete a file for an operation parameter.

        Args:
            operation_name: Name of the operation (for UI context only).
            parameter_name: Name of the parameter.
            filename: Name of the file to delete.

        Returns:
            Tuple of (response dict, status code).
        """
        try:
            parameter_dir = self._ensure_parameter_directory(parameter_name)
            file_path = parameter_dir / filename

            if not file_path.exists():
                return {"error": "File not found"}, 404

            if not file_path.is_file():
                return {"error": "Path is not a file"}, 400

            allowed_extensions = self._get_parameter_file_extensions(parameter_name)
            if file_path.suffix.lower() not in allowed_extensions:
                return {"error": "File extension not allowed for this parameter"}, 400

            file_path.unlink()
            self.log(f"Deleted file {filename} for {operation_name}/{parameter_name}")

            return {"success": True}, 200
        except Exception as e:
            self.log(f"Error deleting operation file: {e}")
            return {"error": str(e)}, 500

    def get_pipeline_config_by_name(self, pipeline_name: str) -> list:
        """
        Get the config data for a pipeline by name.

        Args:
            pipeline_name (str): The name of the pipeline.

        Returns:
            list: The config data for the pipeline.
        """
        with open(os.path.join(src_path, "config", "pipeline_config.json"), "r") as f:
            config = json.load(f)
            if pipeline_name not in config:
                return []
            pipeline_config = config[pipeline_name]

        return self._reorder_pipeline_config(pipeline_config)

    def get_pipeline_names(self) -> list[str]:
        """
        Get the names of all pipelines.

        Returns:
            list[str]: The names of all pipelines.
        """
        with open(os.path.join(src_path, "config", "pipeline_config.json"), "r") as f:
            config = json.load(f)
        return list(config.keys())

    def publish_operation_errors(self, payload: dict[str, Any]) -> None:
        """Publish operation error updates via SSE.

        Args:
            payload: Error payload containing pipeline and operation data.
        """
        try:
            pipeline_name = payload.get("pipeline_name") or "unknown"
            errors = payload.get("errors")
            normalized_payload = {
                "pipeline_name": pipeline_name,
                "errors": errors if isinstance(errors, list) else [],
            }
            with self._pipeline_error_lock:
                self._pipeline_error_cache[pipeline_name] = normalized_payload
                self._pipeline_error_dirty_pipelines.add(pipeline_name)
        except Exception as e:
            self.log(f"Failed to publish pipeline_operation_errors: {e}")

    def save_pipeline_config_by_name(self, pipeline_name: str) -> tuple[dict, int]:
        """
        Save the pipeline config by pipeline name.

        Args:
            pipeline_name (str): The name of the pipeline.

        Returns:
            tuple[dict, int]: A success or failure message.
        """
        with open(os.path.join(src_path, "config", "pipeline_config.json"), "r") as f:
            current_config = json.load(f)
            new_data = request.get_json()

        if pipeline_name not in current_config:
            current_config[pipeline_name] = []

        # Merge operations while preserving existing data and enabling reordering
        existing_ops = {op["uuid"]: op for op in current_config[pipeline_name]}
        updated_operations = []
        for operation in new_data:
            operation_uuid = operation["uuid"]
            operation_name = operation["action_name"]
            operation_params = self._reorder_operation_params(
                operation_name, operation["action_params"]
            )

            if operation_uuid in existing_ops:
                # Merge incoming data into existing operation
                merged_op = existing_ops[operation_uuid].copy()
                for key, value in operation.items():
                    if key == "action_params":
                        merged_op["action_params"].update(operation_params)
                    else:
                        merged_op[key] = value
            else:
                # New operation
                merged_op = operation.copy()
                merged_op["action_params"] = operation_params

            updated_operations.append(merged_op)

        current_config[pipeline_name] = updated_operations

        with open(os.path.join(src_path, "config", "pipeline_config.json"), "w") as f:
            json.dump(current_config, f, indent=4)

        # use callback to prevent circular imports
        pipeline_objects = self.pipeline_objects_callback()
        if pipeline_name in pipeline_objects:
            pipeline_objects[pipeline_name].update_operations_config(request.get_json())

        return {"message": "Pipeline config saved successfully"}, 200

    def delete_pipeline_by_name(self, pipeline_name: str) -> tuple[dict, int]:
        """
        Delete a pipeline by name.

        Args:
            pipeline_name (str): The name of the pipeline.
        """
        with open(os.path.join(src_path, "config", "pipeline_config.json"), "r") as f:
            current_config = json.load(f)
            if pipeline_name in current_config:
                del current_config[pipeline_name]
            else:
                return {"message": PIPELINE_NOT_FOUND_MESSAGE}, 404
        with open(os.path.join(src_path, "config", "pipeline_config.json"), "w") as f:
            json.dump(current_config, f, indent=4)
        return {"message": "Pipeline deleted successfully"}, 200

    def _reorder_pipeline_config(
        self, pipeline_config: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Reorder operation parameters for a pipeline config list.

        Args:
            pipeline_config: Configuration list for the pipeline.

        Returns:
            Reordered pipeline config list.
        """
        reordered_pipeline = []
        for operation in pipeline_config:
            operation_name = operation["action_name"]
            action_params = self._reorder_operation_params(
                operation_name, operation["action_params"]
            )

            operation["action_params"] = action_params

            reordered_pipeline.append(operation)

        return reordered_pipeline

    def start_visualize(
        self, pipeline_name: str, operation_uuid: str
    ) -> tuple[dict, int]:
        """
        Start visualizing the pipeline.

        Args:
            pipeline_name: Name of the pipeline to visualize.
            operation_uuid: UUID of the operation instance to visualize.

        Returns:
            A response message and HTTP status code.
        """
        try:
            pipeline = self.pipeline_objects_callback()[pipeline_name]
            operation = pipeline.get_operation_by_uuid(operation_uuid)
            if operation is None:
                return {"message": "Operation not found"}, 404
            if not self._instance_has_visualization(operation.instance):
                with pipeline.visualization_data_lock:
                    pipeline.set_visualize = False
                    pipeline.visualization_operation_uuid = None
                    pipeline.visualization_data = None
                return {"message": "Operation has no visualization"}, 400
            pipeline.start_visualize(operation.uuid)
        except KeyError:
            return {"message": PIPELINE_NOT_FOUND_MESSAGE}, 404
        return {"message": "Pipeline visualized successfully"}, 200

    def stop_visualize(self, pipeline_name: str) -> tuple[dict, int]:
        """
        Stop visualizing the pipeline.

        Args:
            pipeline_name: Name of the pipeline.
        """
        try:
            self.pipeline_objects_callback()[pipeline_name].stop_visualize()
        except KeyError:
            return {"message": PIPELINE_NOT_FOUND_MESSAGE}, 404
        return {"message": "Pipeline visualized stopped"}, 200

    def visualize(self, pipeline_name: str) -> Response:
        """
        Visualize the pipeline.

        Args:
            pipeline_name: Name of the pipeline.

        Returns the image as JPEG binary data.
        """
        try:
            pipeline = self.pipeline_objects_callback()[pipeline_name]
        except KeyError:
            return Response(
                PIPELINE_NOT_FOUND_MESSAGE, status=404, mimetype=TEXT_PLAIN_MIMETYPE
            )

        # Get visualization data from pipeline
        with pipeline.visualization_data_lock:
            visualization_data = pipeline.visualization_data

        if visualization_data is None:
            return Response(
                "No visualization data available",
                status=500,
                mimetype=TEXT_PLAIN_MIMETYPE,
            )

        # Get the visualized frame from the visualization data
        image_array = visualization_data.get("visualization_data")

        if image_array is None:
            return Response(
                "Function has no visualization",
                status=500,
                mimetype=TEXT_PLAIN_MIMETYPE,
            )

        # Encode the numpy array to JPEG format
        success, encoded_image = cv2.imencode(".jpg", image_array)
        if not success:
            return Response(
                "Failed to encode image", status=500, mimetype=TEXT_PLAIN_MIMETYPE
            )

        # Return the encoded image as binary data with proper content type
        return Response(encoded_image.tobytes(), mimetype="image/jpeg")

    def visualize_stream(self, pipeline_name: str) -> Response:
        """Stream visualization frames as MJPEG."""
        try:
            pipeline = self.pipeline_objects_callback()[pipeline_name]
        except KeyError:
            return Response(
                PIPELINE_NOT_FOUND_MESSAGE, status=404, mimetype=TEXT_PLAIN_MIMETYPE
            )

        return Response(
            self._visualization_frame_generator(pipeline),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    def _visualization_frame_generator(
        self, pipeline: "Pipeline"
    ) -> Generator[bytes, Any, Any]:
        frame_interval = 1.0 / VISUALIZATION_STREAM_FPS
        last_frame_time = 0.0
        while True:
            now = time.time()
            elapsed = now - last_frame_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
            last_frame_time = time.time()

            with pipeline.visualization_data_lock:
                visualization_data = pipeline.visualization_data

            image_array = None
            if visualization_data is not None:
                image_array = visualization_data.get("visualization_data")

            if image_array is None:
                frame_bytes = no_image_jpeg_bytes
            else:
                success, encoded_image = cv2.imencode(".jpg", image_array)
                frame_bytes = (
                    encoded_image.tobytes() if success else no_image_jpeg_bytes
                )

            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

    def _instance_has_visualization(
        self, operation_instance: OperationInstance
    ) -> bool:
        """Check whether an operation instance has a custom visualize method.

        Determines if the operation instance has overridden the base visualize
        method from OperationInstance, indicating it provides visualization
        capabilities.

        Args:
            operation_instance (OperationInstance): The operation instance to
                check for visualization support.

        Returns:
            bool: True if the instance's class has overridden the visualize
                method, False if it uses the default OperationInstance.visualize.
        """
        return operation_instance.__class__.visualize is not OperationInstance.visualize

    def restart_backend(self) -> tuple[dict, int]:
        """
        Restart the backend.
        """
        self.restart_callback()
        return {"message": "Backend restarted successfully"}, 200

    def set_restart_required(self) -> tuple[dict, int]:
        """
        Set the restart required flag.
        """
        self.restart_required_for_config = True
        return {"message": "Restart required for config set successfully"}, 200

    def get_restart_required(self) -> tuple[dict, int]:
        """
        Get the restart required flag.
        """
        return {"restart_required": self.restart_required_for_config}, 200

    def get_log_messages(self) -> tuple[dict, int]:
        """
        Get all log messages from the logger instance.

        Returns:
            tuple[dict, int]: Dictionary containing log messages and HTTP status code.
        """
        if self.logger is None:
            return {"messages": [], "error": "Logger instance not available"}, 503

        try:
            log_lines = self.logger.message_history.to_file_lines()

            return {"messages": log_lines, "total_count": len(log_lines)}, 200
        except Exception as e:
            self.logger.log(f"Error retrieving log messages: {e}")
            return {"messages": [], "error": str(e)}, 500

    def _log_monitor_loop(self) -> None:
        """
        Monitor the logger for new messages and publish them via SSE.
        """
        if self.logger is None:
            return

        while True:
            try:
                current_message_count = len(self.logger.message_history.messages)

                if current_message_count > self.last_log_message_count:
                    message_lines = self.logger.message_history.to_file_lines()

                    if message_lines:
                        self._publish_event(
                            "log_update",
                            {
                                "messages": message_lines,
                            },
                        )

                    self.last_log_message_count = current_message_count

                time.sleep(0.1)
            except Exception as e:
                self.logger.log(f"Error in log monitor loop: {e}")
                time.sleep(1.0)

    def get_pipeline_active(self, pipeline_name: str) -> tuple[dict, int]:
        """
        Return activity status for a pipeline.

        Args:
            pipeline_name: Name of the pipeline.

        Returns:
            tuple[dict, int]: Dictionary containing active flag.
        """
        try:
            pipeline = self.pipeline_objects_callback()[pipeline_name]
        except KeyError:
            return {"message": PIPELINE_NOT_FOUND_MESSAGE}, 404

        try:
            active = bool(pipeline.is_active())
        except Exception as error:
            self.log(
                f"Failed to read active status for pipeline {pipeline_name}: {error}"
            )
            active = False
        return {"pipeline": pipeline_name, "active": active}, 200

    def get_system_status(self) -> tuple[dict, int]:
        """
        Get current system status metrics.

        Returns:
            tuple[dict, int]: Dictionary containing system metrics.
        """
        payload = self._build_system_status_payload()
        return payload, 200

    def _system_status_loop(self) -> None:
        """
        Publish system status metrics via SSE on a fixed interval.
        """
        while True:
            try:
                payload = self._build_system_status_payload()
                self._publish_event("system_status", payload)
            except Exception as e:
                self.log(f"Error publishing system status: {e}")
            time.sleep(self._system_status_interval)

    def _build_system_status_payload(self) -> dict[str, Any]:
        """
        Build the system status payload with platform-aware fallbacks.

        Returns:
            dict[str, Any]: Structured system status payload.
        """
        cpu_payload: dict[str, Any] = {"status": "unavailable"}
        memory_payload: dict[str, Any] = {"status": "unavailable"}
        storage_payload: dict[str, Any] = {"status": "unavailable"}
        pipeline_payload = self._build_pipeline_status_list()

        try:
            import psutil

            cpu_payload = {
                "percent": float(psutil.cpu_percent(interval=None)),
                "cores": int(psutil.cpu_count(logical=True) or 0),
                "status": "ok",
            }
            memory = psutil.virtual_memory()
            memory_payload = {
                "percent": float(memory.percent),
                "used_mb": float(memory.used / (1024 * 1024)),
                "total_mb": float(memory.total / (1024 * 1024)),
                "status": "ok",
            }
            disk = psutil.disk_usage("/")
            storage_payload = {
                "percent": float(disk.percent),
                "used_gb": float(disk.used / (1024 * 1024 * 1024)),
                "total_gb": float(disk.total / (1024 * 1024 * 1024)),
                "status": "ok",
            }
            self._system_status_error_logged = False
        except Exception as e:
            message = str(e)
            cpu_payload = {"status": "unavailable", "error": message}
            memory_payload = {"status": "unavailable", "error": message}
            storage_payload = {"status": "unavailable", "error": message}
            if not self._system_status_error_logged:
                self.log(f"System status metrics unavailable: {message}")
                self._system_status_error_logged = True

        return {
            "cpu": cpu_payload,
            "memory": memory_payload,
            "storage": storage_payload,
            "pipelines": pipeline_payload,
        }

    def _build_pipeline_status_list(self) -> list[dict[str, Any]]:
        """
        Build a list of pipelines with live active status.

        Returns:
            list[dict[str, Any]]: Pipeline status list.
        """
        try:
            pipeline_names = self.get_pipeline_names()
        except Exception as error:
            self.log(
                f"{Colors.RED}Error loading pipeline names for status: {error}{Colors.RESET}"
            )
            pipeline_names = []

        try:
            pipeline_objects = self.pipeline_objects_callback()
        except Exception as error:
            self.log(
                f"{Colors.RED}Error loading pipeline objects for status: {error}{Colors.RESET}"
            )
            pipeline_objects = {}

        statuses: list[dict[str, Any]] = []
        pipeline_objects_available = bool(pipeline_objects)
        for pipeline_name in pipeline_names:
            pipeline = pipeline_objects.get(pipeline_name)
            if pipeline is None:
                if pipeline_objects_available:
                    self.log(
                        f"{Colors.YELLOW}Pipeline {pipeline_name} not found in pipeline objects callback.{Colors.RESET}"
                    )
                statuses.append({"name": pipeline_name, "active": False})
                continue

            try:
                is_active = bool(pipeline.is_active())
            except Exception as error:
                self.log(
                    f"{Colors.RED}Failed to read active status for pipeline {pipeline_name}: {error}{Colors.RESET}"
                )
                is_active = False
            statuses.append({"name": pipeline_name, "active": is_active})

        return statuses

    def download_log_file(self) -> tuple[str, int] | tuple[dict, int]:
        """
        Download the log file.
        """
        try:
            with open(os.path.join(self.logger.current_log_file), "r") as f:
                return f.read(), 200
        except Exception as e:
            self.logger.log(
                f"{Colors.RED}Error downloading log file: {e}{Colors.RESET}"
            )
            return {"error": str(e)}, 500

    def get_general_conf(self) -> tuple[dict, int]:
        """
        Get the general configuration.
        """
        try:
            with open("general_conf.json", "r") as f:
                return json.load(f), 200
        except Exception as e:
            return {"error": str(e)}, 500

    def save_general_conf(self) -> tuple[dict, int]:
        """
        Save the general configuration.
        """
        try:
            with open("general_conf.json", "w") as f:
                json.dump(request.get_json(), f)
            return {"message": "General configuration saved successfully"}, 200
        except Exception as e:
            self.logger.log(
                f"{Colors.RED}Error saving general configuration: {e}{Colors.RESET}"
            )
            return {"error": str(e)}, 500

    def get_pipeline_thread_info(self, pipeline_name: str) -> tuple[dict, int]:
        """
        Get thread and timestep information for a pipeline.

        Args:
            pipeline_name: Name of pipeline.

        Returns:
            tuple[dict, int]: Dictionary containing total_threads and operations dict
                with thread and timestep for each operation, plus HTTP status code.
        """
        try:
            pipeline = self.pipeline_objects_callback()[pipeline_name]
            return pipeline.get_pipeline_thread_info(), 200
        except KeyError:
            return {"error": PIPELINE_NOT_FOUND_MESSAGE}, 404


if __name__ == "__main__":
    from src.utils.logging.logger import Logger  # noqa: E402

    logger = Logger()
    interface = EagleEyeInterface(
        dev_mode=False,
        logger=logger,
        restart_callback=lambda: None,
        pipeline_objects_callback=lambda: {},
    )

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.log(f"{Colors.CYAN}Program terminated.{Colors.RESET}")
