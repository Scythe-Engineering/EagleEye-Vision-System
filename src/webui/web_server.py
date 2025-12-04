import json
import logging
import os
import queue
import threading
import time
import traceback
from pathlib import Path
from threading import Thread
from typing import Any, Callable, Generator, List

import cv2
import numpy as np
from flask import Flask, Response, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO

from src.utils.colors import Colors
from src.utils.logging.logger import Logger
from src.webui.web_server_utils.serve_static_files import (
    serve_css,
    serve_index,
    serve_js,
)

current_path = os.path.dirname(__file__)
src_path = os.path.abspath(os.path.join(current_path, os.pardir))

with open(os.path.join(current_path, "assets", "no_image.png"), "rb") as f:
    no_image_bytes = f.read()

no_image = cv2.imdecode(np.frombuffer(no_image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
success, _noimg_jpeg = cv2.imencode(".jpg", no_image)
no_image_jpeg_bytes: bytes = _noimg_jpeg.tobytes() if success else b""

CORS_ALLOWED_ORIGINS = ["http://localhost:5173", "http://localhost:5001"]


class EagleEyeInterface:
    def __init__(
        self,
        restart_callback: Callable,
        pipeline_objects_callback: Callable,
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

        self.app = Flask(
            __name__,
            static_folder=current_path,
            static_url_path="",
        )
        CORS(
            self.app,
            resources={r"/*": {"origins": CORS_ALLOWED_ORIGINS}},
            supports_credentials=True,
        )

        self.socketio = SocketIO(
            self.app,
            cors_allowed_origins=CORS_ALLOWED_ORIGINS,
        )

        # Disable Werkzeug access logging (HTTP request logs)
        logging.getLogger("werkzeug").setLevel(logging.WARNING)
        # Simplified single-client SSE: one queue and a lock to guard it.
        self._sse_queue: queue.Queue | None = None
        self._sse_queue_lock = threading.Lock()

        self.cameras = {}
        self.log(f"Initialized with cameras: {self.cameras}")
        self.frame_list = {}
        self.available_cameras = {}

        self.frame_locks = {}
        self.frame_list_structure_lock = threading.Lock()

        self._register_routes()

        if dev_mode:
            self.run()
        else:
            # Run Flask with SocketIO for WebSocket support in production
            self.app_thread = Thread(
                target=self.socketio.run,
                args=(self.app,),
                kwargs={
                    "host": "0.0.0.0",
                    "port": 5001,
                    "debug": False,
                },
                daemon=True,
            )
            time.sleep(
                5
            )  # might prevent an error, idk bruh, whent away when I added this
            self.app_thread.start()

        # Start heartbeat publisher thread for connection tracking
        self._heartbeat_interval = 5.0
        Thread(target=self._sse_heartbeat_loop, daemon=True).start()

        # Start log monitoring thread for real-time log updates
        Thread(target=self._log_monitor_loop, daemon=True).start()

        @self.app.errorhandler(Exception)
        def _log_and_raise(_):
            self.log(f"Error: {traceback.format_exc()}")
            return {"message": "Internal server error"}, 500

    def _register_routes(self) -> None:
        """
        Register all Flask endpoints.
        """
        self.app.add_url_rule("/", "index", lambda: serve_index())
        self.app.add_url_rule("/script.js", "script", lambda: serve_js())
        self.app.add_url_rule("/main.css", "style", lambda: serve_css())

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
            "/get-pipeline-config/<string:camera_name>/<string:pipeline_name>",
            "get_pipeline_config",
            self.get_pipeline_config,
            methods=["GET"],
        )
        self.app.add_url_rule(
            "/get-pipeline-names-for-camera/<string:camera_name>",
            "get_pipeline_names_for_camera",
            self.get_pipeline_names_for_camera,
            methods=["GET"],
        )
        self.app.add_url_rule(
            "/save-pipeline-config/<string:camera_name>/<string:pipeline_name>",
            "save_pipeline_config",
            self.save_pipeline_config,
            methods=["POST"],
        )
        self.app.add_url_rule(
            "/delete-pipeline/<string:camera_name>/<string:pipeline_name>",
            "delete_pipeline",
            self.delete_pipeline,
            methods=["DELETE"],
        )
        self.app.add_url_rule(
            "/restart-backend",
            "restart_backend",
            self.restart_backend,
            methods=["POST"],
        )
        self.app.add_url_rule(
            "/start-visualize/<string:camera_name>/<string:pipeline_name>/<string:operation_name>",
            "start_visualize",
            self.start_visualize,
            methods=["POST"],
        )
        self.app.add_url_rule(
            "/stop-visualize/<string:camera_name>/<string:pipeline_name>",
            "stop_visualize",
            self.stop_visualize,
            methods=["POST"],
        )
        self.app.add_url_rule(
            "/visualize/<string:camera_name>/<string:pipeline_name>",
            "visualize",
            self.visualize,
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

    def add_camera(self, camera_name: str, camera_id: int | str | None = None) -> None:
        """
        Add a camera to the available cameras list.

        Args:
            camera_name (str): The name of the camera.
            camera_id (int | str | None, optional): The ID of the camera. If None, uses the camera name.
        """
        if camera_id is None:
            camera_id = camera_name

        with self.frame_list_structure_lock:
            self.cameras[camera_name] = camera_id
            if camera_name not in self.frame_list:
                self.frame_list[camera_name] = no_image
                self.frame_locks[camera_name] = threading.Lock()

            url_safe_name = camera_name.replace(" ", "_")
            self.available_cameras[camera_name] = url_safe_name

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

    def set_cameras(self, cameras_dict: dict[str, int | str]) -> None:
        """
        Set multiple cameras at once, replacing the current camera list.

        Args:
            cameras_dict (dict[str, int | str]): A dictionary mapping camera names to camera IDs.
        """
        with self.frame_list_structure_lock:
            self.cameras = cameras_dict.copy()
            self.frame_list = {}
            self.available_cameras = {}
            self.frame_locks = {}

            for camera_name in self.cameras:
                self.frame_list[camera_name] = no_image
                self.frame_locks[camera_name] = threading.Lock()
                url_safe_name = camera_name.replace(" ", "_")
                self.available_cameras[camera_name] = url_safe_name

        self.log(f"Set cameras: {self.cameras}")

    def get_available_cameras(self) -> dict:
        """
        Get a dict of available cameras.

        Returns:
            dict: A dict of available cameras.
        """
        return self.available_cameras

    def run(self) -> None:
        """
        Run the Flask application with SocketIO.
        """
        self.socketio.run(
            self.app,
            host="0.0.0.0",
            port=5001,
            debug=False,
            allow_unsafe_werkzeug=True,
            extra_files=["./static/bundle.js", "./style.css", "./index.html"],
        )

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
        payload = json.dumps(data, allow_nan=False)
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

    def _sse_heartbeat_loop(self) -> None:
        """
        Periodically publish a heartbeat event for connection tracking.
        """
        while True:
            try:
                self._publish_event("heartbeat", {"ts": time.time()})
                # Optional: Uncomment for verbose heartbeat logging
                # self.log(f"Heartbeat sent at {time.time()}")
            except Exception as e:
                self.log(f"Error sending heartbeat: {e}")
            time.sleep(self._heartbeat_interval)

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
            for orig_name, url_name in self.available_cameras.items():
                if url_name == camera_name:
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
                    }
                )

        return {
            "operations": main_operations + secondary_operations,
        }

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
                return json.load(f)
        except FileNotFoundError:
            # Don't log errors for missing configs when trying both locations
            return {}
        except json.JSONDecodeError as e:
            self.log(f"Error loading config for operation {operation_name}: {e}")
            return {}

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

    def _ensure_operation_directory(
        self, operation_name: str, parameter_name: str
    ) -> Path:
        """
        Ensure the operation's parameter-specific file directory exists.

        Args:
            operation_name: Name of the operation.
            parameter_name: Name of the parameter.

        Returns:
            Path to the parameter-specific directory.
        """
        files_base_dir = Path(src_path).parent / "files"
        operation_dir = files_base_dir / operation_name.lower().replace(
            " ", "_"
        ).replace(".py", "")
        parameter_dir = operation_dir / parameter_name
        parameter_dir.mkdir(parents=True, exist_ok=True)
        return parameter_dir

    def get_operation_files(
        self, operation_name: str, parameter_name: str
    ) -> tuple[dict, int]:
        """
        Get list of available files for an operation parameter.

        Args:
            operation_name: Name of the operation.
            parameter_name: Name of the parameter.

        Returns:
            Tuple of (response dict, status code).
        """
        try:
            parameter_dir = self._ensure_operation_directory(
                operation_name, parameter_name
            )
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
            operation_name: Name of the operation.
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

            parameter_dir = self._ensure_operation_directory(
                operation_name, parameter_name
            )
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
            operation_name: Name of the operation.
            parameter_name: Name of the parameter.
            filename: Name of the file to delete.

        Returns:
            Tuple of (response dict, status code).
        """
        try:
            parameter_dir = self._ensure_operation_directory(
                operation_name, parameter_name
            )
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

    def get_pipeline_config(self, camera_name: str, pipeline_name: str) -> list:
        """
        Get the config data for a pipeline.

        Args:
            camera_name (str): The name of the camera.
            pipeline_name (str): The name of the pipeline.

        Returns:
            dict: The config data for the pipeline.
        """
        with open(os.path.join(src_path, "config", "pipeline_config.json"), "r") as f:
            config = json.load(f)
            if camera_name not in config:
                return []
            if pipeline_name not in config[camera_name]:
                return []
            pipeline_config = config[camera_name][pipeline_name]

            # Reorder parameters according to config definitions
            reordered_pipeline = []
            for operation in pipeline_config:
                operation_name = operation["action_name"]
                action_params = self._reorder_operation_params(
                    operation_name, operation["action_params"]
                )

                reordered_pipeline.append(
                    {"action_name": operation_name, "action_params": action_params}
                )

            return reordered_pipeline

    def get_pipeline_names_for_camera(self, camera_name: str) -> list[str]:
        """
        Get the names of the pipelines for a camera.

        Args:
            camera_name (str): The name of the camera.

        Returns:
            list[str]: The names of the pipelines for the camera.
        """
        with open(os.path.join(src_path, "config", "pipeline_config.json"), "r") as f:
            config = json.load(f)
            if camera_name not in config:
                return []
            return list(config[camera_name].keys())

    def save_pipeline_config(
        self, camera_name: str, pipeline_name: str
    ) -> tuple[dict, int]:
        """
        Save the pipeline config.

        Args:
            camera_name (str): The name of the camera.
            pipeline_name (str): The name of the pipeline.

        Returns:
            tuple[dict, int]: A success or failure message.
        """
        with open(os.path.join(src_path, "config", "pipeline_config.json"), "r") as f:
            current_config = json.load(f)
            new_data = request.get_json()

            if camera_name not in current_config:
                current_config[camera_name] = {}

            if pipeline_name not in current_config[camera_name]:
                current_config[camera_name][pipeline_name] = []

            for operation in new_data:
                operation_name = operation["action_name"]
                operation_params = self._reorder_operation_params(
                    operation_name, operation["action_params"]
                )

                operation_names = [
                    operation["action_name"]
                    for operation in current_config[camera_name][pipeline_name]
                ]

                if operation_name in operation_names:
                    for key, value in operation_params.items():
                        current_config[camera_name][pipeline_name][
                            operation_names.index(operation_name)
                        ]["action_params"][key] = value
                else:
                    current_config[camera_name][pipeline_name].append(
                        {
                            "action_name": operation_name,
                            "action_params": operation_params,
                        }
                    )

        with open(os.path.join(src_path, "config", "pipeline_config.json"), "w") as f:
            json.dump(current_config, f, indent=4)

        pipeline_objects = self.pipeline_objects_callback()
        if (
            camera_name in pipeline_objects
            and pipeline_name in pipeline_objects[camera_name]
        ):
            pipeline_objects[camera_name][pipeline_name].update_operations_config(
                request.get_json()
            )

        return {"message": "Pipeline config saved successfully"}, 200

    def delete_pipeline(self, camera_name: str, pipeline_name: str) -> tuple[dict, int]:
        """
        Delete a pipeline.
        """
        with open(os.path.join(src_path, "config", "pipeline_config.json"), "r") as f:
            current_config = json.load(f)
            if (
                camera_name in current_config
                and pipeline_name in current_config[camera_name]
            ):
                del current_config[camera_name][pipeline_name]
            else:
                return {"message": "Pipeline not found"}, 404
        with open(os.path.join(src_path, "config", "pipeline_config.json"), "w") as f:
            json.dump(current_config, f, indent=4)
        return {"message": "Pipeline deleted successfully"}, 200

    def start_visualize(
        self, camera_name: str, pipeline_name: str, operation_name: str
    ) -> tuple[dict, int]:
        """
        Start visualizing the pipeline.

        Args:
            camera_name: Name of the camera whose pipeline should be visualized.
            pipeline_name: Name of the pipeline to visualize.
            operation_name: Name of the operation to visualize.

        Returns:
            A response message and HTTP status code.
        """
        self.pipeline_objects_callback()[camera_name][pipeline_name].start_visualize(
            operation_name
        )
        return {"message": "Pipeline visualized successfully"}, 200

    def stop_visualize(self, camera_name: str, pipeline_name: str) -> tuple[dict, int]:
        """
        Stop visualizing the pipeline.
        """
        self.pipeline_objects_callback()[camera_name][pipeline_name].stop_visualize()
        return {"message": "Pipeline visualized stopped"}, 200

    def visualize(self, camera_name: str, pipeline_name: str) -> Response:
        """
        Visualize the pipeline.

        Returns the image as JPEG binary data.
        """
        pipeline = self.pipeline_objects_callback()[camera_name][pipeline_name]

        # Get visualization data from pipeline
        with pipeline.visualization_data_lock:
            visualization_data = pipeline.visualization_data

        if visualization_data is None:
            return Response(
                "No visualization data available", status=500, mimetype="text/plain"
            )

        # Get the visualized frame from the visualization data
        image_array = visualization_data.get("visualization_data")

        if image_array is None:
            return Response(
                "Function has no visualization", status=500, mimetype="text/plain"
            )

        # Encode the numpy array to JPEG format
        success, encoded_image = cv2.imencode(".jpg", image_array)
        if not success:
            return Response("Failed to encode image", status=500, mimetype="text/plain")

        # Return the encoded image as binary data with proper content type
        return Response(encoded_image.tobytes(), mimetype="image/jpeg")

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

    def download_log_file(self) -> tuple[str, int] | tuple[dict, int]:
        """
        Download the log file.
        """
        try:
            with open(os.path.join(self.logger.current_log_file), "r") as f:
                return f.read(), 200
        except Exception as e:
            self.logger.log(f"Error downloading log file: {e}")
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
            self.logger.log(f"Error saving general configuration: {e}")
            return {"error": str(e)}, 500


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
