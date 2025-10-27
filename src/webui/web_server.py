import json
import logging
import os
import queue
import threading
import time
import traceback
from threading import Thread
from typing import Any, Callable, Generator

import cv2
import numpy as np
from flask import Flask, Response, request, send_from_directory
from flask_cors import CORS
from src.webui.web_server_utils.serve_static_files import (
    serve_css,
    serve_index,
    serve_js,
)
from src.utils.colors import Colors

current_path = os.path.dirname(__file__)
src_path = current_path.split("/src")[0] + "/src"

with open(os.path.join(current_path, "assets", "no_image.png"), "rb") as f:
    no_image_bytes = f.read()

no_image = cv2.imdecode(np.frombuffer(no_image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
success, _noimg_jpeg = cv2.imencode(".jpg", no_image)
no_image_jpeg_bytes: bytes = _noimg_jpeg.tobytes() if success else b""


class EagleEyeInterface:
    def __init__(
        self,
        restart_callback: Callable,
        pipeline_objects_callback: Callable,
        settings_object=None,
        dev_mode: bool = False,
        log: Callable | None = None,
    ):
        """
        Initialize the EagleEyeInterface.

        Starts a Flask server in a separate thread.

        Args:
            settings_object (Constants | None): Optional settings object.
            dev_mode (bool): Whether to run in development mode.
            log (Callable | None): Optional logging function.
        """
        if log is None:

            def colored_log(*messages: object) -> None:
                """Log function with automatic color coding based on message content."""
                message = " ".join(str(m) for m in messages)
                if any(
                    word in message.lower() for word in ["error", "failed", "exception"]
                ):
                    print(f"{Colors.RED}{message}{Colors.RESET}")
                elif any(
                    word in message.lower()
                    for word in ["success", "added", "updated", "started"]
                ):
                    print(f"{Colors.GREEN}{message}{Colors.RESET}")
                elif any(
                    word in message.lower()
                    for word in ["warning", "skipping", "queue full"]
                ):
                    print(f"{Colors.YELLOW}{message}{Colors.RESET}")
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
                    print(f"{Colors.CYAN}{message}{Colors.RESET}")
                else:
                    print(message)

            self.log = colored_log
        else:
            self.log = log

        self.restart_callback = restart_callback
        self.pipeline_objects_callback = pipeline_objects_callback

        self.restart_required_for_config = False

        self.app = Flask(
            __name__,
            static_folder=current_path,
            static_url_path="",
        )
        CORS(
            self.app,
            resources={
                r"/*": {"origins": ["http://localhost:5173", "http://localhost:5001"]}
            },
            supports_credentials=True,
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

        if settings_object is None:
            self.settings_object = None
        else:
            self.settings_object = settings_object

        self._register_routes()

        if dev_mode:
            self.run()
        else:
            # Run Flask normally; SSE streams are served from a route
            self.app_thread = Thread(
                target=self.app.run,
                args=("0.0.0.0", 5001),
                kwargs={"debug": False, "use_reloader": False},
                daemon=True,
            )
            time.sleep(
                5
            )  # might prevent an error, idk bruh, whent away when I added this
            self.app_thread.start()

        # Start heartbeat publisher thread for connection tracking
        self._heartbeat_interval = 5.0
        Thread(target=self._sse_heartbeat_loop, daemon=True).start()

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
            "/start-visualize/<string:camera_name>/<string:pipeline_name>",
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
            "/visualize/<string:camera_name>/<string:pipeline_name>/<string:action_name>",
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
            self.log("Error during shutdown:", e)
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

        self.log("SSE client connected")
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
            self.log("SSE client disconnected")

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
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.log(f"Error loading config for operation {operation_name}: {e}")
            return {}

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
            return config[camera_name][pipeline_name]

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
                operation_params = operation["action_params"]

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

    def start_visualize(self, camera_name: str, pipeline_name: str) -> tuple[dict, int]:
        """
        Start visualizing the pipeline.
        """
        self.pipeline_objects_callback()[camera_name][pipeline_name].start_visualize()
        return {"message": "Pipeline visualized successfully"}, 200

    def stop_visualize(self, camera_name: str, pipeline_name: str) -> tuple[dict, int]:
        """
        Stop visualizing the pipeline.
        """
        self.pipeline_objects_callback()[camera_name][pipeline_name].stop_visualize()
        return {"message": "Pipeline visualized stopped"}, 200

    def visualize(
        self, camera_name: str, pipeline_name: str, action_name: str
    ) -> Response:
        """
        Visualize the pipeline up to the given action name.

        Returns the image as JPEG binary data.
        """
        # Get the numpy array from the pipeline's visualize method
        image_array = self.pipeline_objects_callback()[camera_name][
            pipeline_name
        ].visualize(action_name)

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


if __name__ == "__main__":
    interface = EagleEyeInterface(dev_mode=False)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"{Colors.CYAN}Program terminated.{Colors.RESET}")
