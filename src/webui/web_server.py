import os
import threading
import time
from threading import Thread
from typing import Any, Callable, Generator

import numpy as np
from flask import Flask, Response, request, send_from_directory
from flask_socketio import SocketIO

from src.main_operations.object_detection.src.constants.constants import Constants
from src.utils.camera_utils.get_available_cameras import (
    detect_cameras_with_names,
)
from src.webui.web_server_utils.serve_static_files import (
    serve_css,
    serve_index,
    serve_js,
)

current_path = os.path.dirname(__file__)

with open(os.path.join(current_path, "assets", "no_image.png"), "rb") as f:
    no_image = f.read()


class EagleEyeInterface:
    def __init__(
        self,
        settings_object: Constants | None = None,
        dev_mode: bool = False,
        log: Callable | None = None,
    ):
        """
        Initialize the EagleEyeInterface.

        Starts a Flask server in a separate thread.

        Args:
            settings_object (Constants | None): Optional settings object.
        """
        if log is None:
            self.log = print
        else:
            self.log = log

        self.app = Flask(__name__, static_folder=current_path, static_url_path="")
        self.socketio = SocketIO(
            self.app,
            cors_allowed_origins="*",
            async_mode="threading",
            ping_timeout=60,
            ping_interval=25,
            engineio_logger=False,
            socketio_logger=False,
        )

        self.cameras = detect_cameras_with_names()
        self.log(f"Detected Cameras: {self.cameras}")
        self.frame_list = {}
        for camera in self.cameras:
            self.frame_list[camera] = no_image
        self.available_cameras = {}

        self.frame_list_lock = threading.Lock()

        if settings_object is None:
            self.settings_object = Constants()
        else:
            self.settings_object = settings_object

        self._register_routes()

        if dev_mode:
            self.run()
        else:
            self.app_thread = Thread(
                target=self.socketio.run,
                args=(self.app,),
                kwargs={"host": "0.0.0.0", "port": 5001},
                daemon=True,
            )
            self.app_thread.start()

        @self.app.errorhandler(Exception)
        def _log_and_raise(e):
            self.log("Error:", e)
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
            "/save-settings", "save_settings", self.set_settings, methods=["POST"]
        )
        self.app.add_url_rule(
            "/get-settings", "get_settings", self.get_settings, methods=["GET"]
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
                os.path.join("../", "apriltags", "utils"), "frc2025r2.json"
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

    def get_available_cameras(self) -> dict:
        """
        Get a dict of available cameras.

        Returns:
            dict: A dict of available cameras.
        """
        self.cameras = detect_cameras_with_names()
        self.available_cameras = {}
        for camera_name in self.cameras:
            url_safe_name = camera_name.replace(" ", "_")
            self.available_cameras[camera_name] = url_safe_name
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
            extra_files=["./static/bundle.js", "./style.css", "./index.html"],
        )

    def get_settings(self) -> dict:
        """
        Get the current settings.

        Returns:
            dict: The current settings.
        """
        return self.settings_object.get_config()

    def set_settings(self) -> tuple[dict, int]:
        """
        Set the current settings.

        Returns:
            Response: A success or failure message.
        """
        try:
            settings = request.get_json()
            self.settings_object.load_config_from_json(settings)
            self.log("Settings updated successfully")
            return {"message": "Settings updated successfully"}, 200
        except Exception as e:
            self.log("Error updating settings:", e)
            return {"message": "Failed to update settings"}, 500

    def update_camera_frame(self, camera_name: str, frame: bytes) -> None:
        """
        Update the camera frame.

        Args:
            camera_name (str): The ID of the camera.
            frame: The frame to update.
        """
        with self.frame_list_lock:
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
            with self.frame_list_lock:
                frame = self.frame_list[camera_name]

            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"

            time.sleep(max((1 / 120) - (time.time() - time_start), 0))

    def _frame_generator_no_image(self) -> Generator[bytes, Any, Any]:
        """
        Generate no image frames when camera is not found.

        Yields:
            Generator: The no image feed.
        """
        while True:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + no_image + b"\r\n"
            time.sleep(1 / 30)

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

        # Convert matrix to list for JSON serialization
        matrix_list = transformation_matrix.tolist()
        self.socketio.emit("update_robot_transform", {"transform_matrix": matrix_list})
        self.socketio.sleep(0)

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
                if file.endswith(".glb")
            ]
        }


if __name__ == "__main__":
    interface = EagleEyeInterface(dev_mode=False)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Program terminated.")
