import os
import threading
import time
from threading import Thread
from typing import Any, Callable, Generator

import cv2
import numpy as np
from flask import Flask, Response, request, send_from_directory
from flask_socketio import SocketIO

from src.object_detection.src.constants.constants import Constants
from src.object_detection.src.devices.utils.get_available_cameras import (
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

    def serve_camera_feed(self, camera_name: str, direct_serve: bool = False) -> None:
        """
        Serve the camera feed.

        Args:
            camera_name (str): The ID of the camera.
            direct_serve (bool): Whether to directly serve the camera feed.

        Returns:
            Response: The camera feed.
        """
        # Create URL path and unique endpoint
        url_name = camera_name.replace(" ", "_")
        route = f"/feed/{url_name}"
        endpoint = f"feed_{url_name}"

        # Define view function for this camera
        def _make_feed(name: str = camera_name) -> Response:
            return Response(
                self._frame_generator(name),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        # Register the route with a unique endpoint
        self.app.add_url_rule(route, endpoint, _make_feed, methods=["GET"])

        if direct_serve:
            camera_thread = Thread(
                target=self._update_camera_feed, args=(camera_name,), daemon=True
            )
            camera_thread.start()

        self.available_cameras[camera_name] = self.cameras[camera_name]

        self.log(
            f"Serving camera feed for {camera_name} at /feed/{camera_name.replace(' ', '_')}"
        )

    def _update_camera_feed(self, camera_name: str) -> None:
        """
        Update the camera feed.

        Args:
            camera_name (str): The ID of the camera.
        """
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not camera.isOpened():
            self.log(f"Error: Unable to open camera: {camera_name}.")
            return

        try:
            time_start = time.time()
            while True:
                ret, frame = camera.read()

                if not ret:
                    self.log(
                        f"Warning: Unable to grab frame from camera {camera_name}."
                    )
                    time.sleep(1)
                    continue

                ret, buffer = cv2.imencode(".jpg", frame)
                frame_bytes = buffer.tobytes()

                # Check if processing time exceeds target frame time
                processing_time = time.time() - time_start
                if processing_time > (1 / 30):
                    self.update_camera_frame(camera_name, frame_bytes)
                    time_start = time.time()
        finally:
            camera.release()

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
