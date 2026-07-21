from __future__ import annotations
import gzip
import json
import logging
import os
import queue
import threading
import time
import traceback
from pathlib import Path
from threading import Thread
from typing import TYPE_CHECKING, Any, Callable, Generator

from flask import Flask, Response, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO
from werkzeug.exceptions import HTTPException
from werkzeug.serving import make_server

from src.utils.colors import Colors
from src.utils.logging.logger import Logger
from src.utils.camera_utils.camera_config_manager import (
    CameraConfigRegistry,
)

if TYPE_CHECKING:
    from src.config.utils.pipeline import Pipeline
from src.webui.web_server_utils.serve_static_files import (
    STATIC_DIR,
    serve_css,
    serve_index,
    serve_js,
)
from src.webui.web_server_utils.draco_asset_cache import (
    DracoAssetCache,
    default_gltf_transform_bin,
)

from src.webui.web_server_utils.constants import (
    CORS_ALLOWED_ORIGINS,
    DEFAULT_GENERAL_CONF,
    DEFAULT_VIEW_STREAM_DOWNSCALE,
    GENERAL_CONF_PATH,  # noqa: F401 (for tests)
    PIPELINE_NOT_FOUND_MESSAGE,  # noqa: F401 (for tests)
    PROFILING_PUBLISH_INTERVAL_SECONDS,
    SRC_DIR,
    SSE_SERIALIZATION_WARN_INTERVAL_SECONDS,
    TEXT_PLAIN_MIMETYPE,  # noqa: F401 (for tests)
    WEBUI_DIR,
    WEB_SERVER_HOST,
    WEB_SERVER_PORT,
    VIEW_STREAM_DOWNSCALE_KEY,
)

# Re-export for external callers (main_backend.py, tests)
__all__ = [
    "EagleEyeInterface",
    "DEFAULT_GENERAL_CONF",
    "DEFAULT_VIEW_STREAM_DOWNSCALE",
    "VIEW_STREAM_DOWNSCALE_KEY",
    "WEB_SERVER_HOST",
    "WEB_SERVER_PORT",
]
from src.webui.web_server_utils.asset_manager_mixin import AssetManagerMixin
from src.webui.web_server_utils.camera_calibration_mixin import CameraCalibrationMixin
from src.webui.web_server_utils.camera_config_mixin import CameraConfigMixin
from src.webui.web_server_utils.camera_stream_mixin import CameraStreamMixin
from src.webui.web_server_utils.line_profiling_mixin import LineProfilingMixin
from src.webui.web_server_utils.network_manager_mixin import NetworkManagerMixin
from src.webui.web_server_utils.operation_config_mixin import OperationConfigMixin
from src.webui.web_server_utils.pipeline_config_mixin import PipelineConfigMixin
from src.webui.web_server_utils.system_monitor_mixin import SystemMonitorMixin
from src.webui.web_server_utils.test_video_mixin import TestVideoMixin
from src.webui.web_server_utils.visualization_mixin import VisualizationMixin

current_path = WEBUI_DIR
src_path = SRC_DIR


class EagleEyeInterface(
    AssetManagerMixin,
    CameraCalibrationMixin,
    CameraConfigMixin,
    CameraStreamMixin,
    LineProfilingMixin,
    NetworkManagerMixin,
    OperationConfigMixin,
    PipelineConfigMixin,
    SystemMonitorMixin,
    TestVideoMixin,
    VisualizationMixin,
):
    def __init__(
        self,
        restart_callback: Callable[[], None],
        pipeline_objects_callback: Callable[[], dict[str, Pipeline]],
        dev_mode: bool = False,
        logger: Logger | None = None,
        network_table_instance: Any | None = None,
    ):
        """
        Initialize the EagleEyeInterface.

        Starts a Flask server in a separate thread.

        Args:
            restart_callback: Callable invoked when a restart is requested.
            pipeline_objects_callback: Callable returning the live pipeline dict.
            dev_mode (bool): Whether to run in development mode.
            logger: Logger instance for logging.
            network_table_instance: Optional ntcore NetworkTableInstance for status reporting.
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
        self.runtime_id = f"{os.getpid()}-{time.time_ns()}"
        self.last_log_message_count = 0
        self.network_table_instance = network_table_instance
        self.view_stream_downscale = DEFAULT_VIEW_STREAM_DOWNSCALE
        self._general_conf_lock = threading.Lock()
        self._system_update_lock = threading.Lock()
        self._system_update_in_progress = False
        self._system_update_id = None
        self._latest_system_update_progress = None
        self._system_update_target_branch = None
        self._system_status_interval = 1.5
        self._system_status_error_logged = False
        self._refresh_view_stream_settings()

        self.app = Flask(
            __name__,
            static_folder=str(STATIC_DIR),
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
            async_mode="threading",
        )
        repo_root = Path(current_path).parents[1]
        self.draco_asset_cache = DracoAssetCache(
            assets_dir=Path(current_path) / "assets",
            cache_dir=Path(current_path) / "generated_assets" / "draco",
            gltf_transform_bin=default_gltf_transform_bin(repo_root),
            logger=self.log,
        )
        self.draco_asset_cache.prepare_all()

        self.app_thread: Thread | None = None
        self._http_server = None

        # Disable Werkzeug access logging (HTTP request logs)
        logging.getLogger("werkzeug").setLevel(logging.WARNING)

        class _SuppressHandshakeErrors(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                return "read error in handshake" not in record.getMessage()

        logging.getLogger("werkzeug").addFilter(_SuppressHandshakeErrors())

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

        self._register_response_optimizations()
        self._register_error_handlers()
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

    def _register_error_handlers(self) -> None:
        """Register request error logging without changing Flask responses."""

        @self.app.after_request
        def _log_error_response(response: Response) -> Response:
            if response.status_code >= 400:
                self._log_serving_error(response)
            return response

        @self.app.errorhandler(Exception)
        def _log_and_raise(e: Exception):
            if isinstance(e, HTTPException):
                return e
            self.log(f"Error: {traceback.format_exc()}")
            return {"message": "Internal server error"}, 500

    def _register_response_optimizations(self) -> None:
        """Register low-bandwidth response optimizations for WebUI clients."""

        compressible_mimetypes = {
            "application/json",
            "application/javascript",
            "text/css",
            "text/html",
            "text/javascript",
            "text/plain",
        }
        min_compress_size_bytes = 512

        @self.app.after_request
        def _gzip_response(response: Response) -> Response:
            accept_encoding = request.headers.get("Accept-Encoding", "")
            if "gzip" not in accept_encoding.lower():
                return response

            if (
                response.status_code < 200
                or response.status_code >= 300
                or response.direct_passthrough
                or response.is_streamed
                or response.headers.get("Content-Encoding")
                or response.mimetype not in compressible_mimetypes
            ):
                return response

            payload = response.get_data()
            if len(payload) < min_compress_size_bytes:
                return response

            compressed = gzip.compress(payload, compresslevel=5)
            if len(compressed) >= len(payload):
                return response

            response.set_data(compressed)
            response.headers["Content-Encoding"] = "gzip"
            response.headers["Vary"] = "Accept-Encoding"
            response.headers["Content-Length"] = str(len(compressed))
            return response

    def _log_serving_error(self, response: Response) -> None:
        """Log failed HTTP responses with enough context to find frontend misses."""
        endpoint = request.endpoint or "<unmatched>"
        referrer = request.referrer or "-"
        remote_addr = request.headers.get("X-Forwarded-For", request.remote_addr or "-")
        self.log(
            "Serving error: "
            f"{response.status} for {request.method} {request.full_path.rstrip('?')} "
            f"endpoint={endpoint} remote_addr={remote_addr} referrer={referrer}"
        )

    def _register_routes(self) -> None:
        """
        Register all Flask endpoints.
        """
        self.app.add_url_rule("/", "index", lambda: serve_index())
        self.app.add_url_rule("/js/main.js", "script", lambda: serve_js())
        self.app.add_url_rule("/style.css", "style", lambda: serve_css())

        self.app.add_url_rule(
            "/background.webp",
            "background",
            lambda: send_from_directory(str(Path(__file__).resolve().parent / "assets"), "background.webp"),
        )
        self.app.add_url_rule(
            "/assets/<path:filename>",
            "webui_assets",
            self.serve_webui_asset,
        )
        self.app.add_url_rule(
            "/.well-known/appspecific/com.chrome.devtools.json",
            "chrome_devtools_probe",
            lambda: ("", 204),
        )
        self.app.add_url_rule(
            "/get-available-cameras",
            "get_available_cameras",
            self.get_available_cameras,
            methods=["GET"],
        )
        self.app.add_url_rule(
            "/test-videos",
            "get_test_videos",
            self.get_test_videos,
            methods=["GET"],
        )
        self.app.add_url_rule(
            "/test-videos",
            "upload_test_video",
            self.upload_test_video,
            methods=["POST"],
        )
        self.app.add_url_rule(
            "/test-videos/<path:filename>",
            "delete_test_video",
            self.delete_test_video,
            methods=["DELETE"],
        )
        self.app.add_url_rule(
            "/wifi-networks",
            "get_wifi_networks",
            self.get_wifi_networks,
            methods=["GET"],
        )
        self.app.add_url_rule(
            "/wifi-networks/status",
            "network_manager_status",
            self.network_manager_status,
            methods=["GET"],
        )
        self.app.add_url_rule(
            "/wifi-networks/connect",
            "connect_wifi_network",
            self.connect_wifi_network,
            methods=["POST"],
        )
        self.app.add_url_rule(
            "/wifi-networks/disconnect",
            "disconnect_wifi_network",
            self.disconnect_wifi_network,
            methods=["POST"],
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
            "/camera-config/<string:camera_bus_id>/calibration/feed",
            "calibration_feed",
            self.calibration_feed,
            methods=["GET"],
        )
        self.app.add_url_rule(
            "/camera-config/<string:camera_bus_id>/calibration/capture",
            "capture_calibration_frame",
            self.capture_calibration_frame,
            methods=["POST"],
        )
        self.app.add_url_rule(
            "/camera-config/<string:camera_bus_id>/calibration/frames",
            "get_calibration_frames",
            self.get_calibration_frames,
            methods=["GET"],
        )
        self.app.add_url_rule(
            "/camera-config/<string:camera_bus_id>/calibration/frames/<int:frame_index>",
            "get_calibration_frame_image",
            self.get_calibration_frame_image,
            methods=["GET"],
        )
        self.app.add_url_rule(
            "/camera-config/<string:camera_bus_id>/calibration/frames/<int:frame_index>",
            "delete_calibration_frame",
            self.delete_calibration_frame,
            methods=["DELETE"],
        )
        self.app.add_url_rule(
            "/camera-config/<string:camera_bus_id>/calibration/reset",
            "reset_calibration_frames",
            self.reset_calibration_frames,
            methods=["POST"],
        )
        self.app.add_url_rule(
            "/camera-config/<string:camera_bus_id>/calibration/run",
            "run_camera_calibration",
            self.run_camera_calibration,
            methods=["POST"],
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
            "apriltags_assets",
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
            "/robot-files",
            "get_robot_files",
            self.get_robot_files,
            methods=["GET"],
        )
        self.app.add_url_rule(
            "/robot-files",
            "upload_robot_file",
            self.upload_robot_file,
            methods=["POST"],
        )
        self.app.add_url_rule(
            "/robot-files/<path:filename>/scale",
            "save_robot_file_scale",
            self.save_robot_file_scale,
            methods=["POST"],
        )
        self.app.add_url_rule(
            "/robot-files/<path:filename>",
            "delete_robot_file",
            self.delete_robot_file,
            methods=["DELETE"],
        )
        self.app.add_url_rule(
            "/field-files",
            "get_field_files",
            self.get_field_files,
            methods=["GET"],
        )
        self.app.add_url_rule(
            "/field-files",
            "upload_field_file",
            self.upload_field_file,
            methods=["POST"],
        )
        self.app.add_url_rule(
            "/field-files/<string:year>/<path:filename>/scale",
            "save_field_file_scale",
            self.save_field_file_scale,
            methods=["POST"],
        )
        self.app.add_url_rule(
            "/field-files/<string:year>/<path:filename>",
            "delete_field_file",
            self.delete_field_file,
            methods=["DELETE"],
        )
        self.app.add_url_rule(
            "/get-robot-file/<path:filename>",
            "get_robot_file",
            self.serve_robot_file,
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
            "/get-operation-config-data-batch",
            "get_operation_config_data_batch",
            self.get_operation_config_data_batch,
            methods=["POST"],
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
            "/line-profiling/start/<string:pipeline_name>/<string:operation_uuid>",
            "start_line_profiling",
            self.start_line_profiling,
            methods=["POST"],
        )
        self.app.add_url_rule(
            "/line-profiling/stop/<string:pipeline_name>/<string:operation_uuid>",
            "stop_line_profiling",
            self.stop_line_profiling,
            methods=["POST"],
        )
        self.app.add_url_rule(
            "/line-profiling/status",
            "get_line_profiling_status",
            self.get_line_profiling_status,
            methods=["GET"],
        )
        self.app.add_url_rule(
            "/line-profiling/report/<string:pipeline_name>/<string:operation_uuid>",
            "get_line_profiling_report",
            self.get_line_profiling_report,
            methods=["GET"],
        )
        self.app.add_url_rule(
            "/system-update/status",
            "system_update_status",
            self.system_update_status,
            methods=["GET"],
        )
        self.app.add_url_rule(
            "/system-update/info",
            "system_update_info",
            self.system_update_info,
            methods=["GET"],
        )
        self.app.add_url_rule(
            "/system-update/run",
            "run_system_update",
            self.run_system_update,
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
        self.app_thread = Thread(
            target=self._serve_threaded_wsgi,
            daemon=True,
        )
        self.app_thread.start()

    def _serve_threaded_wsgi(self) -> None:
        """Serve the wrapped Flask app with a threaded WSGI server."""
        self._http_server = make_server(
            WEB_SERVER_HOST,
            WEB_SERVER_PORT,
            self.app,
            threaded=True,
        )
        self._http_server.serve_forever()

    def _format_sse(self, event: str, data: str) -> bytes:
        """
        Format an SSE message with a named event and JSON data payload.
        """
        return f"event: {event}\ndata: {data}\n\n".encode()

    def _sse_stream(self) -> Generator[bytes, Any, Any]:
        """
        Generator that yields SSE messages for a single client using a queue.
        """
        q: queue.Queue = queue.Queue(maxsize=100)
        # assume single client: set queue, replacing any existing queue
        with self._sse_queue_lock:
            self._sse_queue = q

        with self._pipeline_error_lock:
            self._pipeline_error_dirty_pipelines.update(
                self._pipeline_error_cache.keys()
            )

        self._publish_cached_pipeline_errors()
        if hasattr(self, "_replay_cached_system_update_progress"):
            self._replay_cached_system_update_progress()

        try:
            while True:
                try:
                    msg = q.get(timeout=1.0)
                    yield msg
                except queue.Empty:
                    yield b": keepalive\n\n"
                    continue
        except GeneratorExit:
            pass
        finally:
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
        with self._sse_queue_lock:
            q = self._sse_queue
        if q is not None:
            try:
                q.put_nowait(msg)
            except queue.Full:
                try:
                    q.get_nowait()
                    q.put_nowait(msg)
                    self.log(
                        f"SSE queue full, dropped oldest event to add {event_name}"
                    )
                except queue.Empty:
                    self.log(
                        f"SSE queue unexpectedly empty when trying to drop oldest for {event_name}"
                    )
                except queue.Full:
                    self.log(
                        f"SSE queue still full after dropping oldest, dropping {event_name} event"
                    )
            except Exception as e:
                self.log(f"SSE publish error for {event_name}: {e}")

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
