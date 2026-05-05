from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.colors import Colors
from src.webui.web_server_utils.constants import (
    DEFAULT_GENERAL_CONF,
    DEFAULT_VIEW_STREAM_DOWNSCALE,
    MAX_VIEW_STREAM_DOWNSCALE,
    MIN_VIEW_STREAM_DOWNSCALE,
    VIEW_STREAM_DOWNSCALE_KEY,
)


def _request():
    """Return the current Flask request, allowing monkeypatching via web_server.request."""
    import src.webui.web_server as _ws

    return _ws.request


def _general_conf_path():
    """Return GENERAL_CONF_PATH, allowing monkeypatching via web_server.GENERAL_CONF_PATH."""
    import src.webui.web_server as _ws

    return _ws.GENERAL_CONF_PATH


class SystemMonitorMixin:
    def shutdown(self) -> tuple[dict, int]:
        """
        Shutdown the web interface.

        Returns:
            tuple[dict, int]: A success or failure message.
        """
        try:
            import os

            os._exit(0)
        except Exception as e:
            self.log(f"Error during shutdown: {e}")
            return {"message": "Failed to shutdown server"}, 500

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
        body = _request().get_json(silent=True) or {}
        required = bool(body.get("required", True))
        self.restart_required_for_config = required
        return {
            "message": "Restart required flag updated",
            "restart_required": self.restart_required_for_config,
            "runtime_id": getattr(self, "runtime_id", ""),
        }, 200

    def get_restart_required(self) -> tuple[dict, int]:
        """
        Get the restart required flag.
        """
        restart_required = bool(self.restart_required_for_config)
        try:
            restart_state = self._analyze_pipeline_restart_state(
                self._load_pipeline_config_file()
            )
            restart_required = restart_required or bool(
                restart_state.get("restart_required", False)
            )
            self.restart_required_for_config = restart_required
        except Exception as error:
            self.log(f"Failed to analyze pipeline restart state: {error}")

        return {
            "restart_required": restart_required,
            "runtime_id": getattr(self, "runtime_id", ""),
        }, 200

    def _has_active_wifi_connection(self) -> bool:
        try:
            if not hasattr(self, "_run_nmcli"):
                return False
            result = self._run_nmcli(
                ["-t", "-f", "TYPE,STATE", "device", "status"],
                timeout=5.0,
            )
            if result.returncode != 0:
                return False
            for line in result.stdout.splitlines():
                parts = line.split(":")
                if len(parts) >= 2 and parts[0] == "wifi" and parts[1] == "connected":
                    return True
        except Exception:
            return False
        return False

    def _has_internet_access(self) -> bool:
        try:
            socket.create_connection(("github.com", 443), timeout=3.0).close()
            return True
        except OSError:
            return False

    def system_update_status(self) -> tuple[dict, int]:
        """Return whether system update can run over WiFi with internet."""
        if not self._has_active_wifi_connection():
            return {
                "available": False,
                "reason": "Connect to a WiFi network before updating.",
            }, 200

        if not self._has_internet_access():
            return {
                "available": False,
                "reason": "Connected WiFi network does not appear to have internet access.",
            }, 200

        return {"available": True, "reason": "WiFi internet access available."}, 200

    def _repo_root(self) -> Path:
        return Path(__file__).resolve().parents[3]

    def _run_update_command(self, command: list[str], timeout: float) -> str:
        result = subprocess.run(
            command,
            cwd=self._repo_root(),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = "\n".join(
            part.strip() for part in [result.stdout, result.stderr] if part.strip()
        )
        if result.returncode != 0:
            raise RuntimeError(output or f"Command failed: {' '.join(command)}")
        return output

    def run_system_update(self) -> tuple[dict, int]:
        """Run git pull and apt package updates before the frontend restarts backend."""
        status_payload, _ = self.system_update_status()
        if not status_payload.get("available"):
            return {"error": status_payload.get("reason", "WiFi internet required")}, 400

        output_parts: list[str] = []
        try:
            output_parts.append("$ git pull")
            output_parts.append(self._run_update_command(["git", "pull"], timeout=120.0))
            output_parts.append("$ sudo apt update")
            output_parts.append(
                self._run_update_command(["sudo", "apt", "update"], timeout=300.0)
            )
            output_parts.append("$ sudo DEBIAN_FRONTEND=noninteractive apt upgrade -y")
            output_parts.append(
                self._run_update_command(
                    [
                        "sudo",
                        "env",
                        "DEBIAN_FRONTEND=noninteractive",
                        "apt",
                        "upgrade",
                        "-y",
                    ],
                    timeout=1800.0,
                )
            )
        except subprocess.TimeoutExpired as error:
            message = f"Update command timed out: {' '.join(error.cmd)}"
            self.log(message)
            return {"error": message, "output": "\n".join(output_parts)}, 504
        except Exception as error:
            message = str(error)
            self.log(f"System update failed: {message}")
            return {"error": message, "output": "\n".join(output_parts)}, 500

        output = "\n".join(part for part in output_parts if part)
        self.log("System update completed successfully")
        return {"success": True, "output": output}, 200

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
        network_table_payload = self._build_network_table_status()

        try:
            import psutil

            cpu_payload = {
                "percent": float(psutil.cpu_percent(interval=None)),
                "cores": int(psutil.cpu_count(logical=True) or 0),
                "temperature_c": self._read_cpu_temperature_c(psutil),
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
            "network_table": network_table_payload,
        }

    def _read_cpu_temperature_c(self, psutil_module: Any) -> float | None:
        """Read CPU temperature in Celsius, preferring Linux thermal sensors."""
        try:
            thermal_zones = sorted(Path("/sys/class/thermal").glob("thermal_zone*/temp"))
            temperatures: list[float] = []
            for temp_path in thermal_zones:
                try:
                    raw_value = temp_path.read_text(encoding="utf-8").strip()
                    value = float(raw_value)
                    if value > 1000:
                        value /= 1000.0
                    if 0.0 < value < 150.0:
                        temperatures.append(value)
                except Exception:
                    continue
            if temperatures:
                return max(temperatures)
        except Exception:
            pass

        try:
            sensors_temperatures = getattr(psutil_module, "sensors_temperatures", None)
            if sensors_temperatures is None:
                return None
            readings = sensors_temperatures(fahrenheit=False)
            temperatures = []
            for entries in readings.values():
                for entry in entries:
                    current = getattr(entry, "current", None)
                    if isinstance(current, (int, float)) and 0.0 < float(current) < 150.0:
                        label = str(getattr(entry, "label", "") or "").lower()
                        if "cpu" in label or "core" in label or not label:
                            temperatures.append(float(current))
            if temperatures:
                return max(temperatures)
        except Exception:
            pass

        return None

    def _build_network_table_status(self) -> dict[str, Any]:
        """
        Build NetworkTables client connection status for the frontend.

        Returns:
            dict[str, Any]: Connection status and configured server address.
        """
        try:
            server_address = self._read_general_conf().get("network_table_address", "")
        except Exception:
            server_address = ""

        instance = getattr(self, "network_table_instance", None)
        if instance is None:
            return {
                "status": "unavailable",
                "connected": False,
                "server": server_address,
                "connection_count": 0,
            }

        try:
            connected = bool(instance.isConnected())
            connections = instance.getConnections()
            return {
                "status": "ok",
                "connected": connected,
                "server": server_address,
                "connection_count": len(connections),
            }
        except Exception as error:
            return {
                "status": "unavailable",
                "connected": False,
                "server": server_address,
                "connection_count": 0,
                "error": str(error),
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

    def _read_general_conf(self) -> dict[str, Any]:
        """Read general config with defaults for missing optional settings."""
        conf_path = _general_conf_path()
        if not conf_path.exists():
            return DEFAULT_GENERAL_CONF.copy()

        with conf_path.open("r", encoding="utf-8") as f:
            config = json.load(f)

        if not isinstance(config, dict):
            raise ValueError("General configuration must be a JSON object")

        return {**DEFAULT_GENERAL_CONF, **config}

    def _parse_view_stream_downscale(self, value: Any) -> float:
        """Validate the view stream downscale setting."""
        try:
            downscale = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError("View stream downscale must be a number") from error

        if not MIN_VIEW_STREAM_DOWNSCALE <= downscale <= MAX_VIEW_STREAM_DOWNSCALE:
            raise ValueError(
                "View stream downscale must be between "
                f"{MIN_VIEW_STREAM_DOWNSCALE} and {MAX_VIEW_STREAM_DOWNSCALE}"
            )

        return downscale

    def _refresh_view_stream_settings(self) -> None:
        """Load view stream settings from the general configuration file."""
        try:
            general_conf = self._read_general_conf()
            downscale = self._parse_view_stream_downscale(
                general_conf.get(
                    VIEW_STREAM_DOWNSCALE_KEY,
                    DEFAULT_VIEW_STREAM_DOWNSCALE,
                )
            )
        except Exception as error:
            self.log(f"Failed loading view stream settings, using defaults: {error}")
            downscale = DEFAULT_VIEW_STREAM_DOWNSCALE

        with self._general_conf_lock:
            self.view_stream_downscale = downscale

    def get_general_conf(self) -> tuple[dict, int]:
        """
        Get the general configuration.
        """
        try:
            return self._read_general_conf(), 200
        except Exception as e:
            return {"error": str(e)}, 500

    def save_general_conf(self) -> tuple[dict, int]:
        """
        Save the general configuration.
        """
        try:
            payload = _request().get_json(silent=True)
            if not isinstance(payload, dict):
                return {"error": "Expected JSON object payload"}, 400

            config = {**self._read_general_conf(), **payload}
            config[VIEW_STREAM_DOWNSCALE_KEY] = self._parse_view_stream_downscale(
                config.get(
                    VIEW_STREAM_DOWNSCALE_KEY,
                    DEFAULT_VIEW_STREAM_DOWNSCALE,
                )
            )

            with _general_conf_path().open("w", encoding="utf-8") as f:
                json.dump(config, f, indent=4)
                f.write("\n")

            with self._general_conf_lock:
                self.view_stream_downscale = config[VIEW_STREAM_DOWNSCALE_KEY]

            return {"message": "General configuration saved successfully"}, 200
        except ValueError as e:
            return {"error": str(e)}, 400
        except Exception as e:
            self.logger.log(
                f"{Colors.RED}Error saving general configuration: {e}{Colors.RESET}"
            )
            return {"error": str(e)}, 500

    def update_robot_position(self, transformation_matrix: np.ndarray) -> None:
        """
        Push the tracked robot's transformation matrix to the frontend via SSE.

        Args:
            transformation_matrix (np.ndarray): The new transformation matrix as a 4x4 numpy array.
        """
        if transformation_matrix.shape != (4, 4):
            raise ValueError("Transformation matrix must be a 4x4 numpy array.")

        if not np.all(np.isfinite(transformation_matrix)):
            self.log("Skipping publish of robot transform due to non-finite values")
            return

        matrix_list = transformation_matrix.tolist()
        try:
            self._publish_event(
                "update_robot_transform", {"transform_matrix": matrix_list}
            )
        except Exception:
            self.log("Failed to publish update_robot_transform via SSE")

    def update_camera_pose(
        self, camera_bus_id: str, transformation_matrix: np.ndarray
    ) -> None:
        """Publish a camera pose update for 3D visualization via SSE.

        Args:
            camera_bus_id: Stable camera identifier used across pipelines and UI.
            transformation_matrix: 4x4 camera pose transform in world space.
        """
        if transformation_matrix.shape != (4, 4):
            raise ValueError("Transformation matrix must be a 4x4 numpy array.")

        if not np.all(np.isfinite(transformation_matrix)):
            self.log("Skipping publish of camera transform due to non-finite values")
            return

        resolved_camera_name: str | None = None
        for camera_name, camera_info in self.available_cameras.items():
            if not isinstance(camera_info, dict):
                continue
            if str(camera_info.get("bus_id") or "") != str(camera_bus_id):
                continue
            resolved_camera_name = str(camera_name)
            break

        try:
            self._publish_event(
                "update_camera_pose",
                {
                    "camera_bus_id": str(camera_bus_id),
                    "camera_name": resolved_camera_name or str(camera_bus_id),
                    "transform_matrix": transformation_matrix.tolist(),
                    "timestamp_ms": int(time.time() * 1000),
                },
            )
        except Exception:
            self.log("Failed to publish update_camera_pose via SSE")

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
