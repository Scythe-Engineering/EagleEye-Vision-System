from __future__ import annotations

import json
import os
import tempfile
from typing import Any

from flask import request

from src.webui.web_server_utils.constants import PIPELINE_NOT_FOUND_MESSAGE, SRC_DIR


LIMIT_FRAMES_TO_CAMERA_CAPTURE_SPEED_KEY = "limit_frames_to_camera_capture_speed"
DEFAULT_LIMIT_FRAMES_TO_CAMERA_CAPTURE_SPEED = True


class PipelineSettingsMixin:
    """Persist per-pipeline settings that apply after backend restart."""

    def _pipeline_settings_path(self) -> str:
        """Return the absolute path to the persisted pipeline settings file."""
        return os.path.join(SRC_DIR, "config", "pipeline_settings.json")

    def _load_pipeline_settings_file(self) -> dict[str, dict[str, Any]]:
        """Load pipeline settings, treating a missing settings file as empty."""
        try:
            with open(self._pipeline_settings_path(), "r", encoding="utf-8") as file:
                settings = json.load(file)
        except FileNotFoundError:
            return {}
        except (OSError, json.JSONDecodeError) as error:
            self.log(f"Failed to load pipeline settings: {error}")
            return {}

        if not isinstance(settings, dict):
            self.log("Failed to load pipeline settings: root value must be an object")
            return {}

        return {
            pipeline_name: pipeline_settings
            for pipeline_name, pipeline_settings in settings.items()
            if isinstance(pipeline_name, str) and isinstance(pipeline_settings, dict)
        }

    def _write_pipeline_settings_file(
        self, settings: dict[str, dict[str, Any]]
    ) -> None:
        """Atomically serialize and persist all pipeline settings."""
        settings_path = self._pipeline_settings_path()
        file_descriptor, temporary_path = tempfile.mkstemp(
            dir=os.path.dirname(settings_path),
            prefix=".pipeline_settings.",
            suffix=".tmp",
            text=True,
        )
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
                json.dump(settings, file, indent=4)
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_path, settings_path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

    def _pipeline_setting_enabled(self, pipeline_name: str) -> bool:
        """Return a pipeline's frame-limiting setting with its safe default."""
        settings = self._load_pipeline_settings_file().get(pipeline_name, {})
        value = settings.get(LIMIT_FRAMES_TO_CAMERA_CAPTURE_SPEED_KEY)
        return (
            value
            if isinstance(value, bool)
            else DEFAULT_LIMIT_FRAMES_TO_CAMERA_CAPTURE_SPEED
        )

    def _pipeline_exists(self, pipeline_name: str) -> bool:
        """Return whether a pipeline name is present in pipeline_config.json."""
        return pipeline_name in self._load_pipeline_config_file()

    def get_pipeline_settings(
        self, pipeline_name: str
    ) -> tuple[dict[str, bool | str], int]:
        """Return persisted settings for a known pipeline, including defaults."""
        if not self._pipeline_exists(pipeline_name):
            return {"message": PIPELINE_NOT_FOUND_MESSAGE}, 404

        return {
            LIMIT_FRAMES_TO_CAMERA_CAPTURE_SPEED_KEY: self._pipeline_setting_enabled(
                pipeline_name
            )
        }, 200

    def save_pipeline_settings(
        self, pipeline_name: str
    ) -> tuple[dict[str, bool | str], int]:
        """Validate and persist a pipeline setting that requires restart."""
        if not self._pipeline_exists(pipeline_name):
            return {"message": PIPELINE_NOT_FOUND_MESSAGE}, 404

        payload = request.get_json(silent=True)
        enabled = (
            payload.get(LIMIT_FRAMES_TO_CAMERA_CAPTURE_SPEED_KEY)
            if isinstance(payload, dict)
            else None
        )
        if not isinstance(enabled, bool):
            return {
                "error": "Expected boolean field "
                f"'{LIMIT_FRAMES_TO_CAMERA_CAPTURE_SPEED_KEY}'"
            }, 400

        with self._pipeline_settings_lock:
            settings = self._load_pipeline_settings_file()
            settings[pipeline_name] = {
                LIMIT_FRAMES_TO_CAMERA_CAPTURE_SPEED_KEY: enabled
            }
            self._write_pipeline_settings_file(settings)

        self.restart_required_for_config = True
        return {
            LIMIT_FRAMES_TO_CAMERA_CAPTURE_SPEED_KEY: enabled,
            "restart_required": True,
        }, 200

    def remove_pipeline_settings(self, pipeline_name: str) -> None:
        """Remove a deleted pipeline's persisted settings entry, if present."""
        with self._pipeline_settings_lock:
            settings = self._load_pipeline_settings_file()
            if pipeline_name not in settings:
                return
            del settings[pipeline_name]
            self._write_pipeline_settings_file(settings)
