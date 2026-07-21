from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from werkzeug.utils import secure_filename

from src.webui.web_server_utils.constants import TEST_VIDEO_EXTENSION


def _request():
    """Return the current Flask request, allowing monkeypatching via web_server.request."""
    import src.webui.web_server as _ws
    return _ws.request


def _src_path() -> str:
    """Return the src directory path, allowing monkeypatching via web_server.src_path."""
    import src.webui.web_server as _ws
    return _ws.src_path


class TestVideoMixin:
    def _get_test_video_directory(self) -> Path:
        """Return the managed test video directory, creating it if needed.

        Returns:
            Path: Directory containing managed test video files.
        """
        video_dir = Path(_src_path()) / "utils" / "sim_videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        return video_dir

    def _sanitize_test_video_filename(self, filename: str | None) -> str:
        """Validate and sanitize a test video filename.

        Args:
            filename: Raw filename from an upload or URL path.

        Returns:
            str: Safe filename for storage.

        Raises:
            ValueError: If filename is empty, unsafe, or not an MP4 file.
        """
        raw_filename = (filename or "").strip()
        if not raw_filename:
            raise ValueError("No filename provided")

        if "/" in raw_filename or "\\" in raw_filename:
            raise ValueError("Path separators are not allowed in test video filenames")

        sanitized_filename = secure_filename(raw_filename)
        if not sanitized_filename:
            raise ValueError("Filename is not valid")

        if Path(sanitized_filename).name != sanitized_filename:
            raise ValueError("Path traversal is not allowed")

        if Path(sanitized_filename).suffix.lower() != TEST_VIDEO_EXTENSION:
            raise ValueError("Only .mp4 test videos are supported")

        return sanitized_filename

    def _resolve_test_video_path(self, filename: str) -> Path:
        """Resolve a managed test video path after filename validation.

        Args:
            filename: Safe test video filename.

        Returns:
            Path: Absolute path inside the managed test video directory.

        Raises:
            ValueError: If the resolved path escapes the managed directory.
        """
        video_dir = self._get_test_video_directory().resolve()
        video_path = (video_dir / filename).resolve()
        if video_path.parent != video_dir:
            raise ValueError("Test video path escapes managed directory")
        return video_path

    def _get_pipeline_references_for_bus_id(self, bus_id: str) -> list[str]:
        """Return pipeline names referencing a camera bus ID.

        Args:
            bus_id: Camera bus ID to search for.

        Returns:
            list[str]: Sorted pipeline names with device_input operations using
                the provided bus ID.
        """
        config_path = Path(_src_path()) / "config" / "pipeline_config.json"
        if not config_path.exists():
            return []

        try:
            with config_path.open("r", encoding="utf-8") as config_file:
                pipeline_config = json.load(config_file)
        except Exception as error:
            self.log(f"Failed reading pipeline config for test video references: {error}")
            return []

        references: set[str] = set()
        if not isinstance(pipeline_config, dict):
            return []

        for pipeline_name, operations in pipeline_config.items():
            if not isinstance(operations, list):
                continue

            for operation in operations:
                if not isinstance(operation, dict):
                    continue
                if operation.get("action_name") not in {"device_input", "device_input.py"}:
                    continue

                action_params = operation.get("action_params", {})
                if (
                    isinstance(action_params, dict)
                    and str(action_params.get("camera_bus_id", "")) == bus_id
                ):
                    references.add(str(pipeline_name))

        return sorted(references)

    def _is_request_flag_enabled(self, flag_name: str) -> bool:
        """Read a boolean flag from query args, form data, or JSON payload.

        Args:
            flag_name: Name of the flag to read.

        Returns:
            bool: True when the request contains a truthy flag value.
        """
        req = _request()
        raw_value: Any = req.args.get(flag_name)
        if raw_value is None:
            raw_value = req.form.get(flag_name)
        if raw_value is None:
            payload = req.get_json(silent=True)
            if isinstance(payload, dict):
                raw_value = payload.get(flag_name)

        if isinstance(raw_value, bool):
            return raw_value
        if raw_value is None:
            return False
        return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}

    def get_test_videos(self) -> tuple[dict, int]:
        """List managed MP4 test videos available for pipeline injection.

        Returns:
            tuple[dict, int]: Video list payload and HTTP status.
        """
        try:
            videos = []
            for video_path in self._get_test_video_directory().iterdir():
                if (
                    not video_path.is_file()
                    or video_path.suffix.lower() != TEST_VIDEO_EXTENSION
                ):
                    continue

                file_stat = video_path.stat()
                bus_id = video_path.stem
                videos.append(
                    {
                        "filename": video_path.name,
                        "bus_id": bus_id,
                        "size": file_stat.st_size,
                        "modified": file_stat.st_mtime,
                        "pipeline_references": self._get_pipeline_references_for_bus_id(
                            bus_id
                        ),
                    }
                )

            videos.sort(key=lambda video: str(video["filename"]).lower())
            return {"videos": videos}, 200
        except Exception as error:
            self.log(f"Error listing test videos: {error}")
            return {"error": str(error)}, 500

    def upload_test_video(self) -> tuple[dict, int]:
        """Upload a managed MP4 test video file.

        Returns:
            tuple[dict, int]: Upload result payload and HTTP status.
        """
        req = _request()
        if "file" not in req.files:
            return {"error": "No file provided"}, 400

        upload = req.files["file"]
        try:
            filename = self._sanitize_test_video_filename(upload.filename)
            video_path = self._resolve_test_video_path(filename)
        except ValueError as error:
            return {"error": str(error)}, 400

        overwrite = self._is_request_flag_enabled("overwrite")
        if video_path.exists() and not overwrite:
            return {
                "error": "A test video with this filename already exists",
                "filename": filename,
                "requires_overwrite": True,
            }, 409

        try:
            upload.save(str(video_path))
            file_stat = video_path.stat()
            bus_id = video_path.stem
            self.log(f"Uploaded test video {filename}")
            return {
                "success": True,
                "video": {
                    "filename": filename,
                    "bus_id": bus_id,
                    "size": file_stat.st_size,
                    "modified": file_stat.st_mtime,
                    "pipeline_references": self._get_pipeline_references_for_bus_id(
                        bus_id
                    ),
                },
            }, 200
        except Exception as error:
            self.log(f"Error uploading test video: {error}")
            return {"error": "Failed to upload test video"}, 500

    def delete_test_video(self, filename: str) -> tuple[dict, int]:
        """Delete a managed MP4 test video file.

        Args:
            filename: Filename to delete from the managed test video directory.

        Returns:
            tuple[dict, int]: Delete result payload and HTTP status.
        """
        try:
            safe_filename = self._sanitize_test_video_filename(filename)
            video_path = self._resolve_test_video_path(safe_filename)
        except ValueError as error:
            return {"error": str(error)}, 400

        if not video_path.exists():
            return {"error": "Test video not found"}, 404
        if not video_path.is_file():
            return {"error": "Test video path is not a file"}, 400

        bus_id = video_path.stem
        pipeline_references = self._get_pipeline_references_for_bus_id(bus_id)
        force_delete = self._is_request_flag_enabled("force")
        if pipeline_references and not force_delete:
            return {
                "error": "Test video is referenced by pipelines",
                "filename": safe_filename,
                "bus_id": bus_id,
                "pipeline_references": pipeline_references,
                "requires_force": True,
            }, 409

        try:
            video_path.unlink()
            self.log(f"Deleted test video {safe_filename}")
            return {
                "success": True,
                "filename": safe_filename,
                "bus_id": bus_id,
                "pipeline_references": pipeline_references,
            }, 200
        except Exception as error:
            self.log(f"Error deleting test video: {error}")
            return {"error": "Failed to delete test video"}, 500
