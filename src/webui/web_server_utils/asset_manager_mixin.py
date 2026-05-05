from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from flask import send_from_directory
from werkzeug.utils import secure_filename


def _request():
    """Return the current Flask request, allowing monkeypatching via web_server.request."""
    import src.webui.web_server as _ws
    return _ws.request

from src.webui.web_server_utils.apriltag_map_sanitizer import sanitize_apriltag_map_file
from src.webui.web_server_utils.constants import (
    APRILTAG_MAP_EXTENSIONS,
    ASSET_ROTATION_OFFSET_KEY,
    ASSET_SCALE_KEY,
    DEFAULT_ASSET_ROTATION_OFFSET,
    DEFAULT_ASSET_SCALE,
    FIELD_APRILTAG_MAP_DIR_NAME,
    FIELD_ASSET_DIR_NAME,
    FIELD_FILE_DIR_NAME,
    MODEL_ASSET_EXTENSION,
    MODEL_ASSET_METADATA_SUFFIX,
    ROBOT_ASSET_DIR_NAME,
    WEBUI_DIR,
)


class AssetManagerMixin:
    def serve_webui_asset(self, filename: str):
        """Serve WebUI assets, preferring cached Draco-compressed GLB files."""
        cached_or_source_path = self.draco_asset_cache.resolve_asset(filename)
        if cached_or_source_path is not None:
            return send_from_directory(
                str(cached_or_source_path.parent), cached_or_source_path.name
            )
        return send_from_directory(os.path.join(WEBUI_DIR, "assets"), filename)

    def serve_robot_file(self, filename: str):
        """Serve robot GLB files, preferring cached Draco-compressed copies."""
        cached_or_source_path = self.draco_asset_cache.resolve_asset(
            Path("robots") / filename
        )
        if cached_or_source_path is not None:
            return send_from_directory(
                str(cached_or_source_path.parent), cached_or_source_path.name
            )
        return send_from_directory(
            os.path.join(WEBUI_DIR, "assets", "robots"), filename
        )

    def _assets_dir(self) -> Path:
        """Return the WebUI source assets directory."""
        import src.webui.web_server as _ws
        return Path(_ws.current_path) / "assets"

    def _robot_assets_dir(self) -> Path:
        """Return the robot model asset directory, creating it if needed."""
        robot_dir = self._assets_dir() / ROBOT_ASSET_DIR_NAME
        robot_dir.mkdir(parents=True, exist_ok=True)
        return robot_dir

    def _field_file_assets_dir(self, year: str) -> Path:
        """Return the field model asset directory for one game year."""
        field_dir = (
            self._assets_dir() / FIELD_ASSET_DIR_NAME / year / FIELD_FILE_DIR_NAME
        )
        field_dir.mkdir(parents=True, exist_ok=True)
        return field_dir

    def _field_apriltag_map_assets_dir(self, year: str) -> Path:
        """Return the AprilTag map asset directory for one game year."""
        map_dir = (
            self._assets_dir()
            / FIELD_ASSET_DIR_NAME
            / year
            / FIELD_APRILTAG_MAP_DIR_NAME
        )
        map_dir.mkdir(parents=True, exist_ok=True)
        return map_dir

    def _sanitize_asset_filename(self, raw_filename: str) -> str:
        """Validate and sanitize an uploaded GLB asset filename."""
        if "/" in raw_filename or "\\" in raw_filename:
            raise ValueError("Path separators are not allowed in asset filenames")

        filename = secure_filename(raw_filename)
        if not filename:
            raise ValueError("Filename cannot be empty")

        if filename.startswith("_"):
            raise ValueError("Asset filenames cannot start with an underscore")

        if Path(filename).suffix.lower() != MODEL_ASSET_EXTENSION:
            raise ValueError(f"Only {MODEL_ASSET_EXTENSION} files are supported")

        return filename

    def _sanitize_apriltag_map_filename(self, raw_filename: str) -> str:
        """Validate and sanitize an uploaded AprilTag map filename."""
        if "/" in raw_filename or "\\" in raw_filename:
            raise ValueError("Path separators are not allowed in fmap filenames")

        filename = secure_filename(raw_filename)
        if not filename:
            raise ValueError("AprilTag map filename cannot be empty")

        if Path(filename).suffix.lower() not in APRILTAG_MAP_EXTENSIONS:
            allowed_extensions = ", ".join(sorted(APRILTAG_MAP_EXTENSIONS))
            raise ValueError(
                f"Only {allowed_extensions} AprilTag map files are supported"
            )

        return filename

    def _sanitize_field_year(self, raw_year: str) -> str:
        """Validate and sanitize a field asset year/directory name."""
        if "/" in raw_year or "\\" in raw_year:
            raise ValueError("Path separators are not allowed in field years")

        year = secure_filename(raw_year.strip())
        if not year:
            raise ValueError("Field year is required")

        return year

    def _asset_file_details(self, file_path: Path) -> dict[str, Any]:
        """Return the file metadata shape used by asset-manager responses."""
        file_stat = file_path.stat()
        return {
            "filename": file_path.name,
            "size": file_stat.st_size,
            "modified": file_stat.st_mtime,
        }

    def _asset_metadata_path(self, file_path: Path) -> Path:
        """Return the sidecar metadata path for a model asset."""
        return file_path.with_name(f"{file_path.name}{MODEL_ASSET_METADATA_SUFFIX}")

    def _read_asset_metadata(self, file_path: Path) -> dict[str, Any]:
        """Read optional model metadata from the asset sidecar file."""
        metadata_path = self._asset_metadata_path(file_path)
        if not metadata_path.is_file():
            return {}

        try:
            with metadata_path.open("r", encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)
        except (OSError, json.JSONDecodeError) as exc:
            self.log(f"Warning: Could not read asset metadata {metadata_path}: {exc}")
            return {}

        return metadata if isinstance(metadata, dict) else {}

    def _write_asset_metadata(
        self, file_path: Path, metadata: dict[str, Any]
    ) -> None:
        """Write model metadata beside the GLB asset."""
        metadata_path = self._asset_metadata_path(file_path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, indent=2, sort_keys=True)
            metadata_file.write("\n")

    def _delete_asset_metadata(self, file_path: Path) -> None:
        """Delete optional sidecar metadata for a removed model asset."""
        metadata_path = self._asset_metadata_path(file_path)
        if metadata_path.is_file():
            metadata_path.unlink()

    def _asset_scale(self, file_path: Path) -> float:
        """Return the saved positive scale factor for a model asset."""
        raw_scale = self._read_asset_metadata(file_path).get(ASSET_SCALE_KEY)
        try:
            scale = float(raw_scale)
        except (TypeError, ValueError):
            return DEFAULT_ASSET_SCALE

        if not np.isfinite(scale) or scale <= 0:
            return DEFAULT_ASSET_SCALE

        return scale

    def _asset_rotation_offset(self, file_path: Path) -> dict[str, float]:
        """Return saved model rotation offsets in degrees."""
        raw_rotation = self._read_asset_metadata(file_path).get(
            ASSET_ROTATION_OFFSET_KEY, {}
        )
        if not isinstance(raw_rotation, dict):
            raw_rotation = {}

        rotation = DEFAULT_ASSET_ROTATION_OFFSET.copy()
        for axis in rotation:
            try:
                value = float(raw_rotation.get(axis, rotation[axis]))
            except (TypeError, ValueError):
                value = rotation[axis]
            rotation[axis] = value if np.isfinite(value) else rotation[axis]
        return rotation

    def _save_asset_settings(
        self, file_path: Path, scale: float, rotation_offset: dict[str, float] | None = None
    ) -> None:
        """Persist viewer settings for a model asset."""
        metadata = self._read_asset_metadata(file_path)
        metadata[ASSET_SCALE_KEY] = scale
        if rotation_offset is not None:
            metadata[ASSET_ROTATION_OFFSET_KEY] = rotation_offset
        self._write_asset_metadata(file_path, metadata)

    def _asset_settings_from_request(self) -> tuple[float, dict[str, float]]:
        """Parse and validate model viewer settings from JSON or form data."""
        req = _request()
        payload: dict[str, Any] = {}
        get_json = getattr(req, "get_json", None)
        if callable(get_json):
            json_payload = get_json(silent=True)
            if isinstance(json_payload, dict):
                payload = json_payload

        raw_scale = payload.get(ASSET_SCALE_KEY, req.form.get(ASSET_SCALE_KEY))
        try:
            scale = float(raw_scale)
        except (TypeError, ValueError):
            raise ValueError("Scale must be a number") from None

        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Scale must be a positive number")

        raw_rotation = payload.get(ASSET_ROTATION_OFFSET_KEY, {})
        if not isinstance(raw_rotation, dict):
            raw_rotation = {}
        rotation = DEFAULT_ASSET_ROTATION_OFFSET.copy()
        for axis in rotation:
            try:
                value = float(raw_rotation.get(axis, req.form.get(f"rotation_{axis}", 0)))
            except (TypeError, ValueError):
                raise ValueError(f"Rotation {axis.upper()} must be a number") from None
            if not np.isfinite(value):
                raise ValueError(f"Rotation {axis.upper()} must be a finite number")
            rotation[axis] = value

        return scale, rotation

    def _robot_file_detail(self, file_path: Path) -> dict[str, Any]:
        """Return robot GLB metadata including saved scale."""
        detail = self._asset_file_details(file_path)
        detail[ASSET_SCALE_KEY] = self._asset_scale(file_path)
        detail[ASSET_ROTATION_OFFSET_KEY] = self._asset_rotation_offset(file_path)
        return detail

    def _webui_asset_url(self, relative_path: Path) -> str:
        """Return the frontend URL for an asset path relative to assets/."""
        return f"/assets/{relative_path.as_posix()}"

    def _field_game_piece_details(self, year: str) -> list[dict[str, Any]]:
        """Return game-piece GLB asset details for a field year."""
        game_pieces_dir = (
            self._assets_dir() / FIELD_ASSET_DIR_NAME / year / "game_pieces"
        )
        if not game_pieces_dir.is_dir():
            return []

        game_pieces = []
        for file_path in sorted(game_pieces_dir.iterdir(), key=lambda p: p.name):
            if (
                not file_path.is_file()
                or file_path.suffix.lower() != MODEL_ASSET_EXTENSION
                or file_path.name.startswith("_")
            ):
                continue

            relative_path = (
                Path(FIELD_ASSET_DIR_NAME) / year / "game_pieces" / file_path.name
            )
            detail = self._asset_file_details(file_path)
            detail["asset_path"] = relative_path.as_posix()
            detail["url"] = self._webui_asset_url(relative_path)
            game_pieces.append(detail)

        return game_pieces

    def _field_apriltag_map_detail(
        self, field_filename: str, year: str
    ) -> dict[str, Any] | None:
        """Return the AprilTag map associated with a field file, if present."""
        map_dir = (
            self._assets_dir()
            / FIELD_ASSET_DIR_NAME
            / year
            / FIELD_APRILTAG_MAP_DIR_NAME
        )
        if not map_dir.is_dir():
            return None

        field_stem = Path(field_filename).stem
        for extension in sorted(APRILTAG_MAP_EXTENSIONS):
            map_path = map_dir / f"{field_stem}{extension}"
            if not map_path.is_file():
                continue

            relative_path = (
                Path(FIELD_ASSET_DIR_NAME)
                / year
                / FIELD_APRILTAG_MAP_DIR_NAME
                / map_path.name
            )
            detail = self._asset_file_details(map_path)
            detail["asset_path"] = relative_path.as_posix()
            detail["url"] = self._webui_asset_url(relative_path)
            return detail

        return None

    def _field_file_detail(self, file_path: Path, year: str) -> dict[str, Any]:
        """Return field GLB metadata with exact asset URLs."""
        relative_path = (
            Path(FIELD_ASSET_DIR_NAME) / year / FIELD_FILE_DIR_NAME / file_path.name
        )
        game_pieces = self._field_game_piece_details(year)
        detail = self._asset_file_details(file_path)
        detail["year"] = year
        detail["path"] = f"{year}/{FIELD_FILE_DIR_NAME}/{file_path.name}"
        detail["asset_path"] = relative_path.as_posix()
        detail["url"] = self._webui_asset_url(relative_path)
        detail[ASSET_SCALE_KEY] = self._asset_scale(file_path)
        detail[ASSET_ROTATION_OFFSET_KEY] = self._asset_rotation_offset(file_path)
        detail["game_pieces"] = game_pieces
        detail["game_piece_urls"] = [game_piece["url"] for game_piece in game_pieces]
        apriltag_map = self._field_apriltag_map_detail(file_path.name, year)
        detail["apriltag_map"] = apriltag_map
        detail["apriltag_map_url"] = apriltag_map["url"] if apriltag_map else None
        return detail

    def _save_field_apriltag_map(
        self, year: str, field_filename: str, upload: Any
    ) -> None:
        """Save an uploaded AprilTag map with the same stem as its field file."""
        if upload is None or not upload.filename:
            return

        map_filename = self._sanitize_apriltag_map_filename(upload.filename)
        map_extension = Path(map_filename).suffix.lower()
        map_path = (
            self._field_apriltag_map_assets_dir(year)
            / f"{Path(field_filename).stem}{map_extension}"
        )
        upload.save(str(map_path))
        fixes = sanitize_apriltag_map_file(map_path)
        if fixes:
            self.log(
                f"Auto-fixed {fixes} invalid AprilTag map transform values in "
                f"{map_filename}"
            )

    def _overwrite_requested(self) -> bool:
        """Return whether a multipart upload explicitly requested overwrite."""
        return str(_request().form.get("overwrite", "")).lower() == "true"

    def _prepare_draco_asset(self, relative_path: Path) -> None:
        """Prime the Draco cache for an asset after upload."""
        try:
            self.draco_asset_cache.resolve_asset(relative_path)
        except Exception as exc:
            self.log(
                f"Warning: Could not prepare Draco cache for {relative_path}: {exc}"
            )

    def get_available_robots(self) -> dict:
        """
        Get a dict of available robots.

        Returns:
            dict:
                robots: list of dicts with the name and path of the robot file.
                    name: the name of the robot file.
                    path: the path of the robot file.
        """
        payload, _status = self.get_robot_files()
        return payload

    def get_robot_files(self) -> tuple[dict, int]:
        """
        Get available robot GLB files and metadata.

        Returns:
            Tuple of response dict and status code.
        """
        try:
            robot_dir = self._robot_assets_dir()
            files = [
                self._robot_file_detail(file_path)
                for file_path in robot_dir.iterdir()
                if (
                    file_path.is_file()
                    and file_path.suffix.lower() == MODEL_ASSET_EXTENSION
                    and not file_path.name.startswith("_")
                )
            ]
            files.sort(key=lambda item: item["filename"].lower())
            return {
                "robots": [file_info["filename"] for file_info in files],
                "file_details": files,
            }, 200
        except Exception as e:
            self.log(f"Error getting robot files: {e}")
            return {"error": str(e)}, 500

    def upload_robot_file(self) -> tuple[dict, int]:
        """
        Upload or replace a robot GLB file.

        Returns:
            Tuple of response dict and status code.
        """
        try:
            req = _request()
            if "file" not in req.files:
                return {"error": "No file provided"}, 400

            file = req.files["file"]
            filename = self._sanitize_asset_filename(file.filename or "")
            destination = self._robot_assets_dir() / filename

            if destination.exists() and not self._overwrite_requested():
                return {
                    "error": f"{filename} already exists",
                    "filename": filename,
                    "requires_overwrite": True,
                }, 409

            file.save(str(destination))
            self._prepare_draco_asset(Path(ROBOT_ASSET_DIR_NAME) / filename)
            self.log(f"Uploaded robot file {filename}")

            return {
                "success": True,
                "file": self._robot_file_detail(destination),
            }, 200
        except ValueError as e:
            return {"error": str(e)}, 400
        except Exception as e:
            self.log(f"Error uploading robot file: {e}")
            return {"error": str(e)}, 500

    def save_robot_file_scale(self, filename: str) -> tuple[dict, int]:
        """
        Save the 3D viewer scale factor for a robot GLB file.

        Args:
            filename: Robot model filename.

        Returns:
            Tuple of response dict and status code.
        """
        try:
            safe_filename = self._sanitize_asset_filename(filename)
            file_path = self._robot_assets_dir() / safe_filename

            if not file_path.is_file():
                return {"error": "File not found"}, 404

            scale, rotation_offset = self._asset_settings_from_request()
            self._save_asset_settings(file_path, scale, rotation_offset)
            self.log(f"Saved robot file scale {scale} for {safe_filename}")

            return {
                "success": True,
                "file": self._robot_file_detail(file_path),
            }, 200
        except ValueError as e:
            return {"error": str(e)}, 400
        except Exception as e:
            self.log(f"Error saving robot file scale: {e}")
            return {"error": str(e)}, 500

    def delete_robot_file(self, filename: str) -> tuple[dict, int]:
        """
        Delete a robot GLB file.

        Args:
            filename: Robot model filename.

        Returns:
            Tuple of response dict and status code.
        """
        try:
            safe_filename = self._sanitize_asset_filename(filename)
            file_path = self._robot_assets_dir() / safe_filename

            if not file_path.exists():
                return {"error": "File not found"}, 404

            if not file_path.is_file():
                return {"error": "Path is not a file"}, 400

            file_path.unlink()
            self._delete_asset_metadata(file_path)
            self.log(f"Deleted robot file {safe_filename}")

            return {"success": True}, 200
        except ValueError as e:
            return {"error": str(e)}, 400
        except Exception as e:
            self.log(f"Error deleting robot file: {e}")
            return {"error": str(e)}, 500

    def get_field_files(self) -> tuple[dict, int]:
        """
        Get available field GLB files grouped by year.

        Returns:
            Tuple of response dict and status code.
        """
        try:
            fields_root = self._assets_dir() / FIELD_ASSET_DIR_NAME
            fields_by_year: dict[str, list[str]] = {}
            file_details: list[dict[str, Any]] = []

            if fields_root.is_dir():
                for year_dir in sorted(fields_root.iterdir(), key=lambda p: p.name):
                    if not year_dir.is_dir() or year_dir.name.startswith("_"):
                        continue

                    field_dir = year_dir / FIELD_FILE_DIR_NAME
                    if not field_dir.is_dir():
                        continue

                    year_files = []
                    for file_path in sorted(field_dir.iterdir(), key=lambda p: p.name):
                        if (
                            not file_path.is_file()
                            or file_path.suffix.lower() != MODEL_ASSET_EXTENSION
                            or file_path.name.startswith("_")
                        ):
                            continue

                        year_files.append(file_path.name)
                        file_details.append(
                            self._field_file_detail(file_path, year_dir.name)
                        )

                    if year_files:
                        fields_by_year[year_dir.name] = year_files

            return {
                "fields": fields_by_year,
                "file_details": file_details,
            }, 200
        except Exception as e:
            self.log(f"Error getting field files: {e}")
            return {"error": str(e)}, 500

    def upload_field_file(self) -> tuple[dict, int]:
        """
        Upload or replace a field GLB file for one game year.

        Returns:
            Tuple of response dict and status code.
        """
        try:
            req = _request()
            if "file" not in req.files:
                return {"error": "No file provided"}, 400

            file = req.files["file"]
            year = self._sanitize_field_year(str(req.form.get("year", "")))
            filename = self._sanitize_asset_filename(file.filename or "")
            apriltag_map_upload = req.files.get("apriltag_map")
            destination = self._field_file_assets_dir(year) / filename

            if destination.exists() and not self._overwrite_requested():
                return {
                    "error": f"{filename} already exists for {year}",
                    "filename": filename,
                    "year": year,
                    "requires_overwrite": True,
                }, 409

            file.save(str(destination))
            self._save_field_apriltag_map(year, filename, apriltag_map_upload)
            self._prepare_draco_asset(
                Path(FIELD_ASSET_DIR_NAME) / year / FIELD_FILE_DIR_NAME / filename
            )
            self.log(f"Uploaded field file {filename} for {year}")

            return {
                "success": True,
                "file": self._field_file_detail(destination, year),
            }, 200
        except ValueError as e:
            return {"error": str(e)}, 400
        except Exception as e:
            self.log(f"Error uploading field file: {e}")
            return {"error": str(e)}, 500

    def save_field_file_scale(self, year: str, filename: str) -> tuple[dict, int]:
        """
        Save the 3D viewer scale factor for a field GLB file.

        Args:
            year: Field game year/directory name.
            filename: Field model filename.

        Returns:
            Tuple of response dict and status code.
        """
        try:
            safe_year = self._sanitize_field_year(year)
            safe_filename = self._sanitize_asset_filename(filename)
            file_path = self._field_file_assets_dir(safe_year) / safe_filename

            if not file_path.is_file():
                return {"error": "File not found"}, 404

            scale, rotation_offset = self._asset_settings_from_request()
            self._save_asset_settings(file_path, scale, rotation_offset)
            self.log(
                f"Saved field file scale {scale} for {safe_year}/{safe_filename}"
            )

            return {
                "success": True,
                "file": self._field_file_detail(file_path, safe_year),
            }, 200
        except ValueError as e:
            return {"error": str(e)}, 400
        except Exception as e:
            self.log(f"Error saving field file scale: {e}")
            return {"error": str(e)}, 500

    def delete_field_file(self, year: str, filename: str) -> tuple[dict, int]:
        """
        Delete a field GLB file.

        Args:
            year: Field game year/directory name.
            filename: Field model filename.

        Returns:
            Tuple of response dict and status code.
        """
        try:
            safe_year = self._sanitize_field_year(year)
            safe_filename = self._sanitize_asset_filename(filename)
            file_path = self._field_file_assets_dir(safe_year) / safe_filename

            if not file_path.exists():
                return {"error": "File not found"}, 404

            if not file_path.is_file():
                return {"error": "Path is not a file"}, 400

            file_path.unlink()
            self._delete_asset_metadata(file_path)
            for extension in APRILTAG_MAP_EXTENSIONS:
                map_path = (
                    self._field_apriltag_map_assets_dir(safe_year)
                    / f"{Path(safe_filename).stem}{extension}"
                )
                if map_path.is_file():
                    map_path.unlink()
            self.log(f"Deleted field file {safe_filename} for {safe_year}")

            return {"success": True}, 200
        except ValueError as e:
            return {"error": str(e)}, 400
        except Exception as e:
            self.log(f"Error deleting field file: {e}")
            return {"error": str(e)}, 500
