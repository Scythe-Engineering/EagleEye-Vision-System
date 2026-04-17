from __future__ import annotations

import os
from typing import Optional

from flask import request

from src.webui.web_server_utils.constants import SRC_DIR


class CameraConfigMixin:
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

    def _resolve_camera_config(self, camera_bus_id: str):
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
            SRC_DIR,
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
