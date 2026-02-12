"""Camera configuration management for intrinsics and extrinsics."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class CameraExtrinsics:
    """Camera extrinsic parameters.

    Attributes:
        horizontal_fov: Horizontal field of view in degrees.
        vertical_fov: Vertical field of view in degrees.
        pitch: Camera pitch angle in degrees.
        yaw: Camera yaw angle in degrees.
        roll: Camera roll angle in degrees.
        x_offset: X position offset in meters.
        y_offset: Y position offset in meters.
        z_offset: Z position offset in meters.
    """

    horizontal_fov: float = 0.0
    vertical_fov: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    x_offset: float = 0.0
    y_offset: float = 0.0
    z_offset: float = 0.0

    def to_dict(self) -> dict:
        """Convert extrinsics to dictionary format.

        Returns:
            dict: Dictionary representation of extrinsics.
        """
        return {
            "horizontal_fov": self.horizontal_fov,
            "vertical_fov": self.vertical_fov,
            "pitch": self.pitch,
            "yaw": self.yaw,
            "roll": self.roll,
            "x_offset": self.x_offset,
            "y_offset": self.y_offset,
            "z_offset": self.z_offset,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CameraExtrinsics:
        """Create extrinsics from dictionary.

        Args:
            data: Dictionary containing extrinsic parameters.

        Returns:
            CameraExtrinsics: New extrinsics instance.
        """
        return cls(
            horizontal_fov=data.get("horizontal_fov", 0.0),
            vertical_fov=data.get("vertical_fov", 0.0),
            pitch=data.get("pitch", 0.0),
            yaw=data.get("yaw", 0.0),
            roll=data.get("roll", 0.0),
            x_offset=data.get("x_offset", 0.0),
            y_offset=data.get("y_offset", 0.0),
            z_offset=data.get("z_offset", 0.0),
        )


class CameraConfig:
    """Manages configuration for a single camera.

    Stores intrinsics file path and extrinsic parameters with
    persistence to JSON files following the camera_calibrations pattern.
    """

    def __init__(
        self,
        camera_id: str,
        base_path: Optional[str] = None,
    ) -> None:
        """Initialize camera configuration.

        Args:
            camera_id: Unique identifier for this camera.
            base_path: Base directory for camera calibrations. Defaults to
                src/utils/camera_utils/camera_calibrations/.
        """
        self._camera_id: str = camera_id
        self._extrinsics: CameraExtrinsics = CameraExtrinsics()

        if base_path is None:
            base_path = os.path.join(
                os.path.dirname(__file__), "camera_calibrations", camera_id
            )
        self._base_path: str = base_path
        self._intrinsics_file: str = os.path.join(base_path, "intrinsics.json")
        self._intrinsics_path: Optional[str] = self._intrinsics_file
        self._extrinsics_file: str = os.path.join(base_path, "extrinsics.json")

    @property
    def camera_id(self) -> str:
        """Get the camera identifier.

        Returns:
            str: Camera ID.
        """
        return self._camera_id

    @property
    def intrinsics_path(self) -> Optional[str]:
        """Get the intrinsics file path.

        Returns:
            Optional[str]: Path to intrinsics JSON file, or None if not set.
        """
        return self._intrinsics_path

    @intrinsics_path.setter
    def intrinsics_path(self, path: Optional[str]) -> None:
        """Set the intrinsics file path.

        Args:
            path: Path to intrinsics JSON file, or None to clear.
        """
        self._intrinsics_path = path

    @property
    def extrinsics(self) -> CameraExtrinsics:
        """Get the extrinsics object.

        Returns:
            CameraExtrinsics: Current extrinsic parameters.
        """
        return self._extrinsics

    def _set_and_sync_extrinsics_field(self, field_name: str, value: float) -> None:
        """Update one extrinsics field and keep disk/runtime state synchronized.

        Args:
            field_name: Name of the ``CameraExtrinsics`` field to update.
            value: New numeric value for the field.
        """
        setattr(self._extrinsics, field_name, value)
        self.save_and_reload_extrinsics()

    def get_horizontal_fov(self) -> float:
        """Get horizontal field of view.

        Returns:
            float: Horizontal FOV in degrees.
        """
        return self._extrinsics.horizontal_fov

    def get_vertical_fov(self) -> float:
        """Get vertical field of view.

        Returns:
            float: Vertical FOV in degrees.
        """
        return self._extrinsics.vertical_fov

    def get_pitch(self) -> float:
        """Get camera pitch angle.

        Returns:
            float: Pitch in degrees.
        """
        return self._extrinsics.pitch

    def get_yaw(self) -> float:
        """Get camera yaw angle.

        Returns:
            float: Yaw in degrees.
        """
        return self._extrinsics.yaw

    def get_roll(self) -> float:
        """Get camera roll angle.

        Returns:
            float: Roll in degrees.
        """
        return self._extrinsics.roll

    def get_x_offset(self) -> float:
        """Get X position offset.

        Returns:
            float: X offset in meters.
        """
        return self._extrinsics.x_offset

    def get_y_offset(self) -> float:
        """Get Y position offset.

        Returns:
            float: Y offset in meters.
        """
        return self._extrinsics.y_offset

    def get_z_offset(self) -> float:
        """Get Z position offset.

        Returns:
            float: Z offset in meters.
        """
        return self._extrinsics.z_offset

    def set_horizontal_fov(self, value: float) -> None:
        """Set horizontal field of view.

        Args:
            value: Horizontal FOV in degrees.
        """
        self._set_and_sync_extrinsics_field("horizontal_fov", value)

    def set_vertical_fov(self, value: float) -> None:
        """Set vertical field of view.

        Args:
            value: Vertical FOV in degrees.
        """
        self._set_and_sync_extrinsics_field("vertical_fov", value)

    def set_pitch(self, value: float) -> None:
        """Set camera pitch angle.

        Args:
            value: Pitch in degrees.
        """
        self._set_and_sync_extrinsics_field("pitch", value)

    def set_yaw(self, value: float) -> None:
        """Set camera yaw angle.

        Args:
            value: Yaw in degrees.
        """
        self._set_and_sync_extrinsics_field("yaw", value)

    def set_roll(self, value: float) -> None:
        """Set camera roll angle.

        Args:
            value: Roll in degrees.
        """
        self._set_and_sync_extrinsics_field("roll", value)

    def set_x_offset(self, value: float) -> None:
        """Set X position offset.

        Args:
            value: X offset in meters.
        """
        self._set_and_sync_extrinsics_field("x_offset", value)

    def set_y_offset(self, value: float) -> None:
        """Set Y position offset.

        Args:
            value: Y offset in meters.
        """
        self._set_and_sync_extrinsics_field("y_offset", value)

    def set_z_offset(self, value: float) -> None:
        """Set Z position offset.

        Args:
            value: Z offset in meters.
        """
        self._set_and_sync_extrinsics_field("z_offset", value)

    def set_extrinsics_from_json(self, json_data: dict | str) -> None:
        """Set extrinsics from JSON data.

        Args:
            json_data: Dictionary or JSON string containing extrinsic parameters.
                Expected keys: horizontal_fov, vertical_fov, pitch, yaw, roll,
                x_offset, y_offset, z_offset.

        Raises:
            ValueError: If json_data is a string and cannot be parsed.
        """
        if isinstance(json_data, str):
            try:
                json_data = json.loads(json_data)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON string: {e}") from e

        if not isinstance(json_data, dict):
            raise ValueError(
                "Extrinsics JSON payload must decode to a dictionary object."
            )

        self._extrinsics = CameraExtrinsics.from_dict(json_data)
        self.save_and_reload_extrinsics()

    def update_extrinsics_live(self, json_data: dict | str) -> None:
        """Apply, persist, and refresh extrinsics for live config updates.

        This method updates the in-memory values from provided JSON data,
        saves the result to disk, and then reloads from disk so local state
        always mirrors persisted state.

        Args:
            json_data: Dictionary or JSON string containing extrinsic
                parameters.
        """
        self.set_extrinsics_from_json(json_data)

    def save_and_reload_extrinsics(self) -> None:
        """Persist extrinsics to disk and refresh local values from file."""
        self.save_extrinsics()
        self.load_extrinsics()

    def save_extrinsics(self) -> None:
        """Save extrinsics to JSON file.

        Creates the directory if it doesn't exist.
        """
        os.makedirs(self._base_path, exist_ok=True)
        with open(self._extrinsics_file, "w") as f:
            json.dump(self._extrinsics.to_dict(), f, indent=4)

    def load_extrinsics(self) -> None:
        """Load extrinsics from JSON file.

        Raises:
            FileNotFoundError: If extrinsics file doesn't exist.
        """
        if not os.path.exists(self._extrinsics_file):
            raise FileNotFoundError(
                f"Extrinsics file not found: {self._extrinsics_file}"
            )
        with open(self._extrinsics_file, "r") as f:
            data = json.load(f)
        self._extrinsics = CameraExtrinsics.from_dict(data)


class CameraConfigRegistry:
    """Registry for managing multiple camera configurations."""

    def __init__(self, base_path: Optional[str] = None) -> None:
        """Initialize the camera config registry.

        Args:
            base_path: Base directory for camera calibrations. Defaults to
                src/utils/camera_utils/camera_calibrations/.
        """
        self._base_path: Optional[str] = base_path
        self._configs: dict[str, CameraConfig] = {}

    def get_config(self, camera_id: str) -> CameraConfig:
        """Get or create a camera configuration.

        Args:
            camera_id: Unique identifier for the camera.

        Returns:
            CameraConfig: The camera configuration instance.
        """
        if camera_id not in self._configs:
            camera_path = (
                os.path.join(self._base_path, camera_id)
                if self._base_path is not None
                else None
            )
            self._configs[camera_id] = CameraConfig(camera_id, camera_path)
        return self._configs[camera_id]

    def has_config(self, camera_id: str) -> bool:
        """Check if a camera configuration exists.

        Args:
            camera_id: Unique identifier for the camera.

        Returns:
            bool: True if config exists, False otherwise.
        """
        return camera_id in self._configs

    def remove_config(self, camera_id: str) -> bool:
        """Remove a camera configuration from the registry.

        Args:
            camera_id: Unique identifier for the camera.

        Returns:
            bool: True if removed, False if not found.
        """
        if camera_id in self._configs:
            del self._configs[camera_id]
            return True
        return False

    def get_all_camera_ids(self) -> list[str]:
        """Get all registered camera IDs.

        Returns:
            list[str]: List of camera identifiers.
        """
        return list(self._configs.keys())

    def get_all_configs(self) -> dict[str, CameraConfig]:
        """Get all camera configurations.

        Returns:
            dict[str, CameraConfig]: Dictionary mapping camera IDs to configs.
        """
        return self._configs.copy()

    def load_all_from_directory(self) -> int:
        """Load all camera configs from the base directory.

        Scans the base path for subdirectories containing extrinsics.json
        and loads them into the registry.

        Returns:
            int: Number of camera configs loaded.
        """
        if self._base_path is None:
            base = os.path.join(os.path.dirname(__file__), "camera_calibrations")
        else:
            base = self._base_path

        if not os.path.exists(base):
            return 0

        loaded_count = 0
        for camera_id in os.listdir(base):
            camera_dir = os.path.join(base, camera_id)
            extrinsics_file = os.path.join(camera_dir, "extrinsics.json")
            if os.path.isdir(camera_dir) and os.path.exists(extrinsics_file):
                config = CameraConfig(camera_id, camera_dir)
                try:
                    config.load_extrinsics()
                    self._configs[camera_id] = config
                    loaded_count += 1
                except (json.JSONDecodeError, KeyError):
                    continue

        return loaded_count

    def save_all(self) -> None:
        """Save all registered camera extrinsics to files."""
        for config in self._configs.values():
            config.save_extrinsics()
