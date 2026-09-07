"""Regression tests for UI-only camera placement names."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest

import src.utils.camera_utils.camera_config_manager as camera_config_manager
import src.webui.web_server_utils.camera_config_mixin as camera_config_module
import src.webui.web_server_utils.camera_stream_mixin as camera_stream_module
from src.utils.camera_utils.camera_config_manager import CameraConfigRegistry
from src.webui.web_server_utils.camera_config_mixin import CameraConfigMixin
from src.webui.web_server_utils.camera_stream_mixin import CameraStreamMixin
from src.webui.web_server_utils.first_boot_mixin import FirstBootMixin


class _Request:
    """Minimal request object for direct mixin endpoint tests."""

    def __init__(self, payload: Any) -> None:
        """Store the JSON payload returned by ``get_json``.

        Args:
            payload: Request JSON payload.
        """
        self.payload = payload

    def get_json(self, silent: bool = False) -> Any:
        """Return the configured JSON request payload."""
        return self.payload


class _CameraConfigHarness(CameraConfigMixin):
    """Minimal host for camera-name endpoint tests."""

    def __init__(self, base_path: Path) -> None:
        """Create one active camera and a temporary config registry.

        Args:
            base_path: Temporary calibration directory.
        """
        self.available_cameras = {
            "USB Camera": {
                "name": "USB_Camera",
                "id": 7,
                "bus_id": "1-2",
            }
        }
        self.camera_config_registry = CameraConfigRegistry(str(base_path))
        self.log = lambda _message: None


class _FirstBootHarness(FirstBootMixin):
    """Minimal first-boot camera-list harness."""

    def __init__(self, registry: CameraConfigRegistry) -> None:
        self.available_cameras = {
            "USB Camera": {"name": "USB_Camera", "bus_id": "1-2"}
        }
        self.camera_config_registry = registry
        self.frame_list_structure_lock = threading.Lock()


def test_display_name_persists_by_bus_id_without_changing_camera_id(
    tmp_path: Path,
) -> None:
    """Persist placement metadata independently from runtime camera identity."""
    registry = CameraConfigRegistry(str(tmp_path))
    config = registry.get_config("1-2")

    config.set_display_name("Front bumper")

    assert config.camera_id == "1-2"
    assert config.display_name == "Front bumper"
    assert json.loads((tmp_path / "1-2" / "metadata.json").read_text()) == {
        "display_name": "Front bumper"
    }
    assert CameraConfigRegistry(str(tmp_path)).get_config("1-2").display_name == (
        "Front bumper"
    )


def test_display_name_endpoint_validates_and_exposes_ui_name(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Reject invalid names and leave feed and bus identities untouched."""
    harness = _CameraConfigHarness(tmp_path)
    monkeypatch.setattr(
        camera_config_module, "request", _Request({"display_name": "Rear shelf"})
    )

    payload, status = harness.save_camera_display_name("1-2")

    assert status == 200
    assert payload == {
        "success": True,
        "camera_bus_id": "1-2",
        "display_name": "Rear shelf",
    }
    cameras, cameras_status = harness.get_camera_config_cameras()
    assert cameras_status == 200
    assert cameras == {
        "cameras": [
            {
                "name": "USB Camera",
                "display_name": "Rear shelf",
                "bus_id": "1-2",
                "stream_name": "USB_Camera",
            }
        ]
    }

    monkeypatch.setattr(camera_config_module, "request", _Request({"display_name": " "}))
    error, error_status = harness.save_camera_display_name("1-2")

    assert error_status == 400
    assert error == {"error": "Camera name cannot be empty"}
    assert harness.camera_config_registry.get_config("1-2").camera_id == "1-2"


def test_display_name_endpoint_rejects_unknown_bus_id(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Reject unknown IDs before creating a camera configuration directory."""
    harness = _CameraConfigHarness(tmp_path)
    monkeypatch.setattr(
        camera_config_module, "request", _Request({"display_name": "Unsafe"})
    )

    payload, status = harness.save_camera_display_name("..")

    assert status == 404
    assert payload == {"error": "Unknown camera bus ID"}
    assert not (tmp_path / "metadata.json").exists()


@pytest.mark.parametrize("name", [None, 7, "", " ", "x" * 81, "Front\nbumper"])
def test_invalid_display_name_does_not_write(tmp_path: Path, name: Any) -> None:
    config = CameraConfigRegistry(str(tmp_path)).get_config("1-2")
    with pytest.raises(ValueError):
        config.set_display_name(name)
    assert config.display_name is None
    assert not (tmp_path / "1-2" / "metadata.json").exists()


def test_failed_display_name_write_preserves_previous_name(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Keep in-memory and persisted metadata synchronized after write failure."""
    config = CameraConfigRegistry(str(tmp_path)).get_config("1-2")
    config.set_display_name("Original")
    monkeypatch.setattr(
        camera_config_manager.os,
        "replace",
        lambda _source, _destination: (_ for _ in ()).throw(OSError("disk full")),
    )

    try:
        config.set_display_name("Replacement")
    except OSError:
        pass
    else:
        raise AssertionError("Expected metadata replacement to fail")

    assert config.display_name == "Original"
    assert json.loads((tmp_path / "1-2" / "metadata.json").read_text()) == {
        "display_name": "Original"
    }
    assert not list((tmp_path / "1-2").glob(".metadata.*.tmp"))


def test_first_boot_loads_saved_name_after_registry_restart(tmp_path: Path) -> None:
    """Expose saved metadata after restart without requiring extrinsics files."""
    registry = CameraConfigRegistry(str(tmp_path))
    registry.get_config("1-2").set_display_name("Front bumper")

    restarted_registry = CameraConfigRegistry(str(tmp_path))
    records = _FirstBootHarness(restarted_registry)._first_boot_camera_records()

    assert records[0]["display_name"] == "Front bumper"
    assert not (tmp_path / "1-2" / "extrinsics.json").exists()


def test_camera_snapshot_is_finite_and_keeps_stream_identity(monkeypatch: Any) -> None:
    """Return a small JPEG without reserving a persistent streaming connection."""
    harness = CameraStreamMixin()
    harness.cameras = {"USB Camera": 7}
    harness.available_cameras = {
        "USB Camera": {"name": "USB_Camera", "display_name": "Front bumper"}
    }
    harness.frame_locks = {"USB Camera": threading.Lock()}
    harness.frame_list = {"USB Camera": np.zeros((480, 640, 3), dtype=np.uint8)}
    monkeypatch.setattr(camera_stream_module, "request", SimpleNamespace(args={"snapshot": "1"}))
    monkeypatch.setattr(
        camera_stream_module, "Response",
        lambda data, **kwargs: SimpleNamespace(data=data, **kwargs),
    )
    response = harness.serve_camera_feed_route("USB_Camera")
    assert response.mimetype == "image/jpeg"
    assert response.headers["Cache-Control"] == "no-store"
    assert isinstance(response.data, bytes)
    decoded = cv2.imdecode(np.frombuffer(response.data, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[:2] == (240, 320)

    missing = harness.serve_camera_feed_route("disconnected")
    assert missing.mimetype == "image/jpeg"
    assert isinstance(missing.data, bytes)
