"""Tests for ground plane projection configuration sources."""

from __future__ import annotations

import json

import pytest

from src.secondary_operations.ground_plane_intersection import GroundPlaneIntersection
from src.utils.camera_utils.camera_config_manager import CameraConfigRegistry


def test_ground_plane_uses_selected_camera_extrinsics(tmp_path) -> None:
    """Ground plane projection should read height/pitch from camera extrinsics."""
    camera_dir = tmp_path / "cam0"
    camera_dir.mkdir()
    intrinsics_path = camera_dir / "intrinsics.json"
    intrinsics_path.write_text(
        json.dumps(
            {
                "camera_matrix": [
                    [100.0, 0.0, 100.0],
                    [0.0, 100.0, 100.0],
                    [0.0, 0.0, 1.0],
                ],
                "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
                "image_width": 200,
                "image_height": 200,
            }
        ),
        encoding="utf-8",
    )

    registry = CameraConfigRegistry(base_path=str(tmp_path))
    camera_config = registry.get_config("cam0")
    camera_config.intrinsics_path = str(intrinsics_path)
    camera_config.update_extrinsics_live(
        {
            "pitch": 45.0,
            "yaw": 0.0,
            "roll": 0.0,
            "x_offset": 0.0,
            "y_offset": 0.0,
            "z_offset": 2.0,
        }
    )

    operation = GroundPlaneIntersection(
        camera_bus_id="cam0",
        camera_height=999.0,
        camera_pitch=0.0,
        camera_config_registry=registry,
    )

    detections = operation.run([{"bbox": [0.45, 0.45, 0.55, 0.5]}])

    assert len(detections) == 1
    assert detections[0]["position_3d"] == pytest.approx([0.0, 0.0, 2.0])
