from __future__ import annotations

import threading

import numpy as np

from src.webui.web_server_utils.camera_calibration_mixin import CameraCalibrationMixin


class _CalibrationHarness(CameraCalibrationMixin):
    def __init__(self) -> None:
        self.available_cameras = {
            "OBS Camera Extension": {
                "name": "OBS_Camera_Extension",
                "id": "0",
                "bus_id": "0",
            },
            "UVC Camera VendorID_3141 ProductID_25446": {
                "name": "UVC_Camera_VendorID_3141_ProductID_25446",
                "id": "2",
                "bus_id": "2",
            },
        }
        self.frame_list = {
            "OBS Camera Extension": np.full((2, 2, 3), 10, dtype=np.uint8),
            "UVC Camera VendorID_3141 ProductID_25446": np.full(
                (2, 2, 3), 20, dtype=np.uint8
            ),
        }
        self.frame_locks = {
            camera_name: threading.Lock() for camera_name in self.frame_list
        }


def test_latest_camera_frame_resolves_selected_camera_by_bus_id() -> None:
    harness = _CalibrationHarness()

    frame = harness._latest_camera_frame("2")

    assert frame is not None
    assert int(frame[0, 0, 0]) == 20


def test_latest_camera_frame_resolves_selected_camera_by_stream_name() -> None:
    harness = _CalibrationHarness()

    frame = harness._latest_camera_frame("UVC_Camera_VendorID_3141_ProductID_25446")

    assert frame is not None
    assert int(frame[0, 0, 0]) == 20


def test_scaled_camera_matrix_matches_live_frame_size() -> None:
    harness = _CalibrationHarness()
    matrix = np.array([[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]])

    scaled = harness._scaled_camera_matrix(matrix, (1280, 720), (640, 360))

    np.testing.assert_allclose(
        scaled,
        [[400.0, 0.0, 320.0], [0.0, 400.0, 180.0], [0.0, 0.0, 1.0]],
    )


def test_distortion_grid_draws_warped_lines() -> None:
    harness = _CalibrationHarness()
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    matrix = np.array([[120.0, 0.0, 80.0], [0.0, 120.0, 60.0], [0.0, 0.0, 1.0]])

    output = harness._draw_distortion_grid(
        frame, matrix, np.array([-0.3, 0.1, 0.0, 0.0, 0.0])
    )

    assert output.shape == frame.shape
    assert np.count_nonzero(output) > 0


def test_straight_reference_grid_draws_lines() -> None:
    harness = _CalibrationHarness()
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    output = harness._draw_straight_grid(frame)

    assert output.shape == frame.shape
    assert np.count_nonzero(output) > 0
