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
