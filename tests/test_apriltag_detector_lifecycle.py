"""Regression tests for AprilTag detector native lifecycle handling."""

from __future__ import annotations

import threading
import time

import numpy as np

from src.main_operations.modules.apriltags import apriltag_detector


def test_update_parameters_waits_for_in_flight_detection(monkeypatch) -> None:
    detect_entered = threading.Event()
    release_detect = threading.Event()
    destroyed_ids: list[int] = []

    class FakeDetector:
        next_id = 0

        def __init__(self, *args, **kwargs) -> None:
            self.detector_id = FakeDetector.next_id
            FakeDetector.next_id += 1

        def detect(self, _image):
            if self.detector_id == 0:
                detect_entered.set()
                release_detect.wait(timeout=2)
            return []

        def __del__(self) -> None:
            destroyed_ids.append(self.detector_id)

    monkeypatch.setattr(apriltag_detector, "Detector", FakeDetector)

    detector = apriltag_detector.AprilTagDetector()
    image = np.zeros((16, 16), dtype=np.uint8)

    detect_thread = threading.Thread(target=detector.run_detection, args=(image,))
    detect_thread.start()
    assert detect_entered.wait(timeout=1)

    update_thread = threading.Thread(
        target=detector.update_parameters,
        kwargs={"quad_decimate": 1.0},
    )
    update_thread.start()
    time.sleep(0.05)

    assert update_thread.is_alive()
    assert 0 not in destroyed_ids

    release_detect.set()
    detect_thread.join(timeout=1)
    update_thread.join(timeout=1)

    assert not detect_thread.is_alive()
    assert not update_thread.is_alive()
    assert detector.quad_decimate == 1.0
    assert 0 in destroyed_ids
