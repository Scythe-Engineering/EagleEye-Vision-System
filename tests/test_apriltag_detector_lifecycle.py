"""Regression tests for AprilTag detector native lifecycle handling."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from pytest import MonkeyPatch

from src.main_operations.modules.apriltags import apriltag_detector


def test_update_parameters_waits_for_in_flight_detection(
    monkeypatch: MonkeyPatch,
) -> None:
    """Keep the retired detector alive until its in-flight call returns."""
    detect_entered = threading.Event()
    release_detect = threading.Event()
    destroyed_ids: list[int] = []

    class FakeDetector:
        next_id = 0

        def __init__(self, *args, **kwargs) -> None:
            self.detector_id = FakeDetector.next_id
            FakeDetector.next_id += 1
            self.tag_detector_ptr = object()
            self.tag_families = {"tag36h11": object()}

        def detect(self, _image):
            if self.detector_id == 0:
                detect_entered.set()
                release_detect.wait(timeout=2)
            return []

        def close(self) -> None:
            if self.tag_detector_ptr is not None:
                destroyed_ids.append(self.detector_id)
                self.tag_detector_ptr = None

        def __del__(self) -> None:
            self.close()

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
    assert destroyed_ids == [0, 1, 2]
    detector.close()


def test_tiny_temporal_rois_use_decimate_one_only(monkeypatch: MonkeyPatch) -> None:
    """Route only temporal ROIs below the threshold to decimation one."""
    calls: list[tuple[float, tuple[int, int]]] = []

    class FakeDetector:
        """Record the selected native detector configuration."""

        def __init__(self, *, quad_decimate: float, **_kwargs: object) -> None:
            """Store the decimation setting supplied by the detector wrapper."""
            self.quad_decimate = quad_decimate

        def close(self) -> None:
            """Match the native detector cleanup interface."""

        def detect(self, image: np.ndarray) -> list[object]:
            """Record a detection call without producing a tag."""
            calls.append((self.quad_decimate, image.shape))
            return []

    monkeypatch.setattr(apriltag_detector, "Detector", FakeDetector)
    detector = apriltag_detector.AprilTagDetector(
        quad_decimate=2.0, small_roi_max_px=32
    )

    detector.run_detection(
        [
            (np.zeros((31, 40), dtype=np.uint8), np.zeros(2)),
            (np.zeros((32, 40), dtype=np.uint8), np.zeros(2)),
        ]
    )
    detector.run_detection(np.zeros((31, 40), dtype=np.uint8))

    assert calls == [(1.0, (31, 40)), (2.0, (32, 40)), (2.0, (31, 40))]

    detector.update_parameters(small_roi_max_px=0)
    detector.run_detection([(np.zeros((31, 40), dtype=np.uint8), np.zeros(2))])
    assert calls[-1] == (2.0, (31, 40))


def test_large_temporal_rois_use_the_high_decimation_detector(
    monkeypatch: MonkeyPatch,
) -> None:
    """Route configured large temporal ROIs to their detector bank."""
    calls: list[tuple[float, tuple[int, int]]] = []

    class FakeDetector:
        """Record the selected native detector configuration."""

        def __init__(self, *, quad_decimate: float, **_kwargs: object) -> None:
            """Store the decimation setting supplied by the detector wrapper."""
            self.quad_decimate = quad_decimate

        def close(self) -> None:
            """Match the native detector cleanup interface."""

        def detect(self, image: np.ndarray) -> list[object]:
            """Record a detection call without producing a tag."""
            calls.append((self.quad_decimate, image.shape))
            return []

    monkeypatch.setattr(apriltag_detector, "Detector", FakeDetector)
    detector = apriltag_detector.AprilTagDetector(
        quad_decimate=2.0, large_roi_decimate=3.0, large_roi_min_px=96
    )

    detector.run_detection(
        [
            (np.zeros((95, 120), dtype=np.uint8), np.zeros(2)),
            (np.zeros((96, 120), dtype=np.uint8), np.zeros(2)),
        ]
    )

    assert calls == [(2.0, (95, 120)), (3.0, (96, 120))]


def test_native_close_waits_for_parent_detect(monkeypatch: MonkeyPatch) -> None:
    from src.main_operations.modules.apriltags.native_detector import (
        Detector,
        PupilDetector,
    )

    entered = threading.Event()
    release = threading.Event()
    closed = threading.Event()
    results = []

    def parent_init(self, *args, **kwargs):
        pass

    def parent_detect(self, *args, **kwargs):
        entered.set()
        assert release.wait(2)
        assert not self._closed
        return (args, kwargs)

    monkeypatch.setattr(PupilDetector, "__init__", parent_init)
    monkeypatch.setattr(PupilDetector, "detect", parent_detect)
    detector = Detector("tag16h5", nthreads=1)
    image = np.zeros((16, 16), dtype=np.uint8)
    runner = threading.Thread(
        target=lambda: results.append(detector.detect(image, True, tag_size=0.1))
    )

    def close():
        detector.close()
        closed.set()

    runner.start()
    assert entered.wait(1)
    closer = threading.Thread(target=close)
    closer.start()
    try:
        assert not closed.wait(0.05)
    finally:
        release.set()
        runner.join(2)
        closer.join(2)
    assert not runner.is_alive() and not closer.is_alive()
    assert results[0][0][0] is image
    assert results[0][0][1] is True
    assert results[0][1] == {"tag_size": 0.1}
    with pytest.raises(RuntimeError, match="closed"):
        detector.detect(image)
    detector.close()


@pytest.mark.parametrize("close_during_build", [False, True])
def test_reconfigure_closed_detector_closes_new_bank(
    monkeypatch: MonkeyPatch, close_during_build: bool
) -> None:
    instances = []

    class FakeDetector:
        def __init__(self, **kwargs):
            self.closed = False
            instances.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setattr(apriltag_detector, "Detector", FakeDetector)
    detector = apriltag_detector.AprilTagDetector(full_frame_nthreads=2)
    original_bank = list(instances)
    if close_during_build:
        create = detector._create_detector

        def create_and_close(*args):
            new = create(*args)
            detector.close()
            return new

        monkeypatch.setattr(detector, "_create_detector", create_and_close)
    else:
        detector.close()
    with pytest.raises(RuntimeError, match="closed"):
        detector.update_parameters(quad_sigma=0.5)
    assert all(item.closed for item in instances)
    assert len(instances) == len(original_bank) * (2 if close_during_build else 1)
    assert detector.detector is original_bank[0]
    assert detector.quad_sigma == 0.0


def test_reconfigure_failed_bank_closes_successful_allocations(
    monkeypatch: MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    monkeypatch.setattr(
        apriltag_detector,
        "Detector",
        lambda **kwargs: SimpleNamespace(close=lambda: None),
    )
    detector = apriltag_detector.AprilTagDetector()
    closed = []
    calls = []

    def create(*args):
        calls.append(args)
        if len(calls) == 2:
            raise RuntimeError("construction failed")
        return SimpleNamespace(close=lambda: closed.append(True))

    monkeypatch.setattr(detector, "_create_detector", create)
    with pytest.raises(ValueError, match="construction failed"):
        detector.update_parameters(quad_sigma=0.5)
    assert closed == [True]
    assert detector.ready
    detector.close()
