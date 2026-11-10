"""A camera that stops delivering frames is reopened instead of abandoned."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from src.utils.camera_utils import camera_thread_manager
from src.utils.camera_utils.camera_thread_manager import (
    CameraThreadManager,
    CameraWorker,
)
from src.utils.camera_utils.cameras.captured_frame import CapturedFrame


class FlakyCamera:
    """Streams nothing until reopened, like a camera that was just replugged."""

    camera_ready = True

    def __init__(self) -> None:
        """Start as opened once, with no settings applied."""
        self.opens = 1
        self.applied: list[dict[str, int]] = []

    def get_frame(self) -> CapturedFrame | None:
        """Return a frame only once the camera has been reopened."""
        if self.opens < 2:
            return None
        return CapturedFrame(np.zeros((4, 4), np.uint8), time.monotonic_ns())

    def get_achieved_fps(self) -> int:
        """Report a fast camera so the loop does not sleep long."""
        return 1000

    def reopen(self) -> None:
        """Count a fresh open."""
        self.opens += 1

    def set_controls(self, values: dict[str, int]) -> list[str]:
        """Record the settings applied after a reopen."""
        self.applied.append(values)
        return []

    def close(self) -> None:
        """Nothing to release."""


def test_worker_reopens_a_silent_camera_and_restores_its_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Frames resume after a reopen, and the saved settings are applied again."""
    monkeypatch.setattr(camera_thread_manager, "REOPEN_DELAY_S", 0.0)
    camera = FlakyCamera()
    frames: list[np.ndarray] = []
    manager = CameraThreadManager.__new__(CameraThreadManager)
    manager.web_interface = SimpleNamespace(
        update_camera_frame=lambda _name, image: frames.append(image)
    )
    manager.logger = SimpleNamespace(log=lambda _message: None)
    manager.bus_id_to_name = {"1-1": "cam"}
    manager._config_registry = SimpleNamespace(
        get_config=lambda _bus_id: SimpleNamespace(camera_controls={"gain": 50})
    )
    worker = CameraWorker("cam", camera)
    manager.cameras = {"cam": worker}

    thread = threading.Thread(target=manager.camera_feed_worker, args=(worker,))
    thread.start()
    deadline = time.monotonic() + 5
    while not frames and time.monotonic() < deadline:
        time.sleep(0.01)
    worker.running = False
    thread.join(timeout=5)

    assert frames, "the worker gave up instead of reopening the camera"
    assert camera.opens == 2
    assert camera.applied == [{"gain": 50}]
