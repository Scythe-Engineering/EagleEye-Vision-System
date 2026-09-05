from __future__ import annotations

import ctypes
import sys

import numpy as np
import pytest

from src.utils.camera_utils.camera_thread_manager import CameraWorker
from src.utils.camera_utils.cameras import v4l2_capture
from src.utils.camera_utils.cameras.physical_camera import DEFAULT_FPS, PhysicalCamera
from src.utils.camera_utils.cameras.v4l2_capture import (
    BUF_FLAG_ERROR,
    BUF_FLAG_TIMESTAMP_MONOTONIC,
    BUF_FLAG_TSTAMP_SRC_SOE,
    V4l2Capture,
    V4l2CaptureError,
    v4l2_is_supported,
)

IS_64_BIT_LINUX = sys.platform.startswith("linux") and ctypes.sizeof(ctypes.c_long) == 8


def _unsigned(request: int) -> int:
    """Render a signed ioctl request the way kernel headers document it."""
    return request & 0xFFFFFFFF


def _make_buffer(
    index: int,
    bytesused: int,
    tv_sec: int = 0,
    tv_usec: int = 0,
    flags: int = BUF_FLAG_TIMESTAMP_MONOTONIC | BUF_FLAG_TSTAMP_SRC_SOE,
) -> v4l2_capture._Buffer:
    """Build a dequeued-buffer stand-in."""
    buffer = v4l2_capture._Buffer()
    buffer.index = index
    buffer.bytesused = bytesused
    buffer.flags = flags
    buffer.timestamp.tv_sec = tv_sec
    buffer.timestamp.tv_usec = tv_usec
    return buffer


def _capture_stub(
    payloads: list[bytes], decodes_grayscale: bool | None = False
) -> V4l2Capture:
    """Build a V4l2Capture with buffers but no real device behind it."""
    capture = V4l2Capture.__new__(V4l2Capture)
    capture.device_path = "/dev/video-test"
    capture.log = lambda _message: None
    capture._fd = 3
    capture._buffers = [bytearray(payload) for payload in payloads]
    capture._streaming = True
    capture._warned_about_timestamps = False
    capture.timestamp_source = "unknown"
    capture.decodes_grayscale = decodes_grayscale
    capture._chroma_departures = []
    return capture


def test_ioctl_request_numbers_match_the_kernel_abi() -> None:
    """The encoded requests must match the documented V4L2 values exactly.

    A wrong size in the request number makes every ioctl fail with ENOTTY, so
    pinning these catches a bad struct definition immediately.
    """
    assert _unsigned(v4l2_capture._ior("V", 0, 104)) == 0x80685600  # QUERYCAP
    assert _unsigned(v4l2_capture._iowr("V", 5, 208)) == 0xC0D05605  # S_FMT
    assert _unsigned(v4l2_capture._iowr("V", 17, 88)) == 0xC0585611  # DQBUF (64-bit)
    assert _unsigned(v4l2_capture._iow("V", 18, 4)) == 0x40045612  # STREAMON


def test_format_size_tracks_the_linux_word_size() -> None:
    """Keep pointer-bearing format alignment valid on 32- and 64-bit Linux."""
    expected_size = 208 if ctypes.sizeof(ctypes.c_ulong) == 8 else 204

    assert ctypes.sizeof(v4l2_capture._Format) == expected_size


@pytest.mark.skipif(not IS_64_BIT_LINUX, reason="ABI sizes are 64-bit Linux specific")
def test_structure_sizes_match_the_kernel_abi() -> None:
    """ctypes layout must reproduce the kernel's struct sizes."""
    assert ctypes.sizeof(v4l2_capture._Capability) == 104
    assert ctypes.sizeof(v4l2_capture._Format) == 208
    assert ctypes.sizeof(v4l2_capture._Buffer) == 88
    assert ctypes.sizeof(v4l2_capture._RequestBuffers) == 20
    assert ctypes.sizeof(v4l2_capture._StreamParm) == 204
    assert ctypes.sizeof(v4l2_capture._Control) == 8


def test_monotonic_buffer_timestamp_is_decoded_to_nanoseconds() -> None:
    capture = _capture_stub([b""])

    buffer = _make_buffer(0, bytesused=1, tv_sec=12, tv_usec=345_678)

    assert capture._capture_monotonic_ns(buffer) == 12_345_678_000
    assert capture.timestamp_source == "start-of-exposure"


def test_non_monotonic_timestamps_fall_back_to_delivery_time() -> None:
    """A driver without monotonic stamps must degrade, not report garbage."""
    capture = _capture_stub([b""])
    buffer = _make_buffer(0, bytesused=1, tv_sec=12, tv_usec=345_678, flags=0)

    captured_ns = capture._capture_monotonic_ns(buffer)

    assert captured_ns != 12_345_678_000
    assert capture.timestamp_source == "delivery"


def test_read_drains_to_the_newest_frame_and_requeues_everything(monkeypatch) -> None:
    """Backlogged buffers are discarded undecoded, and all are handed back."""
    capture = _capture_stub([b"oldest", b"middle", b"newest"])
    pending = [
        _make_buffer(0, bytesused=6, tv_sec=1),
        _make_buffer(1, bytesused=6, tv_sec=2),
        _make_buffer(2, bytesused=6, tv_sec=3),
    ]
    requeued: list[int] = []
    decoded: list[bytes] = []

    monkeypatch.setattr(v4l2_capture.select, "select", lambda *_args: ([3], [], []))
    monkeypatch.setattr(
        V4l2Capture, "_dequeue_buffer", lambda _self: pending.pop(0) if pending else None
    )
    monkeypatch.setattr(V4l2Capture, "_queue_buffer", lambda _self, i: requeued.append(i))
    monkeypatch.setattr(
        v4l2_capture.cv2,
        "imdecode",
        lambda data, _flags: decoded.append(bytes(data)) or np.zeros((2, 2, 3)),
    )

    frame = capture.read()

    assert frame is not None
    assert frame.capture_monotonic_ns == 3_000_000_000
    assert requeued == [0, 1, 2]
    assert decoded == [b"newest"]


def test_read_skips_error_buffers_but_still_requeues_them(monkeypatch) -> None:
    capture = _capture_stub([b"broken"])
    pending = [_make_buffer(0, bytesused=6, flags=BUF_FLAG_ERROR)]
    requeued: list[int] = []

    monkeypatch.setattr(v4l2_capture.select, "select", lambda *_args: ([3], [], []))
    monkeypatch.setattr(
        V4l2Capture, "_dequeue_buffer", lambda _self: pending.pop(0) if pending else None
    )
    monkeypatch.setattr(V4l2Capture, "_queue_buffer", lambda _self, i: requeued.append(i))

    assert capture.read() is None
    assert requeued == [0]


def test_read_returns_none_when_no_frame_arrives(monkeypatch) -> None:
    capture = _capture_stub([b""])
    monkeypatch.setattr(v4l2_capture.select, "select", lambda *_args: ([], [], []))

    assert capture.read(timeout_s=0.0) is None


def test_read_on_a_closed_device_raises() -> None:
    capture = _capture_stub([b""])
    capture._fd = -1

    with pytest.raises(V4l2CaptureError, match="closed"):
        capture.read()


@pytest.mark.skipif(v4l2_is_supported(), reason="checks the non-Linux guard")
def test_construction_is_refused_without_v4l2() -> None:
    with pytest.raises(V4l2CaptureError, match="only available on Linux"):
        V4l2Capture("/dev/video0", 1280, 720)


def _solid_bgr(value: tuple[int, int, int]) -> np.ndarray:
    """Build a frame whose every pixel carries the given BGR colour."""
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[:, :] = value
    return frame


def test_probe_latches_grayscale_for_a_neutral_chroma_camera() -> None:
    """A mono sensor ships neutral chroma, so it must decode as grayscale."""
    capture = _capture_stub([b""], decodes_grayscale=None)
    rng = np.random.default_rng(0)

    for _ in range(v4l2_capture.MONO_PROBE_FRAMES):
        assert capture.decodes_grayscale is None  # undecided while probing
        luma = rng.integers(0, 255, size=(64, 64), dtype=np.uint8)
        capture._classify_decode_mode(np.repeat(luma[:, :, None], 3, axis=2))

    assert capture.decodes_grayscale is True


def test_probe_latches_colour_when_chroma_is_present() -> None:
    capture = _capture_stub([b""], decodes_grayscale=None)

    for index in range(v4l2_capture.MONO_PROBE_FRAMES):
        capture._classify_decode_mode(
            _solid_bgr((255, 0, 0) if index % 2 else (0, 0, 255))
        )

    assert capture.decodes_grayscale is False


def test_probe_detects_colour_in_a_uniformly_coloured_scene() -> None:
    """Chroma variance is near zero on a flat red scene, but it is still colour."""
    capture = _capture_stub([b""], decodes_grayscale=None)

    for _ in range(v4l2_capture.MONO_PROBE_FRAMES):
        capture._classify_decode_mode(_solid_bgr((0, 0, 255)))

    assert capture.decodes_grayscale is False


def test_probe_needs_every_frame_to_be_neutral() -> None:
    """One colourful frame is enough to rule out a monochrome camera."""
    capture = _capture_stub([b""], decodes_grayscale=None)
    grey = _solid_bgr((128, 128, 128))

    for index in range(v4l2_capture.MONO_PROBE_FRAMES):
        capture._classify_decode_mode(_solid_bgr((0, 0, 255)) if index == 2 else grey)

    assert capture.decodes_grayscale is False


def test_failed_advertised_fps_negotiation_uses_fallbacks() -> None:
    """Do not report an advertised rate that the backend rejected."""
    camera = PhysicalCamera.__new__(PhysicalCamera)
    camera.get_available_fps_for_resolution = lambda: [120]

    class RejectingBackend:
        """Capture backend that rejects every requested rate."""

        def __init__(self) -> None:
            self.requests: list[int] = []

        def set_frame_rate(self, frames_per_second: int) -> int:
            """Record and reject a requested frame rate."""
            self.requests.append(frames_per_second)
            return 0

    backend = RejectingBackend()

    assert camera._negotiate_frame_rate(backend, []) == DEFAULT_FPS
    assert len(backend.requests) > 1


def test_camera_negotiates_frame_rate_before_streaming() -> None:
    """V4L2 rejects frame-rate changes after streaming starts."""
    events: list[str] = []

    class Backend:
        """Record camera setup calls in order."""

        def set_frame_rate(self, frames_per_second: int) -> int:
            """Accept and record the requested rate."""
            events.append(f"fps:{frames_per_second}")
            return frames_per_second

        def start(self) -> None:
            """Record stream startup."""
            events.append("start")

        def set_control(self, _control_id: int, _value: int) -> bool:
            """Record control setup."""
            events.append("control")
            return True

    camera = PhysicalCamera.__new__(PhysicalCamera)
    camera.camera_index = 0
    camera.frame_width = 1280
    camera.frame_height = 720
    camera.name = "test"
    camera.log = lambda _message: None
    camera.camera_ready = False
    camera.backend = None
    camera.get_available_fps_for_resolution = lambda: (events.append("query") or [100])
    camera._open_backend = lambda: (events.append("open") or Backend())

    camera._start_camera()

    assert events == ["query", "open", "fps:100", "start", "control"]
    assert camera.achieved_fps == 100


def test_worker_does_not_close_camera_while_its_thread_is_alive() -> None:
    """Avoid tearing down capture resources under an active reader."""

    class LiveThread:
        """Thread stand-in that remains alive after joining."""

        def join(self, timeout: float) -> None:
            """Accept the worker's bounded join request."""

        def is_alive(self) -> bool:
            """Report that the simulated reader is still running."""
            return True

    class CameraStub:
        """Camera stand-in that records closure."""

        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            """Record that close was called."""
            self.closed = True

    camera = CameraStub()
    worker = CameraWorker("test", camera)  # type: ignore[arg-type]
    worker.thread = LiveThread()  # type: ignore[assignment]

    worker.stop(timeout=0.0)
    assert not camera.closed

    worker._run(lambda _worker: None)
    assert camera.closed


def test_read_decodes_grayscale_once_the_probe_has_latched(monkeypatch) -> None:
    capture = _capture_stub([b"frame"], decodes_grayscale=True)
    pending = [_make_buffer(0, bytesused=5)]
    flags: list[int] = []

    monkeypatch.setattr(v4l2_capture.select, "select", lambda *_args: ([3], [], []))
    monkeypatch.setattr(
        V4l2Capture, "_dequeue_buffer", lambda _self: pending.pop(0) if pending else None
    )
    monkeypatch.setattr(V4l2Capture, "_queue_buffer", lambda _self, _i: None)
    monkeypatch.setattr(
        v4l2_capture.cv2,
        "imdecode",
        lambda _data, flag: flags.append(flag) or np.zeros((2, 2), dtype=np.uint8),
    )

    assert capture.read() is not None
    assert flags == [v4l2_capture.cv2.IMREAD_GRAYSCALE]
