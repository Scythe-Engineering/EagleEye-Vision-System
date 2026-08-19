"""Minimal V4L2 MMAP capture with kernel-provided capture timestamps.

OpenCV's ``VideoCapture`` decodes MJPEG inside ``read()`` before returning the
buffer to the driver, so a shallow buffer queue starves the camera and a deep
one hides how long a frame waited. Neither case exposes when the frame was
actually captured.

This module talks to V4L2 directly so that:

* several buffers stay queued, keeping the driver fed at full frame rate;
* every dequeued buffer carries the kernel's ``CLOCK_MONOTONIC`` capture
  timestamp, which ``uvcvideo`` derives from the camera's own clock rather than
  from when this process was scheduled;
* the compressed payload is copied out and the buffer is requeued immediately,
  moving JPEG decode out of the window where the driver has nowhere to write.

Linux only. Callers are expected to check :func:`v4l2_is_supported` first.
"""

from __future__ import annotations

import ctypes
import errno
import mmap
import os
import select
import struct
import sys
import time
from typing import Callable

try:  # fcntl and the V4L2 ioctl surface exist only on Linux.
    import fcntl
except ImportError:  # pragma: no cover - exercised on non-Linux dev hosts
    fcntl = None  # type: ignore[assignment]

import cv2
import numpy as np

from src.utils.camera_utils.cameras.captured_frame import CapturedFrame
from src.utils.colors import Colors

# --- ioctl encoding (asm-generic) -------------------------------------------

_IOC_SIZESHIFT = 16
_IOC_DIRSHIFT = 30
_IOC_WRITE = 1
_IOC_READ = 2


def _ioc(direction: int, type_char: str, number: int, size: int) -> int:
    """Encode an ioctl request number the way ``<asm-generic/ioctl.h>`` does."""
    request = (
        (direction << _IOC_DIRSHIFT)
        | (size << _IOC_SIZESHIFT)
        | (ord(type_char) << 8)
        | number
    )
    # fcntl.ioctl wants a C int; V4L2 requests set the high direction bits.
    return struct.unpack("i", struct.pack("I", request))[0]


def _iow(type_char: str, number: int, size: int) -> int:
    return _ioc(_IOC_WRITE, type_char, number, size)


def _ior(type_char: str, number: int, size: int) -> int:
    return _ioc(_IOC_READ, type_char, number, size)


def _iowr(type_char: str, number: int, size: int) -> int:
    return _ioc(_IOC_READ | _IOC_WRITE, type_char, number, size)


# --- structures --------------------------------------------------------------


class _Timeval(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_usec", ctypes.c_long)]


class _Timecode(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("frames", ctypes.c_uint8),
        ("seconds", ctypes.c_uint8),
        ("minutes", ctypes.c_uint8),
        ("hours", ctypes.c_uint8),
        ("userbits", ctypes.c_uint8 * 4),
    ]


class _PixFormat(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("pixelformat", ctypes.c_uint32),
        ("field", ctypes.c_uint32),
        ("bytesperline", ctypes.c_uint32),
        ("sizeimage", ctypes.c_uint32),
        ("colorspace", ctypes.c_uint32),
        ("priv", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("ycbcr_enc", ctypes.c_uint32),
        ("quantization", ctypes.c_uint32),
        ("xfer_func", ctypes.c_uint32),
    ]


class _FormatUnion(ctypes.Union):
    # ``struct v4l2_window`` holds pointers, so the kernel union is 8-aligned.
    # The dummy member reproduces that alignment and therefore the total size.
    _fields_ = [
        ("pix", _PixFormat),
        ("raw_data", ctypes.c_uint8 * 200),
        ("_alignment", ctypes.c_uint64),
    ]


class _Format(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint32), ("fmt", _FormatUnion)]


class _RequestBuffers(ctypes.Structure):
    _fields_ = [
        ("count", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("memory", ctypes.c_uint32),
        ("capabilities", ctypes.c_uint32),
        ("flags", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 3),
    ]


class _BufferUnion(ctypes.Union):
    _fields_ = [
        ("offset", ctypes.c_uint32),
        ("userptr", ctypes.c_ulong),
        ("planes", ctypes.c_void_p),
        ("fd", ctypes.c_int32),
    ]


class _Buffer(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("bytesused", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("field", ctypes.c_uint32),
        ("timestamp", _Timeval),
        ("timecode", _Timecode),
        ("sequence", ctypes.c_uint32),
        ("memory", ctypes.c_uint32),
        ("m", _BufferUnion),
        ("length", ctypes.c_uint32),
        ("reserved2", ctypes.c_uint32),
        ("request_fd", ctypes.c_int32),
    ]


class _Fract(ctypes.Structure):
    _fields_ = [("numerator", ctypes.c_uint32), ("denominator", ctypes.c_uint32)]


class _CaptureParm(ctypes.Structure):
    _fields_ = [
        ("capability", ctypes.c_uint32),
        ("capturemode", ctypes.c_uint32),
        ("timeperframe", _Fract),
        ("extendedmode", ctypes.c_uint32),
        ("readbuffers", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 4),
    ]


class _StreamParmUnion(ctypes.Union):
    _fields_ = [("capture", _CaptureParm), ("raw_data", ctypes.c_uint8 * 200)]


class _StreamParm(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint32), ("parm", _StreamParmUnion)]


class _Capability(ctypes.Structure):
    _fields_ = [
        ("driver", ctypes.c_uint8 * 16),
        ("card", ctypes.c_uint8 * 32),
        ("bus_info", ctypes.c_uint8 * 32),
        ("version", ctypes.c_uint32),
        ("capabilities", ctypes.c_uint32),
        ("device_caps", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 3),
    ]


class _Control(ctypes.Structure):
    _fields_ = [("id", ctypes.c_uint32), ("value", ctypes.c_int32)]


# --- constants ---------------------------------------------------------------

VIDIOC_QUERYCAP = _ior("V", 0, ctypes.sizeof(_Capability))
VIDIOC_S_FMT = _iowr("V", 5, ctypes.sizeof(_Format))
VIDIOC_REQBUFS = _iowr("V", 8, ctypes.sizeof(_RequestBuffers))
VIDIOC_QUERYBUF = _iowr("V", 9, ctypes.sizeof(_Buffer))
VIDIOC_QBUF = _iowr("V", 15, ctypes.sizeof(_Buffer))
VIDIOC_DQBUF = _iowr("V", 17, ctypes.sizeof(_Buffer))
VIDIOC_STREAMON = _iow("V", 18, ctypes.sizeof(ctypes.c_int32))
VIDIOC_STREAMOFF = _iow("V", 19, ctypes.sizeof(ctypes.c_int32))
VIDIOC_G_PARM = _iowr("V", 21, ctypes.sizeof(_StreamParm))
VIDIOC_S_PARM = _iowr("V", 22, ctypes.sizeof(_StreamParm))
VIDIOC_S_CTRL = _iowr("V", 28, ctypes.sizeof(_Control))

BUF_TYPE_VIDEO_CAPTURE = 1
MEMORY_MMAP = 1
CAP_VIDEO_CAPTURE = 0x00000001
PIX_FMT_MJPEG = int.from_bytes(b"MJPG", "little")

BUF_FLAG_ERROR = 0x00000040
BUF_FLAG_TIMESTAMP_MASK = 0x0000E000
BUF_FLAG_TIMESTAMP_MONOTONIC = 0x00002000
BUF_FLAG_TSTAMP_SRC_MASK = 0x00070000
BUF_FLAG_TSTAMP_SRC_SOE = 0x00010000

CID_FOCUS_AUTO = 0x009A090C

DEFAULT_BUFFER_COUNT = 4

# A monochrome sensor still ships 3-component YCbCr JPEG, but both chroma
# planes sit on the neutral point (128). Measured on an OV9281 the mean
# departure from neutral is under 2 counts, all of it JPEG ringing at
# high-contrast edges; a colour camera on a real scene is an order of magnitude
# above that.
NEUTRAL_CHROMA = 128
MONO_CHROMA_THRESHOLD = 6.0
MONO_PROBE_FRAMES = 5


def v4l2_is_supported() -> bool:
    """Return whether this platform exposes the V4L2 capture interface."""
    return sys.platform.startswith("linux") and fcntl is not None


class V4l2CaptureError(RuntimeError):
    """Raised when the V4L2 device cannot be configured or streamed."""


class V4l2Capture:
    """MJPEG capture from a V4L2 device using memory-mapped buffers.

    The queue is kept deliberately deep. Depth costs nothing here because every
    buffer carries its own capture timestamp and :meth:`read` always drains to
    the newest frame, so a backlog is discarded rather than delivered late.
    """

    def __init__(
        self,
        device_path: str,
        frame_width: int,
        frame_height: int,
        buffer_count: int = DEFAULT_BUFFER_COUNT,
        log: Callable[[str], None] = print,
    ) -> None:
        """Open and configure the device, then start streaming.

        Args:
            device_path: Device node, for example ``/dev/video0``.
            frame_width: Requested capture width in pixels.
            frame_height: Requested capture height in pixels.
            buffer_count: Number of driver-owned buffers to request.
            log: Logging callable.

        Raises:
            V4l2CaptureError: If the device cannot be opened or configured.
        """
        self.device_path = device_path
        self.log = log
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_rate = 0
        self.timestamp_source = "unknown"
        self.decodes_grayscale: bool | None = None
        self._chroma_departures: list[float] = []

        self._fd = -1
        self._buffers: list[mmap.mmap] = []
        self._streaming = False
        self._warned_about_timestamps = False

        if not v4l2_is_supported():
            raise V4l2CaptureError("V4L2 capture is only available on Linux")

        try:
            self._fd = os.open(device_path, os.O_RDWR | os.O_NONBLOCK)
        except OSError as error:
            raise V4l2CaptureError(f"Cannot open {device_path}: {error}") from error

        try:
            self._verify_capture_device()
            self._set_format()
            self._map_buffers(buffer_count)
            self._start_streaming()
        except Exception:
            self.close()
            raise

    # --- setup ---------------------------------------------------------------

    def _ioctl(self, request: int, argument: ctypes.Structure) -> None:
        """Issue an ioctl, retrying the interrupted-syscall case."""
        while True:
            try:
                fcntl.ioctl(self._fd, request, argument)
                return
            except InterruptedError:
                continue

    def _verify_capture_device(self) -> None:
        """Confirm the node is a video capture device."""
        capability = _Capability()
        try:
            self._ioctl(VIDIOC_QUERYCAP, capability)
        except OSError as error:
            raise V4l2CaptureError(
                f"{self.device_path} is not a V4L2 device: {error}"
            ) from error

        capabilities = capability.device_caps or capability.capabilities
        if not capabilities & CAP_VIDEO_CAPTURE:
            raise V4l2CaptureError(
                f"{self.device_path} does not support video capture"
            )

    def _set_format(self) -> None:
        """Negotiate MJPEG at the requested resolution."""
        video_format = _Format()
        video_format.type = BUF_TYPE_VIDEO_CAPTURE
        video_format.fmt.pix.width = self.frame_width
        video_format.fmt.pix.height = self.frame_height
        video_format.fmt.pix.pixelformat = PIX_FMT_MJPEG
        video_format.fmt.pix.field = 0  # V4L2_FIELD_ANY

        try:
            self._ioctl(VIDIOC_S_FMT, video_format)
        except OSError as error:
            raise V4l2CaptureError(
                f"{self.device_path} rejected MJPEG "
                f"{self.frame_width}x{self.frame_height}: {error}"
            ) from error

        if video_format.fmt.pix.pixelformat != PIX_FMT_MJPEG:
            raise V4l2CaptureError(
                f"{self.device_path} does not support MJPEG capture"
            )

        # The driver may round the resolution; record what it actually gave us.
        self.frame_width = int(video_format.fmt.pix.width)
        self.frame_height = int(video_format.fmt.pix.height)

    def set_frame_rate(self, frames_per_second: int) -> int:
        """Request a frame interval and return the rate the driver accepted.

        Args:
            frames_per_second: Desired capture rate.

        Returns:
            The achieved rate, or 0 when the device does not report one.
        """
        stream_parameters = _StreamParm()
        stream_parameters.type = BUF_TYPE_VIDEO_CAPTURE
        stream_parameters.parm.capture.timeperframe.numerator = 1
        stream_parameters.parm.capture.timeperframe.denominator = frames_per_second

        for request in (VIDIOC_S_PARM, VIDIOC_G_PARM):
            try:
                self._ioctl(request, stream_parameters)
            except OSError as error:
                self.log(
                    f"{Colors.YELLOW}{self.device_path}: frame rate control "
                    f"unavailable ({error}){Colors.RESET}"
                )
                return self.frame_rate

        interval = stream_parameters.parm.capture.timeperframe
        if interval.numerator > 0 and interval.denominator > 0:
            self.frame_rate = round(interval.denominator / interval.numerator)
        return self.frame_rate

    def set_control(self, control_id: int, value: int) -> bool:
        """Best-effort control write. Returns whether the driver accepted it."""
        control = _Control(id=control_id, value=value)
        try:
            self._ioctl(VIDIOC_S_CTRL, control)
            return True
        except OSError:
            return False

    def _map_buffers(self, buffer_count: int) -> None:
        """Request driver buffers and memory-map each one."""
        request = _RequestBuffers()
        request.count = buffer_count
        request.type = BUF_TYPE_VIDEO_CAPTURE
        request.memory = MEMORY_MMAP

        try:
            self._ioctl(VIDIOC_REQBUFS, request)
        except OSError as error:
            raise V4l2CaptureError(
                f"{self.device_path} refused {buffer_count} buffers: {error}"
            ) from error

        if request.count < 2:
            raise V4l2CaptureError(
                f"{self.device_path} granted only {request.count} buffers; "
                "at least 2 are needed to stream without dropping frames"
            )

        for index in range(request.count):
            buffer = _Buffer()
            buffer.type = BUF_TYPE_VIDEO_CAPTURE
            buffer.memory = MEMORY_MMAP
            buffer.index = index
            self._ioctl(VIDIOC_QUERYBUF, buffer)
            self._buffers.append(
                mmap.mmap(
                    self._fd,
                    buffer.length,
                    flags=mmap.MAP_SHARED,
                    prot=mmap.PROT_READ | mmap.PROT_WRITE,
                    offset=buffer.m.offset,
                )
            )

    def _start_streaming(self) -> None:
        """Queue every buffer and turn the stream on."""
        for index in range(len(self._buffers)):
            self._queue_buffer(index)

        stream_type = ctypes.c_int32(BUF_TYPE_VIDEO_CAPTURE)
        try:
            self._ioctl(VIDIOC_STREAMON, stream_type)
        except OSError as error:
            raise V4l2CaptureError(
                f"{self.device_path} failed to start streaming: {error}"
            ) from error
        self._streaming = True

    # --- capture -------------------------------------------------------------

    def _queue_buffer(self, index: int) -> None:
        """Hand a buffer back to the driver."""
        buffer = _Buffer()
        buffer.type = BUF_TYPE_VIDEO_CAPTURE
        buffer.memory = MEMORY_MMAP
        buffer.index = index
        self._ioctl(VIDIOC_QBUF, buffer)

    def _dequeue_buffer(self) -> _Buffer | None:
        """Take one filled buffer, or None when the queue is empty."""
        buffer = _Buffer()
        buffer.type = BUF_TYPE_VIDEO_CAPTURE
        buffer.memory = MEMORY_MMAP
        try:
            self._ioctl(VIDIOC_DQBUF, buffer)
        except OSError as error:
            if error.errno == errno.EAGAIN:
                return None
            raise
        return buffer

    def _capture_monotonic_ns(self, buffer: _Buffer) -> int:
        """Convert a buffer timestamp to CLOCK_MONOTONIC nanoseconds.

        Falls back to the delivery time when the driver does not report a
        monotonic timestamp, which is no worse than the old OpenCV behaviour.
        """
        if buffer.flags & BUF_FLAG_TIMESTAMP_MASK != BUF_FLAG_TIMESTAMP_MONOTONIC:
            if not self._warned_about_timestamps:
                self._warned_about_timestamps = True
                self.timestamp_source = "delivery"
                self.log(
                    f"{Colors.YELLOW}{self.device_path}: driver does not provide "
                    f"monotonic capture timestamps; falling back to delivery "
                    f"time{Colors.RESET}"
                )
            return time.monotonic_ns()

        if self.timestamp_source in {"unknown", "delivery"}:
            self.timestamp_source = (
                "start-of-exposure"
                if buffer.flags & BUF_FLAG_TSTAMP_SRC_MASK == BUF_FLAG_TSTAMP_SRC_SOE
                else "end-of-frame"
            )

        return buffer.timestamp.tv_sec * 1_000_000_000 + buffer.timestamp.tv_usec * 1000

    def _classify_decode_mode(self, image: np.ndarray) -> None:
        """Latch grayscale decoding once enough frames show no chroma.

        A monochrome camera is not identifiable from the JPEG header: the
        firmware wraps the sensor's single plane in a 3-component YCbCr stream,
        so the header claims colour. Measuring the chroma planes answers the
        question the header cannot.

        The measure is how far chroma sits from neutral, not how much it varies:
        a uniformly red scene has almost no chroma variance but is emphatically
        not monochrome.

        Args:
            image: A colour-decoded frame captured during the probe.
        """
        # An 8x8 subsample characterises the planes well enough and keeps the
        # probe far cheaper than the decode it rides along with.
        chroma = cv2.cvtColor(image[::8, ::8], cv2.COLOR_BGR2YCrCb)[:, :, 1:]
        departure = np.abs(chroma.astype(np.int16) - NEUTRAL_CHROMA).mean()
        self._chroma_departures.append(float(departure))
        if len(self._chroma_departures) < MONO_PROBE_FRAMES:
            return

        # Every probe frame must look neutral; one colourful frame is proof of a
        # colour camera, while one grey frame proves nothing.
        worst = max(self._chroma_departures)
        self.decodes_grayscale = worst < MONO_CHROMA_THRESHOLD
        self.log(
            f"{Colors.CYAN}{self.device_path}: chroma departure {worst:.2f}, "
            f"decoding as {'grayscale' if self.decodes_grayscale else 'colour'}"
            f"{Colors.RESET}"
        )

    def read(self, timeout_s: float = 1.0) -> CapturedFrame | None:
        """Return the newest available frame, or None if none arrived in time.

        Every queued buffer is drained and requeued; only the newest payload is
        decoded. Older frames are discarded undecoded, which both keeps the
        driver fed and avoids paying for a decode that would be thrown away.

        The first few frames are decoded in colour to measure whether the camera
        carries any chroma. A monochrome camera then decodes as single-channel
        for the rest of the session, which is both faster and free of the JPEG
        chroma ringing that colour decoding folds into the luma.

        Args:
            timeout_s: How long to wait for the device to become readable.

        Returns:
            The newest frame with its capture timestamp, or None on timeout or
            when every pending buffer was in error.

        Raises:
            V4l2CaptureError: If the device has been closed.
            OSError: On an unrecoverable ioctl failure.
        """
        if self._fd < 0:
            raise V4l2CaptureError(f"{self.device_path} is closed")

        readable, _, _ = select.select([self._fd], [], [], timeout_s)
        if not readable:
            return None

        payload: bytes | None = None
        capture_monotonic_ns = 0

        while (buffer := self._dequeue_buffer()) is not None:
            usable = not buffer.flags & BUF_FLAG_ERROR and buffer.bytesused > 0
            if usable:
                # Copy the compressed bytes so the buffer can go straight back
                # to the driver; decoding happens after it is requeued.
                payload = self._buffers[buffer.index][: buffer.bytesused]
                capture_monotonic_ns = self._capture_monotonic_ns(buffer)
            self._queue_buffer(buffer.index)

        if payload is None:
            return None

        probing = self.decodes_grayscale is None
        decode_flag = (
            cv2.IMREAD_COLOR if probing or not self.decodes_grayscale
            else cv2.IMREAD_GRAYSCALE
        )
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), decode_flag)
        if image is None:
            return None
        if probing:
            self._classify_decode_mode(image)
        return CapturedFrame(image=image, capture_monotonic_ns=capture_monotonic_ns)

    # --- teardown ------------------------------------------------------------

    def close(self) -> None:
        """Stop streaming, unmap buffers, and close the device."""
        if self._streaming:
            try:
                self._ioctl(VIDIOC_STREAMOFF, ctypes.c_int32(BUF_TYPE_VIDEO_CAPTURE))
            except OSError:
                pass
            self._streaming = False

        for buffer in self._buffers:
            try:
                buffer.close()
            except (BufferError, ValueError):
                pass
        self._buffers.clear()

        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1
