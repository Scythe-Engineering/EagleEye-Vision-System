"""Pytest fixtures and module stubs for test environment."""

from __future__ import annotations

import importlib.util
import sys
import types
from typing import Any, Callable

import pytest

from tests.utils.rust_build import ensure_rust_modules_built


def _profile_stub(func: Callable[..., Any]) -> Callable[..., Any]:
    return func


class _LineProfilerStub:
    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        return func

    def add_function(self, _func: Callable[..., Any]) -> None:
        return None

    def print_stats(self, stream: Any | None = None) -> None:
        if stream is not None:
            stream.write("")


line_profiler_stub = types.ModuleType("line_profiler")
setattr(line_profiler_stub, "profile", _profile_stub)
setattr(line_profiler_stub, "LineProfiler", _LineProfilerStub)
sys.modules.setdefault("line_profiler", line_profiler_stub)


cv2_stub = types.ModuleType("cv2")

def _noop(*_args: Any, **_kwargs: Any) -> Any:
    return None


def _imdecode(*_args: Any, **_kwargs: Any) -> Any:
    return None


def _imencode(*_args: Any, **_kwargs: Any) -> Any:
    class _EncodedBytes:
        def tobytes(self) -> bytes:
            return b""

    return True, _EncodedBytes()


setattr(cv2_stub, "imdecode", _imdecode)
setattr(cv2_stub, "imencode", _imencode)
setattr(cv2_stub, "IMREAD_COLOR", 1)
setattr(cv2_stub, "__getattr__", lambda _name: _noop)
if importlib.util.find_spec("cv2") is None:
    sys.modules.setdefault("cv2", cv2_stub)


networktables_stub = types.ModuleType("networktables")

class _NetworkTable:  # noqa: N801
    pass


setattr(networktables_stub, "NetworkTable", _NetworkTable)
sys.modules.setdefault("networktables", networktables_stub)


torch_stub = types.ModuleType("torch")
setattr(torch_stub, "Tensor", object)
sys.modules.setdefault("torch", torch_stub)


flask_stub = types.ModuleType("flask")

class _Flask:  # noqa: N801
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        return None


setattr(flask_stub, "Flask", _Flask)
setattr(flask_stub, "Response", object)
setattr(flask_stub, "request", object)
setattr(flask_stub, "send_from_directory", _noop)
sys.modules.setdefault("flask", flask_stub)

flask_cors_stub = types.ModuleType("flask_cors")
setattr(flask_cors_stub, "CORS", _noop)
sys.modules.setdefault("flask_cors", flask_cors_stub)

flask_socketio_stub = types.ModuleType("flask_socketio")

class _SocketIO:  # noqa: N801
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        return None


setattr(flask_socketio_stub, "SocketIO", _SocketIO)
sys.modules.setdefault("flask_socketio", flask_socketio_stub)

apriltag_stub = types.ModuleType("apriltag")
sys.modules.setdefault("apriltag", apriltag_stub)

pupil_apriltags_stub = types.ModuleType("pupil_apriltags")

class _StubDetection:  # noqa: N801
    def __init__(self, tag_id: int = 0, corners: Any | None = None) -> None:
        self.tag_id = tag_id
        self.corners = corners


class _StubDetector:  # noqa: N801
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        return None

    def detect(self, _image: Any, estimate_tag_pose: bool = False) -> list[_StubDetection]:
        return []


setattr(pupil_apriltags_stub, "Detection", _StubDetection)
setattr(pupil_apriltags_stub, "Detector", _StubDetector)
if importlib.util.find_spec("pupil_apriltags") is None:
    sys.modules.setdefault("pupil_apriltags", pupil_apriltags_stub)


def pytest_sessionstart(session: pytest.Session) -> None:
    """Build Rust extensions before collecting tests."""

    try:
        ensure_rust_modules_built()
    except RuntimeError as exc:
        pytest.exit(str(exc), returncode=1)
