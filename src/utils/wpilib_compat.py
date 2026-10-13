"""Version-selected boundary for WPILib 2026 and 2027 alpha-7 bindings.

Application capture timestamps stay in microseconds; only this boundary uses
native NT units. Unsupported or mixed installations fail at import time.
"""
from __future__ import annotations

import importlib.util
from importlib import import_module
from types import ModuleType
from importlib.metadata import version
from typing import Any

import ntcore
import wpimath


def _detect_api(nt_version: str, math_version: str, *, nt_snake: bool, math_flat: bool) -> str:
    expected = "2027" if nt_snake else "2026"
    def supported(value: str) -> bool:
        return value == "2027.0.0a7" if nt_snake else value.startswith("2026.")
    if nt_snake != math_flat or not supported(nt_version) or not supported(math_version):
        raise ImportError(
            "Unsupported or mixed WPILib installation: "
            f"pyntcore={nt_version}, robotpy-wpimath={math_version}. "
            "Install matching 2026 packages or matching 2027.0.0a7 packages."
        )
    return expected


NT_VERSION = version("pyntcore")
MATH_VERSION = version("robotpy-wpimath")
API = _detect_api(
    NT_VERSION, MATH_VERSION,
    nt_snake=hasattr(ntcore.NetworkTableInstance, "get_default"),
    math_flat=importlib.util.find_spec("wpimath.geometry") is None,
)
NT_TIME_UNITS_PER_US = 1000 if API == "2027" else 1
_geometry: ModuleType
if API == "2026":
    _geometry = import_module("wpimath.geometry")
else:
    _geometry = wpimath
Pose2d = _geometry.Pose2d
Pose3d = _geometry.Pose3d
Quaternion = _geometry.Quaternion
Rotation2d = _geometry.Rotation2d
Rotation3d = _geometry.Rotation3d
Transform2d = _geometry.Transform2d
Transform3d = _geometry.Transform3d
Translation2d = _geometry.Translation2d
Translation3d = _geometry.Translation3d


def _call(obj: Any, modern: str, legacy: str, *args: Any, **kwargs: Any) -> Any:
    return getattr(obj, modern if API == "2027" else legacy)(*args, **kwargs)


def default_instance() -> Any:
    return _call(ntcore.NetworkTableInstance, "get_default", "getDefault")


def create_instance() -> Any:
    return ntcore.NetworkTableInstance.create()


def destroy_instance(instance: Any) -> None:
    ntcore.NetworkTableInstance.destroy(instance)


def start_client(instance: Any, identity: str, server: str, port: int = 5810) -> None:
    _call(instance, "start_client", "startClient4", identity)
    _call(instance, "set_server", "setServer", server, port)


def start_local(instance: Any) -> None:
    _call(instance, "start_local", "startLocal")


def stop_local(instance: Any) -> None:
    _call(instance, "stop_local", "stopLocal")


def start_server(instance: Any, port: int) -> None:
    if API == "2027":
        instance.start_server("", "127.0.0.1", "", port)
    else:
        instance.startServer("", "127.0.0.1", 0, port)


def stop_server(instance: Any) -> None:
    _call(instance, "stop_server", "stopServer")


def get_table(instance: Any, name: str) -> Any:
    return _call(instance, "get_table", "getTable", name)


def table_instance(table: Any) -> Any:
    return _call(table, "get_instance", "getInstance")


def flush(instance: Any) -> None:
    instance.flush()


def is_connected(instance: Any) -> bool:
    return bool(_call(instance, "is_connected", "isConnected"))


def connection_count(instance: Any) -> int:
    return len(_call(instance, "get_connections", "getConnections"))


def topic_names(instance: Any) -> set[str]:
    return {str(_call(item, "get_name", "getName")) for item in _call(instance, "get_topics", "getTopics")}


def read_bool(table: Any, key: str, default: bool) -> bool:
    return _call(_call(table, "get_entry", "getEntry", key), "get_boolean", "getBoolean", default)


def read_double(table: Any, key: str, default: float) -> float:
    return _call(_call(table, "get_entry", "getEntry", key), "get_double", "getDouble", default)


def pubsub_options(*, keep_duplicates: bool = False, send_all: bool = False,
                   periodic: float = 0.1, poll_storage: int = 0) -> Any:
    # The installed stubs describe one binding version; dispatch selects the
    # corresponding keyword names without requiring both stub sets to exist.
    kwargs: dict[str, Any] = {"periodic": periodic}
    if API == "2027":
        kwargs.update(keep_duplicates=keep_duplicates, send_all=send_all,
                      poll_storage=poll_storage)
    else:
        kwargs.update(keepDuplicates=keep_duplicates, sendAll=send_all,
                      pollStorage=poll_storage)
    return ntcore.PubSubOptions(**kwargs)


def topic(table: Any, kind: str, key: str, struct_type: type | None = None) -> Any:
    legacy = "get" + "".join(part.title() for part in kind.split("_")) + "Topic"
    args = (key, struct_type) if kind in {"struct", "struct_array"} else (key,)
    return _call(table, f"get_{kind}_topic", legacy, *args)


def read_queue(subscriber: Any) -> list[Any]:
    return _call(subscriber, "read_queue", "readQueue")


def sample_timestamp(sample: Any) -> int:
    return sample.time


def raw_nt_now() -> int:
    return ntcore._now()


def to_nt_publish_time(capture_nt_us: int) -> int:
    return capture_nt_us * NT_TIME_UNITS_PER_US


def pose_xyz(pose: Any) -> tuple[float, float, float]:
    return (pose.x, pose.y, pose.z) if API == "2027" else (pose.X(), pose.Y(), pose.Z())


def rotation_xyz(rotation: Any) -> tuple[float, float, float]:
    return pose_xyz(rotation)


def quaternion_wxyz(rotation: Any) -> tuple[float, float, float, float]:
    q = _call(rotation, "get_quaternion", "getQuaternion")
    return (q.w, q.x, q.y, q.z) if API == "2027" else (q.W(), q.X(), q.Y(), q.Z())
