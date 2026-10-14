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


def _detect_api(
    nt_version: str, math_version: str, *, nt_snake: bool, math_flat: bool
) -> str:
    """Select a supported API from package versions and binding shapes.

    Args:
        nt_version: Installed pyntcore version.
        math_version: Installed robotpy-wpimath version.
        nt_snake: Whether NT exposes the 2027 snake-case API.
        math_flat: Whether geometry uses the 2027 flat module layout.

    Returns:
        The matching API year, either "2026" or "2027".

    Raises:
        ImportError: If packages are unsupported or use different API generations.
    """
    expected = "2027" if nt_snake else "2026"

    def supported(value: str) -> bool:
        """Check whether a version matches the detected NT API.

        Args:
            value: Installed package version.

        Returns:
            Whether the version is supported for the detected API.
        """
        return value == "2027.0.0a7" if nt_snake else value.startswith("2026.")

    if (
        nt_snake != math_flat
        or not supported(nt_version)
        or not supported(math_version)
    ):
        raise ImportError(
            "Unsupported or mixed WPILib installation: "
            f"pyntcore={nt_version}, robotpy-wpimath={math_version}. "
            "Install matching 2026 packages or matching 2027.0.0a7 packages."
        )
    return expected


NT_VERSION = version("pyntcore")
MATH_VERSION = version("robotpy-wpimath")
API = _detect_api(
    NT_VERSION,
    MATH_VERSION,
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
    """Call the method name supported by the selected API.

    Args:
        obj: Object or class exposing the version-specific method.
        modern: Method name for 2027.
        legacy: Method name for 2026.
        *args: Positional arguments forwarded to the method.
        **kwargs: Keyword arguments forwarded to the method.

    Returns:
        The result of the selected method.
    """
    return getattr(obj, modern if API == "2027" else legacy)(*args, **kwargs)


def default_instance() -> Any:
    """Get the default NetworkTables instance.

    Returns:
        The default instance for the selected API.
    """
    return _call(ntcore.NetworkTableInstance, "get_default", "getDefault")


def create_instance() -> Any:
    """Create an independent NetworkTables instance.

    Returns:
        A newly created instance that the caller must destroy.
    """
    return ntcore.NetworkTableInstance.create()


def destroy_instance(instance: Any) -> None:
    """Destroy a previously created NetworkTables instance.

    Args:
        instance: Instance returned by create_instance.
    """
    ntcore.NetworkTableInstance.destroy(instance)


def start_client(instance: Any, identity: str, server: str, port: int = 5810) -> None:
    """Start an NT4 client and configure its server.

    Args:
        instance: NetworkTables instance to configure.
        identity: Client identity advertised to the server.
        server: Server hostname or address.
        port: NT4 TCP port; defaults to 5810.
    """
    _call(instance, "start_client", "startClient4", identity)
    _call(instance, "set_server", "setServer", server, port)


def start_local(instance: Any) -> None:
    """Start local operation for a NetworkTables instance.

    Args:
        instance: NetworkTables instance to start.
    """
    _call(instance, "start_local", "startLocal")


def stop_local(instance: Any) -> None:
    """Stop local operation for a NetworkTables instance.

    Args:
        instance: NetworkTables instance to stop.
    """
    _call(instance, "stop_local", "stopLocal")


def start_server(instance: Any, port: int) -> None:
    """Start an NT4 server bound to loopback (127.0.0.1).

    Args:
        instance: NetworkTables instance to configure.
        port: NT4 TCP port on which to listen.
    """
    if API == "2027":
        instance.start_server("", "127.0.0.1", "", port)
    else:
        instance.startServer("", "127.0.0.1", 0, port)


def stop_server(instance: Any) -> None:
    """Stop the NetworkTables server.

    Args:
        instance: NetworkTables instance running the server.
    """
    _call(instance, "stop_server", "stopServer")


def get_table(instance: Any, name: str) -> Any:
    """Get a named NetworkTables table.

    Args:
        instance: NetworkTables instance owning the table.
        name: Table path or name.

    Returns:
        The requested table.
    """
    return _call(instance, "get_table", "getTable", name)


def table_instance(table: Any) -> Any:
    """Get the NetworkTables instance owning a table.

    Args:
        table: NetworkTables table to inspect.

    Returns:
        The owning instance.
    """
    return _call(table, "get_instance", "getInstance")


def flush(instance: Any) -> None:
    """Flush pending NetworkTables updates.

    Args:
        instance: NetworkTables instance to flush.
    """
    instance.flush()


def is_connected(instance: Any) -> bool:
    """Check whether a NetworkTables instance is connected.

    Args:
        instance: NetworkTables instance to inspect.

    Returns:
        Whether the instance has an active connection.
    """
    return bool(_call(instance, "is_connected", "isConnected"))


def connection_count(instance: Any) -> int:
    """Count active NetworkTables connections.

    Args:
        instance: NetworkTables instance to inspect.

    Returns:
        The number of active connections.
    """
    return len(_call(instance, "get_connections", "getConnections"))


def topic_names(instance: Any) -> set[str]:
    """Get all topic names exposed by an instance.

    Args:
        instance: NetworkTables instance to inspect.

    Returns:
        A set of topic names.
    """
    return {
        str(_call(item, "get_name", "getName"))
        for item in _call(instance, "get_topics", "getTopics")
    }


def read_bool(table: Any, key: str, default: bool) -> bool:
    """Read a boolean table entry.

    Args:
        table: Table containing the entry.
        key: Entry name.
        default: Value returned when no boolean value is available.

    Returns:
        The entry value or the default.
    """
    return _call(
        _call(table, "get_entry", "getEntry", key), "get_boolean", "getBoolean", default
    )


def read_double(table: Any, key: str, default: float) -> float:
    """Read a double table entry.

    Args:
        table: Table containing the entry.
        key: Entry name.
        default: Value returned when no double value is available.

    Returns:
        The entry value or the default.
    """
    return _call(
        _call(table, "get_entry", "getEntry", key), "get_double", "getDouble", default
    )


def pubsub_options(
    *,
    keep_duplicates: bool = False,
    send_all: bool = False,
    periodic: float = 0.1,
    poll_storage: int = 0,
) -> Any:
    """Build publication and subscription options for the selected API.

    Args:
        keep_duplicates: Whether to preserve consecutive duplicate values.
        send_all: Whether to send every value change over the network instead of
            only the latest per interval. Queue retention is still bounded by
            poll_storage.
        periodic: Requested network transmission interval in seconds.
        poll_storage: Maximum samples retained between read_queue calls; 0 uses
            the NetworkTables default.

    Returns:
        Native publication and subscription options.
    """
    # The installed stubs describe one binding version; dispatch selects the
    # corresponding keyword names without requiring both stub sets to exist.
    kwargs: dict[str, Any] = {"periodic": periodic}
    if API == "2027":
        kwargs.update(
            keep_duplicates=keep_duplicates,
            send_all=send_all,
            poll_storage=poll_storage,
        )
    else:
        kwargs.update(
            keepDuplicates=keep_duplicates, sendAll=send_all, pollStorage=poll_storage
        )
    return ntcore.PubSubOptions(**kwargs)


def topic(table: Any, kind: str, key: str, struct_type: type | None = None) -> Any:
    """Get a typed NetworkTables topic.

    Args:
        table: Table containing the topic.
        kind: Topic type in snake case, such as "double" or "struct_array".
        key: Topic name.
        struct_type: Struct class for struct and struct-array topics.

    Returns:
        The requested typed topic.
    """
    legacy = "get" + "".join(part.title() for part in kind.split("_")) + "Topic"
    args = (key, struct_type) if kind in {"struct", "struct_array"} else (key,)
    return _call(table, f"get_{kind}_topic", legacy, *args)


def read_queue(subscriber: Any) -> list[Any]:
    """Read queued subscriber samples.

    Args:
        subscriber: Subscriber whose queue should be drained.

    Returns:
        Retained samples in oldest-first order, with timestamps in the selected
        API's native NT units. Retention is bounded by poll_storage; excess
        samples may be dropped, so not every published value is guaranteed.
    """
    return _call(subscriber, "read_queue", "readQueue")


def sample_timestamp(sample: Any) -> int:
    """Read a sample timestamp in native NT clock units.

    Args:
        sample: Timestamped NetworkTables sample.

    Returns:
        The native timestamp: microseconds in 2026 or nanoseconds in 2027.
        Divide by NT_TIME_UNITS_PER_US to normalize to microseconds.
    """
    return sample.time


def raw_nt_now() -> int:
    """Read the current NT clock in native units.

    Returns:
        The native clock value: microseconds in 2026 or nanoseconds in 2027.
        Divide by NT_TIME_UNITS_PER_US to normalize to microseconds.
    """
    return ntcore._now()


def to_nt_publish_time(capture_nt_us: int) -> int:
    """Convert an NT-clock capture timestamp to native publication units.

    Args:
        capture_nt_us: Capture timestamp already expressed in NT-clock
            microseconds, not a raw monotonic-clock reading.

    Returns:
        The unchanged microsecond timestamp for 2026, or the timestamp converted
        to nanoseconds by multiplying by 1000 for 2027.
    """
    return capture_nt_us * NT_TIME_UNITS_PER_US


def pose_xyz(pose: Any) -> tuple[float, float, float]:
    """Read x, y, and z components through the selected geometry API.

    Args:
        pose: Pose or translation, or a rotation passed by rotation_xyz.

    Returns:
        The (x, y, z) components in meters for a pose or translation, or the
        (roll, pitch, yaw) angles in radians when called through rotation_xyz.
    """
    return (pose.x, pose.y, pose.z) if API == "2027" else (pose.X(), pose.Y(), pose.Z())


def rotation_xyz(rotation: Any) -> tuple[float, float, float]:
    """Read the roll, pitch, and yaw of a rotation.

    Args:
        rotation: Three-dimensional rotation to inspect.

    Returns:
        The (roll, pitch, yaw) angles in radians.
    """
    return pose_xyz(rotation)


def quaternion_wxyz(rotation: Any) -> tuple[float, float, float, float]:
    """Read a rotation as a quaternion.

    Args:
        rotation: Three-dimensional rotation to inspect.

    Returns:
        The unitless quaternion components in (w, x, y, z) order.
    """
    q = _call(rotation, "get_quaternion", "getQuaternion")
    return (q.w, q.x, q.y, q.z) if API == "2027" else (q.W(), q.X(), q.Y(), q.Z())
