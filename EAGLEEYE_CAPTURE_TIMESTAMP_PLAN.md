# EagleEye Capture-Time Timestamp and NetworkTables Struct Plan

## Verified WPILib / NetworkTables facts

- NetworkTables value timestamps use integer microseconds.
- When the roboRIO is the NT server, NT server timestamps align with the roboRIO FPGA timestamp timebase used by `Timer.getFPGATimestamp()`, except NT uses microseconds and WPILib `Timer` uses seconds.
- NT4 automatically synchronizes time between clients and the server.
- Publishers can publish a value with an explicit timestamp:
  - Java: `publisher.set(value, timestampMicros)`
  - Python / RobotPy: `publisher.set(value, timestampMicros)`
- RobotPy provides `ntcore._now()`, which returns the current NT timestamp in microseconds on the same timebase used for NT value and connection timestamps.
- Subscribers can read value + timestamp atomically with `getAtomic()`.
- WPILib supports NT struct topics for many built-in types, including `Pose2d`, `Pose3d`, `Translation2d`, `Translation3d`, `Rotation2d`, `Rotation3d`, `Transform2d`, `Transform3d`, `Twist2d`, `Twist3d`, and `Quaternion`.

## Goal

EagleEye should publish arbitrary frontend-configured pipeline outputs to NetworkTables with timestamps representing **when the source frame was captured**, not when the result was published.

Robot code should read these values through typed EagleEye accessors and use the NT atomic timestamp directly for latency-compensated consumers such as WPILib pose estimators.

No separate latency key should be required for normal operation.

## High-level architecture

1. Camera capture creates a timestamped frame packet.
2. Pipeline operations propagate timing metadata with their outputs.
3. NetworkTables publishing uses the propagated capture timestamp as the explicit NT publish timestamp.
4. Robot-side EagleEye accessors read values with `getAtomic()`.
5. Robot consumers pass `atomic.timestamp / 1_000_000.0` to WPILib APIs such as `poseEstimator.addVisionMeasurement(...)`.

## EagleEye data model

Add a shared timing module, for example:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")

@dataclass(frozen=True)
class TimingMetadata:
    capture_nt_us: int
    capture_monotonic_ns: int
    frame_seq: int | None = None
    camera_name: str | None = None
    bus_id: str | None = None
    source: str | None = None
    derived_from: tuple["TimingMetadata", ...] = field(default_factory=tuple)

@dataclass(frozen=True)
class TimedValue(Generic[T]):
    value: T
    timing: TimingMetadata
```

For camera frames, use:

```python
FramePacket = TimedValue[np.ndarray]
```

Helper functions should be added:

```python
def is_timed(value: object) -> bool: ...
def unwrap_timed(value): ...
def get_timing(value) -> TimingMetadata | None: ...
def retime(value, timing: TimingMetadata): ...
def average_timings(timings: list[TimingMetadata]) -> TimingMetadata: ...
```

## Capture timestamp source

Use `ntcore._now()` for `capture_nt_us`.

This is the verified RobotPy function for NT-compatible microsecond timestamps.

Use `time.monotonic_ns()` only for local profiling/debug timing, not for robot estimator timestamps.

Example:

```python
import ntcore
import time

capture_nt_us = ntcore._now()
capture_monotonic_ns = time.monotonic_ns()
```

## CameraThreadManager changes

Current code in `src/utils/camera_utils/camera_thread_manager.py` stores only:

```python
_current_frame
_current_timestamp
```

and computes:

```python
timestamp_from_start = current_time_ms - self.start_time_ms
```

Replace this with full frame packets.

### CameraWorker state

Change from:

```python
self._current_frame: Optional[np.ndarray] = None
self._current_timestamp: float = 0.0
```

to:

```python
self._current_packet: FramePacket | None = None
self._frame_seq: int = 0
```

Add methods:

```python
def set_current_packet(self, packet: FramePacket) -> None: ...
def get_current_packet(self) -> FramePacket | None: ...
def next_frame_seq(self) -> int: ...
```

Keep compatibility methods temporarily if needed:

```python
def get_current_frame(self) -> tuple[np.ndarray, float] | None:
    packet = self.get_current_packet()
    if packet is None:
        return None
    return packet.value.copy(), packet.timing.capture_nt_us / 1000.0
```

### Capture worker timestamping

In `camera_feed_worker`, timestamp immediately after `camera.get_frame()` returns:

```python
frame = camera.get_frame()

if frame is not None:
    capture_nt_us = ntcore._now()
    capture_monotonic_ns = time.monotonic_ns()
    frame_seq = worker.next_frame_seq()

    packet = TimedValue(
        frame,
        TimingMetadata(
            capture_nt_us=capture_nt_us,
            capture_monotonic_ns=capture_monotonic_ns,
            frame_seq=frame_seq,
            camera_name=camera_name,
            bus_id=self.get_bus_id_for_camera_name(camera_name),
            source="camera",
        ),
    )

    worker.set_current_packet(packet)
```

If the camera backend later exposes true hardware frame timestamps, prefer those and convert them to NT time if possible. Until then, timestamping immediately after `get_frame()` is the best practical software capture time.

### Cached-frame behavior

If `camera.get_frame()` fails and the cached frame is reused, do **not** create a new capture timestamp. Reuse the cached frame's original `TimingMetadata` so downstream consumers know the data is stale.

## DeviceInput changes

Current `DeviceInput.run()` returns a raw `np.ndarray`.

Change it to return `FramePacket`:

```python
def run(self, _input_data: Any) -> FramePacket | None:
    packet = self.camera_manager.get_current_packet_by_bus_id(self.bus_id)
    if packet is None:
        return None

    frame = self._apply_rotation(packet.value)
    return TimedValue(frame, packet.timing)
```

Rotation does not change capture time.

## Pipeline timing propagation

The pipeline runner should become timing-aware so individual operations do not all need large custom changes.

Rules:

1. If an operation returns `TimedValue`, preserve it.
2. If an operation returns raw output and has one timed input, wrap the output with the input timing.
3. If an operation returns raw output and has multiple timed current-frame inputs, average their capture timestamps.
4. If an operation uses previous-frame/history data, that historical timing must not contribute to the current output timing.
5. If there is no timing metadata, return raw output unchanged.

Pseudo:

```python
def attach_output_timing(output: Any, inputs: Any) -> Any:
    if isinstance(output, TimedValue):
        return output

    timings = collect_current_frame_timings(inputs)
    if not timings:
        return output

    timing = timings[0] if len(timings) == 1 else average_timings(timings)
    return TimedValue(output, timing)
```

### Averaging multiple inputs

For multiple simultaneous sources:

```python
avg_capture_nt_us = round(sum(t.capture_nt_us for t in timings) / len(timings))
avg_capture_monotonic_ns = round(sum(t.capture_monotonic_ns for t in timings) / len(timings))
```

The averaged timing should record `derived_from=tuple(timings)` for debugging.

## Operation migration strategy

Add unwrap helpers and update operations incrementally.

Most operations should do:

```python
input_value = unwrap_timed(input_data)
```

and then return a raw result. The pipeline runner will reattach timing automatically.

Operations that intentionally combine or generate timing-sensitive data can return `TimedValue` explicitly.

### Minimal compatibility helper

```python
def unwrap_timed(value):
    return value.value if isinstance(value, TimedValue) else value
```

Use this at operation boundaries where `np.ndarray`, dicts, or pose matrices are expected.

## NetworkTables publishing changes

Update `src/secondary_operations/publish_to_networktables.py`.

Current behavior:

```python
self._publisher.set(wpi_value)
```

New behavior:

```python
raw_value = unwrap_timed(value)
timing = get_timing(value)
wpi_value = _coerce_wpilib(raw_value, self.schema)

if timing is not None:
    self._publisher.set(wpi_value, timing.capture_nt_us)
else:
    self._publisher.set(wpi_value)
```

This makes `getAtomic().timestamp` on the robot equal to the source frame capture timestamp.

## NetworkTables struct/type support

Keep and expand the current struct publishing approach.

Supported output schemas should include:

- `pose2d`
- `pose3d`
- `translation2d`
- `translation3d`
- `rotation2d`
- `rotation3d`
- `transform2d`
- `transform3d`
- arrays of supported structs
- primitive doubles, booleans, strings, and arrays where appropriate

Avoid comma-separated string serialization for robot-facing geometric data.

## Robot-side EagleEye accessor design

In `E:/Vs Code Projects/Code-2026/src/main/java/frc/robot/subsystems/eagleEye`, replace or supplement the current fixed gamepiece-only connector with typed accessors.

### Key resolution

Support both full and relative keys:

```java
private static String resolveKey(String key) {
  if (key.startsWith("/")) {
    return key;
  }
  return "/EagleEye/" + key;
}
```

### Timestamped value record

```java
public record TimestampedEagleEyeValue<T>(
    T value,
    double timestampSeconds,
    long timestampMicros,
    boolean valid
) {}
```

### Accessors

Examples:

```java
Optional<TimestampedEagleEyeValue<Pose2d>> getPose2d(String key);
Optional<TimestampedEagleEyeValue<Pose3d>> getPose3d(String key);
Optional<TimestampedEagleEyeValue<Pose2d[]>> getPose2dArray(String key);
Optional<TimestampedEagleEyeValue<double[]>> getDoubleArray(String key);
Optional<TimestampedEagleEyeValue<Double>> getDouble(String key);
Optional<TimestampedEagleEyeValue<String[]>> getStringArray(String key);
```

Each accessor should cache its subscriber so callers can request by key without recreating subscribers every loop.

For `Pose2d`:

```java
StructSubscriber<Pose2d> sub =
    NetworkTableInstance.getDefault()
        .getStructTopic(resolveKey(key), Pose2d.struct)
        .subscribe(new Pose2d());

TimestampedObject<Pose2d> atomic = sub.getAtomic();

return Optional.of(new TimestampedEagleEyeValue<>(
    atomic.value,
    atomic.timestamp / 1_000_000.0,
    atomic.timestamp,
    atomic.timestamp != 0
));
```

## Drive.java usage

`Drive.java` should explicitly ask for the EagleEye key it wants:

```java
eagleEye.getPose2d("RobotPose2D").ifPresent(estimate -> {
  if (estimate.valid()) {
    poseEstimator.addVisionMeasurement(
        estimate.value(),
        estimate.timestampSeconds());
  }
});
```

No latency subtraction is needed because the NT timestamp is already the capture timestamp.

## Debugging and validation

Even though latency keys should not be used as part of the robot estimator path, EagleEye should expose debug information to WebUI/logs:

- capture-to-publish latency in ms
- operation processing duration
- frame age at publish time
- frame sequence number
- camera name / bus id
- publish key

These can be published under debug-only paths or kept in WebUI profiling.

Example debug calculation:

```python
publish_nt_us = ntcore._now()
frame_age_ms = (publish_nt_us - timing.capture_nt_us) / 1000.0
```

## Testing plan

1. Unit test `TimedValue` helpers.
2. Unit test `average_timings()`.
3. Unit test `DeviceInput` returns a `FramePacket` with nonzero `capture_nt_us`.
4. Unit test `publish_to_networktables` calls `publisher.set(value, timestamp)` when timing exists.
5. Integration test with a local NT server/client:
   - publish a `Pose2d` with explicit timestamp
   - subscribe and verify `getAtomic().timestamp` matches the injected timestamp
6. Robot-side test:
   - publish `/EagleEye/RobotPose2D`
   - read with `getPose2d("RobotPose2D")`
   - verify timestamp seconds are plausible and nonzero
7. On-robot validation:
   - compare EagleEye timestamp to `Timer.getFPGATimestamp()`
   - log frame age: `Timer.getFPGATimestamp() - estimate.timestampSeconds()`
   - confirm pose estimator accepts measurements without timestamp warnings or stale-measurement behavior

## Implementation order

1. Add `TimingMetadata` / `TimedValue` utility module.
2. Update `CameraThreadManager` to produce `FramePacket` with `ntcore._now()` capture timestamps.
3. Update `DeviceInput` to return `FramePacket`.
4. Add unwrap/timing propagation helpers to pipeline execution.
5. Update common operations to unwrap timed inputs where needed.
6. Update `publish_to_networktables.py` to publish explicit timestamps.
7. Expand publisher type support for WPILib structs and primitive types.
8. Add robot-side typed EagleEye accessors with key resolution and cached subscribers.
9. Update `Drive.java` or other call sites to use explicit EagleEye accessors.
10. Add tests and debug logging.

## Open implementation details

- Determine exact files in pipeline execution where output timing should be attached globally.
- Decide whether `TimedValue` should be transparent in WebUI JSON/profiling paths or explicitly unwrapped before WebUI publishing.
- Decide whether gamepiece detections should become `Pose2d[]` / `Translation2d[]` struct arrays instead of the current string-array format.
