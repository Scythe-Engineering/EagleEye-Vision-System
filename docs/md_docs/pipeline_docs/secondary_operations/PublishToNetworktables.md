# PublishToNetworktables Operation

## Overview

`PublishToNetworktables` is a pass-through secondary operation that publishes a pipeline value to a NetworkTables topic. It uses native NetworkTables primitive topics and WPILib struct/struct-array topics; it does **not** use Flatpack, raw bytes, `putRaw()`, or an `FPK1` envelope.

The backend creates the `EagleEye` table in `MainBackend`, so a relative `target_key` such as `robot_pose` is published at `/EagleEye/robot_pose`.

## Configuration

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `target_key` | `str` | required | Topic key relative to the injected `EagleEye` table. |
| `schema` | `str` | `"auto"` | Requested output type; see supported schemas below. |
| `data_path` | `str`, sequence, or `null` | `null` | Optional field/index path to select before publishing. Dot-separated strings are supported. |

`data_path` also supports extracting one field from every dictionary in a sequence. For example, `data_path: ["angle_degrees"]` converts a list of detection dictionaries into a double array.

## Supported values and topics

The operation selects a native NT publisher from the value after coercion:

| Input / schema | Published type |
| --- | --- |
| number with `double`, `float`, `number`, or `auto` | double |
| bool with `boolean`, `bool`, or `auto` | boolean |
| string with `string` or `auto` | string |
| numeric sequence or numeric ndarray with `double_array`, `float_array`, `number_array`, or `auto` | double array |
| bool sequence | boolean array |
| string sequence | string array |
| Geometry dictionary or compatible sequence of geometry dictionaries | struct or struct array |

Geometry schemas are `pose2d`, `pose3d`, `translation2d`, `translation3d`, `rotation2d`, `rotation3d`, `transform2d`, and `transform3d`. Dictionaries use the expected coordinate keys, for example `{ "x", "y", "rotation" }` for `Pose2d` and `{ "x", "y", "z", "roll", "pitch", "yaw" }` for `Pose3d`.

A 4×4 NumPy transform publishes as `Pose3d` unless `schema` is explicitly `pose2d`. Unsupported values or empty arrays are not published.

## Capture timestamps

Pipeline values originating at `DeviceInput` carry `TimingMetadata`. When the selected value is timed, this operation calls:

```python
publisher.set(wpi_value, timing.capture_nt_us)
```

`capture_nt_us` is an integer NetworkTables timestamp in microseconds. A robot subscriber can read the pose and its capture time atomically with `getAtomic()` and convert the timestamp to WPILib seconds:

```java
var sample = poseSubscriber.getAtomic();
if (sample.timestamp != 0) {
  poseEstimator.addVisionMeasurement(sample.value, sample.timestamp / 1_000_000.0);
}
```

No separate latency topic or latency subtraction is required for this path. Values with no timing metadata are published using the normal publish-time timestamp.

On Linux, a driver-provided monotonic V4L2 buffer timestamp excludes USB transfer, JPEG decode, and scheduler delay. Drivers without that timestamp flag, and other platforms, fall back to stamping when the frame is delivered. A reused cached frame retains its original timestamp, so robot code should reject measurements that are too old or duplicate according to its application policy.

See [Time Synchronization](../../../overviews/TIME_SYNCHRONIZATION.md) for the full capture-to-robot path and the WPILib pose estimator integration.

## Examples

### Publish a robot pose

```json
{
  "action_name": "publish_to_networktables.py",
  "action_params": {
    "target_key": "robot_pose",
    "schema": "pose3d"
  }
}
```

With the default backend table this publishes a `Pose3d` at `/EagleEye/robot_pose`, preserving the input frame's capture timestamp.

### Publish one numeric field from detections

```json
{
  "action_name": "publish_to_networktables.py",
  "action_params": {
    "target_key": "vision/target_angles",
    "schema": "double_array",
    "data_path": ["angle_degrees"]
  }
}
```

## Pipeline behavior

The operation returns its original input unchanged, allowing it to be placed as a terminal side-effect or in a longer pipeline chain.

## Files

- Implementation: `src/secondary_operations/publish_to_networktables.py`
- Timing model: `src/utils/timing.py`
- Example pipeline: `src/config/pipeline_config.json`
