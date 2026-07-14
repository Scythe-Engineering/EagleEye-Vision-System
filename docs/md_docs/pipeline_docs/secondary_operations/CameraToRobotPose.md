# CameraToRobotPose Operation

## Overview

`CameraToRobotPose` converts a field-relative camera pose into a field-relative robot pose using the configured camera extrinsics. Place it after `pnp_camera_localization` before a robot pose output or NetworkTables publisher.

```text
DeviceInput → DetectApriltags → PnpCameraLocalization
            → CameraToRobotPose → PublishToNetworktables
```

## Transform

The operation constructs `T_robot_from_camera` from the camera configuration, inverts it, and computes:

```text
T_field_from_robot = T_field_from_camera @ T_camera_from_robot
```

Camera extrinsics are loaded from the injected `CameraConfigRegistry` using `camera_bus_id`. If no registry is available, the extrinsic transform is identity. The inverse is cached and rebuilt if the bus ID or an extrinsic field is updated.

## Configuration

| Parameter | Type | Description |
| --- | --- | --- |
| `camera_bus_id` | `str` | Camera configuration whose extrinsics are used. |

```json
{
  "action_name": "camera_to_robot_pose.py",
  "action_params": { "camera_bus_id": "basic_test" }
}
```

## Timing

The operation itself receives a raw 4×4 matrix and returns a raw matrix. The pipeline runner reattaches the input `TimedValue` metadata, so a pose originating from a camera retains its frame capture time through this conversion and into `PublishToNetworktables`.

## Input and output

- Input: finite 4×4 NumPy camera-pose transform, or `None`.
- Output: finite 4×4 NumPy robot-pose transform, or `None` for missing/invalid input.

## Files

- Implementation: `src/secondary_operations/camera_to_robot_pose.py`
- Configuration definition: `src/secondary_operations/config_data/camera_to_robot_pose_config_def.json`
