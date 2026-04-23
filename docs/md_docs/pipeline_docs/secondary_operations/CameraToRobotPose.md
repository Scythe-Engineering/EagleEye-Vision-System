# CameraToRobotPose Operation Overview

## Overview

The `CameraToRobotPose` operation converts a camera pose into a robot pose by
applying the configured camera extrinsics. This is the correct secondary
operation to place between `pnp_camera_localization` and `robot_pose_output`
when the frontend should visualize the robot body rather than the camera frame.

## Why This Exists

`pnp_camera_localization` estimates the pose of the camera in field space. The
frontend robot visualization expects the pose of the robot. If you publish raw
camera pose into the robot visualization path, the model will inherit the
camera's physical tilt and offset, which is why the 3D view can appear pitched
down or displaced.

## Transformation Performed

The operation computes:

```text
T_world_from_robot = T_world_from_camera @ T_camera_from_robot
```

Where:

- `T_world_from_camera` is the input pose from PnP
- `T_robot_from_camera` is built from live camera extrinsics
- `T_camera_from_robot` is the inverse of that extrinsics transform

## Configuration

### Parameters

- `camera_bus_id`: Camera bus ID used to load extrinsics from the camera config
  registry.

### Dependencies

- `camera_config_registry`: Injected automatically by the pipeline runtime when
  available.

## Pipeline Placement

For frontend robot visualization, the intended chain is:

```text
detect_apriltags
  -> pnp_camera_localization
  -> camera_to_robot_pose
  -> robot_pose_output
```

If you want a flattened 2D robot pose after that, place `flatten_pose` after
`camera_to_robot_pose`, not before it.

## Example

```json
{
  "action_name": "camera_to_robot_pose.py",
  "action_params": {
    "camera_bus_id": "front_cam"
  }
}
```

## Files

- Implementation: `src/secondary_operations/camera_to_robot_pose.py`
- Config definition: `src/secondary_operations/config_data/camera_to_robot_pose_config_def.json`
