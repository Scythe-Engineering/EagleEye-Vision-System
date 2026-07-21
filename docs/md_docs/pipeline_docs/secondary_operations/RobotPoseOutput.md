# RobotPoseOutput Operation

## Overview

`RobotPoseOutput` forwards a robot 4×4 pose matrix to `EagleEyeInterface.update_robot_position()` for WebUI output. It is not a NetworkTables publisher; add `PublishToNetworktables` after it when the robot must consume the pose.

When the upstream source is `pnp_camera_localization`, insert `camera_to_robot_pose` first so this operation receives the robot pose rather than the camera pose.

## Behavior

The operation stores the last pose sent. If `np.array_equal()` says the incoming pose is identical, it returns `None` and does not call the WebUI. Otherwise it:

1. calls `web_interface.update_robot_position(pose)`;
2. stores `pose.copy()` as the last pose;
3. returns the pose unchanged.

It expects a NumPy pose matrix and does not validate matrix size, finiteness, coordinate frame, or uncertainty.

## Timing

The pipeline runner reattaches upstream capture metadata to raw outputs. Thus, when this operation is reached by a timed camera pipeline, the returned changed pose still carries the originating frame's capture timestamp for a downstream `PublishToNetworktables` operation. A duplicate pose returns `None`, so it does not produce a downstream measurement.

## Configuration

`web_interface` is injected by the runtime; this operation has no action parameters.

```json
{
  "action_name": "robot_pose_output.py",
  "action_params": {}
}
```

## Files

- Implementation: `src/secondary_operations/robot_pose_output.py`
- Example pipeline: `src/config/pipeline_config.json`
