# PnP Camera Localization Operation

## Overview

`PnpCameraLocalizationDefinition` estimates a field-relative camera pose from AprilTag detections, calibrated camera intrinsics, and an FRC field map. It delegates the solve to `PnpLocalization`.

## Inputs and output

- Input: `list[pupil_apriltags.Detection]` from `detect_apriltags`.
- Output: a 4×4 homogeneous `numpy.ndarray` representing `T_field_from_camera`, or `None` when pose estimation fails.
- Translation is in meters. The rotation is a 3×3 matrix.

For each mapped detection, the solver uses the tag's four field-space corners and detected image-space corners, then calls OpenCV `solvePnP(..., SOLVEPNP_ITERATIVE)`. It inverts the resulting camera-space transform before returning the field-from-camera transform.

The fmap parser converts tag sizes from millimeters to meters. The field map and camera calibration must use the expected field coordinate convention; validate the final published pose on the robot.

## Configuration

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `camera_bus_id` | `str` | required | Resolves the camera configuration and its intrinsics file. |
| `apriltag_map_path` | `str` | required | Path to the AprilTag field-map JSON/fmap file. |
| `jump_threshold` | `float` | `2.0` | Position jump threshold in meters used when retrying a solve without the cached guess. |

`camera_config_registry` is injected by the pipeline runtime. The operation loads `intrinsics_path` from the configuration for `camera_bus_id`; `camera_parameters_path` is not an action parameter.

```json
{
  "action_name": "pnp_camera_localization.py",
  "action_params": {
    "camera_bus_id": "basic_test",
    "apriltag_map_path": "{project_root}/files/apriltag_map_path/frc2025r2.json",
    "jump_threshold": 2.0
  }
}
```

## Timing and robot pose chain

This operation returns a raw matrix, but the pipeline runtime propagates the capture metadata attached by `DeviceInput`. Consequently the camera pose retains the source frame's NetworkTables capture timestamp. Use `camera_to_robot_pose` to convert it to a robot pose, then `publish_to_networktables` to publish a timestamped WPILib pose.

## Limitations

- No reprojection error, tag IDs/count, ambiguity, covariance, or solve-status value is emitted with the matrix.
- The solver has a cached pose guess and jump retry, but it is not a full pose filter.
- Timestamp propagation labels the source frame; it does not make the PnP result a hardware-exposure-time measurement.

## Files

- Definition: `src/main_operations/definitions/pnp_camera_localization.py`
- Solver: `src/main_operations/modules/apriltags/pnp_localization.py`
- Field-map parser: `src/main_operations/modules/apriltags/utils/fmap_parser.py`
