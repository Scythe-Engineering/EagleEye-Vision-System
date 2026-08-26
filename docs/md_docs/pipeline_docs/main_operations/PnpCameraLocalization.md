# PnP camera localization

`pnp_camera_localization` estimates the camera's field-relative pose from AprilTag detections, camera calibration, and a field map.

## Inputs

- `detections`: AprilTag detections containing `tag_id` and four image-space `corners`, normally from `detect_apriltags`.
- `robot_yaw` (optional): the robot heading in radians, WPILib convention, normally from a
  `get_networktables_value` operation reading a key the robot publishes (the robot library's
  `EagleEyeCamera.publishRobotYaw` uses `robot/yaw`). Leave unconnected for the standard solve.

## Outputs

- `camera_pose`: a 4 by 4 NumPy transform, `T_field_from_camera`.
- `pose_meta`: `[tag_count, mean_tag_distance_m, reprojection_error_px]`, the quality metrics the
  robot-side pose estimator derives its standard deviations from.
- Both ports carry `None` when no mapped points are available or OpenCV cannot solve the pose, so
  downstream operations keep their existing `None` handling.

Translation is in meters. The upper-left 3 by 3 block is the camera rotation in field coordinates.

## When to use

Use this operation after AprilTag detection when the pipeline needs a field-relative camera pose. Add `camera_to_robot_pose` afterward when consumers need the robot pose instead.

Publish `pose_meta` alongside the pose whenever robot code consumes it. Both ports are stamped
with the same frame capture time, which is how the robot library pairs them; see
[`library/README.md`](../../../../library/README.md). Only single-input operations may sit between
either port and its publisher; a multi-input operation averages its inputs' capture times, which
leaves the two branches carrying different timestamps.

## Configuration

| Setting | Default | Notes |
| --- | --- | --- |
| `camera_bus_id` | required | Camera whose registered intrinsics file is loaded. Requires restart. |
| `apriltag_map_path` | `{project_root}/config/apriltag_map.fmap` | AprilTag field-map path. Requires restart. |

```json
{
  "camera_bus_id": "0-1",
  "apriltag_map_path": "{project_root}/files/apriltag_map_path/frc2025r2.json"
}
```

## Yaw-constrained mode

When `robot_yaw` carries a finite value, the solver treats the camera orientation as known — the
gyro heading plus the camera mounting extrinsics, assuming the robot sits flat on the field — and
solves only for camera position. With rotation fixed, each corner gives two equations linear in
position, so the solve is one exact least squares: no iteration, and none of the pose ambiguity
that makes single-tag and long-range solves noisy. This is the same idea as Limelight's MegaTag2,
and like it, the gyro is trusted over the tags: a wrong yaw in produces a wrong pose out, so seed
the gyro before relying on it.

The yaw is read at solve time, not at frame exposure, so a robot spinning quickly constrains a
frame with a slightly newer heading. Publish the yaw every robot loop and gate vision on angular
velocity if you range at distance while rotating.

If the constrained system is degenerate the operation falls back to the unconstrained solver for
that frame. When `robot_yaw` is connected but nothing publishes the key,
`get_networktables_value` outputs None and the solve is simply unconstrained.

## Important behavior and limitations

The solver uses only detections whose IDs exist in the map and solves every frame independently with OpenCV's iterative PnP solver. Map tag dimensions are converted from millimeters to meters.

The solver reports contributing-tag count, mean tag distance, and mean reprojection error, but not ambiguity or a full covariance. Reprojection error measures how well the solution explains the corners it was solved from; it does not catch a pose that is confidently wrong because of a bad map or bad calibration. Computing the metrics costs one `projectPoints` call over at most a few dozen points.

The pipeline retains capture timing metadata from the source frame, but that does not turn the result into a hardware exposure-time measurement. Camera calibration and the map must use the expected coordinate conventions.
