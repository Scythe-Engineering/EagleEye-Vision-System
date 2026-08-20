# PnP camera localization

`pnp_camera_localization` estimates the camera's field-relative pose from AprilTag detections, camera calibration, and a field map.

## Inputs

- `detections`: AprilTag detections containing `tag_id` and four image-space `corners`, normally from `detect_apriltags`.

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
[`library/README.md`](../../../../library/README.md). Do not place an operation between
`pose_meta` and its publisher, since that would restamp the value.

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

## Important behavior and limitations

The solver uses only detections whose IDs exist in the map and solves every frame independently with OpenCV's iterative PnP solver. Map tag dimensions are converted from millimeters to meters.

The solver reports contributing-tag count, mean tag distance, and mean reprojection error, but not ambiguity or a full covariance. Reprojection error measures how well the solution explains the corners it was solved from; it does not catch a pose that is confidently wrong because of a bad map or bad calibration. Computing the metrics costs one `projectPoints` call over at most a few dozen points.

The pipeline retains capture timing metadata from the source frame, but that does not turn the result into a hardware exposure-time measurement. Camera calibration and the map must use the expected coordinate conventions.
