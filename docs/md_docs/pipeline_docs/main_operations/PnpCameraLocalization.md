# PnP camera localization

`pnp_camera_localization` estimates the camera's field-relative pose from AprilTag detections, camera calibration, and a field map.

## Inputs

- `detections`: AprilTag detections containing `tag_id` and four image-space `corners`, normally from `detect_apriltags`.

## Outputs

- `camera_pose`: a 4 by 4 NumPy transform, `T_field_from_camera`.
- `pose_meta`: `[tag_count, mean_tag_distance_m, reprojection_error_px]`, the quality metrics the
  robot-side pose estimator derives its standard deviations from.
- Both ports carry `None` when no mapped points are available, OpenCV cannot solve the pose, or a recent pose exposes an implausible single-tag jump. No old pose is substituted.

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
| `refinement_iterations` | `10` | Maximum LM refinement iterations. Zero disables refinement. |
| `use_pose_continuity` | `true` | Resolve single-tag ambiguity and reject large jumps using a recent capture-timed pose. Disable for independent per-frame pose selection. |

```json
{
  "camera_bus_id": "0-1",
  "apriltag_map_path": "{project_root}/files/apriltag_map_path/frc2025r2.json"
}
```

## Important behavior and limitations

The solver uses only detections whose IDs exist in the map. It initializes single-tag poses with IPPE and multi-tag poses with SQPnP, also checking IPPE candidates for coplanar multi-tag layouts, then refines with LM. Map tag dimensions are converted from millimeters to meters.

With continuity enabled, a prior captured within 250 ms can break single-tag ties within 0.25 pixels RMS reprojection error. Candidate selection requires translation within 1 m and rotation within 0.35 radians of that prior, both before and after refinement. Otherwise the solver tries ordinary image-only selection. A resulting single-tag jump over 2 m or 0.75 radians returns no pose. Rejected frames do not renew the prior's age, so recovery cannot remain locked out indefinitely.

Missing, repeated, or backward capture timestamps clear history. Multi-tag solves remain image-only and refresh the anchor. Every published pose comes from the current frame; there is no pose blending, smoothing, or repeated previous-pose output. This is a tracking heuristic, not proof that a pose is correct, and can reject real abrupt camera motion. Disable it for applications where those motion bounds are inappropriate.

The solver reports contributing-tag count, mean tag distance, and mean reprojection error, but not ambiguity or a full covariance. Reprojection error measures how well the solution explains the corners it was solved from; it does not catch a pose that is confidently wrong because of a bad map or bad calibration. Computing the metrics costs one `projectPoints` call over at most a few dozen points.

The pipeline retains capture timing metadata from the source frame, but that does not turn the result into a hardware exposure-time measurement. Camera calibration and the map must use the expected coordinate conventions.
