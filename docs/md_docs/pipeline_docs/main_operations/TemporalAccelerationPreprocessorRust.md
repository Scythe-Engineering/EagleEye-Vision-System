# Temporal acceleration preprocessor

`temporal_acceleration_preprocessor_rust` predicts AprilTag search regions from a camera pose and a known tag map. It passes cropped regions to `detect_apriltags`, which can avoid a full-frame search when a predicted region contains a tag.

## Inputs

- `frame`: required NumPy camera image.
- `camera_pose`: optional 4 by 4 camera pose. The operation rejects other non-null shapes.

## Outputs

- `processed_frame`: `(regions, original_frame)`.

Each region is `(crop, mapping)`. A rectified crop has a 3 by 3 transform from crop coordinates to full-frame coordinates. The fallback crop format uses an `[x, y]` offset.

## When to use

Use this operation immediately before `detect_apriltags` when the pipeline already has a camera pose that can guide the next search. A direct full-frame connection is simpler when no pose estimate is available.

## Configuration

| Setting | Default | Notes |
| --- | --- | --- |
| `camera_bus_id` | required | Camera whose registered intrinsics are loaded. Requires restart. |
| `apriltag_map_path` | required | AprilTag fmap path. Requires restart. |
| `padding_factor` | `0.35` | Fractional padding around projected tag bounds, 0.0 to 2.0. |
| `max_regions` | `20` | Maximum returned regions, 1 to 256. |
| `min_region_size_px` | `16` | Reject regions with smaller sides, 4 to 2048 pixels. |
| `max_detection_distance_m` | `0.0` | Skip tags farther than this distance. `0` disables the limit. |

The last four settings support live updates.

```json
{
  "camera_bus_id": "0-1",
  "apriltag_map_path": "{project_root}/files/apriltag_map_path/frc2025r2.json",
  "padding_factor": 0.35,
  "max_regions": 20
}
```

## Important behavior and limitations

The compiled `temporal_acceleration` extension must be installed. Predictions depend on camera calibration, map geometry, and the supplied pose. If region detection finds no tag, `detect_apriltags` falls back to one full-frame search. This protects detection coverage but removes the intended reduction in searched image area for that frame.
