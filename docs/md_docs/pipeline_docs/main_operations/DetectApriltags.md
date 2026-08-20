# Detect AprilTags

`detect_apriltags` locates AprilTags and reports their IDs and image corners. It accepts either a complete camera frame or regions produced by the temporal acceleration preprocessor.

## Inputs

- `frame`: a BGR or grayscale NumPy image, or `(regions, full_frame)` from `temporal_acceleration_preprocessor_rust`.

Each region is an image paired with either an `[x, y]` offset or a 3 by 3 transform into the full frame.

## Outputs

- `detections`: a list of `pupil_apriltags.Detection` objects for full-frame searches, or compatible detections containing `tag_id` and `corners` for region searches.
- Returns `None` when no tag is found.

## When to use

Use this operation before PnP camera localization or whenever a pipeline needs AprilTag IDs and pixel coordinates.

## Configuration

| Setting | Default | Notes |
| --- | --- | --- |
| `families` | `tag36h11` | Tag family to decode. The editor lists the families supported by `pupil-apriltags`. |
| `nthreads` | `1` | Detector threads, 1 to 16. |
| `quad_decimate` | `2.0` | Quad-search downsampling factor, 1.0 to 10.0. Larger values trade corner precision and detection range for less image work. |
| `quad_sigma` | `0.0` | Blur applied during quad search, 0.0 to 5.0. |
| `refine_edges` | `1` | Use `1` to refine detected edges, or `0` to disable it. |
| `decode_sharpening` | `0.25` | Decode sharpening, 0.0 to 1.0. |

All settings support live updates.

```json
{
  "families": "tag36h11",
  "nthreads": 2,
  "quad_decimate": 2.0,
  "refine_edges": 1
}
```

## Important behavior and limitations

Region mode searches the full frame once when no region contains a tag. Detected region corners are mapped back to full-frame coordinates. The operation does not estimate pose; connect its output to `pnp_camera_localization` for that step. Small, blurred, occluded, or low-contrast tags may not decode.
