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
| `nthreads` | `1` | Detector threads, 1 to 16. Used for region searches and, unless overridden, full-frame searches. |
| `full_frame_nthreads` | `0` | Full-frame-only thread count, 0 to 16. Zero follows `nthreads` without allocating another detector. |
| `quad_decimate` | `2.0` | Quad-search downsampling factor, 1.0 to 10.0. Larger values trade corner precision and detection range for less image work. |
| `quad_sigma` | `0.0` | Blur applied during quad search, 0.0 to 5.0. |
| `refine_edges` | `1` | Use `1` to refine detected edges, or `0` to disable it. |
| `decode_sharpening` | `0.25` | Decode sharpening, 0.0 to 1.0. |
| `small_roi_max_px` | `32` | Regions with a shorter side below this threshold use decimation 1. Zero disables this override. |
| `large_roi_decimate` | `3.0` | Decimation for large region searches. |
| `large_roi_min_px` | `96` | Minimum shorter-side length for the large-region detector. |

All settings support live updates. Separate full-frame threading avoids thread-pool overhead on small temporal crops. The override applies to direct full-frame inputs and recovery searches, not individual regions, even when a region spans the frame. A nonzero override different from `nthreads` allocates one additional native detector.

```json
{
  "families": "tag36h11",
  "nthreads": 1,
  "full_frame_nthreads": 2,
  "quad_decimate": 2.0,
  "refine_edges": 1
}
```

## Important behavior and limitations

Tiny regions use a separate single-resolution detector so decimation does not discard already scarce tag pixels. This does not change direct full-frame detection. The extra detector uses `nthreads`, is reused across frames, and is unnecessary when the base decimation is already 1. Small-region selection takes priority over large-region selection if configured thresholds overlap.

Region mode searches the full frame once when no region contains a tag. Detected region corners are mapped back to full-frame coordinates. The operation does not estimate pose; connect its output to `pnp_camera_localization` for that step. Small, blurred, occluded, or low-contrast tags may not decode.
