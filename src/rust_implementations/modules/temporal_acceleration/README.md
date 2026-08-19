# Temporal acceleration

`temporal_acceleration.TemporalAcceleration` projects known AprilTags into the next camera frame and returns crop regions.

```python
from temporal_acceleration import TemporalAcceleration

acceleration = TemporalAcceleration(
    camera_matrix, distortion_coefficients, tag_ids, tag_corners, tag_centers
)
acceleration.back_propagate_input(world_from_camera)
crop_quads, regions = acceleration.process_frame(width, height)
```

`camera_matrix` has nine row-major values. `tag_corners` has 12 floats per tag and `tag_centers` has three. `world_from_camera` is a 16-element transform. `process_frame()` returns perspective crop quadrilaterals and `[x, y, width, height]` regions. If no tag can be projected, it returns an empty quadrilateral list and one full-frame region.

`update_config(config)` accepts `padding_factor`, `max_regions`, `min_region_size_px`, and `max_detection_distance_m`.

Build from `src/rust_implementations` with `uv run python build.py temporal_acceleration`.
