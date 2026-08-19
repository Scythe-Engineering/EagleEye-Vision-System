# Pose outlier filter

`pose_outlier_filter.PoseOutlierFilter` accepts or rejects 4x4 pose transforms using position and rotation gates.

```python
from pose_outlier_filter import PoseOutlierFilter

filter = PoseOutlierFilter()
accepted = filter.run(pose)  # pose is a row-major sequence of 16 floats
```

`run()` returns the accepted 16-element pose or `None`. The first valid pose is accepted. `update_config(config)` updates any constructor option: `history_size`, `base_sigma`, `growth_rate`, `gate_k`, `max_consecutive_rejections`, `relax_factor`, `angular_gate_threshold`, `velocity_smoothing_alpha`, and `full_reset_threshold`.

Build from `src/rust_implementations` with `uv run python build.py pose_outlier_filter`.
