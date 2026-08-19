# PoseOutlierFilterRust

`PoseOutlierFilterRust` rejects pose measurements that disagree with a constant-velocity prediction built from accepted pose history.

## Inputs

`pose` is a 4 by 4 transformation matrix.

## Outputs

`filtered_pose` is the accepted matrix. Rejected measurements produce `None`.

## When to use

Use this after pose estimation to stop isolated position or angle jumps from reaching localization outputs.

## Configuration

- `history_size`: Accepted poses retained, default `20`.
- `base_sigma`: Base positional uncertainty in meters, default `0.1`.
- `growth_rate`: Uncertainty growth after rejections, default `0.2`.
- `gate_k`: Positional gate multiplier, default `3.0`.
- `angular_gate_threshold`: Angular gate in radians, default `0.5`.
- `velocity_smoothing_alpha`: Velocity update factor, default `0.3`.
- `max_consecutive_rejections`: Rejections before gate relaxation, default `10`.
- `relax_factor`: Relaxed-gate multiplier, default `2.0`.
- `full_reset_threshold`: Rejections before reset, default `10`.

## Limitations

The Rust `pose_outlier_filter` extension must be built and importable. Python does not validate shape or finite values before flattening the input.
