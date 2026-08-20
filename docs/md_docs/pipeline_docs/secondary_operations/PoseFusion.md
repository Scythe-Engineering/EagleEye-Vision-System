# Pose fusion

`pose_fusion` combines valid 4 by 4 pose matrices into one pose. It averages translation directly and averages rotation as quaternions.

## Inputs

- Dynamic inputs named `pose_1`, `pose_2`, and so on.
- The implementation also accepts one 4 by 4 matrix directly.

It ignores `None`, non-finite values, and values that are not 4 by 4 matrices.

## Outputs

- `fused_pose`: a 4 by 4 NumPy transform, or `None` when no input is valid.

A single valid input passes through unchanged.

## When to use

Use pose fusion to combine simultaneous pose estimates expressed in the same coordinate frame and units. It does not convert camera poses into a common frame for you.

## Configuration

| Setting | Default | Notes |
| --- | --- | --- |
| `outlier_threshold` | `1.0` | Maximum composite distance used for rejection when at least four poses are valid. |
| `rotation_weight` | `0.5` | Multiplier for angular distance in the composite metric, 0.0 to 1.0. |

The rejection metric is:

```text
translation distance in meters + rotation_weight * angular distance in radians
```

Connect each source to an indexed `pose_N` input in the pipeline editor.

## Important behavior and limitations

Outlier rejection runs only with four or more valid poses. If it rejects every pose, the operation falls back to averaging the original set. Remaining poses receive inverse-distance weights relative to their cluster center.

The threshold mixes meters and weighted radians, so it is not a pure distance in meters. The operation emits no confidence, covariance, or inlier count. Do not use its output as a quality signal without a separate check.
