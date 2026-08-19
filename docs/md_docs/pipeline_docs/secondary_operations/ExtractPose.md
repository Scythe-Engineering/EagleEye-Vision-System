# ExtractPose

`ExtractPose` reduces a 4 by 4 pose matrix to planar position and heading.

## Inputs

`transform` is a 4 by 4 transformation matrix, or `None`.

## Outputs

`pose_2d` is `{"x": x, "y": y, "rotation": yaw}`. Rotation is in radians. `None` produces `None`.

## When to use

Use this when a downstream operation needs a simple 2D pose dictionary instead of a matrix.

## Configuration

This operation has no user parameters.

## Limitations

A non-`None` input with any other shape raises an error. Height, roll, and pitch are discarded.
