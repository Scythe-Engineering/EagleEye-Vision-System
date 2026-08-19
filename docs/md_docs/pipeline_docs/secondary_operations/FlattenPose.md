# FlattenPose

`FlattenPose` converts a 3D pose matrix to a planar 4 by 4 pose. It keeps X, Y, and yaw, sets Z to zero, and removes roll and pitch.

## Inputs

`pose_3d` is a 4 by 4 transformation matrix.

## Outputs

`pose_2d` is a copied 4 by 4 matrix. The input is not changed.

## When to use

Use this when later matrix-based operations expect a pose constrained to the ground plane.

## Configuration

This operation has no user parameters.

## Limitations

The operation does not validate matrix shape or finite values.
