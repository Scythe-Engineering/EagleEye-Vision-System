# RobotLocalToFieldTransform

`RobotLocalToFieldTransform` converts detection positions from robot-local coordinates to field coordinates with the latest robot pose.

## Inputs

- `detections`: Dictionaries with finite three-element `position_3d` values.
- `robot_pose`: Optional finite 4 by 4 robot-to-field transform.

A detections list can also be supplied directly without a pose update.

## Outputs

`field_detections` is a new list. Transformed items keep the local position in `position_robot` and replace `position_3d` with the field position. `None` detections produce `None`.

## When to use

Use this only after an operation that produces robot-local positions. `GroundPlaneIntersection` already returns field-relative positions.

## Configuration

This operation has no user parameters.

## Limitations

A valid robot pose must arrive before a valid position can be transformed. The latest pose is reused on later runs. Malformed positions pass through unchanged.
