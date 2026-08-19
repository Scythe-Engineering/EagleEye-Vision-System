# CameraLocalToRobotTransform

`CameraLocalToRobotTransform` converts camera-local detection positions into robot-local coordinates using the selected camera's mounting extrinsics.

## Inputs

- `detections`: detection dictionaries. A transformable detection has a finite three-element `position_3d`.

## Outputs

- `robot_detections`: a new list. Transformed detections retain the original value as `position_camera` and replace `position_3d` with the robot-local position. Entries without a valid position pass through unchanged.

## When to use

Use this after an operation that estimates positions in camera coordinates and before robot-relative or field-relative processing.

## Configuration

- `camera_bus_id`: required camera bus ID used to load mounting extrinsics. It supports live updates.

## Limitations

The camera must have finite mounting extrinsics. The operation raises an error when the camera registry or calibration is unavailable.
