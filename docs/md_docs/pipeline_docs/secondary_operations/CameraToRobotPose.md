# CameraToRobotPose

`CameraToRobotPose` converts a field-relative camera pose into a field-relative robot pose using the selected camera's extrinsics.

## Inputs

`camera_pose` is a finite 4 by 4 transformation matrix, or `None`.

## Outputs

`robot_pose` is a 4 by 4 transformation matrix. Invalid matrices and `None` produce `None`.

## When to use

Use this after camera localization when the pipeline needs the robot origin rather than the camera origin.

## Configuration

- `camera_bus_id`: Required bus ID used to read camera pitch, yaw, roll, and XYZ offsets.

## Limitations

The operation assumes project coordinate conventions. If no camera configuration registry is available, it uses identity extrinsics.
