# CameraPoseOutput

`CameraPoseOutput` sends a camera pose to the WebUI 3D view and passes the pose downstream.

## Inputs

- `camera_pose`: a finite 4 by 4 camera transform, or `None`.

## Outputs

- `camera_pose`: the unchanged valid pose. Invalid values produce `None`.

## When to use

Use this when the WebUI should display a localized camera. It is not needed when the pose is only consumed by later pipeline operations.

## Configuration

- `camera_bus_id`: required stable camera bus ID used to identify the camera in the viewer. It supports live updates.

## Limitations

An unchanged pose is not sent twice. This operation updates only the WebUI and does not publish to NetworkTables.
