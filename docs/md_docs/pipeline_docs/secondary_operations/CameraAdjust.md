# CameraAdjust

`CameraAdjust` applies brightness, contrast, saturation, gain, and exposure through Linux V4L2 controls. It does not alter pixels in software.

## Inputs

- `frame`: Required image frame.
- `detections`: Optional AprilTag detections, used only by visualization.

## Outputs

`frame` is the original frame after hardware controls have been enforced.

## When to use

Use this near the start of a camera pipeline when the camera needs fixed manual settings.

## Configuration

- `camera_bus_id`: Required camera bus ID. Changing it requires a restart.
- `brightness`: `-1` to `1`, default `0.0`.
- `contrast`: `0` to `1`, default `0.5`.
- `saturation`: `-1` to `1`, default `0.406`.
- `gain`: `0` to `1`, default `0.0`.
- `exposure`: `0` to `1`, default `0.5`.

## Limitations

This requires Linux, `v4l2-ctl`, and controls supported by the camera driver. It disables supported automatic controls and reapplies settings about once per second.
