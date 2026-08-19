# DeviceInput

`DeviceInput` reads the newest frame from one managed camera and attaches capture timing. It can rotate the frame before processing.

## Inputs

None. This is a data-source operation.

## Outputs

`frame` is a timestamped BGR image packet. The operation returns `None` until a frame is available.

## When to use

Use this as the source of a camera pipeline.

## Configuration

- `camera_bus_id`: Required camera bus ID. Changing it requires a restart.
- `frame_rotation`: Clockwise rotation. Allowed values are `0`, `90`, `180`, and `270`; default `0`.

## Behavior

Rotation preserves capture timing and frame sequence. In asynchronous pipelines, the source waits for a newer sequence and fails if the camera worker stops.
