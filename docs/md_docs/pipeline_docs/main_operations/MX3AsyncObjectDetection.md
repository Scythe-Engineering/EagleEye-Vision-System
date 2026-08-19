# MX3 async object detection

`mx3_async_object_detection` runs asynchronous YOLO detection through a MemryX MX3. It docks directly to a `device_input` operation so each result stays paired with the frame submitted for inference.

## Inputs

- `frame`: supplied through a direct dock to `Device Input: frame`. Frames do not arrive through normal graph execution.

## Outputs

- `frame`: the transformed frame used for inference.
- `detections`: normalized detections for that frame.

Both outputs retain the source frame's timing metadata.

## When to use

Use this operation for a managed YOLO model compiled for MX3. Use `object_detection` for synchronous CPU or CUDA inference.

## Configuration

| Setting | Default | Notes |
| --- | --- | --- |
| `model_id` | required | Managed model with an MX3 DFP and runtime profile. Requires restart. |
| `device_id` | required | Canonical `mx3:N` device ID. Requires restart. |
| `confidence_threshold` | `0.25` | Minimum confidence when the model profile exposes this control. |
| `max_detections` | `100` | Result limit when the profile exposes this control, 1 to 1000. |

Select the managed model and physical MX3 in the pipeline editor, then dock the operation to the `device_input` frame port.

## Important behavior and limitations

One `device_input` can own one docked MX3 detector. The pipeline cannot run or save while the detector is detached. Other branches may still consume the source frame.

Streams using the same physical MX3 must use the same DFP. Runtime failures do not fall back to CPU and require a pipeline or service restart. The current decoder profile expects postprocessor rows of `[x1, y1, x2, y2, confidence, class_id]` in model-input pixel coordinates. Unsupported profiles fail during setup.
