# Object detection

`object_detection` runs synchronous Ultralytics-compatible YOLO detection on a selected CPU or NVIDIA CUDA device.

## Inputs

- `frame`: a BGR NumPy image.

## Outputs

- `detections`: a list of dictionaries containing normalized, clipped `bbox` coordinates, `confidence`, `class_id`, and `class_name` when class names are available.

## When to use

Use this operation for synchronous YOLO detection on CPU or CUDA. Use `mx3_async_object_detection` for a MemryX MX3 device.

## Configuration

| Setting | Default | Notes |
| --- | --- | --- |
| `model_id` | required | Stable model ID selected from Model Library. Requires restart. |
| `device_id` | `cpu` | Canonical `cpu` or `cuda:N` device ID. Requires restart. |
| `confidence_threshold` | `0.25` | Minimum confidence, 0.0 to 1.0. |
| `iou_threshold` | `0.45` | Non-maximum-suppression IoU threshold, 0.0 to 1.0. |
| `max_detections` | `100` | Maximum results per frame, 1 to 1000. |
| `image_size` | `0` | Square inference-size override. `0` uses model metadata. Requires restart. |

## Important behavior and limitations

The operation supports detection models only. It does not support segmentation, pose, classification, fallback, or device load balancing.

Artifact selection is deterministic. CPU prefers ONNX, then PT. CUDA prefers TensorRT, then PT, then ONNX. CUDA ONNX must activate `CUDAExecutionProvider` on the selected device or initialization fails. TensorRT image-size overrides are rejected, and fixed-shape ONNX overrides must match the exported shape. Selecting `mx3:N` fails explicitly.
