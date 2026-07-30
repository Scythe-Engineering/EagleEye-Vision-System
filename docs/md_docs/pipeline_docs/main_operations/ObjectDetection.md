# Object Detection

`object_detection` runs synchronous Ultralytics-compatible YOLO **detection** on one explicitly selected CPU or NVIDIA CUDA device. It does not support segmentation, pose, classification, arbitrary ONNX detectors, fallback, or load balancing.

## Input and output

- Input: BGR NumPy frame.
- Output: a list of detections:

```python
{
    "bbox": [x1, y1, x2, y2],  # normalized and clipped Python floats
    "confidence": float,
    "class_id": int,
    "class_name": str,         # present when model/library names are available
}
```

## Configuration

| Setting | Behavior | Live update |
| --- | --- | --- |
| `model_id` | Required stable ID from Model Library | No |
| `device_id` | Required canonical ID (`cpu`, `cuda:0`, …) | No |
| `confidence_threshold` | Minimum result confidence | Yes |
| `iou_threshold` | NMS IoU threshold | Yes |
| `max_detections` | Maximum results per frame | Yes |
| `image_size` | Square override; `0` uses model/export metadata | No |

The operation owns its Ultralytics model instance. Only changes to `image_size`, `model_id`, or `device_id` require a backend restart.

## Artifact selection

Models are uploaded through **Model Library** and copied below `files/models/<model-id>/`. Pipelines store only the stable model ID.

Selection is deterministic:

- CPU: ONNX, then PT.
- CUDA: TensorRT engine, then PT, then ONNX.

CUDA ONNX is accepted only when `onnxruntime-gpu` activates `CUDAExecutionProvider` on the exact selected index. It fails rather than silently executing on CPU. TensorRT image-size overrides are rejected; fixed-shape ONNX overrides must match the export shape.

## MX3

Legacy synchronous MX3 inference has been removed. Selecting `mx3:N` for this operation produces an explicit unsupported error. Use the separate `MX3 Async Object Detection` operation.
