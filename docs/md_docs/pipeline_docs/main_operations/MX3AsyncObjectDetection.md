# MX3 Async Object Detection

`mx3_async_object_detection` runs YOLO detection through MemryX SDK 2.2's Python-wrapped `memryx.mxapi.MxAccl` API. It is separate from synchronous CPU/CUDA Object Detection.

## Editor binding

Connect `Device Input: frame` directly to `MX3 Async Object Detection: frame`. The detector docks to the Device Input and follows it. One Device Input may own one docked MX3 detector. Detaching preserves settings, but the pipeline cannot be saved or started until it is redocked.

The Device Input frame output remains available to other branches.

## Outputs

- `frame`: the exact transformed camera frame submitted for this inference.
- `detections`: normalized detections from that frame.

Both outputs carry the source frame's timing metadata. Completed intermediate results may be replaced, but in-flight frame/output correlation remains FIFO and bounded.

## Model profile

The managed model requires an `mx3_dfp`, an optional cropped `mx3_postprocessor`, and profile metadata. The initially supported profile is:

```json
{
  "input_width": 320,
  "input_height": 320,
  "color_order": "rgb",
  "layout": "hwzc",
  "normalization": "zero_to_one",
  "use_model_shape": [false, true],
  "decoder": "yolo_nms_xyxy",
  "adjustable_controls": {
    "confidence": true,
    "max_detections": true
  },
  "max_inflight": 8
}
```

`yolo_nms_xyxy` expects the post-model output to contain rows of `[x1, y1, x2, y2, confidence, class_id]` in model-input pixel coordinates. Unsupported profiles fail explicitly.

## Runtime behavior

All streams selecting the same `mx3:N` and DFP share one local-mode `MxAccl` runtime with distinct stream IDs. Selecting different DFPs on one physical MX3 fails. Pipeline disable pauses only its stream; re-enable resumes from the newest camera frame. Runtime failures require a pipeline/service restart and never fall back to CPU.
