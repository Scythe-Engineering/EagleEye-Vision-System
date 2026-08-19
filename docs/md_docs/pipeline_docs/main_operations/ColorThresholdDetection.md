# Color threshold detection

`color_threshold_detection` finds connected regions that fall within configured HSV ranges. It returns one detection per contour after optional blur and morphology cleanup.

## Inputs

- `frame`: a three-channel BGR NumPy image.

## Outputs

- `detections`: a list of dictionaries with `bbox`, `class_id`, `color_name`, and `area`.

`bbox` is `[x1, y1, x2, y2]`, normalized against the unpadded image content. `area` is measured in pixels on the letterboxed image.

## When to use

Use this operation when the target has a predictable color and a trained detection model is unnecessary. HSV ranges usually need retuning when lighting or camera exposure changes.

## Configuration

| Setting | Default | Notes |
| --- | --- | --- |
| `camera_bus_id` | required | Camera whose calibration is used to undistort box corners. Requires restart. |
| `target_size` | `320` | Square letterbox size, 64 to 1024. Requires restart. |
| `color_ranges` | required | List of objects containing `name`, `class_id`, `lower_hsv`, and `upper_hsv`. |
| `min_area` | `100` | Reject contours smaller than this letterboxed pixel area. |
| `max_area` | `50000` | Reject contours larger than this letterboxed pixel area. |
| `blur_kernel_size` | `0` | Gaussian kernel size. Use `0` to disable it; nonzero values must be odd. |
| `morphology_kernel_size` | `5` | Opening and closing kernel size. It must be odd. |
| `morphology_iterations` | `0` | Use `0` to disable morphology. The editor currently validates configured nonzero values from 1 to 10. |

HSV hue uses 0 to 179. Saturation and value use 0 to 255.

```json
{
  "camera_bus_id": "0-1",
  "color_ranges": [
    {
      "name": "red",
      "class_id": 0,
      "lower_hsv": [0, 100, 100],
      "upper_hsv": [10, 255, 255]
    }
  ]
}
```

## Important behavior and limitations

The operation rejects monochrome frames. One physical object can produce several detections if its mask has disconnected regions. Bounds may extend outside 0 to 1 after padding removal or point undistortion because the implementation does not clip them.
