# Most dense color threshold detection

`most_dense_color_threshold_detection` applies the same HSV thresholding as `color_threshold_detection`, then returns only the candidate with the largest or smallest contour area.

## Inputs

- `frame`: a three-channel BGR NumPy image.

## Outputs

- `detection`: a list containing at most one detection dictionary with `bbox`, `class_id`, `color_name`, and `area`.

## When to use

Use this operation when color thresholding may find several regions but the pipeline needs only the largest or smallest one. Use `color_threshold_detection` when every matching region matters.

## Configuration

It supports the same `camera_bus_id`, `target_size`, `color_ranges`, area limits, blur, and morphology settings as [Color threshold detection](ColorThresholdDetection.md).

- `selection_mode`: `most_dense` selects the largest contour area. `least_dense` selects the smallest. Default `most_dense`.

## Important behavior and limitations

Density means contour area, not filled-pixel ratio or physical density. The result can change when lighting, thresholds, morphology, or camera perspective changes. An empty list means no candidate passed the configured filters.
