# AngleToObjects

`AngleToObjects` calculates the horizontal bearing to each color-threshold detection and sorts results from largest area to smallest.

## Inputs

`detections` is a list of dictionaries. Each item must contain normalized `bbox` coordinates, `class_id`, and `color_name`. `area` is optional and defaults to `0`.

## Outputs

`angles` contains `angle_degrees`, `angle_radians`, `bbox`, `class_id`, `color_name`, and `area`. Positive angles are right of image center.

## When to use

Use this when downstream code needs horizontal bearings from color-threshold detections.

## Configuration

- `camera_fov_degrees`: Horizontal field of view. Default `60.0`; range `10.0` to `180.0`.

## Limitations

Bounding boxes must use normalized coordinates from `0` to `1`. The operation does not account for lens distortion.
