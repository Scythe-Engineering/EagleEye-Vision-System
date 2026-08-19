# GroundPlaneIntersection operation

## Overview

`GroundPlaneIntersection` projects the bottom center of each detection bounding box onto the field ground plane. It uses camera intrinsics from the selected camera configuration and a field-relative camera pose supplied by another operation, normally `PnpCameraLocalizationDefinition`.

The operation does not use camera mounting extrinsics from camera configuration.

## Inputs and output

Inputs:

- `detections`: detection dictionaries with normalized `bbox` coordinates.
- `camera_pose`: a finite 4x4 `T_field_from_camera` matrix, such as the output of `pnp_camera_localization`.

Output:

- `positions_3d`: copies of detections whose rays intersect the ground plane. Each copy has `position_3d` set to `[field_x, field_y, field_z]` in meters.

If either input is `None`, the operation returns an empty list. It skips missing bounding boxes, rays above or within 3 degrees of the horizon, and intersections behind the camera.

## Configuration

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `camera_bus_id` | `str` | required | Resolves the camera matrix and calibrated image size. |
| `ground_level` | `float` | `0.0` | Field-relative Z coordinate of the ground plane in meters. |

`camera_config_registry` is injected by the pipeline runtime. The operation reads only the selected camera's `intrinsics_path`.

## Projection

For a bounding box `[x1, y1, x2, y2]`, the operation uses its bottom-center pixel to construct an OpenCV camera ray:

```text
ray_camera = [(x_pixel - cx) / fx, (y_pixel - cy) / fy, 1]
ray_field = R_field_from_camera * ray_camera
scale = (ground_level - camera_field_z) / ray_field_z
position_field = camera_field_position + scale * ray_field
```

Input detections must correspond to an undistorted image because the operation does not apply distortion coefficients.

## Pipeline example

Connect an object detection output to `detections` and the PnP localization output to `camera_pose`:

```text
object_detection -------- detections -----> ground_plane_intersection
pnp_camera_localization -- camera_pose ---> ground_plane_intersection
```
