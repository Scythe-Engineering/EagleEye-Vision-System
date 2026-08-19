# GroundPlaneIntersection

`GroundPlaneIntersection` projects the bottom center of each detection bounding box onto a horizontal field ground plane. It uses calibrated intrinsics and a field-relative camera pose.

## Inputs

- `detections`: Dictionaries with normalized `bbox` coordinates `[x1, y1, x2, y2]` from an undistorted image.
- `camera_pose`: A finite 4 by 4 transform from camera coordinates to field coordinates.

## Outputs

`positions_3d` is a new list. Each retained detection has a field-relative `position_3d` in meters; other fields are preserved.

## When to use

Use this to estimate field positions for objects that touch the ground when a field-relative camera pose is available.

## Configuration

- `camera_bus_id`: Required camera bus ID used to resolve intrinsics.
- `ground_level`: Ground-plane Z coordinate in field meters. Default `0.0`.

## Limitations

Missing detections or camera pose produce an empty list. The operation skips malformed detections, cameras at or below the plane, backward intersections, and rays angled downward by no more than 3 degrees. Intrinsics must match the undistorted image and include a valid camera matrix and image size.
