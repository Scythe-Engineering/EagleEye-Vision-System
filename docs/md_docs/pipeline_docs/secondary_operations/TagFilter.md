# TagFilter

`TagFilter` keeps or removes AprilTag detections according to their tag IDs.

## Inputs

`detections` is a list of `Detection` or `CustomDetection` objects, or `None`. Each item must have a `tag_id`.

## Outputs

`filtered_detections` is a new list containing retained objects. `None` remains `None`.

## When to use

Use this before localization when only selected field tags should contribute, or when known bad tags must be excluded.

## Configuration

- `filter_mode`: `whitelist` or `blacklist`. Default `whitelist`.
- `tag_ids`: List of integer IDs. Default empty.

## Behavior

An empty whitelist keeps every tag. An empty blacklist removes nothing. Visualization draws retained detections in green and excluded detections in red.
