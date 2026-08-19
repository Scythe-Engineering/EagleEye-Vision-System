# MinimumApriltagCount

`MinimumApriltagCount` stops the current pipeline cycle unless an AprilTag result contains enough detections.

## Inputs

- `detections`: a sized AprilTag detection collection, or `None`.

## Outputs

- `detections`: the unchanged collection when it meets the threshold. Otherwise the operation returns the pipeline's skip-cycle signal, so downstream operations do not run for that frame.

## When to use

Use this before pose estimation or fusion when too few visible tags would make the result unacceptable. Do not use it when a valid single-tag estimate should continue through the pipeline.

## Configuration

- `minimum_detections`: required count, at least 1. Default `2`. It supports live updates.

## Limitations

The operation checks only collection length. It does not inspect tag IDs, ambiguity, geometry, or detection quality.
