# DetectedObjectsOutput

`DetectedObjectsOutput` sends field-positioned detections to the WebUI 3D view and passes the original list downstream.

## Inputs

`detections` is a list of dictionaries, or `None`. A publishable item needs a finite three-element `position_3d`. Optional `class_id`, `class_name`, and finite numeric `confidence` fields are included.

## Outputs

The original `detections` list. `None` remains `None`.

## When to use

Use this after an operation that adds `position_3d` when detections should appear in the WebUI.

## Configuration

This operation has no user parameters.

## Behavior

Invalid items are omitted from the WebUI payload. An unchanged sanitized payload is not sent again.
