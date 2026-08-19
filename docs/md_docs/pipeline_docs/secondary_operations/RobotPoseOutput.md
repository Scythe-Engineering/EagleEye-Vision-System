# RobotPoseOutput

`RobotPoseOutput` sends a robot pose matrix to the WebUI and passes changed poses downstream. It does not publish to NetworkTables.

## Inputs

`pose` is a NumPy pose matrix.

## Outputs

The same `pose` when it differs exactly from the last sent matrix. An exact duplicate produces `None`.

## When to use

Use this when the WebUI should display the robot pose. Add `PublishToNetworktables` downstream if the robot also needs it.

## Configuration

This operation has no user parameters.

## Timestamp behavior

The pipeline runner restores upstream timing around the raw matrix returned here. A changed pose from a timed camera pipeline retains the frame's capture timestamp for downstream publishing. A duplicate returns `None` and creates no measurement.

The operation does not validate matrix shape or finite values.
