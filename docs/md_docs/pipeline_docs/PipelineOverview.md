# Pipelines

A pipeline is a directed graph of operations. Camera frames and other values travel through named ports, and each operation transforms or publishes the values it receives. Build and configure pipelines in the WebUI pipeline editor.

## Operation types

- Main operations handle larger tasks such as object detection and localization.
- Secondary operations handle smaller transforms, filters, data sources, and outputs.
- Data-source operations have no input connection. The scheduler runs them before their value is needed.

The distinction does not change how you connect nodes in the editor.

## Connections and outputs

Connect an output port to a compatible input port. For an operation with several outputs, the selected output port determines which value travels along the connection. A dictionary returned through one declared output remains one value; the runtime does not split it into ports.

Some operations have indexed dynamic ports such as `pose_1` and `pose_2`. See [Dynamic port groups](DynamicPortGroups.md).

Timing metadata attached to a camera frame follows derived values through the graph. An operation's value does not need to contain its own timestamp.

## Running a pipeline

The scheduler runs operations when their inputs are ready. Returning no usable result can stop that branch for the current cycle. Multi-output operations return values keyed by their declared output names.

The Pipeline Settings option **Limit frames to camera capture speed** starts another complete run only after every connected `device_input` in the pipeline has published a new frame. It is enabled by default. All `device_input` nodes in that named pipeline must belong to one connected graph. Pipelines without `device_input` continue to run continuously. This setting takes effect after a backend restart.

## Common localization chain

A basic AprilTag localization graph is:

```text
Device Input -> Detect AprilTags -> PnP Camera Localization -> Camera to Robot Pose -> output
```

PnP returns a camera pose. Keep `camera_to_robot_pose` in the chain when the frontend or publisher expects the robot pose.

For operation-specific inputs, outputs, and settings, use the operation pages in this section. Contributors adding an operation should read [Implement a pipeline operation](ImplementPipelineOperation.md).
