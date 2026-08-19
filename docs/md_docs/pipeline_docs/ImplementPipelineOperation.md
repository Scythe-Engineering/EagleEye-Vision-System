# Implement a pipeline operation

This page covers the files and contracts needed to add an operation. For runtime concepts and editor behavior, read [Pipelines](PipelineOverview.md).

## Choose a location

Put a small, self-contained operation in `src/secondary_operations/{name}.py`. For a larger operation, put a thin definition in `src/main_operations/definitions/{name}.py` and its logic below `src/main_operations/modules/`.

Main definition classes use the `Definition` suffix. All operation classes inherit `OperationInstance`.

## Implement the runtime contract

Implement `run(self, input_data)`. Declare ports in the config definition to match the value it accepts and returns.

- One output may return any value, including a dictionary.
- Several outputs must return a dictionary keyed by every declared output name.
- A data source receives `None` and sets `is_data_source` to `true`.
- An operation that consumes downstream feedback may implement `back_propagate_input(self, input_data) -> None`.

The pipeline injects shared services only when their exact parameter names appear in the constructor. Examples include `web_interface`, `device_registry`, `model_library`, `network_table`, `mx3_coordinator`, `camera_config_registry`, and `logger`. Do not request a service the operation does not use.

## Add the config definition

Create one of these files:

- Main operation: `src/main_operations/definitions/config_data/{name}_config_def.json`
- Secondary operation: `src/secondary_operations/config_data/{name}_config_def.json`

A minimal definition is:

```json
{
  "class_name": "ClampValue",
  "description": "Clamp a numeric value",
  "category": "proc",
  "folder": "Processing",
  "input_nodes": [{"name": "value", "has_default": false}],
  "output_nodes": ["value"],
  "parameters": {
    "maximum": {
      "type": "float",
      "default": 1.0,
      "required": false,
      "restart_for_change": false
    }
  }
}
```

`class_name` must match the Python class. Use an existing category and folder. Parameter definitions may include `description`, `min`, `max`, `options`, and UI hints. For managed inference, use `model_id` with `ui_hint: "model_library"` and `device_id` with `ui_hint: "device_registry"`.

Use [Dynamic port groups](DynamicPortGroups.md) when the number of ports depends on graph connections.

## Before submitting

Check that constructor names match `action_params`, config defaults match Python defaults, every declared output is returned, and the operation handles invalid or missing inputs intentionally. Generate a pipeline and run a representative input through it. Add end-user documentation that states the operation's inputs, outputs, purpose, configuration, and real limitations.
