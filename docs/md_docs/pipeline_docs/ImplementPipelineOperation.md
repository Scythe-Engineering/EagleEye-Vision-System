## Implementing pipeline operations (concise)

This file contains the exact, essential rules and minimal examples needed to add a new pipeline operation. For high-level architecture and rationale, see `docs/md_docs/pipeline_docs/PipelineOverview.md`.

### Placement and naming

- Main operations (definitions): put the wrapper in `src/main_operations/definitions/{operation_name}.py` and name the class `CamelCaseDefinition` (e.g., `MyOpDefinition`).
- Secondary operations: put the class in `src/secondary_operations/{operation_name}.py` and name it `CamelCase` (e.g., `MyFilter`).
- For complex logic, implement under `src/modules/{operation_name}/implementation.py` and keep the definition as a thin wrapper.

### Constructor args and automatic injection

- Supply operation-specific parameters via `action_params` in `pipeline_config.json`.
- If the constructor parameter name includes `web_interface`, the pipeline will inject `EagleEyeInterface` automatically.
- If it includes `compute_pool`, the pipeline will inject `ComputePool` automatically.
- Parameter names (not annotations) are used for injection; other args must come from `action_params`.

### run contract

- Implement a `run(self, input)` method. Common types:
    - image/frame processing: input and output `np.ndarray`.
    - detections/features: input `np.ndarray`, output list or structured object.
    - pose estimation: output 4x4 `np.ndarray` transform.
- Document the concrete input/output types in the class docstring.
- **Optional**: Implement `back_propagate_input(self, input_data: YourDataType) -> None` for operations that must adjust state based on downstream BackPropagate results. BackPropagate operations automatically call this method on upstream operations during back-propagation. The method should return `None` unless otherwise specified. For more details, see the BackPropagate operation documentation.

    Example implementation:

    ```python
    class MyStatefulOperation:
        def __init__(self):
            self.adjustment_factor = 1.0

        def run(self, frame: np.ndarray) -> np.ndarray:
            # Apply some transformation using current state
            return frame * self.adjustment_factor

        def back_propagate_input(self, feedback_data: np.ndarray) -> None:
            # Adjust internal state based on back-propagated feedback
            self.adjustment_factor = np.mean(feedback_data)
    ```

### Thin-wrapper pattern (recommended)

- Keep definitions short: instantiate a module implementation and delegate `run`.
- Example structure:
    - `src/modules/{operation_name}/implementation.py` -> heavy logic class `XxxImplementation`.
    - `src/main_operations/definitions/{operation_name}.py` -> wrapper `XxxDefinition` that resolves devices/resources and calls `self.delegate.run(...)`.

Minimal example — implementation:

```python
# src/modules/my_op/implementation.py
import numpy as np
class MyOpImplementation:
    def __init__(self, model_path: str, device: object, threshold: float = 0.1):
        self.model_path = model_path
        self.device = device
    def run(self, frame: np.ndarray) -> np.ndarray:
        return frame
```

Minimal example — wrapper (definition):

```python
# src/main_operations/definitions/my_op.py
from src.modules.my_op.implementation import MyOpImplementation
from src.utils.device_management_utils.compute_pool import ComputePool
class MyOpDefinition:
    def __init__(self, model_path: str, device_id: str, compute_pool: ComputePool, threshold: float = 0.1):
        device = compute_pool.get_compute_device(device_id)
        self.delegate = MyOpImplementation(model_path, device, threshold)
    def run(self, frame):
        return self.delegate.run(frame)
```

### Config snippet

Single action entry in `src/config/pipeline_config.json`:

```json
{
    "action_name": "my_op",
    "action_params": {
        "model_path": "/models/model.onnx",
        "device_id": "MX3",
        "threshold": 0.1
    }
}
```

### Configuration Definitions (config_data)

For both main operations (definitions) and secondary operations, create configuration definition files in:

- Main operations: `src/main_operations/definitions/config_data/{operation_name}_config_def.json`
- Secondary operations: `src/secondary_operations/config_data/{operation_name}_config_def.json`

These JSON files define the operation's parameters, validation rules, and metadata in a structured format.

Configuration definition structure:

```json
{
    "class_name": "OperationClassName",
    "description": "Brief description of what the operation does",
    "category": "cat",
    "is_data_source": false,
    "input_nodes": [],
    "output_nodes": [],
    "parameters": {
        "parameter_name": {
            "type": "str|int|float|bool",
            "description": "Parameter description",
            "default": "default_value",
            "required": true|false,
            "min": 0,           // for numeric types
            "max": 100,         // for numeric types
            "restart_for_change": true|false
        }
    }
}
```

Key elements:

- **class_name**: Must match the Python class name exactly (main operations use "Definition" suffix, e.g., "MyOpDefinition")
- **description**: Human-readable description of the operation's purpose
- **category**: Operation category (e.g., "prep" for preprocessing, "det" for detection, "proc" for processing, "filt" for filtering, "net" for networking)
- **parameters**: Dictionary of parameter definitions with:
    - **type**: Data type ("str", "int", "float", "bool")
    - **description**: Parameter purpose explanation
    - **default**: Default value if not required
    - **required**: Whether parameter must be provided
    - **min/max**: Value range validation for numeric types
    - **options**: Array of valid string values (for str type, optional)
    - **restart_for_change**: Whether pipeline restart is needed when parameter changes

Example config definition file (`flatten_pose_config_def.json`):

```json
{
    "class_name": "FlattenPose",
    "description": "Converts 3D pose data to 2D coordinates by removing rotational and height components",
    "category": "proc",
    "parameters": {}
}
```

Example with parameters from secondary operation (`velocity_based_filtering_config_def.json`):

```json
{
    "class_name": "VelocityBasedFiltering",
    "description": "Filters robot pose estimates using velocity measurements to reject outliers and provide smooth, consistent position tracking",
    "category": "filt",
    "parameters": {
        "velocity_mad_multiplier": {
            "type": "float",
            "description": "Multiplier applied to mean absolute deviation for outlier rejection",
            "default": 3.0,
            "required": false,
            "min": 1.0,
            "max": 10.0,
            "restart_for_change": false
        }
    }
}
```

Example with parameters from main operation (`apriltag_cnn_preprocessor_config_def.json`):

```json
{
    "class_name": "ApriltagCnnPreprocessorDefinition",
    "description": "Enhances AprilTag detection speed by preprocessing camera images with a convolutional neural network to crop the input image to the area of interest",
    "category": "prep",
    "parameters": {
        "model_path": {
            "type": "str",
            "description": "Path to the trained model weights file",
            "default": "{project_root}/models/apriltag_cnn/model.pth",
            "required": true,
            "restart_for_change": true
        },
        "device_id": {
            "type": "str",
            "description": "The id of the computation device for processing",
            "options": ["CPU", "CUDA", "MX3_001", "CORAL"],
            "default": "MX3_001",
            "required": true,
            "restart_for_change": true
        },
        "conf_threshold": {
            "type": "float",
            "description": "Confidence threshold for predictions (0.0 to 1.0)",
            "default": 0.15,
            "min": 0.0,
            "max": 1.0,
            "required": false,
            "restart_for_change": false
        }
    }
}
```

### Data Source Operations

Data source operations are special operations that generate their own data independently (no input connections). To create a data source operation:

1. Set `"is_data_source": true` in the config_def.json file
2. Set `"input_nodes": []` (empty array since data sources have no inputs)
3. Define `"output_nodes"` with the output port names
4. The `run()` method will receive `None` as input and should return the generated data

Data sources execute one timestep before their data is needed to get the most up-to-date value possible. This is important for operations like `GetNetworktablesValue` that read from external sources where data may change between frames.

Example data source config definition (`get_networktables_value_config_def.json`):

```json
{
    "class_name": "GetNetworktablesValue",
    "description": "Read data from NetworkTable and output it to downstream operations",
    "category": "net",
    "is_data_source": true,
    "input_nodes": [],
    "output_nodes": ["data"],
    "parameters": {
        "network_table_key": {
            "type": "str",
            "description": "Key to read from the network table",
            "default": "",
            "required": true,
            "restart_for_change": false
        }
    }
}
```

### Quick checklist

- [ ] Place wrapper in `src/main_operations/definitions/` or class in `src/secondary_operations/`.
- [ ] For main operations: create `{operation_name}_config_def.json` in `src/main_operations/definitions/config_data/`.
- [ ] For secondary operations: create `{operation_name}_config_def.json` in `src/secondary_operations/config_data/`.
- [ ] Keep definitions thin; implementation lives under `src/modules/` when logic is non-trivial.
- [ ] Provide constructor params via `action_params` and rely on automatic injection for `web_interface` / `compute_pool`.
- [ ] Implement `run` and document I/O types.
- [ ] Test by generating pipelines and feeding a small `np.ndarray` frame through `Pipeline.run`.

For broader context, examples, and timing/debug instructions, consult `docs/md_docs/pipeline_docs/PipelineOverview.md`.
