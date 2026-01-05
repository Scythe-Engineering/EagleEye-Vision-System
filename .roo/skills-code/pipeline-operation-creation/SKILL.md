---
name: pipeline-operation-creation
description: Create new pipeline operations (main or secondary) following thin-wrapper pattern, injection rules, and configuration-driven architecture
---

# Pipeline Operation Creation Skill

Use this skill when creating new operations for the EagleEye vision pipeline system.

## Operation Type Decision

**Main Operations (>200 lines)**:
- Implement in `src/main_operations/definitions/{name}.py` (thin wrapper)
- Delegate to `src/main_operations/modules/{category}/{name}/implementation.py` (logic)
- Class name: `{CamelCase}Definition`

**Secondary Operations (<200 lines)**:
- Single file in `src/secondary_operations/{name}.py`
- Class name: `CamelCase` (no "Definition" suffix)

## Constructor Injection Rules

Parameter NAMES determine injection (not type hints):
- Parameter named `web_interface` → receives `EagleEyeInterface` instance
- Parameter named `compute_pool` → receives `ComputePool` instance
- All other parameters come from `action_params` in pipeline config

```python
# ✓ Correct - will receive injected dependencies
def __init__(self, web_interface, compute_pool, threshold: float):
    self.web_interface = web_interface
    self.compute_pool = compute_pool
    self.threshold = threshold

# ✗ Wrong - misspelled parameter names won't inject
def __init__(self, interface, pool, threshold: float):  # Lost injections!
    pass
```

## Run Method Contract

All operations must implement:
```python
def run(self, input) -> Any:
    """Process input and return output.
    
    Args:
        input: Frame (np.ndarray) or detections (list/dict)
    
    Returns:
        Processed output for next operation in pipeline
    """
```

Common input/output patterns:
- Image/frame: `np.ndarray` → `np.ndarray`
- Detection: `np.ndarray` → `list[Dict[str, Any]]`
- Pose: `list[Detection]` → `Optional[np.ndarray]` (4x4 transform)

## Configuration File

Create matching JSON config at:
- Main: `src/main_operations/definitions/config_data/{name}_config_def.json`
- Secondary: `src/secondary_operations/config_data/{name}_config_def.json`

```json
{
    "class_name": "YourOperationName",
    "description": "What this operation does",
    "category": "det",
    "parameters": {
        "threshold": {
            "type": "float",
            "description": "Detection threshold",
            "default": 0.5,
            "min": 0.0,
            "max": 1.0,
            "required": true
        }
    }
}
```

**Categories** (must be exact):
- `prep` - preprocessing
- `det` - detection
- `proc` - processing
- `filt` - filtering
- `net` - networking

## Pipeline Configuration

Add operation entry to `src/config/pipeline_config.json`:
```json
{
    "action_name": "my_operation",
    "action_params": {
        "threshold": 0.75,
        "device_id": "GPU_0"
    },
    "position": {"x": 100, "y": 200}
}
```

## Thin-Wrapper Pattern (Main Operations)

**Wrapper** resolves dependencies:
```python
# src/main_operations/definitions/yolo_detection.py
from src.main_operations.modules.object_detection.yolo_detection.implementation import YoloDetectionImplementation

class YoloDetectionDefinition:
    def __init__(self, model_path: str, device_id: str, compute_pool, threshold: float = 0.5):
        device = compute_pool.get_compute_device(device_id)
        self.delegate = YoloDetectionImplementation(model_path, device, threshold)
    
    def run(self, frame: np.ndarray) -> list:
        return self.delegate.run(frame)
```

**Implementation** contains logic:
```python
# src/main_operations/modules/object_detection/yolo_detection/implementation.py
class YoloDetectionImplementation:
    def __init__(self, model_path: str, device: ComputeDevice, threshold: float = 0.5):
        self.model = device.load_model(model_path)
        self.threshold = threshold
    
    def run(self, frame: np.ndarray) -> list:
        # Heavy detection logic here
        return detections
```

## Code Style Requirements

- **Type hints**: All parameters and return values
- **Google-style docstrings**: For all functions
- **Black formatting**: Python code
- **No comments**: Use descriptive variable names instead
- **Descriptive names**: No single-letter variables

## Common Gotchas

1. **Silent injection failures**: Misspelled parameter names don't error—they're treated as config params
2. **Config file location**: Wrong path means operation won't load
3. **Category values**: Must be exact (prep/det/proc/filt/net)
4. **Device pool fallback**: Invalid device_id silently falls back to CPU
5. **Module organization**: Must match category (e.g., object_detection/yolo_detection/)

## Testing Integration

1. Verify operation loads: Check pipeline generation doesn't error
2. Test with sample frame: `operation.run(test_frame)`
3. Check output type: Matches documented input/output contract
4. Verify injection: Print `self.web_interface` and `self.compute_pool` in constructor
