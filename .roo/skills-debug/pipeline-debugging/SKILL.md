---
name: pipeline-debugging
description: Debug pipeline operations, timing issues, injection failures, and configuration errors
---

# Pipeline Debugging Skill

Use this skill when troubleshooting pipeline execution, operation failures, or performance issues.

## Enable Debugging Output

Set `debug_mode = True` in `src/config/utils/pipeline.py` to print per-operation timing statistics:
```python
# src/config/utils/pipeline.py, around line 30
debug_mode = True  # Enable timing output

# Output: "[Operation] time: 0.045s (22 FPS)"
```

This shows frame-by-frame timing for each operation—not visible with standard logging.

## Operation Injection Failures (Silent)

Injection failures don't error. They silently fall back to treating parameter as config param.

**Symptom**: "Missing parameter 'web_interface'" error when running

**Causes**:
1. Parameter name misspelled (e.g., `web_int` instead of `web_interface`)
2. Parameter name not exactly `web_interface` or `compute_pool`
3. Parameter comes from `action_params` which doesn't have it

**Debug**:
```python
class MyOperation:
    def __init__(self, web_interface, compute_pool, threshold: float):
        # Add debug output
        print(f"web_interface: {web_interface}")
        print(f"compute_pool: {compute_pool}")
        print(f"threshold: {threshold}")
```

If `web_interface` or `compute_pool` print as dictionaries instead of objects, injection failed.

**Fix**: Check parameter names exactly match `web_interface` or `compute_pool`.

## Configuration File Not Found

**Symptom**: Operation appears in config but won't load

**Causes**:
1. Config file in wrong location
2. Filename doesn't match operation name (case-sensitive)
3. Missing `class_name` field in config

**Debug**: Check exact path:
- Main operations: `src/main_operations/definitions/config_data/{name}_config_def.json`
- Secondary operations: `src/secondary_operations/config_data/{name}_config_def.json`

Filename must match operation class name exactly (case-sensitive).

## Vite Dev Server Caching

**Symptom**: Frontend changes don't appear even after refresh

**Causes**:
1. Edited `src/webui/static/` instead of source files
2. Vite excluded `static/**` from watch (by design)
3. Browser cache not cleared

**Debug**:
- Edit source files: `src/webui/js/`, `src/webui/css/`, `src/webui/html/`
- NOT `src/webui/static/` (build artifacts)
- Run `npm run build` for production testing
- Check Network tab: Are old bundles cached?

## Device Pool Silent Fallback

**Symptom**: Operation runs but uses wrong device (slow when expecting GPU)

**Causes**:
1. Device ID typo (e.g., "GPU_1" when only "GPU_0" available)
2. Device not detected at runtime
3. MemryX not available (Linux-only)

**Debug**: Check available devices at startup:
```python
# In operation __init__
device = compute_pool.get_compute_device("GPU_0")
print(f"Got device: {type(device).__name__}")  # Check if CPU fallback
```

Invalid device_id silently returns CPU. No error thrown.

## Port Conflicts

**Symptom**: "Address already in use" on startup

**Ports**:
- Flask backend: port 5001 (hardcoded in `src/main_backend.py`)
- Vite dev server: port 5173 (default in `npm run dev`)

**Debug**:
```bash
# Check what's using port 5001
lsof -i :5001

# Check what's using port 5173
lsof -i :5173

# Kill process (if safe)
kill -9 <PID>
```

Both must be available for development.

## Category Name Errors

**Symptom**: Operation "detected" but won't instantiate

**Causes**:
1. Invalid category in config (must be exact)
2. Categories: `prep`, `det`, `proc`, `filt`, `net` only

**Valid categories** (no others):
- `prep` - preprocessing
- `det` - detection
- `proc` - processing
- `filt` - filtering
- `net` - networking

Check `category` field in operation's `_config_def.json`.

## Module Import Errors

**Symptom**: "ModuleNotFoundError" when loading operation

**Causes**:
1. Implementation module path doesn't match category structure
2. Missing `__init__.py` files in package directories
3. Class name mismatch between definition and implementation

**Expected structure**:
```
src/main_operations/modules/
└── {category}/
    └── {operation_name}/
        ├── __init__.py           # May be empty
        └── implementation.py      # Contains implementation class
```

**Debug**: Verify import path:
```python
# In definition file
from src.main_operations.modules.object_detection.yolo_detection.implementation import YoloDetectionImplementation
```

Path must match exact directory structure.

## Camera Thread Failures

**Symptom**: Camera feeds freeze or show no updates

**Causes**:
1. Camera device disconnected
2. Camera file path invalid (for video files)
3. Camera calibration file missing

**Debug**: Check camera availability:
```bash
python src/utils/get_available_devices.py  # Lists cameras too
python src/utils/camera_utils/get_available_cameras.py
```

**Camera calibration paths**:
- Located in: `src/utils/camera_utils/camera_calibrations/{camera_id}/`
- Files: `intrinsics.json`, `extrinsics.json`

Missing calibration files cause pose estimation to fail silently.

## NetworkTables Connection Issues

**Symptom**: Robot data not reaching NetworkTables

**Causes**:
1. NetworkTables server address wrong
2. Pipeline doesn't reach `publish_to_networktables` operation
3. NetworkTables operation not in pipeline config

**Debug**: Check config:
```json
{
    "action_name": "publish_to_networktables",
    "action_params": {
        "table_name": "vision",
        "key": "robot_pose"
    }
}
```

Also verify NetworkTables server address in `src/general_conf.json`.

## Frame Timing Issues

**Symptom**: Lower FPS than expected

**Enable debug timing**:
```python
# src/config/utils/pipeline.py
debug_mode = True
```

**Interpret output**:
```
[detect_apriltags] time: 0.090s (11 FPS)  # Bottleneck here
[pnp_camera_localization] time: 0.010s
[robot_pose_output] time: 0.002s
Total: 0.102s (10 FPS)
```

**Identify bottleneck**: Which operation takes longest?
- Detection operation: Consider smaller model or lower resolution
- Device issue: Check device assignment vs available hardware
- Camera: Check camera FPS setting and resolution

## Silent Failures in Operations

**Symptom**: Operation runs but output is None or empty

**Common causes**:
1. Exception caught and ignored in operation
2. Input doesn't match expected type
3. Optional return (None is valid output)

**Debug**: Add explicit logging:
```python
def run(self, input):
    print(f"Input type: {type(input)}, shape: {getattr(input, 'shape', 'N/A')}")
    result = self._process(input)
    print(f"Output type: {type(result)}, value: {result}")
    return result
```

Check that input is `np.ndarray` (not list), shape is correct, etc.
