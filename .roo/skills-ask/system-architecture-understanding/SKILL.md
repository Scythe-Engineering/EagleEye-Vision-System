---
name: system-architecture-understanding
description: Understand the EagleEye Vision System architecture including pipeline flow, component interactions, and design decisions
---

# System Architecture Understanding Skill

Use this skill when you need to understand the overall system design, components, and how they interact.

## System Overview

EagleEye Vision System is an FRC (FIRST Robotics Competition) object detection and robot localization framework. It features:
- Flask-based web interface with real-time 3D visualization
- Configurable processing pipelines (configuration-driven)
- Multi-device support (CPU, GPU, MX3 accelerator)
- Real-time camera frame processing with SocketIO updates
- NetworkTables integration for robot communication

## Architecture Layers

**Layer 1: Configuration** (`src/config/`)
- `pipeline_config.json` - Defines per-camera operation chains
- Operations are data-driven, not hardcoded
- Adding new operations requires only JSON configuration

**Layer 2: Operations** (`src/main_operations/`, `src/secondary_operations/`)
- **Main operations**: Complex (>200 lines), split into definition + implementation
- **Secondary operations**: Simple (<200 lines), single file
- All operations inherit from configuration-driven instantiation
- Receive injected dependencies or config parameters

**Layer 3: Device Management** (`src/utils/device_management_utils/`)
- `ComputePool` - Manages CPU/GPU/MX3 devices polymorphically
- All devices implement `ComputeDevice` interface
- Device selection is configuration-only

**Layer 4: Camera Handling** (`src/utils/camera_utils/`)
- `CameraThreadManager` - Manages capture threads for physical and video cameras
- `camera_thread_manager.py` - Orchestrates frame capture
- Calibration files in `camera_calibrations/{camera_id}/`

**Layer 5: Pipeline Orchestration** (`src/config/utils/`)
- `pipeline.py` - Chains operations together
- `generate_all_pipelines.py` - Creates pipelines from JSON config
- Frame flows through operation chain: Operation 1 output → Operation 2 input

**Layer 6: Web Interface** (`src/webui/`)
- Flask backend serves static files and API
- Vite frontend (JavaScript/Tailwind/Three.js)
- SocketIO for real-time pose updates
- Handlebars templating for UI components

## Data Flow

```
Camera Thread
    ↓ (np.ndarray frame)
Pipeline (per-camera)
    ↓
Operation 1: detect_apriltags
    ↓ (List[Detection])
Operation 2: pnp_camera_localization
    ↓ (np.ndarray 4x4 pose)
Operation N: robot_pose_output / publish_to_networktables
    ↓
SocketIO broadcast to frontend
    ↓ (WebUI updates 3D visualization)
NetworkTables update (for robot code)
```

Each operation's output becomes the next operation's input.

## Key Architectural Patterns

### Configuration-Driven Architecture

Entire system behavior defined by JSON:
```json
{
    "camera_name": [
        {
            "action_name": "detect_apriltags",
            "action_params": { "device_id": "GPU_0" },
            "position": {"x": 100, "y": 50}
        }
    ]
}
```

**Benefit**: Add operations, change parameters, reorder pipeline—all without code changes.

### Dependency Injection via Parameter Names

Constructor parameters determine what gets injected:
```python
def __init__(self, web_interface, compute_pool, threshold: float):
    # web_interface and compute_pool are automatically injected
    # threshold comes from action_params
```

**Benefit**: Operations don't create dependencies—they request them. Enables testing and loose coupling.

### Device Abstraction

All devices implement same interface:
```python
device = compute_pool.get_compute_device("GPU_0")  # Or "CPU", "MX3_0"
result = device.run(model_path, input_data, shape, stream_idx)
```

**Benefit**: Operations don't care which device they get. Device selection is configuration-only.

### Thin-Wrapper Pattern (Main Operations)

Main operations split into two files:
- **Definition**: Resolves dependencies and config
- **Implementation**: Contains core logic

**Benefit**: Separates dependency concerns from business logic. Improves testability.

### Real-Time Communication

Backend broadcasts via SocketIO, frontend updates 3D visualization live.

**Benefit**: Users see pose updates in real-time without frontend rebuild.

## Component Relationships

```
Pipeline Config (JSON)
    ↓
Operation Instantiation (Injection)
    ↓
Operation Instance
    ├─ web_interface (broadcast updates)
    ├─ compute_pool (request devices)
    └─ config parameters (from action_params)
    ↓
Device Management
    ├─ CPU
    ├─ GPU
    └─ MX3
    ↓
Frame Processing
    ├─ Detection
    ├─ Pose Estimation
    └─ Filtering
    ↓
WebUI Frontend (SocketIO updates)
    └─ Three.js 3D visualization
```

## Non-Obvious Design Decisions

### Operations are Stateless

Operations don't hold state across frames. Caching happens in:
- Device pool (model caching)
- Web interface singleton (frame buffering)
- Camera thread manager (frame queue)

Why: Enables frame-parallel processing and device sharing.

### Categories are Architectural Boundaries

Categories (`prep`, `det`, `proc`, `filt`, `net`) organize modules:
```
src/main_operations/modules/
├── object_detection/    (det category)
├── apriltags/          (det category)
└── {new_category}/
```

They're not just labels—they guide file organization.

### Device IDs are Discovered at Runtime

Devices detected on startup:
- `CPU` - always available
- `GPU_0`, `GPU_1`, ... - based on CUDA detection
- `MX3_0`, `MX3_1`, ... - based on MemryX availability (Linux only)

Pipeline config can reference any ID. Invalid IDs silently fall back to CPU.

### Pipeline is Per-Camera

Each camera has independent operation chain:
```json
{
    "camera_0": [operation chain],
    "camera_1": [different operation chain]
}
```

Enables different processing per camera view.

## Integration Points

### FIRST Robotics NetworkTables
- `pynetworktables` library for robot communication
- `publish_to_networktables` operation publishes pose
- Integrated into `src/general_conf.json`

### Camera Hardware
- OpenCV for USB cameras
- Video files for testing
- Calibration files required for pose estimation

### Compute Accelerators
- NVIDIA CUDA GPUs (PyTorch)
- MemryX MX3 (Linux only, proprietary SDK)
- CPU fallback (always available)

### WebUI
- Flask serves static assets from `src/webui/static/`
- SocketIO broadcasts real-time updates
- Handlebars templates for UI components

## Documentation Structure

Key docs (ordered by detail level):
1. **docs/overviews/** - High-level system design
   - `WEBUI_OVERVIEW.md` - Frontend architecture
   - `BACKEND_OVERVIEW.md` - Backend architecture
   - `DEVICE_MANAGEMENT_UTILS_OVERVIEW.md` - Device management

2. **docs/md_docs/pipeline_docs/** - Operation-specific docs
   - `PipelineOverview.md` - Pipeline concepts
   - `ImplementPipelineOperation.md` - How to create operations
   - Operation-specific docs in `main_operations/` and `secondary_operations/`

3. **Inline documentation** - In code
   - Google-style docstrings
   - Type hints on all functions
   - Descriptive variable names

## File Locations Quick Reference

- **Pipeline config**: `src/config/pipeline_config.json`
- **General settings**: `src/general_conf.json`
- **Main operations**: `src/main_operations/definitions/{name}.py`
- **Operation configs**: `src/main_operations/definitions/config_data/{name}_config_def.json`
- **WebUI backend**: `src/webui/web_server.py`
- **WebUI frontend**: `src/webui/js/main.js`
- **Device management**: `src/utils/device_management_utils/`
- **Camera handling**: `src/utils/camera_utils/`
- **Logging**: `src/utils/logging/logger.py`

## Why This Architecture?

1. **Configuration-driven**: Enables non-programmers to configure pipelines
2. **Flexible operations**: Easy to add/remove/reorder without code changes
3. **Device abstraction**: Works with any compute hardware
4. **Real-time updates**: SocketIO keeps frontend in sync
5. **Testable**: Injection enables dependency mocking
6. **Scalable**: Per-camera pipelines enable multi-camera systems
