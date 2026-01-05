---
name: pipeline-architecture
description: Design and understand the configuration-driven pipeline architecture, operation chaining, and data flow patterns
---

# Pipeline Architecture Skill

Use this skill when designing or analyzing the EagleEye pipeline system architecture.

## Configuration-Driven Architecture

The entire system is driven by `src/config/pipeline_config.json`:
- Per-camera operation chains (each camera has independent pipeline)
- Operations defined by `action_name` (must match Python class name)
- `action_params` provides operation-specific configuration
- `position` field controls visual placement in pipeline editor

```json
{
    "camera_name": [
        {
            "action_name": "detect_apriltags",
            "action_params": {
                "apriltag_map_path": "files/apriltag_map.json",
                "device_id": "CPU"
            },
            "position": {"x": 100, "y": 50}
        },
        {
            "action_name": "pnp_camera_localization",
            "action_params": {
                "camera_params": "files/camera_params.json"
            },
            "position": {"x": 300, "y": 50}
        }
    ]
}
```

**Key insight**: This is NOT a hardcoded pipeline—it's data-driven. Adding operations is purely configuration.

## Pipeline Flow

Each camera frame flows through its operation chain:
```
Frame Input
    ↓
Operation 1 (e.g., detect_apriltags)
    ↓ output becomes input
Operation 2 (e.g., pnp_camera_localization)
    ↓ output becomes input
Operation N (e.g., robot_pose_output)
    ↓
SocketIO broadcast to frontend
```

Operations have no hardcoded coupling—the pipeline orchestration handles chaining via `src/config/utils/pipeline.py`.

## Operation Injection Pattern

Constructor parameter NAMES determine what gets injected:
- Parameter `web_interface` → receives `EagleEyeInterface` instance
- Parameter `compute_pool` → receives `ComputePool` instance
- All other parameters come from `action_params` in config

This enables operations to access:
- **Web interface**: Update frontend in real-time, broadcast pose updates
- **Compute pool**: Access GPU/MX3/CPU devices polymorphically
- **Config params**: Operation-specific configuration from pipeline config

```python
class MyOperation:
    def __init__(self, web_interface, compute_pool, threshold: float):
        # web_interface and compute_pool are automatically injected
        # threshold comes from action_params in config
        self.web_interface = web_interface  # Broadcast updates to frontend
        self.device = compute_pool.get_compute_device("GPU_0")  # Request device
        self.threshold = threshold  # Config parameter
```

## Operation Categories

Categories are not just labels—they're architectural boundaries:
- `prep` - preprocessing (frame normalization, resizing)
- `det` - detection (YOLO, AprilTags, color thresholding)
- `proc` - processing (pose estimation, filtering)
- `filt` - filtering (outlier rejection, temporal smoothing)
- `net` - networking (NetworkTables publishing, output)

These are used for organization in `src/main_operations/modules/{category}/{operation}/`.

## Device Pool Polymorphism

`ComputePool` provides unified interface across heterogeneous devices:
```python
# Any operation can request any device by ID
device = compute_pool.get_compute_device("GPU_0")  # Or "MX3_0" or "CPU"

# ComputeDevice interface (all devices implement this)
result = device.run(model_path, input_data, input_shape, stream_idx)
```

Available devices (detected at runtime):
- `CPU` - always available
- `GPU_0`, `GPU_1`, ... - NVIDIA CUDA GPUs
- `MX3_0`, `MX3_1`, ... - MemryX accelerators (Linux only)

**Architecture benefit**: Operations don't care which device they get—they work with the same interface. Device selection is configuration-only.

## Dual Process Architecture

System has two independent processes:
- **Backend** (`src/main_backend.py`): Flask server, pipeline orchestration, device management, camera threads
- **Frontend** (Vite dev server or static build): WebUI editor, 3D visualization, real-time updates via SocketIO

They communicate via:
- HTTP API (`http://localhost:5001/`)
- SocketIO for real-time pose broadcasts

**Architecture benefit**: Frontend can be developed independently. Backend can update pose in real-time without frontend rebuild.

## Thin-Wrapper Pattern Rationale

Main operations (>200 lines) split into two files:
- **Wrapper** (`src/main_operations/definitions/{name}.py`): Thin layer handling dependency injection and config resolution
- **Implementation** (`src/main_operations/modules/{category}/{name}/implementation.py`): Core logic

**Why**: Separates DI concerns (what gets injected) from business logic (what the operation does). Makes implementation easier to test and understand.

## Module Organization

Modules organized by functional category:
```
src/main_operations/modules/
├── object_detection/          # Detection algorithms
│   ├── yolo_detection/
│   │   └── implementation.py
│   ├── color_threshold_detection/
│   │   └── implementation.py
│   └── utils/                 # Shared utilities (letterbox, etc.)
├── apriltags/                 # AprilTag detection & pose
│   ├── apriltag_detector.py
│   ├── pnp_localization.py
│   └── utils/
└── {category}/                # New categories as needed
```

## Real-Time Communication

Backend broadcasts pose updates via SocketIO:
```python
# Backend (src/webui/web_server.py)
socketio.emit('update_robot_transform', pose_data)

# Frontend listens and updates 3D visualization
socket.on('update_robot_transform', (data) => {
    updateRobotPosition(data);
});
```

**Architecture benefit**: Frontend gets live updates without polling. Enables smooth 3D visualization.

## Custom PyPI Indices Strategy

Platform-specific dependencies managed via PyPI indices:
- `pytorch-cuda`: Windows CUDA version
- `pytorch-cpu`: Linux CPU fallback
- `memryx`: Proprietary Linux-only accelerator

Defined in `pyproject.toml` with platform markers. `uv` resolver handles selection based on `sys_platform`.

## Integration Points

**Critical coupling points to understand**:
1. Pipeline config defines everything—system behavior is configuration
2. Device pool is singleton—all operations share same device manager
3. Web interface is singleton—all operations can broadcast to frontend
4. SocketIO connects backend and frontend—real-time updates
5. Config files drive operation discovery and instantiation

**Non-obvious constraint**: Operations MUST be stateless. The hidden caching layer (device pool, web interface) assumes this. Keeping state in operation instances will cause issues across frames.

## Architectural Decisions

- **No hardcoded pipeline**: Configuration-driven for flexibility
- **Injection-based DI**: Parameters determine injection, not decorators
- **Device abstraction**: Polymorphic ComputeDevice interface
- **Dual process**: Separate backend orchestration and frontend UI
- **Stateless operations**: Frame processing stateless, caching in pool
- **Real-time communication**: SocketIO for live visualization updates
