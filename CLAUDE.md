# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EagleEye Vision System is an FRC (FIRST Robotics Competition) object detection and robot localization framework. It features a Flask-based web interface with real-time 3D visualization, configurable processing pipelines, and support for multiple compute devices (CPU, GPU, MX3 accelerator). The system processes camera frames through a chain of operations to detect game pieces, estimate robot poses, and publish results to NetworkTables.

## Build & Run Commands

### Python Environment
- **Install dependencies**: `uv sync` (uses uv package manager)
- **Run backend**: `python src/main_backend.py` (initializes Flask server on port 5001, loads pipelines, starts camera threads)
- **Backend initialization flow**: Builds Rust implementations → Detects available devices (CPU/GPU/MX3) → Initializes compute pool → Generates pipelines → Starts camera threads

### Frontend (WebUI)
- **Install dependencies**: `npm install`
- **Development server**: `npm run dev` (runs Vite dev server, typically localhost:5173)
- **Build for production**: `npm run build` (compiles with rolldown, outputs to `src/webui/static/`)
- **Watch mode**: `npm run watch` (continuous rebuild)

### Note
The WebUI must be built (`npm run build`) before the Flask server can serve the pipeline editor UI, otherwise only static endpoints are available.

## Architecture Overview

### Backend Stack
- **Entry point**: `src/main_backend.py` (MainBackend class initializes everything)
- **Web server**: `src/webui/web_server.py` (Flask + SocketIO on port 5001)
- **Pipeline system**:
  - `src/config/utils/pipeline.py` - Orchestrates frame processing through operation chains
  - `src/config/utils/generate_all_pipelines.py` - Creates pipelines from JSON config
  - `src/config/pipeline_config.json` - Defines per-camera operations and parameters
- **Operations**:
  - **Main operations** (`src/main_operations/definitions/`) - Complex operations >200 lines, use thin-wrapper pattern delegating to `src/main_operations/modules/{category}/{operation}/implementation.py`
  - **Secondary operations** (`src/secondary_operations/`) - Simple operations <200 lines, standalone single-file implementations
- **Device management**: `src/utils/device_management_utils/`
  - `compute_pool.py` - Manages CPU/GPU/MX3 device pool
  - `compute_device.py` - Abstract base class
  - `cpu.py`, `gpu.py`, `mx3_accelerator.py` - Device implementations
- **Camera handling**: `src/utils/camera_utils/camera_thread_manager.py` - Manages capture threads for physical and video file cameras
- **Logging**: `src/utils/logging/logger.py` - Centralized logger with Colors output

### Frontend Stack
- **Build tool**: Vite with rolldown (ES6 modules)
- **Entry point**: `src/webui/js/main.js`
- **Styling**: Tailwind CSS v4 with `@tailwindcss/vite` plugin
- **3D visualization**: Three.js for robot/field rendering
- **Real-time comms**: Socket.IO client for position updates
- **Directory structure**:
  - `src/webui/js/main.js` - App initialization
  - `src/webui/js/pipeline/` - Flowchart editor (canvas, nodes, connections, drag-drop)
  - `src/webui/js/feeds/` - Camera feed streaming
  - `src/webui/html/` - Handlebars templates and partials
  - `src/webui/css/` - Component styles
  - `src/webui/static/` - Build output

### Key Concepts

#### Pipelines
A `Pipeline` is a chain of operations applied to each camera frame. Each operation's output feeds into the next operation's input. Pipelines are configured in `src/config/pipeline_config.json` with per-camera operation lists. The config includes:
- `action_name` - Operation identifier (e.g., "detect_apriltags")
- `action_params` - Operation-specific parameters
- `position` - x,y coordinates for visual editor placement

#### Operation Injection
Operations can declare constructor parameters `web_interface` or `compute_pool` to receive injected dependencies. Other parameters come from `action_params` in the config.

#### Device Pool
The `ComputePool` manages compute devices and provides polymorphic inference through the `ComputeDevice` interface. Operations request devices by `device_id` (e.g., "CPU", "GPU_0", "MX3_0").

#### Configuration-Driven Architecture
The entire pipeline is driven by JSON configs. Adding new operations requires:
1. Implement operation class in `src/main_operations/definitions/` or `src/secondary_operations/`
2. Create config definition JSON with parameters and validation
3. Add operation entry to `src/config/pipeline_config.json`

## Important Development Rules

### Code Style
- **Python**: Google-style docstrings, type hints for all parameters and return values, descriptive variable names
- **JavaScript**: ES6 modules, functional components
- **No comments**: Self-documenting code preferred over comments
- **Minimal edits**: Only change necessary lines, avoid reprinting unchanged code

### Implementation Standards

#### Creating Pipeline Operations
Follow the "Pipeline Operation Creation" rules in `.cursor/rules/pipeline-operation-creation.mdc`:
- **Main operations**: Use thin-wrapper pattern with definition in `src/main_operations/definitions/{name}.py` delegating to implementation module
- **Secondary operations**: Single file <200 lines in `src/secondary_operations/{name}.py`
- **Config definition**: JSON file at `src/main_operations/definitions/config_data/{name}_config_def.json` or `src/secondary_operations/config_data/{name}_config_def.json`
- **Categories**: Use "prep" (preprocessing), "det" (detection), "proc" (processing), "filt" (filtering), "net" (networking)

#### Run Method Contract
```python
def run(self, input) -> Any:
    """Process input frame/data and return output for next operation."""
```
Common patterns:
- Image/frame: `np.ndarray` → `np.ndarray`
- Detection: `np.ndarray` → `list` of detections
- Pose: `np.ndarray` → 4x4 `np.ndarray` transform matrix

### Always Use
- **context7 MCP**: For library documentation (mandatory for wpilib, Tailwind, etc.)
- **uv**: For Python package management (`uv run`, `uv pip`)
- **Type hints**: All functions must have parameter and return type hints

### Avoid
- New markdown, test, or example files (unless explicitly requested)
- Command-line arguments in code (use configuration variables)
- Automatic Git commits/staging (user must explicitly request)
- Duplicate logic or premature abstractions
- Re-printing unchanged code when editing

## Common File Locations

### Configuration
- `src/config/pipeline_config.json` - Pipeline definition per camera
- `src/general_conf.json` - General settings (network table address)
- `src/webui/web_server.py` - Flask routes and SocketIO handlers

### Operations
- `src/main_operations/definitions/` - Main operation definitions
- `src/main_operations/modules/{category}/{operation}/` - Implementation modules
- `src/secondary_operations/` - Secondary operations

### Frontend
- `src/webui/index.html` - Main HTML template
- `src/webui/js/main.js` - JavaScript entry point
- `src/webui/js/pipeline/` - Pipeline editor components
- `vite.config.js` - Vite build configuration
- `package.json` - npm dependencies

### Documentation
- `docs/overviews/webui/` - WebUI architecture and API docs
- `docs/overviews/device_management_utils/` - Device management docs
- `docs/md_docs/pipeline_docs/` - Pipeline operation documentation
- `.cursor/rules/` - Development rules and patterns

## API Endpoints

### Frontend to Backend
- Base URL: `http://localhost:5001/`
- GET `/get-settings` - Retrieve application settings
- POST `/save-settings` - Update settings
- GET `/camera/<name>` - MJPEG stream
- GET `/camera/<name>/snapshot` - Single frame
- GET `/get-available-cameras` - Camera list
- POST `/restart-backend` - Trigger restart
- GET `/get-pipeline-objects` - Pipeline configuration

### Real-time Communication
- SocketIO: `update_robot_transform` event broadcasts pose updates
- Connection lifecycle managed by server on port 5001

## Performance Considerations

### Pipeline Debugging
- Set `debug_mode = True` in `src/config/utils/pipeline.py` to print per-operation timing statistics including FPS

### Device Selection
- MX3: Lowest latency for compatible models
- GPU: Highest throughput for batch processing
- CPU: Fallback for all scenarios

## Integration Points

The system integrates with:
- **FIRST Robotics NetworkTables**: Via `pynetworktables` for robot communication
- **Camera hardware**: USB cameras (OpenCV), video files (for testing)
- **Compute accelerators**: MemryX MX3, NVIDIA CUDA GPUs, CPU fallback
- **WebUI**: Real-time updates via SocketIO, static assets served from `src/webui/static/`

## Cursor Rules
Key rules are defined in `.cursor/rules/`:
- `always-rules.mdc` - Global development standards
- `pipeline-operation-creation.mdc` - Operation creation patterns
- `webui-rules.mdc` - Frontend development guidelines
- `batched-commits.mdc` - Git commit practices
- `rust-operation-creation.mdc` - Rust operation patterns
