# Ask Mode Rules (Non-Obvious Only)

This file provides guidance for documentation and explanation tasks in this repository.

## Pipeline Configuration is the Source of Truth

The entire system is driven by `src/config/pipeline_config.json`. Understanding this file is critical:
- Each camera has its own operation chain (per-camera configuration)
- Operations are defined by `action_name` (must match Python class name)
- `action_params` provides operation-specific configuration
- `position` field controls visual placement in editor (x, y coordinates)

## Operation Category Meanings

Categories in config files are not just labels:
- `"prep"` - preprocessing operations (frame normalization, resizing)
- `"det"` - detection operations (YOLO, AprilTags, color thresholding)
- `"proc"` - processing operations (pose estimation, filtering)
- `"filt"` - filtering operations (outlier rejection, temporal smoothing)
- `"net"` - networking operations (NetworkTables publishing, output)

These are hardcoded in various parts of the system - using different values breaks operation discovery.

## Dual Configuration System

The WebUI and backend have separate configuration:
- Python backend: `src/config/pipeline_config.json` - defines actual pipeline
- Frontend: WebUI reads this same file via API but also has local UI state
- Backend API base URL is hardcoded: `http://localhost:5001/`

## Three.js Aliases in Vite

Vite is configured with aliases for Three.js modules:
- `three` → `node_modules/three/build/three.module.js`
- `OrbitControls` → Three.js OrbitControls
- `GLTFLoader` → Three.js GLTF loader
- `DRACOLoader` → Three.js Draco loader (for compressed models)

These aliases are in `vite.config.js` - imports use the alias names not full paths.

## Handlebars Template Locations

Frontend templates split across two directories:
- `src/webui/html/tabs/` - tab/page templates
- `src/webui/html/partials/` - reusable component partials
- Vite handlebars plugin configured to look in both directories

## Custom PyPI Indices

Project uses platform-specific dependency installation:
- `pytorch-cuda` index for Windows (CUDA version)
- `pytorch-cpu` index for Linux (CPU fallback version)
- `memryx` index for MemryX accelerator (Linux only)
- These indices must remain in pyproject.toml - resolver depends on them
