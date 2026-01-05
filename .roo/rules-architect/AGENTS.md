# Architect Mode Rules (Non-Obvious Only)

This file provides guidance for architecture and planning tasks in this repository.

## Pipeline Architecture is Configuration-Driven

The entire system follows a configuration-driven architecture pattern:
- `src/config/pipeline_config.json` defines per-camera operation chains
- Operations are instantiated by name from class registry (based on `action_name` in config)
- Each operation receives injected dependencies (`web_interface`, `compute_pool`) by constructor parameter names
- Pipeline orchestration handles chaining - each operation's output becomes next operation's input

This is fundamentally different from typical hardcoded pipeline architectures.

## Dual Process Architecture (Python + JavaScript)

System splits into two independent processes:
- **Backend**: `python src/main_backend.py` - Flask server on port 5001, runs pipelines, serves WebUI static files
- **Frontend**: `npm run dev` (Vite dev server on 5173) or `npm run build` (static build to `src/webui/static/`)

WebUI must be built (`npm run build`) before Flask can serve the editor UI. Without built artifacts, only API endpoints work.

## Device Pool Polymorphism

`ComputePool` provides polymorphic inference across heterogeneous devices:
- `CPU` - always available fallback
- `GPU_*` - NVIDIA CUDA GPUs (detected at runtime)
- `MX3_*` - MemryX MX3 accelerators (Linux only)
- Operations request devices by ID string - device swap is configuration-only

Device selection affects latency more than correctness - system never breaks if device unavailable (falls back to CPU).

## Thin-Wrapper Pattern for Code Organization

Main operations use mandatory separation:
- Wrapper (`src/main_operations/definitions/{name}.py`) - thin layer for dependency resolution
- Implementation (`src/main_operations/modules/{category}/{name}/implementation.py`) - actual logic
- This isn't about code reuse - it's about separating DI concerns from business logic

Secondary operations (<200 lines) are single files in `src/secondary_operations/` - no wrapper required.

## Real-Time Communication via SocketIO

Frontend and backend share real-time state through SocketIO:
- Backend broadcasts `update_robot_transform` events (pose updates)
- Frontend connects to same Flask server (port 5001)
- This enables live 3D pose visualization without polling

## Three-Layer Frontend Architecture

WebUI organized in three independent layers:
- **Styling**: Tailwind CSS v4 via `@tailwindcss/vite` plugin (not PostCSS)
- **Templating**: Handlebars templates in `src/webui/html/` with partials from two directories
- **Application**: ES6 modules in `src/webui/js/` (main.js entry point)
- **Build Output**: Vite builds to `src/webui/static/` (Flask serves from here)

Vite root is `src/webui/` not project root - all relative paths are relative to this.

## Custom PyPI Index Strategy

Project uses platform-specific and accelerator-specific indices:
- Different PyTorch versions for Windows (CUDA) vs Linux (CPU)
- MemryX package from proprietary index (Linux only)
- uv resolver handles index selection based on `sys_platform` markers in pyproject.toml

This is critical for installation across different target platforms.
