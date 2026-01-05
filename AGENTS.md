# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Build & Test Commands

**Frontend (WebUI)**:
- `npm install` → `npm run build` → outputs to `src/webui/static/` (Flask serves from here)
- WebUI MUST be built before Flask server can serve pipeline editor UI; without it only static endpoints work
- `npm run dev` runs Vite dev server (root: `src/webui`, not project root)
- Vite excludes `static/**` and `web_server.py` from file watch

**Backend**:
- `uv sync` (not pip) - manages custom PyPI indices for pytorch and memryx
- `python src/main_backend.py` starts Flask on port 5001

## Non-Obvious Code Patterns

**Operation Injection**: Constructor parameter NAMES determine injection:
- Include parameter named `web_interface` → receives `EagleEyeInterface`
- Include parameter named `compute_pool` → receives `ComputePool`
- All other parameters come from `action_params` in pipeline config

**Operation Structure**:
- Main operations (>200 lines): thin-wrapper in `src/main_operations/definitions/{name}.py` → delegates to `src/main_operations/modules/{category}/{name}/implementation.py`
- Secondary operations (<200 lines): single file in `src/secondary_operations/{name}.py`
- Run method: `def run(self, input) -> Any:` (input/output types vary by operation category)

**Configuration-Driven**:
- Entire pipeline driven by `src/config/pipeline_config.json`
- Operations need matching JSON config file at `src/main_operations/definitions/config_data/{name}_config_def.json`
- Categories: `"prep"`, `"det"`, `"proc"`, `"filt"`, `"net"`

## Code Style Rules

- **Black-style Python** formatting (from .github/copilot-instructions.md)
- **Type hints mandatory** on all functions (parameters and return value)
- **Google-style docstrings** for all functions
- **No comments** - use descriptive variable names and function extraction instead
- **Prettier.js** with tabWidth: 4 (from package.json)

## Critical Integration Points

- Flask serves WebUI from `src/webui/static/` → requires `npm run build` to populate
- Vite root is `src/webui/` (not project root) - CSS/JS paths are relative to this
- Handlebars partials in: `src/webui/html/tabs/` and `src/webui/html/partials/`
- Backend API: `http://localhost:5001/`
- SocketIO for real-time pose updates
- Three.js for 3D visualization (configured via aliases in vite.config.js)
