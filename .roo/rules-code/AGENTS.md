# Code Mode Rules (Non-Obvious Only)

This file provides guidance for coding tasks in this repository.

## Operation Injection Pattern

Constructor parameter NAMES (not type hints) determine what gets injected:
- Parameter named `web_interface` → automatically receives `EagleEyeInterface` instance
- Parameter named `compute_pool` → automatically receives `ComputePool` instance
- All other constructor parameters must come from `action_params` in pipeline config

This is discovered by reading the pipeline initialization code - not obvious from the operation class itself.

## Thin-Wrapper Pattern for Main Operations

Operations >200 lines MUST split into two files:
- `src/main_operations/definitions/{name}.py` - thin wrapper that receives injected deps and resolves config params
- `src/main_operations/modules/{category}/{name}/implementation.py` - actual logic

The wrapper instantiates the implementation class and delegates the `run()` method. This pattern is not enforced by linters.

## Configuration-Driven Everything

- Pipeline operations MUST have matching JSON config at `src/main_operations/definitions/config_data/{name}_config_def.json`
- Config files define parameter types, defaults, and validation - without this the operation won't load
- Categories must be exactly one of: `"prep"`, `"det"`, `"proc"`, `"filt"`, `"net"` (not other names)

## Package Manager: uv (Not pip)

- Use `uv sync` (not `pip install`) - manages custom PyPI indices
- Custom indices defined in pyproject.toml for pytorch (two variants: cuda vs cpu) and memryx (Linux only)
- These custom indices are critical for correct dependency resolution

## WebUI Build Integration

- Frontend output goes to `src/webui/static/` (Flask serves from this directory)
- Vite root is `src/webui/` not project root - all relative paths in vite.config.js are relative to this directory
- Must run `npm run build` before Flask server can serve the pipeline editor UI
- Vite dev server runs on port 5173 by default, backend on 5001

## Code Style

- **Python**: Black formatting, mandatory type hints, Google-style docstrings, architectural comments encouraged for complex blocks and non-obvious design decisions; avoid frivolous line-by-line explanations and prioritize readable code over comment band-aids
