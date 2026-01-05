# Debug Mode Rules (Non-Obvious Only)

This file provides guidance for debugging tasks in this repository.

## Pipeline Timing Debug

Set `debug_mode = True` in `src/config/utils/pipeline.py` to enable per-operation timing statistics including FPS. This is not printed by default even with standard logging enabled.

## Operation Injection Failures

Operations silently fail to inject if parameter names don't match exactly:
- Parameter must be named exactly `web_interface` (not `web_interface_instance`, `interface`, etc.) to receive injection
- Parameter must be named exactly `compute_pool` (not `device_pool`, `pool`, etc.) to receive device manager
- Misspelled names fall back to trying `action_params` - causes confusing "missing parameter" errors instead of immediate injection failures

## Configuration File Location Errors

Operations won't load if config file is in wrong location:
- Main operations: MUST be at `src/main_operations/definitions/config_data/{name}_config_def.json` (not other subdirectories)
- Secondary operations: MUST be at `src/secondary_operations/config_data/{name}_config_def.json`
- Path must match operation class name exactly (case-sensitive)

## Vite Dev Server Caching

When debugging frontend changes:
- Vite dev server excludes `static/**` from file watch (see vite.config.js)
- Changes to `src/webui/static/` won't trigger rebuild - these are built artifacts
- Always edit source files in `src/webui/js/`, `src/webui/css/`, `src/webui/html/` not the static directory
- Production build outputs here: rebuild with `npm run build` then restart Flask

## Device Pool Silent Failures

`ComputePool.get_compute_device(device_id)` can silently return wrong device:
- Device IDs like "GPU_0", "MX3_0", "CPU" must match what's available
- Check available devices in `src/utils/device_management_utils/` initialization
- Invalid device_id doesn't error - falls back to CPU silently

## Port Conflicts

- Flask backend: port 5001 (hardcoded, check `src/main_backend.py`)
- Vite dev server: port 5173 (default, check package.json dev script)
- Both must be available or server startup will fail
