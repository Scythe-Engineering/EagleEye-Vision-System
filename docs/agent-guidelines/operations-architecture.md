# Operations Architecture

## Injection pattern (constructor parameter names)

Injection is determined by constructor **parameter names** (not type hints):

- `web_interface` → inject `EagleEyeInterface`
- `compute_pool` → inject `ComputePool`
- All other constructor parameters must come from `action_params` in pipeline
  config.

## Operation structure by size

### Main operations (>200 lines)

Use thin-wrapper split pattern:

- Wrapper: `src/main_operations/definitions/{name}.py`
- Implementation:
  `src/main_operations/modules/{category}/{name}/implementation.py`

Wrapper responsibilities:

- Receive injected dependencies
- Resolve configuration parameters
- Instantiate implementation class
- Delegate `run()` execution

### Secondary operations (<200 lines)

- Single file implementation in `src/secondary_operations/{name}.py`

## Configuration-driven loading

- Pipeline source of truth: `src/config/pipeline_config.json`
- Each operation must have a config definition file at:
  `src/main_operations/definitions/config_data/{name}_config_def.json`
- Config definitions must include parameter types/defaults/validation.
- Valid operation categories are exactly:
  `"prep"`, `"det"`, `"proc"`, `"filt"`, `"net"`.

## Run contract

- Operations should expose a `run()` method.
- Input/output types vary by operation category and pipeline stage.
