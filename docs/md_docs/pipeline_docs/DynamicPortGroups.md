# Dynamic Port Groups in Operation Configs

This guide explains how to configure dynamic input/output groups in operation
`*_config_def.json` files, and how runtime behavior works in the pipeline UI.

## Where this is configured

Define dynamic groups inside your operation config JSON using a
`dynamic_group` object.

- Main operations:
  `src/main_operations/definitions/config_data/{operation_name}_config_def.json`
- Secondary operations:
  `src/secondary_operations/config_data/{operation_name}_config_def.json`

The backend normalizes these fields in
[`_normalize_dynamic_group_config()`](src/webui/web_server.py:1298), and the
node UI applies them in
[`normalizeDynamicGroup()`](src/webui/js/pipeline/flowchartNode.js:69).

## Dynamic group schema

```json
{
    "dynamic_group": {
        "input_base_name": "pose",
        "output_base_name": "mirrored_pose",
        "max_inputs": 8,
        "max_outputs": 8,
        "mirrored_output_group": true,
        "coupled_groups": true,
        "output_dynamic_group": false,
        "input_dynamic_group": true
    }
}
```

### Field meanings

- `input_base_name` (string)
  - Base name for dynamic input ports (`pose_1`, `pose_2`, ...)
- `output_base_name` (string)
  - Base name for dynamic output ports (`mirrored_pose_1`, ...)
- `max_inputs` (int, >= 1)
  - Maximum dynamic input ports
- `max_outputs` (int, >= 1)
  - Maximum dynamic output ports (defaults to `max_inputs`)
- `mirrored_output_group` (bool)
  - Enables dynamic output group support for mirrored workflows
- `output_dynamic_group` (bool)
  - Enables dynamic output group even when not mirrored
- `input_dynamic_group` (bool)
  - Enables/disables dynamic input group (default: `true`)
- `coupled_groups` (bool)
  - If `true`, dynamic input and output groups resize together (mirror behavior
    from either side)
  - If omitted, defaults to `mirrored_output_group`

## Runtime behavior

The flowchart keeps one extra open slot while growing:

- If `N` ports are connected on a dynamic side, the UI shows `N + 1` ports
  (until max is reached)
- If connections are removed, the group shrinks accordingly

Behavior is implemented in
[`syncDynamicPorts()`](src/webui/js/pipeline/flowchartNode.js:236) and
[`ensureDynamicPortsForConnectionPort()`](src/webui/js/pipeline/flowchartNode.js:325).

### How this appears in `run()` at runtime

When an operation executes, dynamic ports are passed in the same `input_data`
dictionary as static ports. The keys are concrete port names such as:

- Static input: `reference_pose`
- Dynamic inputs: `pose_1`, `pose_2`, `pose_3`, ...

If dynamic output is enabled, your operation should return concrete output keys
as well:

- Static output: `status`
- Dynamic outputs: `mirrored_pose_1`, `mirrored_pose_2`, ...

In practice, dynamic behavior is not a special runtime type. It is naming +
convention, controlled by config and honored by the WebUI + pipeline runtime.

## Authoring operations with dynamic ports

This is the part most people need when implementing real operations.

### 1) Keep static template ports in config

Always include the base/template port names in `input_nodes`/`output_nodes`, even
when the side is dynamic:

- Dynamic input base `pose` should still exist in `input_nodes`
- Dynamic output base `mirrored_pose` should still exist in `output_nodes`

These static template entries are used to derive base names and render behavior.

### 2) Parse dynamic keys by prefix in `run()`

Typical pattern:

```python
dynamic_entries = [
    (port_name, value)
    for port_name, value in input_data.items()
    if port_name.startswith("pose_")
]
```

Then validate/convert each value before processing.

### 3) Emit outputs using matched index tokens

If input is `pose_7`, output should usually be `mirrored_pose_7` (or same index
for your chosen output base):

```python
index_token = port_name.split("pose_", 1)[-1]
output_name = f"mirrored_pose_{index_token}"
```

This preserves deterministic wiring behavior and makes debugging easier.

### 4) Handle sparse / invalid entries defensively

Real pipelines may produce:

- missing indices (e.g., `pose_1`, `pose_3` but no `pose_2`)
- `None` values
- invalid matrix/array shapes

Best practice: skip invalid dynamic entries rather than failing the whole
operation, unless fail-fast is explicitly desired.

### 5) Return `None` only when there is no usable result

For dynamic reducers/transformers, common pattern:

- build `outputs` dict
- return `outputs or None`

This avoids emitting empty payloads while still allowing partial success.

## End-to-end example (secondary operation)

The operation
[`DynamicPoseGroupTest`](src/secondary_operations/dynamic_pose_group_test.py:10)
is a concrete reference implementation:

- static input: `reference_pose`
- dynamic input base: `pose`
- dynamic output base: `mirrored_pose`
- coupled mirrored behavior configured in
  [`dynamic_pose_group_test_config_def.json`](src/secondary_operations/config_data/dynamic_pose_group_test_config_def.json)

Key implementation points:

- reads dynamic inputs via prefix filtering in
  [`run()`](src/secondary_operations/dynamic_pose_group_test.py:19)
- converts each candidate with
  [`_as_pose()`](src/secondary_operations/dynamic_pose_group_test.py:57)
- computes relative transforms with
  [`_offset_from_reference()`](src/secondary_operations/dynamic_pose_group_test.py:75)
- skips non-invertible reference cases (returns `None`) instead of crashing

## Main operation pattern (wrapper + implementation)

For large operations, dynamic logic belongs in the implementation module, not the
thin definition wrapper.

- Wrapper file should only resolve injected deps/config and delegate `run()`
- Implementation file should parse dynamic keys and build dynamic outputs

This keeps the operation architecture consistent with project standards while
still supporting dynamic ports.

## Naming rules that prevent breakage

1. `input_base_name` must match the prefix your code reads (`pose_` pattern)
2. `output_base_name` must match the prefix your code emits (`mirrored_pose_`)
3. If these drift apart, UI connections may look valid but runtime keys will not
   match what your code expects

## Coupled vs non-coupled behavior in operation design

- `coupled_groups: true`
  - Good for 1:1 transforms (each input produces one output)
  - Example: pose transforms, per-camera filtered stream, per-target confidence
    cleanup
- `coupled_groups: false`
  - Good when input/output fanout differs
  - Example: one dynamic input set reduced into one static output, or one static
    input expanded into many dynamic outputs

Choose based on data-shape semantics, not UI convenience.

## Debugging checklist

If dynamic ports are not behaving as expected:

1. Verify `dynamic_group` exists and is valid in the operation config file.
2. Verify template base ports are present in `input_nodes`/`output_nodes`.
3. Confirm runtime keys in `input_data` actually use the expected prefixes.
4. Confirm your operation emits output keys that match `output_base_name`.
5. Check normalization path in
   [`_normalize_dynamic_group_config()`](src/webui/web_server.py:1298).
6. Check node-side resizing logic in
   [`syncDynamicPorts()`](src/webui/js/pipeline/flowchartNode.js:236).

## Common patterns

### 1) Dynamic inputs only

```json
{
    "input_nodes": [{ "name": "pose", "has_default": false }],
    "output_nodes": ["pose_avg"],
    "dynamic_group": {
        "input_base_name": "pose",
        "max_inputs": 16,
        "input_dynamic_group": true,
        "output_dynamic_group": false,
        "mirrored_output_group": false
    }
}
```

Use this for reducers like averaging/merging operations.

### 2) Mirrored coupled input/output groups (bidirectional coupling)

```json
{
    "input_nodes": [
        { "name": "reference_pose", "has_default": false },
        { "name": "pose", "has_default": false }
    ],
    "output_nodes": ["mirrored_pose"],
    "dynamic_group": {
        "input_base_name": "pose",
        "output_base_name": "mirrored_pose",
        "max_inputs": 8,
        "max_outputs": 8,
        "mirrored_output_group": true,
        "coupled_groups": true
    }
}
```

With `coupled_groups: true`, connecting either side expands both sides.

### 3) Standalone dynamic outputs (not coupled to inputs)

```json
{
    "input_nodes": ["trigger"],
    "output_nodes": ["stream"],
    "dynamic_group": {
        "input_dynamic_group": false,
        "output_dynamic_group": true,
        "output_base_name": "stream",
        "max_outputs": 12,
        "coupled_groups": false
    }
}
```

This gives a dynamic output-only group with normal grow/shrink logic.

## Notes and gotchas

- Dynamic metadata is additive to static ports.
- Base names must align with your operation’s expected runtime key naming.
- If `input_dynamic_group` is `false`, input base name is ignored by the UI.
- If dynamic outputs are enabled, the static output with the same base name is
  treated as the template and excluded from static rendering.
- `max_inputs` / `max_outputs` are UI and connection constraints; your operation
  code should still validate incoming values.
- Prefer predictable index-preserving outputs (`*_N` in, `*_N` out) for easier
  graph reasoning and downstream debugging.
