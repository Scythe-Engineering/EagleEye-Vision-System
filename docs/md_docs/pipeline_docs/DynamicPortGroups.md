# Dynamic port groups

Dynamic groups let the pipeline editor add indexed ports as connections are made. At runtime they are ordinary dictionary keys such as `pose_1` and `pose_2`.

## Configuration

Keep a template port in `input_nodes` or `output_nodes`, then add `dynamic_group`:

```json
{
  "input_nodes": [{"name": "pose", "has_default": false}],
  "output_nodes": ["fused_pose"],
  "dynamic_group": {
    "input_base_name": "pose",
    "input_dynamic_group": true,
    "output_dynamic_group": false,
    "max_inputs": 16
  }
}
```

The editor shows one unused indexed port until it reaches the configured maximum. Removing connections lets the group shrink.

| Field | Meaning |
| --- | --- |
| `input_base_name` | Prefix for indexed inputs. The legacy `input_prefix` spelling is also accepted. |
| `output_base_name` | Prefix for indexed outputs. The legacy `output_prefix` spelling is also accepted. |
| `max_inputs` | Highest allowed input index. Use a positive integer. |
| `max_outputs` | Highest allowed output index. Use a positive integer. |
| `input_dynamic_group` | Enables indexed inputs. Defaults to `true` when a group exists. |
| `output_dynamic_group` | Enables indexed outputs. |
| `mirrored_output_group` | Enables dynamic outputs and defaults input-output coupling to on. |
| `coupled_groups` | Resizes input and output groups together. |

Indexed ports start at 1. Names such as `pose_0`, bare `pose`, and values beyond a numeric maximum are not valid connection ports.

## Runtime contract

Read dynamic inputs by prefix:

```python
poses = [
    value
    for name, value in sorted(input_data.items())
    if name.startswith("pose_") and value is not None
]
```

For a mirrored transform, preserve the index in the output name. An input named `pose_3` should produce `mirrored_pose_3`. Return a dictionary keyed by the concrete output names.

## Important limitations

Dynamic metadata controls editor connections and port validation. It does not validate payload types or shapes. The operation must reject or skip bad values itself. Template base ports belong in the config definition but are not rendered as ordinary static ports on an enabled dynamic side. Avoid the legacy `"unlimited"` maximum in new definitions because the WebUI normalizes a nonnumeric maximum to 1.
