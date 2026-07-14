# GetNetworktablesValue Operation

## Overview

`GetNetworktablesValue` is a source operation that reads one **double** from the injected NetworkTables table and returns it to downstream operations. It ignores its input.

The backend injects the `EagleEye` table by default. A relative key such as `robot_speed` addresses `/EagleEye/robot_speed`. The operation passes `network_table_key` directly to `network_table.getEntry()` without normalizing it.

## Configuration

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `network_table_key` | `str` | Yes | Entry key passed to `network_table.getEntry()`. |

```json
{
  "action_name": "get_networktables_value.py",
  "action_params": {
    "network_table_key": "shooter_angle"
  }
}
```

## Behavior

On every run the operation calls `getDouble(NaN)` on the configured entry:

- returns the numeric value when the entry contains a double;
- returns `None` when the value is absent or cannot be read as a double;
- has no timing metadata and does not expose the NetworkTables entry timestamp.

It currently does **not** support arrays, strings, booleans, raw topics, WPILib structs (`Pose2d`/`Pose3d`), or `getAtomic()` timestamped reads. It should not be used to ingest a timestamped vision pose or other typed robot measurement.

## Files

- Implementation: `src/secondary_operations/get_networktables_value.py`
