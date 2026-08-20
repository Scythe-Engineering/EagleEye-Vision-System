# PublishToNetworktables

`PublishToNetworktables` publishes an upstream value to a native NetworkTables primitive or WPILib struct topic, then passes the original input downstream.

## Inputs

`data` may be a primitive, primitive array, supported pose dictionary, 4 by 4 NumPy pose matrix, or a sequence of supported values. `data_path` can select a nested value first.

## Outputs

The unchanged `data`. Unsupported, missing, `None`, and empty-array selections are not published.

## When to use

Use this when the robot needs a pipeline result through NetworkTables. Choose an explicit schema when automatic conversion is ambiguous.

## Configuration

- `target_key`: Required topic key.
- `schema`: Conversion hint, default `auto`. Supported hints are `double`, `float`, `number`, `boolean`, `bool`, `string`, `double_array`, `float_array`, `number_array`, `pose2d`, `pose3d`, `transform2d`, `transform3d`, `translation2d`, `translation3d`, `rotation2d`, and `rotation3d`.
- `data_path`: Optional keys or indices, supplied as a dotted string or list. One field name on a sequence extracts that field from matching dictionaries.

## Timestamp behavior

Timed values publish with their `capture_nt_us` NetworkTables timestamp. Nested selection preserves that timing. Untimed values use publish time. Robot-side pose estimators need capture time rather than later processing time.

The first publish fixes the topic type. Changing `target_key` creates a new publisher. Changing `schema` alone does not recreate one already in use.
