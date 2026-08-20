# GetNetworktablesValue

`GetNetworktablesValue` reads one double from the pipeline's injected NetworkTable.

## Inputs

None. This is a data-source operation.

## Outputs

`data` is the current double. A missing entry, or one read as `NaN`, produces `None`.

## When to use

Use this to feed a scalar robot value into a pipeline.

## Configuration

- `network_table_key`: Required entry key within the injected table. It can change at runtime.

## Limitations

This operation reads only doubles, not strings, booleans, arrays, or WPILib structs.
