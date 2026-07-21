# UpdateAttributeWithNetworktables

## Status: not implemented

There is no `update_attribute_with_networktables.py` operation in `src/secondary_operations/`, no registered operation definition, and no supported pipeline configuration for this feature.

The project currently provides `GetNetworktablesValue` for reading a scalar double and `PublishToNetworktables` for publishing pipeline output. Neither operation dynamically modifies attributes on another pipeline operation.

Do not add `update_attribute_with_networktables` to a pipeline configuration. Any older descriptions of reflection-based live attribute updates, naming conversion, or configuration examples describe a proposal rather than executable code.
