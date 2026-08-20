"""Validation helpers for declared pipeline operation ports."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIRECTORIES = (
    PROJECT_ROOT / "src" / "secondary_operations" / "config_data",
    PROJECT_ROOT / "src" / "main_operations" / "definitions" / "config_data",
)
PortDirection = Literal["input", "output"]


@dataclass(frozen=True, slots=True)
class OperationPorts:
    """Declared runtime ports for one configured operation."""

    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DynamicPortGroup:
    """Canonical interpretation of one definition's dynamic port group."""

    base_name: str | None
    maximum: int | None


def normalize_action_name(action_name: str) -> str:
    """Return the single canonical action name used for lookup and import.

    Definition files are resolved by base name while operations are imported as
    dotted modules, so both consumers must agree on one normalized value.

    Args:
        action_name: Operation filename or normalized action name.

    Returns:
        Action name without a directory prefix or ``.py`` suffix.
    """
    return Path(action_name).name.removesuffix(".py")


def load_operation_config_definition(action_name: str) -> dict[str, Any]:
    """Load an operation's configuration definition by action name.

    Definitions are immutable at runtime, so parsed files are cached and callers
    receive a private copy.

    Args:
        action_name: Operation filename or normalized action name.

    Returns:
        Parsed operation configuration definition.

    Raises:
        ValueError: If no valid definition exists.
    """
    return deepcopy(_read_operation_config_definition(action_name))


@lru_cache(maxsize=None)
def _read_operation_config_definition(action_name: str) -> dict[str, Any]:
    """Read and parse one operation definition from disk exactly once."""
    normalized_name = normalize_action_name(action_name)
    filename = f"{normalized_name}_config_def.json"
    for directory in CONFIG_DIRECTORIES:
        config_path = directory / filename
        if not config_path.is_file():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid operation definition JSON for {normalized_name}: {error}"
            ) from error
        if not isinstance(config, dict):
            raise ValueError(
                f"Operation definition for {normalized_name} must be an object"
            )
        return config
    raise ValueError(f"Operation definition not found for {normalized_name}")


def _static_port_names(
    config_definition: dict[str, Any], direction: PortDirection
) -> tuple[str, ...]:
    """Extract and validate static port names from a definition."""
    raw_nodes = config_definition.get(f"{direction}_nodes", [])
    if not isinstance(raw_nodes, list):
        raise ValueError(f"{direction}_nodes must be a list")

    names: list[str] = []
    for node in raw_nodes:
        name = node.get("name") if isinstance(node, dict) else node
        if not isinstance(name, str) or not name:
            raise ValueError(f"Invalid {direction} port declaration: {node!r}")
        names.append(name)
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate {direction} port declarations: {names!r}")
    return tuple(names)


def resolve_dynamic_port_group(
    config_definition: dict[str, Any], direction: PortDirection
) -> DynamicPortGroup | None:
    """Resolve the canonical dynamic port group for one direction.

    This is the single interpretation of ``dynamic_group`` metadata shared by
    pipeline runtime validation and the WebUI configuration endpoints.

    Args:
        config_definition: Parsed operation configuration definition.
        direction: Port direction to resolve.

    Returns:
        The resolved group, or ``None`` when the direction has no dynamic ports.

    Raises:
        ValueError: If a declared maximum is not a positive integer.
    """
    dynamic_group = config_definition.get("dynamic_group")
    if not isinstance(dynamic_group, dict):
        return None

    if direction == "input":
        enabled = dynamic_group.get("input_dynamic_group", True) is not False
        base_name_keys = ("input_base_name", "input_prefix")
        maximum_key = "max_inputs"
    else:
        enabled = bool(
            dynamic_group.get("output_dynamic_group", False)
            or dynamic_group.get("mirrored_output_group", False)
        )
        base_name_keys = ("output_base_name", "output_prefix")
        maximum_key = "max_outputs"
    if not enabled:
        return None

    base_name = next(
        (
            value
            for key in base_name_keys
            if isinstance((value := dynamic_group.get(key)), str) and value
        ),
        None,
    )
    static_names = _static_port_names(config_definition, direction)
    if base_name is None and static_names:
        base_name = static_names[-1]
    if base_name is None:
        return None

    raw_maximum = dynamic_group.get(maximum_key)
    maximum: int | None
    if raw_maximum in (None, "unlimited"):
        maximum = None
    else:
        try:
            maximum = int(raw_maximum)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid maximum for dynamic {direction} ports: {raw_maximum!r}"
            ) from error
        if maximum < 1:
            raise ValueError(f"Dynamic {direction} port maximum must be positive")
    return DynamicPortGroup(base_name=base_name, maximum=maximum)


def _is_dynamic_port(port_name: str, dynamic_group: DynamicPortGroup | None) -> bool:
    """Return whether a port matches a definition's indexed dynamic group."""
    if dynamic_group is None or dynamic_group.base_name is None:
        return False
    match = re.fullmatch(rf"{re.escape(dynamic_group.base_name)}_(\d+)", port_name)
    if match is None:
        return False
    index = int(match.group(1))
    return index >= 1 and (
        dynamic_group.maximum is None or index <= dynamic_group.maximum
    )


def validate_declared_port(
    config_definition: dict[str, Any],
    direction: PortDirection,
    port_name: str,
) -> None:
    """Validate one connection port against an operation definition."""
    static_names = _static_port_names(config_definition, direction)
    dynamic_group = resolve_dynamic_port_group(config_definition, direction)
    dynamic_base = dynamic_group.base_name if dynamic_group else None
    effective_static_names = tuple(
        name for name in static_names if name != dynamic_base
    )
    if port_name in effective_static_names or _is_dynamic_port(
        port_name, dynamic_group
    ):
        return
    raise ValueError(
        f"Unknown {direction} port {port_name!r}; declared static ports are "
        f"{effective_static_names!r}"
    )


def resolve_operation_ports(
    config_definition: dict[str, Any],
    input_connection_ports: list[str],
    output_connection_ports: list[str],
) -> OperationPorts:
    """Resolve concrete static and connected dynamic ports for one operation."""
    resolved: dict[PortDirection, tuple[str, ...]] = {}
    for direction, connected_ports in (
        ("input", input_connection_ports),
        ("output", output_connection_ports),
    ):
        static_names = _static_port_names(config_definition, direction)
        dynamic_group = resolve_dynamic_port_group(config_definition, direction)
        dynamic_base = dynamic_group.base_name if dynamic_group else None
        names = [name for name in static_names if name != dynamic_base]
        for port_name in connected_ports:
            validate_declared_port(config_definition, direction, port_name)
            if port_name not in names:
                names.append(port_name)
        resolved[direction] = tuple(names)
    return OperationPorts(inputs=resolved["input"], outputs=resolved["output"])


def validate_pipeline_connections(
    pipeline_config: list[dict[str, Any]],
) -> dict[str, OperationPorts]:
    """Validate operation declarations and every serialized connection.

    Args:
        pipeline_config: One pipeline's serialized operations.

    Returns:
        Concrete ports keyed by operation UUID.

    Raises:
        ValueError: If operations, declarations, or connections are malformed.
    """
    if not isinstance(pipeline_config, list):
        raise ValueError("Pipeline configuration must be a list of operations")

    operations: dict[str, dict[str, Any]] = {}
    definitions: dict[str, dict[str, Any]] = {}
    connections: list[dict[str, Any]] = []
    for operation in pipeline_config:
        if not isinstance(operation, dict):
            raise ValueError("Every pipeline operation must be an object")
        operation_uuid = operation.get("uuid")
        action_name = operation.get("action_name")
        if not isinstance(operation_uuid, str) or not operation_uuid:
            raise ValueError("Every pipeline operation requires a non-empty UUID")
        if operation_uuid in operations:
            raise ValueError(f"Duplicate operation UUID: {operation_uuid}")
        if not isinstance(action_name, str) or not action_name:
            raise ValueError(f"Operation {operation_uuid} has no action_name")
        # Definitions are resolved by base name while runtime instantiation
        # imports a dotted module, so a path-qualified action name would pass
        # validation and then fail to start.
        if action_name.removesuffix(".py") != normalize_action_name(action_name):
            raise ValueError(
                f"Operation {operation_uuid} action_name must not contain a path: "
                f"{action_name!r}"
            )
        operations[operation_uuid] = operation
        definitions[operation_uuid] = load_operation_config_definition(action_name)
        raw_connections = operation.get("connections", [])
        if not isinstance(raw_connections, list):
            raise ValueError(
                f"Connections for operation {operation_uuid} must be a list"
            )
        connections.extend(raw_connections)

    input_ports_by_uuid: dict[str, list[str]] = {
        operation_uuid: [] for operation_uuid in operations
    }
    output_ports_by_uuid: dict[str, list[str]] = {
        operation_uuid: [] for operation_uuid in operations
    }
    occupied_inputs: set[tuple[str, str]] = set()

    for connection in connections:
        if not isinstance(connection, dict):
            raise ValueError("Every connection must be an object")
        required_fields = (
            "from_uuid",
            "from_port",
            "to_uuid",
            "to_port",
            "data_type",
        )
        if any(
            not isinstance(connection.get(field), str) or not connection[field]
            for field in required_fields
        ):
            raise ValueError(f"Malformed connection: {connection!r}")
        from_uuid = connection["from_uuid"]
        to_uuid = connection["to_uuid"]
        from_port = connection["from_port"]
        to_port = connection["to_port"]
        if from_uuid not in operations:
            raise ValueError(f"Connection source operation not found: {from_uuid}")
        if to_uuid not in operations:
            raise ValueError(f"Connection destination operation not found: {to_uuid}")
        validate_declared_port(definitions[from_uuid], "output", from_port)
        validate_declared_port(definitions[to_uuid], "input", to_port)
        occupied_input = (to_uuid, to_port)
        if occupied_input in occupied_inputs:
            raise ValueError(
                f"Input port {to_uuid}:{to_port} has more than one connection"
            )
        occupied_inputs.add(occupied_input)
        output_ports_by_uuid[from_uuid].append(from_port)
        input_ports_by_uuid[to_uuid].append(to_port)

    _validate_docking_relationships(operations, definitions, connections)

    return {
        operation_uuid: resolve_operation_ports(
            definitions[operation_uuid],
            input_ports_by_uuid[operation_uuid],
            output_ports_by_uuid[operation_uuid],
        )
        for operation_uuid in operations
    }


def _validate_docking_relationships(
    operations: dict[str, dict[str, Any]],
    definitions: dict[str, dict[str, Any]],
    connections: list[dict[str, Any]],
) -> None:
    """Validate direct, exclusive source bindings for docked operations."""
    source_owners: set[str] = set()
    for operation_uuid, definition in definitions.items():
        docking = definition.get("docking")
        if not isinstance(docking, dict):
            continue
        source_action = docking.get("source_action")
        source_port = docking.get("source_port")
        target_port = docking.get("target_port")
        matching = [
            connection
            for connection in connections
            if connection.get("to_uuid") == operation_uuid
            and connection.get("to_port") == target_port
        ]
        if len(matching) != 1:
            raise ValueError(
                f"Docked operation {operation_uuid} must bind directly to one "
                f"{source_action}:{source_port} output"
            )
        connection = matching[0]
        source_uuid = connection["from_uuid"]
        actual_action = normalize_action_name(
            str(operations[source_uuid]["action_name"])
        )
        if (
            actual_action != source_action
            or connection.get("from_port") != source_port
            or connection.get("is_default", False)
        ):
            raise ValueError(
                f"Docked operation {operation_uuid} requires a non-default "
                f"{source_action}:{source_port} connection"
            )
        if source_uuid in source_owners:
            raise ValueError(
                f"Device Input {source_uuid} already has a docked asynchronous operation"
            )
        source_owners.add(source_uuid)
