from src.utils.colors import Colors
from src.config.utils.operation import Operation


def detect_connection_cycles(operations: dict[str, Operation]) -> None:
    """Detect and report any cycles in the operation connections.

    Uses depth-first search to detect cycles in the operation dependency graph.

    Args:
        operations: Dictionary of all operations in the pipeline.

    Raises:
        ValueError: If cycles are detected in the pipeline.
    """
    visited: set[str] = set()
    recursion_stack: set[str] = set()
    parent_map: dict[str, str | None] = dict.fromkeys(operations)

    def dfs(node_uuid: str) -> bool:
        """Perform DFS to detect cycles.

        Args:
            node_uuid: Current operation UUID being visited.

        Returns:
            True if a cycle is detected, False otherwise.
        """
        visited.add(node_uuid)
        recursion_stack.add(node_uuid)

        operation = operations[node_uuid]
        for output_conn in operation.output_connections:
            neighbor_uuid = output_conn.to_operation.uuid
            parent_map[neighbor_uuid] = node_uuid

            if neighbor_uuid not in visited:
                if dfs(neighbor_uuid):
                    return True
            elif neighbor_uuid in recursion_stack:
                report_cycle(operations, node_uuid, neighbor_uuid, parent_map)
                return True

        recursion_stack.remove(node_uuid)
        return False

    for op_uuid in operations:
        if op_uuid not in visited and dfs(op_uuid):
            raise ValueError("Cyclic dependency detected in pipeline")


def report_cycle(
    operations: dict[str, Operation],
    cycle_start_uuid: str,
    cycle_end_uuid: str,
    parent_map: dict[str, str | None],
) -> None:
    """Report a detected cycle with all connections involved.

    Args:
        operations: Dictionary of all operations.
        cycle_start_uuid: UUID of the operation that created the back edge.
        cycle_end_uuid: UUID of the operation that completes the cycle.
        parent_map: Map of node parents for reconstructing the cycle path.
    """
    cycle_path: list[str] = []
    current = cycle_start_uuid
    while current is not None:
        cycle_path.insert(0, current)
        if current == cycle_end_uuid:
            break
        current = parent_map.get(current)

    cycle_path.append(cycle_end_uuid)

    cycle_connections = []
    for i in range(len(cycle_path) - 1):
        from_uuid = cycle_path[i]
        to_uuid = cycle_path[i + 1]
        from_op = operations[from_uuid]

        for conn in from_op.output_connections:
            if conn.to_operation.uuid == to_uuid:
                cycle_connections.append(
                    {
                        "from": from_op.name,
                        "from_uuid": from_uuid,
                        "from_port": conn.from_port,
                        "to": conn.to_operation.name,
                        "to_uuid": to_uuid,
                        "to_port": conn.to_port,
                        "data_type": conn.data_type,
                    }
                )
                break

    print(Colors.RED + "ERROR: Cyclic dependency detected in pipeline!" + Colors.RESET)
    print(
        Colors.YELLOW
        + "Cycle path: "
        + " -> ".join([operations[uuid].name for uuid in cycle_path])
        + Colors.RESET
    )
    print(Colors.YELLOW + "Connections in cycle:" + Colors.RESET)
    for conn in cycle_connections:
        print(
            Colors.YELLOW
            + f"  {conn['from']} [{conn['from_port']}] -> {conn['to']} [{conn['to_port']}] (type: {conn['data_type']})"
            + Colors.RESET
        )
    raise ValueError("Cyclic dependency detected in pipeline")
