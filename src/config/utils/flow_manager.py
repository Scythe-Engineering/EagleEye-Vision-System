from src.config.utils.pipeline import Operation, Connection


class FlowManager:
    def __init__(self, operations: dict[str, Operation]):
        self.operations = operations
        self.start_operation = self._find_start_operation()

    def forward_pass_operation_order(self) -> list[list[str]]:
        """Returns the starting execution time of each operation in the flow. Returned as execution groups."""

        visited_operations: list[str] = []
        first_connections: list[Connection] = self.start_operation["connections"]

        for connecion in first_connections:
            next_operation_uuid = connecion["to_uuid"]
            if next_operation_uuid not in visited_operations:
                visited_operations.append(next_operation_uuid)

    def _find_start_operation(self) -> Operation:
        """Finds the starting operation in the flow, always is the device_input operation name."""
        for uuid, operation_data in self.operations.items():
            if operation_data.get("name") == "device_input":
                return self.operations[uuid]
        raise ValueError("No starting operation (device_input) found in operations.")
