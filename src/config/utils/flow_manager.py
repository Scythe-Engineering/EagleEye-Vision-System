from src.config.utils.pipeline import Operation


class FlowManager:
    def __init__(self, operations: dict[str, Operation]):
        self.operations = operations
        self.start_operation = self._find_start_operation()
        
    def forward_pass_operation_order(self) -> list[list[str]]:
        """Returns the order of operations for the forward pass."""
        
        first_connections = self.start_operation["connections"]
        
    def _find_start_operation(self) -> Operation:
        """Finds the starting operation in the flow, always is the device_input operation name."""
        for uuid, operation_data in self.operations.items():
            if operation_data.get("name") == "device_input":
                return self.operations[uuid]
        raise ValueError("No starting operation (device_input) found in operations.")
        