from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.config.utils.line_profiling import line_profiling_manager
from src.main_operations.definitions.base.base_class import OperationInstance
from src.utils.timing import attach_output_timing, unwrap_timed_deep

if TYPE_CHECKING:
    from src.config.utils.thread_object import ThreadObject


class Operation:
    def __init__(self, instance: OperationInstance, uuid: str, name: str, is_data_source: bool = False) -> None:
        """Initialize the object.
        
        Args:
            instance (OperationInstance): Instance.
            uuid (str): Uuid.
            name (str): Name.
            is_data_source (bool): Is data source."""
        self.instance: OperationInstance = instance
        self.uuid: str = uuid
        self.name: str = name
        self.is_data_source: bool = is_data_source

        self.input_connections: list[Connection] = []
        self.output_connections: list[Connection] = []
        self.assigned_thread_object: ThreadObject | None = None

        self.has_output_connections: bool = False
        self.has_input_connections: bool = False

        self.execution_timestep: int | None = None
        self.finish_timestep: int | None = None

    def run(self, input_data: Any) -> Any:
        """Run.
        
        Args:
            input_data (Any): Input data.
        
        Returns:
            Any: Result of run."""
        call_input = (
            input_data
            if getattr(self.instance, "uses_timed_inputs", False)
            else unwrap_timed_deep(input_data)
        )

        if line_profiling_manager.is_active_for(self.uuid):
            output = line_profiling_manager.profile_operation_call(
                operation=self,
                call=lambda: self.instance.run(call_input),
            )
        else:
            output = self.instance.run(call_input)
        return attach_output_timing(output, input_data)

    def is_only_input_connection(self, uuid: str) -> bool:
        """Is only input connection.
        
        Args:
            uuid (str): Uuid.
        
        Returns:
            bool: Result of is only input connection."""
        if len(self.input_connections) == 0:
            raise ValueError("Connections not registered yet")

        input_connections = [
            conn for conn in self.input_connections if conn.from_operation.uuid == uuid
        ]
        return len(input_connections) == 1 and len(self.input_connections) == 1

    def set_thread_object(self, thread_object: ThreadObject) -> None:
        """Set thread object.
        
        Args:
            thread_object (ThreadObject): Thread object."""
        self.assigned_thread_object = thread_object

    def get_thread_object(self) -> ThreadObject | None:
        """Get thread object.
        
        Returns:
            ThreadObject | None: Result of get thread object."""
        return self.assigned_thread_object

    def get_output_connections(self) -> list[Connection]:
        """Get output connections.
        
        Returns:
            list[Connection]: Result of get output connections."""
        return self.output_connections

    def get_input_connections(self) -> list[Connection]:
        """Get input connections.
        
        Returns:
            list[Connection]: Result of get input connections."""
        return self.input_connections

    def add_input_connection(self, connection: Connection) -> None:
        """Add input connection.
        
        Args:
            connection (Connection): Connection."""
        self.input_connections.append(connection)
        self.has_input_connections = True

    def add_output_connection(self, connection: Connection) -> None:
        """Add output connection.
        
        Args:
            connection (Connection): Connection."""
        self.output_connections.append(connection)
        self.has_output_connections = True

    def all_inputs_solved(self) -> bool:
        """All inputs solved.
        
        Returns:
            bool: Result of all inputs solved."""
        non_default_connections = [
            conn for conn in self.input_connections if not conn.is_default
        ]

        if not non_default_connections:
            return True

        return all(
            conn.from_operation.execution_timestep is not None
            for conn in non_default_connections
        )

    def __str__(self) -> str:
        """Return a human-readable string representation.
        
        Returns:
            str: Result of str  ."""
        return f"Operation {self.name} with UUID {self.uuid} with output connections {[str(conn) for conn in self.output_connections]} and input connections {[str(conn) for conn in self.input_connections]}"


class Connection:
    def __init__(
        self,
        from_operation: Operation,
        from_port: str,
        to_operation: Operation,
        to_port: str,
        data_type: str,
        is_default: bool = False,
    ) -> None:
        """Initialize the object.
        
        Args:
            from_operation (Operation): From operation.
            from_port (str): From port.
            to_operation (Operation): To operation.
            to_port (str): To port.
            data_type (str): Data type.
            is_default (bool): Is default."""
        self.from_operation: Operation = from_operation
        self.from_port: str = from_port
        self.to_operation: Operation = to_operation
        self.to_port: str = to_port
        self.data_type: str = data_type
        self.is_default: bool = is_default

        self.from_operation.add_output_connection(self)
        self.to_operation.add_input_connection(self)

    def __str__(self) -> str:
        """Return a human-readable string representation.
        
        Returns:
            str: Result of str  ."""
        return f"Connection from {self.from_operation.name}:{self.from_operation.uuid}:{self.from_port} to {self.to_operation.name}:{self.to_operation.uuid}:{self.to_port}"
