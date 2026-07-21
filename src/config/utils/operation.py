from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.config.utils.line_profiling import line_profiling_manager
from src.main_operations.definitions.base.base_class import OperationInstance
from src.utils.timing import attach_output_timing, unwrap_timed_deep

if TYPE_CHECKING:
    from src.config.utils.thread_object import ThreadObject


class _SkipPipelineCycle:
    """Sentinel returned by an operation to discard the current pipeline cycle."""


SKIP_PIPELINE_CYCLE = _SkipPipelineCycle()


class Operation:
    def __init__(self, instance: OperationInstance, uuid: str, name: str, is_data_source: bool = False) -> None:
        """Initializes the Operation class.

        Args:
            instance (object): The instance of the operation.
            uuid (str): The UUID of the operation.
            name (str): The name of the operation.
            is_data_source (bool): Whether this operation generates its own data.
        """
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
        """Run the operation and propagate capture timing metadata.

        Operations receive raw unwrapped values by default so existing image/dict
        processors remain compatible. Operations that need timing metadata may set
        ``uses_timed_inputs = True`` on the instance.
        """
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
        if output is SKIP_PIPELINE_CYCLE:
            return output
        return attach_output_timing(output, input_data)

    def is_only_input_connection(self, uuid: str) -> bool:
        """
        Check if the passed uuid is the only input connection to this operation.

        Args:
            uuid (str): The uuid of the operation that has its output connected to this operation.

        Returns:
            bool: True if the uuid is the only input connection to this operation, False otherwise.
        Raises:
            ValueError: If the connections have not been registered yet.
        """
        if len(self.input_connections) == 0:
            raise ValueError("Connections not registered yet")

        input_connections = [
            conn for conn in self.input_connections if conn.from_operation.uuid == uuid
        ]
        return len(input_connections) == 1 and len(self.input_connections) == 1

    def set_thread_object(self, thread_object: ThreadObject) -> None:
        """Set the assigned thread object for the operation.

        Args:
            thread_object (ThreadObject): The thread object to assign.
        """
        self.assigned_thread_object = thread_object

    def get_thread_object(self) -> ThreadObject | None:
        """Get the assigned thread object for the operation.

        Returns:
            ThreadObject | None: The assigned thread object for the operation.
        """
        return self.assigned_thread_object

    def get_output_connections(self) -> list[Connection]:
        """Get the output connections of the operation.

        Returns:
            list[Connection]: The output connections of the operation.
        """
        return self.output_connections

    def get_input_connections(self) -> list[Connection]:
        """Get the input connections of the operation.

        Returns:
            list[Connection]: The input connections of the operation.
        """
        return self.input_connections

    def add_input_connection(self, connection: Connection) -> None:
        """Add an input connection to the operation.

        Args:
            connection (Connection): The connection to add.
        """
        self.input_connections.append(connection)
        self.has_input_connections = True

    def add_output_connection(self, connection: Connection) -> None:
        """Add an output connection to the operation.

        Args:
            connection (Connection): The connection to add.
        """
        self.output_connections.append(connection)
        self.has_output_connections = True

    def all_inputs_solved(self) -> bool:
        """Check if all non-default inputs of the operation are solved.

        Default connections use previous frame data and are always available.

        Returns:
            bool: True if all non-default inputs are solved, False otherwise.
        """
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
        """Initializes the Connection class.

        Args:
            from_operation (Operation): The operation that outputs data through this connection.
            from_port (str): The output port name on the from_operation.
            to_operation (Operation): The operation that receives data through this connection.
            to_port (str): The input port name on the to_operation.
            data_type (str): The type of data transmitted through this connection.
            is_default (bool, optional): Whether this connection is a default connection from the from_operation. Defaults to False.
        """
        self.from_operation: Operation = from_operation
        self.from_port: str = from_port
        self.to_operation: Operation = to_operation
        self.to_port: str = to_port
        self.data_type: str = data_type
        self.is_default: bool = is_default

        self.from_operation.add_output_connection(self)
        self.to_operation.add_input_connection(self)

    def __str__(self) -> str:
        return f"Connection from {self.from_operation.name}:{self.from_operation.uuid}:{self.from_port} to {self.to_operation.name}:{self.to_operation.uuid}:{self.to_port}"
