from __future__ import annotations

from collections.abc import Mapping, Sequence
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
    def __init__(
        self,
        instance: OperationInstance,
        uuid: str,
        name: str,
        is_data_source: bool = False,
        input_ports: Sequence[str] = (),
        output_ports: Sequence[str] = (),
    ) -> None:
        """Initializes the Operation class.

        Args:
            instance (object): The instance of the operation.
            uuid (str): The UUID of the operation.
            name (str): The name of the operation.
            is_data_source (bool): Whether this operation generates its own data.
            input_ports: Declared input port names.
            output_ports: Declared output port names.
        """
        self.instance: OperationInstance = instance
        self.uuid: str = uuid
        self.name: str = name
        self.is_data_source: bool = is_data_source
        self.input_ports: tuple[str, ...] = self._normalise_ports(input_ports, "input")
        self.output_ports: tuple[str, ...] = self._normalise_ports(
            output_ports, "output"
        )
        self.routes_output_ports: bool = len(self.output_ports) > 1
        self._declared_output_ports: frozenset[str] = frozenset(self.output_ports)

        self.input_connections: list[Connection] = []
        self.output_connections: list[Connection] = []
        self.assigned_thread_object: ThreadObject | None = None

        self.has_output_connections: bool = False
        self.has_input_connections: bool = False

        self.execution_timestep: int | None = None
        self.finish_timestep: int | None = None

    @staticmethod
    def _normalise_ports(ports: Sequence[str], kind: str) -> tuple[str, ...]:
        """Validate and freeze declared port names."""
        result = tuple(ports)
        if any(not isinstance(port, str) or not port for port in result):
            raise ValueError(f"Declared {kind} ports must be non-empty strings")
        if len(set(result)) != len(result):
            raise ValueError(f"Declared {kind} ports must be unique: {result!r}")
        return result

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

        # Single-output operations are deliberately opaque: in particular, a dict
        # is a value rather than an output-port envelope.
        if not self.routes_output_ports:
            return attach_output_timing(output, input_data)

        if not isinstance(output, Mapping):
            raise ValueError(
                f"Operation {self.name} declares multiple outputs {self.output_ports!r} "
                f"but returned non-mapping {type(output).__name__}"
            )
        actual = frozenset(output.keys())
        missing = self._declared_output_ports - actual
        undeclared = actual - self._declared_output_ports
        if missing or undeclared:
            raise ValueError(
                f"Operation {self.name} output keys do not match declared ports; "
                f"missing={sorted(missing)!r}, undeclared={sorted(undeclared, key=str)!r}"
            )
        # Time each branch, not the envelope. attach_output_timing intentionally
        # leaves an explicitly TimedValue branch untouched.
        return {
            port: attach_output_timing(output[port], input_data)
            for port in self.output_ports
        }

    def resolve_output_port(self, output: Any, port: str) -> Any:
        """Resolve *port* from a runtime output while preserving single-output opacity.

        Args:
            output: Value stored for this operation by the flow manager.
            port: Declared output port selected by one connection.

        Returns:
            The routed branch, or the whole value for single-output operations.

        Raises:
            ValueError: If a multi-output value does not contain the port.
        """
        if not self.routes_output_ports:
            return output
        # ``run`` enforces this invariant; retain a useful error for externally
        # populated output stores and tests.
        if not isinstance(output, Mapping) or port not in output:
            raise ValueError(
                f"Output for operation {self.name} has no routed port {port!r}"
            )
        return output[port]

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
        if from_port not in from_operation.output_ports:
            raise ValueError(
                f"Unknown output port {from_port!r} on operation "
                f"{from_operation.name}; declared ports: {from_operation.output_ports!r}"
            )
        if to_port not in to_operation.input_ports:
            raise ValueError(
                f"Unknown input port {to_port!r} on operation "
                f"{to_operation.name}; declared ports: {to_operation.input_ports!r}"
            )

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
