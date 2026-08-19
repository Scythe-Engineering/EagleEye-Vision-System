"""Runtime routing tests for declared operation ports."""

from typing import Any

import pytest

from src.config.utils.flow_manager import FlowManager
from src.config.utils.operation import Connection, Operation
from src.main_operations.definitions.base.base_class import OperationInstance
from src.utils.timing import TimedValue, TimingMetadata


class _Return(OperationInstance):
    def __init__(self, value: Any) -> None:
        """Store the value returned by this operation stub.

        Args:
            value: Value returned from :meth:`run`.
        """
        self.value = value

    def run(self, *_args: Any, **_kwargs: Any) -> Any:
        """Return the configured value.

        Returns:
            Any: Configured operation output.
        """
        return self.value


def _manager(
    current: dict[str, Any], previous: dict[str, Any] | None = None
) -> FlowManager:
    """Create a minimally initialized flow manager for routing assertions.

    Args:
        current: Outputs produced during the current flow iteration.
        previous: Outputs retained from the preceding flow iteration.

    Returns:
        FlowManager: Manager with only output state initialized.
    """
    manager = FlowManager.__new__(FlowManager)
    manager.operation_outputs = current
    manager.previous_operation_outputs = previous or {}
    return manager


def test_multi_output_routes_selected_current_branch() -> None:
    """Route the requested branch from a current multi-output result."""
    source = Operation(
        _Return({"left": 1, "right": 2}),
        "s",
        "source",
        True,
        output_ports=("left", "right"),
    )
    destination = Operation(_Return(None), "d", "destination", input_ports=("value",))
    Connection(source, "right", destination, "value", "int")

    output = source.run(None)
    assert _manager({"s": output})._gather_operation_inputs(destination) == 2


def test_single_output_dict_remains_an_opaque_value() -> None:
    """Keep a single declared dictionary output intact."""
    value = {"left": 1, "right": 2}
    source = Operation(_Return(value), "s", "source", True, output_ports=("result",))
    destination = Operation(_Return(None), "d", "destination", input_ports=("value",))
    Connection(source, "result", destination, "value", "dict")

    output = source.run(None)
    assert output is value
    assert _manager({"s": output})._gather_operation_inputs(destination) is value


def test_multi_output_timing_is_per_branch_and_explicit_timing_wins() -> None:
    """Preserve branch timing and prefer explicitly supplied timing."""
    inherited = TimingMetadata(1, 2)
    explicit = TimingMetadata(3, 4)
    source = Operation(
        _Return({"raw": 1, "timed": TimedValue(2, explicit)}),
        "s",
        "source",
        output_ports=("raw", "timed"),
    )

    output = source.run(TimedValue("input", inherited))
    assert output["raw"] == TimedValue(1, inherited)
    assert output["timed"] == TimedValue(2, explicit)


def test_previous_output_is_selected_before_deep_unwrap() -> None:
    """Select a previous output branch before unwrapping its timed value."""
    timing = TimingMetadata(1, 2)
    source = Operation(_Return(None), "s", "source", output_ports=("a", "b"))
    destination = Operation(_Return(None), "d", "destination", input_ports=("value",))
    Connection(source, "b", destination, "value", "int", is_default=True)
    previous = {"s": {"a": TimedValue(1, timing), "b": TimedValue({"x": 2}, timing)}}

    assert _manager({}, previous)._gather_operation_inputs(destination) == {"x": 2}


def test_declared_ports_and_multi_output_shape_are_validated() -> None:
    """Reject undeclared connections and incomplete multi-output results."""
    source = Operation(_Return({"a": 1}), "s", "source", output_ports=("a", "b"))
    destination = Operation(_Return(None), "d", "destination", input_ports=("in",))
    with pytest.raises(ValueError, match="Unknown output port"):
        Connection(source, "other", destination, "in", "int")
    with pytest.raises(ValueError, match="missing"):
        source.run(None)
