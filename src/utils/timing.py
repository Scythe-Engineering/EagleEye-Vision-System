from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class TimingMetadata:
    """Capture-time metadata propagated alongside pipeline values."""

    capture_nt_us: int


@dataclass(frozen=True)
class TimedValue(Generic[T]):
    """A pipeline value paired with source-frame capture timing."""

    value: T
    timing: TimingMetadata


FramePacket = TimedValue[Any]


def is_timed(value: object) -> bool:
    """Check whether a value carries timing metadata.

    Args:
        value: Value to inspect.

    Returns:
        True when the value is a TimedValue, otherwise False.
    """
    return isinstance(value, TimedValue)


def unwrap_timed(value: T | TimedValue[T]) -> T:
    """Return the raw value from a timed wrapper.

    Args:
        value: Raw value or TimedValue wrapper.

    Returns:
        The wrapped value when timed, otherwise the original value.
    """
    return value.value if isinstance(value, TimedValue) else value


def unwrap_timed_deep(value: Any) -> Any:
    """Remove TimedValue wrappers recursively while preserving container shape.

    Args:
        value: Value or container that may include TimedValue wrappers.

    Returns:
        A value with all nested TimedValue wrappers replaced by raw values.
    """

    if isinstance(value, TimedValue):
        return unwrap_timed_deep(value.value)
    if isinstance(value, Mapping):
        return {key: unwrap_timed_deep(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(unwrap_timed_deep(item) for item in value)
    if isinstance(value, list):
        return [unwrap_timed_deep(item) for item in value]
    return value


def get_timing(value: object) -> TimingMetadata | None:
    """Get timing metadata from a timed value.

    Args:
        value: Value to inspect.

    Returns:
        Timing metadata when value is timed, otherwise None.
    """
    return value.timing if isinstance(value, TimedValue) else None


def retime(value: T | TimedValue[T], timing: TimingMetadata) -> TimedValue[T]:
    """Wrap a value with replacement timing metadata.

    Args:
        value: Raw value or TimedValue to re-time.
        timing: Timing metadata to attach.

    Returns:
        TimedValue containing the raw value and provided timing metadata.
    """
    return TimedValue(unwrap_timed(value), timing)


def oldest_timing(timings: Sequence[TimingMetadata]) -> TimingMetadata:
    """Select the oldest timing metadata record.

    Args:
        timings: Timing metadata records to compare.

    Returns:
        The timing metadata record with the earliest capture timestamp.
    """
    if not timings:
        raise ValueError("oldest_timing() requires at least one timing")
    return min(timings, key=lambda timing: timing.capture_nt_us)


def collect_timings(value: Any) -> list[TimingMetadata]:
    """Collect timing metadata from TimedValue wrappers in current-frame input.

    Args:
        value: Value or container that may include TimedValue wrappers.

    Returns:
        Timing metadata records found in the value.
    """

    if isinstance(value, TimedValue):
        return [value.timing]
    if isinstance(value, Mapping):
        timings: list[TimingMetadata] = []
        for item in value.values():
            timings.extend(collect_timings(item))
        return timings
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        timings = []
        for item in value:
            timings.extend(collect_timings(item))
        return timings
    return []


def attach_output_timing(output: Any, inputs: Any) -> Any:
    """Attach input capture timing to a raw operation output when possible.

    Args:
        output: Operation output that may need timing metadata.
        inputs: Operation inputs used as the timing metadata source.

    Returns:
        Timed output when input timing exists, otherwise the original output.
    """

    if output is None or isinstance(output, TimedValue):
        return output

    timings = collect_timings(inputs)
    if not timings:
        return output

    timing = oldest_timing(timings)
    return TimedValue(output, timing)
