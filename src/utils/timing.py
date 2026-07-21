from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class TimingMetadata:
    """Capture-time metadata propagated alongside pipeline values."""

    capture_nt_us: int
    capture_monotonic_ns: int
    frame_seq: int | None = None
    camera_name: str | None = None
    bus_id: str | None = None
    source: str | None = None
    derived_from: tuple["TimingMetadata", ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class TimedValue(Generic[T]):
    """A pipeline value paired with source-frame capture timing."""

    value: T
    timing: TimingMetadata


FramePacket = TimedValue[Any]


def is_timed(value: object) -> bool:
    return isinstance(value, TimedValue)


def unwrap_timed(value: T | TimedValue[T]) -> T:
    return value.value if isinstance(value, TimedValue) else value


def unwrap_timed_deep(value: Any) -> Any:
    """Remove TimedValue wrappers recursively while preserving container shape."""

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
    return value.timing if isinstance(value, TimedValue) else None


def retime(value: T | TimedValue[T], timing: TimingMetadata) -> TimedValue[T]:
    return TimedValue(unwrap_timed(value), timing)


def average_timings(timings: Sequence[TimingMetadata]) -> TimingMetadata:
    if not timings:
        raise ValueError("average_timings() requires at least one timing")
    if len(timings) == 1:
        return timings[0]

    return TimingMetadata(
        capture_nt_us=round(sum(t.capture_nt_us for t in timings) / len(timings)),
        capture_monotonic_ns=round(
            sum(t.capture_monotonic_ns for t in timings) / len(timings)
        ),
        source="average",
        derived_from=tuple(timings),
    )


def collect_timings(value: Any) -> list[TimingMetadata]:
    """Collect timing metadata from TimedValue wrappers in a current-frame input."""

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
    """Attach input capture timing to a raw operation output when possible."""

    if output is None or isinstance(output, TimedValue):
        return output

    timings = collect_timings(inputs)
    if not timings:
        return output

    timing = timings[0] if len(timings) == 1 else average_timings(timings)
    return TimedValue(output, timing)


def clone_timing(timing: TimingMetadata, **changes: Any) -> TimingMetadata:
    """Return a copy of timing with selected debug/source fields changed."""

    return replace(timing, **changes)
