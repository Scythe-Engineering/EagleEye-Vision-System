from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import ntcore

T = TypeVar("T")

_nt_monotonic_offset_us: int | None = None


def now_nt_us() -> int:
    """Return the current NetworkTables timestamp in microseconds.

    Wraps the one private ntcore entry point the project depends on so a
    pyntcore change is contained to this module.
    """
    return ntcore._now()


def _measure_nt_monotonic_offset_us(samples: int = 11) -> int:
    """Measure the constant offset between the NT clock and CLOCK_MONOTONIC.

    Both clocks tick at the same rate but need not share an epoch. Sampling
    both in a tight loop and taking the median rejects the occasional sample
    that gets descheduled between the two reads.

    Args:
        samples: Number of paired readings to take.

    Returns:
        Microseconds to add to a monotonic timestamp to reach the NT clock.
    """
    deltas = sorted(
        now_nt_us() - time.monotonic_ns() // 1000 for _ in range(samples)
    )
    return deltas[len(deltas) // 2]


def nt_monotonic_offset_us() -> int:
    """Return the cached NT-to-monotonic offset, measuring it on first use."""
    global _nt_monotonic_offset_us
    if _nt_monotonic_offset_us is None:
        _nt_monotonic_offset_us = _measure_nt_monotonic_offset_us()
    return _nt_monotonic_offset_us


def monotonic_ns_to_nt_us(monotonic_ns: int) -> int:
    """Convert a CLOCK_MONOTONIC timestamp to the NetworkTables clock.

    Args:
        monotonic_ns: Timestamp in nanoseconds on CLOCK_MONOTONIC, such as a
            V4L2 buffer timestamp or ``time.monotonic_ns()``.

    Returns:
        The same instant in NetworkTables microseconds.
    """
    return monotonic_ns // 1000 + nt_monotonic_offset_us()


@dataclass(frozen=True)
class TimingMetadata:
    """Capture-time metadata propagated alongside pipeline values."""

    capture_nt_us: int
    capture_monotonic_ns: int
    frame_seq: int | None = None
    camera_name: str | None = None
    bus_id: str | None = None


@dataclass(frozen=True)
class TimedValue(Generic[T]):
    """A pipeline value paired with source-frame capture timing."""

    value: T
    timing: TimingMetadata


FramePacket = TimedValue[Any]


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
