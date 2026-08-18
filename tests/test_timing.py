from __future__ import annotations

import time

from src.utils.timing import (
    TimedValue,
    TimingMetadata,
    attach_output_timing,
    average_timings,
    get_timing,
    monotonic_ns_to_nt_us,
    now_nt_us,
    unwrap_timed,
    unwrap_timed_deep,
)


def test_timed_value_helpers_unwrap_and_get_timing() -> None:
    timing = TimingMetadata(capture_nt_us=123, capture_monotonic_ns=456)
    value = TimedValue({"frame": [1, 2, 3]}, timing)

    assert unwrap_timed(value) == {"frame": [1, 2, 3]}
    assert get_timing(value) is timing
    assert unwrap_timed_deep({"a": value}) == {"a": {"frame": [1, 2, 3]}}


def test_average_timings_averages_capture_clocks() -> None:
    first = TimingMetadata(capture_nt_us=100, capture_monotonic_ns=1000)
    second = TimingMetadata(capture_nt_us=200, capture_monotonic_ns=2000)

    averaged = average_timings([first, second])

    assert averaged.capture_nt_us == 150
    assert averaged.capture_monotonic_ns == 1500


def test_monotonic_conversion_round_trips_through_the_nt_clock() -> None:
    """A monotonic reading converts to an NT time consistent with ntcore."""
    monotonic_ns = time.monotonic_ns()
    converted_us = monotonic_ns_to_nt_us(monotonic_ns)

    # Both clocks tick at the same rate, so the conversion should land within
    # a few milliseconds of a fresh ntcore reading taken right after.
    assert abs(now_nt_us() - converted_us) < 10_000


def test_monotonic_conversion_preserves_intervals() -> None:
    """Converting is affine, so differences survive exactly."""
    base_ns = time.monotonic_ns()

    earlier = monotonic_ns_to_nt_us(base_ns)
    later = monotonic_ns_to_nt_us(base_ns + 25_000_000)

    assert later - earlier == 25_000


def test_attach_output_timing_wraps_raw_output() -> None:
    timing = TimingMetadata(capture_nt_us=123, capture_monotonic_ns=456)
    output = attach_output_timing("processed", TimedValue("input", timing))

    assert isinstance(output, TimedValue)
    assert output.value == "processed"
    assert output.timing is timing
