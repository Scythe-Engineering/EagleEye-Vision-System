from __future__ import annotations

from src.utils.timing import (
    TimedValue,
    TimingMetadata,
    attach_output_timing,
    average_timings,
    get_timing,
    unwrap_timed,
    unwrap_timed_deep,
)


def test_timed_value_helpers_unwrap_and_get_timing() -> None:
    timing = TimingMetadata(capture_nt_us=123, capture_monotonic_ns=456)
    value = TimedValue({"frame": [1, 2, 3]}, timing)

    assert unwrap_timed(value) == {"frame": [1, 2, 3]}
    assert get_timing(value) is timing
    assert unwrap_timed_deep({"a": value}) == {"a": {"frame": [1, 2, 3]}}


def test_average_timings_records_derived_sources() -> None:
    first = TimingMetadata(capture_nt_us=100, capture_monotonic_ns=1000)
    second = TimingMetadata(capture_nt_us=200, capture_monotonic_ns=2000)

    averaged = average_timings([first, second])

    assert averaged.capture_nt_us == 150
    assert averaged.capture_monotonic_ns == 1500
    assert averaged.derived_from == (first, second)


def test_attach_output_timing_wraps_raw_output() -> None:
    timing = TimingMetadata(capture_nt_us=123, capture_monotonic_ns=456)
    output = attach_output_timing("processed", TimedValue("input", timing))

    assert isinstance(output, TimedValue)
    assert output.value == "processed"
    assert output.timing is timing
