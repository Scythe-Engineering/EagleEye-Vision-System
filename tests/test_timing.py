from __future__ import annotations

from src.utils.timing import (
    TimedValue,
    TimingMetadata,
    attach_output_timing,
    oldest_timing,
    get_timing,
    unwrap_timed,
    unwrap_timed_deep,
)


def test_timed_value_helpers_unwrap_and_get_timing() -> None:
    """Verify timed value helpers unwrap and get timing."""
    timing = TimingMetadata(capture_nt_us=123)
    value = TimedValue({"frame": [1, 2, 3]}, timing)

    assert unwrap_timed(value) == {"frame": [1, 2, 3]}
    assert get_timing(value) is timing
    assert unwrap_timed_deep({"a": value}) == {"a": {"frame": [1, 2, 3]}}


def test_oldest_timing_selects_earliest_capture_timestamp() -> None:
    """Verify oldest timing selects the earliest capture timestamp."""
    first = TimingMetadata(capture_nt_us=100)
    second = TimingMetadata(capture_nt_us=200)

    assert oldest_timing([second, first]) is first


def test_attach_output_timing_wraps_raw_output() -> None:
    """Verify attach output timing wraps raw output."""
    timing = TimingMetadata(capture_nt_us=123)
    output = attach_output_timing("processed", TimedValue("input", timing))

    assert isinstance(output, TimedValue)
    assert output.value == "processed"
    assert output.timing is timing
