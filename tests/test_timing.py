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

    # Both clocks tick at the same rate, so the only thing worth asserting is
    # that the epochs agree. The tolerance is loose on purpose: a scheduling
    # stall between the two readings is not a bug, while a wrong epoch is off
    # by hours and still caught.
    assert abs(now_nt_us() - converted_us) < 1_000_000


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


def test_deep_unwrap_rebuilds_containers_and_handles_subclasses() -> None:
    """Fast paths keep the general contract: wrappers removed, containers fresh."""
    from collections import OrderedDict, namedtuple

    import numpy as np

    from src.utils.timing import collect_timings

    first = TimingMetadata(capture_nt_us=1, capture_monotonic_ns=10)
    second = TimingMetadata(capture_nt_us=2, capture_monotonic_ns=20)
    pair = namedtuple("pair", "a b")
    frame = np.zeros((2, 2))
    detections = [{"id": 3, "corners": [(1.0, 2.0)], "score": np.float64(0.5)}]
    source = {
        "frame": TimedValue(frame, first),
        "detections": TimedValue(detections, second),
        "crops": ([TimedValue(frame, first)], frame),
        "ordered": OrderedDict(x=TimedValue("v", first)),
        "pair": pair(TimedValue(1, first), [2]),
    }

    result = unwrap_timed_deep(source)

    assert result["frame"] is frame
    assert result["detections"] == detections
    assert result["detections"] is not detections
    assert result["detections"][0] is not detections[0]
    assert result["crops"] == ([frame], frame)
    assert result["ordered"] == {"x": "v"} and type(result["ordered"]) is dict
    assert result["pair"] == (1, [2]) and type(result["pair"]) is tuple
    result["detections"][0]["id"] = 99
    assert detections[0]["id"] == 3

    assert collect_timings(source) == [first, second, first, first, first]
    assert collect_timings(detections) == []
    assert collect_timings("text") == []


def test_unknown_objects_pass_through_untouched() -> None:
    """Plain objects (e.g. lazily built crops) are never iterated or rebuilt."""
    from src.utils.timing import collect_timings

    class _Lazy:
        def __init__(self) -> None:
            self.touched = False

        def __iter__(self):
            self.touched = True
            return iter(())

        def __len__(self) -> int:
            self.touched = True
            return 0

    timing = TimingMetadata(capture_nt_us=1, capture_monotonic_ns=10)
    lazy = _Lazy()
    frame_like = TimedValue(([lazy], "frame"), timing)

    unwrapped = unwrap_timed_deep(frame_like)
    assert unwrapped[0][0] is lazy
    assert unwrap_timed_deep(lazy) is lazy
    assert collect_timings(([lazy], {"x": lazy})) == []
    assert attach_output_timing(lazy, frame_like).value is lazy
    assert not lazy.touched
