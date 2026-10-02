"""Explicit measurement timestamps use the synchronized NetworkTables clock."""

import math
from bisect import bisect_left
from typing import Any


def validate_sample(sample: object) -> dict[str, Any]:
    """Validate the sample shape; values remain generic native NT data.

    Args:
        sample: Measurement envelope containing only timestamp_us and value.

    Returns:
        The original sample dictionary with a valid native measurement timestamp.

    Raises:
        ValueError: The envelope is malformed or its timestamp is not a signed
            64-bit integer greater than 1.
    """
    if not isinstance(sample, dict) or set(sample) != {"timestamp_us", "value"}:
        raise ValueError("Expected {'timestamp_us': integer, 'value': native value}")
    timestamp = sample["timestamp_us"]
    if type(timestamp) is not int or not 1 < timestamp <= 2**63 - 1:
        raise ValueError("timestamp_us must be a signed 64-bit integer greater than 1")
    return sample


def align_heading(
    samples: object, capture_us: int, max_gap_us: int, nearest_us: int
) -> tuple[float, dict[str, str | int]]:
    """Circularly interpolate finite selected NWU yaw radians; never extrapolate.

    Args:
        samples: Nonempty list of native timestamped yaw measurements.
        capture_us: Capture time in the synchronized NT clock, in microseconds.
        max_gap_us: Maximum permitted interpolation bracket width.
        nearest_us: Maximum nearest-sample distance outside the history range.

    Returns:
        Wrapped NWU yaw in radians and exact, interpolated, or nearest diagnostics.

    Raises:
        ValueError: Capture or gyro data is invalid, the interpolation gap is too
            large, or no sufficiently recent measurement is available.
    """
    if type(capture_us) is not int or not 1 < capture_us <= 2**63 - 1:
        raise ValueError("missing_capture")
    if not isinstance(samples, list) or not samples:
        raise ValueError("missing_gyro")
    headings = {}
    for sample in samples:
        try:
            sample = validate_sample(sample)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError("invalid_gyro") from exc
        headings[sample["timestamp_us"]] = sample["value"]

    def heading(timestamp: int) -> float:
        """Validate only the measurement selected for alignment.

        Args:
            timestamp: Timestamp of an existing measurement in the history.

        Returns:
            Finite NWU yaw wrapped with a circular remainder in radians.

        Raises:
            ValueError: The selected yaw is not a finite numeric measurement.
        """
        yaw = headings[timestamp]
        try:
            if (
                isinstance(yaw, bool)
                or not isinstance(yaw, (int, float))
                or not math.isfinite(yaw)
            ):
                raise ValueError("Yaw must be finite radians")
            return math.remainder(float(yaw), 2 * math.pi)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError("invalid_gyro") from exc

    times = sorted(headings)
    index = bisect_left(times, capture_us)
    if index < len(times) and times[index] == capture_us:
        return heading(capture_us), {"alignment": "exact", "gyro_delta_us": 0}
    if 0 < index < len(times):
        before, after = times[index - 1 : index + 1]
        before_yaw, after_yaw = heading(before), heading(after)
        if after - before > max_gap_us:
            raise ValueError("gyro_gap_too_large")
        difference = math.remainder(after_yaw - before_yaw, 2 * math.pi)
        yaw = before_yaw + difference * (capture_us - before) / (after - before)
        return math.remainder(yaw, 2 * math.pi), {
            "alignment": "interpolated",
            "gyro_gap_us": after - before,
        }
    nearest = min(times, key=lambda timestamp: abs(timestamp - capture_us))
    delta = nearest - capture_us
    if abs(delta) <= nearest_us:
        return heading(nearest), {"alignment": "nearest", "gyro_delta_us": delta}
    raise ValueError("stale_gyro")
