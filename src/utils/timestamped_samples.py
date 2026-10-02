"""Explicit measurement timestamps use the synchronized NetworkTables clock."""

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
