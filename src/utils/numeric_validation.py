"""Shared numeric validation helpers for operation settings."""

from __future__ import annotations

from typing import Any


def is_integral(value: Any) -> bool:
    """Return whether a numeric setting can be stored as an exact integer.

    Args:
        value: Candidate setting of any type.

    Returns:
        True when the value converts to an equal integer. Non-numeric values,
        booleans, NaN, and the infinities all return False.
    """
    if isinstance(value, bool):
        return False
    try:
        return int(value) == value
    except (TypeError, ValueError, OverflowError):
        return False
