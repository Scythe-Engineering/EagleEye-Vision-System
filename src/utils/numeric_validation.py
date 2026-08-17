"""Shared numeric validation helpers for operation settings."""

from __future__ import annotations

from typing import Any


def is_integral(value: Any) -> bool:
    """Return whether a numeric setting can be stored as an exact integer."""
    if isinstance(value, bool):
        return False
    try:
        return int(value) == value
    except (TypeError, ValueError):
        return False
