from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


_IDENTITY_TRANSFORM = [
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
]


def _finite_number_or_zero(value: Any) -> float:
    """Return a finite numeric JSON value, replacing null/non-finite values with 0."""
    if value is None or isinstance(value, bool):
        return 0.0

    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0

    return number if math.isfinite(number) else 0.0


def _sanitize_transform(transform: Any) -> tuple[list[float], int]:
    """Sanitize one flat 4x4 fmap transform.

    Missing or structurally invalid transforms are replaced with identity, which
    represents zero rotation offset and zero translation. Existing 16-value
    transforms preserve all finite numeric values and replace null/non-finite
    entries with zero.
    """
    if not isinstance(transform, list) or len(transform) != 16:
        return _IDENTITY_TRANSFORM.copy(), 1

    sanitized: list[float] = []
    replacements = 0
    for value in transform:
        sanitized_value = _finite_number_or_zero(value)
        if sanitized_value != value:
            replacements += 1
        sanitized.append(sanitized_value)

    return sanitized, replacements


def sanitize_apriltag_map_file(file_path: Path) -> int:
    """Fix uploaded AprilTag fmap/json files in place.

    Returns the number of fiducial transform entries that were fixed. Files
    without a top-level ``fiducials`` list are left unchanged.
    """
    with file_path.open("r", encoding="utf-8") as map_file:
        data = json.load(map_file)

    fiducials = data.get("fiducials") if isinstance(data, dict) else None
    if not isinstance(fiducials, list):
        return 0

    fixes = 0
    for fiducial in fiducials:
        if not isinstance(fiducial, dict):
            continue
        sanitized_transform, transform_fixes = _sanitize_transform(
            fiducial.get("transform")
        )
        if transform_fixes:
            fiducial["transform"] = sanitized_transform
            fixes += transform_fixes

    if fixes:
        with file_path.open("w", encoding="utf-8") as map_file:
            json.dump(data, map_file, separators=(",", ":"))
            map_file.write("\n")

    return fixes
