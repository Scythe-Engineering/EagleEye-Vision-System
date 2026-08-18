from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CapturedFrame:
    """A raw frame paired with the moment it was captured.

    Attributes:
        image: Frame in BGR order, without rotation applied.
        capture_monotonic_ns: Capture time on ``CLOCK_MONOTONIC``. Sources that
            can report a hardware capture time do so; the rest stamp delivery
            time, which is the best they can offer.
    """

    image: np.ndarray
    capture_monotonic_ns: int
