"""Synthetic heading delivery, not a recording of a physical gyro."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from src.utils.timing import monotonic_ns_to_nt_us


@dataclass(frozen=True)
class GyroSettings:
    """Noise is radians; delay and processing offset use dataset time."""

    seed: int = 0
    noise_std_rad: float = 0.0
    bias_rad: float = 0.0
    delivery_delay_ms: float = 0.0
    processing_offset_ms: float = 0.0
    mode: str = "synthetic"

    def __post_init__(self) -> None:
        """Reject malformed perturbations and ambiguous oracle settings."""
        values = (
            self.noise_std_rad,
            self.bias_rad,
            self.delivery_delay_ms,
            self.processing_offset_ms,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("gyro settings must be finite")
        if (
            min(self.noise_std_rad, self.delivery_delay_ms, self.processing_offset_ms)
            < 0
        ):
            raise ValueError(
                "gyro noise, delay and processing offset must be nonnegative"
            )
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("gyro seed must be a nonnegative integer")
        if self.mode not in ("synthetic", "ideal-oracle"):
            raise ValueError("unknown gyro mode")
        if self.mode == "ideal-oracle" and any(values):
            raise ValueError(
                "ideal-oracle requires zero perturbation, delay and offset"
            )

    def provenance(self) -> dict[str, Any]:
        """Describe the artificial source and native measurement-time contract."""
        return {
            **asdict(self),
            "oracle_equivalent": self.noise_std_rad
            == self.bias_rad
            == self.delivery_delay_ms
            == 0,
            "source": "artificial T_field_from_robot heading; NOT recorded physical gyro",
            "clock": "native double publisher.set(yaw, measurement_nt_us); ntcore translates to receiver-local NT time",
            "clock_sync": "client source fails closed until NT synchronized; server/local need no offset",
            "sampling": "one measurement per annotated capture; no future truth interpolation",
        }


def truth_heading(annotation: dict[str, Any]) -> float:
    """Extract NWU CCW robot yaw, never robot position.

    Args:
        annotation: Truth row containing T_field_from_robot as a 4x4 matrix,
            flattened matrix, or dictionary with a matrix field.

    Returns:
        Robot yaw in radians.

    Raises:
        ValueError: The transform cannot be converted to a finite 4x4 matrix.
    """
    value = annotation.get("T_field_from_robot")
    if isinstance(value, dict):
        value = value.get("matrix")
    matrix = np.asarray(value, dtype=float)
    if matrix.size == 16:
        matrix = matrix.reshape(4, 4)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("gyro requires finite T_field_from_robot")
    return math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))


class SyntheticGyro:
    """Stream causal native-double measurements through the production NT reader."""

    def __init__(self, settings: GyroSettings, epoch_ns: int) -> None:
        """Start a reproducible per-clip stream on the camera replay epoch."""
        self.settings = settings
        self.epoch_ns = epoch_ns
        self.rng = np.random.default_rng(settings.seed)
        self.pending: deque[tuple[int, dict[str, Any]]] = deque()
        self.last_capture_ns: int | None = None

    def advance(self, annotation: dict[str, Any], timestamp_ns: int) -> dict[str, Any]:
        """Accept current truth and deliver samples due by synthetic processing.

        Args:
            annotation: Current capture's truth row with T_field_from_robot.
            timestamp_ns: Strictly increasing capture time relative to the epoch.

        Returns:
            Mode, measurement, processing_timestamp_us, and delivered samples.
            Measurement timestamps use local NT microseconds; delivery timestamps
            are separate and never replace measurement times.

        Raises:
            ValueError: Capture times do not increase, the truth transform is
                invalid, or the perturbed yaw is not finite.
        """
        if self.last_capture_ns is not None and timestamp_ns <= self.last_capture_ns:
            raise ValueError("gyro capture timestamps must increase")
        self.last_capture_ns = timestamp_ns
        yaw = (
            truth_heading(annotation)
            + self.settings.bias_rad
            + float(self.rng.normal(0, self.settings.noise_std_rad))
        )
        if not math.isfinite(yaw):
            raise ValueError("synthetic gyro perturbation produced nonfinite yaw")
        yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
        sample = {
            "timestamp_us": monotonic_ns_to_nt_us(self.epoch_ns + timestamp_ns),
            "value": yaw,
        }
        delivered_at = timestamp_ns + round(self.settings.delivery_delay_ms * 1_000_000)
        self.pending.append((delivered_at, sample))
        instant = timestamp_ns + round(self.settings.processing_offset_ms * 1_000_000)
        delivered = []
        while self.pending and self.pending[0][0] <= instant:
            delivery, item = self.pending.popleft()
            delivered.append(
                {
                    "sample": item,
                    "delivery_timestamp_us": monotonic_ns_to_nt_us(
                        self.epoch_ns + delivery
                    ),
                }
            )
        return {
            "mode": self.settings.mode,
            "measurement": sample,
            "processing_timestamp_us": monotonic_ns_to_nt_us(self.epoch_ns + instant),
            "delivered": delivered,
        }

    def publish(
        self, pipeline: Any, annotation: dict[str, Any], timestamp_ns: int
    ) -> dict[str, Any]:
        """Publish delivered yaw with measurement time, never delivery time.

        Args:
            pipeline: Benchmark pipeline owning _benchmark_gyro_publisher.
            annotation: Current capture's truth row with T_field_from_robot.
            timestamp_ns: Strictly increasing capture time relative to the epoch.

        Returns:
            The advance record, including measurements delivered this cycle.

        Raises:
            ValueError: Capture times do not increase, the truth transform is
                invalid, or the perturbed yaw is not finite.
        """
        record = self.advance(annotation, timestamp_ns)
        publisher = pipeline._benchmark_gyro_publisher
        for delivered in record["delivered"]:
            sample = delivered["sample"]
            publisher.set(sample["value"], sample["timestamp_us"])
        return record
