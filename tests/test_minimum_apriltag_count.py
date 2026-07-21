from __future__ import annotations

import pytest

from src.config.utils.operation import SKIP_PIPELINE_CYCLE
from src.secondary_operations.minimum_apriltag_count import MinimumApriltagCount


def test_skips_pipeline_cycle_below_minimum_detection_count() -> None:
    gate = MinimumApriltagCount()

    assert gate.run([]) is SKIP_PIPELINE_CYCLE
    assert gate.run([object()]) is SKIP_PIPELINE_CYCLE


def test_passes_detections_unchanged_at_or_above_minimum() -> None:
    gate = MinimumApriltagCount()
    detections = [object(), object()]

    assert gate.run(detections) == detections


def test_live_configuration_changes_minimum_detection_count() -> None:
    gate = MinimumApriltagCount()
    detection = [object()]

    gate.update_config({"minimum_detections": 1})

    assert gate.run(detection) == detection


@pytest.mark.parametrize("invalid_minimum", [0, -1, 1.5, True])
def test_rejects_invalid_minimum_detection_count(invalid_minimum: object) -> None:
    with pytest.raises(ValueError, match="minimum_detections"):
        MinimumApriltagCount(invalid_minimum)  # type: ignore[arg-type]
