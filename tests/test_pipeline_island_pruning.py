"""Tests for runtime-only pipeline island pruning."""

from __future__ import annotations

from typing import Any, cast

from src.config.utils.pipeline import Pipeline
from src.utils.camera_utils.camera_config_manager import CameraConfigRegistry
from tests.utils.dummy_dependencies import (
    DummyComputePool,
    FakeCameraThreadManager,
    FakeEagleEyeInterface,
    FakeNetworkTable,
)
from tests.utils.dummy_data import dummy_frame


class CapturingLogger:
    """Minimal logger that keeps messages available for assertions."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def log(self, message: str) -> None:
        self.messages.append(message)


def _make_pipeline(config: list[dict[str, Any]], logger: CapturingLogger) -> Pipeline:
    camera_manager = FakeCameraThreadManager(default_frame=dummy_frame())
    camera_manager.add_camera("basic_test")

    return Pipeline(
        config,
        cast(Any, FakeEagleEyeInterface()),
        cast(Any, DummyComputePool()),
        cast(Any, FakeNetworkTable()),
        cast(Any, logger),
        cast(Any, camera_manager),
        camera_config_registry=CameraConfigRegistry(),
        camera_bus_ids=["basic_test"],
        pipeline_name="test_pipeline",
    )


def test_pipeline_prunes_unreachable_runtime_island() -> None:
    logger = CapturingLogger()
    config = [
        {
            "action_name": "device_input.py",
            "action_params": {"bus_id": "basic_test", "frame_rotation": 0},
            "uuid": "camera",
            "connections": [
                {
                    "from_uuid": "camera",
                    "from_port": "frame",
                    "to_uuid": "reachable",
                    "to_port": "detections",
                    "data_type": "frame",
                }
            ],
        },
        {
            "action_name": "tag_filter.py",
            "action_params": {},
            "uuid": "reachable",
            "connections": [],
        },
        {
            "action_name": "tag_filter.py",
            "action_params": {},
            "uuid": "island_a",
            "connections": [
                {
                    "from_uuid": "island_a",
                    "from_port": "filtered_detections",
                    "to_uuid": "island_b",
                    "to_port": "detections",
                    "data_type": "filtered_detections",
                }
            ],
        },
        {
            "action_name": "tag_filter.py",
            "action_params": {},
            "uuid": "island_b",
            "connections": [],
        },
    ]

    pipeline = _make_pipeline(config, logger)

    assert set(pipeline.operations) == {"camera", "reachable"}
    assert any("operation islands disconnected" in msg for msg in logger.messages)
    assert any("island_a" in msg and "island_b" in msg for msg in logger.messages)


def test_pipeline_keeps_island_connected_to_non_camera_data_source() -> None:
    logger = CapturingLogger()
    config = [
        {
            "action_name": "get_networktables_value.py",
            "action_params": {"network_table_key": "example"},
            "uuid": "nt_source",
            "connections": [
                {
                    "from_uuid": "nt_source",
                    "from_port": "data",
                    "to_uuid": "nt_child",
                    "to_port": "detections",
                    "data_type": "data",
                    "is_default": True,
                }
            ],
        },
        {
            "action_name": "tag_filter.py",
            "action_params": {},
            "uuid": "nt_child",
            "connections": [],
        },
    ]

    pipeline = _make_pipeline(config, logger)

    assert set(pipeline.operations) == {"nt_source", "nt_child"}
    assert not any("operation islands disconnected" in msg for msg in logger.messages)
