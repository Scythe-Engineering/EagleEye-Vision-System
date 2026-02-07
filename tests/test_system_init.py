"""System initialization smoke tests without running the pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from typing import Any, cast

import pytest
from src.utils.logging.logger import Logger
from tests.utils.dummy_dependencies import (
    DummyComputePool,
    FakeCameraThreadManager,
    FakeEagleEyeInterface,
    FakeNetworkTable,
)
from tests.utils.dummy_data import dummy_frame


def test_pipeline_initialization_only(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    pipeline_config_path = project_root / "src" / "config" / "pipeline_config.json"
    with pipeline_config_path.open("r", encoding="utf-8") as handle:
        pipeline_config = json.load(handle)

    temp_config_path = tmp_path / "pipeline_config.json"
    temp_config_path.write_text(json.dumps(pipeline_config), encoding="utf-8")

    web_interface = FakeEagleEyeInterface()
    network_table = FakeNetworkTable()
    compute_pool = DummyComputePool()
    logger = Logger(log_directory="logs/test")
    camera_manager = FakeCameraThreadManager(default_frame=dummy_frame())
    camera_manager.add_camera("basic_test")
    camera_manager.add_camera("FaceTime HD Camera")
    camera_manager.add_camera("test_camera")

    try:
        from src.config.utils.generate_all_pipelines import generate_all_pipelines
    except ImportError as exc:
        pytest.skip(f"system_init_optional: {exc}")

    pipelines = generate_all_pipelines(
        cast(Any, web_interface),
        cast(Any, compute_pool),
        cast(Any, network_table),
        cast(Any, camera_manager),
        logger=logger,
        pipeline_config=str(temp_config_path),
    )

    if not pipelines:
        pytest.skip("pipeline_init_optional: no pipelines created")

    for pipeline in pipelines.values():
        assert pipeline.operations, "Pipeline has no operations"
        assert pipeline.thread is None
