"""The shipped localization preset must stay wireable as operations change."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRESET_PATH = PROJECT_ROOT / "library" / "localization_pipeline_preset.json"
CONFIG_DEF_DIRS = (
    PROJECT_ROOT / "src" / "main_operations" / "definitions" / "config_data",
    PROJECT_ROOT / "src" / "secondary_operations" / "config_data",
)


def _config_def(action_name: str) -> dict[str, Any]:
    """Load the operation definition backing a preset node."""
    file_name = f"{action_name.removesuffix('.py')}_config_def.json"
    for directory in CONFIG_DEF_DIRS:
        candidate = directory / file_name
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
    raise AssertionError(f"No config definition for {action_name}")


def _preset() -> list[dict[str, Any]]:
    pipelines = json.loads(PRESET_PATH.read_text(encoding="utf-8"))
    assert list(pipelines) == ["localization"]
    return pipelines["localization"]


def test_preset_ports_and_targets_resolve() -> None:
    operations = _preset()
    by_uuid = {operation["uuid"]: operation for operation in operations}
    assert len(by_uuid) == len(operations), "duplicate uuid in preset"

    for operation in operations:
        definition = _config_def(operation["action_name"])
        outputs = set(definition.get("output_nodes", []))
        for connection in operation["connections"]:
            assert connection["from_uuid"] == operation["uuid"]
            assert connection["from_port"] in outputs, connection
            target = by_uuid.get(connection["to_uuid"])
            assert target is not None, connection
            target_inputs = {
                node["name"] for node in _config_def(target["action_name"])["input_nodes"]
            }
            assert connection["to_port"] in target_inputs, connection


def test_preset_publishes_the_contract_the_java_library_reads() -> None:
    """EagleEyeCamera joins pose and meta by timestamp under one source subtable."""
    published = {
        operation["action_params"]["target_key"]: operation["action_params"]["schema"]
        for operation in _preset()
        if operation["action_name"] == "publish_to_networktables.py"
    }
    assert published == {
        "localization/front/pose": "pose3d",
        "localization/front/meta": "auto",
    }


def test_preset_feeds_pose_meta_straight_from_the_solver() -> None:
    """Metrics must keep the solver's capture timestamp, so nothing may sit in between."""
    meta_connections = [
        connection
        for operation in _preset()
        for connection in operation["connections"]
        if connection["from_port"] == "pose_meta"
    ]
    assert len(meta_connections) == 1
    publisher = next(
        operation
        for operation in _preset()
        if operation["uuid"] == meta_connections[0]["to_uuid"]
    )
    assert publisher["action_name"] == "publish_to_networktables.py"
