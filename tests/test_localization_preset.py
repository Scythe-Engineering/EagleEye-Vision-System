"""The bundled robot-localization template must stay wireable as operations change."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATES_PATH = (
    PROJECT_ROOT / "src" / "webui" / "js" / "pipeline" / "pipelineTemplates.json"
)
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


def _template() -> list[dict[str, Any]]:
    """Load the bundled robot-localization template."""
    templates = json.loads(TEMPLATES_PATH.read_text(encoding="utf-8"))
    return templates["robot_localization"]["nodes"]


def test_template_ports_and_targets_resolve() -> None:
    """Every template connection must reference declared ports and operations."""
    operations = _template()
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


def test_template_publishes_the_contract_the_java_library_reads() -> None:
    """EagleEyeCamera joins pose and meta by timestamp under one source subtable."""
    published = {
        operation["action_params"]["target_key"]: operation["action_params"]["schema"]
        for operation in _template()
        if operation["action_name"] == "publish_to_networktables.py"
    }
    assert published == {
        "localization/front/pose": "pose3d",
        "localization/front/meta": "auto",
    }

    pose_publisher = next(
        operation
        for operation in _template()
        if operation["action_params"].get("target_key") == "localization/front/pose"
    )
    publishers = {
        connection["to_uuid"]: operation["action_name"]
        for operation in _template()
        for connection in operation["connections"]
    }
    assert publishers[pose_publisher["uuid"]] == "camera_to_robot_pose.py"


def test_both_branches_keep_one_capture_timestamp() -> None:
    """The robot joins pose to meta on exact timestamp equality, so both must keep the solver's.

    A single-input operation passes its capture timing through untouched. A multi-input one
    averages the timings of everything feeding it, which would leave the two branches carrying
    different timestamps and the robot silently dropping every sample.
    """
    operations = _template()
    incoming: dict[str, int] = {operation["uuid"]: 0 for operation in operations}
    for operation in operations:
        for connection in operation["connections"]:
            incoming[connection["to_uuid"]] += 1

    solver = next(
        operation
        for operation in operations
        if operation["action_name"] == "pnp_camera_localization.py"
    )
    downstream = {
        connection["to_uuid"] for connection in solver["connections"]
    }
    assert {
        connection["from_port"] for connection in solver["connections"]
    } == {"camera_pose", "pose_meta"}

    by_uuid = {operation["uuid"]: operation for operation in operations}
    while downstream:
        uuid = downstream.pop()
        assert incoming[uuid] == 1, f"{by_uuid[uuid]['action_name']} averages capture timings"
        downstream.update(
            connection["to_uuid"] for connection in by_uuid[uuid]["connections"]
        )
