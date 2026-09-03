"""Focused checks for first-boot pipeline generation and idle detection."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

import src.main_operations.definitions.object_detection as object_detection_module
import src.webui.web_server_utils.first_boot_mixin as first_boot_module
from src.config.utils.port_validation import validate_pipeline_connections
from src.main_operations.definitions.object_detection import ObjectDetectionDefinition
from src.webui.web_server_utils.first_boot_mixin import (
    FirstBootMixin,
    build_first_boot_pipeline,
)


class _Request:
    """Minimal Flask request replacement for direct mixin tests."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def get_json(self, silent: bool = False) -> dict[str, Any]:
        """Return the configured JSON payload."""
        return self.payload


class _CameraRegistry:
    """Resolve temporary calibrated camera records by bus ID."""

    def __init__(self, intrinsics_by_bus_id: dict[str, Path]) -> None:
        self.intrinsics_by_bus_id = intrinsics_by_bus_id

    def get_config(self, bus_id: str) -> SimpleNamespace:
        """Return the camera config required by the generator."""
        return SimpleNamespace(intrinsics_path=str(self.intrinsics_by_bus_id[bus_id]))


class _FirstBootHarness(FirstBootMixin):
    """File-backed first-boot endpoint harness."""

    def __init__(
        self,
        pipeline_path: Path,
        general_conf_path: Path,
        intrinsics_by_bus_id: dict[str, Path],
    ) -> None:
        self.pipeline_path = pipeline_path
        self.general_conf_path = general_conf_path
        self.available_cameras = {
            "Front Camera": {"name": "Front_Camera", "bus_id": "1"},
            "Front-Camera": {"name": "Front-Camera", "bus_id": "2"},
        }
        self.camera_config_registry = _CameraRegistry(intrinsics_by_bus_id)
        self._general_conf_lock = threading.Lock()
        self._pipeline_settings_lock = threading.RLock()
        self.frame_list_structure_lock = threading.Lock()
        self.restart_required_for_config = False
        self.runtime_id = "runtime-1"
        self.network_table_instance = None
        self.log = lambda _message: None

    def _pipeline_config_path(self) -> str:
        """Return the temporary pipeline path."""
        return str(self.pipeline_path)

    def _load_pipeline_config_file(self) -> dict[str, list[dict[str, Any]]]:
        """Load the temporary pipeline config."""
        return json.loads(self.pipeline_path.read_text(encoding="utf-8"))

    def _write_pipeline_config_file(
        self, config: dict[str, list[dict[str, Any]]]
    ) -> None:
        """Write the temporary pipeline config."""
        self.pipeline_path.write_text(
            json.dumps(config, indent=4) + "\n", encoding="utf-8"
        )

    def _read_general_conf(self) -> dict[str, Any]:
        """Load the temporary general config."""
        if not self.general_conf_path.exists():
            return {
                "network_table_address": "0.0.0.0",
                "view_stream_downscale": 0.5,
            }
        return json.loads(self.general_conf_path.read_text(encoding="utf-8"))

    def pipeline_objects_callback(self) -> dict[str, Any]:
        """Return no runtime pipelines before the requested restart."""
        return {}

    def _build_network_table_status(self) -> dict[str, Any]:
        """Return a disconnected test NetworkTables status."""
        return {"connected": False, "server": "10.0.0.2", "connection_count": 0}


def test_first_boot_generates_unique_multi_camera_pipelines(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Generate valid localize/both pipelines with unique camera sources."""
    pipeline_path = tmp_path / "pipeline_config.json"
    general_conf_path = tmp_path / "general_conf.json"
    pipeline_path.write_text("{}\n", encoding="utf-8")
    intrinsics_by_bus_id = {}
    for bus_id in ("1", "2"):
        path = tmp_path / bus_id / "intrinsics.json"
        path.parent.mkdir()
        path.write_text("{}", encoding="utf-8")
        intrinsics_by_bus_id[bus_id] = path

    harness = _FirstBootHarness(pipeline_path, general_conf_path, intrinsics_by_bus_id)
    monkeypatch.setattr(first_boot_module, "GENERAL_CONF_PATH", general_conf_path)
    monkeypatch.setattr(
        first_boot_module,
        "request",
        _Request(
            {
                "network_table_address": "10.0.0.2",
                "cameras": [
                    {"bus_id": "1", "mode": "both", "model_id": ""},
                    {"bus_id": "2", "mode": "localize", "model_id": ""},
                ],
            }
        ),
    )

    initial_status, initial_code = harness.get_first_boot_status()
    pipeline_path.write_text(
        json.dumps(
            {
                "existing": [
                    {
                        "action_name": "publish_to_networktables.py",
                        "action_params": {
                            "target_key": "localization/front-camera-1",
                            "schema": "pose3d",
                            "data_path": [],
                        },
                        "position": {"x": 0, "y": 0},
                        "uuid": "existing-publisher",
                        "connections": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    payload, status = harness.generate_first_boot_pipelines()

    assert initial_code == 200
    assert initial_status["required"] is True
    assert status == 200
    assert payload["restart_required"] is True
    assert payload["pipelines"] == [
        "wizard-front-camera-1-2-both",
        "wizard-front-camera-2-localize",
    ]

    pipelines = json.loads(pipeline_path.read_text(encoding="utf-8"))
    all_uuids: set[str] = set()
    assert "existing" in pipelines
    for pipeline_name, pipeline in pipelines.items():
        if not pipeline_name.startswith("wizard-"):
            continue
        validate_pipeline_connections(pipeline)
        pipeline_uuids = {operation["uuid"] for operation in pipeline}
        assert all_uuids.isdisjoint(pipeline_uuids)
        all_uuids.update(pipeline_uuids)
        assert {
            operation["action_params"].get("camera_bus_id")
            for operation in pipeline
            if "camera_bus_id" in operation["action_params"]
        } in ({"1"}, {"2"})

    first_operations = pipelines["wizard-front-camera-1-2-both"]
    detector = next(
        operation
        for operation in first_operations
        if operation["action_name"] == "object_detection.py"
    )
    publisher_operations = [
        operation
        for operation in first_operations
        if operation["action_name"] == "publish_to_networktables.py"
    ]
    publishers = {
        operation["action_params"]["target_key"] for operation in publisher_operations
    }
    assert detector["action_params"] == {"model_id": "", "device_id": "cpu"}
    assert publishers == {
        "localization/front-camera-1-2/pose",
        "localization/front-camera-1-2/detections",
    }
    assert (
        next(
            operation
            for operation in publisher_operations
            if operation["action_params"]["target_key"]
            == "localization/front-camera-1-2/detections"
        )["action_params"]["schema"]
        == "json"
    )
    transform = next(
        operation
        for operation in first_operations
        if operation["action_name"] == "camera_to_robot_pose.py"
    )
    assert transform["action_params"] == {"camera_bus_id": "1"}
    robot_output = next(
        operation
        for operation in first_operations
        if operation["action_name"] == "robot_pose_output.py"
    )
    pose_publisher = next(
        operation
        for operation in publisher_operations
        if operation["action_params"]["target_key"]
        == "localization/front-camera-1-2/pose"
    )
    transform_feeds = {
        connection["to_uuid"]: connection["from_uuid"]
        for connection in transform["connections"]
    }
    assert transform_feeds[pose_publisher["uuid"]] == transform["uuid"]
    assert transform_feeds[robot_output["uuid"]] == transform["uuid"]

    localize_operations = pipelines["wizard-front-camera-2-localize"]
    assert {
        operation["action_params"]["target_key"]
        for operation in localize_operations
        if operation["action_name"] == "publish_to_networktables.py"
    } == {
        "localization/front-camera-2/pose",
        "localization/front-camera-2/meta",
    }

    saved_general = json.loads(general_conf_path.read_text(encoding="utf-8"))
    assert saved_general["first_boot_wizard_completed"] is False
    assert saved_general["first_boot_wizard_verification_pending"] is True
    assert saved_general["network_table_address"] == "10.0.0.2"
    assert saved_general["first_boot_networktable_keys"] == [
        {"key": "localization/front-camera-1-2/pose", "required": True},
        {"key": "localization/front-camera-1-2/detections", "required": False},
        {"key": "localization/front-camera-2/pose", "required": True},
        {"key": "localization/front-camera-2/meta", "required": False},
    ]
    pending_status, _ = harness.get_first_boot_status()
    assert pending_status["required"] is False
    assert pending_status["verification_pending"] is True

    finished, finish_status = harness.finish_first_boot()
    assert finish_status == 200
    assert finished == {"completed": True, "verification_pending": False}
    final_general = json.loads(general_conf_path.read_text(encoding="utf-8"))
    assert final_general["first_boot_wizard_completed"] is True
    assert final_general["first_boot_wizard_verification_pending"] is False


def test_first_boot_status_ignores_malformed_persisted_lists(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Render status when persisted wizard list fields have invalid types."""
    pipeline_path = tmp_path / "pipeline_config.json"
    general_conf_path = tmp_path / "general_conf.json"
    pipeline_path.write_text("{}\n", encoding="utf-8")
    general_conf_path.write_text(
        json.dumps(
            {
                "first_boot_pipeline_names": None,
                "first_boot_networktable_keys": 1,
            }
        ),
        encoding="utf-8",
    )
    harness = _FirstBootHarness(pipeline_path, general_conf_path, {})
    monkeypatch.setattr(first_boot_module, "GENERAL_CONF_PATH", general_conf_path)

    payload, status = harness.get_first_boot_status()

    assert status == 200
    assert payload["pipelines"] == []
    assert payload["networktable_keys"] == []


def test_first_boot_model_library_unavailable_is_actionable(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Return a service response instead of dereferencing an unready library."""
    pipeline_path = tmp_path / "pipeline_config.json"
    general_conf_path = tmp_path / "general_conf.json"
    intrinsics_path = tmp_path / "intrinsics.json"
    pipeline_path.write_text("{}\n", encoding="utf-8")
    intrinsics_path.write_text("{}", encoding="utf-8")
    harness = _FirstBootHarness(
        pipeline_path, general_conf_path, {"1": intrinsics_path}
    )
    monkeypatch.setattr(
        first_boot_module,
        "request",
        _Request(
            {
                "network_table_address": "10.0.0.2",
                "cameras": [{"bus_id": "1", "mode": "detect", "model_id": "model-1"}],
            }
        ),
    )

    payload, status = harness.generate_first_boot_pipelines()

    assert status == 503
    assert payload == {"error": "Model library is not available"}


def test_first_boot_rolls_back_pipelines_when_metadata_save_fails(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Do not leave wizard pipelines without matching first-boot metadata."""
    pipeline_path = tmp_path / "pipeline_config.json"
    general_conf_path = tmp_path / "general_conf.json"
    intrinsics_path = tmp_path / "intrinsics.json"
    original = {"existing": []}
    pipeline_path.write_text(json.dumps(original), encoding="utf-8")
    intrinsics_path.write_text("{}", encoding="utf-8")
    harness = _FirstBootHarness(
        pipeline_path, general_conf_path, {"1": intrinsics_path}
    )
    monkeypatch.setattr(
        first_boot_module,
        "request",
        _Request(
            {
                "network_table_address": "10.0.0.2",
                "cameras": [{"bus_id": "1", "mode": "localize", "model_id": ""}],
            }
        ),
    )
    monkeypatch.setattr(
        harness,
        "_save_first_boot_state",
        lambda **_updates: (_ for _ in ()).throw(OSError("disk full")),
    )

    try:
        harness.generate_first_boot_pipelines()
    except OSError as error:
        assert str(error) == "disk full"
    else:
        raise AssertionError("metadata failure was not propagated")

    assert json.loads(pipeline_path.read_text(encoding="utf-8")) == original


def test_empty_detection_slot_loads_the_first_compatible_uploaded_model(
    monkeypatch: Any,
) -> None:
    """Keep detection idle, then start it without rebuilding the pipeline."""

    class _Library:
        def __init__(self) -> None:
            self.models: list[SimpleNamespace] = [
                SimpleNamespace(model_id="existing-model")
            ]

        def list_models(self) -> tuple[SimpleNamespace, ...]:
            return tuple(self.models)

        def resolve_artifact(self, model_id: str, device_id: str) -> object:
            assert model_id in {model.model_id for model in self.models}
            assert device_id == "cpu"
            return object()

    class _Delegate:
        def __init__(self, **kwargs: Any) -> None:
            assert kwargs["model_id"] == "uploaded-model"

        def run(self, _frame: np.ndarray) -> list[dict[str, Any]]:
            return [{"class_name": "game-piece"}]

        def update_live_settings(self, **_kwargs: Any) -> None:
            return None

    library = _Library()
    monkeypatch.setattr(
        object_detection_module, "ObjectDetectionImplementation", _Delegate
    )
    operation = ObjectDetectionDefinition(
        model_id="",
        device_id="cpu",
        device_registry=object(),
        model_library=library,  # type: ignore[arg-type]
    )
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    assert operation.run(frame) == []
    library.models.append(SimpleNamespace(model_id="uploaded-model"))
    operation._next_model_check = 0.0
    assert operation.run(frame) == [{"class_name": "game-piece"}]


def test_detect_mode_builds_a_detection_pipeline() -> None:
    """Build a detect-only pipeline from the CPU object-detection template."""
    templates = json.loads(first_boot_module._TEMPLATE_PATH.read_text(encoding="utf-8"))
    nodes, keys = build_first_boot_pipeline(
        templates,
        camera_bus_id="3",
        source_name="rear-camera",
        mode="detect",
        model_id="cpu-model",
    )
    validate_pipeline_connections(nodes)
    detector = next(
        operation
        for operation in nodes
        if operation["action_name"] == "object_detection.py"
    )
    publishers = {
        operation["action_params"]["target_key"]
        for operation in nodes
        if operation["action_name"] == "publish_to_networktables.py"
    }
    assert detector["action_params"] == {"model_id": "cpu-model", "device_id": "cpu"}
    assert publishers == {"localization/rear-camera/detections"}
    assert not any(
        operation["action_name"] in {"camera_to_robot_pose.py", "camera_pose_output.py"}
        for operation in nodes
    )
    assert keys == [{"key": "localization/rear-camera/detections", "required": True}]
