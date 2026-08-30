"""First-boot state, pipeline generation, and verification endpoints."""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

from flask import request

from src.utils.model_library import ModelLibraryError
from src.webui.web_server_utils.constants import GENERAL_CONF_PATH, SRC_DIR

_TEMPLATE_PATH = Path(SRC_DIR) / "webui" / "js" / "pipeline" / "pipelineTemplates.json"
_APRILTAG_MAP_PATH = (
    "{project_root}/src/webui/assets/fields/2026/apriltag_maps/"
    "FE-2026-_REBUILTTM_Playing_Field.fmap"
)
_COMPLETED_KEY = "first_boot_wizard_completed"
_VERIFICATION_PENDING_KEY = "first_boot_wizard_verification_pending"
_PIPELINES_KEY = "first_boot_pipeline_names"
_VERIFICATION_KEYS_KEY = "first_boot_networktable_keys"
_ALLOWED_MODES = frozenset({"localize", "detect", "both"})
_NT_PUBLISH_ACTION = "publish_to_networktables.py"
_PNP_ACTION = "pnp_camera_localization.py"


def _source_slug(name: str, bus_id: str) -> str:
    """Return a stable NetworkTables-safe camera source segment."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    if slug:
        return slug
    fallback = re.sub(r"[^a-zA-Z0-9]+", "-", bus_id).strip("-").lower()
    return fallback or "camera"


def _fresh_template_nodes(template: dict[str, Any]) -> list[dict[str, Any]]:
    """Clone template nodes and replace every node and connection UUID."""
    nodes = deepcopy(template["nodes"])
    uuid_map = {node["uuid"]: f"op-{uuid.uuid4().hex}" for node in nodes}
    for node in nodes:
        node["uuid"] = uuid_map[node["uuid"]]
        for connection in node.get("connections", []):
            connection["from_uuid"] = uuid_map[connection["from_uuid"]]
            connection["to_uuid"] = uuid_map[connection["to_uuid"]]
    return nodes


def _connection(
    from_uuid: str,
    from_port: str,
    to_uuid: str,
    to_port: str,
    data_type: str,
) -> dict[str, Any]:
    """Build one standard pipeline connection record."""
    return {
        "from_uuid": from_uuid,
        "from_port": from_port,
        "to_uuid": to_uuid,
        "to_port": to_port,
        "data_type": data_type,
        "is_default": False,
        "custom_waypoints": None,
    }


def build_first_boot_pipeline(
    templates: dict[str, Any],
    *,
    camera_bus_id: str,
    source_name: str,
    mode: str,
    model_id: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build one runnable camera pipeline and its verification key records.

    Args:
        templates: Parsed bundled pipeline templates.
        camera_bus_id: Stable camera bus identifier.
        source_name: Unique source segment used in NetworkTables keys.
        mode: One of ``localize``, ``detect``, or ``both``.
        model_id: Optional managed CPU model ID. An empty value leaves detection idle.

    Returns:
        Generated operation list and expected NetworkTables key records.
    """
    if mode not in _ALLOWED_MODES:
        raise ValueError(f"Unsupported camera mode: {mode!r}")

    template_id = (
        "apriltag_localization" if mode == "localize" else "object_detection_cpu"
    )
    nodes = _fresh_template_nodes(templates[template_id])
    nodes_by_action = {node["action_name"]: node for node in nodes}

    for action_name in (
        "device_input.py",
        "temporal_acceleration_preprocessor_rust.py",
        _PNP_ACTION,
        "ground_plane_intersection.py",
    ):
        node = nodes_by_action.get(action_name)
        if node is not None:
            node.setdefault("action_params", {})["camera_bus_id"] = camera_bus_id

    for action_name in ("temporal_acceleration_preprocessor_rust.py", _PNP_ACTION):
        node = nodes_by_action[action_name]
        node["action_params"]["apriltag_map_path"] = _APRILTAG_MAP_PATH

    detector = nodes_by_action.get("object_detection.py")
    if detector is not None:
        detector["action_params"].update({"model_id": model_id, "device_id": "cpu"})

    pnp_node = nodes_by_action[_PNP_ACTION]
    transform_node = nodes_by_action.pop("camera_pose_output.py")
    transform_node["action_name"] = "camera_to_robot_pose.py"
    transform_node["action_params"] = {"camera_bus_id": camera_bus_id}
    robot_output: dict[str, Any] = {
        "action_name": "robot_pose_output.py",
        "action_params": {},
        "position": {
            "x": transform_node.get("position", {}).get("x", 1920) + 340,
            "y": transform_node.get("position", {}).get("y", 100),
        },
        "uuid": f"op-{uuid.uuid4().hex}",
        "connections": [],
    }

    pose_publisher = next(
        node
        for node in nodes
        if node["action_name"] == _NT_PUBLISH_ACTION
        and node.get("action_params", {}).get("target_key") == "camera_pose"
    )
    pose_key = f"localization/{source_name}"
    pose_publisher["action_params"].update(
        {"target_key": pose_key, "schema": "pose3d", "data_path": []}
    )
    pose_publisher["position"]["x"] = robot_output["position"]["x"]
    pnp_node["connections"] = [
        item
        for item in pnp_node["connections"]
        if item["to_uuid"] != pose_publisher["uuid"]
    ]
    transform_node["connections"] = [
        _connection(
            transform_node["uuid"],
            "robot_pose",
            robot_output["uuid"],
            "pose",
            "robot_pose",
        ),
        _connection(
            transform_node["uuid"],
            "robot_pose",
            pose_publisher["uuid"],
            "data",
            "robot_pose",
        ),
    ]
    nodes.append(robot_output)

    verification_keys: list[dict[str, Any]] = []
    if mode == "detect":
        nodes = [node for node in nodes if node["uuid"] != pose_publisher["uuid"]]
        transform_node["connections"] = [
            item
            for item in transform_node["connections"]
            if item["to_uuid"] != pose_publisher["uuid"]
        ]
    else:
        verification_keys.append({"key": pose_key, "required": True})

    if mode != "localize":
        detection_publisher = next(
            node
            for node in nodes
            if node["action_name"] == _NT_PUBLISH_ACTION
            and node.get("action_params", {}).get("target_key") == "detected_objects"
        )
        detection_key = f"detections/{source_name}"
        detection_publisher["action_params"].update(
            {"target_key": detection_key, "schema": "json", "data_path": []}
        )
        verification_keys.append(
            {"key": detection_key, "required": mode == "detect" or bool(model_id)}
        )

    return nodes, verification_keys


def _unique_camera_sources(
    available_by_bus_id: dict[str, dict[str, str]],
    reserved_target_keys: set[str],
) -> dict[str, str]:
    """Assign unique NetworkTables source names for each active camera."""
    grouped: dict[str, list[str]] = {}
    for bus_id, camera in available_by_bus_id.items():
        grouped.setdefault(_source_slug(camera["name"], bus_id), []).append(bus_id)

    sources_by_bus_id: dict[str, str] = {}
    used_sources: set[str] = set()
    for base_source, grouped_bus_ids in grouped.items():
        for bus_id in sorted(grouped_bus_ids):
            source = (
                f"{base_source}-{_source_slug('', bus_id)}"
                if len(grouped_bus_ids) > 1
                else base_source
            )
            candidate = source
            suffix = 2
            while (
                candidate in used_sources
                or f"localization/{candidate}" in reserved_target_keys
                or f"detections/{candidate}" in reserved_target_keys
            ):
                candidate = f"{source}-{suffix}"
                suffix += 1
            sources_by_bus_id[bus_id] = candidate
            used_sources.add(candidate)
    return sources_by_bus_id


def _validated_camera_payload(
    item: Any,
    available_by_bus_id: dict[str, dict[str, str]],
    seen_bus_ids: set[str],
) -> dict[str, str]:
    """Return one validated camera record or raise ValueError."""
    if not isinstance(item, dict) or set(item) - {"bus_id", "mode", "model_id"}:
        raise ValueError("Each camera requires bus_id, mode, and optional model_id")
    bus_id = item.get("bus_id")
    mode = item.get("mode")
    model_id = item.get("model_id", "")
    if not isinstance(bus_id, str) or bus_id not in available_by_bus_id:
        raise ValueError(f"Camera {bus_id!r} is not active")
    if bus_id in seen_bus_ids:
        raise ValueError(f"Camera {bus_id!r} was selected more than once")
    if mode not in _ALLOWED_MODES:
        raise ValueError(f"Unsupported camera mode: {mode!r}")
    if not isinstance(model_id, str):
        raise ValueError("model_id must be a string")
    model_id = model_id.strip()
    if mode == "detect" and not model_id:
        raise ValueError("Detect-only pipelines require a CPU model")
    return {"bus_id": bus_id, "mode": mode, "model_id": model_id}


class FirstBootMixin:
    """Expose the guided first-boot setup contract to the WebUI."""

    def _write_general_conf(self, config: dict[str, Any]) -> None:
        """Atomically persist the complete general configuration."""
        GENERAL_CONF_PATH.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=GENERAL_CONF_PATH.parent,
            prefix=".general_conf.",
            suffix=".tmp",
            text=True,
        )
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
                json.dump(config, stream, indent=4)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, GENERAL_CONF_PATH)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    def _save_first_boot_state(self, **updates: Any) -> dict[str, Any]:
        """Merge first-boot fields into persistent general configuration."""
        with self._general_conf_lock:
            config = self._read_general_conf()
            config.update(updates)
            self._write_general_conf(config)
        return config

    def _first_boot_camera_records(self) -> list[dict[str, str]]:
        """Return active cameras in the wizard's stable public shape."""
        with self.frame_list_structure_lock:
            available_cameras = list(self.available_cameras.items())
        cameras: list[dict[str, str]] = []
        for camera_name, camera_info in available_cameras:
            if not isinstance(camera_info, dict):
                continue
            bus_id = str(camera_info.get("bus_id") or camera_info.get("id") or "")
            if not bus_id:
                continue
            cameras.append(
                {
                    "name": str(camera_name),
                    "bus_id": bus_id,
                    "stream_name": str(camera_info.get("name") or camera_name),
                }
            )
        return sorted(cameras, key=lambda camera: (camera["name"], camera["bus_id"]))

    def _networktable_topic_names(self) -> set[str]:
        """Return currently announced NetworkTables topic names without table prefix."""
        instance = getattr(self, "network_table_instance", None)
        if instance is None:
            return set()
        try:
            names = {str(topic.getName()) for topic in instance.getTopics()}
        except (AttributeError, RuntimeError) as error:
            self.log(
                f"Failed reading NetworkTables topics for setup verification: {error}"
            )
            return set()
        return {
            name.removeprefix("/EagleEye/").removeprefix("EagleEye/") for name in names
        }

    def get_first_boot_status(self) -> tuple[dict[str, Any], int]:
        """Return first-boot state plus live pipeline and NetworkTables checks."""
        config = self._read_general_conf()
        completed = config.get(_COMPLETED_KEY) is True
        verification_pending = config.get(_VERIFICATION_PENDING_KEY) is True
        pipeline_config = self._load_pipeline_config_file()
        required = not completed and not verification_pending and not pipeline_config
        raw_pipeline_names = config.get(_PIPELINES_KEY, [])
        pipeline_names = (
            [name for name in raw_pipeline_names if isinstance(name, str)]
            if isinstance(raw_pipeline_names, list)
            else []
        )
        pipeline_objects = self.pipeline_objects_callback()
        pipelines = []
        for pipeline_name in pipeline_names:
            pipeline = pipeline_objects.get(pipeline_name)
            try:
                active = pipeline is not None and bool(pipeline.is_active())
            except (AttributeError, RuntimeError):
                active = False
            pipelines.append({"name": pipeline_name, "active": active})

        topic_names = self._networktable_topic_names()
        raw_expected_keys = config.get(_VERIFICATION_KEYS_KEY, [])
        expected_keys = (
            [
                item
                for item in raw_expected_keys
                if isinstance(item, dict) and isinstance(item.get("key"), str)
            ]
            if isinstance(raw_expected_keys, list)
            else []
        )
        return {
            "required": required,
            "completed": completed,
            "verification_pending": verification_pending,
            "cameras": self._first_boot_camera_records(),
            "pipelines": pipelines,
            "network_table": self._build_network_table_status(),
            "networktable_keys": [
                {
                    "key": item["key"],
                    "required": item.get("required") is True,
                    "present": item["key"] in topic_names,
                }
                for item in expected_keys
            ],
        }, 200

    def skip_first_boot(self) -> tuple[dict[str, bool], int]:
        """Persist that the user skipped automatic first-boot display."""
        self._save_first_boot_state(
            **{_COMPLETED_KEY: True, _VERIFICATION_PENDING_KEY: False}
        )
        return {"completed": True, "skipped": True}, 200

    def finish_first_boot(self) -> tuple[dict[str, bool], int]:
        """Persist that the user reached and accepted live verification."""
        self._save_first_boot_state(
            **{_COMPLETED_KEY: True, _VERIFICATION_PENDING_KEY: False}
        )
        return {"completed": True, "verification_pending": False}, 200

    def generate_first_boot_pipelines(self) -> tuple[dict[str, Any], int]:
        """Validate wizard input and persist template-generated camera pipelines."""
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return {"error": "Expected JSON object payload"}, 400
        if set(payload) - {"network_table_address", "cameras"}:
            return {"error": "Unexpected setup fields"}, 400

        address = payload.get("network_table_address")
        camera_payloads = payload.get("cameras")
        if not isinstance(address, str) or not address.strip():
            return {"error": "NetworkTables address is required"}, 400
        if not isinstance(camera_payloads, list) or not camera_payloads:
            return {"error": "At least one camera is required"}, 400

        with self._pipeline_settings_lock:
            return self._generate_first_boot_pipelines_locked(address, camera_payloads)

    def _generate_first_boot_pipelines_locked(
        self, address: str, camera_payloads: list[Any]
    ) -> tuple[dict[str, Any], int]:
        """Generate pipelines while holding the pipeline-config lock."""
        registry = getattr(self, "camera_config_registry", None)
        if registry is None:
            return {"error": "Camera configuration is not ready yet"}, 503
        library = getattr(self, "model_library", None)

        available_by_bus_id = {
            camera["bus_id"]: camera for camera in self._first_boot_camera_records()
        }
        config = self._read_general_conf()
        current_pipelines = self._load_pipeline_config_file()
        current_pipelines_before_update = deepcopy(current_pipelines)
        old_pipeline_names = {
            name for name in config.get(_PIPELINES_KEY, []) if isinstance(name, str)
        }
        retained_pipelines = {
            name: operations
            for name, operations in current_pipelines.items()
            if name not in old_pipeline_names
        }
        reserved_pipeline_names = set(retained_pipelines)
        reserved_target_keys = {
            target_key
            for operations in retained_pipelines.values()
            for operation in operations
            if isinstance(operation, dict)
            and operation.get("action_name") == _NT_PUBLISH_ACTION
            and isinstance(operation.get("action_params"), dict)
            and isinstance(
                target_key := operation["action_params"].get("target_key"), str
            )
        }
        sources_by_bus_id = _unique_camera_sources(
            available_by_bus_id, reserved_target_keys
        )

        parsed_cameras: list[dict[str, str]] = []
        seen_bus_ids: set[str] = set()
        for camera_payload in camera_payloads:
            try:
                camera = _validated_camera_payload(
                    camera_payload, available_by_bus_id, seen_bus_ids
                )
            except ValueError as error:
                return {"error": str(error)}, 400
            if camera["model_id"]:
                if library is None:
                    return {"error": "Model library is not available"}, 503
                try:
                    library.resolve_artifact(camera["model_id"], "cpu")
                except ModelLibraryError as error:
                    return {"error": str(error)}, 400
            camera_config = registry.get_config(camera["bus_id"])
            if (
                not camera_config.intrinsics_path
                or not Path(camera_config.intrinsics_path).is_file()
            ):
                return {
                    "error": f"Camera {camera['bus_id']!r} needs intrinsics calibration"
                }, 400
            parsed_cameras.append(camera)
            seen_bus_ids.add(camera["bus_id"])

        templates = json.loads(_TEMPLATE_PATH.read_text(encoding="utf-8"))
        generated: dict[str, list[dict[str, Any]]] = {}
        verification_keys: list[dict[str, Any]] = []
        for camera in parsed_cameras:
            source_name = sources_by_bus_id[camera["bus_id"]]
            pipeline_name = f"wizard-{source_name}-{camera['mode']}"
            suffix = 2
            while (
                pipeline_name in reserved_pipeline_names or pipeline_name in generated
            ):
                pipeline_name = f"wizard-{source_name}-{camera['mode']}-{suffix}"
                suffix += 1
            nodes, expected_keys = build_first_boot_pipeline(
                templates,
                camera_bus_id=camera["bus_id"],
                source_name=source_name,
                mode=camera["mode"],
                model_id=camera["model_id"],
            )
            generated[pipeline_name] = nodes
            verification_keys.extend(expected_keys)

        for old_name in old_pipeline_names:
            current_pipelines.pop(old_name, None)
        current_pipelines.update(generated)
        self._write_pipeline_config_file(current_pipelines)
        try:
            self._save_first_boot_state(
                network_table_address=address.strip(),
                **{
                    _COMPLETED_KEY: False,
                    _VERIFICATION_PENDING_KEY: True,
                    _PIPELINES_KEY: list(generated),
                    _VERIFICATION_KEYS_KEY: verification_keys,
                },
            )
        except Exception:
            # Metadata determines whether the wizard is shown. If it cannot be
            # persisted, restore the exact pipeline snapshot so an untracked
            # wizard pipeline cannot suppress setup on the next status check.
            self._write_pipeline_config_file(current_pipelines_before_update)
            raise
        self.restart_required_for_config = True
        return {
            "completed": False,
            "verification_pending": True,
            "pipelines": list(generated),
            "networktable_keys": verification_keys,
            "restart_required": True,
            "runtime_id": getattr(self, "runtime_id", ""),
        }, 200
