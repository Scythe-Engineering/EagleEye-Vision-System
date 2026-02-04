"""Resolve configuration defaults for operations."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

from tests.utils.operation_discovery import OperationSpec


@dataclass(frozen=True)
class ConfigDefaultsResult:
    """Resolved defaults with source details."""

    action_params: Dict[str, Any]
    sources: Dict[str, str]


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _project_root() -> Path:
    override = os.environ.get("EAGLEEYE_TEST_PROJECT_ROOT")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2]


def _resolve_placeholder(value: Any, root: Path) -> Any:
    if isinstance(value, str):
        return value.replace("{project_root}", str(root))
    return value


def _config_def_path(root: Path, action_name: str, is_secondary: bool) -> Path:
    if is_secondary:
        return (
            root
            / "src"
            / "secondary_operations"
            / "config_data"
            / f"{action_name}_config_def.json"
        )
    return (
        root
        / "src"
        / "main_operations"
        / "definitions"
        / "config_data"
        / f"{action_name}_config_def.json"
    )


def _pipeline_config_path(root: Path) -> Path:
    return root / "src" / "config" / "pipeline_config.json"


def _extract_defaults_from_config_def(config_def: Dict[str, Any], root: Path) -> Tuple[Dict[str, Any], Dict[str, str]]:
    action_params: Dict[str, Any] = {}
    sources: Dict[str, str] = {}
    parameters = config_def.get("parameters", {}) if isinstance(config_def, dict) else {}
    for name, meta in parameters.items():
        if isinstance(meta, dict) and "default" in meta:
            action_params[name] = _resolve_placeholder(meta["default"], root)
            sources[name] = "config_def"
    return action_params, sources


def _extract_defaults_from_pipeline_config(config_data: Dict[str, Any], action_name: str) -> Dict[str, Any]:
    for pipeline_config in config_data.values():
        for operation in pipeline_config:
            if operation.get("action_name") == f"{action_name}.py" or operation.get(
                "action_name"
            ) == action_name:
                return operation.get("action_params", {})
    return {}


def resolve_operation_defaults(spec: OperationSpec) -> ConfigDefaultsResult:
    """Resolve action_params defaults for the given operation."""

    root = _project_root()
    action_params: Dict[str, Any] = {}
    sources: Dict[str, str] = {}

    config_def_path = _config_def_path(root, spec.action_name, spec.is_secondary)
    if config_def_path.exists():
        config_def = _load_json(config_def_path)
        params, param_sources = _extract_defaults_from_config_def(config_def, root)
        action_params.update(params)
        sources.update(param_sources)

    pipeline_path = _pipeline_config_path(root)
    if pipeline_path.exists():
        pipeline_config = _load_json(pipeline_path)
        pipeline_defaults = _extract_defaults_from_pipeline_config(
            pipeline_config, spec.action_name
        )
        for key, value in pipeline_defaults.items():
            if key not in action_params:
                action_params[key] = _resolve_placeholder(value, root)
                sources[key] = "pipeline_config"

    action_params = _apply_overrides(action_params, sources, root)
    return ConfigDefaultsResult(action_params=action_params, sources=sources)


def _apply_overrides(
    action_params: Dict[str, Any], sources: Dict[str, str], root: Path
) -> Dict[str, Any]:
    if "color_ranges" in action_params and action_params["color_ranges"] is None:
        action_params["color_ranges"] = [
            {
                "name": "test",
                "class_id": 0,
                "lower_hsv": [0, 0, 0],
                "upper_hsv": [179, 255, 255],
            }
        ]
        sources.setdefault("color_ranges", "override")
    overrides: Dict[str, Any] = {
        "camera_parameters_path": os.environ.get(
            "EAGLEEYE_TEST_CAMERA_PARAMETERS_PATH",
            str(root / "files" / "camera_parameters_path" / "intrinsics.json"),
        ),
        "apriltag_map_path": os.environ.get(
            "EAGLEEYE_TEST_APRILTAG_MAP_PATH",
            str(root / "files" / "apriltag_map_path" / "frc2025r2.json"),
        ),
        "camera_name": os.environ.get("EAGLEEYE_TEST_CAMERA_NAME", "test_camera"),
        "network_table_key": os.environ.get(
            "EAGLEEYE_TEST_NETWORK_TABLE_KEY", "test_key"
        ),
        "target_key": os.environ.get("EAGLEEYE_TEST_NETWORK_TABLE_KEY", "test_key"),
    }
    if "camera_parameters_path" not in action_params and "intrinsics_path" in action_params:
        overrides["intrinsics_path"] = overrides["camera_parameters_path"]
    for key, value in overrides.items():
        existing = action_params.get(key)
        if isinstance(existing, str) and existing:
            if key.endswith("_path") and Path(existing).exists():
                continue
        elif existing:
            continue
        action_params[key] = value
        sources.setdefault(key, "override")
    return action_params
