import traceback
import os
import json
import time
from pathlib import Path
from typing import Dict, Any

import ntcore

from src.config.utils.pipeline import Pipeline
from src.webui.web_server import EagleEyeInterface
from src.utils.camera_utils.camera_config_manager import CameraConfigRegistry
from src.utils.device_registry import DeviceRegistry
from src.utils.logging.logger import Logger
from src.utils.model_library import ModelLibrary
from src.utils.mx3_runtime import Mx3RuntimeCoordinator
from src.utils.colors import Colors

# Find project root by walking up from this file's directory until we find 'src'
current_path = Path(__file__).resolve().parent
project_root = current_path
while project_root.name != "src" and project_root.parent != project_root:
    project_root = project_root.parent
if project_root.name == "src":
    project_root = str(project_root.parent)
else:
    raise ValueError("Project root not found")

value_map = {"{project_root}": project_root}


def _replace_string_value(text: str) -> str:
    """Replace all placeholders in a string using the value_map."""
    result = text
    for old_value, new_value in value_map.items():
        result = result.replace(old_value, new_value)
    return result


def replace_values(config_data: dict) -> dict:
    """Recursively replace values in nested dictionaries and lists."""
    for key, value in config_data.items():
        if isinstance(value, str):
            config_data[key] = _replace_string_value(value)
        elif isinstance(value, dict):
            config_data[key] = replace_values(value)
        elif isinstance(value, list):
            config_data[key] = [
                (
                    replace_values(item)
                    if isinstance(item, dict)
                    else _replace_string_value(item) if isinstance(item, str) else item
                )
                for item in value
            ]
    return config_data


def _load_pipeline_settings() -> dict[str, dict[str, Any]]:
    """Load pipeline settings, defaulting to an empty mapping if unavailable."""
    settings_path = os.path.join(str(current_path.parent), "pipeline_settings.json")
    try:
        with open(settings_path, "r", encoding="utf-8") as settings_file:
            settings = json.load(settings_file)
    except (OSError, json.JSONDecodeError):
        return {}
    return settings if isinstance(settings, dict) else {}


def _get_device_input_camera_bus_ids(
    pipeline_name: str, pipeline_config: list[dict[str, Any]], logger: Logger
) -> list[str]:
    """Collect bus IDs from a pipeline's device_input operations.

    Args:
        pipeline_name: Name of the pipeline.
        pipeline_config: Configuration list for the pipeline.
        logger: Logger instance for logging.

    Returns:
        List of bus IDs referenced by device_input operations.
    """
    device_input_configs = [
        operation
        for operation in pipeline_config
        if operation.get("action_name") in {"device_input.py", "device_input"}
    ]

    if len(device_input_configs) == 0:
        logger.log(
            f"{Colors.YELLOW}Pipeline {pipeline_name} has no device_input operations.{Colors.RESET}"
        )
        return []

    camera_bus_ids: list[str] = []
    for device_config in device_input_configs:
        action_params = device_config.get("action_params", {})
        camera_bus_id = action_params.get("camera_bus_id")
        if isinstance(camera_bus_id, str) and camera_bus_id:
            camera_bus_ids.append(camera_bus_id)
        else:
            logger.log(
                f"{Colors.RED}Error creating pipeline {pipeline_name}: invalid camera_bus_id in device_input{Colors.RESET}"
            )

    if not camera_bus_ids:
        return []

    logger.log(
        f"{Colors.CYAN}Resolved device_input camera_bus_ids for pipeline {pipeline_name}: {camera_bus_ids}{Colors.RESET}"
    )
    return camera_bus_ids


def generate_all_pipelines(
    web_interface: EagleEyeInterface,
    network_table: ntcore.NetworkTable,
    camera_manager,
    camera_config_registry: CameraConfigRegistry,
    device_registry: DeviceRegistry,
    model_library: ModelLibrary,
    logger: Logger,
    mx3_coordinator: Mx3RuntimeCoordinator | None = None,
    pipeline_config: str | None = None,
) -> Dict[str, Pipeline]:
    """Generate all pipelines from the pipeline_config.json file.

    Args:
        web_interface: The web interface to use for the pipelines.
        network_table: The network table to use for the pipelines.
        camera_manager: The camera manager to use for the pipelines.
        camera_config_registry: Shared camera config registry.
        device_registry: Immutable startup device inventory.
        model_library: Managed inference model library.
        logger: Logger instance for logging.
        mx3_coordinator: Optional shared MX3 runtime coordinator injected into operations.
        pipeline_config: The pipeline configuration to use for the pipelines. (Optional, mostly for testing)

    Returns:
        Dictionary mapping pipeline names to Pipeline objects.
    """
    config_path = (
        os.path.join(str(current_path.parent), "pipeline_config.json")
        if pipeline_config is None
        else pipeline_config
    )
    try:
        with open(config_path, "r") as f:
            config_data = json.load(f)
    except json.JSONDecodeError:
        logger.log(
            f"{Colors.RED}Invalid pipeline config JSON in {config_path}: "
            f"{traceback.format_exc()}{Colors.RESET}"
        )
        if web_interface:
            try:
                web_interface.publish_operation_errors(
                    {
                        "pipeline_name": "Pipeline Config",
                        "errors": [
                            {
                                "uuid": "pipeline_config::json_decode",
                                "name": "Pipeline Config",
                                "message": traceback.format_exc().strip(),
                                "last_seen_ts": time.time(),
                                "count": 1,
                            }
                        ],
                    }
                )
            except Exception:
                pass
        return {}

    # Replace placeholders in the configuration data
    config_data = replace_values(config_data)
    pipeline_settings = _load_pipeline_settings()

    pipelines: Dict[str, Pipeline] = {}

    for pipeline_name, config in config_data.items():
        try:
            camera_bus_ids = _get_device_input_camera_bus_ids(
                pipeline_name, config, logger
            )
            settings = pipeline_settings.get(pipeline_name, {})
            limit_frames = (
                settings.get("limit_frames_to_camera_capture_speed", True) is not False
                if isinstance(settings, dict)
                else True
            )
            pipeline = Pipeline(
                config,
                web_interface,
                network_table,
                logger,
                device_registry,
                model_library,
                camera_manager,
                mx3_coordinator=mx3_coordinator,
                camera_config_registry=camera_config_registry,
                camera_bus_ids=camera_bus_ids,
                pipeline_name=pipeline_name,
                limit_frames_to_camera_capture_speed=limit_frames,
            )
        except Exception:
            logger.log(
                f"{Colors.RED}Error creating pipeline {pipeline_name}: {traceback.format_exc()}{Colors.RESET}"
            )
            if web_interface:
                try:
                    errors = []
                    partial_pipeline = locals().get("pipeline")
                    if (
                        isinstance(partial_pipeline, Pipeline)
                        and partial_pipeline.pipeline_name == pipeline_name
                    ):
                        errors = partial_pipeline.get_operation_errors()
                    if not errors:
                        errors = [
                            {
                                "uuid": f"pipeline_init::{pipeline_name}",
                                "name": "Pipeline Initialization",
                                "message": traceback.format_exc().strip(),
                                "last_seen_ts": time.time(),
                                "count": 1,
                            }
                        ]
                    error_payload = {
                        "pipeline_name": pipeline_name,
                        "errors": errors,
                    }
                    web_interface.publish_operation_errors(error_payload)
                except Exception:
                    pass
            continue
        pipelines[pipeline_name] = pipeline

    return pipelines
