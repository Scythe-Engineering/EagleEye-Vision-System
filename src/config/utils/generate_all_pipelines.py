import traceback
import os
import json
from pathlib import Path
from typing import Dict, Any
from src.config.utils.pipeline import Pipeline
from src.webui.web_server import EagleEyeInterface
from src.utils.device_management_utils.compute_pool import ComputePool
from src.utils.logging.logger import Logger
from src.utils.colors import Colors
from networktables import NetworkTable

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
                replace_values(item)
                if isinstance(item, dict)
                else _replace_string_value(item)
                if isinstance(item, str)
                else item
                for item in value
            ]
    return config_data


def _get_device_input_camera_names(
    pipeline_name: str, pipeline_config: list[dict[str, Any]], logger: Logger
) -> list[str]:
    """Collect camera names from a pipeline's device_input operations.

    Args:
        pipeline_name: Name of the pipeline.
        pipeline_config: Configuration list for the pipeline.
        logger: Logger instance for logging.

    Returns:
        List of camera names referenced by device_input operations.
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

    camera_names: list[str] = []
    for device_config in device_input_configs:
        action_params = device_config.get("action_params", {})
        camera_name = action_params.get("camera_name")
        if isinstance(camera_name, str) and camera_name:
            camera_names.append(camera_name)
        else:
            logger.log(
                f"{Colors.RED}Error creating pipeline {pipeline_name}: invalid camera_name in device_input{Colors.RESET}"
            )

    if not camera_names:
        return []

    logger.log(
        f"{Colors.CYAN}Resolved device_input cameras for pipeline {pipeline_name}: {camera_names}{Colors.RESET}"
    )
    return camera_names


def generate_all_pipelines(
    web_interface: EagleEyeInterface,
    compute_pool: ComputePool,
    network_table: NetworkTable,
    camera_manager,
    logger: Logger,
    pipeline_config: str | None = None,
) -> Dict[str, Pipeline]:
    """Generate all pipelines from the pipeline_config.json file.

    Args:
        web_interface: The web interface to use for the pipelines.
        compute_pool: The compute pool to use for the pipelines.
        network_table: The network table to use for the pipelines.
        camera_manager: The camera manager to use for the pipelines.
        logger: Logger instance for logging.
        pipeline_config: The pipeline configuration to use for the pipelines. (Optional, mostly for testing)

    Returns:
        Dictionary mapping pipeline names to Pipeline objects.
    """
    if pipeline_config is None:
        with open(
            os.path.join(str(current_path.parent), "pipeline_config.json"), "r"
        ) as f:
            config_data = json.load(f)
    else:
        with open(pipeline_config, "r") as f:
            config_data = json.load(f)

    # Replace placeholders in the configuration data
    config_data = replace_values(config_data)

    pipelines: Dict[str, Pipeline] = {}
    pipeline_count = 0

    for pipeline_name, config in config_data.items():
        try:
            camera_names = _get_device_input_camera_names(
                pipeline_name, config, logger
            )
            pipeline = Pipeline(
                config,
                web_interface,
                compute_pool,
                network_table,
                logger,
                camera_manager,
                camera_bus_id=camera_names[0] if len(camera_names) == 1 else None,
                camera_bus_ids=camera_names,
                pipeline_name=pipeline_name,
            )
        except Exception as error:
            logger.log(
                f"{Colors.RED}Error creating pipeline {pipeline_name}: {traceback.format_exc()}{Colors.RESET}"
            )
            if web_interface:
                try:
                    error_payload = {
                        "pipeline_name": pipeline_name,
                        "errors": [
                            {
                                "uuid": f"pipeline_init::{pipeline_name}",
                                "name": "Pipeline Initialization",
                                "message": traceback.format_exc().strip(),
                                "last_seen_ts": time.time(),
                                "count": 1,
                            }
                        ],
                    }
                    web_interface.publish_operation_errors(error_payload)
                except Exception:
                    pass
            continue
        pipelines[pipeline_name] = pipeline
        pipeline_count += 1

    for device in compute_pool.get_compute_devices_by_type("MX3"):
        device.connect_streams(pipeline_count)

    return pipelines
