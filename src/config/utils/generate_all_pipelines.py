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


def _resolve_device_input_camera_name(
    pipeline_name: str, pipeline_config: list[dict[str, Any]], logger: Logger
) -> str | None:
    """Resolve the camera name from a pipeline's device_input operation.

    Args:
        pipeline_name: Name of the pipeline.
        pipeline_config: Configuration list for the pipeline.
        logger: Logger instance for logging.

    Returns:
        Camera name if valid, otherwise None.
    """
    device_input_configs = [
        operation
        for operation in pipeline_config
        if operation.get("action_name") in {"device_input.py", "device_input"}
    ]

    if len(device_input_configs) == 0:
        logger.log(
            f"{Colors.RED}Error creating pipeline {pipeline_name}: missing device_input operation{Colors.RESET}"
        )
        return None

    if len(device_input_configs) > 1:
        logger.log(
            f"{Colors.RED}Error creating pipeline {pipeline_name}: multiple device_input operations found{Colors.RESET}"
        )
        return None

    action_params = device_input_configs[0].get("action_params", {})
    camera_name = action_params.get("camera_name")

    if not isinstance(camera_name, str) or not camera_name:
        logger.log(
            f"{Colors.RED}Error creating pipeline {pipeline_name}: invalid camera_name in device_input{Colors.RESET}"
        )
        return None

    logger.log(
        f"{Colors.CYAN}Resolved camera '{camera_name}' for pipeline {pipeline_name}{Colors.RESET}"
    )
    return camera_name


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
            camera_name = _resolve_device_input_camera_name(
                pipeline_name, config, logger
            )
            if camera_name is None:
                continue
            pipeline = Pipeline(
                config,
                web_interface,
                compute_pool,
                network_table,
                logger,
                camera_manager,
                camera_bus_id=camera_name,
            )
        except Exception as _:
            logger.log(
                f"{Colors.RED}Error creating pipeline {pipeline_name}: {traceback.format_exc()}{Colors.RESET}"
            )
            continue
        pipelines[pipeline_name] = pipeline
        pipeline_count += 1

    for device in compute_pool.get_compute_devices_by_type("MX3"):
        device.connect_streams(pipeline_count)

    return pipelines
