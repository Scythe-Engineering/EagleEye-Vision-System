import os
import json
from pathlib import Path
from typing import Dict
from src.config.utils.pipeline import Pipeline
from src.webui.web_server import EagleEyeInterface
from src.utils.device_management_utils.compute_pool import ComputePool
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


def generate_all_pipelines(
    web_interface: EagleEyeInterface,
    compute_pool: ComputePool,
    network_table: NetworkTable,
    camera_manager,
    pipeline_config: str | None = None,
) -> Dict[str, Dict[str, Pipeline]]:
    """Generate all pipelines from the pipeline_config.json file.

    Args:
        web_interface: The web interface to use for the pipelines.
        compute_pool: The compute pool to use for the pipelines.
        network_table: The network table to use for the pipelines.
        pipeline_config: The pipeline configuration to use for the pipelines. (Optional, mostly for testing)

    Returns:
        A list of Pipeline objects.
    """
    if pipeline_config is None:
        with open(os.path.join(str(current_path), "pipeline_config.json"), "r") as f:
            config_data = json.load(f)
    else:
        with open(pipeline_config, "r") as f:
            config_data = json.load(f)

    # Replace placeholders in the configuration data
    config_data = replace_values(config_data)

    pipelines: Dict[str, Dict[str, Pipeline]] = {}
    pipeline_count = 0

    for camera_name in config_data.keys():
        pipelines[camera_name] = {}
        for pipeline_name in config_data[camera_name].keys():
            config = config_data[camera_name][pipeline_name]

            pipeline = Pipeline(
                config,
                web_interface,
                camera_name,
                compute_pool,
                network_table,
                camera_manager,
            )
            pipelines[camera_name][pipeline_name] = pipeline
            pipeline_count += 1

    for device in compute_pool.get_compute_devices_by_type("MX3"):
        device.connect_streams(pipeline_count)

    return pipelines
