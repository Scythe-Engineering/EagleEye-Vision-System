import os
import json
from typing import Dict
from src.config.utils.pipeline import Pipeline
from src.webui.web_server import EagleEyeInterface
from src.utils.device_management_utils.compute_pool import ComputePool
from networktables import NetworkTable

current_path = os.path.dirname(__file__)


def generate_all_pipelines(
    web_interface: EagleEyeInterface,
    compute_pool: ComputePool,
    network_table: NetworkTable,
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
        with open(
            os.path.join(os.path.dirname(current_path), "pipeline_config.json"), "r"
        ) as f:
            config_data = json.load(f)
    else:
        with open(pipeline_config, "r") as f:
            config_data = json.load(f)

    pipelines: Dict[str, Dict[str, Pipeline]] = {}
    pipeline_count = 0

    for camera_name in config_data.keys():
        pipelines[camera_name] = {}
        for pipeline_name in config_data[camera_name].keys():
            config = config_data[camera_name][pipeline_name]

            pipeline = Pipeline(
                config, web_interface, camera_name, compute_pool, network_table
            )
            pipelines[camera_name][pipeline_name] = pipeline
            pipeline_count += 1

    for device in compute_pool.get_compute_devices_by_type("MX3"):
        device.connect_streams(pipeline_count)

    return pipelines
