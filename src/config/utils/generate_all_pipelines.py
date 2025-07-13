import os
import json
from typing import List
from src.config.utils.pipeline import Pipeline
from src.webui.web_server import EagleEyeInterface

current_path = os.path.dirname(__file__)

def generate_all_pipelines(web_interface: EagleEyeInterface) -> List[Pipeline]:
    """Generate all pipelines from the pipeline_config.json file.

    Args:
        web_interface: The web interface to use for the pipelines.

    Returns:
        A list of Pipeline objects.
    """
    with open(os.path.join(os.path.dirname(current_path), "pipeline_config.json"), "r") as f:
        pipeline_config = json.load(f)

    pipelines = []
    for camera_bus_id in pipeline_config.keys():
        pipeline = Pipeline(pipeline_config[camera_bus_id], web_interface, camera_bus_id)
        pipelines.append(pipeline)

    return pipelines
