import numpy as np
from typing import Any

from networktables import NetworkTable
from src.config.utils.pipeline import Pipeline


def _snake_to_camel(snake_str: str) -> str:
    """Convert snake_case string to CamelCase.

    Args:
        snake_str: String in snake_case format.

    Returns:
        String in CamelCase format.
    """
    components = snake_str.split("_")
    return "".join(word.capitalize() for word in components)


class UpdateAttributeWithNetworktables:
    def __init__(
        self,
        pipeline: Pipeline,
        network_table: NetworkTable,
        action_name: str,
        attribute_name: str,
        network_table_key: str,
    ) -> None:
        """Output the robot pose to the web interface.

        Args:
            pipeline: Pipeline to update.
            network_table: Network table to update.
            action_name: Name of the action to update.
            attribute_name: Name of the attribute to update.
            network_table_key: Key to update in the network table.
        """
        self.pipeline = pipeline
        self.network_table = network_table
        self.action_name = _snake_to_camel(action_name)
        self.attribute_name = attribute_name
        self.network_table_key = network_table_key

    def run(self, passthrough_data: Any) -> Any:
        """Output the robot pose to the web interface."""
        data = self.network_table.getNumber(self.network_table_key, None)
        if data is not None:
            try:
                action_object = self.pipeline.get_operation_by_class_name(
                    self.action_name
                )
                if action_object is not None:
                    action_object.set_attribute(self.attribute_name, data)
                else:
                    raise ValueError(f"Action {self.action_name} not found")
            except AttributeError as e:
                raise ValueError(
                    f"Action {self.action_name} does not have a set_attribute method: {e}"
                )
        return passthrough_data

    def update_config(self, json_config: dict) -> None:
        """Update the configuration of the network tables updater. Only live-updatable parameters are changed.

        Args:
            json_config: JSON configuration for the network tables updater.
        """
        if "action_name" in json_config:
            self.action_name = _snake_to_camel(json_config["action_name"])
        if "attribute_name" in json_config:
            self.attribute_name = json_config["attribute_name"]
        if "network_table_key" in json_config:
            self.network_table_key = json_config["network_table_key"]

    def visualize(self, frame: np.ndarray) -> None:
        """Visualize the network tables updater outputs.

        This operation is a pass-through that doesn't modify data,
        so no frame visualization is available.

        Args:
            frame: Input frame (unused).

        Returns:
            None - no visualization available for pass-through operations.
        """
        return None
