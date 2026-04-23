import math
from typing import Any

import ntcore

from src.main_operations.definitions.base.base_class import OperationInstance


class GetNetworktablesValue(OperationInstance):
    def __init__(
        self,
        network_table: ntcore.NetworkTable,
        network_table_key: str,
    ) -> None:
        """Read data from NetworkTable and output it to downstream operations.

        Args:
            network_table: Network table to read from.
            network_table_key: Key to read from the network table.
        """
        self.network_table = network_table
        self.network_table_key = network_table_key

    def run(self, input_data: Any) -> Any:
        """Read value from NetworkTable and output it.

        Args:
            input_data: Ignored (data source operation receives None).

        Returns:
            The value read from NetworkTable, or None if key not found.
        """
        val = self.network_table.getEntry(self.network_table_key).getDouble(float("nan"))
        return None if math.isnan(val) else val

    def update_config(self, json_config: dict) -> None:
        """Update the configuration of the network tables updater.

        Args:
            json_config: JSON configuration for the network tables updater.
        """
        if "network_table_key" in json_config:
            self.network_table_key = json_config["network_table_key"]
