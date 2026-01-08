from collections.abc import Sequence
from typing import Any

from networktables import NetworkTable
from src.utils.flatpack_schema.registry import registry
from src.secondary_operations.base_class import SecondaryOperation


class PublishToNetworktables(SecondaryOperation):
    def __init__(
        self,
        network_table: NetworkTable,
        target_key: str,
        data_path: str | Sequence[str] | None = None,
    ) -> None:
        """Initialize the NetworkTables publisher operation.

        Args:
            network_table: NetworkTables destination table.
            target_key: Entry key to publish the value to.
            data_path: Optional path describing where to extract the value from the pipeline data.
        """
        self.network_table = network_table
        self.target_key = target_key
        self.data_path_tokens = self._normalize_path(data_path)

    def run(self, data: Any) -> Any:
        """Publish the selected value to NetworkTables.

        Args:
            data: Pipeline data that may contain the value to publish.

        Returns:
            Original pipeline data unchanged.
        """
        value_to_publish = self._select_value(data)
        if value_to_publish is not None:
            self._publish_value(value_to_publish)
        return data

    def update_config(self, json_config: dict) -> None:
        """Apply live configuration updates.

        Args:
            json_config: Dictionary containing configuration fields to update.
        """
        if "target_key" in json_config:
            self.target_key = json_config["target_key"]
        if "data_path" in json_config:
            self.data_path_tokens = self._normalize_path(json_config["data_path"])

    def _normalize_path(self, data_path: str | Sequence[str] | None) -> list[str | int]:
        """Normalize the provided path configuration into a token list.

        Args:
            data_path: Path definition supplied through configuration.

        Returns:
            List of path tokens represented as strings or integers.
        """
        if data_path is None:
            return []
        if isinstance(data_path, str):
            raw_tokens = [token for token in data_path.split(".") if token]
        else:
            raw_tokens = list(data_path)
        normalized_tokens: list[str | int] = []
        for token in raw_tokens:
            if isinstance(token, int):
                normalized_tokens.append(token)
            else:
                token_str = str(token).strip()
                if token_str.isdigit():
                    normalized_tokens.append(int(token_str))
                else:
                    normalized_tokens.append(token_str)
        return normalized_tokens

    def _select_value(self, data: Any) -> Any:
        """Extract the value from the incoming data according to the configured path.

        Args:
            data: Pipeline data used as the source for extraction.

        Returns:
            The extracted value or None when the path cannot be resolved.
        """
        if not self.data_path_tokens:
            return data

        if self._should_extract_sequence_field(data):
            return self._extract_sequence_field(data)

        current_value = data
        for token in self.data_path_tokens:
            if isinstance(token, int):
                if isinstance(current_value, Sequence):
                    try:
                        current_value = current_value[token]
                    except (IndexError, TypeError):
                        return None
                else:
                    return None
            else:
                if isinstance(current_value, dict) and token in current_value:
                    current_value = current_value[token]
                else:
                    return None
        return current_value

    def _publish_value(self, value: Any) -> None:
        """Publish the provided value to NetworkTables using Flatpack serialization.

        Args:
            value: Value selected for publication.

        Raises:
            ValueError: Raised when no Flatpack schema matches the value.
        """
        serialized_bytes, _ = registry.serialize(value)
        self.network_table.putRaw(self.target_key, serialized_bytes)

    def _should_extract_sequence_field(self, data: Any) -> bool:
        """Determine whether extraction should pull fields from a sequence of dicts."""
        return (
            len(self.data_path_tokens) == 1
            and isinstance(self.data_path_tokens[0], str)
            and isinstance(data, Sequence)
            and not isinstance(data, (str, bytes, bytearray))
        )

    def _extract_sequence_field(self, data: Sequence[Any]) -> Any:
        """Extract a field from each mapping inside a sequence."""
        field_name = self.data_path_tokens[0]
        try:
            return [
                item[field_name]
                for item in data
                if isinstance(item, dict) and field_name in item
            ]
        except (KeyError, TypeError):
            return None
