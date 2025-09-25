from typing import Any

import numpy as np

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


class BackPropagate:
    def __init__(
        self,
        pipeline: Pipeline,
        action_name: str,
    ) -> None:
        """Back propagate input data to a previous operation in the pipeline.

        This operation takes its input and calls the back_propagate_input method
        on a specified operation earlier in the pipeline, then passes the input through.

        Args:
            pipeline: Pipeline containing the operations.
            action_name: Name of the action to back propagate to (in snake_case).
        """
        self.pipeline = pipeline
        self.action_name = _snake_to_camel(action_name)

    def run(self, input_data: Any) -> Any:
        """Back propagate the input data to the specified operation.

        Attempts to call a back_propagate_input method on the target operation
        to notify it of the back-propagated input, then passes the original input through.

        Args:
            input_data: Input data to back propagate.

        Returns:
            The original input_data (pass-through behavior).
        """
        target_operation = self.pipeline.get_operation_by_class_name(self.action_name)
        if target_operation is not None:
            if hasattr(target_operation, "back_propagate_input"):
                try:
                    target_operation.back_propagate_input(input_data)
                except Exception as e:
                    raise ValueError(
                        f"Error calling back_propagate_input on target operation {self.action_name}: {e}"
                    )
            else:
                raise ValueError(
                    f"Target operation {self.action_name} does not have a back_propagate_input method"
                )
        else:
            raise ValueError(
                f"Target operation {self.action_name} not found in pipeline"
            )

        return input_data

    def update_config(self, json_config: dict) -> None:
        """Update the configuration of the back propagation operation.

        Only live-updatable parameters are changed.

        Args:
            json_config: JSON configuration for the back propagation operation.
        """
        if "action_name" in json_config:
            self.action_name = _snake_to_camel(json_config["action_name"])

    def visualize(self, frame: np.ndarray) -> None:
        """Visualize the back propagation operation outputs.

        This operation is a pass-through that may modify data flow,
        so no frame visualization is available.

        Args:
            frame: Input frame (unused).

        Returns:
            None - no visualization available for pass-through operations.
        """
        return None
