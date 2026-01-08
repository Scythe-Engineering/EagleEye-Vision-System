from typing import Any, Optional

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
        target_pipeline_name: str | None = None,
    ) -> None:
        """Back propagate input data to a previous operation in the pipeline.

        This operation takes its input and calls the back_propagate_input method
        on a specified operation earlier in the pipeline, then passes the input through.

        Args:
            pipeline: Pipeline containing the operations.
            action_name: Name of the action to back propagate to (in snake_case).
            target_pipeline_name: Optional name of another pipeline to target for back propagation.
        """
        self.pipeline = pipeline
        self.action_name = _snake_to_camel(action_name)
        self.target_pipeline_name: str | None = (
            target_pipeline_name if target_pipeline_name else None
        )
        self._cached_target_operation: Optional[Any] = None
        self._cache_valid: bool = False

    def _invalidate_cache(self) -> None:
        """Reset cached pipeline and operation references.

        Returns:
            None: Indicates that cache references have been cleared.
        """
        self._cached_target_operation = None
        self._cache_valid = False

    def _resolve_target_pipeline(self) -> Pipeline:
        """Resolve the pipeline that should receive the back propagated data.

        Returns:
            Pipeline: Target pipeline for the back propagated data.

        Raises:
            ValueError: If the target pipeline cannot be found.
        """
        if self.target_pipeline_name:
            target_pipeline = self.pipeline.get_pipeline_by_name(
                pipeline_name=self.target_pipeline_name,
                camera_name=self.pipeline.camera_bus_id,
            )
            if target_pipeline is None:
                raise ValueError(
                    f"Target pipeline {self.target_pipeline_name} not found for camera {self.pipeline.camera_bus_id}"
                )
            return target_pipeline
        return self.pipeline

    def run(self, input_data: Any) -> Any:
        """Back propagate the input data to the specified operation.

        Attempts to call a back_propagate_input method on the target operation
        to notify it of the back-propagated input, then passes the original input through.

        Args:
            input_data: Input data to back propagate.

        Returns:
            The original input_data (pass-through behavior).
        """
        if not self._cache_valid:
            target_pipeline = self._resolve_target_pipeline()
            target_operation = target_pipeline.get_operation_by_class_name(
                self.action_name
            )
            if target_operation is None:
                pipeline_label = self.target_pipeline_name or "current"
                raise ValueError(
                    f"Target operation {self.action_name} not found in pipeline {pipeline_label}"
                )
            if not hasattr(target_operation, "back_propagate_input"):
                raise ValueError(
                    f"Target operation {self.action_name} does not have a back_propagate_input method"
                )
            self._cached_target_operation = target_operation
            self._cache_valid = True

        if self._cached_target_operation is None:
            raise ValueError(
                f"Cached target operation is not available for {self.action_name}"
            )

        try:
            self._cached_target_operation.back_propagate_input(input_data)
        except Exception as e:
            raise ValueError(
                f"Error calling back_propagate_input on target operation {self.action_name}: {e}"
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
            self._invalidate_cache()

        if "target_pipeline_name" in json_config:
            raw_pipeline_name = json_config["target_pipeline_name"]
            self.target_pipeline_name = raw_pipeline_name if raw_pipeline_name else None
            self._invalidate_cache()
