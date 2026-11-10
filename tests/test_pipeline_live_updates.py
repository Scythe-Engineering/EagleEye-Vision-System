"""Live parameter edits from the web UI take effect on the next frame."""

from __future__ import annotations

from typing import Any

from src.config.utils.flow_manager import FlowManager
from src.config.utils.operation import Connection, Operation
from src.config.utils.pipeline import Pipeline
from src.main_operations.definitions.base.base_class import OperationInstance
from src.utils.logging.logger import Logger
from src.utils.timing import unwrap_timed


class _Source(OperationInstance):
    def run(self, *_args: Any) -> int:
        return 10


class _Scale(OperationInstance):
    def __init__(self) -> None:
        self.factor = 2

    def update_config(self, params: dict[str, Any]) -> None:
        self.factor = params["factor"]

    def run(self, value: int) -> int:
        return value * self.factor


def test_parameter_update_applies_to_the_next_frame() -> None:
    source = Operation(
        _Source(), "src", "source", is_data_source=True, output_ports=("data",)
    )
    scale = Operation(_Scale(), "scale", "scale", input_ports=("data",))
    Connection(source, "data", scale, "data", "any")
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.logger = None
    pipeline.operations = {source.uuid: source, scale.uuid: scale}
    pipeline.flow_manager = FlowManager(
        pipeline.operations, Logger(log_directory="logs/test"), pipeline_name="Live"
    )

    pipeline.flow_manager.run_flow()
    assert unwrap_timed(pipeline.get_operation_output("scale")) == 20
    snapshot = pipeline.flow_manager.get_latest_profile_snapshot()
    assert snapshot is not None and snapshot["frame_seq"] == 1

    status = pipeline.update_operations_config(
        [{"uuid": "scale", "action_name": "scale", "action_params": {"factor": 5}}]
    )
    assert status == "applied"
    pipeline.flow_manager.run_flow()
    assert unwrap_timed(pipeline.get_operation_output("scale")) == 50
    snapshot = pipeline.flow_manager.get_latest_profile_snapshot()
    assert snapshot is not None and snapshot["frame_seq"] == 2
    assert set(snapshot["operations"]) == {"src", "scale"}
