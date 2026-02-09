"""Tests for flow profiling and SSE profiling publication."""

from __future__ import annotations

import time
from typing import Any

from src.config.utils.flow_manager import FlowManager
from src.config.utils.operation import Connection, Operation
from src.main_operations.definitions.base.base_class import OperationInstance
from src.webui.web_server import EagleEyeInterface
from src.utils.logging.logger import Logger


class _SourceOperation(OperationInstance):
    def __init__(self, value: Any) -> None:
        self.value = value

    def run(self, *_args: Any, **_kwargs: Any) -> Any:
        return self.value


class _PassOperation(OperationInstance):
    def run(self, input_data: Any) -> Any:
        return input_data


class _MergeOperation(OperationInstance):
    def run(self, input_data: dict[str, Any]) -> Any:
        left = input_data.get("left")
        right = input_data.get("right")
        return (left or 0) + (right or 0)


class _NoneAbortOperation(OperationInstance):
    def run(self, _input_data: Any) -> Any:
        raise TypeError("None input is unsupported")


def _build_direct_flow_manager() -> FlowManager:
    source = Operation(_SourceOperation(3), "op-source", "source", is_data_source=True)
    transform = Operation(_PassOperation(), "op-transform", "transform")
    Connection(source, "data", transform, "data", "any")
    operations = {source.uuid: source, transform.uuid: transform}
    return FlowManager(operations, Logger(log_directory="logs/test"), pipeline_name="Direct")


def _build_threaded_flow_manager() -> FlowManager:
    src_a = Operation(_SourceOperation(2), "op-a-src", "src_a", is_data_source=True)
    src_b = Operation(_SourceOperation(5), "op-b-src", "src_b", is_data_source=True)
    pass_a = Operation(_PassOperation(), "op-a-pass", "pass_a")
    pass_b = Operation(_PassOperation(), "op-b-pass", "pass_b")
    merge = Operation(_MergeOperation(), "op-merge", "merge")

    Connection(src_a, "data", pass_a, "data", "any")
    Connection(src_b, "data", pass_b, "data", "any")
    Connection(pass_a, "data", merge, "left", "any")
    Connection(pass_b, "data", merge, "right", "any")

    operations = {
        src_a.uuid: src_a,
        src_b.uuid: src_b,
        pass_a.uuid: pass_a,
        pass_b.uuid: pass_b,
        merge.uuid: merge,
    }
    return FlowManager(
        operations,
        Logger(log_directory="logs/test"),
        pipeline_name="Threaded",
    )


def _assert_profile_contract(snapshot: dict[str, Any]) -> None:
    required_keys = {
        "pipeline_name",
        "frame_seq",
        "frame_time_ms",
        "timestamp_ms",
        "operations",
        "timesteps",
    }
    assert required_keys.issubset(snapshot.keys())
    assert snapshot["frame_seq"] >= 1
    assert snapshot["frame_time_ms"] >= 0.0
    assert snapshot["timestamp_ms"] > 0
    assert isinstance(snapshot["operations"], dict)
    assert isinstance(snapshot["timesteps"], list)

    for operation_row in snapshot["operations"].values():
        assert operation_row["execution_time_ms"] >= 0.0

    operations = snapshot["operations"]
    max_timestep_total_ms = 0.0
    for row in snapshot["timesteps"]:
        row_total = float(row["total_time_ms"])
        max_timestep_total_ms = max(max_timestep_total_ms, row_total)
        assert row_total >= 0.0
        assert float(row["max_operation_time_ms"]) >= 0.0

        timestep_ops = [
            op
            for op in operations.values()
            if int(op.get("timestep", -1)) == int(row["timestep"])
        ]
        if not timestep_ops:
            continue

        max_operation_row = max(
            timestep_ops,
            key=lambda op: float(op.get("execution_time_ms", 0.0)),
        )
        assert abs(
            float(row["max_operation_time_ms"])
            - float(max_operation_row["execution_time_ms"])
        ) < 1e-6

    assert float(snapshot["frame_time_ms"]) + 1e-6 >= max_timestep_total_ms


def test_direct_flow_records_profile_snapshot() -> None:
    flow_manager = _build_direct_flow_manager()
    flow_manager.run_flow()

    snapshot = flow_manager.get_latest_profile_snapshot()
    assert snapshot is not None
    _assert_profile_contract(snapshot)


def test_threaded_flow_records_profile_snapshot() -> None:
    flow_manager = _build_threaded_flow_manager()
    assert flow_manager.num_threads > 1

    flow_manager.run_flow()
    snapshot = flow_manager.get_latest_profile_snapshot()
    assert snapshot is not None
    _assert_profile_contract(snapshot)


def test_none_abort_does_not_publish_profile_snapshot() -> None:
    source = Operation(
        _NoneAbortOperation(),
        "op-abort",
        "abort",
        is_data_source=True,
    )
    passthrough = Operation(_PassOperation(), "op-pass", "pass")
    Connection(source, "data", passthrough, "data", "any")
    flow_manager = FlowManager(
        {source.uuid: source, passthrough.uuid: passthrough},
        Logger(log_directory="logs/test"),
        pipeline_name="Abort",
    )

    flow_manager.run_flow()
    assert flow_manager.get_latest_profile_snapshot() is None


def test_sse_profiling_update_is_published_on_new_sequence() -> None:
    published_events: list[tuple[str, dict[str, Any]]] = []
    snapshot = {
        "pipeline_name": "FrontCam",
        "frame_seq": 3,
        "frame_time_ms": 12.5,
        "timestamp_ms": int(time.time() * 1000),
        "operations": {},
        "timesteps": [],
    }

    class _PipelineStub:
        def get_latest_profile_snapshot(self) -> dict[str, Any]:
            return snapshot

    interface = EagleEyeInterface.__new__(EagleEyeInterface)
    interface._last_profiling_publish_ts = 0.0
    interface._profiling_publish_interval = 0.0
    interface._pipeline_profile_last_seq_sent = {}
    interface.pipeline_objects_callback = lambda: {"FrontCam": _PipelineStub()}
    interface._publish_event = (
        lambda event_name, data: published_events.append((event_name, data))
    )
    interface.log = lambda *_args, **_kwargs: None

    EagleEyeInterface._publish_profiling_updates(interface)
    assert len(published_events) == 1
    event_name, payload = published_events[0]
    assert event_name == "profiling_update"
    for required_field in (
        "pipeline_name",
        "frame_seq",
        "frame_time_ms",
        "timestamp_ms",
        "operations",
        "timesteps",
    ):
        assert required_field in payload

    EagleEyeInterface._publish_profiling_updates(interface)
    assert len(published_events) == 1
