"""Tests for flow profiling and SSE profiling publication."""

from __future__ import annotations

import time
from typing import Any

import pytest

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
    source = Operation(
        _SourceOperation(3),
        "op-source",
        "source",
        is_data_source=True,
        output_ports=("data",),
    )
    transform = Operation(
        _PassOperation(), "op-transform", "transform", input_ports=("data",)
    )
    Connection(source, "data", transform, "data", "any")
    operations = {source.uuid: source, transform.uuid: transform}
    return FlowManager(operations, Logger(log_directory="logs/test"), pipeline_name="Direct")


def _build_threaded_flow_manager() -> FlowManager:
    src_a = Operation(
        _SourceOperation(2),
        "op-a-src",
        "src_a",
        is_data_source=True,
        output_ports=("data",),
    )
    src_b = Operation(
        _SourceOperation(5),
        "op-b-src",
        "src_b",
        is_data_source=True,
        output_ports=("data",),
    )
    pass_a = Operation(
        _PassOperation(),
        "op-a-pass",
        "pass_a",
        input_ports=("data",),
        output_ports=("data",),
    )
    pass_b = Operation(
        _PassOperation(),
        "op-b-pass",
        "pass_b",
        input_ports=("data",),
        output_ports=("data",),
    )
    merge = Operation(
        _MergeOperation(), "op-merge", "merge", input_ports=("left", "right")
    )

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
    assert flow_manager.run_flow()

    snapshot = flow_manager.get_latest_profile_snapshot()
    assert snapshot is not None
    _assert_profile_contract(snapshot)


def test_profile_cycle_time_can_include_pipeline_input_wait() -> None:
    flow_manager = _build_direct_flow_manager()
    flow_manager.run_flow()

    flow_manager.set_latest_profile_cycle_time(42.5)

    snapshot = flow_manager.get_latest_profile_snapshot()
    assert snapshot is not None
    assert snapshot["cycle_time_ms"] == pytest.approx(42.5)
    assert snapshot["cycle_time_ms"] >= snapshot["frame_time_ms"]


def test_threaded_flow_records_profile_snapshot() -> None:
    flow_manager = _build_threaded_flow_manager()
    assert flow_manager.num_threads > 1

    assert flow_manager.run_flow()
    snapshot = flow_manager.get_latest_profile_snapshot()
    assert snapshot is not None
    _assert_profile_contract(snapshot)


def test_none_abort_does_not_publish_profile_snapshot() -> None:
    source = Operation(
        _NoneAbortOperation(),
        "op-abort",
        "abort",
        is_data_source=True,
        output_ports=("data",),
    )
    passthrough = Operation(_PassOperation(), "op-pass", "pass", input_ports=("data",))
    Connection(source, "data", passthrough, "data", "any")
    flow_manager = FlowManager(
        {source.uuid: source, passthrough.uuid: passthrough},
        Logger(log_directory="logs/test"),
        pipeline_name="Abort",
    )

    assert not flow_manager.run_flow()
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


def test_snapshot_built_on_read_matches_recorded_frame() -> None:
    """The lazily built payload carries exactly the latest frame's timings."""
    flow_manager = _build_threaded_flow_manager()
    assert flow_manager.run_flow()
    assert flow_manager.run_flow()
    groups = flow_manager.execution_time_groups
    merge = flow_manager.operations["op-merge"]
    op_times = {uuid: 0.5 for uuid in flow_manager.operations}
    op_times["op-merge"] = 2.0
    flow_manager._record_profile_snapshot(
        frame_time_ms=9.0,
        operation_time_by_uuid_ms=op_times,
        timestep_total_ms={step: 1.5 for step in range(len(groups))},
    )
    flow_manager.set_latest_profile_cycle_time(11.0)
    flow_manager.set_latest_profile_capture_latency(30.0)

    snapshot = flow_manager.get_latest_profile_snapshot()
    assert snapshot is not None
    thread_info = flow_manager.get_thread_and_timestep_info()
    assert snapshot["operations"] == {
        uuid: {
            "name": operation.name,
            "timestep": thread_info[uuid]["timestep"],
            "thread": thread_info[uuid]["thread"],
            "execution_time_ms": op_times[uuid],
        }
        for uuid, operation in flow_manager.operations.items()
    }
    merge_row = snapshot["timesteps"][merge.execution_timestep]
    assert merge_row == {
        "timestep": merge.execution_timestep,
        "total_time_ms": 1.5,
        "max_operation_uuid": "op-merge",
        "max_operation_name": "merge",
        "max_operation_time_ms": 2.0,
        "operation_count": 1,
    }
    assert len(snapshot["timesteps"]) == len(groups)
    assert snapshot["frame_seq"] == 3
    assert snapshot["frame_time_ms"] == 9.0
    assert snapshot["cycle_time_ms"] == 11.0
    assert snapshot["capture_latency_ms"] == 30.0
    assert snapshot["pipeline_name"] == "Threaded"
    assert "skipped" not in snapshot

    # Readers get independent copies, and the next frame replaces everything.
    snapshot["operations"].clear()
    assert flow_manager.get_latest_profile_snapshot()["operations"]
    assert flow_manager.run_flow()
    latest = flow_manager.get_latest_profile_snapshot()
    assert latest["frame_seq"] == 4
    assert "cycle_time_ms" not in latest
    _assert_profile_contract(latest)
