"""Real-binding regressions shared by the 2026 and alpha-7 test jobs."""
from __future__ import annotations

import socket
import time
from types import SimpleNamespace

import numpy as np
import pytest

from src.config.utils.pipeline import Pipeline
from src.secondary_operations.get_networktables_value import GetNetworktablesValue
from src.secondary_operations.publish_to_networktables import PublishToNetworktables
from src.utils import timing, wpilib_compat as wpi
from src.utils.timing import TimedValue, TimingMetadata
from src.webui.web_server_utils.first_boot_mixin import FirstBootMixin
from src.webui.web_server_utils.system_monitor_mixin import SystemMonitorMixin


@pytest.fixture
def local_table():
    instance = wpi.create_instance()
    wpi.start_local(instance)
    try:
        yield instance, wpi.get_table(instance, "EagleEye")
    finally:
        wpi.stop_local(instance)
        wpi.destroy_instance(instance)


def test_real_nt_clock_scale_and_epoch_conversion():
    before_mono = time.monotonic_ns() // 1000
    before_nt = timing.now_nt_us()
    time.sleep(0.05)
    elapsed_nt = timing.now_nt_us() - before_nt
    elapsed_mono = time.monotonic_ns() // 1000 - before_mono
    assert abs(elapsed_nt - elapsed_mono) < 20_000
    timing.nt_monotonic_offset_us()
    converted = timing.monotonic_ns_to_nt_us(time.monotonic_ns())
    assert abs(timing.now_nt_us() - converted) < 2_000


def test_clock_sanity_check_rejects_wrong_units(monkeypatch):
    monkeypatch.setattr(timing, "now_nt_us", lambda: time.monotonic_ns())
    with pytest.raises(RuntimeError, match="clock unit mismatch"):
        timing._measure_nt_monotonic_offset_us()


def test_clock_sanity_check_retries_a_scheduler_skew(monkeypatch):
    # First pair spans 200 ms on monotonic but only 50 ms on NT because the
    # process was paused between reads; the next pair advances both by 50 ms.
    mono_readings = iter([0, 200_000_000, 200_000_000, 250_000_000])
    nt_readings = iter([0, 50_000, 50_000, 100_000])
    sleeps = []
    monkeypatch.setattr(timing.time, "monotonic_ns", lambda: next(mono_readings, 250_000_000))
    monkeypatch.setattr(timing, "now_nt_us", lambda: next(nt_readings, 100_000))
    monkeypatch.setattr(timing.time, "sleep", sleeps.append)
    assert timing._measure_nt_monotonic_offset_us() == -150_000
    assert sleeps == [0.05, 0.05]


@pytest.mark.parametrize("nt_version,math_version,nt_snake,math_flat", [
    ("2027.0.0a7", "2026.2.2", True, False),
    ("2026.2.2", "2027.0.0a7", False, True),
    ("2027.0.0a7", "2027.0.0a7", True, False),
    ("2027.0.0a8", "2027.0.0a8", True, True),
    ("2028.1.1", "2028.1.1", False, False),
])
def test_mixed_or_unsupported_bindings_rejected(nt_version, math_version, nt_snake, math_flat):
    with pytest.raises(ImportError, match="Unsupported or mixed WPILib"):
        wpi._detect_api(nt_version, math_version, nt_snake=nt_snake, math_flat=math_flat)


def test_real_pipeline_disable_and_missing_default(local_table):
    _, table = local_table
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.network_table = table
    pipeline.pipeline_name = "camera"
    pipeline._last_nt_active_state = True
    pipeline.logger = None
    assert pipeline._is_enabled_from_networktables() is True
    from src.config.utils.pipeline import NT_COMMAND_PREFIX, NT_ACTIVE_COMMAND
    key = f"{NT_COMMAND_PREFIX}/camera/{NT_ACTIVE_COMMAND}"
    publisher = wpi.topic(table, "boolean", key).publish()
    publisher.set(False)
    assert pipeline._is_enabled_from_networktables() is False
    publisher.set(True)
    assert pipeline._is_enabled_from_networktables() is True


def test_real_double_read_and_missing_default(local_table):
    _, table = local_table
    operation = GetNetworktablesValue(table, "number")
    assert operation.run(None) is None
    publisher = wpi.topic(table, "double", "number").publish()
    publisher.set(42.25)
    assert operation.run(None) == 42.25
    assert wpi.read_bool(table, "missing", True) is True
    assert wpi.read_bool(table, "missing", False) is False


def test_real_monitor_and_setup_topics(local_table):
    instance, table = local_table
    publisher = wpi.topic(table, "double", "setup").publish()
    publisher.set(1.0)
    owner = SimpleNamespace(network_table_instance=instance, log=lambda _: None,
                            _read_general_conf=lambda: {"network_table_address": "127.0.0.1"})
    assert "setup" in FirstBootMixin._networktable_topic_names(owner)
    status = SystemMonitorMixin._build_network_table_status(owner)
    assert status["status"] == "ok"
    assert status["connection_count"] == 0
    assert status["connected"] == wpi.is_connected(instance)


def _wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    pytest.fail("NetworkTables loopback timed out")


def test_real_client_server_pose_meta_timestamp_join():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server, client = wpi.create_instance(), wpi.create_instance()
    try:
        wpi.start_server(server, port)
        table = wpi.get_table(server, "EagleEye")
        opts = wpi.pubsub_options(keep_duplicates=True, send_all=True, periodic=0.01, poll_storage=20)
        poses = wpi.topic(table, "struct", "pose", wpi.Pose3d).subscribe(wpi.Pose3d(), opts)
        metas = wpi.topic(table, "double_array", "meta").subscribe([], opts)
        wpi.start_client(client, "compat-test", "127.0.0.1", port)
        _wait_until(lambda: wpi.is_connected(client))
        # Let subscriptions and initial time synchronization cross the loopback.
        time.sleep(0.15)
        client_table = wpi.get_table(client, "EagleEye")
        posepub = PublishToNetworktables(client_table, "pose", "pose3d")
        metapub = PublishToNetworktables(client_table, "meta", "auto")
        captures = []
        for _ in range(3):
            capture = timing.now_nt_us()
            captures.append(capture)
            frame = TimingMetadata(capture, time.monotonic_ns())
            posepub.run(TimedValue(np.eye(4), frame))
            metapub.run(TimedValue([2., 3., .5], frame))
            wpi.flush(client)
            time.sleep(0.02)
        pose_samples, meta_samples = [], []
        def received():
            pose_samples.extend(wpi.read_queue(poses))
            meta_samples.extend(wpi.read_queue(metas))
            return len(pose_samples) >= 3 and len(meta_samples) >= 3
        _wait_until(received)
        assert len(pose_samples) == len(meta_samples) == 3
        pose_times = [wpi.sample_timestamp(sample) for sample in pose_samples]
        meta_times = [wpi.sample_timestamp(sample) for sample in meta_samples]
        assert pose_times == meta_times
        assert len(set(pose_times)) == 3
        for sample, capture in zip(pose_samples, captures):
            assert abs(wpi.sample_timestamp(sample) // wpi.NT_TIME_UNITS_PER_US - capture) < 20_000
    finally:
        wpi.stop_server(server)
        wpi.destroy_instance(client)
        wpi.destroy_instance(server)
