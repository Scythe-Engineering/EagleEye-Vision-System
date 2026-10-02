"""Native NT queues retain measurement times in the receiver-local clock."""

import itertools
import socket
import time
from pathlib import Path
from types import SimpleNamespace

import ntcore
import pytest

from src.secondary_operations.get_networktables_value import GetNetworktablesValue
from src.secondary_operations.publish_to_networktables import PublishToNetworktables
from src.utils.timing import TimedValue, TimingMetadata


@pytest.fixture
def local_nt():
    instance = ntcore.NetworkTableInstance.create()
    instance.startLocal()
    yield instance.getTable("timestamp_tests")
    instance.stopLocal()
    ntcore.NetworkTableInstance.destroy(instance)


def test_queued_samples_key_update_and_bounds(local_nt):
    reader = GetNetworktablesValue(local_nt, "a", timestamped=True, history_size=3)
    publisher = local_nt.getDoubleTopic("a").publish(
        ntcore.PubSubOptions(keepDuplicates=True, sendAll=True)
    )
    try:
        for timestamp in (10, 20, 30, 40):
            publisher.set(0.5, timestamp)
        output = reader.run(None)
        assert not isinstance(output, TimedValue)
        assert output == [{"timestamp_us": t, "value": 0.5} for t in (20, 30, 40)]
        assert reader.run(None) == output
        reader.update_config({"network_table_key": "b"})
        assert reader.run(None) == []
        publisher_b = local_nt.getStringTopic("b").publish()
        try:
            publisher_b.set("not JSON", 50)
            assert reader.run(None) == [{"timestamp_us": 50, "value": "not JSON"}]
            publisher.set(1, 70)
            assert len(reader.run(None)) == 1
        finally:
            publisher_b.close()
    finally:
        reader.close()
        publisher.close()


@pytest.mark.parametrize(
    "topic,value",
    [
        ("Double", 1.25),
        ("Boolean", True),
        ("String", "plain text"),
        ("DoubleArray", [1.0, 2.0]),
        ("BooleanArray", [True, False]),
        ("StringArray", ["a", "b"]),
        ("Raw", b"\x00\xff"),
    ],
)
def test_native_types_and_future_measurement(local_nt, topic, value):
    reader = GetNetworktablesValue(local_nt, topic, timestamped=True)
    native_topic = getattr(local_nt, f"get{topic}Topic")(topic)
    publisher = (
        native_topic.publish("raw") if topic == "Raw" else native_topic.publish()
    )
    try:
        timestamp = ntcore._now() + 1_000_000
        publisher.set(value, timestamp)
        assert reader.run(None) == [{"timestamp_us": timestamp, "value": value}]
        assert reader.last_error is None
    finally:
        reader.close()
        publisher.close()


def test_existing_publisher_reverse_contract(local_nt):
    reader = GetNetworktablesValue(local_nt, "yaw", timestamped=True)
    publisher = PublishToNetworktables(local_nt, "yaw", "double")
    try:
        timestamp = ntcore._now() - 500_000
        publisher.run(TimedValue(0.75, TimingMetadata(timestamp, 123)))
        assert reader.run(None) == [{"timestamp_us": timestamp, "value": 0.75}]
    finally:
        reader.close()
        publisher._publisher.close()


def test_legacy_mode_hotupdates(local_nt):
    reader = GetNetworktablesValue(local_nt, "first")
    assert reader.run(None) is None
    local_nt.getEntry("first").setDouble(3)
    local_nt.getEntry("second").setDouble(5)
    assert reader.run(None) == 3
    reader.update_config({"network_table_key": "second"})
    assert reader.run(None) == 5
    reader.update_config({"timestamped": True})
    try:
        assert reader.run(None)[0]["value"] == 5
    finally:
        reader.close()


def test_invalid_bounds(local_nt):
    for size in (0, 4097, True, 1.5):
        with pytest.raises(ValueError, match="history_size"):
            GetNetworktablesValue(local_nt, "x", timestamped=True, history_size=size)


def test_client_unsynced_disconnect_and_reconnect(local_nt):
    # Use real native queue/publisher; fake only the production clock interface.
    reader = GetNetworktablesValue(local_nt, "clock", timestamped=True)
    publisher = local_nt.getDoubleTopic("clock").publish()
    clock = SimpleNamespace(connected=True, offset=None)
    instance = SimpleNamespace(
        getNetworkMode=lambda: ntcore.NetworkTableInstance.NetworkMode.kNetModeClient4,
        isConnected=lambda: clock.connected,
        getServerTimeOffset=lambda: clock.offset,
    )
    reader.network_table = SimpleNamespace(getInstance=lambda: instance)
    try:
        publisher.set(1, 100)
        assert reader.run(None) == []
        assert "clock" in reader.last_error and "synchronization" in reader.last_error
        publisher.set(2, 200)  # Queued before sync: never relabel as valid.
        clock.offset = 345_000
        assert reader.run(None) == []
        publisher.set(3, 300)
        assert reader.run(None) == [{"timestamp_us": 300, "value": 3.0}]
        clock.connected = False
        publisher.set(4, 400)
        assert reader.run(None) == []
        clock.connected = True
        publisher.set(5, 500)
        assert reader.run(None) == []
        publisher.set(6, 600)
        assert reader.run(None) == [{"timestamp_us": 600, "value": 6.0}]
        assert reader.last_error is None
    finally:
        reader.close()
        publisher.close()


@pytest.mark.parametrize("timestamp", [0, 1, -1, 2**63])
def test_invalid_native_timestamp_clears_history(local_nt, timestamp):
    reader = GetNetworktablesValue(local_nt, "bad", timestamped=True)
    # Native set(time=0) means now; use a production-interface queue double
    # only to exercise timestamps that the public publisher cannot emit.
    reader.close()
    queue = [
        SimpleNamespace(time=lambda: timestamp, value=lambda: 1, isValid=lambda: True)
    ]
    reader._subscriber = SimpleNamespace(readQueue=lambda: queue)
    reader._history[20] = {"timestamp_us": 20, "value": 2}
    try:
        assert reader.run(None) == []
        assert "bad" in reader.last_error and "timestamp" in reader.last_error
    finally:
        reader.close()


def wait_until(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    pytest.fail("NT connection/clock/queued sample did not become ready")


def test_real_server_client_conversion(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = ntcore.NetworkTableInstance.create()
    client = ntcore.NetworkTableInstance.create()
    reader = publisher = None
    try:
        server.startServer(str(tmp_path / "nt.json"), "127.0.0.1", 0, port)
        client.setServer("127.0.0.1", port)
        client.startClient4("timestamp-test")
        reader = GetNetworktablesValue(client.getTable("test"), "yaw", timestamped=True)
        publisher = server.getTable("test").getDoubleTopic("yaw").publish()
        publisher.set(0.1, ntcore._now() - 900_000)
        wait_until(
            lambda: client.isConnected() and client.getServerTimeOffset() is not None
        )
        assert reader.run(None) == []  # First ready poll drains pre-sync data.
        # Ensure initial cached value has arrived before the fresh publication.
        wait_until(lambda: client.getTopic("/test/yaw").exists())
        time.sleep(0.1)
        reader.run(None)
        timestamp = ntcore._now() - 500_000
        publisher.set(0.75, timestamp)
        server.flush()
        output = []

        def received():
            nonlocal output
            output = reader.run(None)
            return any(sample["value"] == 0.75 for sample in output)

        wait_until(received)
        sample = next(sample for sample in output if sample["value"] == 0.75)
        # Same process NT clock: native conversion is near zero, not publish-now.
        assert abs(sample["timestamp_us"] - timestamp) < 10_000
        assert reader.last_error is None
    finally:
        if reader:
            reader.close()
        if publisher:
            publisher.close()
        client.stopClient()
        server.stopServer()
        ntcore.NetworkTableInstance.destroy(client)
        ntcore.NetworkTableInstance.destroy(server)


def test_native_client_delivery_cadence_without_flush(tmp_path: Path) -> None:
    """The production reader requests timely batches, not default 100ms bursts.

    A native sender client avoids the server's independent 100ms local-input
    pump. Neither client nor server is flushed; both clocks sync before sampling.
    """
    import statistics

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = ntcore.NetworkTableInstance.create()
    sender = ntcore.NetworkTableInstance.create()
    receiver = ntcore.NetworkTableInstance.create()
    reader = publisher = None
    try:
        server.startServer(str(tmp_path / "cadence.json"), "127.0.0.1", 0, port)
        for client, name in ((sender, "cadence-sender"), (receiver, "cadence-reader")):
            client.setServer("127.0.0.1", port)
            client.startClient4(name)
        reader = GetNetworktablesValue(
            receiver.getTable("cadence"), "yaw", timestamped=True
        )
        publisher = (
            sender.getTable("cadence")
            .getDoubleTopic("yaw")
            .publish(
                ntcore.PubSubOptions(periodic=0.01, keepDuplicates=True, sendAll=True)
            )
        )
        wait_until(
            lambda: all(
                client.isConnected() and client.getServerTimeOffset() is not None
                for client in (sender, receiver)
            )
        )
        assert reader.run(None) == []  # Drain once at the clock-ready transition.
        start = next_publish = time.monotonic()
        published: dict[int, int] = {}
        retained: set[int] = set()
        batches: list[float] = []
        ages_ms: list[float] = []
        latest = None
        while time.monotonic() - start < 2.2:
            now = time.monotonic()
            if now >= next_publish:
                index = len(published)
                timestamp = ntcore._now()
                published[index] = timestamp
                publisher.set(float(index), timestamp)
                next_publish = now + 0.01
            samples = reader.run(None)
            assert reader.last_error is None
            for sample in samples:
                index = int(sample["value"])
                # Explicit measurement time survives native clock conversion.
                assert abs(sample["timestamp_us"] - published[index]) < 10_000
                retained.add(index)
            if samples:
                ages_ms.append((ntcore._now() - samples[-1]["timestamp_us"]) / 1000)
                if samples[-1]["timestamp_us"] != latest:
                    batches.append(now)
                    latest = samples[-1]["timestamp_us"]
            time.sleep(0.002)
        # Aggregate windows tolerate scheduler jitter, but fail 100ms batching.
        assert len(retained) > 50, (
            "native relay did not retain enough fresh measurements"
        )
        assert len(batches) > 30, "reader still receives default 100ms network bursts"
        gaps = [later - earlier for earlier, later in itertools.pairwise(batches)]
        assert statistics.median(gaps) < 0.075
        print(
            f"native cadence: {len(batches)} batches, {len(retained)}/{len(published)} "
            f"retained, {time.monotonic() - start:.3f}s elapsed, "
            f"median gap {statistics.median(gaps) * 1000:.2f}ms, "
            f"median/max source age {statistics.median(ages_ms):.2f}/{max(ages_ms):.2f}ms"
        )
    finally:
        if reader:
            reader.close()
        if publisher:
            publisher.close()
        for client in (sender, receiver):
            client.stopClient()
            ntcore.NetworkTableInstance.destroy(client)
        server.stopServer()
        ntcore.NetworkTableInstance.destroy(server)
