import logging
import math
from typing import Any

import ntcore

from src.main_operations.definitions.base.base_class import OperationInstance
from src.utils.timestamped_samples import validate_sample


class GetNetworktablesValue(OperationInstance):
    """Latest double, or native NT values with receiver-local measurement times.

    Publishers use .set(value, measurement_nt_us); ntcore performs clock
    conversion. Output is bounded plain {timestamp_us, value} sample history.
    """

    def __init__(
        self,
        network_table: ntcore.NetworkTable,
        network_table_key: str,
        timestamped: bool = False,
        history_size: int = 256,
    ) -> None:
        """Select legacy scalar or bounded native queued measurement mode."""
        self.network_table = network_table
        self.network_table_key = network_table_key
        self.timestamped = timestamped
        self.history_size = self._validate_size(history_size)
        self._subscriber: ntcore.GenericSubscriber | None = None
        self._history: dict[int, dict[str, Any]] = {}
        self._clock_ready = False
        self.last_error: str | None = None
        self._configure_subscriber()

    @staticmethod
    def _validate_size(value: object) -> int:
        """Reject noninteger or unbounded queue/history sizes."""
        if type(value) is not int or not 1 <= value <= 4096:
            raise ValueError("history_size must be an integer between 1 and 4096")
        return value

    def close(self) -> None:
        """Release the subscriber; safe to call repeatedly."""
        # GenericSubscriber releases its native handle on destruction (no close API).
        self._subscriber = None

    def _configure_subscriber(self) -> None:
        """Close the old subscription and discard history when source changes."""
        self.close()
        self._history.clear()
        self.last_error = None
        self._clock_ready = False
        if self.timestamped:
            self._subscriber = self.network_table.getTopic(
                self.network_table_key
            ).genericSubscribe(
                ntcore.PubSubOptions(
                    periodic=0.01,
                    pollStorage=self.history_size,
                    keepDuplicates=True,
                    sendAll=True,
                ),
            )

    def run(self, input_data: Any) -> float | list[dict[str, Any]] | None:
        """Drain native samples; discard history/queues while client clock is unsafe.

        Server and isolated startLocal modes already share the local NT clock.
        Client sync transitions drain old values before accepting new samples.
        No age gate: later gyro samples and explicitly backdated replay are valid.
        """
        if not self.timestamped:
            val = self.network_table.getEntry(self.network_table_key).getDouble(
                float("nan")
            )
            return None if math.isnan(val) else val
        self.last_error = None
        if self._subscriber is None:
            self.last_error = (
                f"NT key {self.network_table_key!r}: timestamped subscriber is closed"
            )
            return []
        instance = self.network_table.getInstance()
        mode = instance.getNetworkMode()
        modes = ntcore.NetworkTableInstance.NetworkMode
        client = mode & (modes.kNetModeClient3 | modes.kNetModeClient4)
        ready = (
            instance.isConnected() and instance.getServerTimeOffset() is not None
            if client
            else bool(mode & (modes.kNetModeServer | modes.kNetModeLocal))
        )
        queued_values = self._subscriber.readQueue()
        if not ready or (client and not self._clock_ready):
            self._history.clear()
            self._clock_ready = bool(ready)
            self.last_error = (
                f"NT key {self.network_table_key!r}: waiting for connected NT clock "
                "synchronization; discarded queued measurements, publish fresh samples"
            )
            return []
        self._clock_ready = True
        for queued in queued_values:
            if not queued.isValid():
                continue  # Native unassigned-topic notification is not a measurement.
            try:
                sample = {"timestamp_us": queued.time(), "value": queued.value()}
                validate_sample(sample)
                self._history[sample["timestamp_us"]] = sample
            except (ValueError, TypeError, OverflowError) as exc:
                self.last_error = f"NT key {self.network_table_key!r}: invalid native measurement timestamp: {exc}"
        self._history = dict(sorted(self._history.items())[-self.history_size :])
        if self.last_error:
            logging.getLogger(__name__).warning(self.last_error)
            self._history.clear()
            return []
        return list(self._history.values())

    def update_config(self, json_config: dict[str, Any]) -> None:
        """Hot-update key, mode, or finite storage bounds, resetting history."""
        key = json_config.get("network_table_key", self.network_table_key)
        timestamped = json_config.get("timestamped", self.timestamped)
        size = self._validate_size(json_config.get("history_size", self.history_size))
        if (key, timestamped, size) != (
            self.network_table_key,
            self.timestamped,
            self.history_size,
        ):
            self.network_table_key, self.timestamped, self.history_size = (
                key,
                timestamped,
                size,
            )
            self._configure_subscriber()
