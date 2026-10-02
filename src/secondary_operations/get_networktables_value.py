import logging
import math
import threading
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
        """Select legacy scalar or bounded native queued measurement mode.

        Args:
            network_table: Native table providing the configured topic.
            network_table_key: Entry or topic key to read.
            timestamped: Whether to retain queued measurement times and values.
            history_size: Maximum queued and retained sample count (1 to 4096).

        Raises:
            ValueError: The history size is not a bounded integer.
        """
        self._state_lock = threading.RLock()
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
        """Reject noninteger or unbounded queue/history sizes.

        Args:
            value: Requested maximum sample count.

        Returns:
            The validated integer sample count.

        Raises:
            ValueError: The count is not an integer between 1 and 4096.
        """
        if type(value) is not int or not 1 <= value <= 4096:
            raise ValueError("history_size must be an integer between 1 and 4096")
        return value

    def close(self) -> None:
        """Release the subscriber after active consumption; safe to call repeatedly."""
        with self._state_lock:
            # GenericSubscriber releases its native handle on destruction (no close API).
            self._subscriber = None

    def _configure_subscriber(self) -> None:
        """Close the old subscription and discard history when source changes."""
        with self._state_lock:
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
        Own the subscription and history until consumption finishes; live updates
        and release wait so an old-source batch cannot enter new-source history.
        No age gate: later gyro samples and explicitly backdated replay are valid.

        Args:
            input_data: Unused operation input; the configured NT source is read.

        Returns:
            Latest native double or None when absent in legacy mode. Timestamped
            mode returns bounded sample history, or an empty list when the clock,
            subscriber, or measurement timestamps are invalid.
        """
        with self._state_lock:
            if not self.timestamped:
                latest_value = self.network_table.getEntry(
                    self.network_table_key
                ).getDouble(float("nan"))
                return None if math.isnan(latest_value) else latest_value
            self.last_error = None
            subscriber = self._subscriber
            if subscriber is None:
                self.last_error = f"NT key {self.network_table_key!r}: timestamped subscriber is closed"
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
            queued_values = subscriber.readQueue()
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
        """Hot-update key, mode, or finite storage bounds, resetting history.

        Args:
            json_config: Key, timestamped mode, or history size overrides.

        Raises:
            ValueError: The history size is not a bounded integer.
        """
        with self._state_lock:
            key = json_config.get("network_table_key", self.network_table_key)
            timestamped = json_config.get("timestamped", self.timestamped)
            size = self._validate_size(
                json_config.get("history_size", self.history_size)
            )
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
