import queue
import re
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional

current_dir = Path(__file__).parent


class MessageEntry:
    """Represents a single log message."""

    def __init__(self, timestamp: str, message: str):
        self.timestamp = timestamp
        self.message = message

    def to_file_lines(self) -> List[str]:
        return [f"[{self.timestamp}] {self.message}\n"]

    def __str__(self) -> str:
        return "".join(self.to_file_lines())


class MessageSequence:
    """Represents a collapsed sequence of repeated messages."""

    def __init__(
        self, timestamp: str, messages: List[MessageEntry], repeat_count: int = 1
    ):
        self.timestamp = timestamp
        self.messages = messages
        self.repeat_count = repeat_count

    def try_add_messages(self, messages: List[MessageEntry]) -> bool:
        """Try to add messages to the sequence.

        Args:
            messages: List of messages to try adding

        Returns:
            True if the messages were added, False otherwise
        """
        if len(messages) != len(self.messages):
            return False
        if all(
            message.message == self.messages[i].message
            for i, message in enumerate(messages)
        ):
            self.repeat_count += 1
            return True
        return False

    def to_file_lines(self) -> List[str]:
        """Convert this entry to file lines.

        Returns:
            List of formatted log lines
        """
        lines = [f"[{self.timestamp}] [Sequence repeated x{self.repeat_count}]\n"]
        for message in self.messages:
            lines.append(f"  [{message.timestamp}] {message.message}\n")
        return lines

    def __str__(self) -> str:
        return "\n".join(self.to_file_lines())


class MessageHistory:
    """Manages message history with automatic sequence detection and memory limits."""

    def __init__(
        self, max_size: int = 10000, max_sequence_search_length: int = 10
    ):
        """Initialize a MessageHistory instance.

        Args:
            max_size: Maximum number of messages to keep in memory. Oldest
                messages are discarded to prevent unbounded memory growth.
            max_sequence_search_length: Max length of sequences to examine
                for pattern collapsing.
        """
        self.max_size = max_size
        self.max_sequence_search_length = max_sequence_search_length
        self.messages: List[MessageEntry | MessageSequence] = []
        self.dirty = False

    def add_message(self, message: MessageEntry) -> None:
        """Add a message to history and detect sequences.

        Args:
            message: The message to add
        """
        self.messages.append(message)

        if len(self.messages) > self.max_size:
            self.messages.pop(0)
            self.dirty = True

        old_len = len(self.messages)
        self.detect_and_collapse_sequences()
        if len(self.messages) != old_len:
            self.dirty = True

    def to_file_lines(self) -> List[str]:
        """Convert the message history to file lines.

        Returns:
            List of formatted log lines
        """
        lines = []
        for item in self.messages:
            lines.extend(item.to_file_lines())
        return lines

    def detect_and_collapse_sequences(self) -> None:
        """Detect and collapse sequences in recent message history."""
        if len(self.messages) == 0:
            return

        search_window = min(
            self.max_sequence_search_length * 3, len(self.messages)
        )

        for chunk_size in range(
            min(self.max_sequence_search_length, search_window), 0, -1
        ):
            if len(self.messages) < chunk_size:
                continue

            current_chunk = self._extract_chunk_from_end(chunk_size)

            if current_chunk is None:
                continue

            if self._try_add_to_existing_sequence(current_chunk, chunk_size):
                return

            if self._try_create_from_matching_chunks(current_chunk, chunk_size):
                return

    def _extract_chunk_from_end(self, chunk_size: int) -> Optional[List[MessageEntry]]:
        """Extract the last chunk_size messages from history.

        Args:
            chunk_size: Number of messages to extract

        Returns:
            List of MessageEntry objects from the end of history, or None if any item is a MessageSequence
        """
        chunk: List[MessageEntry] = []
        for i in range(chunk_size):
            idx = len(self.messages) - 1 - i
            if isinstance(self.messages[idx], MessageSequence):
                return None
            chunk.insert(0, self.messages[idx])
        return chunk

    def _try_add_to_existing_sequence(
        self, current_chunk: List[MessageEntry], chunk_size: int
    ) -> bool:
        """Try to add current chunk to an existing sequence.

        Args:
            current_chunk: The chunk to try adding
            chunk_size: Size of the chunk

        Returns:
            True if added successfully, False otherwise
        """
        if len(self.messages) <= chunk_size:
            return False

        prev_idx = len(self.messages) - chunk_size - 1
        prev_item = self.messages[prev_idx]

        if not isinstance(prev_item, MessageSequence):
            return False

        if prev_item.try_add_messages(current_chunk):
            self.messages = self.messages[:-chunk_size]
            return True

        return False

    def _try_create_from_matching_chunks(
        self, current_chunk: List[MessageEntry], chunk_size: int
    ) -> bool:
        """Try to create a new sequence from matching chunks.

        Args:
            current_chunk: The current chunk
            chunk_size: Size of the chunk

        Returns:
            True if sequence created successfully, False otherwise
        """
        if len(self.messages) < max(chunk_size * 2, 2):
            return False

        if chunk_size == 1:
            prev_item = self.messages[-2]
            if isinstance(prev_item, MessageSequence):
                if len(prev_item.messages) == 1 and prev_item.messages[0].message == current_chunk[0].message:
                    prev_item.repeat_count += 1
                    self.messages = self.messages[:-1]
                    return True
                return False

            if isinstance(prev_item, MessageEntry) and current_chunk[0].message == prev_item.message:
                messages_list = list(current_chunk)
                timestamp = current_chunk[0].timestamp
                new_sequence = MessageSequence(timestamp, messages_list, repeat_count=2)
                self.messages = self.messages[:-2] + [new_sequence]
                return True
            return False

        if len(self.messages) < chunk_size * 2:
            return False

        previous_chunk = self._extract_chunk_from_end(chunk_size * 2)
        if previous_chunk is None or len(previous_chunk) != chunk_size * 2:
            return False

        previous_chunk_first_half = previous_chunk[:chunk_size]

        messages_match = all(
            prev.message == curr.message
            for prev, curr in zip(previous_chunk_first_half, current_chunk)
        )

        if not messages_match:
            return False

        messages_list = list(current_chunk)
        timestamp = current_chunk[0].timestamp
        new_sequence = MessageSequence(timestamp, messages_list, repeat_count=2)
        self.messages = self.messages[: -(chunk_size * 2)] + [new_sequence]
        return True

    def __call__(self) -> List[str]:
        return self.to_file_lines()

    def __str__(self) -> str:
        return "\n".join(self.to_file_lines())


class Logger:
    """Thread-safe logger with append-only writes and sequence deduplication."""

    def __init__(
        self,
        log_directory: str = "logs",
        max_file_size_mb: int = 50,
        max_history_size: int = 10000,
    ) -> None:
        """Initialize the logger with queue and processing thread.

        Args:
            log_directory: Directory to store log files
            max_file_size_mb: Maximum log file size in MB before rotation
            max_history_size: Maximum number of messages to keep in memory
        """
        self.log_directory = current_dir / log_directory
        self.log_directory.mkdir(parents=True, exist_ok=True)
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

        self.message_queue: queue.Queue = queue.Queue()
        self.current_log_file = self._create_log_file()
        self.message_history: MessageHistory = MessageHistory(max_size=max_history_size)
        self.lock = threading.RLock()
        self.last_written_index = 0

        self.processing_thread = threading.Thread(
            target=self._process_queue, daemon=True
        )
        self.processing_thread.start()

    def log(self, message: str) -> None:
        """Add a message to the logging queue for asynchronous processing.

        Args:
            message: The message to log
        """
        print(message)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.message_queue.put((timestamp, message))

    def _create_log_file(self) -> Path:
        """Create a new log file with timestamp-based filename.

        Returns:
            Path to the newly created log file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_directory / f"log_{timestamp}.txt"
        log_file.touch()
        return log_file

    def _check_and_rotate_file(self) -> bool:
        """Check if file rotation is needed and rotate if necessary.

        Returns:
            True if file was rotated, False otherwise
        """
        try:
            if self.current_log_file.stat().st_size >= self.max_file_size_bytes:
                self.current_log_file = self._create_log_file()
                self.last_written_index = 0
                return True
        except Exception:
            traceback.print_exc()
        return False

    def _strip_ansi_codes(self, text: str) -> str:
        """Remove ANSI color codes from text.

        Args:
            text: Text potentially containing ANSI codes

        Returns:
            Text with ANSI codes removed
        """
        return re.sub(r"\x1b\[[0-9;]*m", "", text)

    def _write_to_file(self) -> None:
        """Write pending messages to file (append or full rewrite)."""
        with self.lock:
            try:
                rotated = self._check_and_rotate_file()

                if self.message_history.dirty or rotated:
                    with open(self.current_log_file, "w", encoding="utf-8") as f:
                        for line in self.message_history.to_file_lines():
                            f.write(self._strip_ansi_codes(line))
                    self.message_history.dirty = False
                    self.last_written_index = len(self.message_history.messages)
                else:
                    new_messages = self.message_history.messages[self.last_written_index:]
                    if new_messages:
                        with open(self.current_log_file, "a", encoding="utf-8") as f:
                            for item in new_messages:
                                for line in item.to_file_lines():
                                    f.write(self._strip_ansi_codes(line))
                        self.last_written_index = len(self.message_history.messages)

            except Exception:
                traceback.print_exc()

    def _process_queue(self) -> None:
        """Process messages from the queue and write to file.

        Runs in a separate daemon thread.
        """
        while True:
            try:
                timestamp, message = self.message_queue.get(timeout=1)

                with self.lock:
                    entry = MessageEntry(timestamp, message)
                    self.message_history.add_message(entry)

                self._write_to_file()

            except queue.Empty:
                continue

            except Exception:
                traceback.print_exc()
                continue
