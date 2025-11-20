import queue
import re
import threading
import traceback
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import List

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
    """Represents a single log message or a collapsed sequence."""

    def __init__(
        self, timestamp: str, messages: List[MessageEntry], repeat_count: int = 1
    ):
        self.timestamp = timestamp
        self.messages = messages
        self.repeat_count = repeat_count

    def try_add_messages(self, messages: List[MessageEntry]) -> bool:
        """Try to add a message to the sequence.

        Returns:
            True if the message was added, False otherwise
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
    """Represents the history of messages."""

    def __init__(self, max_sequence_search_length: int = 10):
        self.max_sequence_search_length = max_sequence_search_length
        self.messages: List[MessageEntry | MessageSequence] = []

    def add_message(self, message: MessageEntry):
        self.messages.append(message)
        self.detect_and_collapse_sequences()

    def to_file_lines(self) -> List[str]:
        """Convert the message history to file lines."""
        lines = []
        for item in self.messages:
            lines.extend(item.to_file_lines())
        return lines

    def detect_and_collapse_sequences(self) -> None:
        """Detect and collapse sequences in the message history."""
        if len(self.messages) == 0:
            return

        for chunk_size in range(
            1, min(self.max_sequence_search_length, len(self.messages)) + 1
        ):
            if len(self.messages) < chunk_size:
                continue

            current_chunk = self._extract_chunk_from_end(chunk_size)
            if len(current_chunk) != chunk_size:
                continue

            if self._try_add_to_existing_sequence(current_chunk, chunk_size):
                return

            if isinstance(self.messages[-chunk_size], MessageSequence):
                continue

            if self._try_create_from_matching_chunks(current_chunk, chunk_size):
                return

    def _extract_chunk_from_end(self, chunk_size: int) -> List[MessageEntry]:
        """Extract the last chunk_size messages from history.

        Args:
            chunk_size: Number of messages to extract

        Returns:
            List of MessageEntry objects from the end of history
        """
        chunk: List[MessageEntry] = []
        for i in range(chunk_size):
            idx = len(self.messages) - 1 - i
            if isinstance(self.messages[idx], MessageSequence):
                break
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
        if not isinstance(self.messages[prev_idx], MessageSequence):
            return False

        if self.messages[prev_idx].try_add_messages(current_chunk):
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
            if isinstance(self.messages[-2], MessageSequence):
                return False

            if current_chunk[0].message == self.messages[-2].message:
                messages_list = list(current_chunk)
                timestamp = current_chunk[0].timestamp
                new_sequence = MessageSequence(timestamp, messages_list, repeat_count=2)
                self.messages = self.messages[:-2] + [new_sequence]
                return True
            else:
                return False

        previous_chunk = self._extract_chunk_from_end(chunk_size * 2)[:chunk_size]
        if len(previous_chunk) != chunk_size:
            return False

        messages_match = all(
            prev.message == curr.message
            for prev, curr in zip(previous_chunk, current_chunk)
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
    """Thread-safe logger that processes messages asynchronously with sequence deduplication in file only."""

    def __init__(
        self,
        log_directory: str = "logs",
        max_file_size_mb: int = 50,
    ) -> None:
        """Initialize the logger with queue and processing thread.

        Args:
            log_directory: Directory to store log files
            max_file_size_mb: Maximum log file size in MB before rotation
            min_sequence_length: Minimum length of sequences to detect (default 2)
        """
        self.log_directory = current_dir / log_directory
        self.log_directory.mkdir(exist_ok=True)
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

        self.message_queue: queue.Queue = queue.Queue()
        self.current_log_file = self._create_log_file()
        self.message_history: MessageHistory = MessageHistory()

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

    def _check_file_rotation(self) -> None:
        """Rotate to a new log file if current file exceeds maximum size."""
        if self.current_log_file.stat().st_size >= self.max_file_size_bytes:
            self.current_log_file = self._create_log_file()
            self._write_full_history()

    def _write_full_history(self) -> None:
        """Write the complete message history to the log file."""
        self._check_file_rotation()

        try:
            with open(self.current_log_file, "w", encoding="utf-8") as f:
                f.writelines(
                    [
                        re.sub(r"\x1b\[[0-9;]*m", "", line)
                        for line in self.message_history.to_file_lines()
                    ]
                )
        except Exception:
            traceback.print_exc()

    def _process_queue(self) -> None:
        """Process messages from the queue and write to file.

        Runs in a separate daemon thread.
        """
        while True:
            try:
                timestamp, message = self.message_queue.get(timeout=1)

                entry = MessageEntry(timestamp, message)
                self.message_history.add_message(entry)

                self._write_full_history()

                sleep(0.01)
            except queue.Empty:
                sleep(0.01)
                continue

            except Exception:
                traceback.print_exc()
                continue
