from __future__ import annotations

import getpass
import os
import shlex
import socket
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

from flask import request


TERMINAL_HOME = str(Path.home().resolve())
TERMINAL_USER = getpass.getuser()
TERMINAL_HOST = socket.gethostname().split(".")[0] or "eagleeye"
TERMINAL_TIMEOUT_SECONDS = 60

_TERMINAL_LOCK = threading.Lock()


class TerminalMixin:
    """Provide a stateful shell session for the Raspberry Pi terminal UI."""

    _terminal_cwd = TERMINAL_HOME

    def _terminal_prompt(self) -> str:
        """Build the shell prompt string for the current working directory.

        Returns:
            Prompt string in ``user@host:path$`` form, with ``~`` for home.
        """
        cwd = self._terminal_cwd
        if cwd == TERMINAL_HOME:
            display_cwd = "~"
        elif cwd.startswith(TERMINAL_HOME + os.sep):
            display_cwd = "~" + cwd[len(TERMINAL_HOME) :]
        else:
            display_cwd = cwd
        return f"{TERMINAL_USER}@{TERMINAL_HOST}:{display_cwd}$"

    def _terminal_payload(
        self, output: str = "", error: str = "", exit_code: int = 0
    ) -> dict[str, Any]:
        """Build a terminal API response payload.

        Args:
            output: Captured stdout text.
            error: Captured stderr or error text.
            exit_code: Process exit code.

        Returns:
            Dictionary describing terminal state and command results.
        """
        return {
            "cwd": self._terminal_cwd,
            "prompt": self._terminal_prompt(),
            "output": output,
            "error": error,
            "exit_code": exit_code,
            "success": exit_code == 0,
        }

    def get_terminal_cwd(self) -> tuple[dict[str, Any], int]:
        """Return the current terminal working directory and prompt.

        Returns:
            Tuple of response payload and HTTP status code.
        """
        return self._terminal_payload(), 200

    def execute_terminal_command(self) -> tuple[dict[str, Any], int]:
        """Execute a shell command in the persistent terminal working directory.

        Returns:
            Tuple of response payload and HTTP status code.
        """
        body = request.get_json(silent=True) or {}
        command = str(body.get("command", "")).strip()
        if not command:
            return self._terminal_payload(error="Command is required", exit_code=1), 400

        # ponytail: one lock for the whole process, the UI drives a single session.
        with _TERMINAL_LOCK:
            return self._run_terminal_command(command)

    def _run_terminal_command(self, command: str) -> tuple[dict[str, Any], int]:
        """Run one command and carry any ``cd`` it performs over to the next call.

        Args:
            command: Shell command text to execute.

        Returns:
            Tuple of response payload and HTTP status code.
        """
        cwd_file = tempfile.NamedTemporaryFile(
            mode="w", prefix="eagleeye-terminal-", suffix=".cwd", delete=False
        )
        cwd_file.close()
        try:
            completed_process = subprocess.run(
                [
                    "bash",
                    "-c",
                    f"{command}\n"
                    "__ee_status=$?\n"
                    f"pwd -P > {shlex.quote(cwd_file.name)}\n"
                    "exit $__ee_status\n",
                ],
                cwd=self._terminal_cwd,
                capture_output=True,
                text=True,
                timeout=TERMINAL_TIMEOUT_SECONDS,
            )
            next_cwd = Path(cwd_file.name).read_text(encoding="utf-8").strip()
            if next_cwd and Path(next_cwd).is_dir():
                self._terminal_cwd = next_cwd

            return self._terminal_payload(
                output=completed_process.stdout.rstrip("\n"),
                error=completed_process.stderr.rstrip("\n"),
                exit_code=completed_process.returncode,
            ), 200
        except subprocess.TimeoutExpired:
            return self._terminal_payload(
                error=f"Command timed out after {TERMINAL_TIMEOUT_SECONDS} seconds",
                exit_code=124,
            ), 408
        except OSError as error:
            return self._terminal_payload(
                error=f"Failed to execute command: {error}", exit_code=1
            ), 500
        finally:
            try:
                os.unlink(cwd_file.name)
            except OSError:
                pass
