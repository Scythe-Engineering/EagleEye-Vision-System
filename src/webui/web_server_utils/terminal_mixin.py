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


def _request():
    """Return the current Flask request, allowing monkeypatching via web_server.request."""
    import src.webui.web_server as _ws

    return _ws.request


class TerminalMixin:
    """Provide a stateful shell session for the Raspberry Pi terminal UI."""

    _TERMINAL_COMMAND_TIMEOUT_SECONDS = 60.0

    def _ensure_terminal_state(self) -> None:
        """Initialize terminal session state on first use."""
        if getattr(self, "_terminal_state_initialized", False):
            return

        self._terminal_lock = threading.Lock()
        self._terminal_cwd = str(Path.home().resolve())
        self._terminal_home = str(Path.home().resolve())
        self._terminal_user = getpass.getuser()
        self._terminal_host = socket.gethostname().split(".")[0] or "eagleeye"
        self._terminal_state_initialized = True

    def _get_terminal_cwd(self) -> str:
        """Return the current terminal working directory.

        Returns:
            Absolute path of the active terminal working directory.
        """
        self._ensure_terminal_state()
        return self._terminal_cwd

    def _format_prompt_cwd(self, cwd: str) -> str:
        """Format a working directory for prompt display.

        Args:
            cwd: Absolute working directory path.

        Returns:
            Prompt-friendly path using ``~`` for the home directory.
        """
        self._ensure_terminal_state()
        home_directory = self._terminal_home
        if cwd == home_directory:
            return "~"
        if cwd.startswith(home_directory + os.sep):
            return "~" + cwd[len(home_directory) :]
        return cwd

    def _build_terminal_prompt(self, cwd: str | None = None) -> str:
        """Build the shell prompt string for the current session.

        Args:
            cwd: Optional working directory override.

        Returns:
            Prompt string in ``user@host:path$`` form.
        """
        self._ensure_terminal_state()
        active_cwd = cwd if cwd is not None else self._terminal_cwd
        display_cwd = self._format_prompt_cwd(active_cwd)
        return f"{self._terminal_user}@{self._terminal_host}:{display_cwd}$"

    def _build_terminal_payload(
        self,
        *,
        output: str = "",
        error: str = "",
        exit_code: int | None = None,
    ) -> dict[str, Any]:
        """Build a terminal API response payload.

        Args:
            output: Captured stdout text.
            error: Captured stderr or error text.
            exit_code: Optional process exit code.

        Returns:
            Dictionary describing terminal state and command results.
        """
        self._ensure_terminal_state()
        payload: dict[str, Any] = {
            "cwd": self._terminal_cwd,
            "prompt": self._build_terminal_prompt(),
            "user": self._terminal_user,
            "host": self._terminal_host,
            "output": output,
            "error": error,
        }
        if exit_code is not None:
            payload["exit_code"] = exit_code
        return payload

    def get_terminal_cwd(self) -> tuple[dict[str, Any], int]:
        """Return the current terminal working directory and prompt.

        Returns:
            Tuple of response payload and HTTP status code.
        """
        self._ensure_terminal_state()
        return self._build_terminal_payload(), 200

    def reset_terminal_cwd(self) -> tuple[dict[str, Any], int]:
        """Reset the terminal working directory to the user home directory.

        Returns:
            Tuple of response payload and HTTP status code.
        """
        self._ensure_terminal_state()
        with self._terminal_lock:
            self._terminal_cwd = self._terminal_home
        return self._build_terminal_payload(), 200

    def execute_terminal_command(self) -> tuple[dict[str, Any], int]:
        """Execute a shell command in the persistent terminal working directory.

        Returns:
            Tuple of response payload and HTTP status code.
        """
        self._ensure_terminal_state()
        body = _request().get_json(silent=True) or {}
        command = str(body.get("command", "")).strip()
        if not command:
            return {
                **self._build_terminal_payload(error="Command is required"),
                "success": False,
            }, 400

        with self._terminal_lock:
            return self._execute_command_locked(command)

    def _execute_command_locked(self, command: str) -> tuple[dict[str, Any], int]:
        """Execute a shell command while holding the terminal lock.

        Args:
            command: Shell command text to execute.

        Returns:
            Tuple of response payload and HTTP status code.
        """
        current_cwd = self._terminal_cwd
        meta_path: str | None = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                delete=False,
                prefix="eagleeye-terminal-",
                suffix=".meta",
            ) as meta_file:
                meta_path = meta_file.name

            shell_script = (
                f"cd {shlex.quote(current_cwd)} || exit 1\n"
                f"{command}\n"
                "__ee_status=$?\n"
                f"pwd -P > {shlex.quote(meta_path)}\n"
                f'printf "%s\\n" "$__ee_status" >> {shlex.quote(meta_path)}\n'
                "exit $__ee_status\n"
            )

            completed_process = subprocess.run(
                ["bash", "-c", shell_script],
                capture_output=True,
                text=True,
                timeout=self._TERMINAL_COMMAND_TIMEOUT_SECONDS,
                env=os.environ.copy(),
            )

            next_cwd, exit_code = self._read_command_metadata(
                meta_path,
                fallback_cwd=current_cwd,
                fallback_exit_code=completed_process.returncode,
            )
            if Path(next_cwd).is_dir():
                self._terminal_cwd = next_cwd

            stdout_text = completed_process.stdout or ""
            stderr_text = completed_process.stderr or ""
            payload = self._build_terminal_payload(
                output=stdout_text.rstrip("\n"),
                error=stderr_text.rstrip("\n"),
                exit_code=exit_code,
            )
            payload["success"] = exit_code == 0
            return payload, 200
        except subprocess.TimeoutExpired:
            payload = self._build_terminal_payload(
                error=(
                    "Command timed out after "
                    f"{int(self._TERMINAL_COMMAND_TIMEOUT_SECONDS)} seconds"
                ),
                exit_code=124,
            )
            payload["success"] = False
            return payload, 408
        except Exception as error:
            payload = self._build_terminal_payload(
                error=f"Failed to execute command: {error}",
                exit_code=1,
            )
            payload["success"] = False
            return payload, 500
        finally:
            if meta_path is not None:
                try:
                    os.unlink(meta_path)
                except OSError:
                    pass

    def _read_command_metadata(
        self,
        meta_path: str,
        *,
        fallback_cwd: str,
        fallback_exit_code: int,
    ) -> tuple[str, int]:
        """Read cwd and exit code written by the shell wrapper script.

        Args:
            meta_path: Path to the temporary metadata file.
            fallback_cwd: Working directory to use when metadata is missing.
            fallback_exit_code: Exit code to use when metadata is missing.

        Returns:
            Tuple of resolved working directory and exit code.
        """
        try:
            metadata_text = Path(meta_path).read_text(encoding="utf-8")
        except OSError:
            return fallback_cwd, fallback_exit_code

        metadata_lines = [
            line.strip() for line in metadata_text.splitlines() if line.strip()
        ]
        if len(metadata_lines) < 2:
            return fallback_cwd, fallback_exit_code

        resolved_cwd = metadata_lines[0]
        try:
            exit_code = int(metadata_lines[1])
        except ValueError:
            exit_code = fallback_exit_code

        return resolved_cwd, exit_code
