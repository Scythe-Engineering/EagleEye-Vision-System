from __future__ import annotations

import getpass
import os
import shlex
import shutil
import signal
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
TERMINAL_OUTPUT_LIMIT_BYTES = 1_000_000
BASH_EXECUTABLE = shutil.which("bash") or "/bin/bash"

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
        body = request.get_json(silent=True)
        command = body.get("command") if isinstance(body, dict) else None
        if not isinstance(command, str) or not command.strip():
            return self._terminal_payload(error="Command is required", exit_code=1), 400
        command = command.strip()
        password = body.get("sudo_password") if isinstance(body, dict) else None
        if password is not None and not isinstance(password, str):
            return self._terminal_payload(
                error="sudo_password must be a string", exit_code=1
            ), 400
        if password is not None and len(password) > 4096:
            return self._terminal_payload(
                error="sudo_password is too long", exit_code=1
            ), 400

        # ponytail: one lock for the whole process, the UI drives a single session.
        with _TERMINAL_LOCK:
            return self._run_terminal_command(command, password)

    def _run_terminal_command(
        self, command: str, sudo_password: str | None = None
    ) -> tuple[dict[str, Any], int]:
        """Run one command and carry any ``cd`` it performs over to the next call.

        Args:
            command: Shell command text to execute.
            sudo_password: Password supplied to sudo through a temporary askpass helper.

        Returns:
            Tuple of response payload and HTTP status code.
        """
        if not Path(self._terminal_cwd).is_dir():
            self._terminal_cwd = TERMINAL_HOME

        cwd_file = tempfile.NamedTemporaryFile(
            mode="w", prefix="eagleeye-terminal-", suffix=".cwd", delete=False
        )
        cwd_file.close()
        password_read_fd: int | None = None
        try:
            with (
                tempfile.TemporaryFile() as stdout_file,
                tempfile.TemporaryFile() as stderr_file,
                tempfile.TemporaryDirectory(prefix="eagleeye-sudo-") as sudo_dir,
            ):
                env = os.environ.copy()
                shell_prefix = ""
                pass_fds: tuple[int, ...] = ()
                if sudo_password is not None:
                    password_read_fd, password_write_fd = os.pipe()
                    os.write(password_write_fd, sudo_password.encode())
                    os.close(password_write_fd)
                    askpass_file = Path(sudo_dir, "askpass")
                    askpass_file.write_text(
                        '#!/bin/sh\ncat <&"$EAGLEEYE_SUDO_PASSWORD_FD"\n',
                        encoding="utf-8",
                    )
                    askpass_file.chmod(0o700)
                    env["SUDO_ASKPASS"] = str(askpass_file)
                    env["EAGLEEYE_SUDO_PASSWORD_FD"] = str(password_read_fd)
                    pass_fds = (password_read_fd,)
                    shell_prefix = (
                        'sudo() { command sudo -C "$((EAGLEEYE_SUDO_PASSWORD_FD + 1))" '
                        '-A "$@"; }\n'
                    )

                process = subprocess.Popen(
                    [
                        BASH_EXECUTABLE,
                        "-c",
                        shell_prefix
                        + f"{command}\n"
                        "__ee_status=$?\n"
                        f"pwd -P > {shlex.quote(cwd_file.name)}\n"
                        "exit $__ee_status\n",
                    ],
                    cwd=self._terminal_cwd,
                    env=env,
                    pass_fds=pass_fds,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    start_new_session=True,
                )
                try:
                    exit_code = process.wait(timeout=TERMINAL_TIMEOUT_SECONDS)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    return self._terminal_payload(
                        error=f"Command timed out after {TERMINAL_TIMEOUT_SECONDS} seconds",
                        exit_code=124,
                    ), 408

                stdout_file.seek(0)
                stderr_file.seek(0)
                output = stdout_file.read(TERMINAL_OUTPUT_LIMIT_BYTES + 1)
                error = stderr_file.read(TERMINAL_OUTPUT_LIMIT_BYTES + 1)

            if len(output) > TERMINAL_OUTPUT_LIMIT_BYTES:
                output = output[:TERMINAL_OUTPUT_LIMIT_BYTES] + b"\n[output truncated]"
            if len(error) > TERMINAL_OUTPUT_LIMIT_BYTES:
                error = error[:TERMINAL_OUTPUT_LIMIT_BYTES] + b"\n[output truncated]"

            next_cwd = Path(cwd_file.name).read_text(encoding="utf-8").strip()
            if next_cwd and Path(next_cwd).is_dir():
                self._terminal_cwd = next_cwd

            return self._terminal_payload(
                output=output.decode(errors="replace").rstrip("\n"),
                error=error.decode(errors="replace").rstrip("\n"),
                exit_code=exit_code,
            ), 200
        except OSError as error:
            return self._terminal_payload(
                error=f"Failed to execute command: {error}", exit_code=1
            ), 500
        finally:
            if password_read_fd is not None:
                os.close(password_read_fd)
            try:
                os.unlink(cwd_file.name)
            except OSError:
                pass
