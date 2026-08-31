from __future__ import annotations

import json
import os
import re
import selectors
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.colors import Colors
from src.webui.web_server_utils.constants import (
    DEFAULT_GENERAL_CONF,
    DEFAULT_VIEW_STREAM_DOWNSCALE,
    MAX_VIEW_STREAM_DOWNSCALE,
    MIN_VIEW_STREAM_DOWNSCALE,
    VIEW_STREAM_DOWNSCALE_KEY,
)

SYSTEM_UPDATE_APT_PHASES: list[tuple[str, list[str], float]] = [
    ("apt_update", ["sudo", "apt", "update"], 300.0),
    (
        "apt_upgrade",
        [
            "sudo",
            "env",
            "DEBIAN_FRONTEND=noninteractive",
            "apt",
            "upgrade",
            "-y",
        ],
        1800.0,
    ),
]

SYSTEM_UPDATE_PHASE_COUNT = 1 + len(SYSTEM_UPDATE_APT_PHASES)
SYSTEM_UPDATE_DEFAULT_BRANCH = "main"
_GIT_BRANCH_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._/\-]+$")


def _request():
    """Return the current Flask request, allowing monkeypatching via web_server.request."""
    import src.webui.web_server as _ws

    return _ws.request


def _general_conf_path():
    """Return GENERAL_CONF_PATH, allowing monkeypatching via web_server.GENERAL_CONF_PATH."""
    import src.webui.web_server as _ws

    return _ws.GENERAL_CONF_PATH


class SystemMonitorMixin:
    def shutdown(self) -> tuple[dict, int]:
        """
        Shutdown the web interface.

        Returns:
            tuple[dict, int]: A success or failure message.
        """
        try:
            import os

            os._exit(0)
        except Exception as e:
            self.log(f"Error during shutdown: {e}")
            return {"message": "Failed to shutdown server"}, 500

    def restart_backend(self) -> tuple[dict, int]:
        """
        Restart the backend.
        """
        self.restart_callback()
        return {"message": "Backend restarted successfully"}, 200

    def reboot_system(self) -> tuple[dict, int]:
        """Reboot the host machine on Linux via ``sudo reboot``.

        Returns:
            tuple[dict, int]: Acceptance or error payload with HTTP status.
        """
        if sys.platform != "linux":
            return {"error": "System reboot is only supported on Linux."}, 400

        try:
            result = subprocess.run(
                ["sudo", "-n", "reboot"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            self.log(f"Failed to initiate system reboot: {error}")
            return {"error": "Failed to initiate system reboot."}, 500

        if result.returncode != 0:
            error = result.stderr.strip() or "sudo reboot failed"
            self.log(f"Failed to initiate system reboot: {error}")
            return {"error": error}, 500

        self.log("System reboot initiated")
        return {"message": "System reboot initiated"}, 200

    def set_restart_required(self) -> tuple[dict, int]:
        """
        Set the restart required flag.
        """
        body = _request().get_json(silent=True) or {}
        required = bool(body.get("required", True))
        self.restart_required_for_config = required
        return {
            "message": "Restart required flag updated",
            "restart_required": self.restart_required_for_config,
            "runtime_id": getattr(self, "runtime_id", ""),
        }, 200

    def get_restart_required(self) -> tuple[dict, int]:
        """
        Get the restart required flag.
        """
        restart_required = bool(self.restart_required_for_config)
        try:
            restart_state = self._analyze_pipeline_restart_state(
                self._load_pipeline_config_file()
            )
            restart_required = restart_required or bool(
                restart_state.get("restart_required", False)
            )
            self.restart_required_for_config = restart_required
        except Exception as error:
            self.log(f"Failed to analyze pipeline restart state: {error}")

        return {
            "restart_required": restart_required,
            "runtime_id": getattr(self, "runtime_id", ""),
        }, 200

    def _has_active_wifi_connection(self) -> bool:
        try:
            if not hasattr(self, "_run_nmcli"):
                return False
            result = self._run_nmcli(
                ["-t", "-f", "TYPE,STATE", "device", "status"],
                timeout=5.0,
            )
            if result.returncode != 0:
                return False
            for line in result.stdout.splitlines():
                parts = line.split(":")
                if len(parts) >= 2 and parts[0] == "wifi" and parts[1] == "connected":
                    return True
        except Exception:
            return False
        return False

    def _has_internet_access(self) -> bool:
        try:
            socket.create_connection(("github.com", 443), timeout=3.0).close()
            return True
        except OSError:
            return False

    def system_update_status(self) -> tuple[dict, int]:
        """Return whether system update can run over WiFi with internet."""
        self._ensure_system_update_state()
        if not self._has_active_wifi_connection():
            payload: dict[str, Any] = {
                "available": False,
                "reason": "Connect to a WiFi network before updating.",
            }
        elif not self._has_internet_access():
            payload = {
                "available": False,
                "reason": "Connected WiFi network does not appear to have internet access.",
            }
        else:
            payload = {
                "available": True,
                "reason": "WiFi internet access available.",
            }

        payload["in_progress"] = bool(self._system_update_in_progress)
        payload["update_id"] = self._system_update_id
        if self._latest_system_update_progress is not None:
            payload["latest_progress"] = self._latest_system_update_progress
        return payload, 200

    def _repo_root(self) -> Path:
        """Return the repository root used by update commands."""
        return Path(__file__).resolve().parents[3]

    def _ensure_system_update_state(self) -> None:
        """Initialize system-update lock state if missing (e.g. in tests)."""
        if not hasattr(self, "_system_update_lock"):
            self._system_update_lock = threading.Lock()
        if not hasattr(self, "_system_update_in_progress"):
            self._system_update_in_progress = False
        if not hasattr(self, "_system_update_id"):
            self._system_update_id: str | None = None
        if not hasattr(self, "_latest_system_update_progress"):
            self._latest_system_update_progress: dict[str, Any] | None = None
        if not hasattr(self, "_system_update_target_branch"):
            self._system_update_target_branch: str | None = None

    def _run_git_command(self, args: list[str], timeout: float = 30.0) -> str:
        """Run a git command in the repository root and return stdout.

        Args:
            args: Git arguments after ``git``.
            timeout: Maximum seconds to wait for the command.

        Returns:
            str: Stripped stdout from the command.

        Raises:
            RuntimeError: If the command exits non-zero or times out.
        """
        result = subprocess.run(
            ["git", *args],
            cwd=self._repo_root(),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = "\n".join(
            part.strip() for part in [result.stdout, result.stderr] if part.strip()
        )
        if result.returncode != 0:
            raise RuntimeError(output or f"git {' '.join(args)} failed")
        return result.stdout.strip()

    def _normalize_git_branch_name(self, branch_name: str) -> str:
        """Validate and normalize a git branch name for update targeting.

        Args:
            branch_name: Requested branch name from the client.

        Returns:
            str: Normalized branch name.

        Raises:
            ValueError: If the branch name is empty or unsafe.
        """
        normalized_branch_name = branch_name.strip()
        if not normalized_branch_name:
            raise ValueError("Branch name is required.")
        if normalized_branch_name.startswith("-"):
            raise ValueError("Branch name cannot start with '-'.")
        if not _GIT_BRANCH_NAME_PATTERN.fullmatch(normalized_branch_name):
            raise ValueError("Branch name contains invalid characters.")
        return normalized_branch_name

    def _list_remote_branches_with_shas(self) -> list[dict[str, str]]:
        """List remote origin branches without downloading repository objects.

        Returns:
            list[dict[str, str]]: Remote branches as ``{name, sha}`` entries.
        """
        ref_output = self._run_git_command(["ls-remote", "--heads", "origin"])
        remote_branches: list[dict[str, str]] = []
        for line in ref_output.splitlines():
            parts = line.split()
            if len(parts) != 2 or not parts[1].startswith("refs/heads/"):
                continue
            full_sha, ref_name = parts
            branch_name = ref_name.removeprefix("refs/heads/")
            remote_branches.append(
                {"name": branch_name, "sha": full_sha[:7], "full_sha": full_sha}
            )
        remote_branches.sort(key=lambda branch: branch["name"].lower())
        return remote_branches

    def system_update_info(self) -> tuple[dict, int]:
        """Return current/remote git SHAs and remote branches for the update modal.

        Returns:
            tuple[dict, int]: Version/branch payload and HTTP status code.
        """
        status_payload, _ = self.system_update_status()
        if not status_payload.get("available"):
            return {
                "error": status_payload.get("reason", "WiFi internet required"),
                "available": False,
            }, 400

        try:
            current_branch = self._run_git_command(
                ["rev-parse", "--abbrev-ref", "HEAD"]
            )
            current_sha = self._run_git_command(["rev-parse", "--short", "HEAD"])
            current_sha_full = self._run_git_command(["rev-parse", "HEAD"])
            remote_branches = self._list_remote_branches_with_shas()
            remote_full_sha_by_branch = {
                branch["name"]: branch["full_sha"] for branch in remote_branches
            }
            remote_sha_by_branch = {
                branch["name"]: branch["sha"] for branch in remote_branches
            }
            remote_sha = remote_sha_by_branch.get(current_branch)
            remote_full_sha = remote_full_sha_by_branch.get(current_branch)
            update_needed = (
                remote_full_sha is None or remote_full_sha != current_sha_full
            )
            return {
                "available": True,
                "default_branch": SYSTEM_UPDATE_DEFAULT_BRANCH,
                "current_branch": current_branch,
                "current_sha": current_sha,
                "remote_sha": remote_sha,
                "update_needed": update_needed,
                "remote_branches": [
                    {"name": branch["name"], "sha": branch["sha"]}
                    for branch in remote_branches
                ],
            }, 200
        except subprocess.TimeoutExpired:
            return {"error": "Timed out while fetching remote git state."}, 504
        except Exception as error:
            message = str(error)
            self.log(f"Failed to load system update info: {message}")
            return {"error": message}, 500

    def _publish_system_update_progress(
        self,
        *,
        phase: str,
        phase_index: int,
        phase_count: int,
        percent: int,
        line: str | None = None,
        done: bool = False,
        error: str | None = None,
    ) -> None:
        """Publish a system update progress event over SSE.

        Args:
            phase: Current update phase identifier.
            phase_index: Zero-based index of the current phase.
            phase_count: Total number of update phases.
            percent: Overall progress percentage from 0 to 100.
            line: Optional terminal output line to append.
            done: Whether the update sequence has finished.
            error: Optional error message when the update fails.
        """
        self._ensure_system_update_state()
        payload: dict[str, Any] = {
            "phase": phase,
            "phase_index": phase_index,
            "phase_count": phase_count,
            "percent": max(0, min(100, int(percent))),
            "done": done,
        }
        if self._system_update_id is not None:
            payload["update_id"] = self._system_update_id
        if line is not None:
            payload["line"] = line
        if error is not None:
            payload["error"] = error
        self._latest_system_update_progress = payload
        self._publish_event("system_update_progress", payload)

    def _replay_cached_system_update_progress(self) -> None:
        """Republish the latest cached system-update event for SSE reconnects."""
        self._ensure_system_update_state()
        if self._latest_system_update_progress is None:
            return
        self._publish_event(
            "system_update_progress", self._latest_system_update_progress
        )

    def _run_update_command_streaming(
        self,
        command: list[str],
        timeout: float,
        *,
        phase: str,
        phase_index: int,
        phase_count: int,
        percent_start: int,
        percent_end: int,
    ) -> None:
        """Run an update command and stream stdout/stderr lines over SSE.

        Args:
            command: Command argv to execute.
            timeout: Maximum seconds to wait for the command.
            phase: Phase identifier for progress events.
            phase_index: Zero-based phase index.
            phase_count: Total phase count.
            percent_start: Progress percent at command start.
            percent_end: Progress percent when the command completes.

        Raises:
            RuntimeError: If the command exits non-zero or times out.
        """
        display_command = " ".join(command)
        self._publish_system_update_progress(
            phase=phase,
            phase_index=phase_index,
            phase_count=phase_count,
            percent=percent_start,
            line=f"$ {display_command}",
        )

        process = subprocess.Popen(
            command,
            cwd=self._repo_root(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        deadline = time.monotonic() + timeout
        assert process.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)

        try:
            while True:
                remaining_seconds = deadline - time.monotonic()
                if remaining_seconds <= 0:
                    process.kill()
                    process.wait(timeout=5)
                    raise RuntimeError(f"Update command timed out: {display_command}")

                ready = selector.select(timeout=min(remaining_seconds, 0.5))
                if ready:
                    line = process.stdout.readline()
                    if line:
                        self._publish_system_update_progress(
                            phase=phase,
                            phase_index=phase_index,
                            phase_count=phase_count,
                            percent=percent_start,
                            line=line.rstrip("\n"),
                        )
                        continue
                    try:
                        return_code = process.wait(timeout=remaining_seconds)
                    except subprocess.TimeoutExpired as error:
                        process.kill()
                        process.wait(timeout=5)
                        raise RuntimeError(
                            f"Update command timed out: {display_command}"
                        ) from error
                else:
                    return_code = process.poll()
                    if return_code is None:
                        continue

                remaining = process.stdout.read()
                if remaining:
                    for leftover_line in remaining.splitlines():
                        self._publish_system_update_progress(
                            phase=phase,
                            phase_index=phase_index,
                            phase_count=phase_count,
                            percent=percent_start,
                            line=leftover_line,
                        )
                if return_code != 0:
                    raise RuntimeError(
                        f"Command failed ({return_code}): {display_command}"
                    )
                self._publish_system_update_progress(
                    phase=phase,
                    phase_index=phase_index,
                    phase_count=phase_count,
                    percent=percent_end,
                )
                return
        finally:
            selector.close()
            process.stdout.close()

    def _execute_system_update(self) -> None:
        """Run the full system update sequence and publish SSE progress."""
        phase_count = SYSTEM_UPDATE_PHASE_COUNT
        target_branch = (
            self._system_update_target_branch or SYSTEM_UPDATE_DEFAULT_BRANCH
        )
        try:
            self._checkout_branch_preserving_pipeline_config(
                target_branch,
                phase_count=phase_count,
            )

            for apt_phase_index, (phase_name, command, timeout) in enumerate(
                SYSTEM_UPDATE_APT_PHASES
            ):
                phase_index = apt_phase_index + 1
                percent_start = int((phase_index / phase_count) * 90)
                percent_end = int(((phase_index + 1) / phase_count) * 90)
                self._run_update_command_streaming(
                    command,
                    timeout,
                    phase=phase_name,
                    phase_index=phase_index,
                    phase_count=phase_count,
                    percent_start=percent_start,
                    percent_end=percent_end,
                )

            self._publish_system_update_progress(
                phase="complete",
                phase_index=phase_count,
                phase_count=phase_count,
                percent=100,
                line="Update completed successfully. Restarting backend...",
                done=True,
            )
            self.log("System update completed successfully")
        except Exception as error:
            message = str(error)
            self.log(f"System update failed: {message}")
            self._publish_system_update_progress(
                phase="error",
                phase_index=0,
                phase_count=phase_count,
                percent=0,
                line=message,
                done=True,
                error=message,
            )
        finally:
            self._ensure_system_update_state()
            with self._system_update_lock:
                self._system_update_in_progress = False

    def _checkout_branch_preserving_pipeline_config(
        self,
        target_branch: str,
        *,
        phase_count: int,
    ) -> None:
        """Fetch and checkout a branch while restoring local pipeline configuration.

        Args:
            target_branch: Remote branch name to check out.
            phase_count: Total update phase count for progress reporting.
        """
        repo_root = self._repo_root()
        pipeline_path = repo_root / "src" / "config" / "pipeline_config.json"
        relative_path = pipeline_path.relative_to(repo_root).as_posix()
        pipeline_existed = pipeline_path.exists()
        pipeline_contents = pipeline_path.read_bytes() if pipeline_existed else None

        status = self._run_git_command(
            ["status", "--porcelain", "--", relative_path],
            timeout=30.0,
        )
        stash_created = bool(status)
        if stash_created:
            self._publish_system_update_progress(
                phase="git_pull",
                phase_index=0,
                phase_count=phase_count,
                percent=0,
                line="Backing up local pipeline configuration...",
            )
            self._run_git_command(
                [
                    "stash",
                    "push",
                    "--include-untracked",
                    "--message",
                    "EagleEye system update pipeline backup",
                    "--",
                    relative_path,
                ],
                timeout=30.0,
            )

        try:
            self._run_update_command_streaming(
                [
                    "git",
                    "fetch",
                    "--depth=1",
                    "origin",
                    f"+refs/heads/{target_branch}:refs/remotes/origin/{target_branch}",
                ],
                120.0,
                phase="git_pull",
                phase_index=0,
                phase_count=phase_count,
                percent_start=0,
                percent_end=15,
            )
            self._run_update_command_streaming(
                [
                    "git",
                    "checkout",
                    "-B",
                    target_branch,
                    f"origin/{target_branch}",
                ],
                60.0,
                phase="git_pull",
                phase_index=0,
                phase_count=phase_count,
                percent_start=15,
                percent_end=30,
            )
        finally:
            if pipeline_contents is None:
                pipeline_path.unlink(missing_ok=True)
            else:
                pipeline_path.parent.mkdir(parents=True, exist_ok=True)
                pipeline_path.write_bytes(pipeline_contents)
                self._publish_system_update_progress(
                    phase="git_pull",
                    phase_index=0,
                    phase_count=phase_count,
                    percent=30,
                    line="Restored src/config/pipeline_config.json",
                )

            if stash_created:
                self._run_git_command(
                    ["stash", "drop", "stash@{0}"],
                    timeout=30.0,
                )

    def _pull_updates_preserving_pipeline_config(self) -> str:
        """Pull repository updates while restoring the local pipeline configuration.

        Returns:
            str: Combined stdout from the git pull command.
        """
        repo_root = self._repo_root()
        pipeline_path = repo_root / "src" / "config" / "pipeline_config.json"
        relative_path = pipeline_path.relative_to(repo_root).as_posix()
        pipeline_existed = pipeline_path.exists()
        pipeline_contents = pipeline_path.read_bytes() if pipeline_existed else None

        status = self._run_git_command(
            ["status", "--porcelain", "--", relative_path],
            timeout=30.0,
        )
        stash_created = bool(status)
        if stash_created:
            self._run_git_command(
                [
                    "stash",
                    "push",
                    "--include-untracked",
                    "--message",
                    "EagleEye system update pipeline backup",
                    "--",
                    relative_path,
                ],
                timeout=30.0,
            )

        try:
            return self._run_git_command(["pull"], timeout=120.0)
        finally:
            if pipeline_contents is None:
                pipeline_path.unlink(missing_ok=True)
            else:
                pipeline_path.parent.mkdir(parents=True, exist_ok=True)
                pipeline_path.write_bytes(pipeline_contents)

            if stash_created:
                self._run_git_command(
                    ["stash", "drop", "stash@{0}"],
                    timeout=30.0,
                )

    def run_system_update(self) -> tuple[dict, int]:
        """Start git checkout/pull and apt updates in a background thread; stream via SSE.

        Returns:
            tuple[dict, int]: Acceptance payload and HTTP status code.
        """
        self._ensure_system_update_state()
        status_payload, _ = self.system_update_status()
        if not status_payload.get("available"):
            return {
                "error": status_payload.get("reason", "WiFi internet required")
            }, 400

        body = _request().get_json(silent=True) or {}
        requested_branch = body.get("branch")
        try:
            if isinstance(requested_branch, str) and requested_branch.strip():
                target_branch = self._normalize_git_branch_name(requested_branch)
            else:
                target_branch = SYSTEM_UPDATE_DEFAULT_BRANCH
        except ValueError as error:
            return {"error": str(error)}, 400
        except Exception as error:
            return {"error": f"Unable to resolve target branch: {error}"}, 500

        update_id = str(uuid.uuid4())
        with self._system_update_lock:
            if self._system_update_in_progress:
                return {"error": "A system update is already in progress."}, 409
            self._system_update_in_progress = True
            self._system_update_id = update_id
            self._system_update_target_branch = target_branch
            self._latest_system_update_progress = None

        phase_count = SYSTEM_UPDATE_PHASE_COUNT
        self._publish_system_update_progress(
            phase="starting",
            phase_index=0,
            phase_count=phase_count,
            percent=0,
            line=f"Starting system update on branch '{target_branch}'...",
        )
        update_thread = threading.Thread(
            target=self._execute_system_update,
            name="system-update",
            daemon=True,
        )
        update_thread.start()
        return {
            "started": True,
            "message": "System update started",
            "update_id": update_id,
            "branch": target_branch,
        }, 202

    def get_log_messages(self) -> tuple[dict, int]:
        """
        Get all log messages from the logger instance.

        Returns:
            tuple[dict, int]: Dictionary containing log messages and HTTP status code.
        """
        if self.logger is None:
            return {"messages": [], "error": "Logger instance not available"}, 503

        try:
            log_lines = self.logger.message_history.to_file_lines()

            return {"messages": log_lines, "total_count": len(log_lines)}, 200
        except Exception as e:
            self.logger.log(f"Error retrieving log messages: {e}")
            return {"messages": [], "error": str(e)}, 500

    def download_log_file(self) -> tuple[str, int] | tuple[dict, int]:
        """
        Download the log file.
        """
        try:
            with open(os.path.join(self.logger.current_log_file), "r") as f:
                return f.read(), 200
        except Exception as e:
            self.logger.log(
                f"{Colors.RED}Error downloading log file: {e}{Colors.RESET}"
            )
            return {"error": str(e)}, 500

    def _log_monitor_loop(self) -> None:
        """
        Monitor the logger for new messages and publish them via SSE.
        """
        if self.logger is None:
            return

        while True:
            try:
                current_message_count = len(self.logger.message_history.messages)

                if current_message_count > self.last_log_message_count:
                    message_lines = self.logger.message_history.to_file_lines()

                    if message_lines:
                        self._publish_event(
                            "log_update",
                            {
                                "messages": message_lines,
                            },
                        )

                    self.last_log_message_count = current_message_count

                time.sleep(0.1)
            except Exception as e:
                self.logger.log(f"Error in log monitor loop: {e}")
                time.sleep(1.0)

    def get_system_status(self) -> tuple[dict, int]:
        """
        Get current system status metrics.

        Returns:
            tuple[dict, int]: Dictionary containing system metrics.
        """
        payload = self._build_system_status_payload()
        return payload, 200

    def _system_status_loop(self) -> None:
        """
        Publish system status metrics via SSE on a fixed interval.
        """
        while True:
            try:
                payload = self._build_system_status_payload()
                self._publish_event("system_status", payload)
            except Exception as e:
                self.log(f"Error publishing system status: {e}")
            time.sleep(self._system_status_interval)

    def _build_system_status_payload(self) -> dict[str, Any]:
        """
        Build the system status payload with platform-aware fallbacks.

        Returns:
            dict[str, Any]: Structured system status payload.
        """
        cpu_payload: dict[str, Any] = {"status": "unavailable"}
        memory_payload: dict[str, Any] = {"status": "unavailable"}
        storage_payload: dict[str, Any] = {"status": "unavailable"}
        pipeline_payload = self._build_pipeline_status_list()
        network_table_payload = self._build_network_table_status()

        try:
            import psutil

            cpu_payload = {
                "percent": float(psutil.cpu_percent(interval=None)),
                "cores": int(psutil.cpu_count(logical=True) or 0),
                "temperature_c": self._read_cpu_temperature_c(psutil),
                "status": "ok",
            }
            memory = psutil.virtual_memory()
            memory_payload = {
                "percent": float(memory.percent),
                "used_mb": float(memory.used / (1024 * 1024)),
                "total_mb": float(memory.total / (1024 * 1024)),
                "status": "ok",
            }
            disk = psutil.disk_usage("/")
            storage_payload = {
                "percent": float(disk.percent),
                "used_gb": float(disk.used / (1024 * 1024 * 1024)),
                "total_gb": float(disk.total / (1024 * 1024 * 1024)),
                "status": "ok",
            }
            self._system_status_error_logged = False
        except Exception as e:
            message = str(e)
            cpu_payload = {"status": "unavailable", "error": message}
            memory_payload = {"status": "unavailable", "error": message}
            storage_payload = {"status": "unavailable", "error": message}
            if not self._system_status_error_logged:
                self.log(f"System status metrics unavailable: {message}")
                self._system_status_error_logged = True

        return {
            "cpu": cpu_payload,
            "memory": memory_payload,
            "storage": storage_payload,
            "pipelines": pipeline_payload,
            "network_table": network_table_payload,
        }

    def _read_cpu_temperature_c(self, psutil_module: Any) -> float | None:
        """Read CPU temperature in Celsius, preferring Linux thermal sensors."""
        try:
            thermal_zones = sorted(
                Path("/sys/class/thermal").glob("thermal_zone*/temp")
            )
            temperatures: list[float] = []
            for temp_path in thermal_zones:
                try:
                    raw_value = temp_path.read_text(encoding="utf-8").strip()
                    value = float(raw_value)
                    if value > 1000:
                        value /= 1000.0
                    if 0.0 < value < 150.0:
                        temperatures.append(value)
                except Exception:
                    continue
            if temperatures:
                return max(temperatures)
        except Exception:
            pass

        try:
            sensors_temperatures = getattr(psutil_module, "sensors_temperatures", None)
            if sensors_temperatures is None:
                return None
            readings = sensors_temperatures(fahrenheit=False)
            temperatures = []
            for entries in readings.values():
                for entry in entries:
                    current = getattr(entry, "current", None)
                    if (
                        isinstance(current, (int, float))
                        and 0.0 < float(current) < 150.0
                    ):
                        label = str(getattr(entry, "label", "") or "").lower()
                        if "cpu" in label or "core" in label or not label:
                            temperatures.append(float(current))
            if temperatures:
                return max(temperatures)
        except Exception:
            pass

        return None

    def _build_network_table_status(self) -> dict[str, Any]:
        """
        Build NetworkTables client connection status for the frontend.

        Returns:
            dict[str, Any]: Connection status and configured server address.
        """
        try:
            server_address = self._read_general_conf().get("network_table_address", "")
        except Exception:
            server_address = ""

        instance = getattr(self, "network_table_instance", None)
        if instance is None:
            return {
                "status": "unavailable",
                "connected": False,
                "server": server_address,
                "connection_count": 0,
            }

        try:
            connected = bool(instance.isConnected())
            connections = instance.getConnections()
            return {
                "status": "ok",
                "connected": connected,
                "server": server_address,
                "connection_count": len(connections),
            }
        except Exception as error:
            return {
                "status": "unavailable",
                "connected": False,
                "server": server_address,
                "connection_count": 0,
                "error": str(error),
            }

    def _build_pipeline_status_list(self) -> list[dict[str, Any]]:
        """
        Build a list of pipelines with live active status.

        Returns:
            list[dict[str, Any]]: Pipeline status list.
        """
        try:
            pipeline_names = self.get_pipeline_names()
        except Exception as error:
            self.log(
                f"{Colors.RED}Error loading pipeline names for status: {error}{Colors.RESET}"
            )
            pipeline_names = []

        try:
            pipeline_objects = self.pipeline_objects_callback()
        except Exception as error:
            self.log(
                f"{Colors.RED}Error loading pipeline objects for status: {error}{Colors.RESET}"
            )
            pipeline_objects = {}

        statuses: list[dict[str, Any]] = []
        pipeline_objects_available = bool(pipeline_objects)
        for pipeline_name in pipeline_names:
            pipeline = pipeline_objects.get(pipeline_name)
            if pipeline is None:
                if pipeline_objects_available:
                    self.log(
                        f"{Colors.YELLOW}Pipeline {pipeline_name} not found in pipeline objects callback.{Colors.RESET}"
                    )
                statuses.append({"name": pipeline_name, "active": False})
                continue

            try:
                is_active = bool(pipeline.is_active())
            except Exception as error:
                self.log(
                    f"{Colors.RED}Failed to read active status for pipeline {pipeline_name}: {error}{Colors.RESET}"
                )
                is_active = False
            statuses.append({"name": pipeline_name, "active": is_active})

        return statuses

    def _read_general_conf(self) -> dict[str, Any]:
        """Read general config with defaults for missing optional settings."""
        conf_path = _general_conf_path()
        if not conf_path.exists():
            return DEFAULT_GENERAL_CONF.copy()

        with conf_path.open("r", encoding="utf-8") as f:
            config = json.load(f)

        if not isinstance(config, dict):
            raise ValueError("General configuration must be a JSON object")

        return {**DEFAULT_GENERAL_CONF, **config}

    def _parse_view_stream_downscale(self, value: Any) -> float:
        """Validate the view stream downscale setting."""
        try:
            downscale = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError("View stream downscale must be a number") from error

        if not MIN_VIEW_STREAM_DOWNSCALE <= downscale <= MAX_VIEW_STREAM_DOWNSCALE:
            raise ValueError(
                "View stream downscale must be between "
                f"{MIN_VIEW_STREAM_DOWNSCALE} and {MAX_VIEW_STREAM_DOWNSCALE}"
            )

        return downscale

    def _refresh_view_stream_settings(self) -> None:
        """Load view stream settings from the general configuration file."""
        try:
            general_conf = self._read_general_conf()
            downscale = self._parse_view_stream_downscale(
                general_conf.get(
                    VIEW_STREAM_DOWNSCALE_KEY,
                    DEFAULT_VIEW_STREAM_DOWNSCALE,
                )
            )
        except Exception as error:
            self.log(f"Failed loading view stream settings, using defaults: {error}")
            downscale = DEFAULT_VIEW_STREAM_DOWNSCALE

        with self._general_conf_lock:
            self.view_stream_downscale = downscale

    def get_general_conf(self) -> tuple[dict, int]:
        """
        Get the general configuration.
        """
        try:
            return self._read_general_conf(), 200
        except Exception as e:
            return {"error": str(e)}, 500

    def save_general_conf(self) -> tuple[dict, int]:
        """
        Save the general configuration.
        """
        try:
            payload = _request().get_json(silent=True)
            if not isinstance(payload, dict):
                return {"error": "Expected JSON object payload"}, 400

            with self._general_conf_lock:
                config = {**self._read_general_conf(), **payload}
                config[VIEW_STREAM_DOWNSCALE_KEY] = self._parse_view_stream_downscale(
                    config.get(
                        VIEW_STREAM_DOWNSCALE_KEY,
                        DEFAULT_VIEW_STREAM_DOWNSCALE,
                    )
                )

                with _general_conf_path().open("w", encoding="utf-8") as config_file:
                    json.dump(config, config_file, indent=4)
                    config_file.write("\n")

                self.view_stream_downscale = config[VIEW_STREAM_DOWNSCALE_KEY]

            return {"message": "General configuration saved successfully"}, 200
        except ValueError as e:
            return {"error": str(e)}, 400
        except Exception as e:
            self.logger.log(
                f"{Colors.RED}Error saving general configuration: {e}{Colors.RESET}"
            )
            return {"error": str(e)}, 500

    def update_robot_position(self, transformation_matrix: np.ndarray) -> None:
        """
        Push the tracked robot's transformation matrix to the frontend via SSE.

        Args:
            transformation_matrix (np.ndarray): The new transformation matrix as a 4x4 numpy array.
        """
        if transformation_matrix.shape != (4, 4):
            raise ValueError("Transformation matrix must be a 4x4 numpy array.")

        if not np.all(np.isfinite(transformation_matrix)):
            self.log("Skipping publish of robot transform due to non-finite values")
            return

        matrix_list = transformation_matrix.tolist()
        try:
            self._publish_event(
                "update_robot_transform", {"transform_matrix": matrix_list}
            )
        except Exception:
            self.log("Failed to publish update_robot_transform via SSE")

    def update_camera_pose(
        self, camera_bus_id: str, transformation_matrix: np.ndarray
    ) -> None:
        """Publish a camera pose update for 3D visualization via SSE.

        Args:
            camera_bus_id: Stable camera identifier used across pipelines and UI.
            transformation_matrix: 4x4 camera pose transform in world space.
        """
        if transformation_matrix.shape != (4, 4):
            raise ValueError("Transformation matrix must be a 4x4 numpy array.")

        if not np.all(np.isfinite(transformation_matrix)):
            self.log("Skipping publish of camera transform due to non-finite values")
            return

        resolved_camera_name: str | None = None
        for camera_name, camera_info in self.available_cameras.items():
            if not isinstance(camera_info, dict):
                continue
            if str(camera_info.get("bus_id") or "") != str(camera_bus_id):
                continue
            resolved_camera_name = str(camera_name)
            break

        try:
            self._publish_event(
                "update_camera_pose",
                {
                    "camera_bus_id": str(camera_bus_id),
                    "camera_name": resolved_camera_name or str(camera_bus_id),
                    "transform_matrix": transformation_matrix.tolist(),
                    "timestamp_ms": int(time.time() * 1000),
                },
            )
        except Exception:
            self.log("Failed to publish update_camera_pose via SSE")

    def update_detected_objects(self, detections: list[dict[str, Any]]) -> None:
        """
        Publish detected objects for 3D visualization.

        Args:
            detections (list[dict[str, Any]]): Detected objects with 3D positions and metadata.

        Returns:
            None: This method does not return a value.
        """
        if not isinstance(detections, list):
            return

        validated_detections: list[dict[str, Any]] = []
        for detection in detections:
            if not isinstance(detection, dict):
                continue

            position = detection.get("position_3d")
            if not (
                isinstance(position, (list, tuple))
                and len(position) == 3
                and all(isinstance(coord, (int, float)) for coord in position)
            ):
                continue

            position_values = [float(coord) for coord in position]
            if not np.all(np.isfinite(position_values)):
                continue

            detection_payload: dict[str, Any] = {"position_3d": position_values}

            class_id = detection.get("class_id")
            if isinstance(class_id, (int, float, str)):
                detection_payload["class_id"] = class_id

            confidence = detection.get("confidence")
            if isinstance(confidence, (int, float)) and np.isfinite(confidence):
                detection_payload["confidence"] = float(confidence)

            class_name = detection.get("class_name")
            if class_name is not None:
                detection_payload["class_name"] = str(class_name)

            validated_detections.append(detection_payload)

        try:
            self._publish_event(
                "update_detected_objects", {"detections": validated_detections}
            )
        except Exception:
            self.log("Failed to publish update_detected_objects via SSE")
