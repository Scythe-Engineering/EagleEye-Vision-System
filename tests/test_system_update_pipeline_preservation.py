from __future__ import annotations

import subprocess
from pathlib import Path

from src.webui.web_server_utils.system_monitor_mixin import SystemMonitorMixin


class _UpdateHarness(SystemMonitorMixin):
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root

    def _repo_root(self) -> Path:
        return self.repo_root


class _UpdateInfoHarness(SystemMonitorMixin):
    def system_update_status(self) -> tuple[dict, int]:
        return {"available": True}, 200

    def _run_git_command(self, args: list[str], timeout: float = 30.0) -> str:
        del timeout
        responses = {
            ("fetch", "origin", "--prune"): "",
            ("rev-parse", "--abbrev-ref", "HEAD"): "feature/test",
            ("rev-parse", "--short", "HEAD"): "1111111",
            ("rev-parse", "HEAD"): "1" * 40,
            (
                "for-each-ref",
                "--format=%(refname:short) %(objectname:short) %(objectname)",
                "refs/remotes/origin",
            ): "\n".join(
                [
                    f"origin/feature/test 1111111 {'1' * 40}",
                    f"origin/main 2222222 {'2' * 40}",
                ]
            ),
        }
        return responses[tuple(args)]

    def log(self, _message: str) -> None:
        return None


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _configure_git_identity(repo: Path) -> None:
    _git(repo, "config", "user.email", "eagleeye-tests@example.com")
    _git(repo, "config", "user.name", "EagleEye Tests")


def test_git_pull_restores_local_pipeline_configuration(tmp_path: Path) -> None:
    origin = tmp_path / "origin.git"
    seed = tmp_path / "seed"
    working = tmp_path / "working"
    updater = tmp_path / "updater"

    origin.mkdir()
    _git(origin, "init", "--bare")

    seed.mkdir()
    _git(seed, "init", "--initial-branch=main")
    _configure_git_identity(seed)
    pipeline_path = seed / "src" / "config" / "pipeline_config.json"
    pipeline_path.parent.mkdir(parents=True)
    pipeline_path.write_text('{"source": "template-v1"}\n', encoding="utf-8")
    (seed / "version.txt").write_text("v1\n", encoding="utf-8")
    _git(seed, "add", ".")
    _git(seed, "commit", "-m", "Initial")
    _git(seed, "remote", "add", "origin", str(origin))
    _git(seed, "push", "-u", "origin", "main")
    _git(origin, "symbolic-ref", "HEAD", "refs/heads/main")

    _git(tmp_path, "clone", str(origin), str(working))
    local_pipeline = b'{"source": "local-runtime"}\n'
    working_pipeline = working / "src" / "config" / "pipeline_config.json"
    working_pipeline.write_bytes(local_pipeline)

    _git(tmp_path, "clone", str(origin), str(updater))
    _configure_git_identity(updater)
    (updater / "src" / "config" / "pipeline_config.json").write_text(
        '{"source": "template-v2"}\n', encoding="utf-8"
    )
    (updater / "version.txt").write_text("v2\n", encoding="utf-8")
    _git(updater, "add", ".")
    _git(updater, "commit", "-m", "Remote update")
    _git(updater, "push")

    output = _UpdateHarness(working)._pull_updates_preserving_pipeline_config()

    assert "Updating" in output
    assert working_pipeline.read_bytes() == local_pipeline
    assert (working / "version.txt").read_text(encoding="utf-8") == "v2\n"
    assert _git(working, "stash", "list") == ""
    assert "src/config/pipeline_config.json" in _git(
        working, "status", "--porcelain"
    )


def test_failed_pull_still_restores_pipeline_configuration(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--initial-branch=main")
    _configure_git_identity(repo)
    pipeline_path = repo / "src" / "config" / "pipeline_config.json"
    pipeline_path.parent.mkdir(parents=True)
    pipeline_path.write_text('{"source": "template"}\n', encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "Initial")

    local_pipeline = b'{"source": "local-runtime"}\n'
    pipeline_path.write_bytes(local_pipeline)

    try:
        _UpdateHarness(repo)._pull_updates_preserving_pipeline_config()
    except RuntimeError:
        pass
    else:
        raise AssertionError("git pull unexpectedly succeeded without a remote")

    assert pipeline_path.read_bytes() == local_pipeline
    assert _git(repo, "stash", "list") == ""


def test_update_info_tracks_main_instead_of_current_branch() -> None:
    payload, status = _UpdateInfoHarness().system_update_info()

    assert status == 200
    assert payload["default_branch"] == "main"
    assert payload["current_branch"] == "feature/test"
    assert payload["current_sha"] == "1111111"
    assert payload["remote_sha"] == "2222222"
    assert payload["update_needed"] is True
