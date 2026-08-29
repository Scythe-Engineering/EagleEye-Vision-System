"""Tests for the curl-fetched installer script and fresh-install setup."""

from __future__ import annotations

import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INSTALL_SCRIPT = PROJECT_ROOT / "install.sh"
PIPELINE_CONFIG = PROJECT_ROOT / "src" / "config" / "pipeline_config.json"


def _run_installer_function(script: str) -> subprocess.CompletedProcess[str]:
    """Source the installer and run a shell snippet without installing.

    Args:
        script: Shell source to run after loading installer functions.

    Returns:
        The completed Bash process.
    """
    return subprocess.run(
        [
            "bash",
            "-c",
            f'EAGLEEYE_INSTALL_LIB_ONLY=1 . "{INSTALL_SCRIPT}"\nset +e\n{script}',
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def test_install_script_is_syntactically_valid() -> None:
    """Verify Bash accepts the installer syntax."""
    result = subprocess.run(
        ["bash", "-n", str(INSTALL_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_sourcing_the_script_does_not_run_the_install() -> None:
    """Verify library mode only defines installer functions."""
    result = _run_installer_function('echo "sourced-only"')
    assert result.returncode == 0, result.stderr
    assert "sourced-only" in result.stdout
    assert "[1/" not in result.stdout


def test_check_platform_warns_but_continues_on_untested_platform() -> None:
    """Verify unsupported platforms warn without stopping installation."""
    result = _run_installer_function(
        'check_platform x86_64 ubuntu 24.04; echo "EXIT=$?"'
    )
    assert "EXIT=0" in result.stdout
    assert "Untested architecture 'x86_64'" in result.stderr
    assert "Untested OS 'ubuntu 24.04'" in result.stderr


def test_check_platform_is_silent_on_the_tested_platform() -> None:
    """Verify the supported platform emits no compatibility warning."""
    result = _run_installer_function('check_platform aarch64 debian 12; echo "EXIT=$?"')
    assert "EXIT=0" in result.stdout
    assert "Untested" not in result.stderr


def test_check_not_already_installed_refuses_and_points_at_the_web_ui(
    tmp_path: Path,
) -> None:
    """Verify existing installs are preserved and directed to the updater.

    Args:
        tmp_path: Temporary directory supplied by pytest.
    """
    existing = tmp_path / "EagleEye-Vision-System"
    existing.mkdir()

    result = _run_installer_function(
        f'check_not_already_installed "{existing}"; echo "EXIT=$?"'
    )
    assert "EXIT=1" in result.stdout
    assert "already exists" in result.stderr
    assert "System Update" in result.stderr


def test_check_not_already_installed_allows_a_fresh_target(tmp_path: Path) -> None:
    """Verify a missing target directory passes the freshness check.

    Args:
        tmp_path: Temporary directory supplied by pytest.
    """
    missing = tmp_path / "EagleEye-Vision-System"
    result = _run_installer_function(
        f'check_not_already_installed "{missing}"; echo "EXIT=$?"'
    )
    assert "EXIT=0" in result.stdout


def test_helper_arguments_do_not_overwrite_main_install_path(tmp_path: Path) -> None:
    """Verify helper-local arguments do not change main state.

    Args:
        tmp_path: Temporary directory supplied by pytest.
    """
    missing = tmp_path / "EagleEye-Vision-System"
    result = _run_installer_function(
        f'install_dir="/home/pilot/EagleEye-Vision-System"; '
        f'check_not_already_installed "{missing}"; echo "$install_dir"'
    )
    assert result.stdout.strip() == "/home/pilot/EagleEye-Vision-System"


def test_system_artifacts_refuse_a_fresh_install(tmp_path: Path) -> None:
    """Verify existing system files block a fresh installation.

    Args:
        tmp_path: Temporary directory supplied by pytest.
    """
    service = tmp_path / "eagleeye.service"
    sudoers = tmp_path / "eagleeye-sudoers"
    service.write_text("existing", encoding="utf-8")
    result = _run_installer_function(
        f'check_no_system_artifacts "{service}" "{sudoers}"; echo "EXIT=$?"'
    )
    assert "EXIT=1" in result.stdout
    assert "previous EagleEye service" in result.stderr
    assert service.read_text(encoding="utf-8") == "existing"


def test_system_artifact_check_allows_clean_paths(tmp_path: Path) -> None:
    """Verify absent system files pass the freshness check.

    Args:
        tmp_path: Temporary directory supplied by pytest.
    """
    result = _run_installer_function(
        f'check_no_system_artifacts "{tmp_path / "service"}" '
        f'"{tmp_path / "sudoers"}"; echo "EXIT=$?"'
    )
    assert "EXIT=0" in result.stdout


def test_os_release_value_reads_quoted_fields(tmp_path: Path) -> None:
    """Verify OS metadata parsing removes field quotes.

    Args:
        tmp_path: Temporary directory supplied by pytest.
    """
    os_release = tmp_path / "os-release"
    os_release.write_text(
        'PRETTY_NAME="Debian GNU/Linux 12 (bookworm)"\nID=debian\nVERSION_ID="12"\n',
        encoding="utf-8",
    )
    result = _run_installer_function(
        f'os_release_value ID "{os_release}"; os_release_value VERSION_ID "{os_release}"'
    )
    assert result.stdout.split() == ["debian", "12"]


def test_render_service_unit_uses_the_current_user_and_path() -> None:
    """Verify the generated service uses only argument-derived identity and paths."""
    result = _run_installer_function(
        "render_service_unit pilot /home/pilot /home/pilot/EagleEye-Vision-System"
    )
    unit = result.stdout
    assert "User=pilot\n" in unit
    assert "User=eagle\n" not in unit
    assert "WorkingDirectory=/home/pilot/EagleEye-Vision-System" in unit
    assert "Environment=SERVICE_NAME=eagleeye" in unit
    assert (
        "ExecStart=/home/pilot/EagleEye-Vision-System/.venv/bin/python -m src.main_backend"
        in unit
    )
    assert "WantedBy=multi-user.target" in unit
    # uv and rustup install into per-user bin directories that systemd omits.
    assert "/home/pilot/.local/bin" in unit
    assert "/home/pilot/.cargo/bin" in unit


def test_render_sudoers_policy_allows_only_backend_commands() -> None:
    """Verify the sudoers policy grants only required backend commands."""
    result = _run_installer_function("render_sudoers_policy pilot 1001")
    policy = result.stdout
    assert result.returncode == 0, result.stderr
    assert "#1001 ALL=(root) NOPASSWD:" in policy
    assert "/usr/bin/apt update" in policy
    assert "/usr/bin/env DEBIAN_FRONTEND=noninteractive apt upgrade -y" in policy
    assert "/usr/bin/systemctl restart eagleeye" in policy
    assert "/usr/sbin/reboot" in policy
    assert "ALL=(ALL" not in policy
    assert "*" not in policy
    assert "sudo visudo -cf" in INSTALL_SCRIPT.read_text(encoding="utf-8")


def test_repository_url_can_be_overridden_for_pre_release_testing() -> None:
    """Verify pre-release tests can override the repository URL."""
    result = subprocess.run(
        [
            "bash",
            "-c",
            (
                f'EAGLEEYE_REPO_URL="file:///tmp/eagleeye.git" '
                f'EAGLEEYE_INSTALL_LIB_ONLY=1 . "{INSTALL_SCRIPT}"; echo "$REPO_URL"'
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.stdout.strip() == "file:///tmp/eagleeye.git"


def test_failed_install_cleanup_removes_only_its_staging_and_target(
    tmp_path: Path,
) -> None:
    """Verify failed fresh installs remove only paths created by that run.

    Args:
        tmp_path: Temporary directory supplied by pytest.
    """
    staging = tmp_path / ".EagleEye-Vision-System.installing.test"
    target = tmp_path / "EagleEye-Vision-System"
    staging.mkdir()
    target.mkdir()
    (staging / "partial").write_text("staging", encoding="utf-8")
    (target / "partial").write_text("target", encoding="utf-8")
    result = _run_installer_function(
        f'staging_dir="{staging}"; install_dir="{target}"; '
        "install_dir_created=1; cleanup_failed_install 1"
    )
    assert result.returncode == 1
    assert not staging.exists()
    assert not target.exists()


def test_web_readiness_failure_prints_the_service_journal() -> None:
    """Verify readiness timeouts print actionable service logs."""
    result = _run_installer_function(
        "WEB_SERVER_READY_TIMEOUT=0; "
        "curl() { return 1; }; "
        "date() { echo 0; }; "
        'sudo() { printf "sudo:%s\\n" "$*"; }; '
        'wait_for_web_server; echo "EXIT=$?"'
    )
    assert "EXIT=1" in result.stdout
    assert "127.0.0.1:5001" in result.stderr
    assert "sudo:journalctl -u eagleeye.service -n 50 --no-pager" in result.stderr


def test_failed_verification_preserves_completed_install(tmp_path: Path) -> None:
    """Verify failed readiness leaves the completed install for diagnosis.

    Args:
        tmp_path: Temporary directory supplied by pytest.
    """
    result = _run_installer_function(
        f'HOME="{tmp_path}"; '
        "check_user() { :; }; check_platform() { :; }; "
        "check_not_already_installed() { :; }; check_no_system_artifacts() { :; }; "
        "install_apt_packages() { :; }; install_uv() { :; }; install_node() { :; }; "
        "install_rust() { :; }; clone_repository() { :; }; "
        "install_python_dependencies() { :; }; install_frontend() { :; }; "
        "configure_camera_permissions() { :; }; install_sudoers_policy() { :; }; "
        "install_service() { :; }; verify_install() { return 1; }; "
        "main; status=$?; "
        'printf "STATUS=%s FLAGS=%s,%s,%s TARGET=%s\\n" "$status" '
        '"$install_dir_created" "$service_installed" "$sudoers_installed" '
        '"$(test -d "$install_dir" && echo present || echo missing)"'
    )
    assert "STATUS=1 FLAGS=0,0,0 TARGET=present" in result.stdout
    assert f"Remove {tmp_path}/EagleEye-Vision-System before retrying" in result.stderr


def test_fresh_install_has_no_preconfigured_pipelines() -> None:
    """Verify the wizard is the only fresh-install pipeline setup path."""
    with PIPELINE_CONFIG.open("r", encoding="utf-8") as handle:
        assert handle.read().strip() == "{}"
