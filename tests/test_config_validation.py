"""Tests for configuration file validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


def test_claude_settings_local_json_valid() -> None:
    """Test that .claude/settings.local.json is valid JSON."""
    config_path = Path(__file__).resolve().parents[1] / ".claude" / "settings.local.json"

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    assert isinstance(config, dict), "Config must be a dictionary"
    assert "permissions" in config, "Config must have 'permissions' key"
    assert "allow" in config["permissions"], "Permissions must have 'allow' key"
    assert isinstance(config["permissions"]["allow"], list), "Allow list must be a list"


def test_claude_settings_permissions_format() -> None:
    """Test that Claude settings permissions are properly formatted."""
    config_path = Path(__file__).resolve().parents[1] / ".claude" / "settings.local.json"

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    permissions = config["permissions"]["allow"]

    for permission in permissions:
        assert isinstance(permission, str), f"Permission must be string: {permission}"
        assert len(permission) > 0, "Permission cannot be empty string"


def test_claude_settings_mcp_servers() -> None:
    """Test that MCP server configuration is valid."""
    config_path = Path(__file__).resolve().parents[1] / ".claude" / "settings.local.json"

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    assert "enableAllProjectMcpServers" in config, "Must have enableAllProjectMcpServers"
    assert isinstance(config["enableAllProjectMcpServers"], bool), "enableAllProjectMcpServers must be boolean"

    if "enabledMcpjsonServers" in config:
        assert isinstance(config["enabledMcpjsonServers"], list), "enabledMcpjsonServers must be a list"
        for server in config["enabledMcpjsonServers"]:
            assert isinstance(server, str), f"MCP server name must be string: {server}"


def test_gitignore_file_exists() -> None:
    """Test that .gitignore file exists and is readable."""
    gitignore_path = Path(__file__).resolve().parents[1] / ".gitignore"

    assert gitignore_path.exists(), ".gitignore file must exist"
    assert gitignore_path.is_file(), ".gitignore must be a file"

    with open(gitignore_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert len(content) > 0, ".gitignore cannot be empty"


def test_gitignore_contains_common_patterns() -> None:
    """Test that .gitignore contains expected patterns."""
    gitignore_path = Path(__file__).resolve().parents[1] / ".gitignore"

    with open(gitignore_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Check for common Python patterns
    assert "__pycache__" in content, ".gitignore should ignore __pycache__"
    assert "*.pyc" in content or "*.py[cod]" in content, ".gitignore should ignore compiled Python files"

    # Check for common IDE patterns
    assert ".idea" in content, ".gitignore should ignore .idea directory"

    # Check for environment patterns
    assert ".venv" in content or "venv/" in content, ".gitignore should ignore virtual environments"


def test_gitignore_project_specific_patterns() -> None:
    """Test that .gitignore contains project-specific patterns."""
    gitignore_path = Path(__file__).resolve().parents[1] / ".gitignore"

    with open(gitignore_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Check for project-specific patterns
    assert "node_modules/" in content, ".gitignore should ignore node_modules"
    assert "*.pth" in content or "*model.pth" in content, ".gitignore should ignore model files"
    assert "uv.lock" in content, ".gitignore should ignore uv.lock"


def test_no_duplicate_patterns_in_gitignore() -> None:
    """Test that .gitignore doesn't contain exact duplicate patterns."""
    gitignore_path = Path(__file__).resolve().parents[1] / ".gitignore"

    with open(gitignore_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    seen: set[str] = set()
    duplicates: list[str] = []

    for line in lines:
        if line in seen:
            duplicates.append(line)
        seen.add(line)

    assert len(duplicates) == 0, f"Found duplicate patterns in .gitignore: {duplicates}"


def test_copilot_instructions_valid_format() -> None:
    """Test that copilot instructions file is properly formatted."""
    instructions_path = Path(__file__).resolve().parents[1] / ".github" / "copilot-instructions.md"

    assert instructions_path.exists(), "copilot-instructions.md must exist"

    with open(instructions_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert len(content) > 0, "copilot-instructions.md cannot be empty"

    # Check for numbered rules (common pattern in instruction files)
    assert any(line.strip().startswith(("1.", "2.", "3.")) for line in content.split("\n")), \
        "Instructions should contain numbered rules"


def test_copilot_instructions_contains_style_rules() -> None:
    """Test that copilot instructions contain expected style rules."""
    instructions_path = Path(__file__).resolve().parents[1] / ".github" / "copilot-instructions.md"

    with open(instructions_path, "r", encoding="utf-8") as f:
        content = f.read().lower()

    # Check for expected content
    assert "black" in content, "Instructions should mention black formatting"
    assert "docstring" in content, "Instructions should mention docstrings"
    assert "type hint" in content, "Instructions should mention type hints"


def test_general_conf_json_if_exists() -> None:
    """Test general_conf.json if it exists (it's in .gitignore but may be present)."""
    config_path = Path(__file__).resolve().parents[1] / "general_conf.json"

    if not config_path.exists():
        pytest.skip("general_conf.json not present (expected, it's in .gitignore)")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    assert isinstance(config, dict), "general_conf.json must be a dictionary"


def test_package_json_exists() -> None:
    """Test that package.json exists for frontend dependencies."""
    package_json_path = Path(__file__).resolve().parents[1] / "src" / "webui" / "package.json"

    if not package_json_path.exists():
        # Try alternate location
        package_json_path = Path(__file__).resolve().parents[1] / "package.json"

    if package_json_path.exists():
        with open(package_json_path, "r", encoding="utf-8") as f:
            package_data = json.load(f)

        assert isinstance(package_data, dict), "package.json must be a dictionary"
        assert "name" in package_data or "scripts" in package_data, \
            "package.json should contain name or scripts"


def test_pyproject_toml_exists() -> None:
    """Test that pyproject.toml exists for Python project configuration."""
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"

    assert pyproject_path.exists(), "pyproject.toml must exist"
    assert pyproject_path.is_file(), "pyproject.toml must be a file"

    with open(pyproject_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert len(content) > 0, "pyproject.toml cannot be empty"
    assert "[project]" in content or "[tool" in content, \
        "pyproject.toml should contain project or tool sections"