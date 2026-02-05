"""Tests for GitHub workflow file validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

# Use safe yaml loading
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


@pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
def test_frontend_build_workflow_valid_yaml() -> None:
    """Test that FrontendBuild.yaml is valid YAML."""
    workflow_path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "FrontendBuild.yaml"

    assert workflow_path.exists(), "FrontendBuild.yaml must exist"

    with open(workflow_path, "r", encoding="utf-8") as f:
        workflow = yaml.safe_load(f)

    assert isinstance(workflow, dict), "Workflow must be a dictionary"
    assert "name" in workflow, "Workflow must have a name"
    # Note: 'on' is parsed as True (boolean) by YAML, so check for True key
    assert "on" in workflow or True in workflow, "Workflow must have trigger events"
    assert "jobs" in workflow, "Workflow must have jobs"


@pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
def test_frontend_build_workflow_structure() -> None:
    """Test that FrontendBuild.yaml has correct structure."""
    workflow_path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "FrontendBuild.yaml"

    with open(workflow_path, "r", encoding="utf-8") as f:
        workflow = yaml.safe_load(f)

    # Check trigger events (YAML parses 'on' as True)
    triggers = workflow.get("on", workflow.get(True, {}))
    assert "push" in triggers or "pull_request" in triggers, \
        "Workflow should trigger on push or pull_request"

    # Check jobs structure
    assert "build" in workflow["jobs"], "Workflow should have a build job"
    build_job = workflow["jobs"]["build"]

    assert "runs-on" in build_job, "Build job must specify runs-on"
    assert "steps" in build_job, "Build job must have steps"
    assert isinstance(build_job["steps"], list), "Steps must be a list"
    assert len(build_job["steps"]) > 0, "Build job must have at least one step"


@pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
def test_frontend_build_workflow_node_setup() -> None:
    """Test that FrontendBuild.yaml sets up Node.js correctly."""
    workflow_path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "FrontendBuild.yaml"

    with open(workflow_path, "r", encoding="utf-8") as f:
        workflow = yaml.safe_load(f)

    steps = workflow["jobs"]["build"]["steps"]

    # Check for Node.js setup
    node_setup_found = False
    for step in steps:
        if "uses" in step and "setup-node" in step["uses"]:
            node_setup_found = True
            assert "with" in step, "Node setup should have 'with' configuration"
            assert "node-version" in step["with"], "Node setup should specify version"

    assert node_setup_found, "Workflow should include Node.js setup"


@pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
def test_frontend_build_workflow_build_step() -> None:
    """Test that FrontendBuild.yaml includes build step."""
    workflow_path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "FrontendBuild.yaml"

    with open(workflow_path, "r", encoding="utf-8") as f:
        workflow = yaml.safe_load(f)

    steps = workflow["jobs"]["build"]["steps"]

    # Check for build command
    build_step_found = False
    for step in steps:
        if "run" in step and "npm run build" in step["run"]:
            build_step_found = True

    assert build_step_found, "Workflow should include 'npm run build' step"


@pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
def test_frontend_build_workflow_working_directory() -> None:
    """Test that FrontendBuild.yaml uses correct working directory."""
    workflow_path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "FrontendBuild.yaml"

    with open(workflow_path, "r", encoding="utf-8") as f:
        workflow = yaml.safe_load(f)

    build_job = workflow["jobs"]["build"]

    # Check for working directory (could be in defaults or individual steps)
    if "defaults" in build_job and "run" in build_job["defaults"]:
        assert "working-directory" in build_job["defaults"]["run"], \
            "Should specify working directory"
        working_dir = build_job["defaults"]["run"]["working-directory"]
        assert "webui" in working_dir, "Working directory should include webui path"


@pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
def test_pytests_workflow_valid_yaml() -> None:
    """Test that Pytests.yaml is valid YAML."""
    workflow_path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "Pytests.yaml"

    assert workflow_path.exists(), "Pytests.yaml must exist"

    with open(workflow_path, "r", encoding="utf-8") as f:
        workflow = yaml.safe_load(f)

    assert isinstance(workflow, dict), "Workflow must be a dictionary"
    assert "name" in workflow, "Workflow must have a name"
    assert "on" in workflow, "Workflow must have trigger events"
    assert "jobs" in workflow, "Workflow must have jobs"


@pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
def test_pytests_workflow_structure() -> None:
    """Test that Pytests.yaml has correct structure."""
    workflow_path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "Pytests.yaml"

    with open(workflow_path, "r", encoding="utf-8") as f:
        workflow = yaml.safe_load(f)

    # Check jobs structure
    assert "test" in workflow["jobs"], "Workflow should have a test job"
    test_job = workflow["jobs"]["test"]

    assert "runs-on" in test_job, "Test job must specify runs-on"
    assert "steps" in test_job, "Test job must have steps"
    assert isinstance(test_job["steps"], list), "Steps must be a list"
    assert len(test_job["steps"]) > 0, "Test job must have at least one step"


@pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
def test_pytests_workflow_python_setup() -> None:
    """Test that Pytests.yaml sets up Python correctly."""
    workflow_path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "Pytests.yaml"

    with open(workflow_path, "r", encoding="utf-8") as f:
        workflow = yaml.safe_load(f)

    steps = workflow["jobs"]["test"]["steps"]

    # Check for Python setup
    python_setup_found = False
    for step in steps:
        if "uses" in step and "setup-python" in step["uses"]:
            python_setup_found = True
            assert "with" in step, "Python setup should have 'with' configuration"
            assert "python-version" in step["with"], "Python setup should specify version"

    assert python_setup_found, "Workflow should include Python setup"


@pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
def test_pytests_workflow_uv_installation() -> None:
    """Test that Pytests.yaml installs uv package manager."""
    workflow_path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "Pytests.yaml"

    with open(workflow_path, "r", encoding="utf-8") as f:
        workflow = yaml.safe_load(f)

    steps = workflow["jobs"]["test"]["steps"]

    # Check for uv installation
    uv_install_found = False
    for step in steps:
        if "run" in step and "uv" in step["run"]:
            uv_install_found = True
            break

    assert uv_install_found, "Workflow should install uv package manager"


@pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
def test_pytests_workflow_test_execution() -> None:
    """Test that Pytests.yaml runs pytest."""
    workflow_path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "Pytests.yaml"

    with open(workflow_path, "r", encoding="utf-8") as f:
        workflow = yaml.safe_load(f)

    steps = workflow["jobs"]["test"]["steps"]

    # Check for pytest execution
    pytest_found = False
    for step in steps:
        if "run" in step and "pytest" in step["run"]:
            pytest_found = True

    assert pytest_found, "Workflow should execute pytest"


@pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
def test_both_workflows_use_ubuntu_runner() -> None:
    """Test that both workflows use Ubuntu runner."""
    frontend_path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "FrontendBuild.yaml"
    pytest_path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "Pytests.yaml"

    for workflow_path in [frontend_path, pytest_path]:
        with open(workflow_path, "r", encoding="utf-8") as f:
            workflow = yaml.safe_load(f)

        jobs = workflow["jobs"]
        for job_name, job_config in jobs.items():
            runner = job_config["runs-on"]
            assert "ubuntu" in runner, f"Job {job_name} should use Ubuntu runner"


@pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
def test_workflows_checkout_repository() -> None:
    """Test that workflows checkout the repository."""
    frontend_path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "FrontendBuild.yaml"
    pytest_path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "Pytests.yaml"

    for workflow_path in [frontend_path, pytest_path]:
        with open(workflow_path, "r", encoding="utf-8") as f:
            workflow = yaml.safe_load(f)

        jobs = workflow["jobs"]
        for job_name, job_config in jobs.items():
            steps = job_config["steps"]

            checkout_found = False
            for step in steps:
                if "uses" in step and "checkout" in step["uses"]:
                    checkout_found = True
                    break

            assert checkout_found, f"Job {job_name} should checkout repository"


def test_workflow_files_exist() -> None:
    """Test that expected workflow files exist."""
    workflows_dir = Path(__file__).resolve().parents[1] / ".github" / "workflows"

    assert workflows_dir.exists(), ".github/workflows directory must exist"
    assert workflows_dir.is_dir(), ".github/workflows must be a directory"

    frontend_workflow = workflows_dir / "FrontendBuild.yaml"
    pytest_workflow = workflows_dir / "Pytests.yaml"

    assert frontend_workflow.exists(), "FrontendBuild.yaml must exist"
    assert pytest_workflow.exists(), "Pytests.yaml must exist"


def test_workflow_files_not_empty() -> None:
    """Test that workflow files are not empty."""
    frontend_path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "FrontendBuild.yaml"
    pytest_path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "Pytests.yaml"

    for workflow_path in [frontend_path, pytest_path]:
        with open(workflow_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert len(content) > 0, f"{workflow_path.name} cannot be empty"
        assert content.strip() != "", f"{workflow_path.name} cannot be only whitespace"