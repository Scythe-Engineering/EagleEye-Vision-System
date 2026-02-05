"""Tests specifically for files changed in this pull request.

This test suite validates the structure and content of configuration,
documentation, and guideline files that were modified.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class TestClaudeSettings:
    """Tests for .claude/settings.local.json changes."""

    def test_permissions_structure(self) -> None:
        """Test that permissions are properly structured."""
        config_path = Path(__file__).resolve().parents[1] / ".claude" / "settings.local.json"

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        assert "permissions" in config
        assert "allow" in config["permissions"]
        assert isinstance(config["permissions"]["allow"], list)

        # Verify all permissions are strings
        for perm in config["permissions"]["allow"]:
            assert isinstance(perm, str)
            assert len(perm) > 0

    def test_bash_permissions_format(self) -> None:
        """Test that Bash permissions follow expected format."""
        config_path = Path(__file__).resolve().parents[1] / ".claude" / "settings.local.json"

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        bash_perms = [p for p in config["permissions"]["allow"] if p.startswith("Bash(")]

        # All Bash permissions should have closing parenthesis
        for perm in bash_perms:
            assert perm.endswith(")"), f"Bash permission should end with ')': {perm}"
            assert ":" in perm, f"Bash permission should contain ':': {perm}"

    def test_mcp_server_configuration(self) -> None:
        """Test MCP server configuration is valid."""
        config_path = Path(__file__).resolve().parents[1] / ".claude" / "settings.local.json"

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        assert isinstance(config["enableAllProjectMcpServers"], bool)
        assert isinstance(config["enabledMcpjsonServers"], list)

        # ScytheContextEngine should be enabled
        assert "ScytheContextEngine" in config["enabledMcpjsonServers"]

    def test_allowed_commands_are_safe(self) -> None:
        """Test that allowed bash commands are safe and expected."""
        config_path = Path(__file__).resolve().parents[1] / ".claude" / "settings.local.json"

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        allowed_commands = [
            p.split("(")[1].split(":")[0]
            for p in config["permissions"]["allow"]
            if p.startswith("Bash(")
        ]

        # Check for expected safe commands
        expected_safe_commands = ["find", "wc", "grep", "ls", "tree"]
        for cmd in expected_safe_commands:
            assert cmd in allowed_commands, f"Expected safe command '{cmd}' should be allowed"

        # Build commands should be allowed
        build_perms = [p for p in config["permissions"]["allow"] if "npm run build" in p]
        assert len(build_perms) > 0, "npm run build should be allowed"


class TestGitignore:
    """Tests for .gitignore changes."""

    def test_python_patterns_present(self) -> None:
        """Test that Python-specific patterns are present."""
        gitignore_path = Path(__file__).resolve().parents[1] / ".gitignore"

        with open(gitignore_path, "r", encoding="utf-8") as f:
            content = f.read()

        python_patterns = [
            "__pycache__",
            "*.py[cod]",
            ".Python",
            "*.egg-info",
            ".pytest_cache",
        ]

        for pattern in python_patterns:
            assert pattern in content, f"Python pattern '{pattern}' should be in .gitignore"

    def test_ide_patterns_present(self) -> None:
        """Test that IDE-specific patterns are present."""
        gitignore_path = Path(__file__).resolve().parents[1] / ".gitignore"

        with open(gitignore_path, "r", encoding="utf-8") as f:
            content = f.read()

        ide_patterns = [".idea", ".vscode", "*.iml"]

        found_count = sum(1 for pattern in ide_patterns if pattern in content)
        assert found_count >= 1, "At least one IDE pattern should be in .gitignore"

    def test_project_specific_ignores(self) -> None:
        """Test that project-specific ignores are present."""
        gitignore_path = Path(__file__).resolve().parents[1] / ".gitignore"

        with open(gitignore_path, "r", encoding="utf-8") as f:
            content = f.read()

        project_patterns = [
            "node_modules/",
            "*.pth",
            "uv.lock",
            ".build_cache.json",
            "src/webui/static/",
        ]

        for pattern in project_patterns:
            assert pattern in content or pattern.replace("/", "") in content, \
                f"Project pattern '{pattern}' should be in .gitignore"

    def test_model_files_ignored(self) -> None:
        """Test that model and training files are ignored."""
        gitignore_path = Path(__file__).resolve().parents[1] / ".gitignore"

        with open(gitignore_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Should ignore various model file types
        model_patterns = ["*.pth", "*.pt", "model.pth"]

        found = any(pattern in content for pattern in model_patterns)
        assert found, "Model files should be ignored"


@pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
class TestWorkflows:
    """Tests for GitHub workflow changes."""

    def test_frontend_workflow_structure(self) -> None:
        """Test FrontendBuild.yaml structure is valid."""
        workflow_path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "FrontendBuild.yaml"

        with open(workflow_path, "r", encoding="utf-8") as f:
            workflow = yaml.safe_load(f)

        # Check basic structure
        assert workflow["name"] == "Verify WebUI Build"
        assert "on" in workflow
        assert "jobs" in workflow
        assert "build" in workflow["jobs"]

    def test_frontend_workflow_steps(self) -> None:
        """Test FrontendBuild.yaml has correct steps."""
        workflow_path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "FrontendBuild.yaml"

        with open(workflow_path, "r", encoding="utf-8") as f:
            workflow = yaml.safe_load(f)

        steps = workflow["jobs"]["build"]["steps"]
        step_names = [step.get("name", "") for step in steps]

        # Expected steps
        expected_steps = [
            "Set up Node.js",
            "Install frontend dependencies",
            "Run Vite production build",
        ]

        for expected in expected_steps:
            assert any(expected in name for name in step_names), \
                f"Expected step '{expected}' not found in workflow"

    def test_frontend_workflow_working_directory(self) -> None:
        """Test FrontendBuild.yaml uses correct working directory."""
        workflow_path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "FrontendBuild.yaml"

        with open(workflow_path, "r", encoding="utf-8") as f:
            workflow = yaml.safe_load(f)

        build_job = workflow["jobs"]["build"]

        # Should have working directory set to src/webui
        if "defaults" in build_job:
            working_dir = build_job["defaults"]["run"]["working-directory"]
            assert "webui" in working_dir

    def test_pytests_workflow_structure(self) -> None:
        """Test Pytests.yaml structure is valid."""
        workflow_path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "Pytests.yaml"

        with open(workflow_path, "r", encoding="utf-8") as f:
            workflow = yaml.safe_load(f)

        # Check basic structure
        assert workflow["name"] == "Run EagleEye Tests"
        assert "on" in workflow
        assert "jobs" in workflow
        assert "test" in workflow["jobs"]

    def test_pytests_workflow_uses_uv(self) -> None:
        """Test Pytests.yaml uses uv package manager."""
        workflow_path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "Pytests.yaml"

        with open(workflow_path, "r", encoding="utf-8") as f:
            workflow = yaml.safe_load(f)

        steps = workflow["jobs"]["test"]["steps"]

        # Should install uv and use it
        uv_related_steps = [
            step for step in steps
            if "run" in step and "uv" in step["run"]
        ]

        assert len(uv_related_steps) > 0, "Workflow should use uv package manager"

    def test_both_workflows_trigger_correctly(self) -> None:
        """Test both workflows trigger on correct events."""
        frontend_path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "FrontendBuild.yaml"
        pytest_path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "Pytests.yaml"

        for workflow_path in [frontend_path, pytest_path]:
            with open(workflow_path, "r", encoding="utf-8") as f:
                workflow = yaml.safe_load(f)

            triggers = workflow.get("on", workflow.get(True, {}))

            # Should trigger on both push and pull_request
            assert "push" in triggers
            assert "pull_request" in triggers

            # Push should trigger on main branch
            if isinstance(triggers["push"], dict):
                assert "main" in triggers["push"]["branches"]


class TestCopilotInstructions:
    """Tests for copilot-instructions.md changes."""

    def test_contains_formatting_rules(self) -> None:
        """Test that instructions contain formatting rules."""
        instructions_path = Path(__file__).resolve().parents[1] / ".github" / "copilot-instructions.md"

        with open(instructions_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "black" in content.lower()
        assert "docstring" in content.lower()
        assert "type hint" in content.lower()

    def test_contains_naming_conventions(self) -> None:
        """Test that instructions contain naming conventions."""
        instructions_path = Path(__file__).resolve().parents[1] / ".github" / "copilot-instructions.md"

        with open(instructions_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Should mention descriptive variables
        assert "variable" in content.lower() and "descriptive" in content.lower()

    def test_contains_comment_policy(self) -> None:
        """Test that instructions contain comment policy."""
        instructions_path = Path(__file__).resolve().parents[1] / ".github" / "copilot-instructions.md"

        with open(instructions_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Should have comments section
        assert "comment" in content.lower()


class TestRooConfiguration:
    """Tests for .roo directory configuration files."""

    @pytest.mark.parametrize("mode", ["architect", "ask", "code", "debug"])
    def test_mode_rules_are_complete(self, mode: str) -> None:
        """Test that mode-specific rules contain necessary information."""
        rules_path = Path(__file__).resolve().parents[1] / ".roo" / f"rules-{mode}" / "AGENTS.md"

        with open(rules_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Should have header
        assert content.startswith("#")

        # Should contain mode-specific information
        assert len(content) > 100, f"Rules for {mode} mode should have substantial content"

    def test_codebase_context_mentions_scythe(self) -> None:
        """Test that codebase context mentions ScytheContextEngine."""
        context_path = Path(__file__).resolve().parents[1] / ".roo" / "rules" / "01-codebase-context.md"

        with open(context_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "ScytheContextEngine" in content or "context engine" in content.lower()

    @pytest.mark.parametrize("mode", ["architect", "ask", "code"])
    def test_system_prompts_exist_and_valid(self, mode: str) -> None:
        """Test that system prompts exist and are valid."""
        prompt_path = Path(__file__).resolve().parents[1] / ".roo" / f"system-prompt-{mode}"

        assert prompt_path.exists(), f"system-prompt-{mode} must exist"

        with open(prompt_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert len(content) > 500, f"System prompt for {mode} should be substantial"

    @pytest.mark.parametrize("skill_name", [
        "pipeline-architecture",
        "system-architecture-understanding",
        "pipeline-operation-creation",
        "webui-development",
        "pipeline-debugging",
    ])
    def test_skills_have_proper_structure(self, skill_name: str) -> None:
        """Test that skill files have proper structure."""
        # Find skill in any skills directory
        roo_dir = Path(__file__).resolve().parents[1] / ".roo"
        skill_files = list(roo_dir.glob(f"skills-*/{skill_name}/SKILL.md"))

        if not skill_files:
            pytest.skip(f"Skill {skill_name} not found")

        skill_path = skill_files[0]

        with open(skill_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Should have YAML frontmatter
        assert content.startswith("---"), f"{skill_name} should have YAML frontmatter"

        # Should have substantial content
        assert len(content) > 200, f"{skill_name} should have substantial content"


class TestDocumentation:
    """Tests for documentation file changes."""

    def test_agents_md_completeness(self) -> None:
        """Test that AGENTS.md contains complete information."""
        agents_path = Path(__file__).resolve().parents[1] / "AGENTS.md"

        with open(agents_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Should contain key sections
        required_sections = [
            "Build",
            "operation",
            "injection",
        ]

        content_lower = content.lower()
        for section in required_sections:
            assert section.lower() in content_lower, \
                f"AGENTS.md should contain information about {section}"

    def test_claude_md_completeness(self) -> None:
        """Test that CLAUDE.md contains complete information."""
        claude_path = Path(__file__).resolve().parents[1] / "CLAUDE.md"

        with open(claude_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Should contain key sections
        required_sections = [
            "Project Overview",
            "Build",
            "Architecture",
            "Pipeline",
        ]

        for section in required_sections:
            assert section in content, f"CLAUDE.md should contain {section} section"

    def test_pipeline_overview_updated(self) -> None:
        """Test that PipelineOverview.md is complete."""
        overview_path = Path(__file__).resolve().parents[1] / "docs" / "md_docs" / "pipeline_docs" / \
                        "PipelineOverview.md"

        with open(overview_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Should describe pipeline concepts
        assert "pipeline" in content.lower()
        assert "operation" in content.lower()

    def test_implement_pipeline_operation_has_examples(self) -> None:
        """Test that ImplementPipelineOperation.md has code examples."""
        impl_path = Path(__file__).resolve().parents[1] / "docs" / "md_docs" / "pipeline_docs" / \
                    "ImplementPipelineOperation.md"

        with open(impl_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Should have code examples
        assert "```python" in content or "```json" in content

        # Should mention injection pattern
        assert "injection" in content.lower() or "inject" in content.lower()

    def test_operation_docs_contain_examples(self) -> None:
        """Test that operation documentation contains usage examples."""
        color_threshold_path = Path(__file__).resolve().parents[1] / "docs" / "md_docs" / "pipeline_docs" / \
                               "main_operations" / "ColorThresholdDetection.md"

        with open(color_threshold_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Should have configuration examples
        assert "```" in content, "Should contain code blocks"
        assert "color" in content.lower()

    def test_device_input_doc_describes_integration(self) -> None:
        """Test that DeviceInput.md describes camera integration."""
        device_input_path = Path(__file__).resolve().parents[1] / "docs" / "md_docs" / "pipeline_docs" / \
                            "secondary_operations" / "DeviceInput.md"

        with open(device_input_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Should describe camera integration
        assert "camera" in content.lower()
        assert "frame" in content.lower()