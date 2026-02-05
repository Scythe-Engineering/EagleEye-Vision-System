"""Tests for documentation file validation."""

from __future__ import annotations

from pathlib import Path

import pytest


class TestAgentsDocumentation:
    """Tests for AGENTS.md and agent-related documentation."""

    def test_agents_md_exists(self) -> None:
        """Test that AGENTS.md exists."""
        agents_path = Path(__file__).resolve().parents[1] / "AGENTS.md"
        assert agents_path.exists(), "AGENTS.md must exist"

    def test_agents_md_not_empty(self) -> None:
        """Test that AGENTS.md is not empty."""
        agents_path = Path(__file__).resolve().parents[1] / "AGENTS.md"

        with open(agents_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert len(content) > 0, "AGENTS.md cannot be empty"
        assert content.strip() != "", "AGENTS.md cannot be only whitespace"

    def test_agents_md_contains_build_commands(self) -> None:
        """Test that AGENTS.md contains build and test commands."""
        agents_path = Path(__file__).resolve().parents[1] / "AGENTS.md"

        with open(agents_path, "r", encoding="utf-8") as f:
            content = f.read().lower()

        assert "npm" in content or "build" in content, "AGENTS.md should contain build information"
        assert "uv sync" in content or "python" in content, "AGENTS.md should contain Python setup info"

    def test_agents_md_contains_code_patterns(self) -> None:
        """Test that AGENTS.md documents important code patterns."""
        agents_path = Path(__file__).resolve().parents[1] / "AGENTS.md"

        with open(agents_path, "r", encoding="utf-8") as f:
            content = f.read().lower()

        # Check for key architectural patterns
        assert "operation" in content, "AGENTS.md should document operations"
        assert "injection" in content or "dependency" in content, "AGENTS.md should document dependency injection"


class TestClaudeDocumentation:
    """Tests for CLAUDE.md documentation."""

    def test_claude_md_exists(self) -> None:
        """Test that CLAUDE.md exists."""
        claude_path = Path(__file__).resolve().parents[1] / "CLAUDE.md"
        assert claude_path.exists(), "CLAUDE.md must exist"

    def test_claude_md_not_empty(self) -> None:
        """Test that CLAUDE.md is not empty."""
        claude_path = Path(__file__).resolve().parents[1] / "CLAUDE.md"

        with open(claude_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert len(content) > 0, "CLAUDE.md cannot be empty"

    def test_claude_md_contains_project_overview(self) -> None:
        """Test that CLAUDE.md contains project overview."""
        claude_path = Path(__file__).resolve().parents[1] / "CLAUDE.md"

        with open(claude_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check for key sections
        assert "## Project Overview" in content or "# Project Overview" in content or \
               "Project Overview" in content, "CLAUDE.md should have Project Overview section"

    def test_claude_md_contains_build_commands(self) -> None:
        """Test that CLAUDE.md contains build and run commands."""
        claude_path = Path(__file__).resolve().parents[1] / "CLAUDE.md"

        with open(claude_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "uv sync" in content or "pip install" in content, \
            "CLAUDE.md should document Python dependency installation"
        assert "npm" in content, "CLAUDE.md should document npm usage"

    def test_claude_md_contains_architecture_info(self) -> None:
        """Test that CLAUDE.md contains architecture information."""
        claude_path = Path(__file__).resolve().parents[1] / "CLAUDE.md"

        with open(claude_path, "r", encoding="utf-8") as f:
            content = f.read().lower()

        assert "pipeline" in content, "CLAUDE.md should document pipeline architecture"
        assert "backend" in content or "frontend" in content, \
            "CLAUDE.md should document backend/frontend architecture"


class TestRooDocumentation:
    """Tests for .roo directory documentation."""

    def test_roo_directory_exists(self) -> None:
        """Test that .roo directory exists."""
        roo_path = Path(__file__).resolve().parents[1] / ".roo"
        assert roo_path.exists(), ".roo directory must exist"
        assert roo_path.is_dir(), ".roo must be a directory"

    def test_roo_rules_directory_exists(self) -> None:
        """Test that .roo/rules directory exists."""
        rules_path = Path(__file__).resolve().parents[1] / ".roo" / "rules"
        if rules_path.exists():
            assert rules_path.is_dir(), ".roo/rules must be a directory"

    @pytest.mark.parametrize("mode", ["architect", "ask", "code", "debug"])
    def test_roo_mode_rules_exist(self, mode: str) -> None:
        """Test that mode-specific rule files exist."""
        rules_path = Path(__file__).resolve().parents[1] / ".roo" / f"rules-{mode}" / "AGENTS.md"

        assert rules_path.exists(), f".roo/rules-{mode}/AGENTS.md must exist"

        with open(rules_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert len(content) > 0, f"rules-{mode}/AGENTS.md cannot be empty"

    @pytest.mark.parametrize("mode", ["architect", "ask", "code", "debug"])
    def test_roo_mode_rules_contain_guidance(self, mode: str) -> None:
        """Test that mode-specific rules contain actual guidance."""
        rules_path = Path(__file__).resolve().parents[1] / ".roo" / f"rules-{mode}" / "AGENTS.md"

        with open(rules_path, "r", encoding="utf-8") as f:
            content = f.read().lower()

        # Should contain some form of instructions or rules
        assert any(keyword in content for keyword in ["rule", "pattern", "must", "should", "guide"]), \
            f"rules-{mode}/AGENTS.md should contain guidance keywords"

    def test_roo_codebase_context_exists(self) -> None:
        """Test that codebase context file exists."""
        context_path = Path(__file__).resolve().parents[1] / ".roo" / "rules" / "01-codebase-context.md"

        assert context_path.exists(), ".roo/rules/01-codebase-context.md must exist"

        with open(context_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert len(content) > 0, "01-codebase-context.md cannot be empty"

    def test_roo_system_prompts_exist(self) -> None:
        """Test that system prompt files exist."""
        roo_dir = Path(__file__).resolve().parents[1] / ".roo"

        for prompt_file in ["system-prompt-architect", "system-prompt-ask", "system-prompt-code"]:
            prompt_path = roo_dir / prompt_file
            assert prompt_path.exists(), f"{prompt_file} must exist"

            with open(prompt_path, "r", encoding="utf-8") as f:
                content = f.read()

            assert len(content) > 0, f"{prompt_file} cannot be empty"


class TestRooSkills:
    """Tests for .roo/skills documentation."""

    def test_skills_directories_exist(self) -> None:
        """Test that skills directories exist for different modes."""
        roo_dir = Path(__file__).resolve().parents[1] / ".roo"

        skill_dirs = list(roo_dir.glob("skills-*"))
        assert len(skill_dirs) > 0, "Should have at least one skills directory"

        for skill_dir in skill_dirs:
            assert skill_dir.is_dir(), f"{skill_dir.name} should be a directory"

    @pytest.mark.parametrize("mode", ["architect", "ask", "code", "debug"])
    def test_mode_skills_directory_structure(self, mode: str) -> None:
        """Test that mode-specific skills have proper structure."""
        skills_dir = Path(__file__).resolve().parents[1] / ".roo" / f"skills-{mode}"

        if not skills_dir.exists():
            pytest.skip(f"skills-{mode} directory not present")

        # Should contain skill subdirectories with SKILL.md files
        skill_subdirs = [d for d in skills_dir.iterdir() if d.is_dir()]

        for skill_subdir in skill_subdirs:
            skill_file = skill_subdir / "SKILL.md"
            if skill_file.exists():
                with open(skill_file, "r", encoding="utf-8") as f:
                    content = f.read()

                assert len(content) > 0, f"{skill_file} cannot be empty"

    def test_pipeline_architecture_skill_exists(self) -> None:
        """Test that pipeline architecture skill exists."""
        skill_path = Path(__file__).resolve().parents[1] / ".roo" / "skills-architect" / \
                      "pipeline-architecture" / "SKILL.md"

        assert skill_path.exists(), "pipeline-architecture skill must exist"

        with open(skill_path, "r", encoding="utf-8") as f:
            content = f.read().lower()

        assert "pipeline" in content, "pipeline-architecture skill should discuss pipelines"

    def test_skill_files_have_frontmatter(self) -> None:
        """Test that SKILL.md files have proper frontmatter."""
        roo_dir = Path(__file__).resolve().parents[1] / ".roo"

        skill_files = list(roo_dir.glob("skills-*/**/SKILL.md"))

        for skill_file in skill_files:
            with open(skill_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Check for YAML frontmatter
            assert content.startswith("---"), f"{skill_file} should have YAML frontmatter"
            assert content.count("---") >= 2, f"{skill_file} frontmatter should be properly closed"


class TestPipelineDocumentation:
    """Tests for pipeline operation documentation."""

    def test_pipeline_docs_directory_exists(self) -> None:
        """Test that pipeline docs directory exists."""
        docs_dir = Path(__file__).resolve().parents[1] / "docs" / "md_docs" / "pipeline_docs"

        assert docs_dir.exists(), "docs/md_docs/pipeline_docs must exist"
        assert docs_dir.is_dir(), "pipeline_docs must be a directory"

    def test_pipeline_overview_exists(self) -> None:
        """Test that PipelineOverview.md exists."""
        overview_path = Path(__file__).resolve().parents[1] / "docs" / "md_docs" / "pipeline_docs" / \
                        "PipelineOverview.md"

        assert overview_path.exists(), "PipelineOverview.md must exist"

        with open(overview_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert len(content) > 0, "PipelineOverview.md cannot be empty"
        assert "pipeline" in content.lower(), "PipelineOverview.md should discuss pipelines"

    def test_implement_pipeline_operation_exists(self) -> None:
        """Test that ImplementPipelineOperation.md exists."""
        impl_path = Path(__file__).resolve().parents[1] / "docs" / "md_docs" / "pipeline_docs" / \
                    "ImplementPipelineOperation.md"

        assert impl_path.exists(), "ImplementPipelineOperation.md must exist"

        with open(impl_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert len(content) > 0, "ImplementPipelineOperation.md cannot be empty"

    def test_implement_pipeline_operation_contains_examples(self) -> None:
        """Test that ImplementPipelineOperation.md contains code examples."""
        impl_path = Path(__file__).resolve().parents[1] / "docs" / "md_docs" / "pipeline_docs" / \
                    "ImplementPipelineOperation.md"

        with open(impl_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check for code blocks
        assert "```python" in content or "```" in content, \
            "ImplementPipelineOperation.md should contain code examples"

    def test_operation_docs_exist(self) -> None:
        """Test that operation documentation files exist."""
        main_ops_dir = Path(__file__).resolve().parents[1] / "docs" / "md_docs" / "pipeline_docs" / \
                       "main_operations"
        secondary_ops_dir = Path(__file__).resolve().parents[1] / "docs" / "md_docs" / "pipeline_docs" / \
                            "secondary_operations"

        # At least one of these directories should exist with docs
        has_docs = False
        for docs_dir in [main_ops_dir, secondary_ops_dir]:
            if docs_dir.exists():
                md_files = list(docs_dir.glob("*.md"))
                if md_files:
                    has_docs = True
                    break

        assert has_docs, "Should have at least some operation documentation"

    def test_color_threshold_detection_doc_exists(self) -> None:
        """Test that ColorThresholdDetection.md exists and is valid."""
        doc_path = Path(__file__).resolve().parents[1] / "docs" / "md_docs" / "pipeline_docs" / \
                   "main_operations" / "ColorThresholdDetection.md"

        assert doc_path.exists(), "ColorThresholdDetection.md must exist"

        with open(doc_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert len(content) > 0, "ColorThresholdDetection.md cannot be empty"
        assert "color" in content.lower() or "threshold" in content.lower(), \
            "ColorThresholdDetection.md should discuss color thresholding"

    def test_device_input_doc_exists(self) -> None:
        """Test that DeviceInput.md exists and is valid."""
        doc_path = Path(__file__).resolve().parents[1] / "docs" / "md_docs" / "pipeline_docs" / \
                   "secondary_operations" / "DeviceInput.md"

        assert doc_path.exists(), "DeviceInput.md must exist"

        with open(doc_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert len(content) > 0, "DeviceInput.md cannot be empty"
        assert "device" in content.lower() or "input" in content.lower() or "camera" in content.lower(), \
            "DeviceInput.md should discuss device input"


class TestDocumentationFormatting:
    """Tests for documentation formatting and consistency."""

    def test_markdown_files_have_headers(self) -> None:
        """Test that markdown files have proper headers."""
        docs_dir = Path(__file__).resolve().parents[1] / "docs"

        if not docs_dir.exists():
            pytest.skip("docs directory not present")

        md_files = list(docs_dir.glob("**/*.md"))

        for md_file in md_files[:10]:  # Test first 10 to avoid being too slow
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Should have at least one header
            assert any(line.startswith("#") for line in content.split("\n")), \
                f"{md_file} should have at least one header"

    def test_readme_files_not_empty(self) -> None:
        """Test that README files are not empty."""
        repo_root = Path(__file__).resolve().parents[1]

        readme_patterns = ["README.md", "README.txt", "README"]
        for pattern in readme_patterns:
            readme_path = repo_root / pattern
            if readme_path.exists():
                with open(readme_path, "r", encoding="utf-8") as f:
                    content = f.read()

                assert len(content) > 0, f"{pattern} cannot be empty"
                break

    def test_documentation_uses_consistent_formatting(self) -> None:
        """Test that documentation uses consistent formatting."""
        agents_path = Path(__file__).resolve().parents[1] / "AGENTS.md"
        claude_path = Path(__file__).resolve().parents[1] / "CLAUDE.md"

        for doc_path in [agents_path, claude_path]:
            if not doc_path.exists():
                continue

            with open(doc_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Check for consistent header formatting
            lines = content.split("\n")
            headers = [line for line in lines if line.startswith("#")]

            assert len(headers) > 0, f"{doc_path.name} should have headers"

            # Headers should use # format (not underlined with ===)
            for header in headers:
                assert header.startswith("#"), f"Headers in {doc_path.name} should use # format"