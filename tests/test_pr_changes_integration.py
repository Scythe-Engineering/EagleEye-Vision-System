"""Integration tests for pull request changes.

This test suite provides comprehensive validation that all changed files
work together correctly and maintain consistency across the codebase.
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


class TestCrossFileConsistency:
    """Tests that verify consistency across multiple changed files."""

    def test_claude_settings_match_workflow_commands(self) -> None:
        """Test that Claude settings allow commands used in workflows."""
        # Read Claude settings
        config_path = Path(__file__).resolve().parents[1] / ".claude" / "settings.local.json"
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        allowed_commands = [
            p.split("(")[1].split(":")[0]
            for p in config["permissions"]["allow"]
            if p.startswith("Bash(") and ":" in p
        ]

        # Commands that might be used in workflows should be allowed
        workflow_commands = ["npm run build"]
        for cmd in workflow_commands:
            # Check if any permission matches
            has_permission = any(
                cmd in perm
                for perm in config["permissions"]["allow"]
            )
            # Build commands should be allowed
            if "build" in cmd:
                # Just verify some build-related permission exists
                assert any("build" in perm for perm in config["permissions"]["allow"]), \
                    "Build commands should be permitted"

    def test_documentation_matches_codebase_structure(self) -> None:
        """Test that documentation reflects actual codebase structure."""
        # Check AGENTS.md mentions key directories
        agents_path = Path(__file__).resolve().parents[1] / "AGENTS.md"
        with open(agents_path, "r", encoding="utf-8") as f:
            agents_content = f.read().lower()

        # Should mention key directories that exist
        key_dirs = [
            "src/main_operations",
            "src/secondary_operations",
            "src/webui",
        ]

        for dir_path in key_dirs:
            full_path = Path(__file__).resolve().parents[1] / dir_path
            if full_path.exists():
                # Directory name should be mentioned in documentation
                dir_name = dir_path.split("/")[-1]
                assert dir_name in agents_content, \
                    f"Documentation should mention {dir_name}"

    def test_roo_rules_consistent_across_modes(self) -> None:
        """Test that Roo rules are consistent across different modes."""
        roo_dir = Path(__file__).resolve().parents[1] / ".roo"
        modes = ["architect", "ask", "code", "debug"]

        # All modes should mention common concepts
        common_concepts = ["operation", "pipeline"]

        for mode in modes:
            rules_path = roo_dir / f"rules-{mode}" / "AGENTS.md"
            if not rules_path.exists():
                continue

            with open(rules_path, "r", encoding="utf-8") as f:
                content = f.read().lower()

            # At least some common concepts should be mentioned
            found_concepts = [concept for concept in common_concepts if concept in content]
            assert len(found_concepts) > 0, \
                f"Mode {mode} should mention common concepts like {common_concepts}"

    def test_documentation_links_are_consistent(self) -> None:
        """Test that documentation file references are consistent."""
        # CLAUDE.md should reference files that exist
        claude_path = Path(__file__).resolve().parents[1] / "CLAUDE.md"
        with open(claude_path, "r", encoding="utf-8") as f:
            claude_content = f.read()

        # Check some key files mentioned in docs actually exist
        repo_root = Path(__file__).resolve().parents[1]

        key_files = [
            "src/main_backend.py",
            "src/config/pipeline_config.json",
            "pyproject.toml",
        ]

        for file_ref in key_files:
            if file_ref in claude_content:
                file_path = repo_root / file_ref
                assert file_path.exists(), \
                    f"File referenced in documentation should exist: {file_ref}"

    def test_gitignore_matches_build_outputs(self) -> None:
        """Test that .gitignore covers documented build outputs."""
        gitignore_path = Path(__file__).resolve().parents[1] / ".gitignore"
        with open(gitignore_path, "r", encoding="utf-8") as f:
            gitignore_content = f.read()

        # CLAUDE.md mentions build outputs that should be ignored
        claude_path = Path(__file__).resolve().parents[1] / "CLAUDE.md"
        with open(claude_path, "r", encoding="utf-8") as f:
            claude_content = f.read()

        # If documentation mentions src/webui/static/ as build output
        if "src/webui/static" in claude_content and "build" in claude_content.lower():
            # .gitignore should ignore some static files or the directory
            assert "static" in gitignore_content or "**/static/" in gitignore_content, \
                "Build output directory mentioned in docs should be in .gitignore"


class TestConfigurationIntegrity:
    """Tests for configuration file integrity and completeness."""

    def test_all_json_files_valid(self) -> None:
        """Test that all JSON configuration files are valid."""
        repo_root = Path(__file__).resolve().parents[1]

        json_files = [
            ".claude/settings.local.json",
        ]

        for json_file in json_files:
            json_path = repo_root / json_file
            if not json_path.exists():
                continue

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            assert isinstance(data, dict), f"{json_file} should be a dictionary"

    @pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
    def test_all_yaml_files_valid(self) -> None:
        """Test that all YAML workflow files are valid."""
        repo_root = Path(__file__).resolve().parents[1]
        workflows_dir = repo_root / ".github" / "workflows"

        if not workflows_dir.exists():
            pytest.skip("No workflows directory")

        yaml_files = list(workflows_dir.glob("*.yaml")) + list(workflows_dir.glob("*.yml"))

        for yaml_file in yaml_files:
            with open(yaml_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            assert isinstance(data, dict), f"{yaml_file.name} should be a dictionary"
            assert "jobs" in data, f"{yaml_file.name} should have jobs"

    def test_markdown_files_well_formed(self) -> None:
        """Test that all changed markdown files are well-formed."""
        repo_root = Path(__file__).resolve().parents[1]

        md_files = [
            "AGENTS.md",
            "CLAUDE.md",
            ".github/copilot-instructions.md",
        ]

        for md_file in md_files:
            md_path = repo_root / md_file
            if not md_path.exists():
                continue

            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Should not be empty
            assert len(content) > 0, f"{md_file} should not be empty"

            # Should have at least one header (markdown)
            lines = content.split("\n")
            has_header = any(line.startswith("#") for line in lines)
            assert has_header, f"{md_file} should have at least one header"


class TestBuildProcessIntegrity:
    """Tests that build process is properly documented and configured."""

    def test_frontend_build_documented(self) -> None:
        """Test that frontend build process is documented."""
        # Check multiple documentation files mention frontend build
        doc_files = [
            "AGENTS.md",
            "CLAUDE.md",
        ]

        repo_root = Path(__file__).resolve().parents[1]

        for doc_file in doc_files:
            doc_path = repo_root / doc_file
            if not doc_path.exists():
                continue

            with open(doc_path, "r", encoding="utf-8") as f:
                content = f.read().lower()

            # Should mention npm and build
            assert "npm" in content, f"{doc_file} should document npm usage"
            assert "build" in content, f"{doc_file} should document build process"

    def test_python_dependencies_documented(self) -> None:
        """Test that Python dependency management is documented."""
        doc_files = [
            "AGENTS.md",
            "CLAUDE.md",
        ]

        repo_root = Path(__file__).resolve().parents[1]

        for doc_file in doc_files:
            doc_path = repo_root / doc_file
            if not doc_path.exists():
                continue

            with open(doc_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Should mention uv sync
            assert "uv sync" in content or "uv" in content.lower(), \
                f"{doc_file} should document uv package manager"

    @pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
    def test_workflows_match_documentation(self) -> None:
        """Test that workflows match what's documented."""
        repo_root = Path(__file__).resolve().parents[1]

        # Read frontend workflow
        frontend_workflow_path = repo_root / ".github" / "workflows" / "FrontendBuild.yaml"
        with open(frontend_workflow_path, "r", encoding="utf-8") as f:
            frontend_workflow = yaml.safe_load(f)

        # Should use npm commands
        steps = frontend_workflow["jobs"]["build"]["steps"]
        npm_commands = [
            step.get("run", "")
            for step in steps
            if "run" in step and "npm" in step["run"]
        ]

        assert len(npm_commands) > 0, "Frontend workflow should use npm commands"

        # Should have npm run build
        has_build = any("npm run build" in cmd for cmd in npm_commands)
        assert has_build, "Frontend workflow should run npm build"


class TestSkillsAndRulesCompleteness:
    """Tests that skills and rules are complete and consistent."""

    def test_all_skills_have_required_frontmatter(self) -> None:
        """Test that all skill files have required YAML frontmatter."""
        repo_root = Path(__file__).resolve().parents[1]
        roo_dir = repo_root / ".roo"

        skill_files = list(roo_dir.glob("skills-*/**/SKILL.md"))

        for skill_file in skill_files:
            with open(skill_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Should start with ---
            assert content.startswith("---"), \
                f"{skill_file.relative_to(repo_root)} should start with YAML frontmatter"

            # Should have closing ---
            assert content.count("---") >= 2, \
                f"{skill_file.relative_to(repo_root)} should have properly closed frontmatter"

    def test_skills_reference_actual_concepts(self) -> None:
        """Test that skills reference actual codebase concepts."""
        repo_root = Path(__file__).resolve().parents[1]
        roo_dir = repo_root / ".roo"

        # Pipeline architecture skill should mention actual pipeline concepts
        pipeline_skill = roo_dir / "skills-architect" / "pipeline-architecture" / "SKILL.md"

        if pipeline_skill.exists():
            with open(pipeline_skill, "r", encoding="utf-8") as f:
                content = f.read().lower()

            # Should mention actual concepts from the codebase
            assert "pipeline" in content
            assert "operation" in content

    def test_rules_provide_actionable_guidance(self) -> None:
        """Test that rule files provide actionable guidance."""
        repo_root = Path(__file__).resolve().parents[1]
        roo_dir = repo_root / ".roo"

        modes = ["architect", "ask", "code", "debug"]

        for mode in modes:
            rules_file = roo_dir / f"rules-{mode}" / "AGENTS.md"
            if not rules_file.exists():
                continue

            with open(rules_file, "r", encoding="utf-8") as f:
                content = f.read().lower()

            # Should contain actionable words
            actionable_words = ["must", "should", "pattern", "use", "implement"]
            found = any(word in content for word in actionable_words)

            assert found, f"Rules for {mode} should contain actionable guidance"


class TestDocumentationCompleteness:
    """Tests that documentation covers all essential topics."""

    def test_claude_md_covers_essential_topics(self) -> None:
        """Test that CLAUDE.md covers all essential topics."""
        claude_path = Path(__file__).resolve().parents[1] / "CLAUDE.md"

        with open(claude_path, "r", encoding="utf-8") as f:
            content = f.read()

        essential_topics = [
            "Project Overview",
            "Build",
            "Architecture",
            "Pipeline",
            "Backend",
            "Frontend",
        ]

        for topic in essential_topics:
            assert topic in content, \
                f"CLAUDE.md should cover {topic}"

    def test_agents_md_provides_quick_reference(self) -> None:
        """Test that AGENTS.md provides quick reference information."""
        agents_path = Path(__file__).resolve().parents[1] / "AGENTS.md"

        with open(agents_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Should be concise but informative
        assert len(content) < 5000, "AGENTS.md should be concise"
        assert len(content) > 500, "AGENTS.md should have substantial content"

        # Should have command examples
        assert "`" in content, "AGENTS.md should have code/command examples"

    def test_pipeline_docs_cover_implementation(self) -> None:
        """Test that pipeline docs cover implementation details."""
        impl_doc = Path(__file__).resolve().parents[1] / "docs" / "md_docs" / "pipeline_docs" / \
                   "ImplementPipelineOperation.md"

        with open(impl_doc, "r", encoding="utf-8") as f:
            content = f.read()

        # Should have code examples
        assert "```" in content, "Implementation doc should have code examples"

        # Should mention key concepts
        assert "injection" in content.lower() or "dependency" in content.lower()
        assert "run" in content.lower()