"""
Script to create a new Rust module from the template.

Usage:
    uv run python create_module.py module_name "Module description"
"""

import sys
import shutil
from pathlib import Path


# ANSI color codes for colored console output
class Colors:
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"


def create_module(module_name: str, description: str) -> bool:
    """Create a new module from the template."""
    root_dir = Path(__file__).parent
    template_dir = root_dir / "module_template"
    modules_dir = root_dir / "modules"
    new_module_dir = modules_dir / module_name

    if not template_dir.exists():
        print(
            f"{Colors.RED}Error: Template directory not found at {template_dir}{Colors.RESET}"
        )
        return False

    if new_module_dir.exists():
        print(f"{Colors.RED}Error: Module '{module_name}' already exists{Colors.RESET}")
        return False

    try:
        shutil.copytree(template_dir, new_module_dir)

        # Update template files
        replacements = {
            "{{MODULE_NAME}}": module_name,
            "{{MODULE_CLASS_NAME}}": "".join(
                word.capitalize() for word in module_name.split("_")
            ),
            "{{MODULE_DESCRIPTION}}": description,
        }

        # Update files with replacements
        for file_path in new_module_dir.rglob("*"):
            if file_path.is_file():
                try:
                    content = file_path.read_text()
                    for old, new in replacements.items():
                        content = content.replace(old, new)
                    file_path.write_text(content)
                except UnicodeDecodeError:
                    # Skip binary files
                    pass
    except (OSError, PermissionError, IOError) as e:
        # Clean up partially created module directory
        try:
            if new_module_dir.exists():
                shutil.rmtree(new_module_dir)
        except (OSError, PermissionError, IOError):
            # Don't mask the original error if cleanup fails
            pass
        # Re-raise the original exception with additional context
        raise RuntimeError(f"Failed to create module '{module_name}': {e}") from e

    print(f"{Colors.GREEN}✓ Created new module: {module_name}{Colors.RESET}")
    print(f"{Colors.CYAN}  Location: {new_module_dir}{Colors.RESET}")
    print(f"{Colors.CYAN}  Description: {description}{Colors.RESET}")
    print()
    print(f"{Colors.CYAN}Next steps:{Colors.RESET}")
    print(
        f"{Colors.CYAN}  1. Edit {new_module_dir}/src/lib.rs to implement your functionality{Colors.RESET}"
    )
    print(
        f"{Colors.CYAN}  2. Update {new_module_dir}/Cargo.toml with any additional dependencies{Colors.RESET}"
    )
    print(
        f"{Colors.CYAN}  3. Build the module: uv run python build.py {module_name}{Colors.RESET}"
    )

    return True


def main():
    """Main entry point."""
    if len(sys.argv) != 3:
        print(
            f'{Colors.RED}Usage: uv run python create_module.py module_name "Module description"{Colors.RESET}'
        )
        print(
            f'{Colors.YELLOW}Example: uv run python create_module.py image_processor "High-performance image processing"{Colors.RESET}'
        )
        sys.exit(1)

    module_name = sys.argv[1]
    description = sys.argv[2]

    # Validate module name (should be valid Python/Rust identifier)
    if not module_name.replace("_", "").isalnum() or not module_name[0].isalpha():
        print(
            f"{Colors.RED}Error: Module name must be a valid identifier (letters, numbers, underscores only, starting with a letter){Colors.RESET}"
        )
        sys.exit(1)

    success = create_module(module_name, description)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
