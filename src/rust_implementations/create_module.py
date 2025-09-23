"""
Script to create a new Rust module from the template.

Usage:
    uv run python create_module.py module_name "Module description"
"""

import sys
import shutil
from pathlib import Path


def create_module(module_name: str, description: str) -> None:
    """Create a new module from the template."""
    root_dir = Path(__file__).parent
    template_dir = root_dir / "module_template"
    modules_dir = root_dir / "modules"
    new_module_dir = modules_dir / module_name

    if not template_dir.exists():
        print(f"Error: Template directory not found at {template_dir}")
        return False

    if new_module_dir.exists():
        print(f"Error: Module '{module_name}' already exists")
        return False

    # Copy template to new module directory
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

    print(f"✓ Created new module: {module_name}")
    print(f"  Location: {new_module_dir}")
    print(f"  Description: {description}")
    print()
    print("Next steps:")
    print(f"  1. Edit {new_module_dir}/src/lib.rs to implement your functionality")
    print(f"  2. Update {new_module_dir}/Cargo.toml with any additional dependencies")
    print(f"  3. Build the module: uv run python build.py {module_name}")

    return True


def main():
    """Main entry point."""
    if len(sys.argv) != 3:
        print('Usage: uv run python create_module.py module_name "Module description"')
        print(
            'Example: uv run python create_module.py image_processor "High-performance image processing"'
        )
        sys.exit(1)

    module_name = sys.argv[1]
    description = sys.argv[2]

    # Validate module name (should be valid Python/Rust identifier)
    if not module_name.replace("_", "").isalnum() or not module_name[0].isalpha():
        print(
            "Error: Module name must be a valid identifier (letters, numbers, underscores only, starting with a letter)"
        )
        sys.exit(1)

    success = create_module(module_name, description)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
