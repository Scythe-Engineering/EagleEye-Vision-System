"""
Master build script for Rust implementations.

This script manages building multiple Rust extension modules.
It detects changes in source files and rebuilds only modules that have changed
or haven't been built yet.

Usage:
    python build.py                    # Build all modules that need rebuilding
    python build.py --all             # Force rebuild all modules
    python build.py module_name       # Build specific module
    python build.py --clean           # Clean all build artifacts
"""

import subprocess
import sys
import argparse
from pathlib import Path
import hashlib
import json


# ANSI color codes for colored console output
class Colors:
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"


class RustModuleBuilder:
    """Manages building of Rust extension modules."""

    def __init__(self, root_dir: Path):
        """Initialize the builder with the root directory."""
        self.root_dir = root_dir
        self.modules_dir = root_dir / "modules"
        self.build_cache_file = root_dir / ".build_cache.json"

    def get_modules(self) -> list[Path]:
        """Get all module directories."""
        if not self.modules_dir.exists():
            return []

        return [
            d
            for d in self.modules_dir.iterdir()
            if d.is_dir() and (d / "Cargo.toml").exists()
        ]

    def get_module_hash(self, module_dir: Path) -> str:
        """Calculate hash of all source files in a module."""
        hasher = hashlib.md5()

        # Include Cargo.toml
        cargo_toml = module_dir / "Cargo.toml"
        if cargo_toml.exists():
            hasher.update(cargo_toml.read_bytes())

        # Include all Rust source files
        src_dir = module_dir / "src"
        if src_dir.exists():
            for rust_file in src_dir.rglob("*.rs"):
                hasher.update(rust_file.read_bytes())

        return hasher.hexdigest()

    def load_build_cache(self) -> dict:
        """Load the build cache from disk."""
        if self.build_cache_file.exists():
            try:
                with open(self.build_cache_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {}

    def clean_build_cache(self, cache: dict) -> dict:
        """Remove cache entries for modules that no longer exist."""
        existing_modules = {module_dir.name for module_dir in self.get_modules()}

        # Remove entries for modules that no longer exist
        cleaned_cache = {}
        for module_name, data in cache.items():
            if module_name in existing_modules:
                cleaned_cache[module_name] = data
            else:
                print(
                    f"{Colors.YELLOW}Removed cache entry for deleted module: {module_name}{Colors.RESET}"
                )

        return cleaned_cache

    def save_build_cache(self, cache: dict) -> None:
        """Save the build cache to disk."""
        try:
            with open(self.build_cache_file, "w") as f:
                json.dump(cache, f, indent=2)
        except IOError:
            print(
                f"{Colors.YELLOW}Warning: Could not save build cache to {self.build_cache_file}{Colors.RESET}"
            )

    def needs_rebuild(self, module_dir: Path, cache: dict) -> bool:
        """Check if a module needs rebuilding."""
        module_name = module_dir.name
        current_hash = self.get_module_hash(module_dir)
        cached_hash = cache.get(module_name, {}).get("hash")

        return current_hash != cached_hash

    def test_module_import(self, module_name: str) -> bool:
        """Test that a module can be imported in Python."""
        print(f"{Colors.CYAN}Testing import of {module_name}...{Colors.RESET}")

        try:
            result = subprocess.run(
                ["uv", "run", "python", "-c", f"import {module_name}"],
                cwd=self.root_dir.parent,  # Run from workspace root
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                print(
                    f"{Colors.GREEN}✓ Successfully imported {module_name}{Colors.RESET}"
                )
                return True
            else:
                print(f"{Colors.RED}✗ Failed to import {module_name}{Colors.RESET}")
                if result.stderr:
                    print(f"{Colors.RED}Import error:{Colors.RESET}")
                    print(result.stderr)
                return False

        except subprocess.CalledProcessError as e:
            print(
                f"{Colors.RED}✗ Import test failed for {module_name}: {e}{Colors.RESET}"
            )
            return False
        except FileNotFoundError as e:
            print(
                f"{Colors.RED}✗ Import test failed for {module_name}: {e}{Colors.RESET}"
            )
            return False

    def build_module(self, module_dir: Path) -> bool:
        """Build a single module."""
        module_name = module_dir.name
        print(f"{Colors.CYAN}Building module: {module_name}{Colors.RESET}")

        try:
            # Check if module has its own build.py
            build_script = module_dir / "build.py"
            if build_script.exists():
                # Run module-specific build script
                result = subprocess.run(
                    ["uv", "run", "python", str(build_script)],
                    cwd=module_dir,
                    capture_output=True,
                    text=True,
                )
            else:
                # Fallback to direct maturin build
                if not self.check_dependencies():
                    return False

                result = subprocess.run(
                    ["maturin", "develop"],
                    cwd=module_dir,
                    capture_output=True,
                    text=True,
                )

            if result.returncode == 0:
                print(f"{Colors.GREEN}✓ Successfully built {module_name}{Colors.RESET}")
                if result.stdout:
                    print(result.stdout)

                # Test that the module can be imported
                if not self.test_module_import(module_name):
                    print(
                        f"{Colors.RED}✗ Build verification failed for {module_name}{Colors.RESET}"
                    )
                    return False

                return True
            else:
                print(f"{Colors.RED}✗ Failed to build {module_name}{Colors.RESET}")
                if result.stderr:
                    print(f"{Colors.RED}Error output:{Colors.RESET}")
                    print(result.stderr)
                return False

        except subprocess.CalledProcessError as e:
            print(f"{Colors.RED}✗ Build failed for {module_name}: {e}{Colors.RESET}")
            return False
        except FileNotFoundError as e:
            print(f"{Colors.RED}✗ Build failed for {module_name}: {e}{Colors.RESET}")
            return False

    def check_dependencies(self) -> bool:
        """Check if required build dependencies are available."""
        try:
            # Check cargo
            subprocess.run(["cargo", "--version"], capture_output=True, check=True)

            # Check maturin
            result = subprocess.run(["maturin", "--version"], capture_output=True)
            if result.returncode != 0:
                print(f"{Colors.YELLOW}Installing maturin...{Colors.RESET}")
                subprocess.run(["uv", "pip", "install", "maturin"], check=True)

            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(
                f"{Colors.RED}Error: Rust and maturin are required for building.{Colors.RESET}"
            )
            print(
                f"{Colors.RED}Please install Rust from https://rustup.rs/{Colors.RESET}"
            )
            return False

    def clean_module(self, module_dir: Path) -> None:
        """Clean build artifacts for a module."""
        module_name = module_dir.name
        print(f"{Colors.CYAN}Cleaning {module_name}...{Colors.RESET}")

        # Remove target directory
        target_dir = module_dir / "target"
        if target_dir.exists():
            import shutil

            shutil.rmtree(target_dir)
            print(f"{Colors.CYAN}Removed {target_dir}{Colors.RESET}")

        # Remove any .so files in the module directory
        for so_file in module_dir.glob("*.so"):
            so_file.unlink()
            print(f"{Colors.CYAN}Removed {so_file}{Colors.RESET}")

    def build_all(self, force: bool = False) -> bool:
        """Build all modules that need rebuilding."""
        modules = self.get_modules()
        if not modules:
            print(
                f"{Colors.YELLOW}No modules found in modules/ directory{Colors.RESET}"
            )
            return True

        cache = self.load_build_cache()
        success = True

        for module_dir in modules:
            module_name = module_dir.name

            if force or self.needs_rebuild(module_dir, cache):
                if not self.build_module(module_dir):
                    success = False
                else:
                    # Update cache with new hash
                    cache[module_name] = {
                        "hash": self.get_module_hash(module_dir),
                        "last_built": str(Path(module_dir).stat().st_mtime),
                    }
            else:
                print(f"{Colors.GREEN}✓ {module_name} is up to date{Colors.RESET}")

        # Clean cache of non-existent modules and save
        cleaned_cache = self.clean_build_cache(cache)
        self.save_build_cache(cleaned_cache)
        return success

    def build_specific(self, module_name: str) -> bool:
        """Build a specific module."""
        module_dir = self.modules_dir / module_name
        if not module_dir.exists():
            print(f"{Colors.RED}Module '{module_name}' not found{Colors.RESET}")
            return False

        if not (module_dir / "Cargo.toml").exists():
            print(
                f"{Colors.RED}'{module_name}' is not a valid Rust module{Colors.RESET}"
            )
            return False

        cache = self.load_build_cache()

        if self.build_module(module_dir):
            # Update cache
            cache[module_name] = {
                "hash": self.get_module_hash(module_dir),
                "last_built": str(module_dir.stat().st_mtime),
            }
            # Clean cache of non-existent modules and save
            cleaned_cache = self.clean_build_cache(cache)
            self.save_build_cache(cleaned_cache)
            return True

        return False

    def clean_all(self) -> None:
        """Clean all modules."""
        modules = self.get_modules()
        for module_dir in modules:
            self.clean_module(module_dir)

        # Remove build cache
        if self.build_cache_file.exists():
            self.build_cache_file.unlink()
            print(f"{Colors.CYAN}Removed build cache{Colors.RESET}")


def main(ran_directly=False):
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Build Rust extension modules")
    parser.add_argument("module", nargs="?", help="Specific module to build")
    parser.add_argument("--all", action="store_true", help="Force rebuild all modules")
    parser.add_argument(
        "--clean", action="store_true", help="Clean all build artifacts"
    )
    parser.add_argument(
        "--list", action="store_true", help="List all available modules"
    )

    args = parser.parse_args()

    root_dir = Path(__file__).parent
    builder = RustModuleBuilder(root_dir)

    if args.list:
        modules = builder.get_modules()
        if modules:
            print(f"{Colors.CYAN}Available modules:{Colors.RESET}")
            for module in modules:
                print(f"  - {module.name}")
        else:
            print(f"{Colors.YELLOW}No modules found{Colors.RESET}")
        return

    if args.clean:
        builder.clean_all()
        return

    if args.module:
        success = builder.build_specific(args.module)
    else:
        success = builder.build_all(force=args.all)

    if ran_directly:
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main(ran_directly=True)
