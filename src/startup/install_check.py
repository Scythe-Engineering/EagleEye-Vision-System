from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from shutil import which
from typing import Protocol

from src.rust_implementations.build import RustModuleBuilder
from src.utils.colors import Colors


class _LoggerLike(Protocol):
    def log(self, message: str) -> None: ...


PROJECT_REQUIRED_IMPORTS = (
    "cv2",
    "flask",
    "flask_cors",
    "flask_socketio",
    "networktables",
    "numpy",
    "psutil",
)
WEBUI_REQUIRED_ASSETS = ("index.html", "bundle.js", "main.css")


class StartupInstallChecker:
    """Validate the local install before backend startup."""

    def __init__(self, logger: _LoggerLike, repo_root: Path | None = None) -> None:
        self.logger = logger
        self.repo_root = repo_root or Path(__file__).resolve().parents[2]
        self.venv_dir = self.repo_root / ".venv"
        self.venv_python = self.venv_dir / "bin" / "python"
        self.package_json = self.repo_root / "package.json"
        self.node_modules_dir = self.repo_root / "node_modules"
        self.vite_bin = self.node_modules_dir / ".bin" / "vite"
        self.webui_dir = self.repo_root / "src" / "webui"
        self.static_dir = self.webui_dir / "static"
        self.rust_builder = RustModuleBuilder(
            self.repo_root / "src" / "rust_implementations",
            logger=logger,
        )

    def ensure_startup_requirements(self) -> None:
        """Validate and prepare the runtime environment."""
        self._check_python_runtime()
        uv_sync_ran = self._ensure_uv_environment()
        self._ensure_rust_toolchain()
        self._ensure_npm_dependencies()
        self._ensure_webui_build()
        if uv_sync_ran:
            self._log(
                f"{Colors.YELLOW}uv sync ran during startup. Editable Rust modules "
                "may have been removed; the existing Rust bootstrap will rebuild "
                f"them next.{Colors.RESET}"
            )

    def _check_python_runtime(self) -> None:
        if sys.version_info < (3, 11):
            raise RuntimeError(
                "Python 3.11 or newer is required. "
                f"Current interpreter: {sys.executable} "
                f"({sys.version.split()[0]})."
            )

        self._log(
            f"{Colors.CYAN}Python runtime:{Colors.RESET} "
            f"{sys.executable} ({sys.version.split()[0]})"
        )

    def _ensure_uv_environment(self) -> bool:
        uv_path = which("uv")
        if uv_path is None:
            raise RuntimeError(
                "uv is required but was not found on PATH. Install uv and rerun startup."
            )

        self._log(f"{Colors.CYAN}uv executable:{Colors.RESET} {uv_path}")

        current_python = Path(sys.executable).resolve()
        if not self.venv_python.exists() or not self._imports_available(current_python):
            reason = "current Python environment is missing required packages"
            if not self.venv_python.exists():
                reason = f"virtualenv Python not found at {self.venv_python}"
            self._log(
                f"{Colors.YELLOW}Python environment is not ready ({reason}); "
                f"running `uv sync`...{Colors.RESET}"
            )
            self._run_command(["uv", "sync"], cwd=self.repo_root, timeout=1800)

            if not self.venv_python.exists():
                raise RuntimeError(
                    f"uv sync completed, but the repository virtualenv was not created at {self.venv_python}."
                )

            if self._imports_available(current_python):
                self._log(
                    f"{Colors.GREEN}Current Python environment is ready after uv sync.{Colors.RESET}"
                )
                return True

            if self.venv_python.exists() and self._imports_available(self.venv_python):
                raise RuntimeError(
                    "Dependencies were synced into the repository virtualenv, but "
                    f"the backend is running with {current_python} instead of "
                    f"{self.venv_python}. Start the backend with `uv run python "
                    "src/main_backend.py` or the repo venv Python."
                )

            raise RuntimeError(
                "uv sync completed, but required Python packages are still unavailable "
                f"from {current_python}."
            )

        if self.venv_python.exists() and current_python != self.venv_python.resolve():
            self._log(
                f"{Colors.YELLOW}Backend is not using the repo virtualenv. "
                f"current_python={current_python} expected_python={self.venv_python}{Colors.RESET}"
            )
        else:
            self._log(
                f"{Colors.GREEN}Python environment is ready without running uv sync.{Colors.RESET}"
            )

        return False

    def _ensure_rust_toolchain(self) -> None:
        self._log(f"{Colors.CYAN}Checking Rust and maturin toolchain...{Colors.RESET}")
        if not self.rust_builder.check_dependencies():
            raise RuntimeError(
                "Rust toolchain or maturin is unavailable. Backend initialization "
                "cannot continue."
            )

    def _ensure_npm_dependencies(self) -> None:
        if not self.package_json.is_file():
            raise RuntimeError(f"package.json not found at {self.package_json}")

        npm_path = which("npm")
        if npm_path is None:
            raise RuntimeError("npm is required for the WebUI but was not found on PATH.")

        self._log(f"{Colors.CYAN}npm executable:{Colors.RESET} {npm_path}")

        if self._npm_install_ready():
            self._log(
                f"{Colors.GREEN}npm dependencies are already installed.{Colors.RESET}"
            )
            return

        self._log(
            f"{Colors.YELLOW}npm dependencies are missing; running `npm install`...{Colors.RESET}"
        )
        self._run_command(["npm", "install"], cwd=self.repo_root, timeout=1800)

        if not self._npm_install_ready():
            raise RuntimeError(
                "npm install completed but required frontend packages are still missing."
            )

    def _ensure_webui_build(self) -> None:
        if not self._webui_build_required():
            self._log(f"{Colors.GREEN}WebUI build artifacts are up to date.{Colors.RESET}")
            return

        self._log(
            f"{Colors.YELLOW}WebUI build artifacts are missing or stale; "
            f"running `npm run build`...{Colors.RESET}"
        )
        self._run_command(["npm", "run", "build"], cwd=self.repo_root, timeout=1800)

        missing_assets = self._missing_webui_assets()
        if missing_assets:
            raise RuntimeError(
                "WebUI build completed but required assets are still missing: "
                f"{missing_assets}"
            )

    def _imports_available(self, python_executable: Path) -> bool:
        if not python_executable.exists():
            return False

        result = subprocess.run(
            [
                str(python_executable),
                "-c",
                f"import {', '.join(PROJECT_REQUIRED_IMPORTS)}",
            ],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0

    def _npm_install_ready(self) -> bool:
        return (
            self.node_modules_dir.is_dir()
            and self.vite_bin.exists()
            and (self.node_modules_dir / "three" / "package.json").is_file()
        )

    def _missing_webui_assets(self) -> list[str]:
        return [
            asset_name
            for asset_name in WEBUI_REQUIRED_ASSETS
            if not (self.static_dir / asset_name).is_file()
        ]

    def _webui_build_required(self) -> bool:
        missing_assets = self._missing_webui_assets()
        if missing_assets:
            self._log(
                f"{Colors.YELLOW}Missing WebUI build assets: {missing_assets} "
                f"in {self.static_dir}{Colors.RESET}"
            )
            return True

        source_files = self._webui_source_files()
        if not source_files:
            return False

        latest_source_mtime = max(path.stat().st_mtime for path in source_files)
        oldest_output_mtime = min(
            (self.static_dir / asset_name).stat().st_mtime
            for asset_name in WEBUI_REQUIRED_ASSETS
        )
        return latest_source_mtime > oldest_output_mtime

    def _webui_source_files(self) -> list[Path]:
        source_files = [
            self.repo_root / "package.json",
            self.repo_root / "vite.config.js",
            self.webui_dir / "index.html",
            self.webui_dir / "style.css",
        ]
        for relative_dir, glob_pattern in (
            ("js", "*.js"),
            ("css", "*.css"),
            ("html", "*.html"),
        ):
            root = self.webui_dir / relative_dir
            if root.is_dir():
                source_files.extend(root.rglob(glob_pattern))
        return [path for path in source_files if path.exists()]

    def _run_command(
        self,
        command: list[str],
        *,
        cwd: Path,
        timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0:
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            raise RuntimeError(
                f"Command failed: {' '.join(command)} (cwd={cwd}) "
                f"exit_code={result.returncode} stdout={stdout[-4000:]} "
                f"stderr={stderr[-4000:]}"
            )
        return result

    def _log(self, message: str) -> None:
        self.logger.log(message)
