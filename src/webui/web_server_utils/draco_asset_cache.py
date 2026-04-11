from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path
from typing import Any, Callable

from src.utils.colors import Colors


DRACO_EXTENSION = "KHR_draco_mesh_compression"
COMPRESSIBLE_MODEL_EXTENSIONS = {".glb"}
DRACO_CACHE_SETTINGS_VERSION = "draco-edgebreaker-v1"
MANIFEST_FILE_NAME = "manifest.json"


def default_gltf_transform_bin(repo_root: Path) -> Path:
    """Return the platform-specific local glTF-Transform CLI path."""
    binary_name = "gltf-transform.cmd" if os.name == "nt" else "gltf-transform"
    return repo_root / "node_modules" / ".bin" / binary_name


class DracoAssetCache:
    """Cache Draco-compressed copies of WebUI GLB assets."""

    def __init__(
        self,
        assets_dir: Path,
        cache_dir: Path,
        gltf_transform_bin: Path,
        logger: Callable[[str], None] | None = None,
        timeout_seconds: int = 1800,
    ) -> None:
        self.assets_dir = assets_dir.resolve()
        self.cache_dir = cache_dir.resolve()
        self.gltf_transform_bin = gltf_transform_bin
        self.logger = logger
        self.timeout_seconds = timeout_seconds
        self.manifest_path = self.cache_dir / MANIFEST_FILE_NAME
        self._lock = threading.Lock()
        self._compressor_available: bool | None = None
        self._compressor_version: str | None = None

    def prepare_all(self) -> None:
        """Compress all uncompressed GLB assets into the cache."""
        if not self.assets_dir.is_dir():
            self._log(
                f"{Colors.YELLOW}3D asset directory is missing: "
                f"{self.assets_dir}{Colors.RESET}"
            )
            return

        if not self._is_compressor_available():
            self._log(
                f"{Colors.YELLOW}Draco asset compression is disabled because "
                f"gltf-transform was not found at {self.gltf_transform_bin}. "
                f"Original GLB assets will be served.{Colors.RESET}"
            )
            return

        model_paths = sorted(
            path
            for path in self.assets_dir.rglob("*")
            if self._is_compressible_model(path)
        )
        if not model_paths:
            return

        compressed_count = 0
        cached_count = 0
        skipped_count = 0
        for source_path in model_paths:
            if self.asset_uses_draco(source_path):
                skipped_count += 1
                continue

            cache_path = self.cached_path_for_source(source_path)
            if self._is_cache_current(source_path, cache_path):
                cached_count += 1
                continue

            if self.ensure_cached(source_path) is not None:
                compressed_count += 1

        self._log(
            f"{Colors.GREEN}Draco asset cache ready: compressed={compressed_count} "
            f"cached={cached_count} already_draco={skipped_count}{Colors.RESET}"
        )

    def resolve_asset(self, relative_path: str | Path) -> Path | None:
        """Return the cached Draco asset when available, otherwise the source asset."""
        source_path = self._source_path(relative_path)
        if source_path is None or not source_path.is_file():
            return None

        if not self._is_compressible_model(source_path):
            return source_path

        if self.asset_uses_draco(source_path):
            return source_path

        cached_path = self.ensure_cached(source_path)
        return cached_path if cached_path is not None else source_path

    def ensure_cached(self, source_path: Path) -> Path | None:
        """Create or refresh the cached compressed copy for one source asset."""
        source_path = source_path.resolve()
        if not self._is_source_asset(source_path):
            return None

        if self.asset_uses_draco(source_path):
            return source_path

        cache_path = self.cached_path_for_source(source_path)
        with self._lock:
            if self._is_cache_current(source_path, cache_path):
                return cache_path

            if not self._is_compressor_available():
                return None

            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = cache_path.with_name(f"{cache_path.stem}.tmp{cache_path.suffix}")
            self._remove_sidecar_files(temp_path, cache_path)
            if temp_path.exists():
                temp_path.unlink()

            try:
                self._run_compressor(source_path, temp_path)
                if not self.asset_uses_draco(temp_path):
                    raise RuntimeError(
                        f"Compressed output does not declare {DRACO_EXTENSION}"
                )
                temp_path.replace(cache_path)
                self._remove_sidecar_files(temp_path, cache_path)
                self._record_cache_entry(source_path, cache_path)
            except Exception as exc:
                if temp_path.exists():
                    temp_path.unlink()
                self._remove_sidecar_files(temp_path, cache_path)
                self._log(
                    f"{Colors.YELLOW}Failed to Draco-compress {source_path}: "
                    f"{exc}. Serving original asset.{Colors.RESET}"
                )
                return None

        return cache_path

    def cached_path_for_source(self, source_path: Path) -> Path:
        relative_path = source_path.resolve().relative_to(self.assets_dir)
        return self.cache_dir / relative_path

    def asset_uses_draco(self, model_path: Path) -> bool:
        gltf_json = self._read_gltf_json(model_path)
        if gltf_json is None:
            return False
        return DRACO_EXTENSION in gltf_json

    def _run_compressor(self, source_path: Path, output_path: Path) -> None:
        result = subprocess.run(
            [
                str(self.gltf_transform_bin),
                "draco",
                str(source_path),
                str(output_path),
                "--method",
                "edgebreaker",
            ],
            cwd=self.assets_dir,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        if result.returncode != 0:
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            raise RuntimeError(
                "gltf-transform draco failed "
                f"exit_code={result.returncode} stdout={stdout[-2000:]} "
                f"stderr={stderr[-2000:]}"
            )

    def _source_path(self, relative_path: str | Path) -> Path | None:
        path = Path(relative_path)
        if path.is_absolute() or ".." in path.parts:
            return None
        source_path = (self.assets_dir / path).resolve()
        if not self._is_source_asset(source_path):
            return None
        return source_path

    def _is_source_asset(self, path: Path) -> bool:
        try:
            path.relative_to(self.assets_dir)
        except ValueError:
            return False
        return True

    def _is_compressible_model(self, path: Path) -> bool:
        return path.is_file() and path.suffix.lower() in COMPRESSIBLE_MODEL_EXTENSIONS

    def _is_compressor_available(self) -> bool:
        if self._compressor_available is not None:
            return self._compressor_available

        if not self.gltf_transform_bin.exists():
            self._compressor_available = False
            return False

        version = self._gltf_transform_version()
        self._compressor_available = version is not None
        return self._compressor_available

    def _gltf_transform_version(self) -> str | None:
        if self._compressor_version is not None:
            return self._compressor_version

        try:
            result = subprocess.run(
                [str(self.gltf_transform_bin), "--version"],
                cwd=self.assets_dir,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except Exception:
            return None

        if result.returncode != 0:
            return None

        self._compressor_version = result.stdout.strip() or result.stderr.strip()
        return self._compressor_version

    def _is_cache_current(self, source_path: Path, cache_path: Path) -> bool:
        if not cache_path.is_file():
            return False

        manifest = self._read_manifest()
        key = self._manifest_key(source_path)
        entry = manifest.get(key)
        if not isinstance(entry, dict):
            return False

        source_stat = source_path.stat()
        return (
            entry.get("source_size") == source_stat.st_size
            and entry.get("source_mtime_ns") == source_stat.st_mtime_ns
            and entry.get("settings_version") == DRACO_CACHE_SETTINGS_VERSION
            and entry.get("compressor_version") == self._gltf_transform_version()
            and self.asset_uses_draco(cache_path)
        )

    def _record_cache_entry(self, source_path: Path, cache_path: Path) -> None:
        manifest = self._read_manifest()
        source_stat = source_path.stat()
        cache_stat = cache_path.stat()
        manifest[self._manifest_key(source_path)] = {
            "source_size": source_stat.st_size,
            "source_mtime_ns": source_stat.st_mtime_ns,
            "cache_size": cache_stat.st_size,
            "settings_version": DRACO_CACHE_SETTINGS_VERSION,
            "compressor_version": self._gltf_transform_version(),
        }
        self._write_manifest(manifest)

    def _remove_sidecar_files(self, *model_paths: Path) -> None:
        """Remove external buffers that can be produced by temp extension guesses."""
        for model_path in model_paths:
            for sidecar_path in (
                Path(f"{model_path}.bin"),
                model_path.with_suffix(".bin"),
            ):
                if sidecar_path.exists():
                    sidecar_path.unlink()

    def _read_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.is_file():
            return {}
        try:
            with self.manifest_path.open("r", encoding="utf-8") as manifest_file:
                manifest = json.load(manifest_file)
        except (OSError, json.JSONDecodeError):
            return {}
        return manifest if isinstance(manifest, dict) else {}

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        temp_path = self.manifest_path.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file, indent=2, sort_keys=True)
            manifest_file.write("\n")
        temp_path.replace(self.manifest_path)

    def _manifest_key(self, source_path: Path) -> str:
        return source_path.resolve().relative_to(self.assets_dir).as_posix()

    def _read_gltf_json(self, model_path: Path) -> str | None:
        suffix = model_path.suffix.lower()
        try:
            if suffix == ".gltf":
                return model_path.read_text(encoding="utf-8")
            return self._read_glb_json_chunk(model_path)
        except (OSError, UnicodeDecodeError):
            return None

    def _read_glb_json_chunk(self, model_path: Path) -> str | None:
        with model_path.open("rb") as model_file:
            header = model_file.read(20)
            if len(header) < 20 or header[0:4] != b"glTF":
                return None
            if int.from_bytes(header[4:8], "little") != 2:
                return None
            json_chunk_length = int.from_bytes(header[12:16], "little")
            chunk_type = header[16:20]
            if chunk_type != b"JSON":
                return None
            json_bytes = model_file.read(json_chunk_length)
        return json_bytes.decode("utf-8").rstrip("\x00 \t\r\n")

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger(message)
