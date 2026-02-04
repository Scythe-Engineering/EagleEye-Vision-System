"""Rust build helpers for tests."""

from __future__ import annotations

from functools import lru_cache

from src.rust_implementations.build import main as rust_build
from src.utils.logging.logger import Logger


@lru_cache(maxsize=1)
def ensure_rust_modules_built() -> None:
    """Build Rust extension modules required by tests."""

    logger = Logger(log_directory="logs/test")
    try:
        build_success = rust_build(logger=logger)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Failed to build Rust implementations for tests. "
            "Install the Rust toolchain and maturin, then run "
            "`uv run python src/rust_implementations/build.py --all`."
        ) from exc

    if not build_success:
        raise RuntimeError(
            "Rust implementation build returned failure. "
            "Run `uv run python src/rust_implementations/build.py --all` "
            "and ensure cargo and maturin are available."
        )
