"""Tests for static asset path resolution."""

from __future__ import annotations

from pathlib import Path

from src.webui.web_server_utils import serve_static_files


def test_static_dir_points_to_built_webui_assets() -> None:
    expected = Path(__file__).resolve().parents[1] / "src" / "webui" / "static"
    assert serve_static_files.STATIC_DIR == expected
    assert serve_static_files.STATIC_DIR.is_dir()
    assert (serve_static_files.STATIC_DIR / "index.html").is_file()

