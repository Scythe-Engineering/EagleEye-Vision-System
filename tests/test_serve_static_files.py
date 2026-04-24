"""Tests for static asset path resolution."""

from __future__ import annotations

from pathlib import Path

from src.webui.web_server_utils import serve_static_files


def test_static_dir_points_to_built_webui_assets() -> None:
    expected = Path(__file__).resolve().parents[1] / "src" / "webui" / "static"
    assert serve_static_files.STATIC_DIR == expected
    assert serve_static_files.STATIC_DIR.is_dir()
    assert (serve_static_files.STATIC_DIR / "index.html").is_file()
    assert (serve_static_files.STATIC_DIR / "bundle.js").is_file()
    assert (serve_static_files.STATIC_DIR / "bundle2.js").is_file()


def test_pipeline_popup_modules_are_bundled_not_runtime_source_requests() -> None:
    webui_dir = Path(__file__).resolve().parents[1] / "src" / "webui"

    pipeline_creator = webui_dir / "js" / "pipeline" / "pipelineCreator.js"
    settings_popup = webui_dir / "js" / "pipeline" / "settingsPopup.js"
    pipeline_tab = webui_dir / "html" / "tabs" / "pipeline_tab_content.html"

    pipeline_creator_source = pipeline_creator.read_text(encoding="utf-8")
    settings_popup_source = settings_popup.read_text(encoding="utf-8")
    pipeline_tab_source = pipeline_tab.read_text(encoding="utf-8")

    assert (
        'import { registerSettingsPopup } from "./settingsPopup.js";'
        in pipeline_creator_source
    )
    assert "registerSettingsPopup();" in pipeline_creator_source
    assert (
        'import { registerFileManagerPopup } from "./fileManager.js";'
        in settings_popup_source
    )
    assert "registerFileManagerPopup();" in settings_popup_source
    assert "document.createElement(\"script\")" not in pipeline_creator_source
    assert "/js/pipeline/" not in pipeline_creator_source
    assert "../../js/pipeline/" not in pipeline_creator_source
    assert 'src="./js/pipeline/' not in pipeline_tab_source


def test_built_webui_bundle_includes_popup_modules_without_source_js_requests() -> None:
    static_dir = serve_static_files.STATIC_DIR
    built_scripts = "\n".join(
        [
            (static_dir / "bundle.js").read_text(encoding="utf-8"),
            (static_dir / "bundle2.js").read_text(encoding="utf-8"),
        ]
    )

    assert 'src="/js/pipeline/' not in built_scripts
    assert "src='/js/pipeline/" not in built_scripts
    assert 'src="./js/pipeline/' not in built_scripts
    assert "src='./js/pipeline/" not in built_scripts
    assert 'import("/js/pipeline/' not in built_scripts
    assert "import('/js/pipeline/" not in built_scripts
    assert "fileManagerOverlay" in built_scripts
    assert "operationSettingsOverlay" in built_scripts
