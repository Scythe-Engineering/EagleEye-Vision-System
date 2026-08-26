from pathlib import Path
import re


STYLE_PATH = Path(__file__).parents[1] / "src" / "webui" / "style.css"


def _rule(css: str, selector: str) -> str:
    """Return the declarations for an exact CSS selector."""
    match = re.search(rf"(?:^|\n){re.escape(selector)}\s*\{{([^}}]+)\}}", css)
    assert match, f"Missing shared style for {selector}"
    return match.group(1)


def test_common_native_controls_have_global_project_styles() -> None:
    css = STYLE_PATH.read_text(encoding="utf-8")

    assert "scrollbar-color:" in _rule(css, "*")
    assert "background-color: var(--eagle-scrollbar-thumb)" in _rule(
        css, "*::-webkit-scrollbar-thumb"
    )
    assert "background-image:" in _rule(css, "select")
    assert "color-scheme: dark" in _rule(css, 'input[type="number"]')
    assert "filter:" in _rule(
        css, 'input[type="number"]::-webkit-inner-spin-button'
    )
    assert "appearance: none" in _rule(css, 'input[type="checkbox"]')
