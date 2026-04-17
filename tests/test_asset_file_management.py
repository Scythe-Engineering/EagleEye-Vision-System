"""Tests for managed robot and field GLB files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.webui import web_server
from src.webui.web_server import EagleEyeInterface


class _FakeUpload:
    def __init__(self, filename: str, content: bytes = b"glb") -> None:
        self.filename = filename
        self.content = content

    def save(self, destination: str) -> None:
        Path(destination).write_bytes(self.content)


class _FakeRequest:
    def __init__(
        self,
        files: dict[str, Any] | None = None,
        form: dict[str, Any] | None = None,
    ) -> None:
        self.files = files or {}
        self.form = form or {}


class _FakeDracoCache:
    def __init__(self) -> None:
        self.resolved_paths: list[Path] = []

    def resolve_asset(self, relative_path: Path) -> None:
        self.resolved_paths.append(relative_path)


def _interface() -> EagleEyeInterface:
    interface = EagleEyeInterface.__new__(EagleEyeInterface)
    interface.log = lambda *_args, **_kwargs: None
    interface.draco_asset_cache = _FakeDracoCache()
    return interface


def _setup_webui_assets(tmp_path: Path, monkeypatch: Any) -> Path:
    webui_dir = tmp_path / "webui"
    (webui_dir / "assets" / "robots").mkdir(parents=True, exist_ok=True)
    (webui_dir / "assets" / "fields" / "2025" / "field_files").mkdir(
        parents=True,
        exist_ok=True,
    )
    monkeypatch.setattr(web_server, "current_path", str(webui_dir))
    return webui_dir


def test_robot_files_are_listed_uploaded_and_draco_prepared(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    webui_dir = _setup_webui_assets(tmp_path, monkeypatch)
    robot_dir = webui_dir / "assets" / "robots"
    (robot_dir / "Practice.glb").write_bytes(b"robot")
    (robot_dir / "_hidden.glb").write_bytes(b"hidden")
    (robot_dir / "notes.txt").write_text("ignored", encoding="utf-8")

    interface = _interface()
    payload, status = interface.get_robot_files()

    assert status == 200
    assert payload["robots"] == ["Practice.glb"]
    assert payload["file_details"][0]["filename"] == "Practice.glb"

    monkeypatch.setattr(
        web_server,
        "request",
        _FakeRequest(files={"file": _FakeUpload("New Bot.glb", b"new")}),
    )
    payload, status = interface.upload_robot_file()

    assert status == 200
    assert payload["file"]["filename"] == "New_Bot.glb"
    assert (robot_dir / "New_Bot.glb").read_bytes() == b"new"
    assert interface.draco_asset_cache.resolved_paths == [
        Path("robots") / "New_Bot.glb"
    ]


def test_robot_upload_requires_explicit_overwrite(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    webui_dir = _setup_webui_assets(tmp_path, monkeypatch)
    robot_path = webui_dir / "assets" / "robots" / "Existing.glb"
    robot_path.write_bytes(b"old")
    interface = _interface()

    monkeypatch.setattr(
        web_server,
        "request",
        _FakeRequest(files={"file": _FakeUpload("Existing.glb", b"new")}),
    )
    payload, status = interface.upload_robot_file()

    assert status == 409
    assert payload["requires_overwrite"] is True
    assert robot_path.read_bytes() == b"old"

    monkeypatch.setattr(
        web_server,
        "request",
        _FakeRequest(
            files={"file": _FakeUpload("Existing.glb", b"new")},
            form={"overwrite": "true"},
        ),
    )
    payload, status = interface.upload_robot_file()

    assert status == 200
    assert robot_path.read_bytes() == b"new"


def test_field_files_are_grouped_uploaded_and_deleted(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    webui_dir = _setup_webui_assets(tmp_path, monkeypatch)
    field_dir = webui_dir / "assets" / "fields" / "2025" / "field_files"
    game_piece_dir = webui_dir / "assets" / "fields" / "2025" / "game_pieces"
    game_piece_dir.mkdir(parents=True, exist_ok=True)
    field_path = field_dir / "Field.glb"
    field_path.write_bytes(b"field")
    (game_piece_dir / "Piece.glb").write_bytes(b"piece")
    interface = _interface()

    payload, status = interface.get_field_files()

    assert status == 200
    assert payload["fields"] == {"2025": ["Field.glb"]}
    field_detail = payload["file_details"][0]
    assert field_detail["path"] == "2025/field_files/Field.glb"
    assert field_detail["asset_path"] == "fields/2025/field_files/Field.glb"
    assert field_detail["url"] == "/assets/fields/2025/field_files/Field.glb"
    assert field_detail["game_piece_urls"] == [
        "/assets/fields/2025/game_pieces/Piece.glb"
    ]
    assert field_detail["apriltag_map_url"] is None

    monkeypatch.setattr(
        web_server,
        "request",
        _FakeRequest(
            files={
                "file": _FakeUpload("Custom Field.glb", b"custom"),
                "apriltag_map": _FakeUpload("Custom Field.fmap", b'{"fiducials": []}'),
            },
            form={"year": "2026"},
        ),
    )
    payload, status = interface.upload_field_file()

    assert status == 200
    assert payload["file"]["path"] == "2026/field_files/Custom_Field.glb"
    assert payload["file"]["url"] == "/assets/fields/2026/field_files/Custom_Field.glb"
    assert payload["file"]["game_piece_urls"] == []
    assert (
        payload["file"]["apriltag_map_url"]
        == "/assets/fields/2026/apriltag_maps/Custom_Field.fmap"
    )
    assert (
        webui_dir / "assets" / "fields" / "2026" / "field_files" / "Custom_Field.glb"
    ).read_bytes() == b"custom"
    assert (
        webui_dir / "assets" / "fields" / "2026" / "apriltag_maps" / "Custom_Field.fmap"
    ).read_bytes() == b'{"fiducials": []}'
    assert interface.draco_asset_cache.resolved_paths == [
        Path("fields") / "2026" / "field_files" / "Custom_Field.glb"
    ]

    monkeypatch.setattr(web_server, "request", _FakeRequest())
    payload, status = interface.delete_field_file("2025", "Field.glb")

    assert status == 200
    assert payload == {"success": True}
    assert not field_path.exists()


def test_asset_upload_rejects_invalid_names(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    _setup_webui_assets(tmp_path, monkeypatch)
    interface = _interface()

    monkeypatch.setattr(
        web_server,
        "request",
        _FakeRequest(files={"file": _FakeUpload("../bad.glb")}),
    )
    payload, status = interface.upload_robot_file()

    assert status == 400
    assert "Path separators" in payload["error"]

    monkeypatch.setattr(
        web_server,
        "request",
        _FakeRequest(files={"file": _FakeUpload("bad.txt")}),
    )
    payload, status = interface.upload_robot_file()

    assert status == 400
    assert "Only .glb" in payload["error"]
