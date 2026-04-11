"""Tests for Draco-compressed WebUI asset caching."""

from __future__ import annotations

import json
from pathlib import Path

from src.webui.web_server_utils.draco_asset_cache import (
    DRACO_EXTENSION,
    DracoAssetCache,
)


def _write_glb(path: Path, *, uses_draco: bool) -> None:
    gltf_json: dict[str, object] = {
        "asset": {"version": "2.0", "generator": "test"},
        "scene": 0,
        "scenes": [{"nodes": []}],
    }
    if uses_draco:
        gltf_json["extensionsUsed"] = [DRACO_EXTENSION]
        gltf_json["extensionsRequired"] = [DRACO_EXTENSION]

    json_bytes = json.dumps(gltf_json, separators=(",", ":")).encode("utf-8")
    padding = (4 - len(json_bytes) % 4) % 4
    padded_json = json_bytes + (b" " * padding)
    total_length = 12 + 8 + len(padded_json)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as glb_file:
        glb_file.write(b"glTF")
        glb_file.write((2).to_bytes(4, "little"))
        glb_file.write(total_length.to_bytes(4, "little"))
        glb_file.write(len(padded_json).to_bytes(4, "little"))
        glb_file.write(b"JSON")
        glb_file.write(padded_json)


def test_asset_uses_draco_reads_glb_json_chunk(tmp_path: Path) -> None:
    assets_dir = tmp_path / "assets"
    cache = DracoAssetCache(
        assets_dir=assets_dir,
        cache_dir=tmp_path / "cache",
        gltf_transform_bin=tmp_path / "missing-gltf-transform",
    )
    uncompressed = assets_dir / "field.glb"
    compressed = assets_dir / "robot.glb"
    _write_glb(uncompressed, uses_draco=False)
    _write_glb(compressed, uses_draco=True)

    assert cache.asset_uses_draco(uncompressed) is False
    assert cache.asset_uses_draco(compressed) is True


def test_resolve_asset_creates_and_reuses_compressed_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    assets_dir = tmp_path / "assets"
    source = assets_dir / "fields" / "2025" / "field_files" / "field.glb"
    _write_glb(source, uses_draco=False)

    cache = DracoAssetCache(
        assets_dir=assets_dir,
        cache_dir=tmp_path / "cache",
        gltf_transform_bin=tmp_path / "gltf-transform",
    )
    compressor_calls: list[Path] = []

    def fake_run_compressor(source_path: Path, output_path: Path) -> None:
        compressor_calls.append(source_path)
        _write_glb(output_path, uses_draco=True)

    monkeypatch.setattr(cache, "_is_compressor_available", lambda: True)
    monkeypatch.setattr(cache, "_gltf_transform_version", lambda: "fake-version")
    monkeypatch.setattr(cache, "_run_compressor", fake_run_compressor)

    resolved_path = cache.resolve_asset("fields/2025/field_files/field.glb")
    assert resolved_path == tmp_path / "cache" / "fields" / "2025" / "field_files" / "field.glb"
    assert cache.asset_uses_draco(resolved_path)
    assert compressor_calls == [source.resolve()]

    assert cache.resolve_asset("fields/2025/field_files/field.glb") == resolved_path
    assert compressor_calls == [source.resolve()]


def test_resolve_asset_rejects_path_traversal(tmp_path: Path) -> None:
    cache = DracoAssetCache(
        assets_dir=tmp_path / "assets",
        cache_dir=tmp_path / "cache",
        gltf_transform_bin=tmp_path / "gltf-transform",
    )

    assert cache.resolve_asset("../secret.glb") is None
