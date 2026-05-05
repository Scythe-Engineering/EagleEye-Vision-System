import json

from src.webui.web_server_utils.apriltag_map_sanitizer import sanitize_apriltag_map_file


def test_sanitize_apriltag_map_file_replaces_null_transform_values(tmp_path):
    map_path = tmp_path / "fiducials.fmap"
    map_path.write_text(
        json.dumps(
            {
                "fiducials": [
                    {
                        "id": 1,
                        "transform": [1, 0, 0, None, 0, 1, 0, 2, 0, 0, 1, 3, 0, 0, 0, 1],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    fixes = sanitize_apriltag_map_file(map_path)

    data = json.loads(map_path.read_text(encoding="utf-8"))
    assert fixes == 1
    assert data["fiducials"][0]["transform"][3] == 0.0


def test_sanitize_apriltag_map_file_replaces_invalid_transform_with_identity(tmp_path):
    map_path = tmp_path / "fiducials.fmap"
    map_path.write_text(
        json.dumps({"fiducials": [{"id": 1, "transform": None}]}),
        encoding="utf-8",
    )

    fixes = sanitize_apriltag_map_file(map_path)

    data = json.loads(map_path.read_text(encoding="utf-8"))
    assert fixes == 1
    assert data["fiducials"][0]["transform"] == [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
