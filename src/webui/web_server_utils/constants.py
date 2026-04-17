from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

WEBUI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.dirname(WEBUI_DIR)

with open(os.path.join(WEBUI_DIR, "assets", "no_image.png"), "rb") as _f:
    _no_image_bytes = _f.read()

no_image = cv2.imdecode(np.frombuffer(_no_image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
_success, _noimg_jpeg = cv2.imencode(".jpg", no_image)
no_image_jpeg_bytes: bytes = _noimg_jpeg.tobytes() if _success else b""

CORS_ALLOWED_ORIGINS = "*"
PIPELINE_NOT_FOUND_MESSAGE = "Pipeline not found"
TEXT_PLAIN_MIMETYPE = "text/plain"
VISUALIZATION_STREAM_FPS = 12
PROFILING_PUBLISH_INTERVAL_SECONDS = 0.3
PIPELINE_ERROR_PUBLISH_FRAME_INTERVAL = 10
PIPELINE_ERROR_FALLBACK_PUBLISH_INTERVAL_SECONDS = 1.0
SSE_SERIALIZATION_WARN_INTERVAL_SECONDS = 5.0
WEB_SERVER_HOST = "0.0.0.0"
WEB_SERVER_PORT = 5001
TEST_VIDEO_EXTENSION = ".mp4"
MODEL_ASSET_EXTENSION = ".glb"
MODEL_ASSET_METADATA_SUFFIX = ".metadata.json"
ROBOT_ASSET_DIR_NAME = "robots"
FIELD_ASSET_DIR_NAME = "fields"
FIELD_FILE_DIR_NAME = "field_files"
FIELD_APRILTAG_MAP_DIR_NAME = "apriltag_maps"
APRILTAG_MAP_EXTENSIONS = {".fmap", ".json"}
ASSET_SCALE_KEY = "scale"
DEFAULT_ASSET_SCALE = 1.0
GENERAL_CONF_PATH = Path(SRC_DIR) / "general_conf.json"
VIEW_STREAM_DOWNSCALE_KEY = "view_stream_downscale"
DEFAULT_VIEW_STREAM_DOWNSCALE = 0.5
MIN_VIEW_STREAM_DOWNSCALE = 0.1
MAX_VIEW_STREAM_DOWNSCALE = 1.0
VIEW_STREAM_JPEG_QUALITY = 70
DEFAULT_GENERAL_CONF: dict[str, Any] = {
    "network_table_address": "0.0.0.0",
    VIEW_STREAM_DOWNSCALE_KEY: DEFAULT_VIEW_STREAM_DOWNSCALE,
}
