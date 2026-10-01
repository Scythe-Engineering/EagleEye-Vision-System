"""Rust operation adapter, with explicit ABI 2 libraries for frozen comparisons."""

from __future__ import annotations

import ctypes as ct
import math
from contextlib import suppress
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
from pupil_apriltags import Detection


class _Config(ct.Structure):
    _fields_ = [
        ("abi_version", ct.c_uint32),
        ("struct_size", ct.c_uint32),
        ("families", ct.c_char_p),
        ("nthreads", ct.c_int32),
        ("refine_edges", ct.c_int32),
        ("quad_decimate", ct.c_double),
        ("quad_sigma", ct.c_double),
        ("decode_sharpening", ct.c_double),
    ]


class _Detection(ct.Structure):
    _fields_ = [
        ("family_index", ct.c_uint32),
        ("tag_id", ct.c_int32),
        ("hamming", ct.c_int32),
        ("rotation", ct.c_int32),
        ("decision_margin", ct.c_double),
        ("corners", ct.c_double * 8),
        ("center", ct.c_double * 2),
        ("homography", ct.c_double * 9),
    ]


class CustomNativeDetector:
    """Own one non-reentrant native handle; callers must serialize detect/close."""

    def __init__(
        self,
        *,
        library: str = "",
        families: str = "tag36h11",
        nthreads: int = 1,
        quad_decimate: float = 2.0,
        quad_sigma: float = 0.0,
        refine_edges: int = 1,
        decode_sharpening: float = 0.25,
    ) -> None:
        """Use the Rust operation by default; explicit libraries must provide ABI 2."""
        self._handle = ct.c_void_p()
        self._rust_detector: Any | None = None
        if not isinstance(families, str) or not families.split() or "\0" in families:
            raise ValueError("families must contain space-separated family names")
        self._families = [name.encode("ascii") for name in families.split()]
        if len(set(self._families)) != len(self._families):
            raise ValueError("families must not contain duplicates")
        if not 1 <= nthreads <= 2**31 - 1 or int(nthreads) != nthreads:
            raise ValueError("nthreads must be a positive int32")
        if refine_edges not in (0, 1):
            raise ValueError("refine_edges must be 0 or 1")
        if not all(
            math.isfinite(v) for v in (quad_decimate, quad_sigma, decode_sharpening)
        ):
            raise ValueError("decimation, sigma and sharpening must be finite")
        if quad_decimate < 1 or quad_sigma < 0 or decode_sharpening < 0:
            raise ValueError(
                "quad_decimate must be >= 1; sigma and sharpening must be >= 0"
            )
        rust_module = None
        if not library:
            try:
                rust_module = import_module("custom_apriltag_detector")
            except ModuleNotFoundError as exc:
                raise ImportError(
                    "Custom Rust detector is not built. Run: "
                    "uv run python src/rust_implementations/build.py custom_apriltag_detector"
                ) from exc
            library = rust_module.native_library_path()
        path = Path(library)
        if not path.is_absolute() or not path.is_file():
            raise FileNotFoundError(
                "EAGLEEYE_CUSTOM_TAG_LIBRARY must name an existing absolute shared-library path"
            )
        # Keep the real shared-library object for existing benchmark provenance.
        self.libc = ct.CDLL(str(path))
        if rust_module is not None:
            self._rust_detector = rust_module.CustomAprilTagDetector(
                families=" ".join(name.decode("ascii") for name in self._families),
                nthreads=int(nthreads),
                quad_decimate=quad_decimate,
                quad_sigma=quad_sigma,
                refine_edges=int(refine_edges),
                decode_sharpening=decode_sharpening,
            )
            return
        try:
            self.libc.et_abi_version.argtypes = []
            self.libc.et_abi_version.restype = ct.c_uint32
            if self.libc.et_abi_version() != 2:
                raise RuntimeError("Eagle Tags ABI mismatch: expected ABI 2")
            self.libc.et_create.argtypes = [
                ct.POINTER(_Config),
                ct.POINTER(ct.c_void_p),
                ct.c_char_p,
                ct.c_uint32,
            ]
            self.libc.et_create.restype = ct.c_int
            self.libc.et_detect_mapped.argtypes = [
                ct.c_void_p,
                ct.POINTER(ct.c_uint8),
                ct.c_uint32,
                ct.c_uint32,
                ct.c_uint32,
                ct.POINTER(ct.c_double),
                ct.c_uint32,
                ct.c_uint32,
                ct.POINTER(ct.POINTER(_Detection)),
                ct.POINTER(ct.c_uint32),
                ct.c_char_p,
                ct.c_uint32,
            ]
            self.libc.et_detect_mapped.restype = ct.c_int
            self.libc.et_destroy.argtypes = [ct.c_void_p]
            self.libc.et_destroy.restype = None
        except AttributeError as exc:
            raise RuntimeError(
                "Not an Eagle Tags ABI 2 library: missing et_* symbol"
            ) from exc
        config = _Config(
            2,
            ct.sizeof(_Config),
            b" ".join(self._families),
            int(nthreads),
            int(refine_edges),
            quad_decimate,
            quad_sigma,
            decode_sharpening,
        )
        error = ct.create_string_buffer(1024)
        status = self.libc.et_create(
            ct.byref(config), ct.byref(self._handle), error, len(error)
        )
        if status or not self._handle:
            self.close()
            raise ValueError(
                f"Eagle Tags configuration rejected: {error.value.decode(errors='replace')}"
            )

    def detect(
        self,
        gray: np.ndarray,
        *,
        source_map: np.ndarray | None = None,
        source_shape: tuple[int, int] | None = None,
    ) -> list[Detection]:
        """Detect observed pixels; optional map uses OpenCV integer pixel centers.

        Without physical source geometry, the input is intentionally unmasked.
        Callers must keep pixels unchanged until detection finishes.
        """
        if self._rust_detector is None and not self._handle:
            raise RuntimeError("Eagle Tags detector is closed")
        if not isinstance(gray, np.ndarray) or gray.dtype != np.uint8 or gray.ndim != 2:
            raise ValueError("Eagle Tags requires a 2D uint8 grayscale image")
        height, width = gray.shape
        if not 0 < min(height, width) or max(height, width) > 2**32 - 1:
            raise ValueError(
                "Eagle Tags image dimensions must be positive uint32 values"
            )
        if (
            gray.strides[1] != 1
            or gray.strides[0] < width
            or gray.strides[0] > 2**32 - 1
        ):
            gray = np.ascontiguousarray(gray)
        mapping = None
        source_height = source_width = 0
        if (source_map is None) != (source_shape is None):
            raise ValueError("source_map and source_shape must be supplied together")
        if source_map is not None:
            mapping = np.asarray(source_map, dtype=np.float64)
            if mapping.shape == (2,):
                mapping = np.array(
                    [[1.0, 0.0, mapping[0]], [0.0, 1.0, mapping[1]], [0.0, 0.0, 1.0]]
                )
            if (
                mapping.shape != (3, 3)
                or not np.isfinite(mapping).all()
                or np.linalg.det(mapping) == 0
            ):
                raise ValueError(
                    "source_map must be a finite nonsingular 3x3 map or XY offset"
                )
            mapping = np.ascontiguousarray(mapping)
            if (
                source_shape is None
                or len(source_shape) != 2
                or any(
                    not isinstance(v, (int, np.integer)) or not 0 < v <= 2**32 - 1
                    for v in source_shape
                )
            ):
                raise ValueError(
                    "source_shape must contain positive uint32 height and width"
                )
            source_height, source_width = source_shape
        if self._rust_detector is not None:
            native_results = self._rust_detector.run(
                gray,
                source_map=None if mapping is None else mapping.ravel().tolist(),
                source_shape=None
                if mapping is None
                else (int(source_height), int(source_width)),
            )
        else:
            results = ct.POINTER(_Detection)()
            count = ct.c_uint32()
            error = ct.create_string_buffer(1024)
            status = self.libc.et_detect_mapped(
                self._handle,
                gray.ctypes.data_as(ct.POINTER(ct.c_uint8)),
                width,
                height,
                gray.strides[0],
                None
                if mapping is None
                else mapping.ctypes.data_as(ct.POINTER(ct.c_double)),
                source_width,
                source_height,
                ct.byref(results),
                ct.byref(count),
                error,
                len(error),
            )
            if status:
                raise RuntimeError(
                    f"Eagle Tags detection failed: {error.value.decode(errors='replace')}"
                )
            if count.value and not results:
                raise RuntimeError("Eagle Tags returned a null detection array")
            native_results = (results[index] for index in range(count.value))
        detections = []
        for native in native_results:
            if native.family_index >= len(self._families):
                raise RuntimeError("Eagle Tags returned an invalid family index")
            detection = Detection()
            detection.tag_family = self._families[native.family_index]
            detection.tag_id = native.tag_id
            detection.hamming = native.hamming
            detection.decision_margin = native.decision_margin
            detection.corners = np.array(native.corners).reshape(4, 2)
            detection.center = np.array(native.center)
            detection.homography = np.array(native.homography).reshape(3, 3)
            detections.append(detection)
        return detections

    def close(self) -> None:
        """Release this handle exactly once, including after partial construction."""
        if self._rust_detector is not None:
            self._rust_detector.close()
            self._rust_detector = None
        handle = self._handle
        if handle:
            self.libc.et_destroy(handle)
            self._handle = ct.c_void_p()

    def __del__(self) -> None:
        """Best-effort cleanup during partial initialization or interpreter shutdown."""
        with suppress(Exception):
            self.close()
