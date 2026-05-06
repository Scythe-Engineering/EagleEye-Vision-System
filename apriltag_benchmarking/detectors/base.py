from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: float
    height: float

    @property
    def params(self) -> tuple[float, float, float, float]:
        return (self.fx, self.fy, self.cx, self.cy)


@dataclass
class TagDetection:
    tag_family: str
    tag_id: int
    corners: np.ndarray  # shape (4, 2), pixels
    center: np.ndarray  # shape (2,), pixels
    pose_t: np.ndarray | None = None  # shape (3,), meters in OpenCV camera coords
    decision_margin: float | None = None
    hamming: int | None = None


class AprilTagDetectorImplementation(Protocol):
    name: str

    def detect(
        self,
        image_bgr: np.ndarray,
        intrinsics: CameraIntrinsics,
        tag_size_m: float,
    ) -> list[TagDetection]:
        ...
