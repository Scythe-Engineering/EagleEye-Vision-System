"""Temporal ROI segments whose rectified crop is warped only when it is read."""

from __future__ import annotations

import cv2
import numpy as np


class LazySegment:
    """A `(crop, mapping)` pair that warps its crop from the frame on first access.

    Unpacks and indexes like the tuple it replaces, so `for crop, mapping in
    segments` keeps working. Detectors can read `shape` and `mapping` without
    building the crop, e.g. for segments a tracker already covers.

    Deliberately neither a tuple nor a registered `Sequence`: the pipeline's timing
    utilities rebuild those item by item, which would build every crop.
    """

    __slots__ = ("_destination", "_frame", "_image", "_source", "mapping", "shape")

    def __init__(
        self,
        frame: np.ndarray,
        source: np.ndarray,
        destination: np.ndarray,
        mapping: np.ndarray,
    ) -> None:
        """Describe a crop without warping it.

        Args:
            frame: Full frame the crop is sampled from; must not change while the
                segment can still be read.
            source: Shape-(4, 2) float32 full-frame quad.
            destination: Shape-(4, 2) float32 square crop corners matching `source`.
            mapping: Shape-(3, 3) transform from crop to full-frame coordinates.
        """
        side = int(destination.max()) + 1
        self.mapping = mapping
        self.shape = (side, side) + frame.shape[2:]
        self._frame = frame
        self._source = source
        self._destination = destination
        self._image: np.ndarray | None = None

    @property
    def image(self) -> np.ndarray:
        """The rectified crop, warped from the frame once."""
        if self._image is None:
            side = self.shape[0]
            crop_from_frame = cv2.getPerspectiveTransform(
                self._source, self._destination
            )
            self._image = cv2.warpPerspective(
                self._frame, crop_from_frame, (side, side)
            )
        return self._image

    def __iter__(self):
        yield self.image
        yield self.mapping

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        return (self.image, self.mapping)[index]


def segment_shape_and_mapping(segment) -> tuple[tuple[int, ...], np.ndarray]:
    """Return a segment's image shape and full-frame mapping without building it.

    Args:
        segment: A `LazySegment` or a `(image, mapping)` pair.
    """
    if isinstance(segment, LazySegment):
        return segment.shape, segment.mapping
    image, mapping = segment
    return getattr(image, "shape", ()), mapping
