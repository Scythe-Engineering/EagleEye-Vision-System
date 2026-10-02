"""Python-owned cleanup for the pupil-apriltags native detector."""

from pupil_apriltags import Detector as PupilDetector


class Detector(PupilDetector):
    """Keep pupil's constructor/detection API, but own native destruction order."""

    def close(self) -> None:
        """Release the detector before its families, including partial construction."""
        pointer = getattr(self, "tag_detector_ptr", None)
        if pointer is not None:
            # Upstream frees families first, but detector_destroy accesses their
            # userdata to free decoding tables: that order is a use-after-free.
            destroy = self.libc.apriltag_detector_destroy
            destroy.restype = None
            destroy.argtypes = [type(pointer)]
            destroy(pointer)
            self.tag_detector_ptr = None
        families = getattr(self, "tag_families", {})
        for family, pointer in list(families.items()):
            destroy = getattr(self.libc, family + "_destroy")
            destroy.restype = None
            destroy.argtypes = [type(pointer)]
            destroy(pointer)
            del families[family]

    def __del__(self) -> None:
        """Use the same idempotent cleanup when construction fails or GC runs."""
        self.close()
