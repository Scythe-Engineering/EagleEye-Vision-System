"""Camera hardware controls: driver enumeration, live changes and saved settings."""

from __future__ import annotations

import json
from types import SimpleNamespace

from src.utils.camera_utils.camera_config_manager import CameraConfigRegistry
from src.utils.camera_utils.cameras import v4l2_controls
from src.utils.camera_utils.cameras.physical_camera import PhysicalCamera
from src.utils.camera_utils.cameras.v4l2_capture import V4l2Capture
from src.webui.web_server_utils import camera_controls_mixin
from src.webui.web_server_utils.camera_config_mixin import CameraConfigMixin
from src.webui.web_server_utils.camera_controls_mixin import CameraControlsMixin

EXPOSURE_ID = 0x100
AUTO_EXPOSURE_ID = 0x200
CLASS_ID = 0x300
GAIN_ID = 0x400
AUTO_MANUAL, AUTO_APERTURE_PRIORITY = 1, 3


class FakeDriver:
    """A UVC-like driver where manual exposure only works with auto exposure off.

    Exposure is listed before auto exposure, as focus_absolute is before
    focus_automatic_continuous on real UVC cameras.
    """

    def __init__(self) -> None:
        self.values = {
            EXPOSURE_ID: 157,
            AUTO_EXPOSURE_ID: AUTO_APERTURE_PRIORITY,
            GAIN_ID: 0,
        }

    def _describe(self, query) -> None:
        auto = self.values[AUTO_EXPOSURE_ID] == AUTO_APERTURE_PRIORITY
        query.flags = 0
        if query.id == EXPOSURE_ID:
            query.type, query.name = 1, b"Exposure Time, Absolute"
            query.minimum, query.maximum, query.step = 1, 5000, 1
            query.default_value = 157
            query.flags = v4l2_controls.CTRL_FLAG_INACTIVE if auto else 0
        elif query.id == AUTO_EXPOSURE_ID:
            query.type, query.name = 3, b"Auto Exposure"
            query.minimum, query.maximum, query.step = 0, 3, 1
            query.default_value = AUTO_APERTURE_PRIORITY
        elif query.id == CLASS_ID:
            query.type, query.name = 6, b"Camera Controls"
        else:
            query.type, query.name = 1, b"Gain"
            query.minimum, query.maximum, query.step = 0, 100, 1
            query.default_value = 0

    def ioctl(self, request: int, argument) -> None:
        ids = [EXPOSURE_ID, AUTO_EXPOSURE_ID, CLASS_ID, GAIN_ID]
        if request == v4l2_controls.VIDIOC_QUERYCTRL:
            requested = argument.id & ~v4l2_controls.CTRL_FLAG_NEXT_CTRL
            following = [i for i in ids if i > requested]
            if not following:
                raise OSError(22, "end of controls")
            argument.id = following[0]
            self._describe(argument)
        elif request == v4l2_controls.VIDIOC_QUERYMENU:
            labels = {AUTO_MANUAL: b"Manual Mode", AUTO_APERTURE_PRIORITY: b"Aperture Priority Mode"}
            if argument.index not in labels:
                raise OSError(22, "not a menu entry")
            argument.item.name = labels[argument.index]
        elif request == v4l2_controls.VIDIOC_G_CTRL:
            argument.value = self.values[argument.id]
        elif request == v4l2_controls.VIDIOC_S_CTRL:
            if (
                argument.id == EXPOSURE_ID
                and self.values[AUTO_EXPOSURE_ID] == AUTO_APERTURE_PRIORITY
            ):
                raise OSError(13, "inactive while auto exposure is on")
            self.values[argument.id] = argument.value
        else:
            raise AssertionError(f"unexpected ioctl {request:#x}")


def _camera(driver: FakeDriver) -> PhysicalCamera:
    """A PhysicalCamera wired to the fake driver, without opening a device."""
    capture = V4l2Capture.__new__(V4l2Capture)
    capture.ioctl = driver.ioctl
    camera = PhysicalCamera.__new__(PhysicalCamera)
    camera.backend = capture
    return camera


def test_controls_are_listed_with_keys_menus_and_auto_state() -> None:
    """Every driver control except class headers is listed in driver order."""
    controls = v4l2_controls.list_controls(FakeDriver().ioctl)

    assert [c.key for c in controls] == [
        "exposure_time_absolute",
        "auto_exposure",
        "gain",
    ]
    exposure, auto, _ = controls
    assert exposure.inactive and exposure.value == 157
    assert auto.type == "menu"
    assert auto.menu == [
        {"value": 1, "label": "Manual Mode"},
        {"value": 3, "label": "Aperture Priority Mode"},
    ]


def test_set_controls_retries_values_blocked_by_an_auto_mode() -> None:
    """Manual exposure lands even though it is applied before auto exposure turns off."""
    driver = FakeDriver()

    refused = _camera(driver).set_controls(
        {"exposure_time_absolute": 10, "auto_exposure": AUTO_MANUAL, "zoom": 2}
    )

    assert refused == ["zoom"]
    assert driver.values[EXPOSURE_ID] == 10
    assert driver.values[AUTO_EXPOSURE_ID] == AUTO_MANUAL


class _ControlsServer(CameraControlsMixin, CameraConfigMixin):
    """Just the web server parts the control routes use."""

    def __init__(self, camera: PhysicalCamera, registry: CameraConfigRegistry) -> None:
        self.camera_config_registry = registry
        self.camera_manager = type(
            "Manager", (), {"get_camera_by_bus_id": lambda _self, bus_id: camera}
        )()
        self.log = lambda _message: None


def _post(monkeypatch, payload) -> None:
    """Make the next handler call see ``payload`` as its JSON body."""
    monkeypatch.setattr(
        camera_controls_mixin,
        "request",
        SimpleNamespace(get_json=lambda silent=True: payload),
    )


def test_settings_change_live_and_are_saved_for_startup(tmp_path, monkeypatch) -> None:
    """A live change is saved on request and comes back from the camera config."""
    server = _ControlsServer(_camera(FakeDriver()), CameraConfigRegistry(str(tmp_path)))

    _post(monkeypatch, {"auto_exposure": AUTO_MANUAL, "exposure_time_absolute": 10})
    applied, applied_status = server.set_camera_controls("1-1")
    saved, saved_status = server.save_camera_controls("1-1")

    assert applied_status == 200 and applied["refused"] == []
    assert saved_status == 200
    expected = {"exposure_time_absolute": 10, "auto_exposure": AUTO_MANUAL, "gain": 0}
    assert saved["saved"] == expected
    assert json.loads((tmp_path / "1-1" / "controls.json").read_text()) == expected
    assert CameraConfigRegistry(str(tmp_path)).get_config("1-1").camera_controls == (
        expected
    )
    _post(monkeypatch, {"gain": "high"})
    assert server.set_camera_controls("1-1")[1] == 400
