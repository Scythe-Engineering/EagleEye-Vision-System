"""Enumerate, read and write a V4L2 device's user controls.

Controls are listed with ``VIDIOC_QUERYCTRL`` and the next-control flag, so
whatever the driver exposes (exposure, gain, white balance, focus, ...) shows
up without a per-camera table. Each control gets a stable ``key`` derived from
its driver name the way ``v4l2-ctl`` does ("Exposure Time, Absolute" becomes
``exposure_time_absolute``), which is what saved settings are keyed by.
"""

from __future__ import annotations

import ctypes
import errno
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

from src.utils.camera_utils.cameras.v4l2_capture import (
    VIDIOC_S_CTRL,
    _Control,
    _iowr,
)

Ioctl = Callable[[int, ctypes.Structure], None]


class _QueryControl(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("name", ctypes.c_char * 32),
        ("minimum", ctypes.c_int32),
        ("maximum", ctypes.c_int32),
        ("step", ctypes.c_int32),
        ("default_value", ctypes.c_int32),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 2),
    ]


class _QueryMenuUnion(ctypes.Union):
    _fields_ = [("name", ctypes.c_char * 32), ("value", ctypes.c_int64)]


class _QueryMenu(ctypes.Structure):
    # ``struct v4l2_querymenu`` is declared packed: 44 bytes, no tail padding.
    _pack_ = 1
    _fields_ = [
        ("id", ctypes.c_uint32),
        ("index", ctypes.c_uint32),
        ("item", _QueryMenuUnion),
        ("reserved", ctypes.c_uint32),
    ]


VIDIOC_G_CTRL = _iowr("V", 27, ctypes.sizeof(_Control))
VIDIOC_QUERYCTRL = _iowr("V", 36, ctypes.sizeof(_QueryControl))
VIDIOC_QUERYMENU = _iowr("V", 37, ctypes.sizeof(_QueryMenu))

CTRL_FLAG_NEXT_CTRL = 0x80000000
CTRL_FLAG_DISABLED = 0x0001
CTRL_FLAG_READ_ONLY = 0x0004
CTRL_FLAG_INACTIVE = 0x0010
CTRL_FLAG_WRITE_ONLY = 0x0040

CTRL_TYPE_INTEGER = 1
CTRL_TYPE_BOOLEAN = 2
CTRL_TYPE_MENU = 3
CTRL_TYPE_INTEGER_MENU = 9
_TYPE_NAMES = {
    CTRL_TYPE_INTEGER: "integer",
    CTRL_TYPE_BOOLEAN: "boolean",
    CTRL_TYPE_MENU: "menu",
    CTRL_TYPE_INTEGER_MENU: "menu",
}


@dataclass
class CameraControl:
    """One adjustable camera control as the driver describes it."""

    id: int
    key: str
    name: str
    type: str
    minimum: int
    maximum: int
    step: int
    default: int
    value: int | None
    inactive: bool
    read_only: bool
    menu: list[dict[str, int | str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return the control as JSON-ready data."""
        return asdict(self)


def control_key(name: str) -> str:
    """Return the ``v4l2-ctl`` style key for a driver control name."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _menu_items(ioctl: Ioctl, query: _QueryControl) -> list[dict[str, int | str]]:
    """Return the selectable entries of a menu control."""
    items: list[dict[str, int | str]] = []
    for index in range(query.minimum, query.maximum + 1):
        menu = _QueryMenu(id=query.id, index=index)
        try:
            ioctl(VIDIOC_QUERYMENU, menu)
        except OSError:
            continue  # Drivers leave gaps in menus; skipped indexes are invalid.
        label = (
            menu.item.name.decode(errors="replace")
            if query.type == CTRL_TYPE_MENU
            else str(menu.item.value)
        )
        items.append({"value": index, "label": label})
    return items


def get_control(ioctl: Ioctl, control_id: int) -> int | None:
    """Read a control's current value, or None when the driver refuses."""
    control = _Control(id=control_id)
    try:
        ioctl(VIDIOC_G_CTRL, control)
    except OSError:
        return None
    return int(control.value)


def set_control(ioctl: Ioctl, control_id: int, value: int) -> bool:
    """Write a control. Returns whether the driver accepted it."""
    if not -(2**31) <= value < 2**31:
        return False  # ctypes would silently wrap it into the int32 field.
    try:
        ioctl(VIDIOC_S_CTRL, _Control(id=control_id, value=int(value)))
    except OSError:
        return False
    return True


def list_controls(ioctl: Ioctl) -> list[CameraControl]:
    """Enumerate the device's integer, boolean and menu controls in driver order."""
    controls: list[CameraControl] = []
    keys: set[str] = set()
    next_id = CTRL_FLAG_NEXT_CTRL
    while True:
        query = _QueryControl(id=next_id)
        try:
            ioctl(VIDIOC_QUERYCTRL, query)
        except OSError as error:
            if error.errno == errno.EINVAL:
                return controls  # EINVAL marks the end of the list.
            raise
        next_id = query.id | CTRL_FLAG_NEXT_CTRL
        if query.flags & CTRL_FLAG_DISABLED or query.type not in _TYPE_NAMES:
            continue
        name = query.name.decode(errors="replace")
        key = control_key(name)
        if key in keys:
            key = f"{key}_{query.id:x}"  # Keep keys unique if two names normalize alike.
        keys.add(key)
        write_only = bool(query.flags & CTRL_FLAG_WRITE_ONLY)
        controls.append(
            CameraControl(
                id=int(query.id),
                key=key,
                name=name,
                type=_TYPE_NAMES[query.type],
                minimum=int(query.minimum),
                maximum=int(query.maximum),
                step=max(1, int(query.step)),
                default=int(query.default_value),
                value=None if write_only else get_control(ioctl, query.id),
                inactive=bool(query.flags & CTRL_FLAG_INACTIVE),
                read_only=bool(query.flags & CTRL_FLAG_READ_ONLY),
                menu=_menu_items(ioctl, query)
                if query.type in (CTRL_TYPE_MENU, CTRL_TYPE_INTEGER_MENU)
                else [],
            )
        )
