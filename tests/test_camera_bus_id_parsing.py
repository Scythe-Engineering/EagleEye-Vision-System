from __future__ import annotations

from src.utils.camera_utils.get_available_cameras import _extract_bus_id


def test_extract_bus_id_handles_usb_xhci_format() -> None:
    device_name = "Arducam OV9782 USB Camera: Ardu (usb-xhci-hcd.1-1):"

    assert _extract_bus_id(device_name) == "1-1"


def test_extract_bus_id_handles_standard_usb_format() -> None:
    device_name = "Logitech C920: usb-0000:00:14.0-1:"

    assert _extract_bus_id(device_name) == "1"


def test_extract_bus_id_returns_unknown_when_no_usb_token_exists() -> None:
    device_name = "Integrated Camera: platform:abcd"

    assert _extract_bus_id(device_name) == "unknown"
