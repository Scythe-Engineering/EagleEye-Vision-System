from __future__ import annotations

import subprocess

from src.webui.web_server_utils.network_manager_mixin import NetworkManagerMixin


class DummyNetworkManager(NetworkManagerMixin):
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.calls: list[tuple[list[str], float]] = []
        self.logged: list[str] = []

    def log(self, message: str) -> None:
        self.logged.append(message)

    def _run_nmcli(
        self,
        args: list[str],
        timeout: float = 15.0,
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append((args, timeout))
        return subprocess.CompletedProcess(
            ["nmcli", *args],
            self.returncode,
            stdout=self.stdout,
            stderr=self.stderr,
        )


class SequencedNetworkManager(NetworkManagerMixin):
    def __init__(self, results: list[subprocess.CompletedProcess[str]]):
        self.results = results
        self.calls: list[tuple[list[str], float]] = []

    def log(self, message: str) -> None:
        pass

    def _run_nmcli(
        self,
        args: list[str],
        timeout: float = 15.0,
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append((args, timeout))
        return self.results.pop(0)


def test_parse_nmcli_fields_handles_escaped_colons() -> None:
    manager = DummyNetworkManager()

    assert manager._parse_nmcli_fields(r"*:Team\:3322:95:WPA2:AA\:BB") == [
        "*",
        "Team:3322",
        "95",
        "WPA2",
        "AA:BB",
    ]


def test_scan_wifi_networks_merges_duplicate_ssids_and_sorts_connected_first() -> None:
    manager = DummyNetworkManager(
        stdout="\n".join(
            [
                r":Practice:30:WPA2:11\:22",
                r"*:RobotNet:60:WPA2:22\:33",
                r":Practice:80:WPA2:33\:44",
                r":OpenPit:55::44\:55",
            ]
        )
    )

    networks = manager._scan_wifi_networks()

    assert [network["ssid"] for network in networks] == [
        "RobotNet",
        "Practice",
        "OpenPit",
    ]
    assert networks[0]["connected"] is True
    assert networks[1]["signal"] == 80
    assert networks[1]["bssids"] == ["11:22", "33:44"]
    assert networks[2]["security"] == "Open"


def test_scan_wifi_networks_reports_nmcli_failure() -> None:
    manager = DummyNetworkManager(stderr="scan failed", returncode=10)

    payload, status = manager.get_wifi_networks()

    assert status == 500
    assert payload["error"] == "scan failed"


def test_disconnect_active_wifi_devices_discovers_connected_wifi_device() -> None:
    manager = SequencedNetworkManager(
        [
            subprocess.CompletedProcess(
                ["nmcli"],
                0,
                stdout="wlan0:wifi:connected\neth0:ethernet:connected\n",
                stderr="",
            ),
            subprocess.CompletedProcess(
                ["nmcli"],
                0,
                stdout="Device 'wlan0' successfully disconnected.\n",
                stderr="",
            ),
        ]
    )

    disconnected, message = manager._disconnect_active_wifi_devices()

    assert disconnected is True
    assert message == "Disconnected devices: wlan0"
    assert manager.calls == [
        (["-t", "-f", "DEVICE,TYPE,STATE", "device", "status"], 10.0),
        (["device", "disconnect", "wlan0"], 15.0),
    ]


def test_network_manager_status_requires_linux(monkeypatch) -> None:
    manager = DummyNetworkManager()

    monkeypatch.setattr(
        "src.webui.web_server_utils.network_manager_mixin.sys.platform",
        "darwin",
    )

    payload, status = manager.network_manager_status()

    assert status == 200
    assert payload["available"] is False
    assert payload["requires"] == "linux"
