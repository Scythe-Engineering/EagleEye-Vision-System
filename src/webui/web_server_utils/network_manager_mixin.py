from __future__ import annotations

import shutil
import subprocess
import sys
from typing import Any


def _request():
    """Return the current Flask request, allowing monkeypatching via web_server.request."""
    import src.webui.web_server as _ws

    return _ws.request


class NetworkManagerMixin:
    def network_manager_status(self) -> tuple[dict, int]:
        """Return whether backend WiFi management can run on this host."""
        is_linux = sys.platform.startswith("linux")
        return {
            "available": is_linux,
            "platform": sys.platform,
            "requires": "linux",
            "nmcli_available": self._nmcli_available(),
        }, 200

    def _nmcli_available(self) -> bool:
        return shutil.which("nmcli") is not None

    def _run_nmcli(
        self,
        args: list[str],
        timeout: float = 15.0,
    ) -> subprocess.CompletedProcess[str]:
        if not self._nmcli_available():
            raise FileNotFoundError("nmcli is not installed")

        return subprocess.run(
            ["nmcli", *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def _parse_nmcli_fields(self, line: str) -> list[str]:
        fields: list[str] = []
        current: list[str] = []
        escaped = False

        for char in line:
            if escaped:
                current.append(char)
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == ":":
                fields.append("".join(current))
                current = []
                continue
            current.append(char)

        if escaped:
            current.append("\\")
        fields.append("".join(current))
        return fields

    def _scan_wifi_networks(self) -> list[dict[str, Any]]:
        result = self._run_nmcli(
            [
                "-t",
                "-f",
                "IN-USE,SSID,SIGNAL,SECURITY,BSSID",
                "device",
                "wifi",
                "list",
                "--rescan",
                "yes",
            ],
            timeout=20.0,
        )
        if result.returncode != 0:
            raise RuntimeError(
                result.stderr.strip() or "Failed to scan WiFi networks"
            )

        networks_by_ssid: dict[str, dict[str, Any]] = {}
        for line in result.stdout.splitlines():
            if not line.strip():
                continue

            fields = self._parse_nmcli_fields(line)
            if len(fields) < 5:
                continue

            in_use, ssid, signal, security, bssid = fields[:5]
            ssid = ssid.strip()
            if not ssid:
                continue

            try:
                signal_value = int(signal)
            except ValueError:
                signal_value = 0

            network = {
                "ssid": ssid,
                "signal": signal_value,
                "security": security.strip() or "Open",
                "connected": in_use.strip() == "*",
                "bssids": [bssid.strip()] if bssid.strip() else [],
            }

            existing = networks_by_ssid.get(ssid)
            if existing is None or signal_value > int(existing.get("signal", 0)):
                if existing is not None:
                    network["bssids"] = sorted(
                        set(existing.get("bssids", [])) | set(network["bssids"])
                    )
                    network["connected"] = (
                        bool(existing.get("connected")) or network["connected"]
                    )
                networks_by_ssid[ssid] = network
            else:
                existing["connected"] = (
                    bool(existing.get("connected")) or network["connected"]
                )
                existing["bssids"] = sorted(
                    set(existing.get("bssids", [])) | set(network["bssids"])
                )

        return sorted(
            networks_by_ssid.values(),
            key=lambda network: (
                not bool(network.get("connected")),
                -int(network.get("signal", 0)),
                str(network.get("ssid", "")).lower(),
            ),
        )

    def _disconnect_active_wifi_devices(self) -> tuple[bool, str]:
        result = self._run_nmcli(
            ["-t", "-f", "DEVICE,TYPE,STATE", "device", "status"],
            timeout=10.0,
        )
        if result.returncode != 0:
            return False, result.stderr.strip()

        disconnected_devices: list[str] = []
        errors: list[str] = []
        for line in result.stdout.splitlines():
            fields = self._parse_nmcli_fields(line)
            if len(fields) < 3:
                continue
            device, connection_type, state = fields[:3]
            if connection_type != "wifi" or state != "connected":
                continue

            disconnect_result = self._run_nmcli(
                ["device", "disconnect", device],
                timeout=15.0,
            )
            if disconnect_result.returncode == 0:
                disconnected_devices.append(device)
            else:
                errors.append(disconnect_result.stderr.strip())

        if disconnected_devices:
            return True, f"Disconnected devices: {', '.join(disconnected_devices)}"
        return False, "; ".join(error for error in errors if error)

    def get_wifi_networks(self) -> tuple[dict, int]:
        """Return WiFi networks visible to the backend host."""
        try:
            return {"networks": self._scan_wifi_networks()}, 200
        except FileNotFoundError:
            return {
                "error": "Network management is unavailable because nmcli is not installed",
                "networks": [],
            }, 503
        except subprocess.TimeoutExpired:
            return {"error": "WiFi scan timed out", "networks": []}, 504
        except Exception as error:
            self.log(f"Error scanning WiFi networks: {error}")
            return {"error": str(error), "networks": []}, 500

    def connect_wifi_network(self) -> tuple[dict, int]:
        """Connect to a visible WiFi network by SSID."""
        body = _request().get_json(silent=True) or {}
        ssid = str(body.get("ssid", "")).strip()
        password = str(body.get("password", ""))

        if not ssid:
            return {"error": "SSID is required"}, 400

        args = ["device", "wifi", "connect", ssid]
        if password:
            args.extend(["password", password])

        try:
            result = self._run_nmcli(args, timeout=30.0)
            if result.returncode != 0:
                return {
                    "error": result.stderr.strip() or "Failed to connect to WiFi network",
                }, 400
            self.log(f"Connected to WiFi network {ssid}")
            return {
                "success": True,
                "ssid": ssid,
                "message": result.stdout.strip(),
            }, 200
        except FileNotFoundError:
            return {
                "error": "Network management is unavailable because nmcli is not installed"
            }, 503
        except subprocess.TimeoutExpired:
            return {"error": "WiFi connection attempt timed out"}, 504
        except Exception as error:
            self.log(f"Error connecting to WiFi network {ssid}: {error}")
            return {"error": "Failed to connect to WiFi network"}, 500

    def disconnect_wifi_network(self) -> tuple[dict, int]:
        """Disconnect from a WiFi network by connection name/SSID."""
        body = _request().get_json(silent=True) or {}
        ssid = str(body.get("ssid", "")).strip()

        if not ssid:
            return {"error": "SSID is required"}, 400

        try:
            result = self._run_nmcli(
                ["connection", "down", "id", ssid],
                timeout=15.0,
            )
            if result.returncode != 0:
                disconnected, fallback_message = self._disconnect_active_wifi_devices()
                if not disconnected:
                    return {
                        "error": (
                            result.stderr.strip()
                            or fallback_message
                            or "Failed to disconnect from WiFi network"
                        ),
                    }, 400
                self.log(f"Disconnected active WiFi device for network {ssid}")
                return {
                    "success": True,
                    "ssid": ssid,
                    "message": fallback_message,
                }, 200
            self.log(f"Disconnected from WiFi network {ssid}")
            return {
                "success": True,
                "ssid": ssid,
                "message": result.stdout.strip(),
            }, 200
        except FileNotFoundError:
            return {
                "error": "Network management is unavailable because nmcli is not installed"
            }, 503
        except subprocess.TimeoutExpired:
            return {"error": "WiFi disconnect timed out"}, 504
        except Exception as error:
            self.log(f"Error disconnecting from WiFi network {ssid}: {error}")
            return {"error": "Failed to disconnect from WiFi network"}, 500
