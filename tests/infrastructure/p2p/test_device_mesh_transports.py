import pytest

from packages.p2p.core.device_mesh import DeviceMesh
from packages.p2p.core.p2p_node import P2PNode


class DummyBleakClient:
    def __init__(self, address):
        self.address = address
        self.is_connected = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class DummyIface:
    def remove_all_network_profiles(self):
        pass

    def add_network_profile(self, profile):
        return profile

    def connect(self, profile):
        pass

    def disconnect(self):
        pass

    def status(self):
        return DummyConst.IFACE_CONNECTED


class DummyPyWiFi:
    def interfaces(self):
        return [DummyIface()]


class DummyConst:
    AUTH_ALG_OPEN = 0
    AKM_TYPE_NONE = 0
    CIPHER_TYPE_NONE = 0
    IFACE_CONNECTED = 1


class DummyProfile:
    def __init__(self):
        self.ssid = ""
        self.auth = None
        self.akm = []
        self.cipher = None


@pytest.mark.asyncio
async def test_try_bluetooth_connect(monkeypatch):
    """Ensure bluetooth connection uses bleak when enabled."""

    def fake_bt_dep(self):
        self._bleak_client = DummyBleakClient
        return True

    monkeypatch.setattr(DeviceMesh, "_check_bluetooth", lambda _: True)
    monkeypatch.setattr(DeviceMesh, "_check_bluetooth_dependencies", fake_bt_dep)
    monkeypatch.setattr(DeviceMesh, "_check_wifi_direct", lambda _: False)

    mesh = DeviceMesh(P2PNode())
    result = await mesh._try_bluetooth_connect("AA:BB:CC:DD:EE:FF")
    assert result == "bluetooth"


@pytest.mark.asyncio
async def test_try_wifi_direct_connect(monkeypatch):
    """Ensure wifi direct connection uses pywifi when enabled."""

    def fake_wifi_dep(self):
        self._pywifi_cls = DummyPyWiFi
        self._wifi_const = DummyConst
        self._wifi_profile_cls = DummyProfile
        return True

    monkeypatch.setattr(DeviceMesh, "_check_wifi_direct", lambda _: True)
    monkeypatch.setattr(DeviceMesh, "_check_wifi_direct_dependencies", fake_wifi_dep)
    monkeypatch.setattr(DeviceMesh, "_check_bluetooth", lambda _: False)

    mesh = DeviceMesh(P2PNode())
    result = await mesh._try_wifi_direct_connect("test-ssid")
    assert result == "wifi_direct"
