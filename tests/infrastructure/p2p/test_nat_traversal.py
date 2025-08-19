import socket as real_socket

from packages.p2p.core.nat_traversal import NATInfo, NATTraversal, NATType


class DummyUDPSocket:
    def __init__(self):
        self.sent = []

    def settimeout(self, timeout):
        pass

    def sendto(self, data, addr):
        self.sent.append((data, addr))

    def close(self):
        pass


class DummyTCPSocket:
    def __init__(self):
        self.connected = []

    def settimeout(self, timeout):
        pass

    def connect(self, addr):
        self.connected.append(addr)

    def close(self):
        pass


def test_full_cone_hole_punching(monkeypatch):
    trav = NATTraversal()
    monkeypatch.setattr(trav, "detect_nat", lambda: NATInfo("1.2.3.4", 1234, NATType.FULL_CONE))

    def socket_factory(_family, sock_type):
        assert sock_type == real_socket.SOCK_DGRAM
        return DummyUDPSocket()

    import packages.p2p.core.nat_traversal as mod

    monkeypatch.setattr(mod, "socket", real_socket)
    monkeypatch.setattr(mod.socket, "socket", socket_factory)

    assert trav.connect("5.6.7.8", 7777)


def test_symmetric_nat_uses_relay(monkeypatch):
    relays = [("relay.example.com", 9000)]
    trav = NATTraversal(relay_servers=relays)
    monkeypatch.setattr(trav, "detect_nat", lambda: NATInfo("9.9.9.9", 9999, NATType.SYMMETRIC))
    attempts: list[int] = []

    def predict(base, attempt):
        attempts.append(attempt)
        return base + attempt

    monkeypatch.setattr(trav, "_predict_port", predict)

    class FailingUDPSocket(DummyUDPSocket):
        def sendto(self, _data, _addr):
            raise OSError

    class RelaySocket(DummyTCPSocket):
        pass

    def socket_factory(_family, sock_type):
        if sock_type == real_socket.SOCK_DGRAM:
            return FailingUDPSocket()
        return RelaySocket()

    import packages.p2p.core.nat_traversal as mod

    monkeypatch.setattr(mod, "socket", real_socket)
    monkeypatch.setattr(mod.socket, "socket", socket_factory)

    assert trav.connect("8.8.8.8", 8000)
    assert attempts == [0, 1, 2]


def test_port_restricted_success(monkeypatch):
    trav = NATTraversal()
    monkeypatch.setattr(
        trav,
        "detect_nat",
        lambda: NATInfo("1.1.1.1", 1111, NATType.PORT_RESTRICTED_CONE),
    )

    def socket_factory(_family, sock_type):
        if sock_type == real_socket.SOCK_DGRAM:
            return DummyUDPSocket()
        raise AssertionError

    import packages.p2p.core.nat_traversal as mod

    monkeypatch.setattr(mod, "socket", real_socket)
    monkeypatch.setattr(mod.socket, "socket", socket_factory)

    assert trav.connect("2.2.2.2", 2222)
