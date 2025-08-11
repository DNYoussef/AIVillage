import socket as real_socket
import time

from src.infrastructure.p2p.nat_traversal import NATInfo, NATTraversal, NATType


class DummyUDPSocket:
    def __init__(self) -> None:
        self.sent: list[tuple[bytes, tuple[str, int]]] = []

    def settimeout(self, timeout: float) -> None:  # - simple stub
        pass

    def sendto(self, data: bytes, addr: tuple[str, int]) -> None:
        self.sent.append((data, addr))

    def close(self) -> None:  # pragma: no cover - nothing to clean
        pass


class DummyTCPSocket:
    def __init__(self) -> None:
        self.connected: list[tuple[str, int]] = []

    def settimeout(self, timeout: float) -> None:  # - stub
        pass

    def connect(self, addr: tuple[str, int]) -> None:
        self.connected.append(addr)

    def close(self) -> None:  # pragma: no cover - nothing to clean
        pass


def test_nat_traversal_flow(monkeypatch) -> None:
    # ---------- Hole punching success ----------
    trav = NATTraversal()
    monkeypatch.setattr(trav, "detect_nat", lambda: NATInfo("1.2.3.4", 1234, NATType.FULL_CONE))

    udp_socket = DummyUDPSocket()

    def socket_factory(_family: int, sock_type: int):
        assert sock_type == real_socket.SOCK_DGRAM
        return udp_socket

    import src.infrastructure.p2p.nat_traversal as mod

    monkeypatch.setattr(mod, "socket", real_socket)
    monkeypatch.setattr(mod.socket, "socket", socket_factory)

    start = time.perf_counter()
    assert trav.connect("5.6.7.8", 7777)
    hole_time = time.perf_counter() - start
    # First packet should be the initial hole punch; keepalives may follow
    assert udp_socket.sent[0] == (b"punch", ("5.6.7.8", 7777))
    assert hole_time < 1.0

    # ---------- Relay fallback ----------
    trav = NATTraversal(relay_servers=[("relay.example.com", 9000)])
    monkeypatch.setattr(trav, "detect_nat", lambda: NATInfo("9.9.9.9", 9999, NATType.SYMMETRIC))

    class FailingUDPSocket(DummyUDPSocket):
        def sendto(self, _data: bytes, _addr: tuple[str, int]) -> None:
            raise OSError

    relay_sock = DummyTCPSocket()

    def socket_factory2(_family: int, sock_type: int):
        if sock_type == real_socket.SOCK_DGRAM:
            return FailingUDPSocket()
        return relay_sock

    monkeypatch.setattr(mod.socket, "socket", socket_factory2)

    start = time.perf_counter()
    assert trav.connect("8.8.8.8", 8000)
    relay_time = time.perf_counter() - start
    assert relay_sock.connected == [("relay.example.com", 9000)]
    assert relay_time < 1.0
