from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
import logging
import socket
import threading
import time

# ruff: noqa: S104,TRY300,PERF203

logger = logging.getLogger(__name__)


class NATType(Enum):
    """Types of NAT devices encountered in networks."""

    FULL_CONE = "full_cone"
    RESTRICTED_CONE = "restricted_cone"
    PORT_RESTRICTED_CONE = "port_restricted_cone"
    SYMMETRIC = "symmetric"
    UNKNOWN = "unknown"


@dataclass
class NATInfo:
    """Information about the external network presence."""

    external_ip: str
    external_port: int
    nat_type: NATType


__all__ = ["NATInfo", "NATTraversal", "NATType"]


class NATTraversal:
    """Minimal NAT traversal helper implementing hole punching and relay fallback.

    The implementation focuses on providing a deterministic, testable sequence of
    operations rather than perfect network behaviour. It attempts UDP hole
    punching first and falls back to relay servers on failure. STUN lookups are
    performed lazily and failures simply return UNKNOWN NAT type, allowing the
    class to operate in restricted environments or tests without network access.
    """

    def __init__(
        self,
        stun_host: str = "stun.l.google.com",
        stun_port: int = 19302,
        relay_servers: Sequence[tuple[str, int]] | None = None,
    ) -> None:
        """Initialize NAT traversal helper."""
        self.stun_host = stun_host
        self.stun_port = stun_port
        self.relay_servers = list(relay_servers or [("127.0.0.1", 9000)])
        self._nat_info: NATInfo | None = None

    # ------------------------------------------------------------------
    # STUN helpers
    # ------------------------------------------------------------------
    def detect_nat(self) -> NATInfo:
        """Discover external address and NAT type using STUN if available."""
        if self._nat_info:
            return self._nat_info
        try:  # pragma: no cover - network may be unavailable
            import stun  # type: ignore[import-not-found]

            nat_type, external_ip, external_port = stun.get_ip_info(
                stun_host=self.stun_host, stun_port=self.stun_port
            )
            nat_enum = NATType.UNKNOWN
            if nat_type:
                formatted = nat_type.replace(" ", "_").upper()
                nat_enum = NATType.__members__.get(formatted, NATType.UNKNOWN)
            self._nat_info = NATInfo(
                external_ip or "0.0.0.0", int(external_port or 0), nat_enum
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("STUN detection failed: %s", exc)
            self._nat_info = NATInfo("0.0.0.0", 0, NATType.UNKNOWN)
        return self._nat_info

    # ------------------------------------------------------------------
    # Hole punching
    # ------------------------------------------------------------------
    def _predict_port(self, base_port: int, attempt: int) -> int:
        """Simple sequential port prediction for symmetric NATs."""
        return base_port + attempt

    def _keep_alive(
        self,
        sock: socket.socket,
        address: tuple[str, int],
        interval: float,
        duration: float,
    ) -> None:
        """Send periodic packets to keep NAT mapping alive."""
        end_time = time.time() + duration
        while time.time() < end_time:
            try:
                sock.sendto(b"keepalive", address)
            except OSError:
                break
            time.sleep(interval)

    def connect(
        self,
        peer_ip: str,
        peer_port: int,
        on_success: Callable[[socket.socket], None] | None = None,
        timeout: float = 5.0,
        retries: int = 3,
    ) -> bool:
        """Attempt to establish connectivity with a peer using hole punching.

        Returns True if either a direct UDP punch succeeds or a relay connection
        is established. The method is deterministic and largely mock-friendly for
        tests; real network errors are logged and ignored.
        """
        nat_info = self.detect_nat()
        deadline = time.time() + timeout

        for attempt in range(retries):
            if time.time() > deadline:
                break
            target_port = peer_port
            if nat_info.nat_type == NATType.SYMMETRIC:
                target_port = self._predict_port(peer_port, attempt)
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.settimeout(max(0.1, timeout / retries))
                sock.sendto(b"punch", (peer_ip, target_port))
                if on_success:
                    on_success(sock)
                t = threading.Thread(
                    target=self._keep_alive,
                    args=(sock, (peer_ip, target_port), 15.0, 30.0),
                    daemon=True,
                )
                t.start()
                return True
            except (
                OSError
            ) as exc:  # pragma: no cover - network operations mocked in tests
                logger.debug("Hole punching attempt %s failed: %s", attempt + 1, exc)
                time.sleep(0.2)

        # Relay fallback
        for relay_ip, relay_port in self.relay_servers:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(timeout)
                sock.connect((relay_ip, relay_port))
                if on_success:
                    on_success(sock)
                return True
            except OSError as exc:  # pragma: no cover
                logger.debug(
                    "Relay connection to %s:%s failed: %s", relay_ip, relay_port, exc
                )
                continue
        return False
