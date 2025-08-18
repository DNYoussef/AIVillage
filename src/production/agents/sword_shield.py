from __future__ import annotations

import logging
import random
import time
import traceback
from dataclasses import dataclass, field
from trace import Trace
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

logger = logging.getLogger(__name__)


class WasiSandbox:
    """Very small wrapper simulating WASI sandbox execution."""

    def __init__(self, timeout: float = 1.0) -> None:
        """Create sandbox with a timeout in seconds."""
        self.timeout = timeout

    def run(self, func: Callable[[bytes], Any], data: bytes) -> tuple[bool, set[int], dict[str, Any]]:
        """Execute ``func`` with ``data`` and capture coverage."""
        tracer = Trace(count=True, trace=False)
        crashed = False
        info: dict[str, Any] = {}
        try:
            tracer.runfunc(func, data)
        except Exception as exc:  # pragma: no cover - emulate crash
            crashed = True
            info["error"] = str(exc)
            info["traceback"] = traceback.format_exc()
        results = tracer.results()
        coverage = set(results.counts.keys())
        return crashed, coverage, info


def _bitflip(data: bytearray) -> bytearray:
    if not data:
        return bytearray(b"\x00")
    idx = random.randrange(len(data))
    data[idx] ^= 0xFF
    return data


def _byte_duplicate(data: bytearray) -> bytearray:
    if not data:
        return bytearray(b"A")
    idx = random.randrange(len(data))
    data.insert(idx, data[idx])
    return data


MUTATORS: list[Callable[[bytearray], bytearray]] = [_bitflip, _byte_duplicate]


class SwordAgent:
    """Security testing specialist implementing a tiny fuzzing harness."""

    common_patterns: ClassVar[list[bytes]] = [
        b"' OR '1'='1",
        b"../",
        b"DROP TABLE",
        b"CRASH",
    ]

    def __init__(self, sandbox: WasiSandbox | None = None) -> None:
        """Initialize the fuzzing agent."""
        self.sandbox = sandbox or WasiSandbox()
        self.crashes: list[tuple[bytes, dict[str, Any]]] = []
        self.coverage: set[int] = set()

    def mutate(self, seed: bytes) -> bytes:
        data = bytearray(seed)
        mutator = random.choice(MUTATORS)
        return bytes(mutator(data))

    def fuzz(
        self,
        target: Callable[[bytes], Any],
        seeds: Sequence[bytes],
        iterations: int = 128,
    ) -> dict[str, Any]:
        """Run a very small AFL style fuzzing loop."""
        start = time.time()
        for _ in range(iterations):
            seed = random.choice(seeds)
            candidate = self.mutate(seed)
            crashed, cov, info = self.sandbox.run(target, candidate)
            self.coverage.update(cov)
            if crashed:
                self.crashes.append((candidate, info))
        duration = time.time() - start
        return {
            "crashes_found": len(self.crashes),
            "coverage": len(self.coverage),
            "duration": duration,
        }

    def penetration_test(self, api_client: Any, attempts: int = 16) -> list[str]:
        """Perform a few basic penetration tests against ``api_client``."""
        findings: list[str] = []
        for _ in range(attempts):
            payload = self.mutate(b"test")
            try:
                api_client.request(payload.decode("utf-8", "ignore"))
            except Exception as exc:
                # pragma: no cover - fuzzing expects errors
                findings.append(str(exc))
        return findings


@dataclass
class SecurityEvent:
    event: str
    details: dict[str, Any]
    blocked: bool
    timestamp: float = field(default_factory=time.time)


class ShieldAgent:
    """Policy enforcement agent providing execution hooks."""

    blocked_patterns: ClassVar[list[str]] = ["../", "rm -rf", "DROP TABLE"]

    def __init__(self, sandbox: WasiSandbox | None = None) -> None:
        """Create shield agent with optional sandbox."""
        self.sandbox = sandbox or WasiSandbox()
        self.events: list[SecurityEvent] = []

    def _log(self, event: str, details: dict[str, Any], *, blocked: bool) -> None:
        self.events.append(SecurityEvent(event, details, blocked))
        logger.debug("%s: %s", event, details)
        if blocked:
            self._alert_admin(event, details)

    def _alert_admin(self, event: str, details: dict[str, Any]) -> None:
        logger.warning("ALERT %s: %s", event, details)

    def _auto_remediate(self, details: dict[str, Any]) -> None:
        details["remediated"] = True

    def pre_execute(self, data: bytes) -> None:
        text = data.decode("utf-8", "ignore")
        for pattern in self.blocked_patterns:
            if pattern in text:
                self._log(
                    "pre_execution_block",
                    {"pattern": pattern, "data": text},
                    blocked=True,
                )
                msg = "policy violation detected"
                raise ValueError(msg)

    def post_execute(self, *, crashed: bool, info: dict[str, Any]) -> None:
        if crashed:
            self._log("execution_crash", info, blocked=False)

    def run(self, func: Callable[[bytes], Any], data: bytes) -> tuple[bool, set[int]]:
        """Execute ``func`` under policy enforcement."""
        self.pre_execute(data)
        crashed, cov, info = self.sandbox.run(func, data)
        self.post_execute(crashed=crashed, info=info)
        return crashed, cov


class SwordAndShieldAgent:
    """Composite agent combining Sword and Shield capabilities."""

    def __init__(self, spec=None) -> None:
        """Instantiate both Sword and Shield agents."""
        self.spec = spec  # Store specification for compatibility
        self.sandbox = WasiSandbox()
        self.sword = SwordAgent(self.sandbox)
        self.shield = ShieldAgent(self.sandbox)

    def fuzz(
        self,
        target: Callable[[bytes], Any],
        seeds: Sequence[bytes],
        iterations: int = 128,
    ) -> dict[str, Any]:
        """Delegate to :class:`SwordAgent` fuzzing."""
        return self.sword.fuzz(target, seeds, iterations)

    def secure_run(self, func: Callable[[bytes], Any], data: bytes) -> tuple[bool, set[int]]:
        """Run ``func`` through :class:`ShieldAgent` policy enforcement."""
        return self.shield.run(func, data)
