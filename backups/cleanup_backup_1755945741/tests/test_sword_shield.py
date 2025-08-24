from pathlib import Path
import sys
import unittest

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from production.agents.sword_shield import ShieldAgent, SwordAgent


class DummyAPIClient:
    def request(self, payload: str) -> None:
        if payload != "test":
            msg = "unexpected payload"
            raise RuntimeError(msg)


def vulnerable_target(data: bytes) -> None:
    text = data.decode("utf-8", "ignore")
    if "CRASH" in text:
        msg = "boom"
        raise RuntimeError(msg)


class TestSwordAgent(unittest.TestCase):
    def test_fuzz_detects_crash(self) -> None:
        sword = SwordAgent()
        result = sword.fuzz(vulnerable_target, [b"seed", b"CRASH"], iterations=32)
        assert result["crashes_found"] >= 1

    def test_penetration_test_reports_findings(self) -> None:
        sword = SwordAgent()
        findings = sword.penetration_test(DummyAPIClient(), attempts=4)
        assert findings


class TestShieldAgent(unittest.TestCase):
    def test_blocks_malicious_input(self) -> None:
        shield = ShieldAgent()
        with pytest.raises(ValueError, match="policy violation detected"):
            shield.run(lambda _: None, b"../etc/passwd")
        assert any(ev.event == "pre_execution_block" for ev in shield.events)

    def test_allows_safe_input(self) -> None:
        shield = ShieldAgent()
        crashed, _ = shield.run(lambda d: d, b"hello")
        assert not crashed
        assert not shield.events


if __name__ == "__main__":  # pragma: no cover - manual run
    unittest.main()
