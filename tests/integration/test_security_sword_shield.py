from dataclasses import dataclass


class Sword:
    def payloads(self) -> list[bytes]:
        return [b"seed", b"CRASH"]


@dataclass
class Shield:
    blocked: list[bytes]

    def inspect(self, payload: bytes) -> None:
        if b"CRASH" in payload:
            self.blocked.append(payload)
            raise ValueError("violation blocked")


@dataclass
class Sandbox:
    flag: str = "safe"


def dummy_target(payload: bytes, sandbox: Sandbox) -> None:
    if b"CRASH" in payload:
        sandbox.flag = "compromised"


def test_sword_shield_security() -> None:
    sword = Sword()
    shield = Shield(blocked=[])
    sandbox = Sandbox()

    for payload in sword.payloads():
        try:
            shield.inspect(payload)
            dummy_target(payload, sandbox)
        except ValueError:
            # violation was blocked
            pass

    # shield should have blocked the malicious payload
    assert shield.blocked == [b"CRASH"]
    # sandbox remains intact
    assert sandbox.flag == "safe"
    # ensure no privilege escalation (no unexpected globals)
    assert "pwned" not in globals()
