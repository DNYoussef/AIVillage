"""Unit tests for the SecurityRiskGate"""

import pytest

from src.twin_runtime.guard import risk_gate


@pytest.mark.parametrize(
    "content",
    [
        "'; DROP TABLE users; --",
        "rm -rf /",
        "format c:",
        "<script>alert('XSS')</script>",
        "../../etc/passwd",
        "eval(__import__('os').system('rm -rf /'))",
        "DELETE FROM users WHERE 1=1",
        "os.system('shutdown -h now')",
        "subprocess.call(['rm', '-rf', '/'])",
    ],
)
def test_dangerous_inputs_denied(content):
    assert risk_gate({"content": content, "type": "command"}) == "deny"


@pytest.mark.parametrize(
    "content",
    [
        "My SSN is 123-45-6789",
        "Credit card: 4111111111111111",
        "api_key=sk_live_xxxxx",
        "password=supersecret123",
        "private_key=-----BEGIN RSA PRIVATE KEY-----",
    ],
)
def test_sensitive_inputs_flagged(content):
    assert risk_gate({"content": content, "type": "text"}) == "ask"


@pytest.mark.parametrize(
    "content",
    ["Hello, how are you?", "Calculate 2 + 2", "What's the weather today?"],
)
def test_safe_inputs_allowed(content):
    assert risk_gate({"content": content, "type": "text"}) == "allow"


def test_risk_thresholds():
    assert risk_gate({"content": "hello", "type": "text"}, 0.2) == "allow"
    assert risk_gate({"content": "select * from users", "type": "query"}, 0.5) == "ask"
    assert risk_gate({"content": "DROP TABLE users; DELETE FROM logs;", "type": "command"}, 0.9) == "deny"
