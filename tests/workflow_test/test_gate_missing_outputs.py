import pytest


def gate_condition(security_gate_passed=None, emergency_bypass=None):
    """Replicate gate condition logic with default fallbacks."""
    return (
        (security_gate_passed or 'unknown') == 'true'
        or (emergency_bypass or 'unknown') == 'true'
    )


def test_gate_missing_outputs_defaults_to_false():
    """When upstream outputs are missing, gate should not authorize."""
    assert gate_condition() is False


def test_gate_allows_emergency_bypass():
    assert gate_condition(emergency_bypass='true') is True


def test_gate_allows_security_pass():
    assert gate_condition(security_gate_passed='true') is True
