from src.production.monitoring.mobile import resource_management as rm


def test_env_policy_triggers_low_resource_behavior(monkeypatch):
    monkeypatch.setenv("BATTERY", "15")
    monkeypatch.setenv("AIV_MOBILE_PROFILE", "low_ram")
    monkeypatch.setenv("THERMAL", "hot")
    policy = rm.evaluate_policy_from_env()
    assert policy["chunk_size"] < rm.DEFAULT_CHUNK_SIZE
    assert policy["transport"] == rm.TransportPreference.BITCHAT_PREFERRED
