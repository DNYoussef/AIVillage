from agent_forge.core.generator import (
    AGENT_REGISTRY,
    HeraldAgent,
    ScribeAgent,
)


def test_agents_register_and_basic_commands():
    scribe = ScribeAgent()
    herald = HeraldAgent()

    # start agents and ensure status changes
    scribe.start()
    herald.start()
    assert scribe.status() == "running"
    assert herald.status() == "running"

    # agents should be registered
    assert "Scribe" in AGENT_REGISTRY
    assert "Herald" in AGENT_REGISTRY

    # test heartbeat metric emission
    hb_before = herald.metrics["heartbeats"]
    herald.emit_heartbeat()
    assert herald.metrics["heartbeats"] == hb_before + 1

    # test simple messaging
    scribe.send_message("Herald", "ping")
    assert ("Scribe", "ping") in herald.messages

    # stop agent and verify status
    herald.stop()
    assert herald.status() == "stopped"
