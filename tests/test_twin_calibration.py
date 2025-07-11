import importlib
import asyncio


from services.twin.schemas import ChatRequest


def test_calibrated_prob_feature(monkeypatch):
    monkeypatch.setenv("CALIBRATION_ENABLED", "1")
    if "services.twin.app" in importlib.sys.modules:
        del importlib.sys.modules["services.twin.app"]
    mod = importlib.import_module("services.twin.app")
    agent = mod.TwinAgent(mod.settings.model_path)
    req = ChatRequest(message="hello", user_id="u1")
    out = asyncio.run(agent.chat(req))
    assert out.calibrated_prob is not None
    assert 0.0 <= out.calibrated_prob <= 1.0
