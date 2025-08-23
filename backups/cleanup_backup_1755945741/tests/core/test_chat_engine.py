from core.chat_engine import ChatEngine


class DummyResp:
    def __init__(self, data):
        self._data = data
        self.status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


def test_process_chat(monkeypatch):
    ce = ChatEngine()

    called = {}

    def fake_post(url, json, timeout):
        called["payload"] = json
        return DummyResp({"response": "ok", "raw_prob": 0.6})

    monkeypatch.setattr("requests.post", fake_post)
    out = ce.process_chat("hi", "cid")
    assert out["response"] == "ok"
    assert out["conversation_id"] == "cid"
    assert out["raw_prob"] == 0.6
    assert "timestamp" in out
    assert called["payload"]["prompt"] == "hi"


def test_process_chat_prefers_server_calib(monkeypatch):
    monkeypatch.setenv("CALIBRATION_ENABLED", "1")
    ce = ChatEngine()

    def fake_post(url, json, timeout):
        return DummyResp({"response": "hi", "raw_prob": 0.4, "calibrated_prob": 0.8})

    monkeypatch.setattr("requests.post", fake_post)
    out = ce.process_chat("hello", "x")
    assert out["calibrated_prob"] == 0.8
