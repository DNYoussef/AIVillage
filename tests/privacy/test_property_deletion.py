import os
import uuid

import pytest
import requests
from hypothesis import given
from hypothesis import strategies as st

GW = os.getenv("GW_URL", "http://localhost:8000")


def _gateway_up() -> bool:
    try:
        return requests.get(f"{GW}/healthz", timeout=1).status_code == 200
    except requests.RequestException:
        return False


@given(st.text(min_size=1, max_size=50))
@pytest.mark.skipif(not _gateway_up(), reason="gateway not running")
def test_any_message_is_erasable(msg):
    uid = str(uuid.uuid4())
    r = requests.post(f"{GW}/v1/chat", json={"message": msg, "user_id": uid})
    assert r.ok
    resp = requests.delete(f"{GW.replace(':8000', ':8001')}/v1/user/{uid}")
    assert resp.ok
    assert resp.json()["deleted_conversations"] >= 1
    new = requests.post(
        f"{GW}/v1/chat", json={"message": "hello", "user_id": uid}
    ).json()
    assert new["conversation_id"] != r.json()["conversation_id"]
