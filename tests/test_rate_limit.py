import time
import os
import requests
import pytest

GW = os.getenv("GW_URL", "http://localhost:8000")


def _gateway_up() -> bool:
    try:
        return requests.get(f"{GW}/healthz", timeout=1).status_code == 200
    except requests.RequestException:
        return False


@pytest.mark.skipif(not _gateway_up(), reason="gateway not running")
def test_rate_limit_resets():
    payload = {"message": "ping", "user_id": "ttl"}
    for _ in range(101):
        requests.post(f"{GW}/v1/chat", json=payload)
    assert requests.post(f"{GW}/v1/chat", json=payload).status_code == 429
    time.sleep(61)
    assert requests.post(f"{GW}/v1/chat", json=payload).status_code == 200
