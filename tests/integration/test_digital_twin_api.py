import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
import pytest
from fastapi.testclient import TestClient

from experimental.services.services.twin.app import app


@pytest.mark.asyncio
async def test_digital_twin_api_integration():
    # Create a test client
    client = TestClient(app)

    # Send a request to the chat endpoint
    response = client.post("/v1/chat", json={"message": "Hello, world!"})

    # Check that the response is correct
    assert response.status_code == 200
    assert "Hello, world!" in response.json()["response"]
