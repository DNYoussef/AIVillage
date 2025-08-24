import base64
from pathlib import Path
import sys

import httpx
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from digital_twin.api.service import app, service


def _profile() -> dict:
    return {
        "student_id": "user1",
        "name": "Alice",
        "age": 10,
        "grade_level": 5,
        "language": "en",
        "region": "US",
        "learning_style": "visual",
        "strengths": [],
        "challenges": [],
        "interests": [],
        "attention_span_minutes": 30,
        "preferred_session_times": [],
        "parent_constraints": {},
        "accessibility_needs": [],
        "motivation_triggers": [],
    }


def _session() -> dict:
    return {
        "session_id": "s1",
        "student_id": "user1",
        "tutor_model_id": "model1",
        "start_time": "2024-01-01T00:00:00",
        "end_time": "2024-01-01T00:30:00",
        "duration_minutes": 30,
        "concepts_covered": ["fractions"],
        "questions_asked": 10,
        "questions_correct": 8,
        "engagement_score": 0.8,
        "difficulty_level": 0.5,
        "adaptations_made": [],
        "parent_feedback": None,
        "student_mood": "happy",
        "session_notes": "",
    }


@pytest.mark.asyncio
async def test_digital_twin_api_flow():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post("/twin/create", params={"user_id": "user1"}, json=_profile())
        assert r.status_code == 200

        r = await client.post("/twin/user1/learn", json=_session())
        assert r.status_code == 200
        data = r.json()
        assert "learning_velocity" in data

        r = await client.get("/twin/user1/recommendations")
        rec = r.json()
        assert "recommended_topics" in rec

        r = await client.get("/twin/user1/analytics")
        assert "engagement_score" in r.json()

        r = await client.post("/twin/user1/marketplace/share", json={"revenue_percentage": 0.5})
        assert "listing_id" in r.json()

        r = await client.get("/twin/user1/health")
        assert r.json()["vault_status"] == "encrypted"

        # Mobile sync round-trip
        twin = service.twins["user1"]
        payload = twin.encrypt_for_mobile({"students": {}})
        encoded = base64.b64encode(payload).decode()
        r = await client.post("/twin/user1/sync", json={"data": encoded})
        assert r.status_code == 200
        returned = base64.b64decode(r.json()["data"])
        assert twin.decrypt_mobile_data(returned) == {"students": {}}
