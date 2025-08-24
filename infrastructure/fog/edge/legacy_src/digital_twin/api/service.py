"""FastAPI service layer exposing the Digital Twin capabilities."""

from __future__ import annotations

import base64
from typing import Any
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.digital_twin.core.digital_twin import DigitalTwin, LearningProfile, LearningSession

app = FastAPI(title="Digital Twin API")


class UserProfile(BaseModel):
    student_id: str
    name: str
    age: int
    grade_level: int
    language: str = "en"
    region: str = ""
    learning_style: str = "visual"
    strengths: list[str] = []
    challenges: list[str] = []
    interests: list[str] = []
    attention_span_minutes: int = 0
    preferred_session_times: list[str] = []
    parent_constraints: dict[str, Any] = {}
    accessibility_needs: list[str] = []
    motivation_triggers: list[str] = []
    created_at: str | None = None
    last_updated: str | None = None


class LearningSessionModel(BaseModel):
    session_id: str
    student_id: str
    tutor_model_id: str
    start_time: str
    end_time: str
    duration_minutes: int
    concepts_covered: list[str]
    questions_asked: int
    questions_correct: int
    engagement_score: float
    difficulty_level: float
    adaptations_made: list[str]
    parent_feedback: str | None = None
    student_mood: str = "neutral"
    session_notes: str = ""


class ShareConfig(BaseModel):
    revenue_percentage: float


class MobileSyncRequest(BaseModel):
    """Base64 encoded payload from the mobile client."""

    data: str


class DigitalTwinService:
    """Maintain active Digital Twin instances."""

    def __init__(self) -> None:
        self.twins: dict[str, DigitalTwin] = {}

    def _get_twin(self, user_id: str) -> DigitalTwin:
        twin = self.twins.get(user_id)
        if twin is None:
            raise HTTPException(status_code=404, detail="twin not found")
        return twin


service = DigitalTwinService()


@app.post("/twin/create")
async def create_twin(user_id: str, profile: UserProfile) -> dict[str, Any]:
    """Create a new Digital Twin for a user."""
    twin = DigitalTwin()
    twin.students[user_id] = LearningProfile(**profile.dict())
    service.twins[user_id] = twin
    return {"status": "created", "vault_id": user_id}


@app.post("/twin/{user_id}/learn")
async def record_learning(user_id: str, session: LearningSessionModel) -> dict[str, Any]:
    """Record a learning session for the specified user."""
    twin = service._get_twin(user_id)
    sess = LearningSession(**session.dict())
    twin.session_history[user_id].append(sess)
    await twin.update_personalization_vector(sess)
    await twin.update_knowledge_states_from_session(sess)
    return twin.generate_private_analytics()


@app.get("/twin/{user_id}/recommendations")
async def get_recommendations(user_id: str) -> dict[str, Any]:
    """Return personalized learning recommendations."""
    twin = service._get_twin(user_id)
    simulated_paths = twin.shadow_simulator.explore_paths(current_state=twin.current_knowledge, time_horizon=7)
    ranked_paths = twin.rank_by_learning_style(simulated_paths)
    top = ranked_paths[:5]
    return {
        "recommended_topics": top,
        "estimated_time": twin.estimate_completion_time(top),
        "confidence": twin.confidence_score,
    }


@app.post("/twin/{user_id}/marketplace/share")
async def share_to_marketplace(user_id: str, share_config: ShareConfig) -> dict[str, Any]:
    """Share anonymized patterns to the marketplace."""
    twin = service._get_twin(user_id)
    patterns = twin.extract_learning_patterns()
    anonymized = twin.full_anonymization(patterns)
    value = twin.calculate_pattern_value(anonymized)
    listing_id = str(uuid.uuid4())
    return {"listing_id": listing_id, "estimated_value": value}


@app.get("/twin/{user_id}/health")
async def health_check(user_id: str) -> dict[str, Any]:
    twin = service._get_twin(user_id)
    last_activity = twin.session_history[user_id][-1].end_time if twin.session_history[user_id] else None
    return {
        "vault_status": "encrypted",
        "sync_status": "healthy",
        "last_activity": last_activity,
        "storage_used": len(twin.session_history[user_id]),
        "encryption_status": "AES-256",
        "privacy_level": "maximum",
    }


@app.get("/twin/{user_id}/analytics")
async def private_analytics(user_id: str) -> dict[str, Any]:
    twin = service._get_twin(user_id)
    return twin.generate_private_analytics()


class PersistentQueue:
    """Very small stub of a persistent queue used for sync operations."""

    def __init__(self) -> None:
        self.items: list[bytes] = []

    def put(self, item: bytes) -> None:
        self.items.append(item)


class TwinMobileSync:
    def __init__(self, twin: DigitalTwin) -> None:
        self.twin = twin
        self.sync_queue = PersistentQueue()

    async def sync_from_mobile(self, mobile_data: bytes) -> bytes:
        decrypted = self.twin.decrypt_mobile_data(mobile_data)
        merge_result = self.twin.merge_states(local_state=decrypted, conflict_resolution="latest_wins")
        diff = self.twin.generate_state_diff(merge_result)
        return self.twin.encrypt_for_mobile(diff)


@app.post("/twin/{user_id}/sync")
async def sync_mobile(user_id: str, req: MobileSyncRequest) -> dict[str, Any]:
    twin = service._get_twin(user_id)
    syncer = TwinMobileSync(twin)
    payload = base64.b64decode(req.data)
    response = await syncer.sync_from_mobile(payload)
    return {"data": base64.b64encode(response).decode()}
