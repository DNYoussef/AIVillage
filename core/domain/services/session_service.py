"""
Session Service

Provides session management capabilities for agents.
This is a reference implementation to resolve import issues after reorganization.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any
import uuid


@dataclass
class Session:
    """Session data structure."""

    session_id: str
    agent_id: str
    created_at: datetime
    last_activity: datetime
    metadata: dict[str, Any]
    active: bool = True

    def __post_init__(self):
        if not self.metadata:
            self.metadata = {}


class SessionService:
    """
    Service for managing agent sessions.

    This is a reference implementation to resolve import dependencies
    during the reorganization process.
    """

    def __init__(self):
        """Initialize the session service."""
        self._sessions: dict[str, Session] = {}
        self._agent_sessions: dict[str, str] = {}

    def create_session(self, agent_id: str) -> Session:
        """Create a new session for an agent."""
        session_id = str(uuid.uuid4())
        now = datetime.now()

        session = Session(session_id=session_id, agent_id=agent_id, created_at=now, last_activity=now, metadata={})

        self._sessions[session_id] = session
        self._agent_sessions[agent_id] = session_id
        return session

    def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    def get_agent_session(self, agent_id: str) -> Session | None:
        """Get the active session for an agent."""
        session_id = self._agent_sessions.get(agent_id)
        if session_id:
            return self.get_session(session_id)
        return None

    def update_session_activity(self, session_id: str) -> bool:
        """Update session last activity time."""
        session = self.get_session(session_id)
        if session and session.active:
            session.last_activity = datetime.now()
            return True
        return False

    def close_session(self, session_id: str) -> bool:
        """Close a session."""
        session = self.get_session(session_id)
        if session:
            session.active = False
            # Remove from agent sessions mapping
            for agent_id, sid in list(self._agent_sessions.items()):
                if sid == session_id:
                    del self._agent_sessions[agent_id]
                    break
            return True
        return False

    def close_agent_session(self, agent_id: str) -> bool:
        """Close the active session for an agent."""
        session = self.get_agent_session(agent_id)
        if session:
            return self.close_session(session.session_id)
        return False
