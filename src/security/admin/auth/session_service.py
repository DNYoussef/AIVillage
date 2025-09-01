"""
Session Management Service
Extracted from SecureAdminServer for focused responsibility.
"""

import secrets
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Set

from ..constants import (
    SESSION_TIMEOUT_MINUTES, MAX_SESSIONS_PER_USER, ROLE_PERMISSIONS
)

logger = logging.getLogger(__name__)


class SessionService:
    """Secure session management for admin users"""
    
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        self.session_timeout = timedelta(minutes=SESSION_TIMEOUT_MINUTES)
        self.max_sessions_per_user = MAX_SESSIONS_PER_USER
    
    def create_session(self, user_id: str, user_roles: Set[str], client_ip: str) -> str:
        """Create secure admin session"""
        session_id = secrets.token_urlsafe(32)
        
        # Clean old sessions for user
        self._cleanup_user_sessions(user_id)
        
        session_data = {
            "session_id": session_id,
            "user_id": user_id,
            "roles": user_roles,
            "client_ip": client_ip,
            "created_at": datetime.utcnow(),
            "last_activity": datetime.utcnow(),
            "mfa_verified": False,
            "permissions": self._get_user_permissions(user_roles)
        }
        
        self.sessions[session_id] = session_data
        logger.info(f"Created admin session for user {user_id} from {client_ip}")
        
        return session_id
    
    def validate_session(self, session_id: str, client_ip: str) -> Optional[Dict]:
        """Validate and update session"""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        now = datetime.utcnow()
        
        # Check timeout
        if now - session["last_activity"] > self.session_timeout:
            self.destroy_session(session_id)
            logger.info(f"Session {session_id} expired due to inactivity")
            return None
        
        # Check IP consistency
        if session["client_ip"] != client_ip:
            logger.warning(f"Session {session_id} IP mismatch: {session['client_ip']} vs {client_ip}")
            self.destroy_session(session_id)
            return None
        
        # Update last activity
        session["last_activity"] = now
        return session
    
    def destroy_session(self, session_id: str):
        """Destroy session"""
        if session_id in self.sessions:
            user_id = self.sessions[session_id]["user_id"]
            del self.sessions[session_id]
            logger.info(f"Destroyed session {session_id} for user {user_id}")
    
    def require_mfa_verification(self, session_id: str) -> bool:
        """Mark session as requiring MFA verification"""
        if session_id in self.sessions:
            self.sessions[session_id]["mfa_verified"] = False
            return True
        return False
    
    def verify_mfa(self, session_id: str, mfa_token: str) -> bool:
        """Verify MFA token (simplified implementation)"""
        # In production, this would integrate with TOTP/SMS/hardware tokens
        if session_id not in self.sessions:
            return False
        
        # Simplified MFA check (replace with real implementation)
        if len(mfa_token) == 6 and mfa_token.isdigit():
            self.sessions[session_id]["mfa_verified"] = True
            logger.info(f"MFA verified for session {session_id}")
            return True
        
        logger.warning(f"MFA verification failed for session {session_id}")
        return False
    
    def _cleanup_user_sessions(self, user_id: str):
        """Clean up old sessions for user"""
        user_sessions = [
            (sid, session) for sid, session in self.sessions.items()
            if session["user_id"] == user_id
        ]
        
        # Sort by last activity and keep only recent ones
        user_sessions.sort(key=lambda x: x[1]["last_activity"], reverse=True)
        
        # Remove excess sessions
        for session_id, _ in user_sessions[self.max_sessions_per_user:]:
            self.destroy_session(session_id)
    
    def _get_user_permissions(self, roles: Set[str]) -> Set[str]:
        """Get permissions for user roles"""
        permissions = set()
        for role in roles:
            permissions.update(ROLE_PERMISSIONS.get(role, set()))
        
        return permissions