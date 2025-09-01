"""
Authentication Route Handlers
Extracted from SecureAdminServer for focused responsibility.
"""

import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, Set

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse

from ..constants import (
    HTTP_STATUS_UNAUTHORIZED, HTTP_STATUS_FORBIDDEN, HTTP_STATUS_OK,
    SESSION_TIMEOUT_MINUTES
)

logger = logging.getLogger(__name__)


class AuthHandlers:
    """Authentication route handlers"""
    
    def __init__(self, session_service):
        self.session_service = session_service
    
    async def admin_login(self, request: Request, credentials: Dict[str, str]):
        """Admin login with session creation"""
        
        username = credentials.get("username")
        password = credentials.get("password")
        client_ip = getattr(request.client, "host", "127.0.0.1") if request.client else "127.0.0.1"
        
        # Validate credentials (integrate with your auth system)
        if not self._validate_admin_credentials(username, password):
            raise HTTPException(
                status_code=HTTP_STATUS_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        # Get user roles
        user_roles = self._get_user_roles(username)
        if "admin" not in user_roles and "super_admin" not in user_roles:
            raise HTTPException(
                status_code=HTTP_STATUS_FORBIDDEN,
                detail="Admin privileges required"
            )
        
        # Create session
        session_id = self.session_service.create_session(username, user_roles, client_ip)
        
        return JSONResponse({
            "message": "Login successful - MFA required",
            "session_id": session_id,
            "requires_mfa": True,
            "expires_at": (datetime.utcnow() + timedelta(minutes=SESSION_TIMEOUT_MINUTES)).isoformat()
        })
    
    async def verify_mfa(self, request: Request, mfa_data: Dict[str, str]):
        """Verify MFA token"""
        
        session_id = mfa_data.get("session_id")
        mfa_token = mfa_data.get("mfa_token")
        
        if not session_id or not mfa_token:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Session ID and MFA token required"
            )
        
        client_ip = getattr(request.client, "host", "127.0.0.1") if request.client else "127.0.0.1"
        session = self.session_service.validate_session(session_id, client_ip)
        
        if not session:
            raise HTTPException(
                status_code=HTTP_STATUS_UNAUTHORIZED,
                detail="Invalid or expired session"
            )
        
        if not self.session_service.verify_mfa(session_id, mfa_token):
            raise HTTPException(
                status_code=HTTP_STATUS_UNAUTHORIZED,
                detail="Invalid MFA token"
            )
        
        return JSONResponse({
            "message": "MFA verification successful",
            "session_id": session_id,
            "permissions": list(session["permissions"]),
            "expires_at": (datetime.utcnow() + timedelta(minutes=SESSION_TIMEOUT_MINUTES)).isoformat()
        })
    
    async def admin_logout(self, request: Request):
        """Admin logout"""
        
        session_id = request.headers.get("x-admin-session")
        if session_id:
            self.session_service.destroy_session(session_id)
        
        return JSONResponse({"message": "Logged out successfully"})
    
    def _validate_admin_credentials(self, username: str, password: str) -> bool:
        """Validate admin credentials (integrate with your auth system)"""
        # This is a simplified implementation
        # In production, integrate with your authentication system
        
        # Example credentials (replace with real auth)
        admin_users = {
            "admin": hashlib.sha256("secure_admin_password".encode()).hexdigest(),
            "superadmin": hashlib.sha256("super_secure_password".encode()).hexdigest()
        }
        
        if username not in admin_users:
            return False
        
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        return password_hash == admin_users[username]
    
    def _get_user_roles(self, username: str) -> Set[str]:
        """Get user roles"""
        user_roles = {
            "admin": {"admin"},
            "superadmin": {"super_admin", "admin"}
        }
        
        return user_roles.get(username, set())