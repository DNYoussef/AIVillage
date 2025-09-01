"""
Security Context Management
Extracted from SecureAdminServer for focused responsibility.
"""

import logging
from datetime import datetime

from fastapi import HTTPException, Request, status

from ..constants import HTTP_STATUS_UNAUTHORIZED
from ..boundaries.secure_boundary_contracts import SecurityContext, SecurityLevel

logger = logging.getLogger(__name__)


class SecurityContextService:
    """Security context extraction and validation"""
    
    def __init__(self, session_service):
        self.session_service = session_service
    
    async def extract_security_context(self, request: Request) -> SecurityContext:
        """Extract security context from request"""
        
        session_id = request.headers.get("x-admin-session")
        if not session_id:
            raise HTTPException(
                status_code=HTTP_STATUS_UNAUTHORIZED,
                detail="Admin session required"
            )
        
        client_ip = getattr(request.client, "host", "127.0.0.1") if request.client else "127.0.0.1"
        session = self.session_service.validate_session(session_id, client_ip)
        
        if not session:
            raise HTTPException(
                status_code=HTTP_STATUS_UNAUTHORIZED,
                detail="Invalid or expired session"
            )
        
        if not session["mfa_verified"]:
            raise HTTPException(
                status_code=HTTP_STATUS_UNAUTHORIZED,
                detail="MFA verification required"
            )
        
        return SecurityContext(
            user_id=session["user_id"],
            session_id=session_id,
            roles=frozenset(session["roles"]),
            permissions=frozenset(session["permissions"]),
            security_level=SecurityLevel.RESTRICTED,
            source_ip=client_ip,
            user_agent=request.headers.get("user-agent", "unknown"),
            timestamp=datetime.utcnow()
        )