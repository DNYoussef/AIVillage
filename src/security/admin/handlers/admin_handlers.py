"""
Admin Operation Route Handlers
Extracted from SecureAdminServer for focused responsibility.
"""

import uuid
import logging
from datetime import datetime
from typing import Any, Dict

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse

from ..constants import HTTP_STATUS_FORBIDDEN, EMERGENCY_CONFIRMATION_TEXT

logger = logging.getLogger(__name__)


class AdminHandlers:
    """Admin operation route handlers"""
    
    def __init__(self, session_service, admin_boundary):
        self.session_service = session_service
        self.admin_boundary = admin_boundary
    
    async def get_system_status(self, request: Request):
        """Get comprehensive system status"""
        context = request.state.security_context
        
        return await self.admin_boundary.execute_secured(
            context,
            lambda: self._get_system_status()
        )
    
    async def get_audit_logs(self, request: Request, limit: int = 100, offset: int = 0):
        """Get audit logs"""
        context = request.state.security_context
        
        if "admin:security" not in context.permissions:
            raise HTTPException(
                status_code=HTTP_STATUS_FORBIDDEN,
                detail="Security admin privileges required"
            )
        
        return await self.admin_boundary.execute_secured(
            context,
            lambda: self._get_audit_logs(limit, offset)
        )
    
    async def get_threat_analysis(self, request: Request):
        """Get current threat analysis"""
        context = request.state.security_context
        
        return await self.admin_boundary.execute_secured(
            context,
            lambda: self._get_threat_analysis()
        )
    
    async def trigger_security_scan(self, request: Request):
        """Trigger comprehensive security scan"""
        context = request.state.security_context
        
        if "admin:system" not in context.permissions:
            raise HTTPException(
                status_code=HTTP_STATUS_FORBIDDEN,
                detail="System admin privileges required"
            )
        
        return await self.admin_boundary.execute_secured(
            context,
            lambda: self._trigger_security_scan(context.user_id)
        )
    
    async def get_users(self, request: Request):
        """Get user list"""
        context = request.state.security_context
        
        if "admin:users" not in context.permissions:
            raise HTTPException(
                status_code=HTTP_STATUS_FORBIDDEN,
                detail="User admin privileges required"
            )
        
        return await self.admin_boundary.execute_secured(
            context,
            lambda: self._get_users()
        )
    
    async def emergency_shutdown(self, request: Request, confirmation: Dict[str, str]):
        """Emergency system shutdown"""
        context = request.state.security_context
        
        if "super_admin" not in context.roles:
            raise HTTPException(
                status_code=HTTP_STATUS_FORBIDDEN,
                detail="Super admin privileges required for emergency actions"
            )
        
        if confirmation.get("confirm") != EMERGENCY_CONFIRMATION_TEXT:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Emergency shutdown requires explicit confirmation"
            )
        
        return await self.admin_boundary.execute_secured(
            context,
            lambda: self._emergency_shutdown(context.user_id)
        )
    
    def _get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            "services": {
                "admin_interface": {"status": "healthy", "localhost_only": True},
                "security_boundary": {"status": "active", "violations": 0},
                "session_manager": {"status": "active", "active_sessions": len(self.session_service.sessions)}
            },
            "security": {
                "localhost_binding": True,
                "mfa_enabled": True,
                "audit_logging": True,
                "security_headers": True
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _get_audit_logs(self, limit: int, offset: int) -> Dict[str, Any]:
        """Get audit logs"""
        # In production, this would read from actual audit log files
        return {
            "logs": [
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "event": "admin_login_success",
                    "user": "admin", 
                    "ip": "127.0.0.1"
                }
            ],
            "total": 1,
            "limit": limit,
            "offset": offset
        }
    
    def _get_threat_analysis(self) -> Dict[str, Any]:
        """Get threat analysis"""
        return {
            "current_threats": [],
            "risk_level": "low",
            "last_scan": datetime.utcnow().isoformat(),
            "localhost_access_attempts": 0,
            "external_access_attempts": 0
        }
    
    def _trigger_security_scan(self, user_id: str) -> Dict[str, Any]:
        """Trigger security scan"""
        scan_id = str(uuid.uuid4())
        logger.info(f"Security scan {scan_id} triggered by user {user_id}")
        
        return {
            "scan_id": scan_id,
            "status": "initiated",
            "estimated_duration": "5-10 minutes",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _get_users(self) -> Dict[str, Any]:
        """Get user list"""
        return {
            "users": [
                {"id": "1", "username": "admin", "roles": ["admin"], "last_login": datetime.utcnow().isoformat()},
                {"id": "2", "username": "superadmin", "roles": ["super_admin"], "last_login": datetime.utcnow().isoformat()}
            ],
            "total": 2
        }
    
    def _emergency_shutdown(self, user_id: str) -> Dict[str, Any]:
        """Emergency shutdown"""
        logger.critical(f"EMERGENCY SHUTDOWN initiated by user {user_id}")
        
        # In production, this would actually shutdown systems
        return {
            "status": "emergency_shutdown_initiated",
            "initiated_by": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "estimated_completion": "30 seconds"
        }