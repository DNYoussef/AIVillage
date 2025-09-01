"""
Security Middleware Components
Extracted from SecureAdminServer for focused responsibility.
"""

import json
import logging
import time
import uuid
import ipaddress
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set

from fastapi import Request
from fastapi.responses import PlainTextResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ..constants import (
    SECURITY_HEADERS, LOCALHOST_IPS, MAX_BLOCKED_ATTEMPTS,
    BLOCKING_DURATION_MINUTES, HTTP_STATUS_FORBIDDEN, HTTP_STATUS_TOO_MANY_REQUESTS,
    DEFAULT_AUDIT_LOG_PATH
)

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses"""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Apply security headers
        for header, value in SECURITY_HEADERS.items():
            response.headers[header] = value
        
        # Remove server information
        response.headers.pop("server", None)
        
        return response


class LocalhostOnlyMiddleware(BaseHTTPMiddleware):
    """Enforce localhost-only access with IP validation"""
    
    def __init__(self, app: ASGIApp, allowed_ips: Set[str] = None):
        super().__init__(app)
        self.allowed_ips = allowed_ips or LOCALHOST_IPS
        self.blocked_attempts: Dict[str, List[datetime]] = {}
        self.max_attempts = MAX_BLOCKED_ATTEMPTS
        self.blocking_duration = timedelta(minutes=BLOCKING_DURATION_MINUTES)
    
    async def dispatch(self, request: Request, call_next):
        client_ip = self._get_client_ip(request)
        
        # Check if IP is blocked
        if self._is_ip_blocked(client_ip):
            logger.warning(f"Blocked request from rate-limited IP: {client_ip}")
            return PlainTextResponse(
                "Too many invalid requests. Access temporarily blocked.",
                status_code=HTTP_STATUS_TOO_MANY_REQUESTS
            )
        
        # Validate localhost access
        if not self._is_allowed_ip(client_ip):
            self._record_blocked_attempt(client_ip)
            logger.warning(f"Blocked non-localhost admin access from {client_ip}")
            
            # Return generic error to avoid information disclosure
            return PlainTextResponse(
                "Admin interface only accessible from localhost",
                status_code=HTTP_STATUS_FORBIDDEN
            )
        
        return await call_next(request)
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP with support for proxies"""
        # Check forwarded headers (be careful with these in production)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        forwarded = request.headers.get("x-forwarded")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        # Get from connection info
        if hasattr(request, "client") and request.client:
            return request.client.host
        
        return "unknown"
    
    def _is_allowed_ip(self, ip: str) -> bool:
        """Check if IP is in allowed list"""
        if ip in self.allowed_ips:
            return True
        
        try:
            ip_obj = ipaddress.ip_address(ip)
            
            # Check if it's a loopback address
            if ip_obj.is_loopback:
                return True
            
            # Check if it's a private address (for development)
            # Note: In production, you might want to remove this
            if ip_obj.is_private and ip.startswith('192.168.'):
                logger.warning(f"Allowing private IP {ip} - remove in production")
                return True
        
        except ValueError:
            pass
        
        return False
    
    def _is_ip_blocked(self, ip: str) -> bool:
        """Check if IP is currently blocked"""
        if ip not in self.blocked_attempts:
            return False
        
        # Clean old attempts
        now = datetime.utcnow()
        self.blocked_attempts[ip] = [
            attempt for attempt in self.blocked_attempts[ip]
            if now - attempt < self.blocking_duration
        ]
        
        return len(self.blocked_attempts[ip]) >= self.max_attempts
    
    def _record_blocked_attempt(self, ip: str):
        """Record a blocked access attempt"""
        now = datetime.utcnow()
        
        if ip not in self.blocked_attempts:
            self.blocked_attempts[ip] = []
        
        self.blocked_attempts[ip].append(now)


class AuditLoggingMiddleware(BaseHTTPMiddleware):
    """Comprehensive audit logging for admin operations"""
    
    def __init__(self, app: ASGIApp, log_file: Optional[str] = None):
        super().__init__(app)
        self.log_file = Path(log_file) if log_file else Path(DEFAULT_AUDIT_LOG_PATH)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup audit logger
        self.audit_logger = logging.getLogger("admin_audit")
        handler = logging.FileHandler(self.log_file)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.audit_logger.addHandler(handler)
        self.audit_logger.setLevel(logging.INFO)
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # Extract request info
        client_ip = getattr(request.client, "host", "unknown") if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        method = request.method
        url = str(request.url)
        
        # Log request start
        audit_entry = {
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "admin_request_start",
            "client_ip": client_ip,
            "user_agent": user_agent,
            "method": method,
            "url": url,
            "headers": dict(request.headers)
        }
        
        self.audit_logger.info(json.dumps(audit_entry))
        
        # Add request ID to request state
        request.state.request_id = request_id
        
        try:
            response = await call_next(request)
            status_code = response.status_code
            
            # Log successful response
            audit_entry.update({
                "event_type": "admin_request_complete",
                "status_code": status_code,
                "duration_ms": round((time.time() - start_time) * 1000, 2),
                "success": True
            })
            
        except Exception as e:
            # Log exception
            audit_entry.update({
                "event_type": "admin_request_error", 
                "error": str(e),
                "duration_ms": round((time.time() - start_time) * 1000, 2),
                "success": False
            })
            self.audit_logger.error(json.dumps(audit_entry))
            raise
        
        self.audit_logger.info(json.dumps(audit_entry))
        return response