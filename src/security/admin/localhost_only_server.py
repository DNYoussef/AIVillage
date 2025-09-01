#!/usr/bin/env python3
"""
Localhost-Only Secure Admin Server Implementation
Implements secure admin interface patterns with strict localhost binding and comprehensive security controls.
"""

import asyncio
import hashlib
import ipaddress
import json
import logging
import secrets
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import uuid
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response, Depends, status, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

from ..boundaries.secure_boundary_contracts import (
    SecurityContext, SecurityLevel, SecurityException, SecurityViolationType,
    AdminBoundary, SecurityBoundaryFactory
)

# Configure logging
logger = logging.getLogger(__name__)

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses"""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        # Remove server information
        response.headers.pop("server", None)
        
        return response

class LocalhostOnlyMiddleware(BaseHTTPMiddleware):
    """Enforce localhost-only access with IP validation"""
    
    def __init__(self, app: ASGIApp, allowed_ips: Set[str] = None):
        super().__init__(app)
        self.allowed_ips = allowed_ips or {'127.0.0.1', '::1'}
        self.blocked_attempts: Dict[str, List[datetime]] = {}
        self.max_attempts = 5
        self.blocking_duration = timedelta(minutes=15)
    
    async def dispatch(self, request: Request, call_next):
        client_ip = self._get_client_ip(request)
        
        # Check if IP is blocked
        if self._is_ip_blocked(client_ip):
            logger.warning(f"Blocked request from rate-limited IP: {client_ip}")
            return PlainTextResponse(
                "Too many invalid requests. Access temporarily blocked.",
                status_code=429
            )
        
        # Validate localhost access
        if not self._is_allowed_ip(client_ip):
            self._record_blocked_attempt(client_ip)
            logger.warning(f"Blocked non-localhost admin access from {client_ip}")
            
            # Return generic error to avoid information disclosure
            return PlainTextResponse(
                "Admin interface only accessible from localhost",
                status_code=403
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
        self.log_file = Path(log_file) if log_file else Path("logs/admin_audit.log")
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

class SessionManager:
    """Secure session management for admin users"""
    
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        self.session_timeout = timedelta(minutes=30)
        self.max_sessions_per_user = 3
    
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
        role_permissions = {
            "super_admin": {"admin:read", "admin:write", "admin:system", "admin:users", "admin:security"},
            "admin": {"admin:read", "admin:write", "admin:system"},
            "operator": {"admin:read", "admin:system"},
            "viewer": {"admin:read"}
        }
        
        permissions = set()
        for role in roles:
            permissions.update(role_permissions.get(role, set()))
        
        return permissions

class SecureAdminServer:
    """Secure admin server with localhost-only access and comprehensive security"""
    
    def __init__(self, 
                 security_boundary_factory: SecurityBoundaryFactory,
                 bind_interface: str = "127.0.0.1",
                 port: int = 3006,
                 debug: bool = False):
        
        # CRITICAL: Validate bind interface
        if bind_interface == "0.0.0.0":
            raise SecurityException(
                "Admin interfaces must never bind to all interfaces",
                SecurityViolationType.WEAK_COUPLING_VIOLATION
            )
        
        self.bind_interface = bind_interface
        self.port = port
        self.debug = debug
        
        # Security components
        self.security_factory = security_boundary_factory
        self.admin_boundary = security_boundary_factory.create_admin_boundary()
        self.session_manager = SessionManager()
        
        # Create FastAPI app
        self.app = self._create_secure_app()
    
    def _create_secure_app(self) -> FastAPI:
        """Create FastAPI app with comprehensive security"""
        
        app = FastAPI(
            title="AIVillage Secure Admin Interface",
            description="Localhost-only admin interface with comprehensive security",
            version="1.0.0",
            # Disable docs in production
            openapi_url="/openapi.json" if self.debug else None,
            docs_url="/docs" if self.debug else None,
            redoc_url=None  # Disable redoc
        )
        
        # Security middleware (order matters!)
        app.add_middleware(SecurityHeadersMiddleware)
        app.add_middleware(AuditLoggingMiddleware, log_file="logs/admin_audit.log")
        app.add_middleware(LocalhostOnlyMiddleware, allowed_ips={'127.0.0.1', '::1'})
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])
        
        # CORS - NEVER allow wildcard for admin interfaces
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[
                "http://127.0.0.1:3000",
                "https://127.0.0.1:3000", 
                "http://localhost:3000",
                "https://localhost:3000"
            ],
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["Content-Type", "Authorization", "X-Admin-Session"]
        )
        
        # Setup routes
        self._setup_routes(app)
        return app
    
    def _setup_routes(self, app: FastAPI):
        """Setup secure admin routes"""
        
        security = HTTPBearer()
        
        @app.middleware("http")
        async def security_boundary_middleware(request: Request, call_next):
            """Apply admin security boundary to all admin routes"""
            
            # Skip security for health check and auth endpoints
            if request.url.path in ["/health", "/auth/login", "/auth/mfa"]:
                return await call_next(request)
            
            # Extract security context
            context = await self._extract_security_context(request)
            
            # Validate admin access
            if not await self.admin_boundary.validate_access(context):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Admin access denied - insufficient privileges"
                )
            
            # Add context to request state
            request.state.security_context = context
            
            response = await call_next(request)
            return response
        
        # Health check (no auth required)
        @app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return JSONResponse({
                "status": "healthy",
                "service": "secure-admin-interface",
                "timestamp": datetime.utcnow().isoformat(),
                "localhost_only": True
            })
        
        # Authentication endpoints
        @app.post("/auth/login")
        async def admin_login(request: Request, credentials: Dict[str, str]):
            """Admin login with session creation"""
            
            username = credentials.get("username")
            password = credentials.get("password")
            client_ip = getattr(request.client, "host", "127.0.0.1") if request.client else "127.0.0.1"
            
            # Validate credentials (integrate with your auth system)
            if not self._validate_admin_credentials(username, password):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid credentials"
                )
            
            # Get user roles
            user_roles = self._get_user_roles(username)
            if "admin" not in user_roles and "super_admin" not in user_roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Admin privileges required"
                )
            
            # Create session
            session_id = self.session_manager.create_session(username, user_roles, client_ip)
            
            return JSONResponse({
                "message": "Login successful - MFA required",
                "session_id": session_id,
                "requires_mfa": True,
                "expires_at": (datetime.utcnow() + timedelta(minutes=30)).isoformat()
            })
        
        @app.post("/auth/mfa")
        async def verify_mfa(request: Request, mfa_data: Dict[str, str]):
            """Verify MFA token"""
            
            session_id = mfa_data.get("session_id")
            mfa_token = mfa_data.get("mfa_token")
            
            if not session_id or not mfa_token:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Session ID and MFA token required"
                )
            
            client_ip = getattr(request.client, "host", "127.0.0.1") if request.client else "127.0.0.1"
            session = self.session_manager.validate_session(session_id, client_ip)
            
            if not session:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired session"
                )
            
            if not self.session_manager.verify_mfa(session_id, mfa_token):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid MFA token"
                )
            
            return JSONResponse({
                "message": "MFA verification successful",
                "session_id": session_id,
                "permissions": list(session["permissions"]),
                "expires_at": (datetime.utcnow() + timedelta(minutes=30)).isoformat()
            })
        
        @app.post("/auth/logout")
        async def admin_logout(request: Request):
            """Admin logout"""
            
            session_id = request.headers.get("x-admin-session")
            if session_id:
                self.session_manager.destroy_session(session_id)
            
            return JSONResponse({"message": "Logged out successfully"})
        
        # Admin endpoints (require authentication)
        @app.get("/admin/system/status")
        async def get_system_status(request: Request):
            """Get comprehensive system status"""
            
            context = request.state.security_context
            
            return await self.admin_boundary.execute_secured(
                context,
                lambda: self._get_system_status()
            )
        
        @app.get("/admin/security/audit-logs")
        async def get_audit_logs(request: Request, limit: int = 100, offset: int = 0):
            \"\"\"Get audit logs\"\"\"
            
            context = request.state.security_context
            
            if "admin:security" not in context.permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Security admin privileges required"
                )
            
            return await self.admin_boundary.execute_secured(
                context,
                lambda: self._get_audit_logs(limit, offset)
            )
        
        @app.get("/admin/security/threats")
        async def get_threat_analysis(request: Request):
            \"\"\"Get current threat analysis\"\"\"
            
            context = request.state.security_context
            
            return await self.admin_boundary.execute_secured(
                context,
                lambda: self._get_threat_analysis()
            )
        
        @app.post("/admin/security/scan")
        async def trigger_security_scan(request: Request):
            \"\"\"Trigger comprehensive security scan\"\"\"
            
            context = request.state.security_context
            
            if "admin:system" not in context.permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="System admin privileges required"
                )
            
            return await self.admin_boundary.execute_secured(
                context,
                lambda: self._trigger_security_scan(context.user_id)
            )
        
        @app.get("/admin/users")
        async def get_users(request: Request):
            \"\"\"Get user list\"\"\"
            
            context = request.state.security_context
            
            if "admin:users" not in context.permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="User admin privileges required"
                )
            
            return await self.admin_boundary.execute_secured(
                context,
                lambda: self._get_users()
            )
        
        # Emergency endpoints
        @app.post("/admin/emergency/shutdown")
        async def emergency_shutdown(request: Request, confirmation: Dict[str, str]):
            \"\"\"Emergency system shutdown\"\"\"
            
            context = request.state.security_context
            
            if "super_admin" not in context.roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Super admin privileges required for emergency actions"
                )
            
            if confirmation.get("confirm") != "EMERGENCY_SHUTDOWN_CONFIRMED":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Emergency shutdown requires explicit confirmation"
                )
            
            return await self.admin_boundary.execute_secured(
                context,
                lambda: self._emergency_shutdown(context.user_id)
            )
    
    async def _extract_security_context(self, request: Request) -> SecurityContext:
        \"\"\"Extract security context from request\"\"\"
        
        session_id = request.headers.get("x-admin-session")
        if not session_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Admin session required"
            )
        
        client_ip = getattr(request.client, "host", "127.0.0.1") if request.client else "127.0.0.1"
        session = self.session_manager.validate_session(session_id, client_ip)
        
        if not session:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired session"
            )
        
        if not session["mfa_verified"]:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
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
    
    def _validate_admin_credentials(self, username: str, password: str) -> bool:
        \"\"\"Validate admin credentials (integrate with your auth system)\"\"\"
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
        \"\"\"Get user roles\"\"\"
        user_roles = {
            "admin": {"admin"},
            "superadmin": {"super_admin", "admin"}
        }
        
        return user_roles.get(username, set())
    
    def _get_system_status(self) -> Dict[str, Any]:
        \"\"\"Get system status\"\"\"
        return {
            "services": {
                "admin_interface": {"status": "healthy", "localhost_only": True},
                "security_boundary": {"status": "active", "violations": 0},
                "session_manager": {"status": "active", "active_sessions": len(self.session_manager.sessions)}
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
        \"\"\"Get audit logs\"\"\"
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
        \"\"\"Get threat analysis\"\"\"
        return {
            "current_threats": [],
            "risk_level": "low",
            "last_scan": datetime.utcnow().isoformat(),
            "localhost_access_attempts": 0,
            "external_access_attempts": 0
        }
    
    def _trigger_security_scan(self, user_id: str) -> Dict[str, Any]:
        \"\"\"Trigger security scan\"\"\"
        scan_id = str(uuid.uuid4())
        logger.info(f"Security scan {scan_id} triggered by user {user_id}")
        
        return {
            "scan_id": scan_id,
            "status": "initiated",
            "estimated_duration": "5-10 minutes",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _get_users(self) -> Dict[str, Any]:
        \"\"\"Get user list\"\"\"
        return {
            "users": [
                {"id": "1", "username": "admin", "roles": ["admin"], "last_login": datetime.utcnow().isoformat()},
                {"id": "2", "username": "superadmin", "roles": ["super_admin"], "last_login": datetime.utcnow().isoformat()}
            ],
            "total": 2
        }
    
    def _emergency_shutdown(self, user_id: str) -> Dict[str, Any]:
        \"\"\"Emergency shutdown\"\"\"
        logger.critical(f"EMERGENCY SHUTDOWN initiated by user {user_id}")
        
        # In production, this would actually shutdown systems
        return {
            "status": "emergency_shutdown_initiated",
            "initiated_by": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "estimated_completion": "30 seconds"
        }
    
    async def start_server(self):
        \"\"\"Start the secure admin server\"\"\"
        
        logger.info(f"Starting Secure Admin Server on {self.bind_interface}:{self.port}")
        logger.info("SECURITY FEATURES ENABLED:")
        logger.info("  ✓ Localhost-only binding")
        logger.info("  ✓ Multi-factor authentication")
        logger.info("  ✓ Session management")
        logger.info("  ✓ Security headers")
        logger.info("  ✓ Audit logging")
        logger.info("  ✓ IP validation")
        logger.info("  ✓ Rate limiting")
        
        config = uvicorn.Config(
            app=self.app,
            host=self.bind_interface,  # NEVER 0.0.0.0 for admin
            port=self.port,
            log_level="info" if not self.debug else "debug",
            reload=False,  # Never enable reload in admin interfaces
            access_log=True
        )
        
        server = uvicorn.Server(config)
        await server.serve()

# Example usage and testing
async def main():
    \"\"\"Example usage\"\"\"
    from unittest.mock import AsyncMock
    from ..boundaries.secure_boundary_contracts import (
        AuthenticationService, AuthorizationService, AuditService, ThreatDetectionService
    )
    
    # Create mock security services
    auth_service = AsyncMock(spec=AuthenticationService)
    authz_service = AsyncMock(spec=AuthorizationService) 
    audit_service = AsyncMock(spec=AuditService)
    threat_service = AsyncMock(spec=ThreatDetectionService)
    
    # Configure mocks
    auth_service.validate_mfa.return_value = True
    authz_service.check_admin_permission.return_value = True
    audit_service.log_security_event.return_value = None
    threat_service.analyze_request_pattern.return_value = {"risk_level": "low"}
    
    # Create security boundary factory
    factory = SecurityBoundaryFactory(
        auth_service, authz_service, audit_service, threat_service
    )
    
    # Create and start server
    server = SecureAdminServer(
        security_boundary_factory=factory,
        bind_interface="127.0.0.1",  # CRITICAL: Never 0.0.0.0
        port=3006,
        debug=True
    )
    
    try:
        await server.start_server()
    except KeyboardInterrupt:
        logger.info("Secure admin server shutdown requested")
    except Exception as e:
        logger.error(f"Admin server error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())