#!/usr/bin/env python3
"""
Refactored Secure Admin Server Implementation
Modular architecture with clean separation of concerns.
"""

import asyncio
import logging
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from .auth import create_session_service
from .middleware import create_security_middleware_stack
from .handlers import create_handler_registry
from .security import create_security_context_service
from .constants import (
    DEFAULT_BIND_INTERFACE, DEFAULT_ADMIN_PORT, CORS_ALLOWED_ORIGINS,
    CORS_ALLOWED_METHODS, CORS_ALLOWED_HEADERS
)

from ..boundaries.secure_boundary_contracts import (
    SecurityException, SecurityViolationType, SecurityBoundaryFactory
)

# Configure logging
logger = logging.getLogger(__name__)


class SecureAdminServer:
    """Refactored secure admin server with modular architecture"""
    
    def __init__(self, 
                 security_boundary_factory: SecurityBoundaryFactory,
                 bind_interface: str = DEFAULT_BIND_INTERFACE,
                 port: int = DEFAULT_ADMIN_PORT,
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
        
        # Initialize services using dependency injection
        self._initialize_services(security_boundary_factory)
        
        # Create FastAPI app
        self.app = self._create_secure_app()
    
    def _initialize_services(self, security_boundary_factory: SecurityBoundaryFactory):
        """Initialize all services with dependency injection"""
        
        # Core services
        self.security_factory = security_boundary_factory
        self.admin_boundary = security_boundary_factory.create_admin_boundary()
        
        # Create modular services
        self.session_service = create_session_service()
        self.security_context_service = create_security_context_service(self.session_service)
        self.handlers = create_handler_registry(self.session_service, self.admin_boundary)
    
    def _create_secure_app(self) -> FastAPI:
        """Create FastAPI app with modular security components"""
        
        app = FastAPI(
            title="AIVillage Secure Admin Interface",
            description="Localhost-only admin interface with comprehensive security",
            version="2.0.0",
            # Disable docs in production
            openapi_url="/openapi.json" if self.debug else None,
            docs_url="/docs" if self.debug else None,
            redoc_url=None  # Disable redoc
        )
        
        # Apply security middleware stack
        middleware_classes = create_security_middleware_stack()
        for middleware_class in middleware_classes:
            if middleware_class.__name__ == 'AuditLoggingMiddleware':
                app.add_middleware(middleware_class, log_file="logs/admin_audit.log")
            elif middleware_class.__name__ == 'LocalhostOnlyMiddleware':
                app.add_middleware(middleware_class, allowed_ips={'127.0.0.1', '::1'})
            else:
                app.add_middleware(middleware_class)
        
        # Standard middleware
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])
        
        # CORS - NEVER allow wildcard for admin interfaces
        app.add_middleware(
            CORSMiddleware,
            allow_origins=CORS_ALLOWED_ORIGINS,
            allow_credentials=True,
            allow_methods=CORS_ALLOWED_METHODS,
            allow_headers=CORS_ALLOWED_HEADERS
        )
        
        # Setup routes with modular handlers
        self._setup_routes(app)
        return app
    
    def _setup_routes(self, app: FastAPI):
        """Setup routes using modular handlers"""
        
        @app.middleware("http")
        async def security_boundary_middleware(request: Request, call_next):
            """Apply admin security boundary to all admin routes"""
            
            # Skip security for health check and auth endpoints
            if request.url.path in ["/health", "/auth/login", "/auth/mfa"]:
                return await call_next(request)
            
            # Extract security context using service
            context = await self.security_context_service.extract_security_context(request)
            
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
                "localhost_only": True,
                "version": "2.0.0",
                "architecture": "modular"
            })
        
        # Authentication endpoints
        @app.post("/auth/login")
        async def admin_login(request: Request, credentials: dict):
            return await self.handlers['auth'].admin_login(request, credentials)
        
        @app.post("/auth/mfa")
        async def verify_mfa(request: Request, mfa_data: dict):
            return await self.handlers['auth'].verify_mfa(request, mfa_data)
        
        @app.post("/auth/logout")
        async def admin_logout(request: Request):
            return await self.handlers['auth'].admin_logout(request)
        
        # Admin endpoints
        @app.get("/admin/system/status")
        async def get_system_status(request: Request):
            return await self.handlers['admin'].get_system_status(request)
        
        @app.get("/admin/security/audit-logs")
        async def get_audit_logs(request: Request, limit: int = 100, offset: int = 0):
            return await self.handlers['admin'].get_audit_logs(request, limit, offset)
        
        @app.get("/admin/security/threats")
        async def get_threat_analysis(request: Request):
            return await self.handlers['admin'].get_threat_analysis(request)
        
        @app.post("/admin/security/scan")
        async def trigger_security_scan(request: Request):
            return await self.handlers['admin'].trigger_security_scan(request)
        
        @app.get("/admin/users")
        async def get_users(request: Request):
            return await self.handlers['admin'].get_users(request)
        
        @app.post("/admin/emergency/shutdown")
        async def emergency_shutdown(request: Request, confirmation: dict):
            return await self.handlers['admin'].emergency_shutdown(request, confirmation)
    
    async def start_server(self):
        """Start the secure admin server"""
        
        logger.info(f"Starting Refactored Secure Admin Server v2.0 on {self.bind_interface}:{self.port}")
        logger.info("MODULAR ARCHITECTURE ENABLED:")
        logger.info("  OK Auth module - Session & MFA management")
        logger.info("  OK Middleware module - Security headers, localhost validation, audit logging")
        logger.info("  OK Handlers module - Route handlers with dependency injection")
        logger.info("  OK Security module - Context management & policy enforcement")
        logger.info("  OK Constants module - Centralized configuration")
        logger.info("SECURITY FEATURES:")
        logger.info("  OK Localhost-only binding")
        logger.info("  OK Multi-factor authentication")
        logger.info("  OK Session management")
        logger.info("  OK Security headers")
        logger.info("  OK Audit logging")
        logger.info("  OK IP validation")
        logger.info("  OK Rate limiting")
        
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
    """Example usage with refactored architecture"""
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
    
    # Create and start refactored server
    server = SecureAdminServer(
        security_boundary_factory=factory,
        bind_interface="127.0.0.1",  # CRITICAL: Never 0.0.0.0
        port=3006,
        debug=True
    )
    
    try:
        await server.start_server()
    except KeyboardInterrupt:
        logger.info("Refactored secure admin server shutdown requested")
    except Exception as e:
        logger.error(f"Refactored admin server error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())