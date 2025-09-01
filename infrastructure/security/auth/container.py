"""Authentication Module Dependency Injection Container.

Provides dependency injection for authentication services following
the Dependency Inversion Principle. Extracted from the God class
to enable proper service composition and testing.
"""

import logging
from typing import Dict, Any, Optional

from .interfaces import IAuthenticationService, ISessionManager, IMFAService
from .services import SessionService, AuthenticationService, MFAService
from .handlers import AuthHandlers, MFAHandlers, SessionHandlers

logger = logging.getLogger(__name__)


class AuthContainer:
    """Dependency injection container for authentication services.

    Manages the lifecycle and dependencies of authentication services
    following the Service Locator and Dependency Injection patterns.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize auth container."""
        self.config = config or {}
        self._services: Dict[str, Any] = {}
        self._initialized = False

    async def initialize(self, rbac_system=None, encryption_service=None) -> None:
        """Initialize all authentication services with proper dependency injection."""
        try:
            logger.info("Initializing authentication module services...")

            # Core services configuration
            session_config = self.config.get("session", {})
            mfa_config = self.config.get("mfa", {})
            auth_config = self.config.get("authentication", {})

            # Initialize session manager first (no dependencies)
            session_service = SessionService(session_config)
            await session_service.initialize()
            self._services["session_manager"] = session_service

            # Initialize MFA service (no dependencies on other auth services)
            mfa_service = MFAService(mfa_config)
            self._services["mfa_service"] = mfa_service

            # Initialize authentication service (depends on session and MFA services)
            auth_service = AuthenticationService(
                session_manager=session_service, mfa_service=mfa_service, rbac_system=rbac_system, config=auth_config
            )
            self._services["auth_service"] = auth_service

            # Initialize handlers (depend on services)
            auth_handlers = AuthHandlers(auth_service=auth_service, mfa_service=mfa_service, rbac_system=rbac_system)
            self._services["auth_handlers"] = auth_handlers

            mfa_handlers = MFAHandlers(mfa_service=mfa_service, encryption_service=encryption_service)
            self._services["mfa_handlers"] = mfa_handlers

            session_handlers = SessionHandlers(session_manager=session_service)
            self._services["session_handlers"] = session_handlers

            self._initialized = True
            logger.info("Authentication module services initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize authentication services: {e}")
            raise

    async def close(self) -> None:
        """Close all services."""
        try:
            if "session_manager" in self._services:
                await self._services["session_manager"].close()

            self._services.clear()
            self._initialized = False
            logger.info("Authentication module services closed")

        except Exception as e:
            logger.error(f"Error closing authentication services: {e}")

    def get_session_manager(self) -> ISessionManager:
        """Get session manager service."""
        self._ensure_initialized()
        return self._services["session_manager"]

    def get_auth_service(self) -> IAuthenticationService:
        """Get authentication service."""
        self._ensure_initialized()
        return self._services["auth_service"]

    def get_mfa_service(self) -> IMFAService:
        """Get MFA service."""
        self._ensure_initialized()
        return self._services["mfa_service"]

    def get_auth_handlers(self) -> AuthHandlers:
        """Get authentication handlers."""
        self._ensure_initialized()
        return self._services["auth_handlers"]

    def get_mfa_handlers(self) -> MFAHandlers:
        """Get MFA handlers."""
        self._ensure_initialized()
        return self._services["mfa_handlers"]

    def get_session_handlers(self) -> SessionHandlers:
        """Get session handlers."""
        self._ensure_initialized()
        return self._services["session_handlers"]

    def register_routes(self, app, auth_prefix: str = "/auth") -> None:
        """Register all authentication routes."""
        try:
            self._ensure_initialized()

            # Register authentication routes
            auth_handlers = self.get_auth_handlers()
            auth_handlers.register_routes(app, auth_prefix)

            # Register MFA routes
            mfa_handlers = self.get_mfa_handlers()
            mfa_handlers.register_routes(app, f"{auth_prefix}/mfa")

            # Register session routes
            session_handlers = self.get_session_handlers()
            session_handlers.register_routes(app, f"{auth_prefix}/sessions")

            logger.info(f"Authentication routes registered with prefix: {auth_prefix}")

        except Exception as e:
            logger.error(f"Failed to register authentication routes: {e}")
            raise

    async def health_check(self) -> Dict[str, Any]:
        """Get health status of all authentication services."""
        try:
            health = {"status": "healthy", "services": {}}

            if self._initialized:
                # Session manager health
                session_manager = self.get_session_manager()
                session_health = await session_manager.health_check()
                health["services"]["session_manager"] = session_health

                # MFA service health
                health["services"]["mfa_service"] = {
                    "status": "healthy",
                    "totp_available": hasattr(self._services["mfa_service"], "totp_window"),
                }

                # Auth service health
                health["services"]["auth_service"] = {
                    "status": "healthy",
                    "jwt_available": True,  # Assuming JWT is available
                }

            else:
                health["status"] = "not_initialized"
                health["message"] = "Authentication services not initialized"

            return health

        except Exception as e:
            logger.error(f"Authentication health check failed: {e}")
            return {"status": "error", "error": str(e)}

    def get_service(self, service_name: str) -> Optional[Any]:
        """Get service by name (generic accessor)."""
        self._ensure_initialized()
        return self._services.get(service_name)

    def _ensure_initialized(self) -> None:
        """Ensure container is initialized."""
        if not self._initialized:
            raise RuntimeError("Authentication container not initialized. Call initialize() first.")

    @property
    def is_initialized(self) -> bool:
        """Check if container is initialized."""
        return self._initialized

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
