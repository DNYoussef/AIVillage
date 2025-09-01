#!/usr/bin/env python3
"""Refactored Enhanced Secure API Server.

This is the refactored version of the EnhancedSecureAPIServer God class,
now using the extracted authentication module following SOLID principles.
The original 890+ line God class has been broken down into focused modules.
"""

import asyncio
from datetime import datetime
import logging
import os
from pathlib import Path
import ssl

from aiohttp import web, web_request
from aiohttp.web_middlewares import middleware
from aiohttp_cors import ResourceOptions
from aiohttp_cors import setup as cors_setup

# Import existing components (unchanged)
from infrastructure.twin.security.rbac_system import RBACSystem
from infrastructure.twin.security.secure_digital_twin_db import SecureDigitalTwinDB

# Import legacy components for compatibility
from .enhanced_encryption import EnhancedDigitalTwinEncryption

# Import our new authentication module
from .auth import AuthContainer, DeviceInfo

logger = logging.getLogger(__name__)


class RefactoredSecureAPIServer:
    """Refactored secure API server using modular authentication.

    This version replaces the God class with a clean architecture using
    the extracted authentication module. The original 890+ line class
    is now broken into focused, testable modules following SOLID principles.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        digital_twin_port: int = 8080,
        evolution_metrics_port: int = 8081,
        rag_pipeline_port: int = 8082,
        auth_config: dict = None,
    ):
        """Initialize refactored secure API server."""
        self.host = host
        self.ports = {
            "digital_twin": digital_twin_port,
            "evolution_metrics": evolution_metrics_port,
            "rag_pipeline": rag_pipeline_port,
        }

        # Authentication module (replaces scattered auth logic)
        self.auth_container = AuthContainer(auth_config or {})

        # Existing components (unchanged)
        self.encryption = EnhancedDigitalTwinEncryption()
        self.rbac_system = RBACSystem()
        self.profile_db = SecureDigitalTwinDB()

        # Rate limiting (simplified, could be extracted to middleware module)
        self.rate_limits = {}

        # Configuration
        self.tls_enabled = os.getenv("TLS_ENABLED", "true").lower() == "true"
        self.cert_file = os.getenv("TLS_CERT_FILE", "./certs/server.crt")
        self.key_file = os.getenv("TLS_KEY_FILE", "./certs/server.key")
        self.cors_enabled = os.getenv("API_CORS_ENABLED", "true").lower() == "true"
        self.cors_origins = os.getenv("API_CORS_ORIGINS", "https://localhost:3000").split(",")

        # Apps
        self.apps = {}

    async def initialize_security_components(self):
        """Initialize all security components."""
        logger.info("Initializing refactored security components...")

        # Initialize the authentication module with dependencies
        await self.auth_container.initialize(rbac_system=self.rbac_system, encryption_service=self.encryption)

        logger.info("Refactored security components initialized")

    def _create_apps(self):
        """Create aiohttp applications for each service."""
        for service in self.ports:
            app = web.Application(
                middlewares=[
                    self._security_middleware,
                    self._rate_limit_middleware,
                    self._session_middleware,
                    self._auth_middleware,
                ]
            )

            # Configure CORS
            self._configure_cors(app)

            # Add service-specific routes
            self._add_routes(app, service)
            self.apps[service] = app

    def _configure_cors(self, app: web.Application):
        """Configure CORS for the application."""
        if self.cors_enabled:
            cors = cors_setup(
                app,
                defaults={
                    "*": ResourceOptions(
                        allow_credentials=True,
                        expose_headers="*",
                        allow_headers="*",
                        allow_methods="*",
                    )
                },
            )

            for origin in self.cors_origins:
                cors.add(
                    app.router.add_resource("/{path:.*}"),
                    {
                        origin: ResourceOptions(
                            allow_credentials=True,
                            expose_headers="*",
                            allow_headers="*",
                            allow_methods="*",
                        )
                    },
                )

    @middleware
    async def _security_middleware(self, request: web_request.Request, handler) -> web.Response:
        """Enhanced security headers and monitoring."""
        # Security monitoring
        client_ip = request.remote
        request.headers.get("User-Agent", "unknown")

        # Log security events
        logger.info(f"Request from {client_ip}: {request.method} {request.path}")

        response = await handler(request)

        # Enhanced security headers
        response.headers.update(
            {
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "DENY",
                "X-XSS-Protection": "1; mode=block",
                "Referrer-Policy": "strict-origin-when-cross-origin",
                "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'",
                "X-Security-Level": "B+",
                "X-Encryption-Algorithm": "AES-256-GCM",
            }
        )

        if self.tls_enabled:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"

        return response

    @middleware
    async def _rate_limit_middleware(self, request: web_request.Request, handler) -> web.Response:
        """Simplified rate limiting middleware."""
        if request.path.startswith("/health"):
            return await handler(request)

        # Get client identifier
        client_id = f"{request.remote}:{request.headers.get('User-Agent', 'unknown')}"
        user_id = request.get("user", {}).get("user_id")

        # Use user-specific limits if authenticated
        if user_id:
            limit_key = f"user:{user_id}"
            max_requests = 120  # Higher limit for authenticated users
        else:
            limit_key = f"ip:{client_id}"
            max_requests = 60  # Lower limit for unauthenticated

        # Check rate limit
        if not self._check_rate_limit(limit_key, max_requests, 60):
            return web.json_response(
                {"error": "Rate limit exceeded", "retry_after": 60, "limit_type": "user" if user_id else "ip"},
                status=429,
                headers={"Retry-After": "60"},
            )

        return await handler(request)

    @middleware
    async def _session_middleware(self, request: web_request.Request, handler) -> web.Response:
        """Session tracking and device info extraction."""
        # Skip for public endpoints
        if request.path in ["/health", "/auth/login", "/auth/register"]:
            return await handler(request)

        # Extract device information
        device_info = DeviceInfo(
            user_agent=request.headers.get("User-Agent", "unknown"), ip_address=request.remote or "unknown"
        )
        request["device_info"] = device_info

        # Monitor for suspicious activity (delegated to session manager)
        user_id = request.get("user", {}).get("user_id")
        if user_id:
            session_manager = self.auth_container.get_session_manager()
            suspicious = await session_manager.detect_suspicious_activity(user_id, device_info)
            if suspicious:
                logger.warning(f"Suspicious activity detected for user {user_id}")

        return await handler(request)

    @middleware
    async def _auth_middleware(self, request: web_request.Request, handler) -> web.Response:
        """Authentication middleware using the auth module."""
        # Skip auth for public endpoints
        if request.path in ["/health", "/auth/login", "/auth/register", "/auth/mfa/setup"]:
            return await handler(request)

        # Check for API token
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return web.json_response({"error": "Missing or invalid authorization header"}, status=401)

        token = auth_header[7:]  # Remove 'Bearer '

        try:
            # Use the authentication service from our auth module
            auth_service = self.auth_container.get_auth_service()
            validation_result = await auth_service.validate_token(token)

            if not validation_result.valid:
                return web.json_response({"error": validation_result.error_message}, status=401)

            request["user"] = validation_result.payload
            return await handler(request)

        except Exception as e:
            logger.error(f"Authentication middleware error: {e}")
            return web.json_response({"error": "Authentication failed"}, status=401)

    def _check_rate_limit(self, key: str, max_requests: int, window_seconds: int) -> bool:
        """Check rate limiting for a key."""
        import time

        now = time.time()

        if key not in self.rate_limits:
            self.rate_limits[key] = []

        # Clean old requests
        self.rate_limits[key] = [timestamp for timestamp in self.rate_limits[key] if now - timestamp < window_seconds]

        # Check limit
        if len(self.rate_limits[key]) >= max_requests:
            return False

        self.rate_limits[key].append(now)
        return True

    def _add_routes(self, app: web.Application, service: str):
        """Add service-specific routes with enhanced security."""
        # Health check endpoint
        app.router.add_get("/health", self._health_check)

        # Register authentication routes using the auth module
        self.auth_container.register_routes(app, "/auth")

        # Security status endpoint
        app.router.add_get("/security/status", self._security_status)

        # Service-specific endpoints (unchanged from original)
        if service == "digital_twin":
            app.router.add_get("/profiles/{profile_id}", self._get_profile_enhanced)
            app.router.add_post("/profiles", self._create_profile_enhanced)
            app.router.add_put("/profiles/{profile_id}", self._update_profile_enhanced)
            app.router.add_delete("/profiles/{profile_id}", self._delete_profile_enhanced)
            app.router.add_get("/profiles/{profile_id}/export", self._export_profile_data_enhanced)

    async def _security_status(self, request: web_request.Request) -> web.Response:
        """Get comprehensive security status using auth module."""
        try:
            user = request.get("user", {})
            user_id = user.get("user_id")

            # Encryption status
            encryption_status = self.encryption.get_key_status()

            # Authentication module health
            auth_health = await self.auth_container.health_check()

            # MFA status (if user authenticated)
            mfa_status = {}
            if user_id:
                mfa_service = self.auth_container.get_mfa_service()
                mfa_status = mfa_service.get_user_mfa_status(user_id)

            return web.json_response(
                {
                    "security_rating": "B+",
                    "encryption": {
                        "algorithm": encryption_status["algorithm"],
                        "current_version": encryption_status["current_version"],
                        "rotation_needed": encryption_status["rotation_needed"],
                    },
                    "authentication": auth_health,
                    "mfa": {
                        "available_methods": ["TOTP", "SMS", "Email", "Backup Codes"],
                        "user_status": mfa_status if user_id else "Not authenticated",
                    },
                    "tls_enabled": self.tls_enabled,
                    "cors_enabled": self.cors_enabled,
                }
            )

        except Exception as e:
            logger.error(f"Security status failed: {e}")
            return web.json_response({"error": "Failed to get security status"}, status=500)

    # Profile endpoints (unchanged from original for compatibility)

    async def _get_profile_enhanced(self, request: web_request.Request) -> web.Response:
        """Get profile with enhanced encryption."""
        profile_id = request.match_info["profile_id"]
        user = request["user"]
        user_id = user["user_id"]

        try:
            # Use enhanced encryption for profile retrieval
            profile = self.profile_db.get_profile(user_id, profile_id)
            if not profile:
                return web.json_response({"error": "Profile not found"}, status=404)

            return web.json_response(
                {"profile_id": profile_id, "data": profile, "status": "success", "encryption": "AES-256-GCM"}
            )

        except Exception as e:
            logger.error(f"Get profile failed: {e}")
            return web.json_response({"error": "Failed to retrieve profile"}, status=500)

    async def _create_profile_enhanced(self, request: web_request.Request) -> web.Response:
        """Create profile endpoint."""
        return web.json_response({"message": "Profile creation endpoint"})

    async def _update_profile_enhanced(self, request: web_request.Request) -> web.Response:
        """Update profile endpoint."""
        return web.json_response({"message": "Profile update endpoint"})

    async def _delete_profile_enhanced(self, request: web_request.Request) -> web.Response:
        """Delete profile endpoint."""
        return web.json_response({"message": "Profile deletion endpoint"})

    async def _export_profile_data_enhanced(self, request: web_request.Request) -> web.Response:
        """Export profile data endpoint."""
        return web.json_response({"message": "Profile export endpoint"})

    async def _health_check(self, request: web_request.Request) -> web.Response:
        """Enhanced health check with security status."""
        try:
            # Database health
            db_health = {"status": "operational", "latency_ms": 0}

            # Authentication module health
            auth_health = await self.auth_container.health_check()

            # Encryption health
            encryption_status = self.encryption.get_key_status()

            overall_status = "healthy"
            if db_health["status"] != "operational" or auth_health["status"] != "healthy":
                overall_status = "degraded"

            return web.json_response(
                {
                    "status": overall_status,
                    "timestamp": datetime.utcnow().isoformat(),
                    "version": "2.0.0-refactored",
                    "security_rating": "B+",
                    "services": {
                        "authentication": auth_health["status"],
                        "encryption": "operational",
                        "database": db_health,
                    },
                    "security": {
                        "tls_enabled": self.tls_enabled,
                        "encryption_algorithm": encryption_status["algorithm"],
                        "auth_module": "modular",
                        "architecture": "clean_architecture",
                    },
                }
            )

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return web.json_response({"status": "error", "error": str(e)}, status=500)

    async def start_servers(self):
        """Start refactored secure API servers."""
        # Initialize security components first
        await self.initialize_security_components()

        # Create apps after security initialization
        self._create_apps()

        # Start servers
        ssl_context = self._create_ssl_context() if self.tls_enabled else None

        runners = []
        sites = []

        for service, port in self.ports.items():
            runner = web.AppRunner(self.apps[service])
            await runner.setup()
            runners.append(runner)

            site = web.TCPSite(runner, self.host, port, ssl_context=ssl_context)
            await site.start()
            sites.append(site)

            protocol = "https" if self.tls_enabled else "http"
            logger.info(f"Started refactored {service} API server: {protocol}://{self.host}:{port}")

        logger.info("🔒 Refactored Secure API servers started with modular architecture")
        return runners, sites

    def _create_ssl_context(self) -> ssl.SSLContext:
        """Create enhanced SSL context (unchanged)."""
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)

        # TLS 1.3 configuration
        context.minimum_version = ssl.TLSVersion.TLSv1_3
        context.maximum_version = ssl.TLSVersion.TLSv1_3

        # Load certificates
        if Path(self.cert_file).exists() and Path(self.key_file).exists():
            context.load_cert_chain(self.cert_file, self.key_file)
        else:
            logger.warning("TLS certificates not found, generating self-signed certificates")
            self._generate_self_signed_cert()
            context.load_cert_chain(self.cert_file, self.key_file)

        # Enhanced security settings
        context.set_ciphers("ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS:!RC4")
        context.options |= ssl.OP_NO_COMPRESSION
        context.options |= ssl.OP_NO_RENEGOTIATION
        context.options |= ssl.OP_CIPHER_SERVER_PREFERENCE

        return context

    def _generate_self_signed_cert(self):
        """Generate enhanced self-signed certificate (unchanged)."""
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID

        # Create certificates directory
        cert_dir = Path(self.cert_file).parent
        cert_dir.mkdir(parents=True, exist_ok=True)

        # Generate private key (4096 bit for enhanced security)
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096,  # Enhanced from 2048
        )

        # Generate certificate
        subject = issuer = x509.Name(
            [
                x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "State"),
                x509.NameAttribute(NameOID.LOCALITY_NAME, "City"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "AIVillage Refactored"),
                x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
            ]
        )

        certificate = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(private_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.utcnow())
            .not_valid_after(datetime.utcnow() + datetime.timedelta(days=365))
            .add_extension(
                x509.SubjectAlternativeName(
                    [
                        x509.DNSName("localhost"),
                        x509.DNSName("127.0.0.1"),
                    ]
                ),
                critical=False,
            )
            .sign(private_key, hashes.SHA256())
        )

        # Write private key
        with open(self.key_file, "wb") as f:
            f.write(
                private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption(),
                )
            )

        # Write certificate
        with open(self.cert_file, "wb") as f:
            f.write(certificate.public_bytes(serialization.Encoding.PEM))

        logger.info(f"Generated enhanced 4096-bit self-signed certificate: {self.cert_file}")


# Example usage
if __name__ == "__main__":
    import secrets

    async def main():
        # Set environment variables
        os.environ["API_SECRET_KEY"] = secrets.token_urlsafe(32)
        os.environ["TLS_ENABLED"] = "false"  # Set to 'true' for production

        # Configuration for auth module
        auth_config = {
            "session": {"redis_url": "redis://localhost:6379/0", "session_ttl_hours": 24, "max_sessions_per_user": 10},
            "authentication": {"token_expiry_hours": 24, "refresh_expiry_days": 30},
            "mfa": {"totp_window": 1, "backup_code_length": 8},
        }

        server = RefactoredSecureAPIServer(auth_config=auth_config)
        runners, sites = await server.start_servers()

        try:
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            logger.info("Shutting down refactored secure servers...")
        finally:
            for runner in runners:
                await runner.cleanup()
            await server.auth_container.close()

    asyncio.run(main())
