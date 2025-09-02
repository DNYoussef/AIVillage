"""MFA Request Handlers.

Handles multi-factor authentication related HTTP endpoints.
Extracted from the EnhancedSecureAPIServer for better modularity.
"""

import logging

from aiohttp import web, web_request

from ..interfaces import IMFAService, MFAMethodType

logger = logging.getLogger(__name__)


class MFAHandlers:
    """MFA request handlers.

    Handles MFA setup, verification, and management endpoints.
    """

    def __init__(self, mfa_service: IMFAService, encryption_service=None):
        """Initialize MFA handlers."""
        self.mfa_service = mfa_service
        self.encryption_service = encryption_service

    async def handle_mfa_setup(self, request: web_request.Request) -> web.Response:
        """Handle MFA setup for user."""
        try:
            user = request.get("user", {})
            user_id = user.get("user_id")

            if not user_id:
                return web.json_response({"error": "Authentication required"}, status=401)

            data = await request.json()
            method = data.get("method", "totp")

            try:
                mfa_method = MFAMethodType(method)
            except ValueError:
                return web.json_response({"error": f"Unsupported MFA method: {method}"}, status=400)

            if mfa_method == MFAMethodType.TOTP:
                # Get user email (in real implementation, from database)
                email = user.get("email", f"user-{user_id}@example.com")

                setup_result = self.mfa_service.setup_totp(user_id, email)

                if not setup_result.success:
                    return web.json_response({"error": setup_result.error_message}, status=500)

                # In production, encrypt and store the secret securely
                if self.encryption_service and setup_result.secret:
                    self.encryption_service.encrypt_sensitive_field(
                        setup_result.secret, "totp_secret"
                    )
                    # Store encrypted_secret in database

                return web.json_response(
                    {
                        "method": method,
                        "qr_code": setup_result.qr_code,
                        "backup_codes": setup_result.backup_codes,
                        "setup_complete": False,  # Requires verification
                        "message": "Scan QR code with authenticator app and verify with a code",
                    }
                )

            else:
                return web.json_response({"error": f"MFA method '{method}' not yet implemented"}, status=400)

        except Exception as e:
            logger.error(f"MFA setup failed: {e}")
            return web.json_response({"error": "MFA setup failed"}, status=500)

    async def handle_mfa_verify(self, request: web_request.Request) -> web.Response:
        """Handle MFA token verification."""
        try:
            data = await request.json()
            user = request.get("user", {})
            user_id = user.get("user_id")

            if not user_id:
                return web.json_response({"error": "Authentication required"}, status=401)

            method = data.get("method")
            token = data.get("token")

            if not method or not token:
                return web.json_response({"error": "Method and token required"}, status=400)

            try:
                mfa_method = MFAMethodType(method)
            except ValueError:
                return web.json_response({"error": f"Invalid MFA method: {method}"}, status=400)

            # Verify MFA token
            verified = self.mfa_service.verify_mfa(user_id, mfa_method, token)

            if verified and mfa_method == MFAMethodType.TOTP:
                # Enable TOTP after successful verification
                self.mfa_service.enable_totp(user_id)

            return web.json_response(
                {
                    "verified": verified,
                    "method": method,
                    "message": "MFA verified successfully" if verified else "Invalid MFA token",
                }
            )

        except Exception as e:
            logger.error(f"MFA verification failed: {e}")
            return web.json_response({"error": "MFA verification failed"}, status=500)

    async def handle_mfa_disable(self, request: web_request.Request) -> web.Response:
        """Handle MFA disable for user."""
        try:
            user = request.get("user", {})
            user_id = user.get("user_id")

            if not user_id:
                return web.json_response({"error": "Authentication required"}, status=401)

            data = await request.json()
            method = data.get("method")
            confirmation_password = data.get("password")

            if not method:
                return web.json_response({"error": "MFA method required"}, status=400)

            if not confirmation_password:
                return web.json_response({"error": "Password confirmation required"}, status=400)

            # In production, verify password before disabling MFA
            # For now, just disable the method
            try:
                mfa_method = MFAMethodType(method)
            except ValueError:
                return web.json_response({"error": f"Invalid MFA method: {method}"}, status=400)

            success = self.mfa_service.disable_mfa(user_id, mfa_method)

            if success:
                logger.info(f"MFA {method} disabled for user {user_id}")
                return web.json_response({"message": f"MFA {method} disabled successfully"})
            else:
                return web.json_response({"error": "Failed to disable MFA"}, status=500)

        except Exception as e:
            logger.error(f"MFA disable failed: {e}")
            return web.json_response({"error": "MFA disable failed"}, status=500)

    async def handle_get_backup_codes(self, request: web_request.Request) -> web.Response:
        """Get backup codes for user."""
        try:
            user = request.get("user", {})
            user_id = user.get("user_id")

            if not user_id:
                return web.json_response({"error": "Authentication required"}, status=401)

            # Generate new backup codes
            backup_codes = self.mfa_service.generate_backup_codes(user_id)

            if backup_codes:
                return web.json_response(
                    {
                        "backup_codes": backup_codes,
                        "message": "Store these codes in a safe place. Each code can only be used once.",
                    }
                )
            else:
                return web.json_response({"error": "Failed to generate backup codes"}, status=500)

        except Exception as e:
            logger.error(f"Get backup codes failed: {e}")
            return web.json_response({"error": "Failed to get backup codes"}, status=500)

    async def handle_mfa_status(self, request: web_request.Request) -> web.Response:
        """Get MFA status for user."""
        try:
            user = request.get("user", {})
            user_id = user.get("user_id")

            if not user_id:
                return web.json_response({"error": "Authentication required"}, status=401)

            mfa_status = self.mfa_service.get_user_mfa_status(user_id)

            return web.json_response(
                {
                    "totp_enabled": mfa_status.totp_enabled,
                    "sms_enabled": mfa_status.sms_enabled,
                    "email_enabled": mfa_status.email_enabled,
                    "backup_codes_available": mfa_status.backup_codes_available,
                    "methods_available": mfa_status.methods_available or [],
                    "mfa_required": self.mfa_service.is_mfa_required(user_id),
                }
            )

        except Exception as e:
            logger.error(f"Get MFA status failed: {e}")
            return web.json_response({"error": "Failed to get MFA status"}, status=500)

    def register_routes(self, app: web.Application, prefix: str = "/auth/mfa"):
        """Register MFA routes."""
        app.router.add_post(f"{prefix}/setup", self.handle_mfa_setup)
        app.router.add_post(f"{prefix}/verify", self.handle_mfa_verify)
        app.router.add_post(f"{prefix}/disable", self.handle_mfa_disable)
        app.router.add_get(f"{prefix}/backup-codes", self.handle_get_backup_codes)
        app.router.add_get(f"{prefix}/status", self.handle_mfa_status)
