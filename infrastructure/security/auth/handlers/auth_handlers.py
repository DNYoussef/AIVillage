"""Authentication Request Handlers.

Handles authentication-related HTTP endpoints including login, logout,
token refresh, and registration. Extracted from the EnhancedSecureAPIServer.
"""

import logging
from typing import Dict, Any
from aiohttp import web, web_request

from ..interfaces import IAuthenticationService, IMFAService, AuthCredentials, MFAMethodType

logger = logging.getLogger(__name__)


class AuthHandlers:
    """Authentication request handlers.
    
    Extracted from the EnhancedSecureAPIServer God class to handle
    authentication-related endpoints following SRP.
    """

    def __init__(
        self,
        auth_service: IAuthenticationService,
        mfa_service: IMFAService,
        rbac_system=None
    ):
        """Initialize auth handlers."""
        self.auth_service = auth_service
        self.mfa_service = mfa_service
        self.rbac_system = rbac_system

    async def handle_login(self, request: web_request.Request) -> web.Response:
        """Handle user login with MFA support."""
        try:
            data = await request.json()
            username = data.get("username")
            password = data.get("password")
            mfa_token = data.get("mfa_token")
            mfa_method = data.get("mfa_method")

            if not username or not password:
                return web.json_response(
                    {"error": "Username and password required"}, 
                    status=400
                )

            # Create credentials
            credentials = AuthCredentials(
                username=username,
                password=password,
                mfa_token=mfa_token,
                mfa_method=MFAMethodType(mfa_method) if mfa_method else None
            )

            # Authenticate user
            auth_result = await self.auth_service.authenticate_user(credentials)
            
            if not auth_result.success:
                # Check if MFA is required
                if "MFA verification required" in auth_result.error_message:
                    # Get MFA status for better error response
                    user = self.rbac_system.get_user(username) if self.rbac_system else None
                    if user:
                        mfa_status = self.mfa_service.get_user_mfa_status(user["user_id"])
                        return web.json_response(
                            {
                                "error": "MFA verification required",
                                "mfa_required": True,
                                "available_methods": mfa_status.methods_available or []
                            },
                            status=403
                        )
                
                return web.json_response(
                    {"error": auth_result.error_message}, 
                    status=401
                )

            # Create session and tokens
            device_info = request.get("device_info")
            if not device_info:
                # Fallback device info
                from ..interfaces import DeviceInfo
                device_info = DeviceInfo(
                    user_agent=request.headers.get("User-Agent", "unknown"),
                    ip_address=request.remote or "unknown"
                )

            tokens = await self.auth_service.create_session_tokens(
                user_id=auth_result.user_id,
                device_info=device_info,
                roles=auth_result.roles,
                permissions=auth_result.permissions,
                mfa_verified=auth_result.mfa_verified
            )

            logger.info(f"User {auth_result.user_id} logged in successfully")
            return web.json_response(tokens)

        except Exception as e:
            logger.error(f"Login failed: {e}")
            return web.json_response({"error": "Login failed"}, status=500)

    async def handle_logout(self, request: web_request.Request) -> web.Response:
        """Handle user logout."""
        try:
            user = request.get("user", {})
            session_id = user.get("session_id")

            if session_id:
                revoked = await self.auth_service.revoke_session(session_id)
                return web.json_response({
                    "message": "Logged out successfully", 
                    "session_revoked": revoked
                })
            else:
                return web.json_response({"message": "No active session to logout"})

        except Exception as e:
            logger.error(f"Logout failed: {e}")
            return web.json_response({"error": "Logout failed"}, status=500)

    async def handle_logout_all_sessions(self, request: web_request.Request) -> web.Response:
        """Handle logout from all sessions."""
        try:
            user = request.get("user", {})
            user_id = user.get("user_id")

            if not user_id:
                return web.json_response({"error": "User ID required"}, status=400)

            revoked_count = await self.auth_service.logout_all_user_sessions(user_id)

            return web.json_response({
                "message": f"Logged out {revoked_count} sessions",
                "sessions_revoked": revoked_count
            })

        except Exception as e:
            logger.error(f"Logout all sessions failed: {e}")
            return web.json_response({"error": "Logout all failed"}, status=500)

    async def handle_token_refresh(self, request: web_request.Request) -> web.Response:
        """Handle token refresh."""
        try:
            data = await request.json()
            refresh_token = data.get("refresh_token")

            if not refresh_token:
                return web.json_response(
                    {"error": "Refresh token required"}, 
                    status=400
                )

            # Refresh token
            tokens = await self.auth_service.refresh_token(refresh_token)
            
            return web.json_response(tokens)

        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            return web.json_response({"error": "Token refresh failed"}, status=401)

    async def handle_register(self, request: web_request.Request) -> web.Response:
        """Handle user registration."""
        try:
            data = await request.json()
            username = data.get("username")
            password = data.get("password")
            email = data.get("email")

            if not username or not password or not email:
                return web.json_response(
                    {"error": "Username, password, and email required"}, 
                    status=400
                )

            # In a real implementation, this would create the user in the database
            # For now, we'll return a success message
            logger.info(f"Registration attempted for user: {username}")
            
            return web.json_response({
                "message": "Registration successful",
                "username": username
            })

        except Exception as e:
            logger.error(f"Registration failed: {e}")
            return web.json_response({"error": "Registration failed"}, status=500)

    async def handle_change_password(self, request: web_request.Request) -> web.Response:
        """Handle password change."""
        try:
            user = request.get("user", {})
            user_id = user.get("user_id")
            
            if not user_id:
                return web.json_response({"error": "Authentication required"}, status=401)

            data = await request.json()
            current_password = data.get("current_password")
            new_password = data.get("new_password")

            if not current_password or not new_password:
                return web.json_response(
                    {"error": "Current and new password required"}, 
                    status=400
                )

            # In a real implementation, this would verify current password
            # and update the password in the database
            logger.info(f"Password change requested for user: {user_id}")
            
            return web.json_response({"message": "Password changed successfully"})

        except Exception as e:
            logger.error(f"Password change failed: {e}")
            return web.json_response({"error": "Password change failed"}, status=500)

    def register_routes(self, app: web.Application, prefix: str = "/auth"):
        """Register authentication routes."""
        app.router.add_post(f"{prefix}/login", self.handle_login)
        app.router.add_post(f"{prefix}/logout", self.handle_logout)
        app.router.add_post(f"{prefix}/logout-all", self.handle_logout_all_sessions)
        app.router.add_post(f"{prefix}/refresh", self.handle_token_refresh)
        app.router.add_post(f"{prefix}/register", self.handle_register)
        app.router.add_post(f"{prefix}/change-password", self.handle_change_password)