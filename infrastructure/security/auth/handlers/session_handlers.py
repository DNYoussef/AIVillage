"""Session Management Request Handlers.

Handles session-related HTTP endpoints including session listing,
revocation, and management. Extracted from the EnhancedSecureAPIServer.
"""

import logging
from aiohttp import web, web_request

from ..interfaces import ISessionManager

logger = logging.getLogger(__name__)


class SessionHandlers:
    """Session management request handlers.
    
    Handles session-related endpoints for user session management.
    """

    def __init__(self, session_manager: ISessionManager):
        """Initialize session handlers."""
        self.session_manager = session_manager

    async def handle_get_user_sessions(self, request: web_request.Request) -> web.Response:
        """Get user's active sessions."""
        try:
            user = request.get("user", {})
            user_id = user.get("user_id")

            if not user_id:
                return web.json_response({"error": "Authentication required"}, status=401)

            sessions = await self.session_manager.get_user_sessions(user_id)

            session_list = []
            for session in sessions:
                session_list.append({
                    "session_id": session.session_id,
                    "created_at": session.created_at.isoformat(),
                    "last_activity": session.last_activity.isoformat(),
                    "device_info": {
                        "device_fingerprint": session.device_info.device_fingerprint,
                        "ip_address": session.device_info.ip_address,
                        "user_agent": session.device_info.user_agent[:100],  # Truncate for security
                    },
                    "is_current": session.session_id == user.get("session_id"),
                    "is_active": session.is_active,
                    "tenant_id": session.tenant_id
                })

            return web.json_response({
                "sessions": session_list,
                "total_sessions": len(session_list)
            })

        except Exception as e:
            logger.error(f"Get user sessions failed: {e}")
            return web.json_response({"error": "Failed to get sessions"}, status=500)

    async def handle_revoke_session(self, request: web_request.Request) -> web.Response:
        """Revoke specific session."""
        try:
            user = request.get("user", {})
            user_id = user.get("user_id")
            session_id = request.match_info.get("session_id")

            if not user_id:
                return web.json_response({"error": "Authentication required"}, status=401)

            if not session_id:
                return web.json_response({"error": "Session ID required"}, status=400)

            # Verify the session belongs to the user
            user_sessions = await self.session_manager.get_user_sessions(user_id)
            session_ids = [s.session_id for s in user_sessions]

            if session_id not in session_ids:
                return web.json_response(
                    {"error": "Session not found or not owned by user"}, 
                    status=404
                )

            # Revoke the session
            revoked = await self.session_manager.revoke_session(session_id)

            if revoked:
                logger.info(f"Session {session_id} revoked by user {user_id}")
                return web.json_response({
                    "message": "Session revoked successfully",
                    "session_id": session_id
                })
            else:
                return web.json_response(
                    {"error": "Failed to revoke session"}, 
                    status=500
                )

        except Exception as e:
            logger.error(f"Revoke session failed: {e}")
            return web.json_response({"error": "Failed to revoke session"}, status=500)

    async def handle_revoke_all_sessions(self, request: web_request.Request) -> web.Response:
        """Revoke all sessions for user except current."""
        try:
            user = request.get("user", {})
            user_id = user.get("user_id")
            current_session_id = user.get("session_id")

            if not user_id:
                return web.json_response({"error": "Authentication required"}, status=401)

            # Get all user sessions
            user_sessions = await self.session_manager.get_user_sessions(user_id)
            
            revoked_count = 0
            for session in user_sessions:
                # Skip current session unless explicitly requested
                if session.session_id != current_session_id:
                    if await self.session_manager.revoke_session(session.session_id):
                        revoked_count += 1

            logger.info(f"Revoked {revoked_count} sessions for user {user_id}")
            return web.json_response({
                "message": f"Revoked {revoked_count} sessions",
                "sessions_revoked": revoked_count
            })

        except Exception as e:
            logger.error(f"Revoke all sessions failed: {e}")
            return web.json_response({"error": "Failed to revoke sessions"}, status=500)

    async def handle_session_info(self, request: web_request.Request) -> web.Response:
        """Get current session information."""
        try:
            user = request.get("user", {})
            session_id = user.get("session_id")

            if not session_id:
                return web.json_response({"error": "No active session"}, status=400)

            session_data = await self.session_manager.get_session(session_id)

            if not session_data:
                return web.json_response({"error": "Session not found"}, status=404)

            return web.json_response({
                "session_id": session_data.session_id,
                "user_id": session_data.user_id,
                "created_at": session_data.created_at.isoformat(),
                "last_activity": session_data.last_activity.isoformat(),
                "device_info": {
                    "ip_address": session_data.device_info.ip_address,
                    "user_agent": session_data.device_info.user_agent,
                },
                "roles": session_data.roles,
                "permissions": session_data.permissions,
                "tenant_id": session_data.tenant_id,
                "is_active": session_data.is_active
            })

        except Exception as e:
            logger.error(f"Get session info failed: {e}")
            return web.json_response({"error": "Failed to get session info"}, status=500)

    async def handle_extend_session(self, request: web_request.Request) -> web.Response:
        """Extend current session."""
        try:
            user = request.get("user", {})
            session_id = user.get("session_id")

            if not session_id:
                return web.json_response({"error": "No active session"}, status=400)

            # Update session activity (extends TTL)
            updated = await self.session_manager.update_session_activity(session_id)

            if updated:
                return web.json_response({"message": "Session extended successfully"})
            else:
                return web.json_response(
                    {"error": "Failed to extend session"}, 
                    status=500
                )

        except Exception as e:
            logger.error(f"Extend session failed: {e}")
            return web.json_response({"error": "Failed to extend session"}, status=500)

    async def handle_session_health(self, request: web_request.Request) -> web.Response:
        """Get session manager health status."""
        try:
            health = await self.session_manager.health_check()
            return web.json_response(health)

        except Exception as e:
            logger.error(f"Session health check failed: {e}")
            return web.json_response({"error": "Health check failed"}, status=500)

    def register_routes(self, app: web.Application, prefix: str = "/auth/sessions"):
        """Register session management routes."""
        app.router.add_get(f"{prefix}", self.handle_get_user_sessions)
        app.router.add_delete(f"{prefix}/{{session_id}}", self.handle_revoke_session)
        app.router.add_delete(f"{prefix}/revoke-all", self.handle_revoke_all_sessions)
        app.router.add_get(f"{prefix}/current", self.handle_session_info)
        app.router.add_post(f"{prefix}/extend", self.handle_extend_session)
        app.router.add_get(f"{prefix}/health", self.handle_session_health)