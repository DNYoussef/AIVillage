"""Session Service.

Manages user sessions with Redis backend, providing session creation,
validation, expiration, and cleanup functionality. Extracted from the
EnhancedSecureAPIServer God class to follow Single Responsibility Principle.
"""

import asyncio
import json
import logging
import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

try:
    import redis.asyncio as redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from ..interfaces import ISessionManager, SessionData, DeviceInfo

logger = logging.getLogger(__name__)


class SessionError(Exception):
    """Session-related error."""

    pass


class SessionService(ISessionManager):
    """Session manager with Redis backend.

    This class was extracted from the EnhancedSecureAPIServer to provide
    dedicated session management functionality with proper separation of concerns.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize session service."""
        self.config = config or {}

        # Redis configuration
        self.redis_url = self.config.get("redis_url", "redis://localhost:6379/0")
        self.session_prefix = self.config.get("session_prefix", "session:")
        self.user_sessions_prefix = self.config.get("user_sessions_prefix", "user_sessions:")
        self.token_prefix = self.config.get("token_prefix", "token:")

        # Session configuration
        self.session_ttl = self.config.get("session_ttl_hours", 24) * 3600
        self.max_sessions_per_user = self.config.get("max_sessions_per_user", 10)
        self.cleanup_interval = self.config.get("cleanup_interval_minutes", 30) * 60

        # Connection
        self.redis_client: Optional[redis.Redis] = None

        # Fallback storage for when Redis is not available
        self.memory_sessions: Dict[str, SessionData] = {}
        self.user_sessions_map: Dict[str, List[str]] = {}
        self.token_revocation_set: set[str] = set()

        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

    async def initialize(self) -> None:
        """Initialize session manager."""
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
                await self.redis_client.ping()
                logger.info("Connected to Redis for session management")
            except Exception as e:
                logger.warning(f"Redis connection failed, using memory storage: {e}")
                self.redis_client = None
        else:
            logger.warning("Redis not available, using memory storage for sessions")

        # Start cleanup task
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def close(self) -> None:
        """Close session manager."""
        self._running = False

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        if self.redis_client:
            await self.redis_client.close()

    async def create_session(
        self,
        user_id: str,
        device_info: DeviceInfo,
        roles: List[str] = None,
        permissions: List[str] = None,
        tenant_id: Optional[str] = None,
        metadata: Dict[str, Any] = None,
    ) -> str:
        """Create new user session."""
        try:
            # Generate session ID
            session_id = f"sess_{secrets.token_urlsafe(32)}"

            # Create session data
            now = datetime.utcnow()
            session_data = SessionData(
                session_id=session_id,
                user_id=user_id,
                created_at=now,
                last_activity=now,
                device_info=device_info,
                roles=roles or [],
                permissions=permissions or [],
                tenant_id=tenant_id,
                metadata=metadata or {},
            )

            # Check session limits
            user_sessions = await self._get_user_sessions_internal(user_id)
            if len(user_sessions) >= self.max_sessions_per_user:
                # Remove oldest session
                oldest_session = min(user_sessions, key=lambda s: s.created_at)
                await self.revoke_session(oldest_session.session_id)

            # Store session
            if self.redis_client:
                await self._store_session_redis(session_data)
            else:
                await self._store_session_memory(session_data)

            logger.info(f"Created session {session_id} for user {user_id}")
            return session_id

        except Exception as e:
            logger.error(f"Session creation error: {e}")
            raise SessionError(f"Failed to create session: {e}")

    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """Get session by ID."""
        try:
            if self.redis_client:
                session_data = await self._get_session_redis(session_id)
            else:
                session_data = await self._get_session_memory(session_id)

            if not session_data:
                return None

            # Check if session is expired
            if not session_data.is_active:
                await self.revoke_session(session_id)
                return None

            # Check TTL
            now = datetime.utcnow()
            if now - session_data.last_activity > timedelta(seconds=self.session_ttl):
                await self.revoke_session(session_id)
                return None

            return session_data

        except Exception as e:
            logger.error(f"Session retrieval error: {e}")
            return None

    async def is_session_active(self, session_id: str) -> bool:
        """Check if session is active."""
        session_data = await self.get_session(session_id)
        return session_data is not None and session_data.is_active

    async def update_session_activity(self, session_id: str) -> bool:
        """Update session last activity."""
        try:
            session_data = await self._get_session_data_internal(session_id)
            if not session_data:
                return False

            session_data.last_activity = datetime.utcnow()

            if self.redis_client:
                await self._store_session_redis(session_data)
            else:
                await self._store_session_memory(session_data)

            return True

        except Exception as e:
            logger.error(f"Session activity update error: {e}")
            return False

    async def revoke_session(self, session_id: str) -> bool:
        """Revoke session."""
        try:
            session_data = await self._get_session_data_internal(session_id)
            if not session_data:
                return False

            user_id = session_data.user_id

            if self.redis_client:
                await self._delete_session_redis(session_id, user_id)
            else:
                await self._delete_session_memory(session_id, user_id)

            logger.info(f"Revoked session {session_id} for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Session revocation error: {e}")
            return False

    async def revoke_all_user_sessions(self, user_id: str) -> int:
        """Revoke all sessions for user."""
        try:
            user_sessions = await self._get_user_sessions_internal(user_id)
            revoked_count = 0

            for session_data in user_sessions:
                if await self.revoke_session(session_data.session_id):
                    revoked_count += 1

            logger.info(f"Revoked {revoked_count} sessions for user {user_id}")
            return revoked_count

        except Exception as e:
            logger.error(f"User session revocation error: {e}")
            return 0

    async def get_user_sessions(self, user_id: str) -> List[SessionData]:
        """Get all active sessions for user."""
        try:
            user_sessions = await self._get_user_sessions_internal(user_id)
            return [session for session in user_sessions if session.is_active]

        except Exception as e:
            logger.error(f"User sessions retrieval error: {e}")
            return []

    async def add_token_to_session(self, session_id: str, jti: str, token_type: str) -> bool:
        """Add token to session tracking."""
        try:
            if self.redis_client:
                # Store token metadata in Redis
                token_key = f"{self.token_prefix}{jti}"
                token_data = {
                    "session_id": session_id,
                    "token_type": token_type,
                    "created_at": datetime.utcnow().isoformat(),
                    "revoked": False,
                }
                await self.redis_client.hset(token_key, mapping=token_data)
                await self.redis_client.expire(token_key, self.session_ttl)
            else:
                # In memory, we'll track this in session metadata
                session_data = await self._get_session_data_internal(session_id)
                if session_data:
                    if "tokens" not in session_data.metadata:
                        session_data.metadata["tokens"] = {}
                    session_data.metadata["tokens"][jti] = {
                        "token_type": token_type,
                        "created_at": datetime.utcnow().isoformat(),
                        "revoked": False,
                    }
                    await self._store_session_memory(session_data)

            return True

        except Exception as e:
            logger.error(f"Add token to session error: {e}")
            return False

    async def is_token_revoked(self, jti: str) -> bool:
        """Check if token is revoked."""
        try:
            if self.redis_client:
                token_key = f"{self.token_prefix}{jti}"
                token_data = await self.redis_client.hgetall(token_key)
                return token_data.get("revoked", "false") == "true"
            else:
                return jti in self.token_revocation_set

        except Exception as e:
            logger.error(f"Token revocation check error: {e}")
            return True  # Fail secure

    async def revoke_token(self, jti: str) -> bool:
        """Revoke specific token."""
        try:
            if self.redis_client:
                token_key = f"{self.token_prefix}{jti}"
                await self.redis_client.hset(token_key, "revoked", "true")
            else:
                self.token_revocation_set.add(jti)

            return True

        except Exception as e:
            logger.error(f"Token revocation error: {e}")
            return False

    async def detect_suspicious_activity(self, user_id: str, device_info: DeviceInfo) -> bool:
        """Detect suspicious activity."""
        try:
            user_sessions = await self._get_user_sessions_internal(user_id)

            # Check for multiple different IPs
            ip_addresses = set()
            user_agents = set()

            for session in user_sessions:
                ip_addresses.add(session.device_info.ip_address)
                user_agents.add(session.device_info.user_agent)

            # Add current request
            ip_addresses.add(device_info.ip_address)
            user_agents.add(device_info.user_agent)

            # Simple heuristics for suspicious activity
            if len(ip_addresses) > 5:  # Too many different IPs
                return True

            if len(user_agents) > 3:  # Too many different user agents
                return True

            return False

        except Exception as e:
            logger.error(f"Suspicious activity detection error: {e}")
            return False

    async def health_check(self) -> Dict[str, Any]:
        """Health check for session manager."""
        try:
            health = {
                "status": "healthy",
                "backend": "redis" if self.redis_client else "memory",
                "active_sessions": 0,
                "redis_connected": False,
            }

            if self.redis_client:
                try:
                    await self.redis_client.ping()
                    health["redis_connected"] = True

                    # Count active sessions
                    session_keys = []
                    async for key in self.redis_client.scan_iter(match=f"{self.session_prefix}*"):
                        session_keys.append(key)
                    health["active_sessions"] = len(session_keys)

                except Exception as e:
                    health["status"] = "degraded"
                    health["error"] = str(e)
            else:
                health["active_sessions"] = len(self.memory_sessions)

            return health

        except Exception as e:
            logger.error(f"Health check error: {e}")
            return {"status": "error", "error": str(e)}

    # Redis implementation methods

    async def _store_session_redis(self, session_data: SessionData):
        """Store session in Redis."""
        session_key = f"{self.session_prefix}{session_data.session_id}"
        user_sessions_key = f"{self.user_sessions_prefix}{session_data.user_id}"

        # Convert session data to dict
        session_dict = {
            "session_id": session_data.session_id,
            "user_id": session_data.user_id,
            "created_at": session_data.created_at.isoformat(),
            "last_activity": session_data.last_activity.isoformat(),
            "device_info": json.dumps(session_data.device_info.__dict__),
            "roles": json.dumps(session_data.roles),
            "permissions": json.dumps(session_data.permissions),
            "tenant_id": session_data.tenant_id or "",
            "is_active": str(session_data.is_active),
            "metadata": json.dumps(session_data.metadata or {}),
        }

        pipe = self.redis_client.pipeline()

        # Store session data
        pipe.hset(session_key, mapping=session_dict)
        pipe.expire(session_key, self.session_ttl)

        # Add to user sessions set
        pipe.sadd(user_sessions_key, session_data.session_id)
        pipe.expire(user_sessions_key, self.session_ttl)

        await pipe.execute()

    async def _get_session_redis(self, session_id: str) -> Optional[SessionData]:
        """Get session from Redis."""
        session_key = f"{self.session_prefix}{session_id}"
        session_dict = await self.redis_client.hgetall(session_key)

        if not session_dict:
            return None

        # Convert back to SessionData
        device_info_dict = json.loads(session_dict["device_info"])
        device_info = DeviceInfo(**device_info_dict)

        return SessionData(
            session_id=session_dict["session_id"],
            user_id=session_dict["user_id"],
            created_at=datetime.fromisoformat(session_dict["created_at"]),
            last_activity=datetime.fromisoformat(session_dict["last_activity"]),
            device_info=device_info,
            roles=json.loads(session_dict["roles"]),
            permissions=json.loads(session_dict["permissions"]),
            tenant_id=session_dict["tenant_id"] if session_dict["tenant_id"] else None,
            is_active=session_dict["is_active"] == "True",
            metadata=json.loads(session_dict["metadata"]),
        )

    async def _delete_session_redis(self, session_id: str, user_id: str):
        """Delete session from Redis."""
        session_key = f"{self.session_prefix}{session_id}"
        user_sessions_key = f"{self.user_sessions_prefix}{user_id}"

        pipe = self.redis_client.pipeline()
        pipe.delete(session_key)
        pipe.srem(user_sessions_key, session_id)
        await pipe.execute()

    # Memory implementation methods (fallback)

    async def _store_session_memory(self, session_data: SessionData):
        """Store session in memory."""
        self.memory_sessions[session_data.session_id] = session_data

        user_id = session_data.user_id
        if user_id not in self.user_sessions_map:
            self.user_sessions_map[user_id] = []

        if session_data.session_id not in self.user_sessions_map[user_id]:
            self.user_sessions_map[user_id].append(session_data.session_id)

    async def _get_session_memory(self, session_id: str) -> Optional[SessionData]:
        """Get session from memory."""
        return self.memory_sessions.get(session_id)

    async def _delete_session_memory(self, session_id: str, user_id: str):
        """Delete session from memory."""
        if session_id in self.memory_sessions:
            del self.memory_sessions[session_id]

        if user_id in self.user_sessions_map and session_id in self.user_sessions_map[user_id]:
            self.user_sessions_map[user_id].remove(session_id)

            if not self.user_sessions_map[user_id]:
                del self.user_sessions_map[user_id]

    # Helper methods

    async def _get_session_data_internal(self, session_id: str) -> Optional[SessionData]:
        """Get session data object (internal method)."""
        if self.redis_client:
            return await self._get_session_redis(session_id)
        else:
            return await self._get_session_memory(session_id)

    async def _get_user_sessions_internal(self, user_id: str) -> List[SessionData]:
        """Get all sessions for user (internal method)."""
        if self.redis_client:
            return await self._get_user_sessions_redis(user_id)
        else:
            return await self._get_user_sessions_memory(user_id)

    async def _get_user_sessions_redis(self, user_id: str) -> List[SessionData]:
        """Get user sessions from Redis."""
        user_sessions_key = f"{self.user_sessions_prefix}{user_id}"
        session_ids = await self.redis_client.smembers(user_sessions_key)

        sessions = []
        for session_id in session_ids:
            session_data = await self._get_session_redis(session_id)
            if session_data:
                sessions.append(session_data)

        return sessions

    async def _get_user_sessions_memory(self, user_id: str) -> List[SessionData]:
        """Get user sessions from memory."""
        session_ids = self.user_sessions_map.get(user_id, [])

        sessions = []
        for session_id in session_ids:
            session_data = self.memory_sessions.get(session_id)
            if session_data:
                sessions.append(session_data)

        return sessions

    # Background tasks

    async def _cleanup_loop(self):
        """Background cleanup loop."""
        while self._running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                if self._running:
                    await self._cleanup_expired_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Session cleanup loop error: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    async def _cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions."""
        try:
            cleaned_count = 0
            now = datetime.utcnow()

            if self.redis_client:
                cleaned_count = await self._cleanup_sessions_redis(now)
            else:
                cleaned_count = await self._cleanup_sessions_memory(now)

            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} expired sessions")

            return cleaned_count

        except Exception as e:
            logger.error(f"Session cleanup error: {e}")
            return 0

    async def _cleanup_sessions_redis(self, current_time: datetime) -> int:
        """Clean up expired sessions in Redis."""
        # Redis TTL handles automatic cleanup, but we can clean user session sets
        pattern = f"{self.user_sessions_prefix}*"
        cleaned_count = 0

        async for key in self.redis_client.scan_iter(match=pattern):
            key.replace(self.user_sessions_prefix, "")
            session_ids = await self.redis_client.smembers(key)

            for session_id in session_ids:
                session_key = f"{self.session_prefix}{session_id}"
                if not await self.redis_client.exists(session_key):
                    await self.redis_client.srem(key, session_id)
                    cleaned_count += 1

        return cleaned_count

    async def _cleanup_sessions_memory(self, current_time: datetime) -> int:
        """Clean up expired sessions in memory."""
        expired_sessions = []

        for session_id, session_data in self.memory_sessions.items():
            if not session_data.is_active or current_time - session_data.last_activity > timedelta(
                seconds=self.session_ttl
            ):
                expired_sessions.append((session_id, session_data.user_id))

        for session_id, user_id in expired_sessions:
            await self._delete_session_memory(session_id, user_id)

        return len(expired_sessions)
