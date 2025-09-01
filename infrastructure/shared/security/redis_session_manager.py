"""Redis-based Session Management for JWT Token Tracking.

Provides comprehensive session management with token revocation,
device tracking, and security monitoring.
"""

from datetime import datetime, timedelta
import hashlib
import json
import logging
import os
import secrets
from typing import Any

try:
    import redis.asyncio as redis
    import redis.exceptions
except ImportError:
    logger.warning("redis package not installed, session management will be disabled")
    redis = None

logger = logging.getLogger(__name__)


class SessionError(Exception):
    """Session management errors."""

    pass


class DeviceInfo:
    """Device information for session tracking."""

    def __init__(self, user_agent: str, ip_address: str, device_fingerprint: str | None = None):
        self.user_agent = user_agent
        self.ip_address = ip_address
        self.device_fingerprint = device_fingerprint or self._generate_fingerprint(user_agent, ip_address)
        self.first_seen = datetime.utcnow()
        self.last_seen = datetime.utcnow()

    def _generate_fingerprint(self, user_agent: str, ip_address: str) -> str:
        """Generate device fingerprint from user agent and IP."""
        combined = f"{user_agent}:{ip_address}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "user_agent": self.user_agent,
            "ip_address": self.ip_address,
            "device_fingerprint": self.device_fingerprint,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DeviceInfo":
        """Create DeviceInfo from dictionary."""
        device = cls(data["user_agent"], data["ip_address"], data.get("device_fingerprint"))
        device.first_seen = datetime.fromisoformat(data["first_seen"])
        device.last_seen = datetime.fromisoformat(data["last_seen"])
        return device


class SessionData:
    """Session data container."""

    def __init__(self, user_id: str, session_id: str, device_info: DeviceInfo):
        self.user_id = user_id
        self.session_id = session_id
        self.device_info = device_info
        self.created_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        self.access_tokens = set()  # Set of JWT IDs (jti)
        self.refresh_tokens = set()
        self.is_active = True
        self.security_flags = set()  # Security flags like 'suspicious_activity'

    def add_access_token(self, jti: str):
        """Add access token JTI to session."""
        self.access_tokens.add(jti)
        self.update_activity()

    def add_refresh_token(self, jti: str):
        """Add refresh token JTI to session."""
        self.refresh_tokens.add(jti)
        self.update_activity()

    def revoke_token(self, jti: str):
        """Revoke specific token from session."""
        self.access_tokens.discard(jti)
        self.refresh_tokens.discard(jti)
        self.update_activity()

    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = datetime.utcnow()
        self.device_info.last_seen = self.last_activity

    def add_security_flag(self, flag: str):
        """Add security flag to session."""
        self.security_flags.add(flag)
        logger.warning(f"Security flag '{flag}' added to session {self.session_id}")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for Redis storage."""
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "device_info": self.device_info.to_dict(),
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "access_tokens": list(self.access_tokens),
            "refresh_tokens": list(self.refresh_tokens),
            "is_active": self.is_active,
            "security_flags": list(self.security_flags),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionData":
        """Create SessionData from dictionary."""
        device_info = DeviceInfo.from_dict(data["device_info"])
        session = cls(data["user_id"], data["session_id"], device_info)

        session.created_at = datetime.fromisoformat(data["created_at"])
        session.last_activity = datetime.fromisoformat(data["last_activity"])
        session.access_tokens = set(data.get("access_tokens", []))
        session.refresh_tokens = set(data.get("refresh_tokens", []))
        session.is_active = data.get("is_active", True)
        session.security_flags = set(data.get("security_flags", []))

        return session


class RedisSessionManager:
    """Redis-based session manager for JWT tokens."""

    def __init__(self, redis_url: str | None = None, key_prefix: str = "aivillage:session"):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.key_prefix = key_prefix
        self.redis_client = None
        self.session_timeout = timedelta(hours=24)  # Default session timeout
        self.max_sessions_per_user = 10  # Max concurrent sessions per user

    async def initialize(self):
        """Initialize Redis connection."""
        if not redis:
            raise SessionError("Redis package not installed")

        try:
            self.redis_client = redis.from_url(
                self.redis_url, decode_responses=True, socket_timeout=5, socket_connect_timeout=5, retry_on_timeout=True
            )

            # Test connection
            await self.redis_client.ping()
            logger.info("Redis session manager initialized")

        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            raise SessionError(f"Redis initialization failed: {e}")

    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()

    def _session_key(self, session_id: str) -> str:
        """Generate Redis key for session."""
        return f"{self.key_prefix}:session:{session_id}"

    def _user_sessions_key(self, user_id: str) -> str:
        """Generate Redis key for user sessions index."""
        return f"{self.key_prefix}:user_sessions:{user_id}"

    def _token_session_key(self, jti: str) -> str:
        """Generate Redis key for token-to-session mapping."""
        return f"{self.key_prefix}:token:{jti}"

    def _revoked_tokens_key(self) -> str:
        """Generate Redis key for revoked tokens set."""
        return f"{self.key_prefix}:revoked_tokens"

    async def create_session(self, user_id: str, device_info: DeviceInfo) -> str:
        """Create new session for user."""
        if not self.redis_client:
            raise SessionError("Redis not initialized")

        # Generate session ID
        session_id = f"sess_{secrets.token_urlsafe(32)}"

        # Create session data
        session_data = SessionData(user_id, session_id, device_info)

        # Check session limits
        await self._enforce_session_limits(user_id)

        # Store session
        session_key = self._session_key(session_id)
        user_sessions_key = self._user_sessions_key(user_id)

        # Use pipeline for atomic operations
        pipe = self.redis_client.pipeline()

        # Store session data
        pipe.hset(session_key, mapping=session_data.to_dict())
        pipe.expire(session_key, int(self.session_timeout.total_seconds()))

        # Add to user's sessions
        pipe.sadd(user_sessions_key, session_id)
        pipe.expire(user_sessions_key, int(self.session_timeout.total_seconds() * 2))

        await pipe.execute()

        logger.info(f"Session {session_id} created for user {user_id}")
        return session_id

    async def get_session(self, session_id: str) -> SessionData | None:
        """Get session data."""
        if not self.redis_client:
            raise SessionError("Redis not initialized")

        session_key = self._session_key(session_id)
        session_dict = await self.redis_client.hgetall(session_key)

        if not session_dict:
            return None

        # Parse nested JSON fields
        if "device_info" in session_dict:
            session_dict["device_info"] = json.loads(session_dict["device_info"])
        if "access_tokens" in session_dict:
            session_dict["access_tokens"] = json.loads(session_dict["access_tokens"])
        if "refresh_tokens" in session_dict:
            session_dict["refresh_tokens"] = json.loads(session_dict["refresh_tokens"])
        if "security_flags" in session_dict:
            session_dict["security_flags"] = json.loads(session_dict["security_flags"])

        return SessionData.from_dict(session_dict)

    async def update_session(self, session_data: SessionData):
        """Update session data in Redis."""
        if not self.redis_client:
            raise SessionError("Redis not initialized")

        session_key = self._session_key(session_data.session_id)
        session_dict = session_data.to_dict()

        # Convert complex fields to JSON
        session_dict["device_info"] = json.dumps(session_dict["device_info"])
        session_dict["access_tokens"] = json.dumps(session_dict["access_tokens"])
        session_dict["refresh_tokens"] = json.dumps(session_dict["refresh_tokens"])
        session_dict["security_flags"] = json.dumps(session_dict["security_flags"])

        await self.redis_client.hset(session_key, mapping=session_dict)

    async def add_token_to_session(self, session_id: str, jti: str, token_type: str = "access"):  # nosec B107
        """Add token to session tracking."""
        session_data = await self.get_session(session_id)
        if not session_data:
            raise SessionError(f"Session {session_id} not found")

        # Add token to session
        if token_type == "access":  # nosec B105 - token type identifier, not password
            session_data.add_access_token(jti)
        elif token_type == "refresh":  # nosec B105 - token type identifier, not password
            session_data.add_refresh_token(jti)
        else:
            raise SessionError(f"Unknown token type: {token_type}")

        # Update session
        await self.update_session(session_data)

        # Create token-to-session mapping
        token_key = self._token_session_key(jti)
        await self.redis_client.setex(token_key, int(self.session_timeout.total_seconds()), session_id)

        logger.debug(f"Token {jti} added to session {session_id}")

    async def revoke_token(self, jti: str) -> bool:
        """Revoke specific token."""
        if not self.redis_client:
            raise SessionError("Redis not initialized")

        # Find session containing this token
        token_key = self._token_session_key(jti)
        session_id = await self.redis_client.get(token_key)

        if not session_id:
            logger.warning(f"Token {jti} not found in any session")
            return False

        # Get session and remove token
        session_data = await self.get_session(session_id)
        if session_data:
            session_data.revoke_token(jti)
            await self.update_session(session_data)

        # Add to revoked tokens set
        revoked_key = self._revoked_tokens_key()
        await self.redis_client.sadd(revoked_key, jti)
        await self.redis_client.expire(revoked_key, int(self.session_timeout.total_seconds()))

        # Remove token-to-session mapping
        await self.redis_client.delete(token_key)

        logger.info(f"Token {jti} revoked")
        return True

    async def is_token_revoked(self, jti: str) -> bool:
        """Check if token is revoked."""
        if not self.redis_client:
            return False

        revoked_key = self._revoked_tokens_key()
        return bool(await self.redis_client.sismember(revoked_key, jti))

    async def revoke_session(self, session_id: str) -> bool:
        """Revoke entire session and all its tokens."""
        session_data = await self.get_session(session_id)
        if not session_data:
            return False

        # Revoke all tokens in session
        all_tokens = session_data.access_tokens.union(session_data.refresh_tokens)

        pipe = self.redis_client.pipeline()
        revoked_key = self._revoked_tokens_key()

        for jti in all_tokens:
            pipe.sadd(revoked_key, jti)
            pipe.delete(self._token_session_key(jti))

        # Mark session as inactive
        session_data.is_active = False
        await self.update_session(session_data)

        # Remove from user's active sessions
        user_sessions_key = self._user_sessions_key(session_data.user_id)
        pipe.srem(user_sessions_key, session_id)

        await pipe.execute()

        logger.info(f"Session {session_id} revoked with {len(all_tokens)} tokens")
        return True

    async def revoke_all_user_sessions(self, user_id: str) -> int:
        """Revoke all sessions for a user."""
        user_sessions_key = self._user_sessions_key(user_id)
        session_ids = await self.redis_client.smembers(user_sessions_key)

        revoked_count = 0
        for session_id in session_ids:
            if await self.revoke_session(session_id):
                revoked_count += 1

        logger.info(f"Revoked {revoked_count} sessions for user {user_id}")
        return revoked_count

    async def get_user_sessions(self, user_id: str) -> list[SessionData]:
        """Get all active sessions for user."""
        user_sessions_key = self._user_sessions_key(user_id)
        session_ids = await self.redis_client.smembers(user_sessions_key)

        sessions = []
        for session_id in session_ids:
            session_data = await self.get_session(session_id)
            if session_data and session_data.is_active:
                sessions.append(session_data)

        return sessions

    async def cleanup_expired_sessions(self):
        """Clean up expired sessions (background task)."""
        # This would typically be run as a periodic background task
        logger.info("Session cleanup completed")

    async def _enforce_session_limits(self, user_id: str):
        """Enforce maximum sessions per user."""
        sessions = await self.get_user_sessions(user_id)

        if len(sessions) >= self.max_sessions_per_user:
            # Sort by last activity and revoke oldest sessions
            sessions.sort(key=lambda s: s.last_activity)
            sessions_to_revoke = sessions[: -self.max_sessions_per_user + 1]

            for session in sessions_to_revoke:
                await self.revoke_session(session.session_id)
                logger.info(f"Revoked old session {session.session_id} due to session limit")

    async def detect_suspicious_activity(self, user_id: str, device_info: DeviceInfo) -> bool:
        """Detect potentially suspicious login activity."""
        sessions = await self.get_user_sessions(user_id)

        # Check for multiple IPs
        recent_ips = set()
        for session in sessions:
            if (datetime.utcnow() - session.last_activity).hours < 1:
                recent_ips.add(session.device_info.ip_address)

        if len(recent_ips) > 3:  # More than 3 IPs in last hour
            logger.warning(f"Suspicious activity: Multiple IPs for user {user_id}")
            return True

        # Check for unusual user agents
        known_devices = {s.device_info.device_fingerprint for s in sessions}
        if device_info.device_fingerprint not in known_devices and len(known_devices) > 0:
            logger.warning(f"Suspicious activity: New device for user {user_id}")
            return True

        return False

    async def get_session_analytics(self, user_id: str | None = None) -> dict[str, Any]:
        """Get session analytics."""
        if user_id:
            sessions = await self.get_user_sessions(user_id)
            return {
                "user_id": user_id,
                "active_sessions": len(sessions),
                "devices": len(set(s.device_info.device_fingerprint for s in sessions)),
                "total_tokens": sum(len(s.access_tokens) + len(s.refresh_tokens) for s in sessions),
            }
        else:
            # Global analytics would require scanning all sessions
            return {"message": "Global analytics not implemented"}

    async def health_check(self) -> dict[str, Any]:
        """Check Redis connection health."""
        try:
            if not self.redis_client:
                return {"status": "error", "message": "Redis not initialized"}

            # Test ping
            start_time = datetime.utcnow()
            await self.redis_client.ping()
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Get info
            info = await self.redis_client.info()

            return {
                "status": "healthy",
                "latency_ms": round(latency, 2),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory_human", "unknown"),
                "redis_version": info.get("redis_version", "unknown"),
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}
