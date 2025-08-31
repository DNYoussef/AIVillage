"""Session Manager.

Manages user sessions with Redis backend, providing session creation,
validation, expiration, and cleanup functionality.
"""

import asyncio
import json
import logging
import secrets
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import asdict

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from ..core.exceptions import SessionError

logger = logging.getLogger(__name__)


class SessionData:
    """Session data structure."""
    
    def __init__(
        self,
        session_id: str,
        user_id: str,
        created_at: datetime,
        last_activity: datetime,
        device_info: Dict[str, Any],
        roles: List[str] = None,
        permissions: List[str] = None,
        tenant_id: Optional[str] = None,
        is_active: bool = True,
        metadata: Dict[str, Any] = None
    ):
        self.session_id = session_id
        self.user_id = user_id
        self.created_at = created_at
        self.last_activity = last_activity
        self.device_info = device_info
        self.roles = roles or []
        self.permissions = permissions or []
        self.tenant_id = tenant_id
        self.is_active = is_active
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "device_info": self.device_info,
            "roles": self.roles,
            "permissions": self.permissions,
            "tenant_id": self.tenant_id,
            "is_active": self.is_active,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionData':
        """Create from dictionary."""
        return cls(
            session_id=data["session_id"],
            user_id=data["user_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_activity=datetime.fromisoformat(data["last_activity"]),
            device_info=data.get("device_info", {}),
            roles=data.get("roles", []),
            permissions=data.get("permissions", []),
            tenant_id=data.get("tenant_id"),
            is_active=data.get("is_active", True),
            metadata=data.get("metadata", {}),
        )


class SessionManager:
    """Session manager with Redis backend."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Redis configuration
        self.redis_url = config.get("redis_url", "redis://localhost:6379/0")
        self.session_prefix = config.get("session_prefix", "session:")
        self.user_sessions_prefix = config.get("user_sessions_prefix", "user_sessions:")
        
        # Session configuration
        self.session_ttl = config.get("session_ttl_hours", 24) * 3600
        self.max_sessions_per_user = config.get("max_sessions_per_user", 10)
        self.cleanup_interval = config.get("cleanup_interval_minutes", 30) * 60
        
        # Connection
        self.redis_client: Optional[redis.Redis] = None
        
        # Fallback storage for when Redis is not available
        self.memory_sessions: Dict[str, SessionData] = {}
        self.user_sessions: Dict[str, List[str]] = {}
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def initialize(self):
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
    
    async def close(self):
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
        device_info: Dict[str, Any],
        roles: List[str] = None,
        permissions: List[str] = None,
        tenant_id: Optional[str] = None,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
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
                metadata=metadata or {}
            )
            
            # Check session limits
            user_sessions = await self._get_user_sessions(user_id)
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
            
            return {
                "session_id": session_id,
                "user_id": user_id,
                "created_at": now.isoformat(),
                "expires_at": (now + timedelta(seconds=self.session_ttl)).isoformat(),
            }
            
        except Exception as e:
            logger.error(f"Session creation error: {e}")
            raise SessionError(f"Failed to create session: {e}")
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
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
            
            return session_data.to_dict()
            
        except Exception as e:
            logger.error(f"Session retrieval error: {e}")
            return None
    
    async def update_activity(self, session_id: str) -> bool:
        """Update session last activity."""
        try:
            session_data = await self._get_session_data(session_id)
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
            session_data = await self._get_session_data(session_id)
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
            user_sessions = await self._get_user_sessions(user_id)
            revoked_count = 0
            
            for session_data in user_sessions:
                if await self.revoke_session(session_data.session_id):
                    revoked_count += 1
            
            logger.info(f"Revoked {revoked_count} sessions for user {user_id}")
            return revoked_count
            
        except Exception as e:
            logger.error(f"User session revocation error: {e}")
            return 0
    
    async def get_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all active sessions for user."""
        try:
            user_sessions = await self._get_user_sessions(user_id)
            return [session.to_dict() for session in user_sessions if session.is_active]
            
        except Exception as e:
            logger.error(f"User sessions retrieval error: {e}")
            return []
    
    async def cleanup_expired_sessions(self) -> int:
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
    
    async def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        try:
            if self.redis_client:
                stats = await self._get_session_stats_redis()
            else:
                stats = await self._get_session_stats_memory()
            
            return stats
            
        except Exception as e:
            logger.error(f"Session stats error: {e}")
            return {"active_sessions": 0, "total_users": 0}
    
    # Redis implementation
    
    async def _store_session_redis(self, session_data: SessionData):
        """Store session in Redis."""
        session_key = f"{self.session_prefix}{session_data.session_id}"
        user_sessions_key = f"{self.user_sessions_prefix}{session_data.user_id}"
        
        pipe = self.redis_client.pipeline()
        
        # Store session data
        pipe.hset(session_key, mapping=session_data.to_dict())
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
        
        return SessionData.from_dict(session_dict)
    
    async def _delete_session_redis(self, session_id: str, user_id: str):
        """Delete session from Redis."""
        session_key = f"{self.session_prefix}{session_id}"
        user_sessions_key = f"{self.user_sessions_prefix}{user_id}"
        
        pipe = self.redis_client.pipeline()
        pipe.delete(session_key)
        pipe.srem(user_sessions_key, session_id)
        await pipe.execute()
    
    async def _cleanup_sessions_redis(self, current_time: datetime) -> int:
        """Clean up expired sessions in Redis."""
        # Redis TTL handles automatic cleanup, but we can clean user session sets
        pattern = f"{self.user_sessions_prefix}*"
        cleaned_count = 0
        
        async for key in self.redis_client.scan_iter(match=pattern):
            user_id = key.replace(self.user_sessions_prefix, "")
            session_ids = await self.redis_client.smembers(key)
            
            for session_id in session_ids:
                session_key = f"{self.session_prefix}{session_id}"
                if not await self.redis_client.exists(session_key):
                    await self.redis_client.srem(key, session_id)
                    cleaned_count += 1
        
        return cleaned_count
    
    async def _get_session_stats_redis(self) -> Dict[str, Any]:
        """Get session statistics from Redis."""
        session_pattern = f"{self.session_prefix}*"
        user_pattern = f"{self.user_sessions_prefix}*"
        
        active_sessions = 0
        async for _ in self.redis_client.scan_iter(match=session_pattern):
            active_sessions += 1
        
        total_users = 0
        async for _ in self.redis_client.scan_iter(match=user_pattern):
            total_users += 1
        
        return {
            "active_sessions": active_sessions,
            "total_users": total_users,
            "storage_backend": "redis",
        }
    
    # Memory implementation (fallback)
    
    async def _store_session_memory(self, session_data: SessionData):
        """Store session in memory."""
        self.memory_sessions[session_data.session_id] = session_data
        
        user_id = session_data.user_id
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = []
        
        if session_data.session_id not in self.user_sessions[user_id]:
            self.user_sessions[user_id].append(session_data.session_id)
    
    async def _get_session_memory(self, session_id: str) -> Optional[SessionData]:
        """Get session from memory."""
        return self.memory_sessions.get(session_id)
    
    async def _delete_session_memory(self, session_id: str, user_id: str):
        """Delete session from memory."""
        if session_id in self.memory_sessions:
            del self.memory_sessions[session_id]
        
        if user_id in self.user_sessions and session_id in self.user_sessions[user_id]:
            self.user_sessions[user_id].remove(session_id)
            
            if not self.user_sessions[user_id]:
                del self.user_sessions[user_id]
    
    async def _cleanup_sessions_memory(self, current_time: datetime) -> int:
        """Clean up expired sessions in memory."""
        expired_sessions = []
        
        for session_id, session_data in self.memory_sessions.items():
            if (not session_data.is_active or 
                current_time - session_data.last_activity > timedelta(seconds=self.session_ttl)):
                expired_sessions.append((session_id, session_data.user_id))
        
        for session_id, user_id in expired_sessions:
            await self._delete_session_memory(session_id, user_id)
        
        return len(expired_sessions)
    
    async def _get_session_stats_memory(self) -> Dict[str, Any]:
        """Get session statistics from memory."""
        return {
            "active_sessions": len(self.memory_sessions),
            "total_users": len(self.user_sessions),
            "storage_backend": "memory",
        }
    
    # Helper methods
    
    async def _get_session_data(self, session_id: str) -> Optional[SessionData]:
        """Get session data object."""
        if self.redis_client:
            return await self._get_session_redis(session_id)
        else:
            return await self._get_session_memory(session_id)
    
    async def _get_user_sessions(self, user_id: str) -> List[SessionData]:
        """Get all sessions for user."""
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
        session_ids = self.user_sessions.get(user_id, [])
        
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
                    await self.cleanup_expired_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Session cleanup loop error: {e}")
                await asyncio.sleep(60)  # Wait before retrying