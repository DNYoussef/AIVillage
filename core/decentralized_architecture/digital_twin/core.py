#!/usr/bin/env python3
"""
Digital Twin Core - Core twin management functionality

Extracted from UnifiedDigitalTwinSystem to handle core twin operations
following Single Responsibility Principle.
"""

import asyncio
import hashlib
import logging
import secrets
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..unified_digital_twin_system import TwinAccessLevel, TwinConversation, TwinDataType, TwinMessage, TwinUser

logger = logging.getLogger(__name__)


class DigitalTwinCore:
    """
    Core Digital Twin Management

    Handles user management, authentication, and core twin operations.
    Maintains clean separation from storage and integration concerns.
    """

    def __init__(self, twin_id: str):
        self.twin_id = twin_id

        # In-memory caches for performance
        self.users: Dict[str, TwinUser] = {}
        self.active_conversations: Dict[str, TwinConversation] = {}
        self.session_cache: Dict[str, Dict[str, Any]] = {}

        # Security tokens
        self.access_tokens: Dict[str, Dict[str, Any]] = {}

        # Core metrics
        self.metrics = {
            "total_users": 0,
            "total_conversations": 0,
            "total_messages": 0,
            "active_sessions": 0,
        }

    async def create_user(self, username: str, email: str, password: str, **user_data) -> str:
        """Create new digital twin user with validation."""

        # Validate input parameters
        if not username or not email:
            raise ValueError("Username and email are required")

        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters")

        user_id = str(uuid.uuid4())

        # Hash password securely
        password_hash = self._hash_password(password) if password else None

        # Create user object with dependency injection friendly design
        user = TwinUser(user_id=user_id, username=username, email=email, password_hash=password_hash, **user_data)

        # Cache user for performance
        self.users[user_id] = user
        self.metrics["total_users"] += 1

        logger.info(f"Created digital twin user {username} ({user_id})")
        return user_id

    async def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """Authenticate user and return access token using secure patterns."""

        user = await self._get_user_by_username(username)
        if not user or not user.is_active:
            logger.warning(f"Authentication failed for inactive user: {username}")
            return None

        # Verify password with constant-time comparison
        if not self._verify_password(password, user.password_hash):
            logger.warning(f"Invalid password for user: {username}")
            return None

        # Generate secure token
        token = secrets.token_urlsafe(32)
        self.access_tokens[token] = {
            "user_id": user.user_id,
            "username": username,
            "expires_at": time.time() + 86400,  # 24 hours
            "permissions": user.permissions,
            "created_at": time.time(),
        }

        # Update user activity
        user.last_active = time.time()

        logger.info(f"User {username} authenticated successfully")
        return token

    async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile with controlled data exposure."""

        user = await self._get_user_by_id(user_id)
        if not user:
            return None

        # Return sanitized profile data
        return {
            "user_id": user.user_id,
            "username": user.username,
            "email": user.email,
            "display_name": user.display_name,
            "avatar_url": user.avatar_url,
            "bio": user.bio,
            "language": user.language,
            "timezone": user.timezone,
            "created_at": user.created_at,
            "last_active": user.last_active,
            "access_level": user.access_level.value,
        }

    async def update_user_profile(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """Update user profile with validation."""

        user = await self._get_user_by_id(user_id)
        if not user:
            return False

        # Whitelist allowed fields to prevent injection
        allowed_fields = {"display_name", "avatar_url", "bio", "language", "timezone"}

        for field, value in updates.items():
            if field in allowed_fields:
                setattr(user, field, value)

        logger.info(f"Updated profile for user {user_id}")
        return True

    async def create_conversation(self, user_id: str, title: str = "", **conv_data) -> str:
        """Create new conversation session with proper defaults."""

        # Validate user exists
        user = await self._get_user_by_id(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")

        conversation_id = str(uuid.uuid4())

        conversation = TwinConversation(
            conversation_id=conversation_id, user_id=user_id, title=title or "New Conversation", **conv_data
        )

        # Cache for performance
        self.active_conversations[conversation_id] = conversation
        self.metrics["total_conversations"] += 1

        logger.info(f"Created conversation {conversation_id} for user {user_id}")
        return conversation_id

    async def add_message(
        self, conversation_id: str, user_id: str, content: str, message_type: str = "user", **msg_data
    ) -> str:
        """Add message to conversation with validation."""

        # Validate inputs
        if not content.strip():
            raise ValueError("Message content cannot be empty")

        if message_type not in ["user", "assistant", "system"]:
            raise ValueError("Invalid message type")

        message_id = str(uuid.uuid4())

        message = TwinMessage(
            message_id=message_id,
            conversation_id=conversation_id,
            user_id=user_id,
            content=content,
            message_type=message_type,
            **msg_data,
        )

        # Update conversation metadata
        if conversation_id in self.active_conversations:
            conv = self.active_conversations[conversation_id]
            conv.message_count += 1
            conv.last_message_at = message.timestamp

        self.metrics["total_messages"] += 1
        logger.debug(f"Added message {message_id} to conversation {conversation_id}")
        return message_id

    async def validate_access_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate access token with expiration checks."""

        if token not in self.access_tokens:
            return None

        token_data = self.access_tokens[token]

        # Check expiration
        if time.time() > token_data["expires_at"]:
            del self.access_tokens[token]
            return None

        return token_data

    def _hash_password(self, password: str) -> str:
        """Hash password with salt using secure algorithms."""

        salt = secrets.token_bytes(32)
        password_hash = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100000)
        return salt.hex() + ":" + password_hash.hex()

    def _verify_password(self, password: str, stored_hash: str) -> bool:
        """Verify password against stored hash with constant-time comparison."""

        if not stored_hash:
            return False

        try:
            salt_hex, hash_hex = stored_hash.split(":")
            salt = bytes.fromhex(salt_hex)
            stored_password_hash = bytes.fromhex(hash_hex)

            password_hash = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100000)

            # Use constant-time comparison to prevent timing attacks
            return secrets.compare_digest(password_hash, stored_password_hash)

        except (ValueError, TypeError):
            return False

    async def _get_user_by_id(self, user_id: str) -> Optional[TwinUser]:
        """Get user by ID from cache, with storage fallback handled by composition."""

        # Check cache first for performance
        if user_id in self.users:
            return self.users[user_id]

        # Note: Storage operations delegated to DigitalTwinStorage
        # This maintains separation of concerns
        return None

    async def _get_user_by_username(self, username: str) -> Optional[TwinUser]:
        """Get user by username with efficient lookup."""

        # Check cache first
        for user in self.users.values():
            if user.username == username:
                return user

        # Note: Database lookup delegated to storage layer
        return None

    def get_core_metrics(self) -> Dict[str, Any]:
        """Get core system metrics."""

        return {
            "twin_id": self.twin_id,
            "cached_users": len(self.users),
            "active_conversations": len(self.active_conversations),
            "active_sessions": len(self.access_tokens),
            "performance": self.metrics,
        }

    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired access tokens."""

        current_time = time.time()
        expired_tokens = []

        for token, data in self.access_tokens.items():
            if current_time > data["expires_at"]:
                expired_tokens.append(token)

        for token in expired_tokens:
            del self.access_tokens[token]

        if expired_tokens:
            logger.info(f"Cleaned up {len(expired_tokens)} expired sessions")

        return len(expired_tokens)

    async def get_user_activity_summary(self, user_id: str) -> Dict[str, Any]:
        """Get user activity summary for analytics."""

        user = await self._get_user_by_id(user_id)
        if not user:
            return {}

        user_conversations = [conv for conv in self.active_conversations.values() if conv.user_id == user_id]

        return {
            "user_id": user_id,
            "total_conversations": len(user_conversations),
            "last_active": user.last_active,
            "account_age_days": (time.time() - user.created_at) / 86400,
            "is_active": user.is_active,
        }
