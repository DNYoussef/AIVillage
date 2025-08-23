#!/usr/bin/env python3
"""
Digital Twin Storage - Database and persistence operations

Extracted from UnifiedDigitalTwinSystem to handle all storage concerns
following Single Responsibility Principle.
"""

import asyncio
import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from ..unified_digital_twin_system import TwinAccessLevel, TwinConversation, TwinMessage, TwinUser

logger = logging.getLogger(__name__)


class DigitalTwinStorage:
    """
    Digital Twin Storage Management

    Handles all database operations, file I/O, and data persistence.
    Provides clean interface for storage operations with proper error handling.
    """

    def __init__(self, data_dir: Path, enable_encryption: bool = True):
        self.data_dir = Path(data_dir)
        self.enable_encryption = enable_encryption

        # Database connection management
        self.database_manager = None
        self.db_path = self.data_dir / "twin.db"

        # Encryption key management
        self.encryption_key: Optional[bytes] = None

        # Storage metrics
        self.storage_metrics = {
            "database_operations": 0,
            "encryption_operations": 0,
            "file_operations": 0,
        }

    async def initialize(self):
        """Initialize storage system with proper setup."""

        logger.info("Initializing Digital Twin Storage...")

        # Create data directory
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize database
        await self._initialize_database()

        # Initialize encryption if enabled
        if self.enable_encryption:
            self.encryption_key = self._get_or_create_encryption_key()

        logger.info("✅ Digital Twin Storage initialized successfully")

    async def _initialize_database(self):
        """Initialize database with proper connection management."""

        try:
            # Try to import external database manager
            from infrastructure.twin.database.database_manager import DatabaseManager

            self.database_manager = DatabaseManager(
                database_path=self.db_path, enable_encryption=self.enable_encryption
            )
            await self.database_manager.initialize()

            logger.info("External database manager initialized")

        except ImportError:
            logger.info("Using built-in SQLite database manager")
            await self._initialize_builtin_database()

    async def _initialize_builtin_database(self):
        """Initialize built-in SQLite database with proper schema."""

        self.database_manager = sqlite3.connect(str(self.db_path))
        self.database_manager.row_factory = sqlite3.Row

        # Create tables with proper constraints
        await self._create_database_tables()

        logger.info(f"Built-in database initialized at {self.db_path}")

    async def _create_database_tables(self):
        """Create database tables with proper schema and constraints."""

        cursor = self.database_manager.cursor()

        # Users table with constraints
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT,
                access_level TEXT DEFAULT 'private',
                permissions TEXT DEFAULT '[]',
                display_name TEXT DEFAULT '',
                avatar_url TEXT DEFAULT '',
                bio TEXT DEFAULT '',
                language TEXT DEFAULT 'en',
                timezone TEXT DEFAULT 'UTC',
                privacy_settings TEXT DEFAULT '{}',
                created_at REAL NOT NULL,
                last_active REAL NOT NULL,
                is_active BOOLEAN DEFAULT 1,
                CONSTRAINT chk_access_level CHECK (access_level IN ('public', 'private', 'restricted', 'confidential'))
            )
        """
        )

        # Conversations table with proper relationships
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT DEFAULT '',
                created_at REAL NOT NULL,
                last_message_at REAL NOT NULL,
                message_count INTEGER DEFAULT 0 CHECK (message_count >= 0),
                context_length INTEGER DEFAULT 4000 CHECK (context_length > 0),
                temperature REAL DEFAULT 0.7 CHECK (temperature >= 0.0 AND temperature <= 2.0),
                system_prompt TEXT DEFAULT '',
                is_private BOOLEAN DEFAULT 1,
                retention_days INTEGER DEFAULT 30 CHECK (retention_days >= 0),
                FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE
            )
        """
        )

        # Messages table with proper constraints
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL CHECK (length(content) > 0),
                message_type TEXT DEFAULT 'user' CHECK (message_type IN ('user', 'assistant', 'system')),
                timestamp REAL NOT NULL,
                tokens_used INTEGER DEFAULT 0 CHECK (tokens_used >= 0),
                processing_time_ms REAL DEFAULT 0 CHECK (processing_time_ms >= 0),
                context_data TEXT DEFAULT '{}',
                references TEXT DEFAULT '[]',
                FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id) ON DELETE CASCADE,
                FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE
            )
        """
        )

        # Twin data table with flexible schema
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS twin_data (
                data_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                data_type TEXT NOT NULL,
                data_key TEXT NOT NULL,
                data_value TEXT NOT NULL,
                access_level TEXT DEFAULT 'private',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                expires_at REAL,
                FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE,
                UNIQUE(user_id, data_type, data_key),
                CONSTRAINT chk_data_access_level CHECK (access_level IN ('public', 'private', 'restricted', 'confidential'))
            )
        """
        )

        # Create indexes for performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_conversations_last_message ON conversations(last_message_at)",
            "CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id)",
            "CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_twin_data_user_type ON twin_data(user_id, data_type)",
            "CREATE INDEX IF NOT EXISTS idx_twin_data_expires ON twin_data(expires_at)",
        ]

        for index_sql in indexes:
            cursor.execute(index_sql)

        self.database_manager.commit()
        logger.info("Database tables and indexes created successfully")

    @contextmanager
    def _get_db_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get database connection with proper resource management."""

        if hasattr(self.database_manager, "cursor"):
            # Built-in SQLite connection
            yield self.database_manager
        else:
            # External database manager
            with self.database_manager.get_connection() as conn:
                yield conn

    async def save_user(self, user: TwinUser) -> bool:
        """Save user to database with proper error handling."""

        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT OR REPLACE INTO users
                    (user_id, username, email, password_hash, access_level, permissions,
                     display_name, avatar_url, bio, language, timezone, privacy_settings,
                     created_at, last_active, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        user.user_id,
                        user.username,
                        user.email,
                        user.password_hash,
                        user.access_level.value,
                        json.dumps(user.permissions),
                        user.display_name,
                        user.avatar_url,
                        user.bio,
                        user.language,
                        user.timezone,
                        json.dumps(user.privacy_settings),
                        user.created_at,
                        user.last_active,
                        user.is_active,
                    ),
                )

                conn.commit()
                self.storage_metrics["database_operations"] += 1
                return True

        except Exception as e:
            logger.error(f"Failed to save user {user.user_id}: {e}")
            return False

    async def load_user(self, user_id: str) -> Optional[TwinUser]:
        """Load user from database with proper deserialization."""

        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
                row = cursor.fetchone()

                if not row:
                    return None

                # Convert row to TwinUser object
                user = TwinUser(
                    user_id=row["user_id"],
                    username=row["username"],
                    email=row["email"],
                    password_hash=row["password_hash"],
                    access_level=TwinAccessLevel(row["access_level"]),
                    permissions=json.loads(row["permissions"]),
                    display_name=row["display_name"],
                    avatar_url=row["avatar_url"],
                    bio=row["bio"],
                    language=row["language"],
                    timezone=row["timezone"],
                    privacy_settings=json.loads(row["privacy_settings"]),
                    created_at=row["created_at"],
                    last_active=row["last_active"],
                    is_active=bool(row["is_active"]),
                )

                self.storage_metrics["database_operations"] += 1
                return user

        except Exception as e:
            logger.error(f"Failed to load user {user_id}: {e}")
            return None

    async def save_conversation(self, conversation: TwinConversation) -> bool:
        """Save conversation to database."""

        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT OR REPLACE INTO conversations
                    (conversation_id, user_id, title, created_at, last_message_at,
                     message_count, context_length, temperature, system_prompt,
                     is_private, retention_days)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        conversation.conversation_id,
                        conversation.user_id,
                        conversation.title,
                        conversation.created_at,
                        conversation.last_message_at,
                        conversation.message_count,
                        conversation.context_length,
                        conversation.temperature,
                        conversation.system_prompt,
                        conversation.is_private,
                        conversation.retention_days,
                    ),
                )

                conn.commit()
                self.storage_metrics["database_operations"] += 1
                return True

        except Exception as e:
            logger.error(f"Failed to save conversation {conversation.conversation_id}: {e}")
            return False

    async def save_message(self, message: TwinMessage) -> bool:
        """Save message to database."""

        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO messages
                    (message_id, conversation_id, user_id, content, message_type,
                     timestamp, tokens_used, processing_time_ms, context_data, references)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        message.message_id,
                        message.conversation_id,
                        message.user_id,
                        message.content,
                        message.message_type,
                        message.timestamp,
                        message.tokens_used,
                        message.processing_time_ms,
                        json.dumps(message.context_data),
                        json.dumps(message.references),
                    ),
                )

                conn.commit()
                self.storage_metrics["database_operations"] += 1
                return True

        except Exception as e:
            logger.error(f"Failed to save message {message.message_id}: {e}")
            return False

    async def get_conversation_messages(self, conversation_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get conversation message history with proper pagination."""

        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT * FROM messages
                    WHERE conversation_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """,
                    (conversation_id, limit),
                )

                messages = []
                for row in cursor.fetchall():
                    messages.append(
                        {
                            "message_id": row["message_id"],
                            "content": row["content"],
                            "message_type": row["message_type"],
                            "timestamp": row["timestamp"],
                            "tokens_used": row["tokens_used"],
                            "processing_time_ms": row["processing_time_ms"],
                            "context_data": json.loads(row["context_data"]),
                            "references": json.loads(row["references"]),
                        }
                    )

                self.storage_metrics["database_operations"] += 1
                return list(reversed(messages))  # Return chronological order

        except Exception as e:
            logger.error(f"Failed to get messages for conversation {conversation_id}: {e}")
            return []

    async def store_twin_data(
        self,
        user_id: str,
        data_type: str,
        data_key: str,
        data_value: Any,
        access_level: TwinAccessLevel = TwinAccessLevel.PRIVATE,
        expires_at: Optional[float] = None,
    ) -> Optional[str]:
        """Store arbitrary twin data with encryption."""

        try:
            data_id = f"{user_id}_{data_type}_{data_key}_{int(time.time())}"

            # Serialize data value
            if isinstance(data_value, (dict, list)):
                serialized_value = json.dumps(data_value)
            else:
                serialized_value = str(data_value)

            # Encrypt if enabled
            if self.enable_encryption:
                serialized_value = self._encrypt_data(serialized_value)

            with self._get_db_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT OR REPLACE INTO twin_data
                    (data_id, user_id, data_type, data_key, data_value, access_level,
                     created_at, updated_at, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        data_id,
                        user_id,
                        data_type,
                        data_key,
                        serialized_value,
                        access_level.value,
                        time.time(),
                        time.time(),
                        expires_at,
                    ),
                )

                conn.commit()
                self.storage_metrics["database_operations"] += 1
                return data_id

        except Exception as e:
            logger.error(f"Failed to store twin data for user {user_id}: {e}")
            return None

    async def get_twin_data(self, user_id: str, data_type: str, data_key: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve twin data with decryption."""

        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()

                if data_key:
                    cursor.execute(
                        """
                        SELECT * FROM twin_data
                        WHERE user_id = ? AND data_type = ? AND data_key = ?
                        AND (expires_at IS NULL OR expires_at > ?)
                    """,
                        (user_id, data_type, data_key, time.time()),
                    )
                else:
                    cursor.execute(
                        """
                        SELECT * FROM twin_data
                        WHERE user_id = ? AND data_type = ?
                        AND (expires_at IS NULL OR expires_at > ?)
                    """,
                        (user_id, data_type, time.time()),
                    )

                results = {}
                for row in cursor.fetchall():
                    # Decrypt if needed
                    data_value = row["data_value"]
                    if self.enable_encryption:
                        data_value = self._decrypt_data(data_value)

                    # Deserialize
                    try:
                        deserialized_value = json.loads(data_value)
                    except json.JSONDecodeError:
                        deserialized_value = data_value

                    results[row["data_key"]] = {
                        "value": deserialized_value,
                        "access_level": row["access_level"],
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                    }

                self.storage_metrics["database_operations"] += 1
                return results

        except Exception as e:
            logger.error(f"Failed to get twin data for user {user_id}: {e}")
            return {}

    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key with secure storage."""

        key_file = self.data_dir / "encryption.key"

        if key_file.exists():
            with open(key_file, "rb") as f:
                return f.read()
        else:
            # Generate new key
            import secrets

            key = secrets.token_bytes(32)
            with open(key_file, "wb") as f:
                f.write(key)
            key_file.chmod(0o600)  # Restrict permissions
            return key

    def _encrypt_data(self, data: str) -> str:
        """Encrypt data using AES (simplified implementation)."""

        if not self.enable_encryption or not self.encryption_key:
            return data

        # Simple XOR encryption for demonstration
        # In production, use proper AES encryption
        encrypted = bytearray()
        key_len = len(self.encryption_key)

        for i, byte in enumerate(data.encode("utf-8")):
            encrypted.append(byte ^ self.encryption_key[i % key_len])

        self.storage_metrics["encryption_operations"] += 1
        return encrypted.hex()

    def _decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt data using AES (simplified implementation)."""

        if not self.enable_encryption or not self.encryption_key:
            return encrypted_data

        try:
            encrypted_bytes = bytes.fromhex(encrypted_data)
            decrypted = bytearray()
            key_len = len(self.encryption_key)

            for i, byte in enumerate(encrypted_bytes):
                decrypted.append(byte ^ self.encryption_key[i % key_len])

            return decrypted.decode("utf-8")

        except Exception:
            return encrypted_data  # Return as-is if decryption fails

    async def cleanup_expired_data(self) -> int:
        """Clean up expired data entries."""

        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                current_time = time.time()

                cursor.execute(
                    """
                    DELETE FROM twin_data
                    WHERE expires_at IS NOT NULL AND expires_at < ?
                """,
                    (current_time,),
                )

                deleted_count = cursor.rowcount
                if deleted_count > 0:
                    conn.commit()
                    logger.info(f"Cleaned up {deleted_count} expired data records")

                self.storage_metrics["database_operations"] += 1
                return deleted_count

        except Exception as e:
            logger.error(f"Failed to cleanup expired data: {e}")
            return 0

    async def close(self):
        """Close database connections with proper cleanup."""

        try:
            if hasattr(self.database_manager, "close"):
                self.database_manager.close()
            logger.info("Storage connections closed successfully")

        except Exception as e:
            logger.error(f"Error closing storage connections: {e}")

    def get_storage_metrics(self) -> Dict[str, Any]:
        """Get storage performance metrics."""

        return {
            "database_operations": self.storage_metrics["database_operations"],
            "encryption_operations": self.storage_metrics["encryption_operations"],
            "file_operations": self.storage_metrics["file_operations"],
            "encryption_enabled": self.enable_encryption,
            "database_path": str(self.db_path),
            "data_directory": str(self.data_dir),
        }
