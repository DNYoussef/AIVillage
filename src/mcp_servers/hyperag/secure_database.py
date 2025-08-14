"""Secure Database Manager for HypeRAG.

Fixes SQL injection vulnerabilities identified in security audit.
"""

import asyncio
import hashlib
import json
import logging
import sqlite3
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class SecureDatabaseManager:
    """Secure database manager with parameterized queries to prevent SQL injection."""

    def __init__(self, db_path: str = "data/hyperag_secure.db") -> None:
        self.db_path = db_path
        self.connection = None
        self._init_database()

    def _init_database(self) -> None:
        """Initialize secure database with proper schema."""
        try:
            self.connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                isolation_level=None,  # Autocommit mode
            )

            # Enable foreign keys and secure settings
            self.connection.execute("PRAGMA foreign_keys = ON")
            self.connection.execute("PRAGMA journal_mode = WAL")
            self.connection.execute("PRAGMA synchronous = NORMAL")

            # Create secure schema
            self._create_secure_schema()
            logger.info(f"Secure database initialized: {self.db_path}")

        except Exception as e:
            logger.exception(f"Failed to initialize secure database: {e}")
            raise

    def _create_secure_schema(self) -> None:
        """Create secure database schema with proper constraints."""
        schema_queries = [
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_hash TEXT NOT NULL UNIQUE,
                content TEXT NOT NULL,
                tags TEXT,  -- JSON array
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0,
                importance_score REAL DEFAULT 0.0
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS knowledge_graph (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(subject, predicate, object)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_hash TEXT NOT NULL UNIQUE,
                embedding BLOB NOT NULL,
                dimension INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (content_hash) REFERENCES memories(content_hash)
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_memories_hash ON memories(content_hash)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_memories_tags ON memories(tags)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_kg_subject ON knowledge_graph(subject)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_kg_predicate ON knowledge_graph(predicate)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_embeddings_hash ON embeddings(content_hash)
            """,
        ]

        for query in schema_queries:
            self.connection.execute(query)

    def _generate_content_hash(self, content: str) -> str:
        """Generate secure hash for content using SHA-256."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    async def store_memory(
        self,
        content: str,
        tags: list[str] | None = None,
        importance_score: float = 0.0,
    ) -> str:
        """Store memory with secure parameterized query."""
        content_hash = self._generate_content_hash(content)
        tags_json = json.dumps(tags or [])

        try:
            # Use parameterized query to prevent SQL injection
            self.connection.execute(
                """
                INSERT OR REPLACE INTO memories
                (content_hash, content, tags, importance_score, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (content_hash, content, tags_json, importance_score, datetime.now()),
            )

            logger.info(f"Stored memory with hash: {content_hash}")
            return content_hash

        except Exception as e:
            logger.exception(f"Failed to store memory: {e}")
            raise

    async def retrieve_memory(self, content_hash: str) -> dict[str, Any] | None:
        """Retrieve memory with secure parameterized query."""
        try:
            cursor = self.connection.execute(
                """
                SELECT content, tags, created_at, updated_at, access_count, importance_score
                FROM memories
                WHERE content_hash = ?
                """,
                (content_hash,),
            )

            row = cursor.fetchone()
            if not row:
                return None

            # Update access count securely
            self.connection.execute(
                "UPDATE memories SET access_count = access_count + 1 WHERE content_hash = ?",
                (content_hash,),
            )

            return {
                "content_hash": content_hash,
                "content": row[0],
                "tags": json.loads(row[1]),
                "created_at": row[2],
                "updated_at": row[3],
                "access_count": row[4] + 1,
                "importance_score": row[5],
            }

        except Exception as e:
            logger.exception(f"Failed to retrieve memory: {e}")
            return None

    async def search_memories(self, query: str, tags: list[str] | None = None, limit: int = 10) -> list[dict[str, Any]]:
        """Search memories with secure parameterized queries."""
        try:
            # Build secure query with parameterized placeholders
            base_query = """
                SELECT content_hash, content, tags, created_at, importance_score
                FROM memories
                WHERE content LIKE ?
            """
            params = [f"%{query}%"]

            # Add tag filtering if specified
            if tags:
                tag_conditions = []
                for tag in tags:
                    tag_conditions.append("tags LIKE ?")
                    params.append(f'%"{tag}"%')

                base_query += " AND (" + " OR ".join(tag_conditions) + ")"

            base_query += " ORDER BY importance_score DESC, created_at DESC LIMIT ?"
            params.append(limit)

            cursor = self.connection.execute(base_query, params)
            rows = cursor.fetchall()

            results = []
            for row in rows:
                results.append(
                    {
                        "content_hash": row[0],
                        "content": row[1],
                        "tags": json.loads(row[2]),
                        "created_at": row[3],
                        "importance_score": row[4],
                    }
                )

            return results

        except Exception as e:
            logger.exception(f"Failed to search memories: {e}")
            return []

    async def store_knowledge_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 1.0,
        source: str | None = None,
    ) -> bool:
        """Store knowledge graph triple with secure parameterized query."""
        try:
            self.connection.execute(
                """
                INSERT OR REPLACE INTO knowledge_graph
                (subject, predicate, object, confidence, source)
                VALUES (?, ?, ?, ?, ?)
                """,
                (subject, predicate, obj, confidence, source),
            )

            logger.debug(f"Stored knowledge triple: ({subject}, {predicate}, {obj})")
            return True

        except Exception as e:
            logger.exception(f"Failed to store knowledge triple: {e}")
            return False

    async def query_knowledge_graph(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        obj: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query knowledge graph with secure parameterized queries."""
        try:
            # Build dynamic query with proper parameterization
            conditions = []
            params = []

            if subject:
                conditions.append("subject = ?")
                params.append(subject)

            if predicate:
                conditions.append("predicate = ?")
                params.append(predicate)

            if obj:
                conditions.append("object = ?")
                params.append(obj)

            base_query = "SELECT subject, predicate, object, confidence, source, created_at FROM knowledge_graph"

            if conditions:
                base_query += " WHERE " + " AND ".join(conditions)

            base_query += " ORDER BY confidence DESC, created_at DESC LIMIT ?"
            params.append(limit)

            cursor = self.connection.execute(base_query, params)
            rows = cursor.fetchall()

            results = []
            for row in rows:
                results.append(
                    {
                        "subject": row[0],
                        "predicate": row[1],
                        "object": row[2],
                        "confidence": row[3],
                        "source": row[4],
                        "created_at": row[5],
                    }
                )

            return results

        except Exception as e:
            logger.exception(f"Failed to query knowledge graph: {e}")
            return []

    async def get_database_stats(self) -> dict[str, Any]:
        """Get database statistics securely."""
        try:
            stats = {}

            # Count memories
            cursor = self.connection.execute("SELECT COUNT(*) FROM memories")
            stats["memory_count"] = cursor.fetchone()[0]

            # Count knowledge triples
            cursor = self.connection.execute("SELECT COUNT(*) FROM knowledge_graph")
            stats["knowledge_triple_count"] = cursor.fetchone()[0]

            # Count embeddings
            cursor = self.connection.execute("SELECT COUNT(*) FROM embeddings")
            stats["embedding_count"] = cursor.fetchone()[0]

            # Database size
            cursor = self.connection.execute(
                "SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()"
            )
            stats["database_size_bytes"] = cursor.fetchone()[0]

            return stats

        except Exception as e:
            logger.exception(f"Failed to get database stats: {e}")
            return {}

    def close(self) -> None:
        """Close database connection securely."""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")


# Security utility functions
class SecureHashingManager:
    """Secure hashing manager using SHA-256 instead of MD5."""

    @staticmethod
    def hash_content(content: str, salt: str | None = None) -> str:
        """Generate secure SHA-256 hash."""
        if salt:
            content = f"{content}{salt}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def verify_hash(content: str, hash_value: str, salt: str | None = None) -> bool:
        """Verify content against hash."""
        return SecureHashingManager.hash_content(content, salt) == hash_value


# Input validation utilities
class SecureInputValidator:
    """Secure input validation to prevent injection attacks."""

    @staticmethod
    def sanitize_string(input_str: str, max_length: int = 1000) -> str:
        """Sanitize string input."""
        if not isinstance(input_str, str):
            msg = "Input must be a string"
            raise ValueError(msg)

        # Remove null bytes and control characters
        sanitized = "".join(char for char in input_str if ord(char) >= 32 or char in "\n\r\t")

        # Truncate to max length
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]

        return sanitized

    @staticmethod
    def validate_hash(hash_value: str) -> bool:
        """Validate hash format."""
        if not isinstance(hash_value, str):
            return False

        # SHA-256 hash should be 64 hex characters
        if len(hash_value) != 64:
            return False

        try:
            int(hash_value, 16)
            return True
        except ValueError:
            return False


if __name__ == "__main__":
    # Test the secure database manager
    async def test_secure_database() -> None:
        db = SecureDatabaseManager("test_secure.db")

        # Test memory storage
        hash1 = await db.store_memory("Test content", ["tag1", "tag2"], 0.8)
        print(f"Stored memory: {hash1}")

        # Test memory retrieval
        memory = await db.retrieve_memory(hash1)
        print(f"Retrieved memory: {memory}")

        # Test search
        results = await db.search_memories("Test", ["tag1"])
        print(f"Search results: {results}")

        # Test knowledge graph
        await db.store_knowledge_triple("Python", "is_a", "programming_language", 0.9)
        kg_results = await db.query_knowledge_graph(subject="Python")
        print(f"Knowledge graph: {kg_results}")

        # Test stats
        stats = await db.get_database_stats()
        print(f"Database stats: {stats}")

        db.close()

    asyncio.run(test_secure_database())
