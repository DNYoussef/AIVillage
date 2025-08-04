"""SQLite-based storage backend for HyperAG MCP server."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from typing import Any


class SQLiteStorage:
    """Simple SQLite storage backend for knowledge items."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                content_type TEXT NOT NULL,
                metadata TEXT
            )
            """
        )
        self.conn.commit()

    async def add_knowledge(
        self,
        node_id: str,
        content: str,
        content_type: str = "text",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        def _insert() -> None:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO knowledge (id, content, content_type, metadata) VALUES (?, ?, ?, ?)",
                (node_id, content, content_type, json.dumps(metadata or {})),
            )
            self.conn.commit()

        await asyncio.to_thread(_insert)

    async def get_knowledge(self, node_id: str) -> dict[str, Any] | None:
        def _get() -> dict[str, Any] | None:
            cur = self.conn.cursor()
            cur.execute(
                "SELECT id, content, content_type, metadata FROM knowledge WHERE id=?",
                (node_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            node_id, content, content_type, metadata = row
            return {
                "id": node_id,
                "content": content,
                "content_type": content_type,
                "metadata": json.loads(metadata) if metadata else {},
            }

        return await asyncio.to_thread(_get)

    async def search_knowledge(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        like_query = f"%{query}%"

        def _search() -> list[dict[str, Any]]:
            cur = self.conn.cursor()
            cur.execute(
                "SELECT id, content, content_type, metadata FROM knowledge WHERE content LIKE ? LIMIT ?",
                (like_query, limit),
            )
            rows = cur.fetchall()
            results = []
            for row in rows:
                node_id, content, content_type, metadata = row
                results.append(
                    {
                        "id": node_id,
                        "content": content,
                        "content_type": content_type,
                        "metadata": json.loads(metadata) if metadata else {},
                        "relevance": 1.0,
                    }
                )
            return results

        return await asyncio.to_thread(_search)

    async def update_knowledge(
        self,
        node_id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        def _update() -> None:
            cur = self.conn.cursor()
            if content is not None:
                cur.execute(
                    "UPDATE knowledge SET content=? WHERE id=?", (content, node_id)
                )
            if metadata is not None:
                cur.execute(
                    "UPDATE knowledge SET metadata=? WHERE id=?",
                    (json.dumps(metadata), node_id),
                )
            self.conn.commit()

        await asyncio.to_thread(_update)

    async def delete_knowledge(self, node_id: str) -> None:
        def _delete() -> None:
            cur = self.conn.cursor()
            cur.execute("DELETE FROM knowledge WHERE id=?", (node_id,))
            self.conn.commit()

        await asyncio.to_thread(_delete)

    async def close(self) -> None:
        await asyncio.to_thread(self.conn.close)
