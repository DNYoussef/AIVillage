# rag_system/tracking/knowledge_evolution_tracker.py

from typing import Any, List, Dict
import asyncio
from datetime import datetime
from ..retrieval.vector_store import VectorStore
from ..retrieval.graph_store import GraphStore

class KnowledgeEvolutionTracker:
    def __init__(self, vector_store: VectorStore, graph_store: GraphStore):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.change_log: List[Dict[str, Any]] = []

    async def track_change(self, entity_id: str, old_state: Any, new_state: Any, timestamp: datetime):
        change_record = {
            "entity_id": entity_id,
            "old_state": old_state,
            "new_state": new_state,
            "timestamp": timestamp
        }
        self.change_log.append(change_record)

        await self._store_change_record(change_record)

    async def get_evolution(self, entity_id: str, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        evolution = [
            change for change in self.change_log
            if change["entity_id"] == entity_id and start_time <= change["timestamp"] <= end_time
        ]
        return sorted(evolution, key=lambda x: x["timestamp"])

    async def get_knowledge_snapshot(self, timestamp: datetime) -> Dict[str, Any]:
        vector_snapshot = await self.vector_store.get_snapshot(timestamp)
        graph_snapshot = await self.graph_store.get_snapshot(timestamp)

        return {
            "timestamp": timestamp,
            "vector_knowledge": vector_snapshot,
            "graph_knowledge": graph_snapshot
        }

    async def _store_change_record(self, record: Dict[str, Any]):
        """Persist a change record to a simple JSONL log for now."""
        import json, os
        log_path = os.path.join(os.path.dirname(__file__), "evolution.log")
        async with asyncio.Lock():
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
