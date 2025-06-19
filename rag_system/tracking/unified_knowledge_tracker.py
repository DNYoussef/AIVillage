# rag_system/tracking/unified_knowledge_tracker.py

from datetime import datetime
from typing import Any, Dict, List, Optional
from ..core.structures import RetrievalResult
from dataclasses import dataclass, field

@dataclass
class KnowledgeChange:
    entity: str
    relation: str
    old_value: Any
    new_value: Any
    timestamp: datetime
    source: str

class UnifiedKnowledgeTracker:
    def __init__(self, vector_store, graph_store):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.knowledge_changes: List[KnowledgeChange] = []
        self.knowledge_graph: Dict[str, Dict[str, Any]] = {}
        self.retrieval_log: List[Dict[str, Any]] = []

    def record_change(self, change: KnowledgeChange):
        self.knowledge_changes.append(change)
        self._update_knowledge_graph(change)

    def track_changes(self, result: Dict[str, Any], timestamp: datetime):
        # Implement logic to extract and record changes from the result
        # This is a simplified version and should be expanded based on your specific needs
        for key, value in result.items():
            change = KnowledgeChange(
                entity=key,
                relation="updated",
                old_value=self.knowledge_graph.get(key, {}).get("value"),
                new_value=value,
                timestamp=timestamp,
                source="result_processing"
            )
            self.record_change(change)

    def _update_knowledge_graph(self, change: KnowledgeChange):
        if change.entity not in self.knowledge_graph:
            self.knowledge_graph[change.entity] = {}
        
        self.knowledge_graph[change.entity][change.relation] = change.new_value

    def get_entity_history(self, entity: str) -> List[KnowledgeChange]:
        return [change for change in self.knowledge_changes if change.entity == entity]

    def get_current_knowledge(self, entity: str) -> Dict[str, Any]:
        return self.knowledge_graph.get(entity, {})

    def record_retrieval(
        self,
        query: str,
        results: List["RetrievalResult"],
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Record retrieval results for auditing and analysis."""
        self.retrieval_log.append(
            {
                "query": query,
                "results": results,
                "timestamp": timestamp or datetime.now(),
            }
        )

    def update_vector_store(self):
        """Synchronize recorded changes with the underlying vector store.

        This method iterates over all recorded ``KnowledgeChange`` instances and
        updates or creates corresponding entries in ``self.vector_store``.  The
        expectation is that ``change.new_value`` contains either the raw
        document dictionary used by :class:`VectorStore.add_documents` or a
        dictionary with enough information to construct one.  If a document with
        the same ``id`` already exists in the store it will be updated via
        ``update_document``; otherwise a new document is added.
        """

        for change in self.knowledge_changes:
            new_val = change.new_value

            if not isinstance(new_val, dict) or "embedding" not in new_val:
                # Nothing we can do if there is no embedding information
                continue

            doc_id = new_val.get("id", f"{change.entity}_{change.relation}")
            document = {
                "id": doc_id,
                "content": new_val.get("content", str(new_val)),
                "embedding": new_val["embedding"],
                "timestamp": new_val.get("timestamp", change.timestamp),
            }

            existing = self.vector_store.get_document_by_id(doc_id)
            if existing:
                self.vector_store.update_document(doc_id, document)
            else:
                self.vector_store.add_documents([document])

    def update_graph_store(self):
        """Propagate knowledge changes to the underlying graph store."""

        for change in self.knowledge_changes:
            new_val = change.new_value

            if not isinstance(new_val, dict) or "id" not in new_val:
                continue

            doc_id = new_val["id"]
            existing = self.graph_store.get_document_by_id(doc_id)

            if existing:
                # Update attributes of the existing node
                self.graph_store.graph.nodes[doc_id].update(new_val)
            else:
                # Fallback to add_documents which will create a new node
                self.graph_store.add_documents([new_val])

    def analyze_knowledge_evolution(self) -> Dict[str, Any]:
        """Provide simple statistics describing the evolution of knowledge."""

        summary = {
            "total_changes": len(self.knowledge_changes),
            "entities_changed": len({c.entity for c in self.knowledge_changes}),
            "last_update": max((c.timestamp for c in self.knowledge_changes), default=None),
        }

        if not self.knowledge_changes:
            summary.update({
                "most_changed_entity": None,
                "average_time_between_changes": None,
            })
            return summary

        # Determine which entity has been modified the most
        entity_counts: Dict[str, int] = {}
        for change in self.knowledge_changes:
            entity_counts[change.entity] = entity_counts.get(change.entity, 0) + 1

        most_changed_entity = max(entity_counts.items(), key=lambda x: x[1])[0]

        # Calculate the average time between successive changes
        sorted_changes = sorted(self.knowledge_changes, key=lambda c: c.timestamp)
        if len(sorted_changes) > 1:
            diffs = [
                (sorted_changes[i].timestamp - sorted_changes[i - 1].timestamp).total_seconds()
                for i in range(1, len(sorted_changes))
            ]
            avg_diff = sum(diffs) / len(diffs)
        else:
            avg_diff = None

        summary.update({
            "most_changed_entity": most_changed_entity,
            "average_time_between_changes": avg_diff,
        })

        return summary
