# rag_system/knowledge_tracker.py

from dataclasses import dataclass
from datetime import datetime


@dataclass
class KnowledgeChange:
    entity: str
    relation: str
    old_value: str
    new_value: str
    timestamp: datetime
    source: str


class KnowledgeTracker:
    def __init__(self) -> None:
        self.changes = []
        self.knowledge_graph = {}

    def record_change(self, change: KnowledgeChange) -> None:
        """Record a change made to the knowledge graph.

        :param change: An instance of KnowledgeChange representing the modification.
        """
        self.changes.append(change)
        entity_data = self.knowledge_graph.setdefault(change.entity, {})
        entity_data[change.relation] = change.new_value

    def get_entity_history(self, entity: str):
        """Retrieve the history of changes for a specific entity.

        :param entity: The entity to retrieve history for.
        :return: A list of KnowledgeChange instances related to the entity.
        """
        return [change for change in self.changes if change.entity == entity]

    def rollback_change(self, change_id: int) -> None:
        """Roll back a specific change by its ID.

        :param change_id: The index of the change in the changes list.
        """
        if not (0 <= change_id < len(self.changes)):
            msg = "Invalid change_id"
            raise ValueError(msg)

        change = self.changes.pop(change_id)
        entity_data = self.knowledge_graph.get(change.entity, {})
        if change.relation in entity_data:
            entity_data[change.relation] = change.old_value
