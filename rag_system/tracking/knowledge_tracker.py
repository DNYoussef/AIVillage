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
    def __init__(self):
        self.changes = []

    def record_change(self, change: KnowledgeChange):
        """
        Record a change made to the knowledge graph.

        :param change: An instance of KnowledgeChange representing the modification.
        """
        self.changes.append(change)

    def get_entity_history(self, entity: str):
        """
        Retrieve the history of changes for a specific entity.

        :param entity: The entity to retrieve history for.
        :return: A list of KnowledgeChange instances related to the entity.
        """
        return [change for change in self.changes if change.entity == entity]

    def rollback_change(self, change_id: int):
        """
        Roll back a specific change by its ID.

        :param change_id: The index of the change in the changes list.
        """
        if 0 <= change_id < len(self.changes):
            change = self.changes[change_id]
            # Implement the logic to reverse this change in the knowledge graph
            # This is a placeholder and would need to be implemented based on your KG structure

            # For example:
            # self.knowledge_graph.update_relation(
            #     entity=change.entity,
            #     relation=change.relation,
            #     value=change.old_value
            # )

            self.changes.pop(change_id)
        else:
            raise ValueError("Invalid change_id")
