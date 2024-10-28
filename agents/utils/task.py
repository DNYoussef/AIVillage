"""Task class for agent operations."""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class Task:
    """Task class for agent operations."""
    content: str
    agent: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    type: str = "default"
    priority: int = 1
    task_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Initialize any derived fields after instance creation."""
        if self.task_id is None:
            self.task_id = f"task_{self.created_at.timestamp()}"

    async def run(self) -> Dict[str, Any]:
        """Execute the task."""
        try:
            result = await self.agent.execute_task(self)
            return {
                "status": "success",
                "result": result,
                "task_id": self.task_id,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "task_id": self.task_id,
                "timestamp": datetime.now().isoformat()
            }

    def update_metadata(self, updates: Dict[str, Any]) -> None:
        """Update task metadata."""
        self.metadata.update(updates)

    def add_context(self, context: Dict[str, Any]) -> None:
        """Add context to task metadata."""
        if "context" not in self.metadata:
            self.metadata["context"] = {}
        self.metadata["context"].update(context)

    def get_context(self) -> Dict[str, Any]:
        """Get task context."""
        return self.metadata.get("context", {})

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary format."""
        return {
            "content": self.content,
            "type": self.type,
            "priority": self.priority,
            "task_id": self.task_id,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], agent: Any) -> 'Task':
        """Create task from dictionary format."""
        return cls(
            content=data["content"],
            agent=agent,
            metadata=data.get("metadata", {}),
            type=data.get("type", "default"),
            priority=data.get("priority", 1),
            task_id=data.get("task_id"),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now()
        )
