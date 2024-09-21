from typing import List, Optional
from collections import deque
from .message import Message, Priority

class MessageQueue:
    def __init__(self):
        self._queues: Dict[Priority, deque[Message]] = {
            Priority.LOW: deque(),
            Priority.MEDIUM: deque(),
            Priority.HIGH: deque(),
            Priority.CRITICAL: deque()
        }

    def enqueue(self, message: Message) -> None:
        self._queues[message.priority].append(message)

    def dequeue(self) -> Optional[Message]:
        for priority in reversed(Priority):
            if self._queues[priority]:
                return self._queues[priority].popleft()
        return None

    def is_empty(self) -> bool:
        return all(len(queue) == 0 for queue in self._queues.values())

    def get_messages_by_priority(self, priority: Priority) -> List[Message]:
        return list(self._queues[priority])

    def get_all_messages(self) -> List[Message]:
        all_messages = []
        for priority in reversed(Priority):
            all_messages.extend(self.get_messages_by_priority(priority))
        return all_messages