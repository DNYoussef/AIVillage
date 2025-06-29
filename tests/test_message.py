import unittest
from communications.message import Message, MessageType, Priority

class TestMessageHelpers(unittest.TestCase):
    def test_helper_methods_and_metadata(self):
        msg = Message(
            type=MessageType.TASK,
            sender="a",
            receiver="b",
            content={"x": 1},
            metadata={"m": 1},
        )
        updated = msg.with_updated_content({"x": 2})
        self.assertEqual(updated.content, {"x": 2})
        self.assertEqual(updated.priority, msg.priority)
        self.assertEqual(updated.metadata, msg.metadata)
        pri_msg = msg.with_updated_priority(Priority.HIGH)
        self.assertEqual(pri_msg.priority, Priority.HIGH)
        d = msg.to_dict()
        restored = Message.from_dict(d)
        self.assertEqual(restored.metadata, msg.metadata)

if __name__ == "__main__":
    unittest.main()
