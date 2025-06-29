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

    def test_message_types_exist(self):
        self.assertTrue(hasattr(MessageType, "UPDATE"))
        self.assertTrue(hasattr(MessageType, "COMMAND"))
        self.assertTrue(hasattr(MessageType, "BULK_UPDATE"))
        self.assertTrue(hasattr(MessageType, "PROJECT_UPDATE"))
        self.assertTrue(hasattr(MessageType, "SYSTEM_STATUS_UPDATE"))
        self.assertTrue(hasattr(MessageType, "CONFIG_UPDATE"))

if __name__ == "__main__":
    unittest.main()
