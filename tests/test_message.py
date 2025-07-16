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
        assert updated.content == {"x": 2}
        assert updated.priority == msg.priority
        assert updated.metadata == msg.metadata
        pri_msg = msg.with_updated_priority(Priority.HIGH)
        assert pri_msg.priority == Priority.HIGH
        d = msg.to_dict()
        restored = Message.from_dict(d)
        assert restored.metadata == msg.metadata

    def test_message_types_exist(self):
        assert hasattr(MessageType, "UPDATE")
        assert hasattr(MessageType, "COMMAND")
        assert hasattr(MessageType, "BULK_UPDATE")
        assert hasattr(MessageType, "PROJECT_UPDATE")
        assert hasattr(MessageType, "SYSTEM_STATUS_UPDATE")
        assert hasattr(MessageType, "CONFIG_UPDATE")


if __name__ == "__main__":
    unittest.main()
