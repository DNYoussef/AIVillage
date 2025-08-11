import asyncio
import unittest

from core.error_handling import (
    Message,
    MessageType,
    Priority,
    StandardCommunicationProtocol,
)


class TestProtocol(unittest.IsolatedAsyncioTestCase):
    async def test_send_and_wait(self):
        protocol = StandardCommunicationProtocol()

        async def echo(msg: Message):
            response = Message(
                type=MessageType.RESPONSE,
                sender="receiver",
                receiver=msg.sender,
                content={"echo": True},
                parent_id=msg.id,
            )
            await protocol.send_message(response)

        protocol.subscribe("receiver", echo)

        request = Message(
            type=MessageType.QUERY,
            sender="sender",
            receiver="receiver",
            content={"ping": True},
        )
        response = await protocol.send_and_wait(request)
        assert response.content["echo"]

    async def test_priority_order(self):
        protocol = StandardCommunicationProtocol()
        low = Message(
            type=MessageType.NOTIFICATION,
            sender="s",
            receiver="r",
            content={},
            priority=Priority.LOW,
        )
        high = Message(
            type=MessageType.NOTIFICATION,
            sender="s",
            receiver="r",
            content={},
            priority=Priority.CRITICAL,
        )
        medium = Message(
            type=MessageType.NOTIFICATION,
            sender="s",
            receiver="r",
            content={},
            priority=Priority.MEDIUM,
        )
        await protocol.send_message(low)
        await protocol.send_message(high)
        await protocol.send_message(medium)
        msg1 = await protocol.receive_message("r")
        msg2 = await protocol.receive_message("r")
        msg3 = await protocol.receive_message("r")
        assert [msg1.priority, msg2.priority, msg3.priority] == [
            Priority.CRITICAL,
            Priority.MEDIUM,
            Priority.LOW,
        ]

    async def test_broadcast_and_unsubscribe(self):
        protocol = StandardCommunicationProtocol()
        received = []

        async def cb(msg: Message):
            received.append(msg.receiver)

        protocol.subscribe("a", cb)
        protocol.subscribe("b", cb)
        protocol.unsubscribe("b", cb)

        await protocol.broadcast(sender="sys", message_type=MessageType.NOTIFICATION, content={"msg": 1})
        await asyncio.sleep(0)
        assert received == ["a"]

    async def test_history_and_process(self):
        protocol = StandardCommunicationProtocol()
        processed = []

        async def handler(msg: Message):
            processed.append(msg)

        task = asyncio.create_task(protocol.process_messages(handler))
        await protocol.send_message(Message(type=MessageType.NOTIFICATION, sender="s", receiver="r", content={}))
        await asyncio.sleep(0.05)
        protocol._running = False
        await asyncio.sleep(0.01)
        assert processed
        hist_r = protocol.get_message_history("r")
        assert len(hist_r) == 1
        hist_s = protocol.get_message_history("s")
        assert len(hist_s) == 1
        # ensure filtering by message type works
        assert len(protocol.get_message_history("r", MessageType.NOTIFICATION)) == 1
        assert protocol.get_message_history("r", MessageType.QUERY) == []
        task.cancel()

    async def test_history_no_duplicate_on_receive(self):
        protocol = StandardCommunicationProtocol()
        msg = Message(type=MessageType.NOTIFICATION, sender="a", receiver="b", content={})
        await protocol.send_message(msg)
        # receive the message and ensure history not duplicated
        await protocol.receive_message("b")
        hist = protocol.get_message_history("b")
        assert len(hist) == 1


if __name__ == "__main__":
    unittest.main()
