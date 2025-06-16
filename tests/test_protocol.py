import unittest
import asyncio

from communications.protocol import StandardCommunicationProtocol, Message, MessageType, Priority

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

        request = Message(type=MessageType.QUERY, sender="sender", receiver="receiver", content={"ping": True})
        response = await protocol.send_and_wait(request)
        self.assertEqual(response.content["echo"], True)

    async def test_priority_order(self):
        protocol = StandardCommunicationProtocol()
        low = Message(type=MessageType.NOTIFICATION, sender="s", receiver="r", content={}, priority=Priority.LOW)
        high = Message(type=MessageType.NOTIFICATION, sender="s", receiver="r", content={}, priority=Priority.CRITICAL)
        medium = Message(type=MessageType.NOTIFICATION, sender="s", receiver="r", content={}, priority=Priority.MEDIUM)
        await protocol.send_message(low)
        await protocol.send_message(high)
        await protocol.send_message(medium)
        msg1 = await protocol.receive_message("r")
        msg2 = await protocol.receive_message("r")
        msg3 = await protocol.receive_message("r")
        self.assertEqual([msg1.priority, msg2.priority, msg3.priority], [Priority.CRITICAL, Priority.MEDIUM, Priority.LOW])

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
        self.assertEqual(received, ["a"])

if __name__ == "__main__":
    unittest.main()
