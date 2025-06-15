import unittest
import asyncio

from communications.protocol import StandardCommunicationProtocol, Message, MessageType

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

if __name__ == "__main__":
    unittest.main()
