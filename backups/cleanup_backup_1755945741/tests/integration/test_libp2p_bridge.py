import asyncio

import pytest

from packages.p2p.core.libp2p_mesh import LibP2PMeshNetwork, MeshMessage, MeshMessageType


@pytest.mark.asyncio
async def test_libp2p_bridge_communication():
    # Create two mesh networks
    network1 = LibP2PMeshNetwork()
    network2 = LibP2PMeshNetwork()

    # Start the networks
    await network1.start()
    await network2.start()

    # Create a message to send
    message = MeshMessage(
        type=MeshMessageType.DATA_MESSAGE,
        payload=b"Hello, world!",
    )

    # Send the message from network1 to network2
    await network1.send_message(message)

    # Wait for the message to be received
    await asyncio.sleep(1)

    # Check that the message was received
    assert len(network2.message_handlers) == 1
    assert network2.message_handlers[0].payload == b"Hello, world!"

    # Stop the networks
    await network1.stop()
    await network2.stop()
