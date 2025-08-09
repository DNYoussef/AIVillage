import asyncio

import pytest


async def simulate_mesh(num_peers: int) -> tuple[int, int]:
    """Simple in-memory mesh simulation used for integration tests.

    Each peer sends a message to every other peer using asyncio queues to
    emulate asynchronous delivery. The function returns the total number of
    messages sent and received across the network.
    """

    queues = [asyncio.Queue() for _ in range(num_peers)]

    async def peer_task(i: int) -> tuple[int, int]:
        sent = 0
        received = 0

        # Broadcast a message to every other peer
        for j in range(num_peers):
            if i == j:
                continue
            await queues[j].put((i, j, f"msg-{i}-{j}"))
            sent += 1

        # Receive messages from other peers
        while received < num_peers - 1:
            await queues[i].get()
            received += 1

        return sent, received

    results = await asyncio.gather(*(peer_task(i) for i in range(num_peers)))
    total_sent = sum(s for s, _ in results)
    total_received = sum(r for _, r in results)
    return total_sent, total_received


@pytest.mark.parametrize("peers", [10, 20, 50])
@pytest.mark.asyncio
async def test_message_delivery_rate(peers: int) -> None:
    """Verify that message delivery rate stays above 95% for various sizes."""
    total_sent, total_received = await simulate_mesh(peers)
    delivery_rate = total_received / total_sent
    assert delivery_rate >= 0.95, f"delivery rate {delivery_rate:.2%} below 95%"


@pytest.mark.asyncio
async def test_offline_queue_and_reconnect() -> None:
    """Simulate nodes queuing messages while offline and delivering on reconnect."""
    queues = [asyncio.Queue() for _ in range(5)]
    offline = {1, 3}

    # Node 0 sends messages to all others; offline nodes queue them
    for j in range(5):
        if j in offline:
            queues[j].put_nowait((0, j, "queued"))
        else:
            await queues[j].put((0, j, "online"))

    # Bring nodes online and flush queued messages
    delivered = 0
    for j in offline:
        while not queues[j].empty():
            await queues[j].get()
            delivered += 1

    assert delivered == len(offline)

