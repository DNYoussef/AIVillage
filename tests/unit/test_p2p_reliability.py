#!/usr/bin/env python3
"""
P2P Network Reliability Tests
=============================

Comprehensive tests for P2P networking components including:
- Connection establishment and maintenance
- Message delivery guarantees
- Network partition handling
- Peer discovery and routing
- Failure recovery mechanisms
"""

import asyncio
import pytest
import time
import json
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path

# Mock P2P classes for testing
class MockP2PNode:
    """Mock P2P node for testing."""
    
    def __init__(self, node_id: str, port: int = 0):
        self.node_id = node_id
        self.port = port
        self.peers: Dict[str, 'MockP2PNode'] = {}
        self.message_queue: List[Dict[str, Any]] = []
        self.status = "disconnected"
        self.network_failures = 0
        self.sent_messages = 0
        self.received_messages = 0
        self.connection_attempts = 0
        
    async def connect(self, peer_address: str) -> bool:
        """Connect to peer."""
        self.connection_attempts += 1
        await asyncio.sleep(0.01)  # Simulate network delay
        
        # Simulate occasional connection failures
        if self.connection_attempts % 10 == 0:
            self.network_failures += 1
            return False
            
        self.status = "connected"
        return True
    
    async def send_message(self, peer_id: str, message: Dict[str, Any]) -> bool:
        """Send message to peer."""
        self.sent_messages += 1
        
        # Simulate network failures
        if self.sent_messages % 15 == 0:
            self.network_failures += 1
            return False
        
        # Simulate successful message delivery
        if peer_id in self.peers:
            await self.peers[peer_id].receive_message(self.node_id, message)
            return True
        
        return False
    
    async def receive_message(self, sender_id: str, message: Dict[str, Any]):
        """Receive message from peer."""
        self.received_messages += 1
        self.message_queue.append({
            "sender": sender_id,
            "message": message,
            "timestamp": time.time()
        })
    
    async def discover_peers(self) -> List[str]:
        """Discover available peers."""
        # Simulate peer discovery
        await asyncio.sleep(0.1)
        return list(self.peers.keys())
    
    def add_peer(self, peer: 'MockP2PNode'):
        """Add peer to known peers."""
        self.peers[peer.node_id] = peer
        peer.peers[self.node_id] = self
    
    def get_stats(self) -> Dict[str, Any]:
        """Get node statistics."""
        return {
            "node_id": self.node_id,
            "status": self.status,
            "peer_count": len(self.peers),
            "sent_messages": self.sent_messages,
            "received_messages": self.received_messages,
            "network_failures": self.network_failures,
            "connection_attempts": self.connection_attempts,
            "message_queue_size": len(self.message_queue)
        }


class TestP2PNetworkReliability:
    """Test P2P network reliability and failure handling."""

    @pytest.fixture
    def p2p_node(self):
        """Create test P2P node."""
        return MockP2PNode("test_node_1", port=8001)

    @pytest.fixture
    def p2p_network(self):
        """Create test P2P network with multiple nodes."""
        nodes = []
        for i in range(5):
            node = MockP2PNode(f"node_{i}", port=8000 + i)
            nodes.append(node)
        
        # Connect nodes in a mesh topology
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i != j:
                    node1.add_peer(node2)
        
        return nodes

    @pytest.mark.asyncio
    async def test_basic_connection_establishment(self, p2p_node):
        """Test basic P2P connection establishment."""
        peer_address = "127.0.0.1:8002"
        
        success = await p2p_node.connect(peer_address)
        
        assert success is True
        assert p2p_node.status == "connected"
        assert p2p_node.connection_attempts == 1

    @pytest.mark.asyncio
    async def test_connection_failure_handling(self, p2p_node):
        """Test handling of connection failures."""
        # Force connection failure by connecting 10 times
        failures = 0
        for i in range(10):
            success = await p2p_node.connect(f"127.0.0.1:800{i}")
            if not success:
                failures += 1
        
        assert failures > 0
        assert p2p_node.network_failures > 0

    @pytest.mark.asyncio
    async def test_message_delivery_success(self, p2p_network):
        """Test successful message delivery between peers."""
        sender = p2p_network[0]
        receiver = p2p_network[1]
        
        message = {
            "type": "test_message",
            "content": "Hello, peer!",
            "timestamp": time.time()
        }
        
        success = await sender.send_message(receiver.node_id, message)
        
        assert success is True
        assert sender.sent_messages == 1
        assert receiver.received_messages == 1
        assert len(receiver.message_queue) == 1
        assert receiver.message_queue[0]["message"]["content"] == "Hello, peer!"

    @pytest.mark.asyncio
    async def test_message_delivery_failure(self, p2p_node):
        """Test message delivery failure handling."""
        # Try to send message to non-existent peer
        message = {"content": "Test message"}
        
        success = await p2p_node.send_message("non_existent_peer", message)
        
        assert success is False
        assert p2p_node.sent_messages == 1

    @pytest.mark.asyncio
    async def test_peer_discovery(self, p2p_network):
        """Test peer discovery functionality."""
        node = p2p_network[0]
        
        discovered_peers = await node.discover_peers()
        
        assert len(discovered_peers) == 4  # Should discover 4 other peers
        assert "node_1" in discovered_peers
        assert "node_2" in discovered_peers

    @pytest.mark.asyncio
    async def test_network_partition_handling(self, p2p_network):
        """Test handling of network partitions."""
        # Simulate network partition by removing connections
        partitioned_node = p2p_network[0]
        
        # Remove all peer connections
        original_peers = partitioned_node.peers.copy()
        partitioned_node.peers.clear()
        
        # Try to send messages
        message = {"content": "Partition test"}
        results = []
        
        for peer_id in original_peers.keys():
            success = await partitioned_node.send_message(peer_id, message)
            results.append(success)
        
        # All messages should fail due to partition
        assert all(result is False for result in results)

    @pytest.mark.asyncio
    async def test_high_message_throughput(self, p2p_network):
        """Test network performance under high message load."""
        sender = p2p_network[0]
        receiver = p2p_network[1]
        
        num_messages = 100
        messages = [
            {"id": i, "content": f"Message {i}", "timestamp": time.time()}
            for i in range(num_messages)
        ]
        
        start_time = time.time()
        
        # Send messages concurrently
        tasks = [
            sender.send_message(receiver.node_id, msg) 
            for msg in messages
        ]
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Calculate success rate
        successful_sends = sum(1 for result in results if result)
        success_rate = successful_sends / num_messages
        
        # Performance assertions
        assert success_rate > 0.8  # At least 80% success rate
        assert execution_time < 5.0  # Complete within 5 seconds
        assert receiver.received_messages >= successful_sends * 0.9  # Allow for some async delays

    @pytest.mark.asyncio
    async def test_connection_recovery(self, p2p_node):
        """Test connection recovery after failures."""
        peer_address = "127.0.0.1:8002"
        
        # Simulate multiple connection attempts with failures
        attempts = 0
        max_attempts = 5
        
        while attempts < max_attempts:
            success = await p2p_node.connect(peer_address)
            attempts += 1
            
            if success:
                break
            
            # Wait before retry
            await asyncio.sleep(0.1)
        
        # Should eventually succeed
        assert p2p_node.status == "connected"
        assert attempts <= max_attempts

    @pytest.mark.asyncio
    async def test_message_ordering(self, p2p_network):
        """Test message ordering preservation."""
        sender = p2p_network[0]
        receiver = p2p_network[1]
        
        num_messages = 10
        messages = [
            {"sequence": i, "content": f"Ordered message {i}"}
            for i in range(num_messages)
        ]
        
        # Send messages sequentially
        for message in messages:
            await sender.send_message(receiver.node_id, message)
        
        # Check message ordering
        received_sequences = [
            msg["message"]["sequence"] 
            for msg in receiver.message_queue
        ]
        
        expected_sequences = list(range(num_messages))
        assert received_sequences == expected_sequences

    @pytest.mark.asyncio
    async def test_broadcast_message_delivery(self, p2p_network):
        """Test broadcast message delivery to all peers."""
        broadcaster = p2p_network[0]
        receivers = p2p_network[1:]
        
        broadcast_message = {
            "type": "broadcast",
            "content": "Message for all peers",
            "timestamp": time.time()
        }
        
        # Send to all peers
        tasks = [
            broadcaster.send_message(peer.node_id, broadcast_message)
            for peer in receivers
        ]
        results = await asyncio.gather(*tasks)
        
        # Verify delivery to all peers
        successful_deliveries = sum(1 for result in results if result)
        assert successful_deliveries >= len(receivers) * 0.8  # Allow for some failures
        
        # Check each receiver got the message
        for receiver in receivers:
            if receiver.received_messages > 0:
                latest_message = receiver.message_queue[-1]["message"]
                assert latest_message["content"] == "Message for all peers"

    @pytest.mark.asyncio
    async def test_network_statistics_tracking(self, p2p_network):
        """Test network statistics tracking."""
        node = p2p_network[0]
        peer = p2p_network[1]
        
        # Generate some network activity
        for i in range(5):
            await node.send_message(peer.node_id, {"test": i})
        
        stats = node.get_stats()
        
        required_fields = [
            "node_id", "status", "peer_count", "sent_messages",
            "received_messages", "network_failures", "connection_attempts"
        ]
        
        for field in required_fields:
            assert field in stats
        
        assert stats["sent_messages"] == 5
        assert stats["peer_count"] == 4  # 4 other peers in network

    @pytest.mark.asyncio
    async def test_concurrent_peer_communication(self, p2p_network):
        """Test concurrent communication between multiple peer pairs."""
        communication_tasks = []
        
        # Create concurrent communication tasks
        for i in range(len(p2p_network)):
            for j in range(len(p2p_network)):
                if i != j:
                    sender = p2p_network[i]
                    receiver = p2p_network[j]
                    message = {
                        "from": sender.node_id,
                        "to": receiver.node_id,
                        "content": f"Message from {i} to {j}"
                    }
                    
                    task = sender.send_message(receiver.node_id, message)
                    communication_tasks.append(task)
        
        # Execute all communications concurrently
        results = await asyncio.gather(*communication_tasks)
        
        # Calculate overall network performance
        successful_communications = sum(1 for result in results if result)
        total_communications = len(communication_tasks)
        success_rate = successful_communications / total_communications
        
        assert success_rate > 0.7  # At least 70% success rate under load
        
        # Verify message distribution
        total_messages_received = sum(
            node.received_messages for node in p2p_network
        )
        assert total_messages_received >= successful_communications * 0.9


class TestP2PNetworkFailureRecovery:
    """Test P2P network failure recovery mechanisms."""

    @pytest.mark.asyncio
    async def test_node_failure_detection(self):
        """Test detection of failed nodes."""
        # Create network with one node that will "fail"
        healthy_node = MockP2PNode("healthy_node")
        failing_node = MockP2PNode("failing_node")
        
        healthy_node.add_peer(failing_node)
        
        # Simulate node failure by clearing its peer connections
        failing_node.peers.clear()
        failing_node.status = "failed"
        
        # Try to send message to failed node
        message = {"content": "Test message"}
        success = await healthy_node.send_message(failing_node.node_id, message)
        
        assert success is False  # Should fail to send to failed node

    @pytest.mark.asyncio
    async def test_network_resilience_with_node_failures(self):
        """Test network resilience when nodes fail."""
        # Create network
        nodes = []
        for i in range(6):
            node = MockP2PNode(f"node_{i}")
            nodes.append(node)
        
        # Connect in mesh topology
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i != j:
                    node1.add_peer(node2)
        
        # Simulate failure of 2 nodes
        failed_nodes = nodes[4:6]
        active_nodes = nodes[0:4]
        
        for failed_node in failed_nodes:
            failed_node.status = "failed"
            failed_node.peers.clear()
        
        # Test communication between remaining active nodes
        sender = active_nodes[0]
        receiver = active_nodes[1]
        
        message = {"content": "Resilience test"}
        success = await sender.send_message(receiver.node_id, message)
        
        assert success is True  # Network should still function
        assert receiver.received_messages == 1

    @pytest.mark.asyncio
    async def test_automatic_reconnection(self):
        """Test automatic reconnection after network issues."""
        node1 = MockP2PNode("node1")
        node2 = MockP2PNode("node2")
        
        # Initial connection
        node1.add_peer(node2)
        initial_success = await node1.send_message(node2.node_id, {"test": 1})
        assert initial_success is True
        
        # Simulate temporary disconnection
        node1.peers.clear()
        node2.peers.clear()
        
        # Reconnect
        node1.add_peer(node2)
        
        # Test communication after reconnection
        reconnect_success = await node1.send_message(node2.node_id, {"test": 2})
        assert reconnect_success is True
        
        # Verify both messages were received
        assert node2.received_messages == 2


if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--cov=src.p2p",
        "--cov-report=html",
        "--cov-report=term"
    ])