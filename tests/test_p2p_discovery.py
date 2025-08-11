"""Test P2P Discovery functionality"""

import json
import time

from src.infrastructure.p2p.device_mesh import DeviceMesh, discover_network_peers

print("=== Testing P2P Discovery ===")

# Test 1: Basic discovery
print("\n1. Basic Network Discovery:")
peers = discover_network_peers()
print(f"   Found {len(peers)} peers")
for i, peer in enumerate(peers):
    print(f"   Peer {i + 1}: {peer.get('hostname')} at {peer.get('ip')}:{peer.get('port')}")
    print(
        f"           Platform: {peer.get('platform')}, CPUs: {peer.get('cpu_count')}, Memory: {peer.get('memory_gb', 0):.1f} GB"
    )

# Test 2: Discovery service
print("\n2. Starting Discovery Service:")
mesh = DeviceMesh(port=8765)
mesh.start_discovery_service()
print("   Service started on port 8765")
time.sleep(1)

# Test 3: Peer persistence
print("\n3. Testing Peer Persistence:")
peers_file = mesh.peers_file
print(f"   Peers file location: {peers_file}")
print(f"   Peers file exists: {peers_file.exists()}")
if peers_file.exists():
    saved_peers = json.loads(peers_file.read_text())
    print(f"   Saved peers count: {len(saved_peers)}")

# Test 4: Two instances test
print("\n4. Testing Two Instances Discovery:")
mesh2 = DeviceMesh(port=8766)
mesh2.start_discovery_service()
print("   Started second instance on port 8766")
time.sleep(1)

# Discover from second instance
discovered = mesh2.discover_network_peers()
print(f"   Second instance found {len(discovered)} peers")

# Test 5: Connection testing
print("\n5. Testing Peer Connection:")
if discovered:
    test_peer = discovered[0]
    peer_addr = f"{test_peer.get('ip')}:{test_peer.get('port')}"
    print(f"   Testing connection to {peer_addr}")
    connected = mesh2.connect_to_peer(peer_addr)
    print(f"   Connection successful: {connected}")

# Test 6: Local info
print("\n6. Local Device Information:")
local_info = mesh.local_info
print(f"   Hostname: {local_info['hostname']}")
print(f"   IP: {local_info['ip']}")
print(f"   Port: {local_info['port']}")
print(f"   Platform: {local_info['platform']}")
print(f"   CPUs: {local_info['cpu_count']}")
print(f"   Memory: {local_info['memory_gb']:.1f} GB")

# Test 7: Health checking simulation
print("\n7. Testing Health Check (simulated):")
print("   Adding fake peer for health check test")
mesh.peers["192.168.1.99:8765"] = {
    "hostname": "fake-peer",
    "ip": "192.168.1.99",
    "port": 8765,
}
mesh.peer_failures["192.168.1.99:8765"] = 2  # Already failed twice
print(f"   Current peers: {list(mesh.peers.keys())}")
print("   Simulating health check failure...")
if not mesh.connect_to_peer("192.168.1.99:8765"):
    mesh.peer_failures["192.168.1.99:8765"] += 1
    if mesh.peer_failures["192.168.1.99:8765"] >= 3:
        mesh.peers.pop("192.168.1.99:8765", None)
        print("   Dead peer removed after 3 failures")
print(f"   Remaining peers: {list(mesh.peers.keys())}")

# Cleanup
print("\n8. Cleanup:")
mesh.stop()
mesh2.stop()
print("   Discovery services stopped")

print("\nP2P Discovery Test Complete!")
