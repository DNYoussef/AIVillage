"""
Test that core functions actually work now
"""
import pytest
from src.twin_runtime.runner import chat
from src.twin_runtime.guard import risk_gate
from src.infrastructure.p2p.device_mesh import discover_network_peers
from src.core.resources.resource_monitor import get_all_metrics


def test_chat_works():
    """Test chat produces real responses"""
    response = chat("Hello, how are you?")
    assert isinstance(response, str) and response != ""
    assert "pass" not in str(response)
    print(f"Chat response: {response}")
    

def test_security_gate_works():
    """Test security gate actually evaluates risk"""
    # Safe message
    safe_result = risk_gate({"content": "Hello world"})
    assert safe_result == "allow"
    
    # Dangerous message
    danger_result = risk_gate({"content": "rm -rf /"})
    assert danger_result == "deny"
    
    # Medium risk
    medium_result = risk_gate({"content": "execute command"})
    assert medium_result in ["ask", "allow"]
    

def test_p2p_discovery_works():
    """Test P2P finds at least localhost"""
    peers = discover_network_peers()
    assert len(peers) > 0
    assert any(p['ip'] in ['127.0.0.1', '192.168.1.1'] for p in peers)
    print(f"Found peers: {peers}")
    

def test_resource_monitoring_works():
    """Test resource monitoring returns real data"""
    metrics = get_all_metrics()
    assert metrics is not None
    assert metrics['cpu_percent'] >= 0
    assert metrics['memory']['total_gb'] > 0
    print(f"System metrics: {metrics}")
