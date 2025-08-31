#!/usr/bin/env python3
"""
Integration test for import dependencies and P2P infrastructure.
Validates that all critical modules can be imported successfully.
"""

import pytest
import sys
import importlib
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestImportValidation:
    """Test suite for import validation across the codebase."""
    
    def test_p2p_core_message_types(self):
        """Test that core message types can be imported."""
        from infrastructure.p2p.core.message_types import (
            MessageType, MessagePriority, MessageMetadata, UnifiedMessage, Message
        )
        
        # Test enum values
        assert MessageType.HEARTBEAT.value == "heartbeat"
        assert MessagePriority.HIGH.value == 2
        
        # Test MessageMetadata creation
        metadata = MessageMetadata(
            sender_id="test_sender",
            receiver_id="test_receiver", 
            timestamp=1234567890,
            message_id="test_id"
        )
        assert metadata.sender_id == "test_sender"
        assert metadata.ttl == 10  # default value
        
        # Test UnifiedMessage creation
        unified_msg = UnifiedMessage(
            metadata=metadata,
            content={"test": "data"},
            priority=MessagePriority.HIGH
        )
        assert unified_msg.priority == MessagePriority.HIGH
        assert unified_msg.content == {"test": "data"}
    
    def test_p2p_advanced_modules(self):
        """Test that P2P advanced modules can be imported."""
        from infrastructure.p2p.advanced.libp2p_integration_api import LibP2PIntegrationAPI
        from infrastructure.p2p.advanced.libp2p_enhanced_manager import LibP2PEnhancedManager
        from infrastructure.p2p.advanced.nat_traversal_optimizer import NATTraversalOptimizer
        from infrastructure.p2p.advanced.protocol_multiplexer import ProtocolMultiplexer
        
        # These should not raise import errors
        assert LibP2PIntegrationAPI is not None
        assert LibP2PEnhancedManager is not None
        assert NATTraversalOptimizer is not None
        assert ProtocolMultiplexer is not None
    
    def test_p2p_communications(self):
        """Test P2P communications modules."""
        from infrastructure.p2p.communications.credits_ledger import User, CreditsLedger
        
        # These should not raise import errors
        assert User is not None
        assert CreditsLedger is not None
    
    def test_rag_integration_bridges(self):
        """Test RAG integration bridge modules."""
        from core.rag.integration.fog_compute_bridge import FogComputeBridge
        from core.rag.integration.edge_device_bridge import EdgeDeviceRAGBridge
        from core.rag.integration.p2p_network_bridge import P2PNetworkRAGBridge
        
        # Test instantiation
        fog_bridge = FogComputeBridge()
        assert fog_bridge.initialized is False
        
        # These should not raise import errors  
        assert EdgeDeviceRAGBridge is not None
        assert P2PNetworkRAGBridge is not None
    
    def test_typing_imports_consistency(self):
        """Test that typing imports are consistent across modules."""
        # Import modules that use extensive typing
        from infrastructure.p2p.advanced.libp2p_enhanced_manager import LibP2PEnhancedManager
        from infrastructure.p2p.advanced.protocol_multiplexer import ProtocolMultiplexer
        
        # These should have proper type annotations (no import errors)
        import inspect
        
        # Check that classes have proper annotations
        sig = inspect.signature(LibP2PEnhancedManager.__init__)
        assert sig is not None
        
        sig = inspect.signature(ProtocolMultiplexer.__init__)
        assert sig is not None
    
    def test_no_circular_imports(self):
        """Test that no circular imports exist in critical modules."""
        # Import all critical modules in sequence to detect circular imports
        modules_to_test = [
            "infrastructure.p2p.core.message_types",
            "infrastructure.p2p.advanced.libp2p_integration_api", 
            "infrastructure.p2p.communications.credits_ledger",
            "core.rag.integration.fog_compute_bridge",
            "core.rag.integration.p2p_network_bridge",
        ]
        
        for module_name in modules_to_test:
            try:
                module = importlib.import_module(module_name)
                assert module is not None, f"Failed to import {module_name}"
            except ImportError as e:
                pytest.fail(f"Circular import detected in {module_name}: {e}")
    
    def test_import_performance(self):
        """Test that imports don't take excessive time."""
        import time
        
        start_time = time.time()
        
        # Import several heavy modules
        from infrastructure.p2p.advanced.libp2p_integration_api import LibP2PIntegrationAPI
        from infrastructure.p2p.advanced.libp2p_enhanced_manager import LibP2PEnhancedManager
        from core.rag.integration.fog_compute_bridge import FogComputeBridge
        
        end_time = time.time()
        import_time = end_time - start_time
        
        # Imports should complete within reasonable time (5 seconds)
        assert import_time < 5.0, f"Imports took too long: {import_time:.2f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])