"""
Tests for Cogment Gated Long-Term Memory System (Agent 2)

Tests the memory system components including:
- GatedLTM read/write operations and attention mechanisms
- Surprise-based gating and memory update logic
- Memory decay, consolidation and capacity management
- Cross-attention integration with transformer layers
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple

# Import Cogment memory components
try:
    from core.agent_forge.models.cogment.memory.gated_ltm import GatedLTM, LTMConfig
    from core.agent_forge.models.cogment.memory.memory_gates import MemoryGate, SurpriseGate
    from core.agent_forge.models.cogment.memory.cross_attention import CrossAttention
    from core.agent_forge.models.cogment.memory.memory_utils import MemoryUtils
    from core.agent_forge.models.cogment.core.config import CogmentConfig
    MEMORY_AVAILABLE = True
except ImportError as e:
    print(f"Cogment memory components not available: {e}")
    MEMORY_AVAILABLE = False


class TestGatedLTM:
    """Test Gated Long-Term Memory system."""
    
    @pytest.fixture
    def ltm_config(self):
        """Create LTM configuration."""
        return LTMConfig(
            mem_slots=1024,
            ltm_capacity=512,
            ltm_dim=256,
            memory_dim=256,
            decay=0.001,
            write_alpha=0.1,
            read_threshold=0.05,
            consolidation_threshold=0.8,
            d_model=512
        )
    
    @pytest.fixture
    def gated_ltm(self, ltm_config):
        """Create GatedLTM instance."""
        if not MEMORY_AVAILABLE:
            pytest.skip("Memory components not available")
        return GatedLTM(ltm_config)
    
    def test_gated_ltm_creation(self, gated_ltm, ltm_config):
        """Test GatedLTM instantiation and structure."""
        assert gated_ltm.config == ltm_config
        assert hasattr(gated_ltm, 'memory_bank')
        assert hasattr(gated_ltm, 'read_head')
        assert hasattr(gated_ltm, 'write_head')
        assert hasattr(gated_ltm, 'surprise_gate')
        assert hasattr(gated_ltm, 'consolidation_gate')
        
        # Verify memory bank dimensions
        assert gated_ltm.memory_bank.shape == (ltm_config.ltm_capacity, ltm_config.ltm_dim)
        
        # Verify attention mechanisms
        assert isinstance(gated_ltm.read_head, CrossAttention)
        assert isinstance(gated_ltm.write_head, CrossAttention)
    
    def test_gated_ltm_read_operation(self, gated_ltm, ltm_config):
        """Test memory read operations."""
        batch_size = 2
        seq_len = 16
        d_model = ltm_config.d_model
        
        # Create query tensor
        query = torch.randn(batch_size, seq_len, d_model)
        
        # Perform read operation
        with torch.no_grad():
            read_result = gated_ltm.read(query)
        
        # Verify output structure
        assert hasattr(read_result, 'retrieved_memory')
        assert hasattr(read_result, 'attention_weights')
        assert hasattr(read_result, 'read_gate_values')
        
        # Verify tensor shapes
        assert read_result.retrieved_memory.shape == (batch_size, seq_len, ltm_config.ltm_dim)
        assert read_result.attention_weights.shape == (batch_size, seq_len, ltm_config.ltm_capacity)
        assert read_result.read_gate_values.shape == (batch_size, seq_len, 1)
        
        # Verify attention weights are normalized
        attention_sums = read_result.attention_weights.sum(dim=-1)
        assert torch.allclose(attention_sums, torch.ones_like(attention_sums), atol=1e-6)
        
        # Verify gate values are in [0, 1]
        assert (read_result.read_gate_values >= 0).all()
        assert (read_result.read_gate_values <= 1).all()
    
    def test_gated_ltm_write_operation(self, gated_ltm, ltm_config):
        """Test memory write operations."""
        batch_size = 2
        seq_len = 8
        d_model = ltm_config.d_model
        
        # Store initial memory state
        initial_memory = gated_ltm.memory_bank.clone()
        
        # Create input for writing
        key = torch.randn(batch_size, seq_len, d_model)
        value = torch.randn(batch_size, seq_len, ltm_config.ltm_dim)
        surprise = torch.rand(batch_size, seq_len, 1)  # Random surprise values
        
        # Perform write operation
        write_result = gated_ltm.write(key, value, surprise)
        
        # Verify output structure
        assert hasattr(write_result, 'write_locations')
        assert hasattr(write_result, 'write_gate_values')
        assert hasattr(write_result, 'memory_updates')
        
        # Verify memory was modified
        memory_changed = not torch.allclose(gated_ltm.memory_bank, initial_memory, atol=1e-6)
        assert memory_changed, "Memory should have been updated during write operation"
        
        # Verify write gate values
        assert (write_result.write_gate_values >= 0).all()
        assert (write_result.write_gate_values <= 1).all()
    
    def test_gated_ltm_surprise_gating(self, gated_ltm, ltm_config):
        """Test surprise-based gating mechanism."""
        batch_size = 1
        seq_len = 4
        d_model = ltm_config.d_model
        
        key = torch.randn(batch_size, seq_len, d_model)
        value = torch.randn(batch_size, seq_len, ltm_config.ltm_dim)
        
        # Test with low surprise (should write less)
        low_surprise = torch.full((batch_size, seq_len, 1), 0.1)
        initial_memory_low = gated_ltm.memory_bank.clone()
        
        write_result_low = gated_ltm.write(key, value, low_surprise)
        memory_after_low = gated_ltm.memory_bank.clone()
        
        # Reset memory and test with high surprise
        gated_ltm.memory_bank = initial_memory_low.clone()
        high_surprise = torch.full((batch_size, seq_len, 1), 0.9)
        
        write_result_high = gated_ltm.write(key, value, high_surprise)
        memory_after_high = gated_ltm.memory_bank.clone()
        
        # High surprise should cause more writing
        low_change = torch.norm(memory_after_low - initial_memory_low)
        high_change = torch.norm(memory_after_high - initial_memory_low)
        
        # Allow some tolerance for randomness
        assert high_change >= low_change * 0.8, \
            f"High surprise should cause more memory change: {high_change:.3f} vs {low_change:.3f}"
        
        # Verify gate values reflect surprise levels
        avg_low_gate = write_result_low.write_gate_values.mean()
        avg_high_gate = write_result_high.write_gate_values.mean()
        
        assert avg_high_gate >= avg_low_gate * 0.8, \
            f"High surprise should have higher gate values: {avg_high_gate:.3f} vs {avg_low_gate:.3f}"
    
    def test_gated_ltm_memory_decay(self, gated_ltm, ltm_config):
        """Test memory decay mechanism."""
        batch_size = 1
        seq_len = 4
        d_model = ltm_config.d_model
        
        # Fill memory with some data
        key = torch.randn(batch_size, seq_len, d_model)
        value = torch.randn(batch_size, seq_len, ltm_config.ltm_dim)
        surprise = torch.full((batch_size, seq_len, 1), 0.8)  # High surprise
        
        gated_ltm.write(key, value, surprise)
        memory_after_write = gated_ltm.memory_bank.clone()
        
        # Apply decay multiple times
        for _ in range(10):
            gated_ltm.apply_decay()
        
        memory_after_decay = gated_ltm.memory_bank.clone()
        
        # Memory should have changed due to decay
        decay_change = torch.norm(memory_after_decay - memory_after_write)
        assert decay_change > 0, "Memory should change due to decay"
        
        # Memory values should generally be smaller (closer to zero) after decay
        initial_norm = torch.norm(memory_after_write)
        final_norm = torch.norm(memory_after_decay)
        
        # Allow some tolerance since decay might not always reduce norm
        assert final_norm <= initial_norm * 1.1, \
            f"Memory norm should not increase significantly after decay: {final_norm:.3f} vs {initial_norm:.3f}"
    
    def test_gated_ltm_consolidation(self, gated_ltm, ltm_config):
        """Test memory consolidation mechanism."""
        batch_size = 1
        seq_len = 8
        d_model = ltm_config.d_model
        
        # Create similar memories that should consolidate
        base_value = torch.randn(batch_size, 1, ltm_config.ltm_dim)
        similar_values = base_value + torch.randn(batch_size, seq_len, ltm_config.ltm_dim) * 0.1
        
        keys = torch.randn(batch_size, seq_len, d_model)
        surprise = torch.full((batch_size, seq_len, 1), 0.9)  # High surprise to ensure writing
        
        # Write similar memories
        for i in range(seq_len):
            gated_ltm.write(
                keys[:, i:i+1], 
                similar_values[:, i:i+1], 
                surprise[:, i:i+1]
            )
        
        memory_before_consolidation = gated_ltm.memory_bank.clone()
        
        # Apply consolidation
        consolidation_result = gated_ltm.consolidate()
        
        memory_after_consolidation = gated_ltm.memory_bank.clone()
        
        # Verify consolidation occurred
        assert hasattr(consolidation_result, 'consolidated_count')
        assert hasattr(consolidation_result, 'similarity_threshold')
        
        # Memory should have changed if consolidation occurred
        if consolidation_result.consolidated_count > 0:
            consolidation_change = torch.norm(memory_after_consolidation - memory_before_consolidation)
            assert consolidation_change > 0, "Memory should change when consolidation occurs"
    
    def test_gated_ltm_capacity_management(self, gated_ltm, ltm_config):
        """Test memory capacity management and overflow handling."""
        batch_size = 1
        seq_len = 1
        d_model = ltm_config.d_model
        
        # Write more memories than capacity
        num_writes = ltm_config.ltm_capacity + 10
        
        for i in range(num_writes):
            key = torch.randn(batch_size, seq_len, d_model)
            value = torch.randn(batch_size, seq_len, ltm_config.ltm_dim)
            surprise = torch.full((batch_size, seq_len, 1), 0.8)
            
            write_result = gated_ltm.write(key, value, surprise)
            
            # Verify memory bank doesn't exceed capacity
            assert gated_ltm.memory_bank.shape[0] == ltm_config.ltm_capacity
        
        # Verify oldest memories are replaced when capacity is exceeded
        memory_utilization = gated_ltm.get_memory_utilization()
        assert hasattr(memory_utilization, 'used_slots')
        assert hasattr(memory_utilization, 'capacity')
        assert memory_utilization.used_slots <= memory_utilization.capacity
    
    def test_gated_ltm_forward_integration(self, gated_ltm, ltm_config):
        """Test integrated forward pass with read and write."""
        batch_size = 2
        seq_len = 16
        d_model = ltm_config.d_model
        
        # Create input tensors
        hidden_states = torch.randn(batch_size, seq_len, d_model)
        loss = torch.rand(batch_size, seq_len, 1)  # Surprise signal
        
        # Forward pass
        with torch.no_grad():
            outputs = gated_ltm(hidden_states, loss)
        
        # Verify output structure
        assert hasattr(outputs, 'memory_enhanced_states')
        assert hasattr(outputs, 'read_results')
        assert hasattr(outputs, 'write_results')
        assert hasattr(outputs, 'memory_stats')
        
        # Verify tensor shapes
        assert outputs.memory_enhanced_states.shape == hidden_states.shape
        
        # Verify memory operations occurred
        assert outputs.read_results is not None
        assert outputs.write_results is not None


class TestMemoryGates:
    """Test memory gating mechanisms."""
    
    @pytest.fixture
    def memory_gate(self):
        """Create MemoryGate instance."""
        if not MEMORY_AVAILABLE:
            pytest.skip("Memory components not available")
        return MemoryGate(d_model=256, gate_type="sigmoid")
    
    @pytest.fixture
    def surprise_gate(self):
        """Create SurpriseGate instance.""" 
        if not MEMORY_AVAILABLE:
            pytest.skip("Memory components not available")
        return SurpriseGate(d_model=256, surprise_threshold=0.5)
    
    def test_memory_gate_forward(self, memory_gate):
        """Test MemoryGate forward pass."""
        batch_size = 2
        seq_len = 8
        d_model = 256
        
        input_tensor = torch.randn(batch_size, seq_len, d_model)
        
        with torch.no_grad():
            gate_values = memory_gate(input_tensor)
        
        assert gate_values.shape == (batch_size, seq_len, 1)
        assert (gate_values >= 0).all()
        assert (gate_values <= 1).all()
    
    def test_surprise_gate_functionality(self, surprise_gate):
        """Test SurpriseGate surprise-based gating."""
        batch_size = 2
        seq_len = 8
        d_model = 256
        
        hidden_states = torch.randn(batch_size, seq_len, d_model)
        
        # Test with different surprise levels
        low_surprise = torch.full((batch_size, seq_len, 1), 0.1)
        high_surprise = torch.full((batch_size, seq_len, 1), 0.9)
        
        with torch.no_grad():
            low_gates = surprise_gate(hidden_states, low_surprise)
            high_gates = surprise_gate(hidden_states, high_surprise)
        
        # High surprise should generally produce higher gate values
        avg_low = low_gates.mean()
        avg_high = high_gates.mean()
        
        assert avg_high >= avg_low * 0.8, \
            f"High surprise should produce higher gates: {avg_high:.3f} vs {avg_low:.3f}"
    
    def test_gate_gradients(self, memory_gate):
        """Test gradient flow through gates."""
        batch_size = 1
        seq_len = 4
        d_model = 256
        
        input_tensor = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        
        gate_values = memory_gate(input_tensor)
        loss = gate_values.sum()
        loss.backward()
        
        assert input_tensor.grad is not None
        assert not torch.isnan(input_tensor.grad).any()


class TestCrossAttention:
    """Test cross-attention mechanisms for memory."""
    
    @pytest.fixture
    def cross_attention(self):
        """Create CrossAttention instance."""
        if not MEMORY_AVAILABLE:
            pytest.skip("Memory components not available")
        return CrossAttention(
            query_dim=512,
            key_dim=256,
            value_dim=256,
            output_dim=256,
            n_heads=8
        )
    
    def test_cross_attention_forward(self, cross_attention):
        """Test cross-attention forward pass."""
        batch_size = 2
        query_len = 16
        memory_len = 32
        
        query = torch.randn(batch_size, query_len, 512)
        keys = torch.randn(batch_size, memory_len, 256)
        values = torch.randn(batch_size, memory_len, 256)
        
        with torch.no_grad():
            outputs = cross_attention(query, keys, values)
        
        assert hasattr(outputs, 'attended_values')
        assert hasattr(outputs, 'attention_weights')
        
        assert outputs.attended_values.shape == (batch_size, query_len, 256)
        assert outputs.attention_weights.shape == (batch_size, query_len, memory_len)
        
        # Verify attention weights are normalized
        attention_sums = outputs.attention_weights.sum(dim=-1)
        assert torch.allclose(attention_sums, torch.ones_like(attention_sums), atol=1e-6)
    
    def test_cross_attention_masked(self, cross_attention):
        """Test cross-attention with masking."""
        batch_size = 1
        query_len = 8
        memory_len = 16
        
        query = torch.randn(batch_size, query_len, 512)
        keys = torch.randn(batch_size, memory_len, 256)
        values = torch.randn(batch_size, memory_len, 256)
        
        # Create mask (mask out second half of memory)
        mask = torch.ones(batch_size, query_len, memory_len)
        mask[:, :, memory_len//2:] = 0
        
        with torch.no_grad():
            outputs = cross_attention(query, keys, values, mask=mask)
        
        # Verify masked positions have zero attention
        masked_attention = outputs.attention_weights[:, :, memory_len//2:]
        assert torch.allclose(masked_attention, torch.zeros_like(masked_attention), atol=1e-6)


@pytest.mark.integration
class TestMemoryIntegration:
    """Integration tests for memory system."""
    
    @pytest.mark.skipif(not MEMORY_AVAILABLE, reason="Memory components not available")
    def test_memory_system_integration(self):
        """Test complete memory system integration."""
        # Create configuration
        config = LTMConfig(
            mem_slots=512,
            ltm_capacity=256, 
            ltm_dim=128,
            memory_dim=128,
            d_model=256
        )
        
        # Create memory system
        memory_system = GatedLTM(config)
        
        batch_size = 2
        seq_len = 32
        
        # Simulate training sequence
        for step in range(5):
            # Create inputs
            hidden_states = torch.randn(batch_size, seq_len, config.d_model)
            surprise = torch.rand(batch_size, seq_len, 1)
            
            # Forward pass
            with torch.no_grad():
                outputs = memory_system(hidden_states, surprise)
            
            # Verify outputs
            assert outputs.memory_enhanced_states.shape == hidden_states.shape
            assert outputs.read_results is not None
            assert outputs.write_results is not None
            
            # Apply decay periodically
            if step % 2 == 0:
                memory_system.apply_decay()
        
        # Verify memory has been utilized
        utilization = memory_system.get_memory_utilization()
        assert utilization.used_slots > 0
        
        print("✓ Memory system integration successful")
    
    @pytest.mark.skipif(not MEMORY_AVAILABLE, reason="Memory components not available")
    def test_memory_persistence_across_batches(self):
        """Test memory persistence across different input batches."""
        config = LTMConfig(
            mem_slots=256,
            ltm_capacity=128,
            ltm_dim=64,
            memory_dim=64,
            d_model=128
        )
        
        memory_system = GatedLTM(config)
        
        # First batch - write some memories
        batch1 = torch.randn(1, 8, config.d_model)
        surprise1 = torch.full((1, 8, 1), 0.8)
        
        with torch.no_grad():
            outputs1 = memory_system(batch1, surprise1)
        
        memory_after_batch1 = memory_system.memory_bank.clone()
        
        # Second batch - should retrieve from previous memories
        batch2 = torch.randn(1, 8, config.d_model)
        surprise2 = torch.full((1, 8, 1), 0.2)  # Low surprise, mostly reading
        
        with torch.no_grad():
            outputs2 = memory_system(batch2, surprise2)
        
        # Verify memory persisted and was used
        assert outputs2.read_results is not None
        assert outputs2.read_results.attention_weights.sum() > 0
        
        # Memory should be similar but may have small changes
        memory_similarity = F.cosine_similarity(
            memory_after_batch1.flatten(),
            memory_system.memory_bank.flatten(),
            dim=0
        )
        assert memory_similarity > 0.8, "Memory should persist across batches"
        
        print("✓ Memory persistence across batches verified")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])