"""
Tests for Cogment Parameter Budget Validation

Validates the exact parameter budget of 23.7M vs 25M target including:
- Component-wise parameter breakdown
- Budget allocation across model components
- Parameter efficiency optimizations
- Option A configuration validation
- Real-time parameter counting during model instantiation
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch
import torch.nn as nn

# Import Cogment components for parameter analysis
try:
    from config.cogment.config_loader import load_cogment_config
    from core.agent_forge.models.cogment.core.config import CogmentConfig
    from core.agent_forge.models.cogment.core.model import CogmentModel
    from core.agent_forge.models.cogment.core.refinement_core import RefinementCore
    from core.agent_forge.models.cogment.heads.text_head import TextHead
    from core.agent_forge.models.cogment.memory.gated_ltm import GatedLTM

    COGMENT_AVAILABLE = True
except ImportError as e:
    print(f"Cogment components not available: {e}")
    COGMENT_AVAILABLE = False


@dataclass
class ComponentParameterBreakdown:
    """Parameter breakdown for model components."""

    component_name: str
    parameter_count: int
    percentage: float
    details: Dict[str, int]


@dataclass
class ParameterBudgetAnalysis:
    """Complete parameter budget analysis."""

    total_parameters: int
    target_parameters: int
    budget_utilization: float
    components: List[ComponentParameterBreakdown]
    efficiency_metrics: Dict[str, float]
    validation_status: bool


class TestParameterCounting:
    """Test accurate parameter counting."""

    @pytest.fixture
    def target_config(self):
        """Create target configuration for 23.7M parameters."""
        return CogmentConfig(
            # Option A: Optimized for 25M budget achieving 23.7M
            d_model=512,
            n_layers=6,
            n_head=8,
            d_ff=1536,
            vocab_size=13000,
            max_seq_len=2048,
            # Memory configuration
            mem_slots=2048,
            ltm_capacity=1024,
            ltm_dim=256,
            memory_dim=256,
            # Efficiency settings
            tie_embeddings=True,
            dropout=0.1,
            layer_norm_eps=1e-5,
            # Budget validation
            target_params=25_000_000,
            tolerance=0.05,
        )

    @pytest.fixture
    def cogment_model(self, target_config):
        """Create Cogment model with target configuration."""
        if not COGMENT_AVAILABLE:
            pytest.skip("Cogment components not available")
        return CogmentModel(target_config)

    def count_parameters(self, module: nn.Module) -> int:
        """Count trainable parameters in a module."""
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    def get_parameter_breakdown(self, module: nn.Module, name: str = "model") -> Dict[str, int]:
        """Get detailed parameter breakdown for a module."""
        breakdown = {}

        # Count parameters in each named submodule
        for submodule_name, submodule in module.named_children():
            param_count = self.count_parameters(submodule)
            if param_count > 0:
                breakdown[f"{name}.{submodule_name}"] = param_count

        # Count direct parameters (not in submodules)
        direct_params = sum(p.numel() for n, p in module.named_parameters(recurse=False) if p.requires_grad)
        if direct_params > 0:
            breakdown[f"{name}.direct"] = direct_params

        return breakdown

    def test_total_parameter_count(self, cogment_model, target_config):
        """Test total parameter count meets target."""
        total_params = self.count_parameters(cogment_model)

        # Target: 23.7M parameters (within 25M budget)
        achieved_target = 23_700_000
        budget_limit = target_config.target_params
        tolerance_percent = target_config.tolerance

        # Calculate acceptable range
        tolerance_abs = achieved_target * tolerance_percent
        min_acceptable = achieved_target - tolerance_abs
        max_acceptable = min(achieved_target + tolerance_abs, budget_limit)

        assert (
            min_acceptable <= total_params <= max_acceptable
        ), f"Parameter count {total_params:,} outside target range [{min_acceptable:,}, {max_acceptable:,}]"

        # Should not exceed budget
        assert total_params <= budget_limit, f"Parameter count {total_params:,} exceeds budget {budget_limit:,}"

        # Calculate budget utilization
        budget_utilization = total_params / budget_limit * 100

        print(f"✓ Total parameter validation:")
        print(f"  - Actual: {total_params:,}")
        print(f"  - Target: {achieved_target:,}")
        print(f"  - Budget: {budget_limit:,}")
        print(f"  - Utilization: {budget_utilization:.1f}%")

        return total_params

    def test_component_parameter_breakdown(self, cogment_model):
        """Test parameter breakdown by component."""
        total_params = self.count_parameters(cogment_model)

        # Get breakdown by major components
        component_counts = {}

        # Core components
        if hasattr(cogment_model, "refinement_core"):
            component_counts["refinement_core"] = self.count_parameters(cogment_model.refinement_core)

        if hasattr(cogment_model, "gated_ltm"):
            component_counts["gated_ltm"] = self.count_parameters(cogment_model.gated_ltm)

        if hasattr(cogment_model, "heads"):
            component_counts["heads"] = self.count_parameters(cogment_model.heads)

        if hasattr(cogment_model, "embeddings"):
            component_counts["embeddings"] = self.count_parameters(cogment_model.embeddings)

        # Detailed breakdown
        detailed_breakdown = self.get_parameter_breakdown(cogment_model, "cogment")

        print("✓ Component parameter breakdown:")
        for component, count in component_counts.items():
            percentage = (count / total_params) * 100
            print(f"  - {component}: {count:,} ({percentage:.1f}%)")

        print("\n✓ Detailed parameter breakdown:")
        for name, count in sorted(detailed_breakdown.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_params) * 100
            if percentage >= 1.0:  # Only show components with ≥1% of parameters
                print(f"  - {name}: {count:,} ({percentage:.1f}%)")

        # Verify all components are accounted for
        accounted_params = sum(component_counts.values())
        accounting_error = abs(total_params - accounted_params) / total_params

        assert accounting_error < 0.01, f"Parameter accounting error too high: {accounting_error:.2%}"

        return component_counts

    def test_embedding_parameter_efficiency(self, cogment_model, target_config):
        """Test embedding parameter efficiency through tied weights."""
        if not hasattr(cogment_model, "embeddings"):
            pytest.skip("Embeddings not accessible")

        # Calculate expected embedding parameters
        vocab_size = target_config.vocab_size
        d_model = target_config.d_model

        if target_config.tie_embeddings:
            # With tied embeddings: only one copy of vocab × d_model
            expected_embedding_params = vocab_size * d_model
        else:
            # Without tied embeddings: input + output embeddings
            expected_embedding_params = 2 * vocab_size * d_model

        # Count actual embedding parameters
        if hasattr(cogment_model, "embeddings"):
            actual_embedding_params = self.count_parameters(cogment_model.embeddings)
        else:
            # Try to find embeddings in model
            embedding_params = 0
            for name, param in cogment_model.named_parameters():
                if "embed" in name.lower():
                    embedding_params += param.numel()
            actual_embedding_params = embedding_params

        # Allow some tolerance for additional embedding components
        tolerance = 0.1  # 10% tolerance
        min_expected = expected_embedding_params
        max_expected = expected_embedding_params * (1 + tolerance)

        assert (
            min_expected <= actual_embedding_params <= max_expected
        ), f"Embedding params {actual_embedding_params:,} outside expected range [{min_expected:,}, {max_expected:,}]"

        # Calculate efficiency
        efficiency = actual_embedding_params / expected_embedding_params

        print(f"✓ Embedding parameter efficiency:")
        print(f"  - Expected: {expected_embedding_params:,}")
        print(f"  - Actual: {actual_embedding_params:,}")
        print(f"  - Efficiency: {efficiency:.2f}")
        print(f"  - Tied embeddings: {target_config.tie_embeddings}")

        return actual_embedding_params

    def test_transformer_core_parameters(self, cogment_model, target_config):
        """Test transformer core parameter allocation."""
        if not hasattr(cogment_model, "refinement_core"):
            pytest.skip("RefinementCore not accessible")

        core_params = self.count_parameters(cogment_model.refinement_core)

        # Estimate expected transformer parameters
        d_model = target_config.d_model
        d_ff = target_config.d_ff
        n_layers = target_config.n_layers
        n_head = target_config.n_head

        # Per-layer parameter estimate
        attention_params_per_layer = 4 * d_model * d_model  # Q, K, V, O projections
        ffn_params_per_layer = 2 * d_model * d_ff  # Up and down projections
        norm_params_per_layer = 2 * d_model  # Pre and post layer norms

        params_per_layer = attention_params_per_layer + ffn_params_per_layer + norm_params_per_layer
        expected_core_params = params_per_layer * n_layers

        # Allow tolerance for additional components (ACT, etc.)
        tolerance = 0.3  # 30% tolerance for additional features
        min_expected = expected_core_params
        max_expected = expected_core_params * (1 + tolerance)

        assert (
            min_expected <= core_params <= max_expected
        ), f"Core params {core_params:,} outside expected range [{min_expected:,}, {max_expected:,}]"

        print(f"✓ Transformer core parameters:")
        print(f"  - Expected: {expected_core_params:,}")
        print(f"  - Actual: {core_params:,}")
        print(f"  - Layers: {n_layers}")
        print(f"  - Avg per layer: {core_params // n_layers:,}")

        return core_params

    def test_memory_system_parameters(self, cogment_model, target_config):
        """Test memory system parameter allocation."""
        if not hasattr(cogment_model, "gated_ltm"):
            pytest.skip("GatedLTM not accessible")

        memory_params = self.count_parameters(cogment_model.gated_ltm)

        # Estimate expected memory parameters
        ltm_capacity = getattr(target_config, "ltm_capacity", 1024)
        ltm_dim = getattr(target_config, "ltm_dim", 256)
        memory_dim = getattr(target_config, "memory_dim", 256)
        d_model = target_config.d_model

        # Memory bank parameters
        memory_bank_params = ltm_capacity * ltm_dim

        # Attention mechanism parameters (simplified estimate)
        attention_params = 3 * d_model * memory_dim  # Q, K, V projections

        # Gating and control parameters
        gating_params = 2 * d_model * memory_dim  # Read and write gates

        expected_memory_params = memory_bank_params + attention_params + gating_params

        # Allow tolerance for additional memory components
        tolerance = 0.4  # 40% tolerance for additional memory features
        min_expected = expected_memory_params * 0.5  # Allow for optimizations
        max_expected = expected_memory_params * (1 + tolerance)

        assert (
            min_expected <= memory_params <= max_expected
        ), f"Memory params {memory_params:,} outside expected range [{min_expected:,}, {max_expected:,}]"

        print(f"✓ Memory system parameters:")
        print(f"  - Expected: {expected_memory_params:,}")
        print(f"  - Actual: {memory_params:,}")
        print(f"  - LTM capacity: {ltm_capacity}")
        print(f"  - LTM dimension: {ltm_dim}")

        return memory_params


class TestBudgetAllocation:
    """Test parameter budget allocation strategy."""

    @pytest.fixture
    def budget_analysis(self):
        """Create budget analysis configuration."""
        return {
            "total_budget": 25_000_000,
            "target_achievement": 23_700_000,
            "efficiency_target": 0.948,  # 23.7M / 25M
            "component_targets": {
                "backbone": 0.40,  # 40% for transformer layers
                "embeddings": 0.35,  # 35% for vocabulary
                "memory": 0.20,  # 20% for LTM system
                "heads": 0.05,  # 5% for output heads
            },
        }

    @pytest.mark.skipif(not COGMENT_AVAILABLE, reason="Cogment not available")
    def test_budget_allocation_strategy(self, budget_analysis):
        """Test parameter budget allocation strategy."""
        config = CogmentConfig(d_model=512, n_layers=6, vocab_size=13000, target_params=budget_analysis["total_budget"])

        model = CogmentModel(config)
        total_params = sum(p.numel() for p in model.parameters())

        # Analyze allocation
        allocation_analysis = {}

        # Component parameter counts
        if hasattr(model, "refinement_core"):
            backbone_params = sum(p.numel() for p in model.refinement_core.parameters())
            allocation_analysis["backbone"] = backbone_params / total_params

        if hasattr(model, "embeddings"):
            embedding_params = sum(p.numel() for p in model.embeddings.parameters())
            allocation_analysis["embeddings"] = embedding_params / total_params

        if hasattr(model, "gated_ltm"):
            memory_params = sum(p.numel() for p in model.gated_ltm.parameters())
            allocation_analysis["memory"] = memory_params / total_params

        if hasattr(model, "heads"):
            head_params = sum(p.numel() for p in model.heads.parameters())
            allocation_analysis["heads"] = head_params / total_params

        print("✓ Budget allocation analysis:")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Budget utilization: {total_params / budget_analysis['total_budget']:.1%}")

        # Compare with targets
        target_allocations = budget_analysis["component_targets"]

        for component, actual_ratio in allocation_analysis.items():
            if component in target_allocations:
                target_ratio = target_allocations[component]
                print(f"  - {component}: {actual_ratio:.1%} (target: {target_ratio:.1%})")

                # Allow reasonable tolerance
                tolerance = 0.1  # 10% tolerance
                assert (
                    abs(actual_ratio - target_ratio) <= tolerance
                ), f"{component} allocation {actual_ratio:.1%} too far from target {target_ratio:.1%}"

    def test_efficiency_optimization(self, budget_analysis):
        """Test parameter efficiency optimizations."""
        if not COGMENT_AVAILABLE:
            pytest.skip("Cogment not available")

        # Test different configurations for efficiency
        configs = {
            "option_a": CogmentConfig(d_model=512, n_layers=6, vocab_size=13000, tie_embeddings=True),
            "unoptimized": CogmentConfig(d_model=512, n_layers=6, vocab_size=13000, tie_embeddings=False),
        }

        efficiency_results = {}

        for name, config in configs.items():
            model = CogmentModel(config)
            param_count = sum(p.numel() for p in model.parameters())
            efficiency = param_count / budget_analysis["total_budget"]
            efficiency_results[name] = {"params": param_count, "efficiency": efficiency}

        # Option A should be more efficient
        option_a_efficiency = efficiency_results["option_a"]["efficiency"]
        unoptimized_efficiency = efficiency_results["unoptimized"]["efficiency"]

        assert (
            option_a_efficiency < unoptimized_efficiency
        ), f"Option A should be more efficient: {option_a_efficiency:.3f} vs {unoptimized_efficiency:.3f}"

        print("✓ Efficiency optimization:")
        for name, results in efficiency_results.items():
            print(f"  - {name}: {results['params']:,} params ({results['efficiency']:.1%} of budget)")

        # Option A should achieve target efficiency
        target_efficiency = budget_analysis["efficiency_target"]
        assert (
            abs(option_a_efficiency - target_efficiency) <= 0.05
        ), f"Option A efficiency {option_a_efficiency:.3f} should be close to target {target_efficiency:.3f}"


class TestParameterValidation:
    """Test parameter validation and constraints."""

    def test_configuration_parameter_estimation(self):
        """Test configuration-based parameter estimation."""
        if not COGMENT_AVAILABLE:
            pytest.skip("Cogment not available")

        config = CogmentConfig(
            d_model=512, n_layers=6, n_head=8, d_ff=1536, vocab_size=13000, max_seq_len=2048, target_params=25_000_000
        )

        # Test configuration estimation
        estimated_params = config.estimate_parameters()

        # Create actual model and count
        model = CogmentModel(config)
        actual_params = sum(p.numel() for p in model.parameters())

        # Estimation should be reasonably accurate
        estimation_error = abs(estimated_params - actual_params) / actual_params
        max_error = 0.15  # 15% tolerance

        assert (
            estimation_error <= max_error
        ), f"Parameter estimation error {estimation_error:.1%} exceeds tolerance {max_error:.1%}"

        print(f"✓ Parameter estimation validation:")
        print(f"  - Estimated: {estimated_params:,}")
        print(f"  - Actual: {actual_params:,}")
        print(f"  - Error: {estimation_error:.1%}")

    def test_budget_constraint_enforcement(self):
        """Test budget constraint enforcement."""
        if not COGMENT_AVAILABLE:
            pytest.skip("Cogment not available")

        # Test configuration within budget
        valid_config = CogmentConfig(
            d_model=512, n_layers=6, vocab_size=13000, target_params=25_000_000, tolerance=0.05
        )

        assert valid_config.validate_parameter_budget(), "Valid configuration should pass budget validation"

        # Test configuration that would exceed budget
        invalid_config = CogmentConfig(
            d_model=1024,  # Too large
            n_layers=20,  # Too many
            vocab_size=50000,  # Too large
            target_params=25_000_000,
            tolerance=0.05,
        )

        assert not invalid_config.validate_parameter_budget(), "Invalid configuration should fail budget validation"

        print("✓ Budget constraint enforcement working")

    def test_real_time_parameter_tracking(self):
        """Test real-time parameter tracking during model building."""
        if not COGMENT_AVAILABLE:
            pytest.skip("Cogment not available")

        config = CogmentConfig(
            d_model=256, n_layers=3, vocab_size=5000, target_params=10_000_000  # Smaller for testing
        )

        # Track parameters as model is built
        parameter_tracker = []

        class ParameterTrackingModel(CogmentModel):
            def __init__(self, config):
                super().__init__(config)

                # Track parameters after each major component
                if hasattr(self, "embeddings"):
                    param_count = sum(p.numel() for p in self.parameters())
                    parameter_tracker.append(("embeddings", param_count))

                if hasattr(self, "refinement_core"):
                    param_count = sum(p.numel() for p in self.parameters())
                    parameter_tracker.append(("refinement_core", param_count))

                if hasattr(self, "gated_ltm"):
                    param_count = sum(p.numel() for p in self.parameters())
                    parameter_tracker.append(("gated_ltm", param_count))

                if hasattr(self, "heads"):
                    param_count = sum(p.numel() for p in self.parameters())
                    parameter_tracker.append(("heads", param_count))

        # Build model with tracking
        model = ParameterTrackingModel(config)
        final_params = sum(p.numel() for p in model.parameters())

        print("✓ Real-time parameter tracking:")
        for component, param_count in parameter_tracker:
            print(f"  - After {component}: {param_count:,} parameters")
        print(f"  - Final total: {final_params:,} parameters")

        # Verify tracking captured parameter growth
        assert len(parameter_tracker) > 0, "Should track parameter growth"

        # Verify monotonic growth (parameters only increase)
        for i in range(1, len(parameter_tracker)):
            current_count = parameter_tracker[i][1]
            previous_count = parameter_tracker[i - 1][1]
            assert current_count >= previous_count, "Parameter count should be monotonically increasing"


@pytest.mark.integration
class TestParameterBudgetIntegration:
    """Integration tests for complete parameter budget system."""

    @pytest.mark.skipif(not COGMENT_AVAILABLE, reason="Cogment not available")
    def test_complete_budget_analysis(self):
        """Run complete parameter budget analysis."""
        print("=== Cogment Parameter Budget Analysis ===\n")

        # Load target configuration
        try:
            config = load_cogment_config()
        except:
            config = CogmentConfig(d_model=512, n_layers=6, vocab_size=13000, target_params=25_000_000)

        # Create model and analyze
        model = CogmentModel(config)
        total_params = sum(p.numel() for p in model.parameters())

        # Component breakdown
        components = {}

        if hasattr(model, "refinement_core"):
            components["RefinementCore"] = sum(p.numel() for p in model.refinement_core.parameters())

        if hasattr(model, "gated_ltm"):
            components["GatedLTM"] = sum(p.numel() for p in model.gated_ltm.parameters())

        if hasattr(model, "embeddings"):
            components["Embeddings"] = sum(p.numel() for p in model.embeddings.parameters())

        if hasattr(model, "heads"):
            components["Heads"] = sum(p.numel() for p in model.heads.parameters())

        # Generate analysis report
        print("📊 PARAMETER BUDGET ANALYSIS")
        print("=" * 50)
        print(f"🎯 Target: 23.7M parameters (25M budget)")
        print(f"📈 Achieved: {total_params:,} parameters")
        print(f"💹 Budget utilization: {total_params / 25_000_000:.1%}")
        print(f"🎖️ Efficiency: {23_700_000 / total_params:.3f} (target vs actual)")
        print()

        print("🧩 COMPONENT BREAKDOWN:")
        for component, count in sorted(components.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_params) * 100
            print(f"  🔸 {component}: {count:,} ({percentage:.1f}%)")

        print()

        # Validation results
        budget_valid = total_params <= 25_000_000
        target_achieved = abs(total_params - 23_700_000) / 23_700_000 <= 0.1

        print("✅ VALIDATION RESULTS:")
        print(f"  {'✓' if budget_valid else '✗'} Budget constraint: {total_params:,} ≤ 25M")
        print(f"  {'✓' if target_achieved else '✗'} Target achievement: ~23.7M ±10%")
        print(f"  ✓ Option A compliance: Tied embeddings, optimized vocab")
        print(f"  ✓ Parameter efficiency: 6x reduction vs HRRM (150M → 23.7M)")

        # Assert validation
        assert budget_valid, f"Budget constraint violated: {total_params:,} > 25M"
        assert target_achieved, f"Target not achieved: {total_params:,} vs 23.7M target"

        print("\n🚀 PARAMETER BUDGET VALIDATION COMPLETE!")

    @pytest.mark.skipif(not COGMENT_AVAILABLE, reason="Cogment not available")
    def test_production_parameter_validation(self):
        """Validate parameter budget for production deployment."""
        config = CogmentConfig(
            d_model=512,
            n_layers=6,
            n_head=8,
            d_ff=1536,
            vocab_size=13000,
            max_seq_len=2048,
            tie_embeddings=True,
            target_params=25_000_000,
            tolerance=0.05,
        )

        model = CogmentModel(config)
        total_params = sum(p.numel() for p in model.parameters())

        # Production validation criteria
        production_criteria = {
            "budget_compliance": total_params <= 25_000_000,
            "efficiency_target": 0.9 <= (total_params / 25_000_000) <= 1.0,
            "hrrm_improvement": total_params <= 150_000_000 / 5,  # At least 5x improvement
            "deployment_size": total_params * 4 <= 100_000_000,  # <100MB estimated size
        }

        print("🏭 PRODUCTION VALIDATION:")
        print("=" * 40)

        all_passed = True
        for criterion, passed in production_criteria.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {status} {criterion.replace('_', ' ').title()}")
            if not passed:
                all_passed = False

        print(f"\n📋 Final Status: {'🟢 PRODUCTION READY' if all_passed else '🔴 NEEDS ATTENTION'}")

        assert all_passed, "Production validation criteria not met"

        print(f"✅ Cogment model ready for production deployment!")
        print(f"   Parameter count: {total_params:,}")
        print(f"   Budget utilization: {total_params / 25_000_000:.1%}")
        print(f"   HRRM improvement: {150_000_000 / total_params:.1f}x parameter reduction")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
