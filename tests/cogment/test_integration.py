"""
Tests for Cogment Integration Layer (Agent 6)

Tests the integration components including:
- EvoMerge adapter for single Cogment model workflow
- HuggingFace export for unified model deployment
- Model compatibility validation (ACT and LTM preservation)
- Deployment manager and production pipeline
- Phase controller replacing 3-phase HRRM with 4-stage Cogment
"""

from pathlib import Path
import tempfile

import pytest
import torch

# Import Cogment integration components
try:
    from core.agent_forge.integration.cogment.deployment_manager import CogmentDeploymentManager
    from core.agent_forge.integration.cogment.evomerge_adapter import CogmentEvoMergeAdapter
    from core.agent_forge.integration.cogment.hf_export import CogmentHFExporter
    from core.agent_forge.integration.cogment.model_compatibility import CogmentCompatibilityValidator
    from core.agent_forge.integration.cogment.phase_controller import CogmentPhaseController
    from core.agent_forge.models.cogment.core.config import CogmentConfig
    from core.agent_forge.models.cogment.core.model import CogmentModel

    INTEGRATION_AVAILABLE = True
except ImportError as e:
    print(f"Cogment integration components not available: {e}")
    INTEGRATION_AVAILABLE = False


class TestCogmentEvoMergeAdapter:
    """Test EvoMerge adapter for Cogment integration."""

    @pytest.fixture
    def adapter_config(self):
        """Create EvoMerge adapter configuration."""
        return {
            "preserve_act_components": True,
            "preserve_ltm_components": True,
            "merge_strategy": "weighted_average",
            "compatibility_check": True,
            "backup_original": True,
        }

    @pytest.fixture
    def evomerge_adapter(self, adapter_config):
        """Create CogmentEvoMergeAdapter instance."""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration components not available")
        return CogmentEvoMergeAdapter(adapter_config)

    @pytest.fixture
    def sample_cogment_model(self):
        """Create sample Cogment model for testing."""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration components not available")

        config = CogmentConfig(d_model=256, n_layers=4, vocab_size=5000, max_seq_len=128)
        return CogmentModel(config)

    def test_adapter_creation(self, evomerge_adapter, adapter_config):
        """Test EvoMerge adapter instantiation."""
        assert evomerge_adapter.config == adapter_config
        assert hasattr(evomerge_adapter, "prepare_model_for_merging")
        assert hasattr(evomerge_adapter, "post_merge_validation")

    def test_model_preparation_for_merging(self, evomerge_adapter, sample_cogment_model):
        """Test model preparation for EvoMerge."""
        # Prepare model for merging
        prepared_state = evomerge_adapter.prepare_model_for_merging(sample_cogment_model)

        # Verify preparation state
        assert "model_state_dict" in prepared_state
        assert "act_components" in prepared_state
        assert "ltm_components" in prepared_state
        assert "compatibility_info" in prepared_state

        # Verify critical components are preserved
        assert prepared_state["act_components"] is not None
        assert prepared_state["ltm_components"] is not None

    def test_single_model_workflow(self, evomerge_adapter, sample_cogment_model):
        """Test single model workflow vs HRRM 3-model approach."""
        # Test that Cogment works as single model
        batch_size = 2
        seq_len = 16
        vocab_size = sample_cogment_model.config.vocab_size

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            outputs = sample_cogment_model(input_ids)

        # Should produce all required outputs in single forward pass
        assert hasattr(outputs, "logits")  # Planning (replaces HRRM Planner)
        assert hasattr(outputs, "act_outputs")  # Reasoning (replaces HRRM Reasoner)
        assert hasattr(outputs, "memory_outputs")  # Memory (replaces HRRM Memory)

        # Verify no need for multiple model coordination
        assert outputs.logits.shape == (batch_size, seq_len, vocab_size)

        print("✓ Single model workflow replaces HRRM 3-model approach")

    def test_merge_compatibility_validation(self, evomerge_adapter, sample_cogment_model):
        """Test merge compatibility validation."""
        # Create two models for merging test
        model1 = sample_cogment_model
        model2 = CogmentModel(sample_cogment_model.config)

        # Test compatibility
        is_compatible = evomerge_adapter.validate_merge_compatibility(model1, model2)

        assert isinstance(is_compatible, bool)
        # Same architecture should be compatible
        assert is_compatible, "Same architecture models should be merge compatible"

        # Test incompatible models - different architectures
        incompatible_config = CogmentConfig(d_model=128, n_layers=2, vocab_size=3000, max_seq_len=64)
        model3 = CogmentModel(incompatible_config)

        is_incompatible = evomerge_adapter.validate_merge_compatibility(model1, model3)
        assert not is_incompatible, "Different architecture models should not be merge compatible"

        # Test edge case: None model
        try:
            none_result = evomerge_adapter.validate_merge_compatibility(model1, None)
            assert not none_result, "Validation with None should return False"
        except (TypeError, AttributeError):
            # Should handle None gracefully with exception
            pass

    def test_act_preservation_during_merge(self, evomerge_adapter, sample_cogment_model):
        """Test ACT component preservation during merge."""
        # Extract ACT state before merge
        pre_merge_act_state = evomerge_adapter.extract_act_state(sample_cogment_model)

        # Simulate merge operation (simplified)
        merged_model = evomerge_adapter.simulate_merge([sample_cogment_model])

        # Extract ACT state after merge
        post_merge_act_state = evomerge_adapter.extract_act_state(merged_model)

        # Verify ACT components are preserved
        assert pre_merge_act_state is not None
        assert post_merge_act_state is not None

        # Key ACT parameters should be similar
        act_similarity = evomerge_adapter.compare_act_states(pre_merge_act_state, post_merge_act_state)
        assert act_similarity > 0.9, f"ACT components should be preserved: {act_similarity:.3f}"

    def test_ltm_preservation_during_merge(self, evomerge_adapter, sample_cogment_model):
        """Test LTM component preservation during merge."""
        # Extract LTM state before merge
        pre_merge_ltm_state = evomerge_adapter.extract_ltm_state(sample_cogment_model)

        # Simulate merge operation
        merged_model = evomerge_adapter.simulate_merge([sample_cogment_model])

        # Extract LTM state after merge
        post_merge_ltm_state = evomerge_adapter.extract_ltm_state(merged_model)

        # Verify LTM components are preserved
        assert pre_merge_ltm_state is not None
        assert post_merge_ltm_state is not None

        # Key LTM parameters should be similar
        ltm_similarity = evomerge_adapter.compare_ltm_states(pre_merge_ltm_state, post_merge_ltm_state)
        assert ltm_similarity > 0.9, f"LTM components should be preserved: {ltm_similarity:.3f}"


class TestCogmentHFExporter:
    """Test HuggingFace export functionality."""

    @pytest.fixture
    def export_config(self):
        """Create HF export configuration."""
        return {
            "model_name": "cogment-test",
            "export_tokenizer": True,
            "export_config": True,
            "validate_export": True,
            "compression": "none",
        }

    @pytest.fixture
    def hf_exporter(self, export_config):
        """Create CogmentHFExporter instance."""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration components not available")
        return CogmentHFExporter(export_config)

    @pytest.fixture
    def sample_model(self):
        """Create sample model for export."""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration components not available")

        config = CogmentConfig(d_model=128, n_layers=2, vocab_size=1000, max_seq_len=64)
        return CogmentModel(config)

    def test_hf_exporter_creation(self, hf_exporter, export_config):
        """Test HF exporter instantiation."""
        assert hf_exporter.config == export_config
        assert hasattr(hf_exporter, "export_model")
        assert hasattr(hf_exporter, "validate_exported_model")

    def test_model_export_preparation(self, hf_exporter, sample_model):
        """Test model export preparation."""
        # Prepare model for HF export
        export_state = hf_exporter.prepare_for_export(sample_model)

        # Verify export state
        assert "model_config" in export_state
        assert "model_state_dict" in export_state
        assert "model_architecture" in export_state

        # Verify HF compatibility
        assert export_state["model_config"] is not None
        assert len(export_state["model_state_dict"]) > 0

    def test_unified_model_export(self, hf_exporter, sample_model):
        """Test unified model export vs HRRM 3-model approach."""
        with tempfile.TemporaryDirectory() as temp_dir:
            export_path = Path(temp_dir) / "cogment_model"

            # Export unified Cogment model
            export_result = hf_exporter.export_model(sample_model, export_path)

            # Verify export success
            assert export_result["success"] is True
            assert export_path.exists()

            # Verify single model artifacts (vs 3 separate HRRM models)
            model_files = list(export_path.glob("*.bin")) + list(export_path.glob("*.safetensors"))
            assert len(model_files) >= 1, "Should have model artifacts"

            # Total export size should be smaller than 3 HRRM models
            total_size = sum(f.stat().st_size for f in model_files)
            max_expected_size = 200 * 1024 * 1024  # 200MB (vs ~600MB for 3 HRRM models)

            assert total_size <= max_expected_size, f"Unified model export too large: {total_size / 1024 / 1024:.1f}MB"

            print(f"✓ Unified model export: {total_size / 1024 / 1024:.1f}MB")

    def test_exported_model_validation(self, hf_exporter, sample_model):
        """Test exported model validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            export_path = Path(temp_dir) / "cogment_model"

            # Export model
            hf_exporter.export_model(sample_model, export_path)

            # Validate exported model
            validation_result = hf_exporter.validate_exported_model(export_path)

            # Verify validation
            assert validation_result["valid"] is True
            assert "model_loadable" in validation_result
            assert "forward_pass_works" in validation_result
            assert "output_compatible" in validation_result

            # All checks should pass
            assert validation_result["model_loadable"] is True
            assert validation_result["forward_pass_works"] is True
            assert validation_result["output_compatible"] is True


class TestCogmentCompatibilityValidator:
    """Test model compatibility validation."""

    @pytest.fixture
    def compatibility_validator(self):
        """Create CogmentCompatibilityValidator instance."""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration components not available")
        return CogmentCompatibilityValidator()

    @pytest.fixture
    def sample_models(self):
        """Create sample models for compatibility testing."""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration components not available")

        config = CogmentConfig(d_model=128, n_layers=2, vocab_size=1000, max_seq_len=64)

        model1 = CogmentModel(config)
        model2 = CogmentModel(config)

        return model1, model2

    def test_validator_creation(self, compatibility_validator):
        """Test compatibility validator instantiation."""
        assert hasattr(compatibility_validator, "validate_model_compatibility")
        assert hasattr(compatibility_validator, "validate_act_compatibility")
        assert hasattr(compatibility_validator, "validate_ltm_compatibility")

    def test_model_architecture_compatibility(self, compatibility_validator, sample_models):
        """Test model architecture compatibility validation."""
        model1, model2 = sample_models

        # Test architecture compatibility
        compatibility_result = compatibility_validator.validate_model_compatibility(model1, model2)

        # Verify compatibility structure
        assert "compatible" in compatibility_result
        assert "architecture_match" in compatibility_result
        assert "parameter_count_match" in compatibility_result
        assert "config_compatibility" in compatibility_result

        # Same architecture should be compatible
        assert compatibility_result["compatible"] is True
        assert compatibility_result["architecture_match"] is True

    def test_act_component_compatibility(self, compatibility_validator, sample_models):
        """Test ACT component compatibility."""
        model1, model2 = sample_models

        # Test ACT compatibility
        act_compatibility = compatibility_validator.validate_act_compatibility(model1, model2)

        # Verify ACT compatibility
        assert "compatible" in act_compatibility
        assert "halting_mechanism_match" in act_compatibility
        assert "step_count_compatibility" in act_compatibility

        # Same ACT configuration should be compatible
        assert act_compatibility["compatible"] is True

    def test_ltm_component_compatibility(self, compatibility_validator, sample_models):
        """Test LTM component compatibility."""
        model1, model2 = sample_models

        # Test LTM compatibility
        ltm_compatibility = compatibility_validator.validate_ltm_compatibility(model1, model2)

        # Verify LTM compatibility
        assert "compatible" in ltm_compatibility
        assert "memory_capacity_match" in ltm_compatibility
        assert "gating_mechanism_match" in ltm_compatibility

        # Same LTM configuration should be compatible
        assert ltm_compatibility["compatible"] is True

    def test_cross_version_compatibility(self, compatibility_validator):
        """Test compatibility across different model versions."""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration components not available")

        # Create models with different configurations
        config1 = CogmentConfig(d_model=128, n_layers=2, vocab_size=1000)
        config2 = CogmentConfig(d_model=256, n_layers=4, vocab_size=2000)  # Different

        model1 = CogmentModel(config1)
        model2 = CogmentModel(config2)

        # Test compatibility
        compatibility_result = compatibility_validator.validate_model_compatibility(model1, model2)

        # Different architectures should be incompatible
        assert compatibility_result["compatible"] is False
        assert compatibility_result["architecture_match"] is False


class TestCogmentDeploymentManager:
    """Test deployment manager functionality."""

    @pytest.fixture
    def deployment_config(self):
        """Create deployment configuration."""
        return {
            "deployment_type": "production",
            "enable_monitoring": True,
            "health_check_enabled": True,
            "auto_scaling": True,
            "resource_limits": {"memory": "1GB", "cpu": "2 cores"},
        }

    @pytest.fixture
    def deployment_manager(self, deployment_config):
        """Create CogmentDeploymentManager instance."""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration components not available")
        return CogmentDeploymentManager(deployment_config)

    def test_deployment_manager_creation(self, deployment_manager, deployment_config):
        """Test deployment manager instantiation."""
        assert deployment_manager.config == deployment_config
        assert hasattr(deployment_manager, "prepare_deployment")
        assert hasattr(deployment_manager, "validate_deployment_readiness")

    def test_deployment_preparation(self, deployment_manager):
        """Test deployment preparation."""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration components not available")

        # Create sample model
        config = CogmentConfig(d_model=128, n_layers=2, vocab_size=1000)
        model = CogmentModel(config)

        # Prepare deployment
        deployment_state = deployment_manager.prepare_deployment(model)

        # Verify deployment state
        assert "model_artifacts" in deployment_state
        assert "deployment_config" in deployment_state
        assert "resource_requirements" in deployment_state
        assert "monitoring_setup" in deployment_state

    def test_deployment_resource_requirements(self, deployment_manager):
        """Test deployment resource requirement calculation."""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration components not available")

        config = CogmentConfig(d_model=256, n_layers=4, vocab_size=5000)
        model = CogmentModel(config)

        # Calculate resource requirements
        requirements = deployment_manager.calculate_resource_requirements(model)

        # Verify requirements structure
        assert "memory_mb" in requirements
        assert "cpu_cores" in requirements
        assert "disk_space_mb" in requirements

        # Verify reasonable requirements for Cogment vs HRRM
        assert requirements["memory_mb"] < 2000, f"Memory requirement too high: {requirements['memory_mb']}MB"
        assert requirements["cpu_cores"] <= 4, f"CPU requirement too high: {requirements['cpu_cores']} cores"

        print(f"✓ Deployment requirements: {requirements['memory_mb']}MB RAM, {requirements['cpu_cores']} CPU cores")

    def test_deployment_readiness_validation(self, deployment_manager):
        """Test deployment readiness validation."""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration components not available")

        config = CogmentConfig(d_model=128, n_layers=2, vocab_size=1000)
        model = CogmentModel(config)

        # Validate deployment readiness
        readiness = deployment_manager.validate_deployment_readiness(model)

        # Verify readiness structure
        assert "ready" in readiness
        assert "checks" in readiness
        assert "recommendations" in readiness

        # Check individual readiness checks
        checks = readiness["checks"]
        assert "model_size_acceptable" in checks
        assert "memory_requirements_reasonable" in checks
        assert "performance_requirements_met" in checks


class TestCogmentPhaseController:
    """Test phase controller for curriculum replacement."""

    @pytest.fixture
    def phase_config(self):
        """Create phase controller configuration."""
        return {
            "replace_hrrm_phases": True,
            "cogment_curriculum_stages": 4,
            "stage_progression": "automatic",
            "performance_gating": True,
        }

    @pytest.fixture
    def phase_controller(self, phase_config):
        """Create CogmentPhaseController instance."""
        if not INTEGRATION_AVAILABLE:
            pytest.skip("Integration components not available")
        return CogmentPhaseController(phase_config)

    def test_phase_controller_creation(self, phase_controller, phase_config):
        """Test phase controller instantiation."""
        assert phase_controller.config == phase_config
        assert hasattr(phase_controller, "replace_hrrm_workflow")
        assert hasattr(phase_controller, "manage_stage_transitions")

    def test_hrrm_workflow_replacement(self, phase_controller):
        """Test HRRM 3-phase replacement with 4-stage Cogment."""
        # Test workflow replacement
        replacement_info = phase_controller.replace_hrrm_workflow()

        # Verify replacement structure
        assert "hrrm_phases_replaced" in replacement_info
        assert "cogment_stages_active" in replacement_info
        assert "workflow_mapping" in replacement_info

        # Verify HRRM is replaced
        assert replacement_info["hrrm_phases_replaced"] is True
        assert replacement_info["cogment_stages_active"] == 4

        # Verify workflow mapping
        mapping = replacement_info["workflow_mapping"]
        assert "planning" in mapping  # HRRM Planner -> Cogment Stage
        assert "reasoning" in mapping  # HRRM Reasoner -> Cogment Stage
        assert "memory" in mapping  # HRRM Memory -> Cogment Stage

    def test_stage_transition_management(self, phase_controller):
        """Test 4-stage curriculum transition management."""
        # Test stage transitions
        for stage in range(4):
            transition_info = phase_controller.manage_stage_transitions(stage)

            # Verify transition structure
            assert "current_stage" in transition_info
            assert "next_stage" in transition_info
            assert "transition_criteria" in transition_info
            assert "performance_gates" in transition_info

            # Verify stage progression
            assert transition_info["current_stage"] == stage

            if stage < 3:  # Not final stage
                assert transition_info["next_stage"] == stage + 1
            else:  # Final stage
                assert transition_info["next_stage"] is None or transition_info["next_stage"] == stage

    def test_performance_gating(self, phase_controller):
        """Test performance gating for stage transitions."""
        if not phase_controller.config["performance_gating"]:
            pytest.skip("Performance gating disabled")

        # Test performance gates
        for stage in range(4):
            gate_info = phase_controller.evaluate_performance_gate(stage)

            # Verify gate structure
            assert "gate_passed" in gate_info
            assert "performance_metrics" in gate_info
            assert "requirements" in gate_info

            # Verify gate evaluation
            assert isinstance(gate_info["gate_passed"], bool)
            assert isinstance(gate_info["performance_metrics"], dict)


@pytest.mark.integration
class TestCogmentIntegrationComplete:
    """Complete integration tests for all components."""

    @pytest.mark.skipif(not INTEGRATION_AVAILABLE, reason="Integration components not available")
    def test_complete_integration_workflow(self):
        """Test complete integration workflow from model to deployment."""
        # Create Cogment model
        config = CogmentConfig(d_model=128, n_layers=2, vocab_size=1000)
        model = CogmentModel(config)

        # Test EvoMerge integration
        evomerge_adapter = CogmentEvoMergeAdapter({"preserve_act_components": True, "preserve_ltm_components": True})

        prepared_state = evomerge_adapter.prepare_model_for_merging(model)
        assert prepared_state["act_components"] is not None
        assert prepared_state["ltm_components"] is not None

        # Test HF export
        hf_exporter = CogmentHFExporter({"model_name": "cogment-integration-test", "validate_export": True})

        with tempfile.TemporaryDirectory() as temp_dir:
            export_path = Path(temp_dir) / "model"
            export_result = hf_exporter.export_model(model, export_path)
            assert export_result["success"] is True

        # Test deployment preparation
        deployment_manager = CogmentDeploymentManager({"deployment_type": "production", "enable_monitoring": True})

        deployment_state = deployment_manager.prepare_deployment(model)
        assert "model_artifacts" in deployment_state

        # Test phase controller
        phase_controller = CogmentPhaseController({"replace_hrrm_phases": True, "cogment_curriculum_stages": 4})

        replacement_info = phase_controller.replace_hrrm_workflow()
        assert replacement_info["hrrm_phases_replaced"] is True

        print("✓ Complete integration workflow successful")

    @pytest.mark.skipif(not INTEGRATION_AVAILABLE, reason="Integration components not available")
    def test_integration_performance_vs_hrrm(self):
        """Test integration performance improvements vs HRRM."""
        # Create Cogment model
        config = CogmentConfig(d_model=256, n_layers=4, vocab_size=5000)
        model = CogmentModel(config)

        # Measure model size
        total_params = sum(p.numel() for p in model.parameters())

        # Calculate improvement metrics
        hrrm_baseline_params = 150_000_000  # 3 × 50M models
        param_reduction = hrrm_baseline_params / total_params

        # Should achieve significant reduction
        assert param_reduction >= 4.0, f"Insufficient parameter reduction: {param_reduction:.1f}x"

        # Measure deployment footprint
        deployment_manager = CogmentDeploymentManager({"deployment_type": "production"})
        requirements = deployment_manager.calculate_resource_requirements(model)

        # Should require less resources than 3 HRRM models
        memory_mb = requirements["memory_mb"]
        cpu_cores = requirements["cpu_cores"]

        # Expected improvements
        assert memory_mb < 1500, f"Memory requirement too high: {memory_mb}MB"
        assert cpu_cores <= 4, f"CPU requirement too high: {cpu_cores} cores"

        print("✓ Integration performance improvements:")
        print(f"  - Parameter reduction: {param_reduction:.1f}x ({hrrm_baseline_params:,} → {total_params:,})")
        print(f"  - Memory requirement: {memory_mb}MB")
        print(f"  - CPU requirement: {cpu_cores} cores")

    @pytest.mark.skipif(not INTEGRATION_AVAILABLE, reason="Integration components not available")
    def test_integration_backward_compatibility(self):
        """Test backward compatibility with existing systems."""
        # Create Cogment model
        config = CogmentConfig(d_model=128, n_layers=2, vocab_size=1000)
        model = CogmentModel(config)

        # Test compatibility validation
        validator = CogmentCompatibilityValidator()

        # Should be compatible with itself
        compatibility = validator.validate_model_compatibility(model, model)
        assert compatibility["compatible"] is True

        # Test export compatibility
        hf_exporter = CogmentHFExporter({"validate_export": True})

        with tempfile.TemporaryDirectory() as temp_dir:
            export_path = Path(temp_dir) / "model"
            hf_exporter.export_model(model, export_path)

            # Validate exported model works
            validation = hf_exporter.validate_exported_model(export_path)
            assert validation["valid"] is True
            assert validation["output_compatible"] is True

        print("✓ Integration backward compatibility verified")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
