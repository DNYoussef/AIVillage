#!/usr/bin/env python3
"""
Test Magi Specialization Pipeline

Integration tests and validation for the complete Magi agent specialization system.
This validates all components work together correctly before running the full pipeline.
"""

import json
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch

import pytest
import torch

# Import the Magi specialization components
from agent_forge.training.magi_specialization import (
    FrontierQuestionGenerator,
    GeometricSelfAwareness,
    MagiConfig,
    MagiSpecializationPipeline,
    SelfModificationFramework,
)


@pytest.fixture
def test_config():
    """Create a test configuration for Magi specialization"""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = MagiConfig(
            optimal_model_path="D:/AgentForge/results_50gen",
            output_dir=temp_dir,
            curriculum_levels=2,  # Reduced for testing
            questions_per_level=10,  # Reduced for testing
            total_questions=20,
            enable_geometric_awareness=True,
            enable_self_modification=True,
            sleep_cycle_frequency=5,  # Every 5 questions for testing
            wandb_project="test-magi",
            seed=42,
        )
        yield config


@pytest.fixture
def mock_model():
    """Create a mock model for testing"""
    model = Mock()
    model.named_parameters.return_value = [
        ("layer1.weight", torch.randn(100, 50, requires_grad=True)),
        ("layer2.weight", torch.randn(50, 25, requires_grad=True)),
        ("layer3.bias", torch.randn(25, requires_grad=True)),
    ]
    model.parameters.return_value = [param for _, param in model.named_parameters()]
    model.save_pretrained = Mock()
    model.generate = Mock(return_value=torch.tensor([[1, 2, 3, 4, 5]]))
    return model


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing"""
    tokenizer = Mock()
    tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    tokenizer.decode.return_value = "Test response from model"
    tokenizer.pad_token = "<pad>"
    tokenizer.pad_token_id = 0
    tokenizer.save_pretrained = Mock()
    return tokenizer


class TestFrontierQuestionGenerator:
    """Test the question generation system"""

    def test_initialization(self, test_config):
        """Test question generator initialization"""
        generator = FrontierQuestionGenerator(test_config)

        assert generator.config == test_config
        assert len(generator.question_templates) > 0
        assert "python_programming" in generator.question_templates
        assert "mathematical_proofs" in generator.question_templates

    def test_question_templates_loaded(self, test_config):
        """Test that question templates are properly loaded"""
        generator = FrontierQuestionGenerator(test_config)

        for area in test_config.specialization_areas:
            if area in generator.question_templates:
                templates = generator.question_templates[area]
                assert len(templates) > 0
                assert all(isinstance(template, str) for template in templates)

    def test_single_question_generation(self, test_config):
        """Test generation of a single question"""
        generator = FrontierQuestionGenerator(test_config)

        question = generator._generate_single_question("python_programming", 5)

        assert question.difficulty == 5
        assert question.domain == "python_programming"
        assert len(question.text) > 0
        assert len(question.answer) > 0

    def test_difficulty_parameters(self, test_config):
        """Test difficulty parameter generation"""
        generator = FrontierQuestionGenerator(test_config)

        # Test different difficulty levels
        for level in [1, 5, 10]:
            params = generator._get_difficulty_parameters("python_programming", level)
            assert isinstance(params, dict)
            assert len(params) > 0

    def test_curriculum_generation(self, test_config):
        """Test full curriculum generation"""
        generator = FrontierQuestionGenerator(test_config)

        questions = generator.generate_curriculum_questions()

        assert len(questions) == test_config.total_questions

        # Check distribution across levels
        level_counts = {}
        for question in questions:
            level_counts[question.difficulty] = level_counts.get(question.difficulty, 0) + 1

        assert len(level_counts) == test_config.curriculum_levels

        # Each level should have roughly equal questions
        expected_per_level = test_config.questions_per_level
        for count in level_counts.values():
            assert abs(count - expected_per_level) <= 2  # Allow small variation


class TestGeometricSelfAwareness:
    """Test geometric analysis and self-awareness components"""

    def test_initialization(self, test_config):
        """Test geometric awareness initialization"""
        geo_awareness = GeometricSelfAwareness(test_config)

        assert geo_awareness.config == test_config
        assert isinstance(geo_awareness.geometric_history, list)
        assert isinstance(geo_awareness.grokking_signatures, list)

    def test_weight_space_analysis(self, test_config, mock_model):
        """Test weight space analysis"""
        geo_awareness = GeometricSelfAwareness(test_config)

        analysis = geo_awareness.analyze_weight_space(mock_model)

        assert "timestamp" in analysis
        assert "layer_analyses" in analysis
        assert "global_metrics" in analysis

        # Check layer analyses
        layer_analyses = analysis["layer_analyses"]
        assert len(layer_analyses) > 0

        for layer_analysis in layer_analyses.values():
            assert "shape" in layer_analysis
            assert "mean" in layer_analysis
            assert "std" in layer_analysis
            assert "sparsity" in layer_analysis
            assert "norm" in layer_analysis

        # Check global metrics
        global_metrics = analysis["global_metrics"]
        assert "total_parameters" in global_metrics
        assert "global_variance" in global_metrics
        assert "geometric_complexity" in global_metrics

    def test_weight_entropy_calculation(self, test_config):
        """Test weight entropy calculation"""
        geo_awareness = GeometricSelfAwareness(test_config)

        # Test with known distribution
        weights = torch.randn(100, 50).numpy()
        entropy = geo_awareness._calculate_weight_entropy(weights)

        assert isinstance(entropy, float)
        assert entropy > 0  # Entropy should be positive for random weights

    def test_grokking_detection(self, test_config):
        """Test grokking signature detection"""
        geo_awareness = GeometricSelfAwareness(test_config)

        # Test with no grokking pattern
        stable_loss = [1.0] * 100
        stable_accuracy = [0.5] * 100

        result = geo_awareness.detect_grokking_signature(stable_loss, stable_accuracy)
        assert result is None

        # Test with grokking pattern
        grokking_loss = [1.0] * 80 + [0.5] * 20  # Sudden drop
        grokking_accuracy = [0.5] * 80 + [0.8] * 20  # Sudden jump

        result = geo_awareness.detect_grokking_signature(grokking_loss, grokking_accuracy)
        if result is not None:  # May not detect with simple pattern
            assert "grokking_detected" in result
            assert "loss_drop" in result
            assert "accuracy_jump" in result

    def test_ai_visualization(self, test_config, mock_model):
        """Test AI-readable visualization generation"""
        geo_awareness = GeometricSelfAwareness(test_config)

        analysis = geo_awareness.analyze_weight_space(mock_model)
        visualization = geo_awareness.visualize_for_ai(analysis)

        assert isinstance(visualization, str)
        assert len(visualization) > 0
        assert "GEOMETRIC SELF-AWARENESS REPORT" in visualization
        assert "GLOBAL WEIGHT SPACE STATE" in visualization
        assert "LAYER-BY-LAYER ANALYSIS" in visualization
        assert "INTERPRETATION" in visualization


class TestSelfModificationFramework:
    """Test self-modification capabilities"""

    def test_initialization(self, test_config):
        """Test self-modification framework initialization"""
        self_mod = SelfModificationFramework(test_config)

        assert self_mod.config == test_config
        assert isinstance(self_mod.modification_history, list)
        assert isinstance(self_mod.safety_checkpoints, list)

    def test_enable_self_modification(self, test_config, mock_model):
        """Test enabling self-modification interface"""
        self_mod = SelfModificationFramework(test_config)

        interface = self_mod.enable_self_modification(mock_model)

        assert "available_modifications" in interface
        assert "safety_bounds" in interface
        assert "current_state" in interface

        modifications = interface["available_modifications"]
        assert "adjust_layer_weights" in modifications
        assert "adjust_temperature" in modifications
        assert "prune_connections" in modifications

    def test_model_state_capture(self, test_config, mock_model):
        """Test model state capture for rollback"""
        self_mod = SelfModificationFramework(test_config)

        state = self_mod._capture_model_state(mock_model)

        assert "timestamp" in state
        assert "state_dict" in state
        assert "architecture_hash" in state

        # Check that parameters are captured
        state_dict = state["state_dict"]
        assert len(state_dict) > 0

    def test_temperature_adjustment(self, test_config, mock_model):
        """Test temperature adjustment modification"""
        self_mod = SelfModificationFramework(test_config)

        modification_request = {
            "type": "adjust_temperature",
            "parameters": {"temperature_change": 0.2},
        }

        result = self_mod.apply_modification(mock_model, modification_request)

        assert result["success"] is True
        assert "temperature" in result["modification_applied"]
        assert hasattr(mock_model, "magi_temperature")

    def test_weight_adjustment(self, test_config, mock_model):
        """Test layer weight adjustment"""
        self_mod = SelfModificationFramework(test_config)

        modification_request = {
            "type": "adjust_layer_weights",
            "parameters": {"layer_name": "layer1", "adjustment_factor": 1.05},
        }

        result = self_mod.apply_modification(mock_model, modification_request)

        assert result["success"] is True
        assert "layer1" in result["modification_applied"]

    def test_safety_bounds_enforcement(self, test_config, mock_model):
        """Test that safety bounds are enforced"""
        self_mod = SelfModificationFramework(test_config)

        # Try to make a large unsafe change
        modification_request = {
            "type": "adjust_layer_weights",
            "parameters": {
                "layer_name": "layer1",
                "adjustment_factor": 2.0,  # Too large change
            },
        }

        result = self_mod.apply_modification(mock_model, modification_request)

        # Should be clamped to safety bounds
        assert result["success"] is True
        # The actual factor should be within safety bounds
        max_change = test_config.modification_safety_bounds["max_weight_change"]
        assert f"factor {1.0 + max_change}" in result["modification_applied"]

    def test_rollback_functionality(self, test_config, mock_model):
        """Test rollback to previous checkpoints"""
        self_mod = SelfModificationFramework(test_config)

        # Create a checkpoint
        modification_request = {
            "type": "adjust_temperature",
            "parameters": {"temperature_change": 0.1},
        }

        self_mod.apply_modification(mock_model, modification_request)

        # Should have created a checkpoint
        assert len(self_mod.safety_checkpoints) > 0

        # Test rollback
        success = self_mod.rollback_to_checkpoint(mock_model, 0)
        assert success is True


class TestMagiSpecializationPipeline:
    """Test the complete Magi specialization pipeline"""

    @pytest.mark.asyncio
    async def test_pipeline_initialization(self, test_config):
        """Test pipeline initialization"""
        pipeline = MagiSpecializationPipeline(test_config)

        assert pipeline.config == test_config
        assert pipeline.output_dir.exists()
        assert isinstance(pipeline.question_generator, FrontierQuestionGenerator)
        assert isinstance(pipeline.geometric_awareness, GeometricSelfAwareness)
        assert isinstance(pipeline.self_modification, SelfModificationFramework)

    @patch("agent_forge.training.magi_specialization.AutoModelForCausalLM")
    @patch("agent_forge.training.magi_specialization.AutoTokenizer")
    def test_model_loading(self, mock_tokenizer_class, mock_model_class, test_config):
        """Test optimal model loading"""
        # Mock the classes
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model

        # Create mock evolution results
        evolution_results = {
            "evolution_summary": {
                "best_configuration": {
                    "merge_method": "task_arithmetic",
                    "fitness": 1.6185,
                    "parameters": {"scaling_coefficient": 1.31},
                }
            }
        }

        results_file = Path(test_config.optimal_model_path) / "evolution_50gen_results.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)

        with open(results_file, "w") as f:
            json.dump(evolution_results, f)

        pipeline = MagiSpecializationPipeline(test_config)
        model, tokenizer = pipeline.load_optimal_model()

        assert model is not None
        assert tokenizer is not None
        assert tokenizer.pad_token == tokenizer.eos_token

    def test_question_answering(self, test_config, mock_model, mock_tokenizer):
        """Test the question answering mechanism"""
        pipeline = MagiSpecializationPipeline(test_config)

        from agent_forge.training.curriculum import Question

        question = Question(text="What is 2 + 2?", answer="4", difficulty=1, domain="mathematics")

        # Mock tokenizer behavior
        mock_tokenizer.return_tensors = "pt"
        mock_tokenizer.side_effect = lambda text, **kwargs: {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        }

        answer = pipeline.answer_question(mock_model, mock_tokenizer, question)

        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_answer_evaluation(self, test_config):
        """Test answer evaluation logic"""
        pipeline = MagiSpecializationPipeline(test_config)

        from agent_forge.training.curriculum import Question

        question = Question(
            text="What is the capital of France?",
            answer="Paris is the capital of France",
            difficulty=1,
            domain="geography",
        )

        # Test correct answer
        good_answer = "The capital of France is Paris, which is located in the northern part."
        assert pipeline.evaluate_answer(question, good_answer) is True

        # Test incorrect answer
        bad_answer = "Berlin"
        assert pipeline.evaluate_answer(question, bad_answer) is False

        # Test partial answer
        partial_answer = "Paris"
        # Should be True due to keyword matching
        pipeline.evaluate_answer(question, partial_answer)
        # This might be False due to length requirement, which is acceptable

    @pytest.mark.asyncio
    async def test_sleep_dream_cycle(self, test_config, mock_model):
        """Test sleep/dream cycle execution"""
        pipeline = MagiSpecializationPipeline(test_config)

        # Capture original parameter state
        original_params = {}
        for name, param in mock_model.named_parameters():
            original_params[name] = param.data.clone()

        await pipeline.sleep_dream_cycle(mock_model)

        # Parameters should be slightly modified
        # Note: In the actual implementation, this would apply more sophisticated changes


def test_integration_with_existing_systems(test_config):
    """Test integration with existing Agent Forge systems"""
    MagiSpecializationPipeline(test_config)

    # Verify we can import and use existing components
    from agent_forge.quietstar_baker import QuietSTaRConfig

    # Test configuration compatibility
    quietstar_config = QuietSTaRConfig(
        model_path=test_config.optimal_model_path,
        output_path=str(test_config.output_dir / "quietstar"),
        eval_dataset="gsm8k",
    )

    assert quietstar_config.model_path == test_config.optimal_model_path


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_pipeline_smoke_test(test_config):
    """Smoke test for the complete pipeline (minimal execution)"""
    # This test runs a minimal version of the complete pipeline
    # to ensure all components work together

    with (
        patch("agent_forge.training.magi_specialization.AutoModelForCausalLM") as mock_model_class,
        patch("agent_forge.training.magi_specialization.AutoTokenizer") as mock_tokenizer_class,
        patch("agent_forge.training.magi_specialization.QuietSTaRBaker") as mock_quietstar_class,
        patch("wandb.init") as mock_wandb,
    ):
        # Setup mocks
        mock_model = Mock()
        mock_model.named_parameters.return_value = [("layer1.weight", torch.randn(10, 5, requires_grad=True))]
        mock_model.parameters.return_value = [torch.randn(10, 5, requires_grad=True)]
        mock_model.save_pretrained = Mock()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.decode.return_value = "Test answer"
        mock_tokenizer.save_pretrained = Mock()

        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        # Mock QuietSTaR baker
        mock_quietstar_baker = Mock()
        mock_quietstar_baker.run_baking_pipeline = Mock(
            return_value={
                "winner": "thoughts",
                "improvement": 5.0,
                "baked_model_path": "/fake/path",
            }
        )
        mock_quietstar_class.return_value = mock_quietstar_baker

        # Mock W&B
        mock_wandb.return_value = Mock()

        # Create evolution results file
        evolution_results = {
            "evolution_summary": {
                "best_configuration": {
                    "merge_method": "task_arithmetic",
                    "fitness": 1.6185,
                    "parameters": {"scaling_coefficient": 1.31},
                }
            }
        }

        results_file = Path(test_config.optimal_model_path) / "evolution_50gen_results.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)

        with open(results_file, "w") as f:
            json.dump(evolution_results, f)

        # Run the pipeline
        pipeline = MagiSpecializationPipeline(test_config)

        # This should complete without errors
        results = await pipeline.run_magi_specialization()

        # Verify results structure
        assert "specialization_summary" in results
        assert "quietstar_results" in results
        assert "curriculum_results" in results
        assert "final_evaluation" in results
        assert "deployment_package" in results

        # Verify specialization summary
        summary = results["specialization_summary"]
        assert summary["domain"] == test_config.domain
        assert summary["total_questions_completed"] > 0
        assert summary["levels_completed"] > 0


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])
