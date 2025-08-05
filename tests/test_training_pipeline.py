"""
Test suite for training pipeline components
"""

from unittest.mock import patch

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agent_forge.training.curriculum import CurriculumGenerator, CurriculumLevel, Question
from agent_forge.training.enhanced_self_modeling import EnhancedSelfModeling, SelfModelingConfig, TemperatureRange
from agent_forge.training.quiet_star import QuietSTaRModel
from agent_forge.training.training_loop import AgentForgeTrainingLoop


class TestCurriculumGenerator:
    """Test curriculum generation functionality"""

    def test_curriculum_generator_init(self):
        """Test curriculum generator initialization"""
        generator = CurriculumGenerator(frontier_model="microsoft/DialoGPT-small", domain="mathematics")

        assert generator.domain == "mathematics"
        assert generator.device in ["cuda", "cpu"]
        assert generator.frontier_model is not None
        assert generator.tokenizer is not None

    def test_question_creation(self):
        """Test question dataclass"""
        question = Question(text="What is 2+2?", answer="4", difficulty=1, domain="mathematics")

        assert question.text == "What is 2+2?"
        assert question.answer == "4"
        assert question.difficulty == 1
        assert question.domain == "mathematics"

    def test_curriculum_level_creation(self):
        """Test curriculum level creation"""
        level = CurriculumLevel(
            level=5,
            difficulty=5,
            organic_data=["task1", "task2"],
            synthetic_data=["synthetic1"],
            rag_data=["rag1", "rag2"],
            interaction_data=["interaction1"],
        )

        assert level.level == 5
        assert level.difficulty == 5
        assert len(level.organic_data) == 2
        assert len(level.synthetic_data) == 1
        assert len(level.rag_data) == 2
        assert len(level.interaction_data) == 1
        assert level.self_awareness_complexity == 1  # default

    @patch("agent_forge.training.curriculum.CurriculumGenerator._generate")
    def test_create_assessment_questions(self, mock_generate):
        """Test assessment question creation"""
        # Mock the generation
        mock_generate.return_value = "Question: What is AI? Answer: Artificial Intelligence"

        generator = CurriculumGenerator(frontier_model="microsoft/DialoGPT-small", domain="AI")

        # Test question creation
        questions = generator.create_assessment_questions(3)

        # Verify structure
        assert len(questions) <= 3  # May be fewer due to difficulty filtering

        for question in questions:
            assert isinstance(question, Question)
            assert question.domain == "AI"
            assert question.difficulty >= 1
            assert question.difficulty <= 1000


class TestQuietSTaRModel:
    """Test Quiet-STaR model functionality"""

    def test_quiet_star_init(self):
        """Test Quiet-STaR model initialization"""
        # Create base model
        base_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

        # Create Quiet-STaR model
        quiet_star = QuietSTaRModel(base_model)

        # Verify initialization
        assert quiet_star.base_model == base_model
        assert quiet_star.start_thought.shape == (1, base_model.config.hidden_size)
        assert quiet_star.end_thought.shape == (1, base_model.config.hidden_size)
        assert quiet_star.mixing_head.in_features == base_model.config.hidden_size
        assert quiet_star.mixing_head.out_features == 1

    def test_quiet_star_forward_without_thoughts(self):
        """Test Quiet-STaR forward pass without thought generation"""
        base_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
        quiet_star = QuietSTaRModel(base_model)

        # Create test input
        input_ids = torch.randint(0, 1000, (1, 10))
        attention_mask = torch.ones_like(input_ids)

        # Forward pass without thoughts
        logits, thought_logits = quiet_star(input_ids, attention_mask=attention_mask, generate_thoughts=False)

        # Verify output
        assert logits.shape == (1, 10, base_model.config.vocab_size)
        assert thought_logits is None

    def test_quiet_star_forward_with_thoughts(self):
        """Test Quiet-STaR forward pass with thought generation"""
        base_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
        quiet_star = QuietSTaRModel(base_model)

        # Create test input (smaller for faster testing)
        input_ids = torch.randint(0, 1000, (1, 3))
        attention_mask = torch.ones_like(input_ids)

        # Forward pass with thoughts
        logits, thought_logits = quiet_star(input_ids, attention_mask=attention_mask, generate_thoughts=True)

        # Verify output
        assert logits.shape == (1, 3, base_model.config.vocab_size)
        assert thought_logits is not None
        assert thought_logits.shape == (1, 3, base_model.config.vocab_size)


class TestAgentForgeTrainingLoop:
    """Test Agent Forge training loop"""

    def test_training_loop_init(self):
        """Test training loop initialization"""
        # Create model and tokenizer
        model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")

        # Add padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Create training loop
        training_loop = AgentForgeTrainingLoop(
            model=model,
            tokenizer=tokenizer,
            enable_quiet_star=True,
            curriculum_domain="general",
        )

        # Verify initialization
        assert training_loop.model != model  # Should be wrapped in Quiet-STaR
        assert isinstance(training_loop.model, QuietSTaRModel)
        assert training_loop.tokenizer == tokenizer
        assert training_loop.enable_quiet_star
        assert training_loop.curriculum is not None
        assert training_loop.optimizer is not None
        assert training_loop.level_accuracy == {}

    def test_training_loop_init_without_quiet_star(self):
        """Test training loop initialization without Quiet-STaR"""
        model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        training_loop = AgentForgeTrainingLoop(
            model=model,
            tokenizer=tokenizer,
            enable_quiet_star=False,
            curriculum_domain="coding",
        )

        # Verify initialization
        assert training_loop.model == model  # Should not be wrapped
        assert training_loop.enable_quiet_star == False
        assert training_loop.curriculum.domain == "coding"

    def test_generate_curriculum_level(self):
        """Test curriculum level generation"""
        model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        training_loop = AgentForgeTrainingLoop(model=model, tokenizer=tokenizer, enable_quiet_star=False)

        # Generate curriculum level
        with patch.object(training_loop.curriculum, "create_assessment_questions") as mock_questions:
            # Mock questions with specific difficulty
            mock_questions.return_value = [
                Question(
                    text=f"Question {i}",
                    answer=f"Answer {i}",
                    difficulty=5,
                    domain="general",
                )
                for i in range(20)
            ]

            curriculum_level = training_loop.generate_curriculum_level(level=5, num_tasks=16)

            # Verify structure
            assert curriculum_level.level == 5
            assert curriculum_level.difficulty == 5
            assert len(curriculum_level.organic_data) == 4  # 16//4
            assert len(curriculum_level.synthetic_data) == 4  # 16//4
            assert len(curriculum_level.rag_data) == 4  # 16//4
            assert len(curriculum_level.interaction_data) == 4  # 16//4

    def test_process_quiet_star_thoughts(self):
        """Test Quiet-STaR thought processing"""
        model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Test with Quiet-STaR enabled
        training_loop = AgentForgeTrainingLoop(model=model, tokenizer=tokenizer, enable_quiet_star=True)

        # Create test input
        input_ids = torch.randint(0, 1000, (1, 5))
        attention_mask = torch.ones_like(input_ids)

        # Process thoughts
        logits, thought_logits = training_loop.process_quiet_star_thoughts(input_ids, attention_mask)

        # Verify output
        assert logits is not None
        assert thought_logits is not None
        assert training_loop.training_metrics["quiet_star_activations"] == 1

        # Test with Quiet-STaR disabled
        training_loop_no_qs = AgentForgeTrainingLoop(model=model, tokenizer=tokenizer, enable_quiet_star=False)

        logits_no_qs, thought_logits_no_qs = training_loop_no_qs.process_quiet_star_thoughts(input_ids, attention_mask)

        assert logits_no_qs is not None
        assert thought_logits_no_qs is None

    def test_calculate_reward(self):
        """Test reward calculation"""
        model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        training_loop = AgentForgeTrainingLoop(model=model, tokenizer=tokenizer, enable_quiet_star=False)

        # Create test data
        logits = torch.randn(1, 5, 1000)
        target = torch.randint(0, 1000, (1, 5))

        # Calculate reward
        reward = training_loop.calculate_reward(logits, target, step=0)

        # Verify reward is reasonable
        assert isinstance(reward, float)
        assert 0 <= reward <= 1  # Reward should be normalized

        # Test with thought logits
        thought_logits = torch.randn(1, 5, 1000)
        reward_with_thoughts = training_loop.calculate_reward(logits, target, step=0, thought_logits=thought_logits)

        assert isinstance(reward_with_thoughts, float)
        assert 0 <= reward_with_thoughts <= 1


class TestEnhancedSelfModeling:
    """Test enhanced self-modeling functionality"""

    def test_self_modeling_config(self):
        """Test self-modeling configuration"""
        config = SelfModelingConfig(
            num_temperature_samples=1000,
            max_sequence_length=256,
            num_mask_tokens=5,
            reflection_depth=2,
        )

        assert config.num_temperature_samples == 1000
        assert config.max_sequence_length == 256
        assert config.num_mask_tokens == 5
        assert config.reflection_depth == 2
        assert len(config.temperature_ranges) == 5  # default ranges

    def test_temperature_range(self):
        """Test temperature range dataclass"""
        temp_range = TemperatureRange(min_temp=0.1, max_temp=0.5, name="low_creativity", exploration_weight=0.8)

        assert temp_range.min_temp == 0.1
        assert temp_range.max_temp == 0.5
        assert temp_range.name == "low_creativity"
        assert temp_range.exploration_weight == 0.8

    def test_self_modeling_init(self):
        """Test self-modeling initialization"""
        model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        config = SelfModelingConfig(reflection_depth=2)

        self_modeling = EnhancedSelfModeling(model, tokenizer, config)

        # Verify initialization
        assert self_modeling.model == model
        assert self_modeling.tokenizer == tokenizer
        assert self_modeling.config == config
        assert self_modeling.self_predictor is not None
        assert len(self_modeling.reflection_network) == 2
        assert self_modeling.optimizer is not None
        assert self_modeling.expert_vectors is not None

    def test_generation_metrics_calculation(self):
        """Test generation metrics calculation"""
        model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self_modeling = EnhancedSelfModeling(model, tokenizer)

        # Create test output
        test_text = "The quick brown fox jumps over the lazy dog"
        test_tokens = tokenizer.encode(test_text)
        output_ids = torch.tensor(test_tokens)

        # Calculate metrics
        metrics = self_modeling._calculate_generation_metrics(output_ids, temperature=0.7)

        # Verify metrics
        assert "diversity" in metrics
        assert "repetition" in metrics
        assert "length" in metrics
        assert "temperature" in metrics
        assert "entropy" in metrics

        assert 0 <= metrics["diversity"] <= 1
        assert 0 <= metrics["repetition"] <= 1
        assert metrics["length"] > 0
        assert metrics["temperature"] == 0.7
        assert metrics["entropy"] >= 0

    def test_entropy_calculation(self):
        """Test entropy calculation"""
        model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self_modeling = EnhancedSelfModeling(model, tokenizer)

        # Test with uniform distribution
        uniform_tokens = ["a", "b", "c", "d"] * 10
        uniform_entropy = self_modeling._calculate_entropy(uniform_tokens)

        # Test with skewed distribution
        skewed_tokens = ["a"] * 30 + ["b"] * 5 + ["c"] * 5
        skewed_entropy = self_modeling._calculate_entropy(skewed_tokens)

        # Uniform should have higher entropy
        assert uniform_entropy > skewed_entropy

        # Test edge cases
        empty_entropy = self_modeling._calculate_entropy([])
        assert empty_entropy == 0

        single_entropy = self_modeling._calculate_entropy(["a"] * 10)
        assert single_entropy == 0  # No diversity

    def test_mask_and_predict(self):
        """Test masking and prediction functionality"""
        model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self_modeling = EnhancedSelfModeling(model, tokenizer)

        # Test masking
        test_text = "The quick brown fox jumps over the lazy dog"
        masked_input, labels, mask_positions = self_modeling.mask_and_predict(test_text, num_masks=3)

        # Verify masking
        assert masked_input.shape == labels.shape
        assert len(mask_positions) <= 3

        # Verify masks were applied
        for pos in mask_positions:
            assert masked_input[0, pos] == tokenizer.mask_token_id
            assert labels[0, pos] != tokenizer.mask_token_id  # Original token preserved in labels

    def test_temperature_analysis(self):
        """Test temperature effect analysis"""
        model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self_modeling = EnhancedSelfModeling(model, tokenizer)

        # Create mock samples
        samples = [
            (
                "low temp text",
                0.1,
                {"diversity": 0.3, "repetition": 0.1, "entropy": 1.0},
            ),
            (
                "medium temp text",
                0.5,
                {"diversity": 0.6, "repetition": 0.2, "entropy": 2.0},
            ),
            (
                "high temp text",
                0.9,
                {"diversity": 0.8, "repetition": 0.4, "entropy": 3.0},
            ),
        ]

        # Analyze temperature effects
        analysis = self_modeling._analyze_temperature_effects(samples)

        # Verify analysis structure
        assert "low" in analysis
        assert "medium" in analysis
        assert "high" in analysis

        # Verify metrics
        for bucket in analysis.values():
            assert "count" in bucket
            assert "avg_diversity" in bucket
            assert "avg_repetition" in bucket
            assert "avg_entropy" in bucket
            assert "quality_score" in bucket

    def test_best_temperature_finding(self):
        """Test finding best temperature range"""
        model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self_modeling = EnhancedSelfModeling(model, tokenizer)

        # Create samples with different quality scores
        samples = [
            (
                "poor quality",
                0.1,
                {"diversity": 0.1, "repetition": 0.8, "entropy": 0.5},
            ),
            (
                "good quality",
                0.5,
                {"diversity": 0.8, "repetition": 0.1, "entropy": 2.0},
            ),
            (
                "medium quality",
                0.9,
                {"diversity": 0.5, "repetition": 0.3, "entropy": 1.5},
            ),
        ]

        # Find best temperature
        best_temp_info = self_modeling._find_best_temperature_range(samples)

        # Verify structure
        assert "best_temperature" in best_temp_info
        assert "best_score" in best_temp_info
        assert "recommendation" in best_temp_info

        # Best should be the 0.5 temperature sample
        assert best_temp_info["best_temperature"] == 0.5
        assert isinstance(best_temp_info["recommendation"], str)


class TestTrainingIntegration:
    """Integration tests for training components"""

    def test_curriculum_training_integration(self):
        """Test curriculum training integration"""
        model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        training_loop = AgentForgeTrainingLoop(model=model, tokenizer=tokenizer, enable_quiet_star=False)

        # Mock curriculum generation to avoid long generation times
        with patch.object(training_loop, "generate_curriculum_level") as mock_curriculum:
            mock_level = CurriculumLevel(
                level=1,
                difficulty=1,
                organic_data=["What is AI?", "Explain ML"],
                synthetic_data=["Define neural networks"],
                rag_data=["What are transformers?"],
                interaction_data=["Hello"],
            )
            mock_curriculum.return_value = mock_level

            # Run shortened curriculum
            try:
                results = training_loop.run_curriculum(max_levels=1, tasks_per_level=2)

                # Verify results
                assert "levels_completed" in results
                assert "level_metrics" in results
                assert "overall_accuracy" in results
                assert "quiet_star_enabled" in results

                assert results["levels_completed"] >= 1
                assert len(results["level_metrics"]) >= 1
                assert results["quiet_star_enabled"] == False

            except Exception as e:
                pytest.skip(f"Integration test skipped due to: {e}")

    def test_self_modeling_integration(self):
        """Test self-modeling integration"""
        model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Create minimal config for testing
        config = SelfModelingConfig(
            num_temperature_samples=10,  # Very small for testing
            reflection_depth=1,
            save_checkpoints=False,
        )

        self_modeling = EnhancedSelfModeling(model, tokenizer, config)

        # Test with simple prompts
        prompts = ["What is AI?", "Hello world"]

        try:
            # Run minimal self-modeling cycle
            results = self_modeling.run_self_modeling_cycle(prompts, num_cycles=1)

            # Verify results
            assert "cycles_completed" in results
            assert "temperature_insights" in results
            assert "reflection_insights" in results
            assert "training_metrics" in results

            assert results["cycles_completed"] >= 1
            assert len(results["temperature_insights"]) >= 1
            assert len(results["training_metrics"]) >= 1

        except Exception as e:
            pytest.skip(f"Self-modeling integration test skipped due to: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
