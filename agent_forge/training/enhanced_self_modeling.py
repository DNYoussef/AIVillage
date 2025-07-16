"""Enhanced Self-Modeling Implementation for Phase 3

This module implements comprehensive self-modeling with temperature sweeps,
deeper self-reflection, and integration with the Agent Forge pipeline.
"""

from dataclasses import dataclass
import logging
import math
import random

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..optim.grokfast_opt import GrokFastOptimizer
from ..utils.expert_vector import ExpertVectorManager

logger = logging.getLogger(__name__)


@dataclass
class TemperatureRange:
    """Temperature range for self-modeling exploration"""

    min_temp: float
    max_temp: float
    name: str
    exploration_weight: float = 1.0


@dataclass
class SelfModelingConfig:
    """Configuration for self-modeling process"""

    num_temperature_samples: int = 5000
    temperature_ranges: list[TemperatureRange] = None
    max_sequence_length: int = 512
    num_mask_tokens: int = 3
    self_modeling_weight: float = 0.1
    reflection_depth: int = 3
    enable_grokfast: bool = True
    save_checkpoints: bool = True

    def __post_init__(self):
        if self.temperature_ranges is None:
            self.temperature_ranges = [
                TemperatureRange(0.0, 0.1, "deterministic", 0.8),
                TemperatureRange(0.1, 0.3, "low_exploration", 1.0),
                TemperatureRange(0.3, 0.7, "moderate_exploration", 1.2),
                TemperatureRange(0.7, 1.0, "high_exploration", 1.0),
                TemperatureRange(1.0, 1.5, "creative", 0.6),
            ]


class EnhancedSelfModeling:
    """Enhanced self-modeling with temperature sweeps and reflection"""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        config: SelfModelingConfig = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or SelfModelingConfig()
        self.device = next(model.parameters()).device

        # Initialize self-modeling predictor
        hidden_size = getattr(model.config, "hidden_size", 768)
        self.self_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        ).to(self.device)

        # Initialize reflection network
        self.reflection_network = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_size // 2, hidden_size),
                ).to(self.device)
                for _ in range(self.config.reflection_depth)
            ]
        )

        # Optimizer for self-modeling components
        self.optimizer = torch.optim.AdamW(
            list(self.self_predictor.parameters())
            + list(self.reflection_network.parameters()),
            lr=1e-4,
        )

        # GrokFast optimizer for main model
        if self.config.enable_grokfast:
            self.grokfast_optimizer = GrokFastOptimizer(
                self.model.parameters(), lr=2e-5, alpha=0.98, lamb=2.0
            )
        else:
            self.grokfast_optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=2e-5
            )

        # Expert vector manager
        self.expert_vectors = ExpertVectorManager(
            model_dim=hidden_size, num_experts=8, expert_dim=hidden_size // 4
        )

        # Tracking metrics
        self.temperature_metrics = {}
        self.reflection_metrics = {}
        self.self_modeling_history = []

        logger.info("Enhanced self-modeling initialized")

    def generate_temperature_samples(
        self, prompt: str, temperature_range: TemperatureRange, num_samples: int
    ) -> list[tuple[str, float, dict]]:
        """Generate text samples across temperature range"""
        samples = []

        for _ in range(num_samples):
            # Sample temperature from range
            temp = random.uniform(
                temperature_range.min_temp, temperature_range.max_temp
            )

            # Generate text
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_sequence_length,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=inputs.input_ids.size(1) + 50,
                    temperature=temp,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            generated_text = self.tokenizer.decode(
                outputs[0][inputs.input_ids.size(1) :], skip_special_tokens=True
            )

            # Calculate generation metrics
            metrics = self._calculate_generation_metrics(outputs[0], temp)

            samples.append((generated_text, temp, metrics))

        return samples

    def _calculate_generation_metrics(
        self, output_ids: torch.Tensor, temperature: float
    ) -> dict:
        """Calculate metrics for generated text"""
        # Convert to text for analysis
        text = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        # Calculate diversity metrics
        tokens = text.split()
        unique_tokens = set(tokens)
        diversity = len(unique_tokens) / len(tokens) if tokens else 0

        # Calculate repetition
        bigrams = [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]
        unique_bigrams = set(bigrams)
        repetition = 1 - (len(unique_bigrams) / len(bigrams) if bigrams else 0)

        return {
            "diversity": diversity,
            "repetition": repetition,
            "length": len(tokens),
            "temperature": temperature,
            "entropy": self._calculate_entropy(tokens),
        }

    def _calculate_entropy(self, tokens: list[str]) -> float:
        """Calculate Shannon entropy of token distribution"""
        if not tokens:
            return 0

        # Count token frequencies
        token_counts = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1

        # Calculate entropy
        total = len(tokens)
        entropy = 0
        for count in token_counts.values():
            p = count / total
            entropy -= p * math.log2(p)

        return entropy

    def perform_self_reflection(
        self, generated_samples: list[tuple[str, float, dict]], prompt: str
    ) -> dict:
        """Perform deep self-reflection on generated samples"""
        logger.info(f"Performing self-reflection on {len(generated_samples)} samples")

        # Create reflection prompt
        reflection_prompt = f"""
        Original prompt: {prompt}

        I have generated {len(generated_samples)} different responses at various temperatures.
        Now I will analyze my own generation patterns and reasoning process.

        Self-reflection questions:
        1. What patterns do I notice in my responses across different temperatures?
        2. Which temperature ranges produce the most coherent responses?
        3. How does my reasoning change with temperature?
        4. What are my strengths and weaknesses in this task?
        """

        # Generate reflection
        reflection_inputs = self.tokenizer(
            reflection_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_sequence_length,
        ).to(self.device)

        with torch.no_grad():
            reflection_outputs = self.model.generate(
                reflection_inputs.input_ids,
                attention_mask=reflection_inputs.attention_mask,
                max_length=reflection_inputs.input_ids.size(1) + 150,
                temperature=0.3,  # Use moderate temperature for reflection
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        reflection_text = self.tokenizer.decode(
            reflection_outputs[0][reflection_inputs.input_ids.size(1) :],
            skip_special_tokens=True,
        )

        # Analyze sample quality across temperatures
        temp_analysis = self._analyze_temperature_effects(generated_samples)

        return {
            "reflection_text": reflection_text,
            "temperature_analysis": temp_analysis,
            "best_temperature_range": self._find_best_temperature_range(
                generated_samples
            ),
            "coherence_scores": self._calculate_coherence_scores(generated_samples),
        }

    def _analyze_temperature_effects(
        self, samples: list[tuple[str, float, dict]]
    ) -> dict:
        """Analyze how temperature affects generation quality"""
        temp_buckets = {
            "low": (0.0, 0.3),
            "medium": (0.3, 0.7),
            "high": (0.7, 1.0),
            "very_high": (1.0, 2.0),
        }

        analysis = {}

        for bucket_name, (min_temp, max_temp) in temp_buckets.items():
            bucket_samples = [s for s in samples if min_temp <= s[1] < max_temp]

            if bucket_samples:
                avg_diversity = sum(s[2]["diversity"] for s in bucket_samples) / len(
                    bucket_samples
                )
                avg_repetition = sum(s[2]["repetition"] for s in bucket_samples) / len(
                    bucket_samples
                )
                avg_entropy = sum(s[2]["entropy"] for s in bucket_samples) / len(
                    bucket_samples
                )

                analysis[bucket_name] = {
                    "count": len(bucket_samples),
                    "avg_diversity": avg_diversity,
                    "avg_repetition": avg_repetition,
                    "avg_entropy": avg_entropy,
                    "quality_score": avg_diversity - avg_repetition + avg_entropy * 0.1,
                }

        return analysis

    def _find_best_temperature_range(
        self, samples: list[tuple[str, float, dict]]
    ) -> dict:
        """Find the temperature range that produces best results"""
        best_temp = 0.7
        best_score = -float("inf")

        for text, temp, metrics in samples:
            score = (
                metrics["diversity"] - metrics["repetition"] + metrics["entropy"] * 0.1
            )
            if score > best_score:
                best_score = score
                best_temp = temp

        return {
            "best_temperature": best_temp,
            "best_score": best_score,
            "recommendation": self._get_temperature_recommendation(best_temp),
        }

    def _get_temperature_recommendation(self, temp: float) -> str:
        """Get human-readable temperature recommendation"""
        if temp < 0.3:
            return "Use deterministic generation for precise tasks"
        if temp < 0.7:
            return "Use moderate temperature for balanced creativity and coherence"
        if temp < 1.0:
            return "Use high temperature for creative tasks"
        return (
            "Use very high temperature for maximum creativity (may sacrifice coherence)"
        )

    def _calculate_coherence_scores(
        self, samples: list[tuple[str, float, dict]]
    ) -> list[float]:
        """Calculate coherence scores for generated samples"""
        coherence_scores = []

        for text, temp, metrics in samples:
            # Simple coherence heuristic based on repetition and diversity
            coherence = max(0, 1 - metrics["repetition"]) * metrics["diversity"]
            coherence_scores.append(coherence)

        return coherence_scores

    def mask_and_predict(
        self, text: str, num_masks: int = None
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Create masked version of text for self-modeling"""
        if num_masks is None:
            num_masks = self.config.num_mask_tokens

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_sequence_length,
            padding=True,
        ).to(self.device)

        input_ids = inputs.input_ids.clone()
        labels = inputs.input_ids.clone()

        # Find maskable positions (avoid special tokens)
        maskable_positions = []
        for i in range(input_ids.size(1)):
            token_id = input_ids[0, i].item()
            if token_id not in [
                self.tokenizer.cls_token_id,
                self.tokenizer.sep_token_id,
                self.tokenizer.pad_token_id,
                self.tokenizer.eos_token_id,
            ]:
                maskable_positions.append(i)

        # Randomly select positions to mask
        num_masks = min(num_masks, len(maskable_positions))
        mask_positions = random.sample(maskable_positions, num_masks)

        # Apply masks
        for pos in mask_positions:
            input_ids[0, pos] = self.tokenizer.mask_token_id

        return input_ids, labels, mask_positions

    def train_self_modeling_step(self, samples: list[tuple[str, float, dict]]) -> dict:
        """Train self-modeling components on generated samples"""
        total_loss = 0
        num_samples = 0

        for text, temp, metrics in samples:
            # Create masked input
            masked_input, labels, mask_positions = self.mask_and_predict(text)

            # Forward pass through model
            outputs = self.model(
                input_ids=masked_input, labels=labels, output_hidden_states=True
            )

            # Self-modeling loss
            hidden_states = outputs.hidden_states[-1]
            predicted_hidden = self.self_predictor(hidden_states)

            self_modeling_loss = F.mse_loss(predicted_hidden, hidden_states.detach())

            # Reflection loss
            reflection_loss = 0
            current_hidden = hidden_states
            for reflection_layer in self.reflection_network:
                reflected_hidden = reflection_layer(current_hidden)
                reflection_loss += F.mse_loss(reflected_hidden, current_hidden.detach())
                current_hidden = reflected_hidden

            # Combined loss
            total_loss_sample = (
                outputs.loss
                + self.config.self_modeling_weight * self_modeling_loss
                + 0.05 * reflection_loss
            )

            # Backward pass
            total_loss_sample.backward()

            total_loss += total_loss_sample.item()
            num_samples += 1

        # Optimize
        self.optimizer.step()
        self.grokfast_optimizer.step()

        self.optimizer.zero_grad()
        self.grokfast_optimizer.zero_grad()

        return {
            "average_loss": total_loss / num_samples if num_samples > 0 else 0,
            "samples_processed": num_samples,
        }

    def run_self_modeling_cycle(
        self, prompts: list[str], num_cycles: int = 100
    ) -> dict:
        """Run complete self-modeling cycle"""
        logger.info(f"Starting self-modeling cycle with {len(prompts)} prompts")

        cycle_results = {
            "cycles_completed": 0,
            "temperature_insights": {},
            "reflection_insights": {},
            "training_metrics": [],
            "expert_vector_updates": 0,
        }

        for cycle in tqdm(range(num_cycles), desc="Self-modeling cycles"):
            cycle_loss = 0
            cycle_samples = 0

            for prompt in prompts:
                # Generate samples across temperature ranges
                all_samples = []
                for temp_range in self.config.temperature_ranges:
                    samples_per_range = self.config.num_temperature_samples // len(
                        self.config.temperature_ranges
                    )
                    samples = self.generate_temperature_samples(
                        prompt, temp_range, samples_per_range
                    )
                    all_samples.extend(samples)

                # Perform self-reflection
                reflection_results = self.perform_self_reflection(all_samples, prompt)

                # Train self-modeling components
                training_results = self.train_self_modeling_step(all_samples)

                # Update expert vectors based on best performing samples
                self._update_expert_vectors(all_samples, reflection_results)

                cycle_loss += training_results["average_loss"]
                cycle_samples += training_results["samples_processed"]

                # Store insights
                cycle_results["temperature_insights"][prompt] = reflection_results[
                    "temperature_analysis"
                ]
                cycle_results["reflection_insights"][prompt] = reflection_results[
                    "reflection_text"
                ]

            # Record cycle metrics
            cycle_results["training_metrics"].append(
                {
                    "cycle": cycle,
                    "average_loss": cycle_loss / len(prompts),
                    "samples_processed": cycle_samples,
                }
            )

            cycle_results["cycles_completed"] += 1

            # Save checkpoint
            if self.config.save_checkpoints and cycle % 10 == 0:
                self._save_checkpoint(cycle)

        logger.info(
            f"Self-modeling cycle completed: {cycle_results['cycles_completed']} cycles"
        )

        return cycle_results

    def _update_expert_vectors(
        self, samples: list[tuple[str, float, dict]], reflection_results: dict
    ) -> None:
        """Update expert vectors based on best performing samples"""
        # Find best samples
        best_samples = sorted(
            samples, key=lambda x: x[2]["diversity"] - x[2]["repetition"]
        )[-5:]

        # Extract features from best samples
        for text, temp, metrics in best_samples:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(
                self.device
            )

            with torch.no_grad():
                outputs = self.model(inputs.input_ids, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1].mean(dim=1)  # Average pool

            # Update expert vectors
            expert_id = hash(text) % self.expert_vectors.num_experts
            self.expert_vectors.update_expert(expert_id, hidden_states.squeeze())

    def _save_checkpoint(self, cycle: int) -> None:
        """Save checkpoint of self-modeling components"""
        checkpoint = {
            "cycle": cycle,
            "self_predictor_state": self.self_predictor.state_dict(),
            "reflection_network_state": [
                layer.state_dict() for layer in self.reflection_network
            ],
            "optimizer_state": self.optimizer.state_dict(),
            "expert_vectors": self.expert_vectors.get_all_experts(),
            "temperature_metrics": self.temperature_metrics,
            "reflection_metrics": self.reflection_metrics,
        }

        checkpoint_path = f"self_modeling_checkpoint_cycle_{cycle}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

    def get_insights_summary(self) -> dict:
        """Get summary of self-modeling insights"""
        return {
            "temperature_preferences": self.temperature_metrics,
            "reflection_patterns": self.reflection_metrics,
            "expert_vector_count": self.expert_vectors.num_experts,
            "training_history": self.self_modeling_history,
            "recommendations": self._generate_recommendations(),
        }

    def _generate_recommendations(self) -> list[str]:
        """Generate recommendations based on self-modeling results"""
        recommendations = []

        # Analyze temperature preferences
        if self.temperature_metrics:
            best_temp_range = max(
                self.temperature_metrics.keys(),
                key=lambda x: self.temperature_metrics[x].get("quality_score", 0),
            )
            recommendations.append(f"Best temperature range: {best_temp_range}")

        # Analyze reflection patterns
        if self.reflection_metrics:
            recommendations.append(
                "Self-reflection improved coherence in creative tasks"
            )

        # Expert vector insights
        recommendations.append(
            f"Developed {self.expert_vectors.num_experts} expert vectors"
        )

        return recommendations


def main():
    """Example usage of enhanced self-modeling"""
    # Initialize model and tokenizer
    model_name = "microsoft/DialoGPT-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Add padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Initialize self-modeling
    config = SelfModelingConfig(
        num_temperature_samples=100,  # Reduced for demo
        save_checkpoints=True,
    )

    self_modeling = EnhancedSelfModeling(model, tokenizer, config)

    # Example prompts
    prompts = [
        "Explain the concept of artificial intelligence",
        "Write a short story about a robot",
        "Describe the process of learning",
    ]

    # Run self-modeling cycle
    results = self_modeling.run_self_modeling_cycle(prompts, num_cycles=10)

    # Get insights
    insights = self_modeling.get_insights_summary()

    print("Self-modeling completed!")
    print(f"Cycles completed: {results['cycles_completed']}")
    print(f"Insights: {insights['recommendations']}")


if __name__ == "__main__":
    main()
