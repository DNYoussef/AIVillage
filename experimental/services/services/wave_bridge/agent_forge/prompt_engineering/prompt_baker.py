"""Prompt Baker - Prepare winning prompts for weight integration
Part B: Agent Forge Phase 4 - Prompt Engineering
"""

from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import statistics
from typing import Any

import numpy as np

import wandb

logger = logging.getLogger(__name__)

@dataclass
class WinningPrompt:
    """A high-performing prompt template ready for deployment"""

    variant_id: str
    template_text: str
    performance_score: float
    confidence_score: float
    interaction_count: int
    statistical_significance: float
    configuration: dict[str, Any]
    optimization_history: list[dict[str, Any]]
    deployment_ready: bool = False
    baked_at: str = ""

@dataclass
class PromptWeights:
    """Optimized weights for prompt parameters"""

    greeting_style_weights: dict[str, float]
    hint_complexity_weights: dict[str, float]
    example_type_weights: dict[str, float]
    encouragement_frequency: float
    response_length_target: str
    subject_specialization: dict[str, float]
    confidence_level: float = 0.0

class PromptBaker:
    """Prepare winning prompts for weight integration and production deployment"""

    def __init__(self, project_name: str = "aivillage-tutoring"):
        self.project_name = project_name
        self.winning_prompts = {}
        self.baked_artifacts = []
        self.optimization_history = defaultdict(list)

        # Performance thresholds for "winning" prompts
        self.min_interactions = 100
        self.min_performance_score = 0.75
        self.min_confidence = 0.85
        self.min_statistical_significance = 0.95

        # Weight optimization parameters
        self.weight_update_rate = 0.1
        self.exploration_decay = 0.95
        self.convergence_threshold = 0.001

        # Initialize W&B API for querying data
        self.initialize_wandb_api()

        # Create directories for baked prompts
        self.setup_prompt_directories()

    def initialize_wandb_api(self):
        """Initialize W&B API for data querying"""
        try:
            self.wandb_api = wandb.Api()
            logger.info("W&B API initialized for prompt baking")
        except Exception as e:
            logger.error(f"Failed to initialize W&B API: {e}")
            self.wandb_api = None

    def setup_prompt_directories(self):
        """Create directory structure for baked prompt artifacts"""
        base_path = Path("services/wave_bridge/agent_forge/baked_prompts")

        directories = [
            base_path / "templates",
            base_path / "weights",
            base_path / "configurations",
            base_path / "deployment"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")

    async def identify_winners(self, min_interactions: int = None) -> list[WinningPrompt]:
        """Find best performing prompt templates from W&B data"""
        min_interactions = min_interactions or self.min_interactions

        if not self.wandb_api:
            logger.error("W&B API not available - cannot identify winners")
            return []

        try:
            # Query W&B for prompt test runs
            runs = self.wandb_api.runs(
                self.project_name,
                filters={
                    "config.job_type": "prompt_testing",
                    "state": "finished"
                }
            )

            winners = []

            for run in runs:
                # Check if run has sufficient data
                total_interactions = run.summary.get("total_interactions", 0)
                if total_interactions < min_interactions:
                    continue

                # Extract performance metrics
                performance_metrics = {
                    "student_engagement": run.summary.get("student_engagement", 0),
                    "response_quality": run.summary.get("response_quality", 0),
                    "response_efficiency": run.summary.get("response_efficiency", 0),
                    "overall_performance": run.summary.get("overall_performance", 0)
                }

                overall_score = performance_metrics["overall_performance"]

                # Check if meets performance threshold
                if overall_score < self.min_performance_score:
                    continue

                # Calculate confidence score based on sample size and consistency
                confidence_score = self.calculate_confidence_score(
                    total_interactions,
                    overall_score,
                    run.summary.get("performance_variance", 0.1)
                )

                if confidence_score < self.min_confidence:
                    continue

                # Extract configuration
                config = run.config
                variant_id = config.get("variant_id", run.id)

                # Get optimization history from run history
                optimization_history = await self.extract_optimization_history(run)

                # Create winning prompt object
                winning_prompt = WinningPrompt(
                    variant_id=variant_id,
                    template_text=config.get("template_text", ""),
                    performance_score=overall_score,
                    confidence_score=confidence_score,
                    interaction_count=total_interactions,
                    statistical_significance=run.summary.get("statistical_significance", 0.9),
                    configuration=dict(config),
                    optimization_history=optimization_history,
                    baked_at=datetime.now(timezone.utc).isoformat()
                )

                winners.append(winning_prompt)

                # Log winner identification
                logger.info(f"Identified winner: {variant_id} (score: {overall_score:.3f}, confidence: {confidence_score:.3f})")

            # Sort by performance score
            winners.sort(key=lambda w: w.performance_score, reverse=True)

            # Version and save winning templates
            for winner in winners[:5]:  # Top 5 winners
                await self.version_winning_template(winner)

            # Log summary to W&B
            wandb.log({
                "winners_identified": len(winners),
                "top_performer_score": winners[0].performance_score if winners else 0,
                "avg_confidence": statistics.mean([w.confidence_score for w in winners]) if winners else 0,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            self.winning_prompts = {w.variant_id: w for w in winners}

            return winners

        except Exception as e:
            logger.error(f"Error identifying winners: {e}")
            return []

    def calculate_confidence_score(self,
                                 sample_size: int,
                                 performance_score: float,
                                 variance: float) -> float:
        """Calculate confidence score based on sample size and performance consistency"""
        # Sample size factor (diminishing returns)
        sample_factor = min(1.0, np.log(sample_size) / np.log(1000))

        # Performance factor
        performance_factor = performance_score

        # Consistency factor (lower variance = higher confidence)
        consistency_factor = max(0.0, 1.0 - variance)

        # Combined confidence score
        confidence = (
            sample_factor * 0.4 +
            performance_factor * 0.4 +
            consistency_factor * 0.2
        )

        return min(1.0, confidence)

    async def extract_optimization_history(self, run) -> list[dict[str, Any]]:
        """Extract optimization history from W&B run"""
        try:
            history = []

            # Get run history (limited to key metrics)
            for row in run.scan_history(keys=["student_engagement", "response_quality", "step"]):
                if row.get("step") is not None:
                    history.append({
                        "step": row["step"],
                        "engagement": row.get("student_engagement", 0),
                        "quality": row.get("response_quality", 0),
                        "timestamp": row.get("_timestamp", 0)
                    })

            return history[-100:]  # Keep last 100 steps

        except Exception as e:
            logger.error(f"Error extracting optimization history: {e}")
            return []

    async def version_winning_template(self, winning_prompt: WinningPrompt):
        """Version winning template as W&B artifact"""
        try:
            # Create artifact
            artifact = wandb.Artifact(
                f"winning_prompt_{winning_prompt.variant_id}",
                type="winning_prompt",
                description=f"High-performing tutoring prompt (score: {winning_prompt.performance_score:.3f})",
                metadata={
                    "performance_score": winning_prompt.performance_score,
                    "confidence_score": winning_prompt.confidence_score,
                    "interaction_count": winning_prompt.interaction_count,
                    "statistical_significance": winning_prompt.statistical_significance,
                    "baked_at": winning_prompt.baked_at,
                    **winning_prompt.configuration
                }
            )

            # Save template content
            template_path = f"services/wave_bridge/agent_forge/baked_prompts/templates/{winning_prompt.variant_id}.txt"
            with open(template_path, "w") as f:
                f.write(winning_prompt.template_text)

            artifact.add_file(template_path)

            # Save configuration
            config_path = f"services/wave_bridge/agent_forge/baked_prompts/configurations/{winning_prompt.variant_id}.json"
            with open(config_path, "w") as f:
                json.dump(asdict(winning_prompt), f, indent=2)

            artifact.add_file(config_path)

            # Log artifact
            wandb.log_artifact(artifact)
            self.baked_artifacts.append(artifact)

            logger.info(f"Versioned winning template: {winning_prompt.variant_id}")

        except Exception as e:
            logger.error(f"Error versioning template {winning_prompt.variant_id}: {e}")

    async def optimize_prompt_weights(self, winners: list[WinningPrompt]) -> PromptWeights:
        """Optimize weights based on winning prompt characteristics"""
        if not winners:
            logger.warning("No winners provided for weight optimization")
            return self.get_default_weights()

        # Analyze winning characteristics
        greeting_styles = defaultdict(list)
        hint_complexities = defaultdict(list)
        example_types = defaultdict(list)
        encouragement_frequencies = []
        subject_specializations = defaultdict(list)

        for winner in winners:
            config = winner.configuration
            score = winner.performance_score

            # Collect weighted samples
            greeting_style = config.get("greeting_style", "friendly")
            greeting_styles[greeting_style].append(score)

            hint_complexity = config.get("hint_complexity", "guided")
            hint_complexities[hint_complexity].append(score)

            example_type = config.get("example_type", "real-world")
            example_types[example_type].append(score)

            encouragement_freq = config.get("encouragement_frequency", 0.3)
            encouragement_frequencies.append(encouragement_freq)

            subject = config.get("subject_expertise", "general")
            subject_specializations[subject].append(score)

        # Calculate optimized weights
        greeting_weights = self.calculate_category_weights(greeting_styles)
        hint_weights = self.calculate_category_weights(hint_complexities)
        example_weights = self.calculate_category_weights(example_types)
        subject_weights = self.calculate_category_weights(subject_specializations)

        # Optimal encouragement frequency (weighted average)
        optimal_encouragement = np.average(
            encouragement_frequencies,
            weights=[w.performance_score for w in winners]
        )

        # Calculate overall confidence
        avg_confidence = statistics.mean([w.confidence_score for w in winners])

        optimized_weights = PromptWeights(
            greeting_style_weights=greeting_weights,
            hint_complexity_weights=hint_weights,
            example_type_weights=example_weights,
            encouragement_frequency=optimal_encouragement,
            response_length_target="moderate",  # Most balanced
            subject_specialization=subject_weights,
            confidence_level=avg_confidence
        )

        # Save weights to file
        await self.save_optimized_weights(optimized_weights)

        # Log optimization results
        wandb.log({
            "weights_optimized": True,
            "greeting_entropy": self.calculate_entropy(greeting_weights),
            "hint_entropy": self.calculate_entropy(hint_weights),
            "example_entropy": self.calculate_entropy(example_weights),
            "optimal_encouragement": optimal_encouragement,
            "confidence_level": avg_confidence,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        logger.info(f"Optimized prompt weights with confidence: {avg_confidence:.3f}")

        return optimized_weights

    def calculate_category_weights(self, category_scores: dict[str, list[float]]) -> dict[str, float]:
        """Calculate normalized weights for a category based on performance scores"""
        if not category_scores:
            return {}

        # Calculate average score for each category
        category_averages = {}
        for category, scores in category_scores.items():
            if scores:
                category_averages[category] = statistics.mean(scores)

        if not category_averages:
            return {}

        # Normalize to probabilities (softmax-like)
        total_score = sum(category_averages.values())
        if total_score == 0:
            # Equal weights if all scores are 0
            return {cat: 1.0 / len(category_averages) for cat in category_averages}

        weights = {
            category: score / total_score
            for category, score in category_averages.items()
        }

        return weights

    def calculate_entropy(self, weights: dict[str, float]) -> float:
        """Calculate entropy of weight distribution (higher = more diverse)"""
        if not weights:
            return 0.0

        values = list(weights.values())
        if sum(values) == 0:
            return 0.0

        # Normalize
        total = sum(values)
        probs = [v / total for v in values]

        # Calculate entropy
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)

        return entropy

    def get_default_weights(self) -> PromptWeights:
        """Get default weights when no winners are available"""
        return PromptWeights(
            greeting_style_weights={"friendly": 0.4, "encouraging": 0.3, "playful": 0.2, "formal": 0.1},
            hint_complexity_weights={"guided": 0.5, "socratic": 0.3, "direct": 0.2},
            example_type_weights={"real-world": 0.4, "visual": 0.3, "story-based": 0.2, "abstract": 0.1},
            encouragement_frequency=0.3,
            response_length_target="moderate",
            subject_specialization={"general": 0.4, "mathematics": 0.2, "science": 0.15, "programming": 0.15, "language_arts": 0.05, "history": 0.05},
            confidence_level=0.5
        )

    async def save_optimized_weights(self, weights: PromptWeights):
        """Save optimized weights to file and W&B artifact"""
        try:
            # Save to local file
            weights_path = "services/wave_bridge/agent_forge/baked_prompts/weights/optimized_weights.json"
            with open(weights_path, "w") as f:
                json.dump(asdict(weights), f, indent=2)

            # Create W&B artifact
            artifact = wandb.Artifact(
                "optimized_prompt_weights",
                type="model_weights",
                description=f"Optimized prompt weights (confidence: {weights.confidence_level:.3f})",
                metadata={
                    "confidence_level": weights.confidence_level,
                    "optimization_date": datetime.now(timezone.utc).isoformat(),
                    "encouragement_frequency": weights.encouragement_frequency
                }
            )

            artifact.add_file(weights_path)
            wandb.log_artifact(artifact)

            logger.info(f"Saved optimized weights with confidence: {weights.confidence_level:.3f}")

        except Exception as e:
            logger.error(f"Error saving optimized weights: {e}")

    async def prepare_deployment_package(self,
                                       winners: list[WinningPrompt],
                                       weights: PromptWeights) -> dict[str, Any]:
        """Prepare complete deployment package with winning prompts and weights"""
        deployment_package = {
            "version": "1.0.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "winning_prompts": {
                winner.variant_id: {
                    "template": winner.template_text,
                    "performance_score": winner.performance_score,
                    "confidence_score": winner.confidence_score,
                    "configuration": winner.configuration,
                    "deployment_ready": True
                }
                for winner in winners if winner.performance_score > self.min_performance_score
            },
            "optimized_weights": asdict(weights),
            "deployment_config": {
                "primary_variant": winners[0].variant_id if winners else None,
                "fallback_variants": [w.variant_id for w in winners[1:3]] if len(winners) > 1 else [],
                "weight_update_frequency": "daily",
                "performance_monitoring": True,
                "auto_optimization": True
            },
            "performance_thresholds": {
                "min_engagement": 0.7,
                "max_response_time": 5.0,
                "min_confidence": self.min_confidence,
                "rollback_threshold": 0.6
            }
        }

        # Save deployment package
        deployment_path = "services/wave_bridge/agent_forge/baked_prompts/deployment/deployment_package.json"
        with open(deployment_path, "w") as f:
            json.dump(deployment_package, f, indent=2)

        # Create deployment artifact
        artifact = wandb.Artifact(
            "prompt_deployment_package",
            type="deployment_package",
            description=f"Complete prompt deployment package with {len(winners)} winning variants",
            metadata={
                "winners_count": len(winners),
                "primary_variant": deployment_package["deployment_config"]["primary_variant"],
                "package_version": deployment_package["version"],
                "confidence_level": weights.confidence_level
            }
        )

        artifact.add_file(deployment_path)
        wandb.log_artifact(artifact)

        # Log deployment preparation
        wandb.log({
            "deployment_package_ready": True,
            "winners_included": len(winners),
            "primary_variant": deployment_package["deployment_config"]["primary_variant"],
            "confidence_level": weights.confidence_level,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        logger.info(f"Deployment package prepared with {len(winners)} winning prompts")

        return deployment_package

    async def validate_deployment_readiness(self, winners: list[WinningPrompt]) -> dict[str, Any]:
        """Validate that winning prompts are ready for production deployment"""
        validation_results = {
            "deployment_ready": True,
            "validation_passed": [],
            "validation_failed": [],
            "warnings": [],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        for winner in winners:
            variant_validation = {
                "variant_id": winner.variant_id,
                "checks": {}
            }

            # Check performance threshold
            if winner.performance_score >= self.min_performance_score:
                variant_validation["checks"]["performance"] = "PASS"
            else:
                variant_validation["checks"]["performance"] = "FAIL"
                validation_results["deployment_ready"] = False

            # Check confidence level
            if winner.confidence_score >= self.min_confidence:
                variant_validation["checks"]["confidence"] = "PASS"
            else:
                variant_validation["checks"]["confidence"] = "FAIL"
                validation_results["deployment_ready"] = False

            # Check sample size
            if winner.interaction_count >= self.min_interactions:
                variant_validation["checks"]["sample_size"] = "PASS"
            else:
                variant_validation["checks"]["sample_size"] = "FAIL"
                validation_results["deployment_ready"] = False

            # Check template completeness
            if winner.template_text and len(winner.template_text) > 50:
                variant_validation["checks"]["template"] = "PASS"
            else:
                variant_validation["checks"]["template"] = "FAIL"
                validation_results["deployment_ready"] = False

            # Check for statistical significance
            if winner.statistical_significance >= self.min_statistical_significance:
                variant_validation["checks"]["significance"] = "PASS"
            else:
                variant_validation["checks"]["significance"] = "WARNING"
                validation_results["warnings"].append(
                    f"Variant {winner.variant_id} has low statistical significance"
                )

            # Add to appropriate list
            if all(check == "PASS" for check in variant_validation["checks"].values() if check != "WARNING"):
                validation_results["validation_passed"].append(variant_validation)
            else:
                validation_results["validation_failed"].append(variant_validation)

        # Log validation results
        wandb.log({
            "deployment_validation": True,
            "deployment_ready": validation_results["deployment_ready"],
            "variants_passed": len(validation_results["validation_passed"]),
            "variants_failed": len(validation_results["validation_failed"]),
            "warnings_count": len(validation_results["warnings"]),
            "timestamp": validation_results["timestamp"]
        })

        logger.info(f"Deployment validation: {len(validation_results['validation_passed'])} passed, {len(validation_results['validation_failed'])} failed")

        return validation_results

    async def generate_baking_report(self) -> str:
        """Generate comprehensive prompt baking report"""
        # Identify winners
        winners = await self.identify_winners()

        # Optimize weights
        weights = await self.optimize_prompt_weights(winners)

        # Validate deployment readiness
        validation = await self.validate_deployment_readiness(winners)

        # Prepare deployment package
        deployment_package = await self.prepare_deployment_package(winners, weights)

        # Generate report
        report = f"""
🔥 **Prompt Baking Report**
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}

📊 **Winners Identified: {len(winners)}**
• Top Performer: {winners[0].variant_id if winners else 'None'} (Score: {winners[0].performance_score:.3f})
• Average Confidence: {statistics.mean([w.confidence_score for w in winners]):.3f}
• Total Interactions Analyzed: {sum(w.interaction_count for w in winners)}

⚖️ **Optimized Weights**
• Greeting Styles: {', '.join(f"{k}: {v:.2f}" for k, v in weights.greeting_style_weights.items())}
• Hint Complexity: {', '.join(f"{k}: {v:.2f}" for k, v in weights.hint_complexity_weights.items())}
• Example Types: {', '.join(f"{k}: {v:.2f}" for k, v in weights.example_type_weights.items())}
• Encouragement Frequency: {weights.encouragement_frequency:.2f}
• Overall Confidence: {weights.confidence_level:.3f}

✅ **Deployment Status**
• Ready for Deployment: {'Yes' if validation['deployment_ready'] else 'No'}
• Variants Passed Validation: {len(validation['validation_passed'])}
• Variants Failed Validation: {len(validation['validation_failed'])}
• Warnings: {len(validation['warnings'])}

📦 **Deployment Package**
• Primary Variant: {deployment_package['deployment_config']['primary_variant']}
• Fallback Variants: {len(deployment_package['deployment_config']['fallback_variants'])}
• Package Version: {deployment_package['version']}

🎯 **Next Steps**
{'✅ Deploy to production - all validations passed' if validation['deployment_ready'] else '⚠️ Address validation failures before deployment'}
"""

        return report

# Global instance
prompt_baker = PromptBaker()
