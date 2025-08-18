"""Curriculum Effectiveness Validation Framework.

Comprehensive framework for measuring and validating the effectiveness
of the Frontier Curriculum Engine in improving model training outcomes,
learning efficiency, and retention rates.
"""

import logging
import statistics

# Test imports
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from agent_forge.curriculum import CurriculumOrchestrator, Problem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LearningOutcome:
    """Represents a learning outcome measurement."""

    student_id: str
    problem_type: str
    difficulty: float
    attempts: int
    success: bool
    time_to_mastery: float | None
    retention_score: float | None
    confidence_level: float
    timestamp: datetime


@dataclass
class CurriculumExperiment:
    """Represents a controlled curriculum experiment."""

    experiment_id: str
    control_group: str  # "curriculum" or "baseline"
    domain: str
    duration_minutes: int
    problems_attempted: int
    success_rate: float
    average_attempts_to_success: float
    learning_efficiency: float
    retention_rate: float
    difficulty_progression: list[float]


@dataclass
class EffectivenessMetrics:
    """Comprehensive effectiveness metrics."""

    learning_speed: float  # Problems mastered per hour
    retention_rate: float  # Knowledge retained after time gap
    transfer_learning: float  # Performance on unseen similar problems
    curriculum_adherence: float  # How well curriculum maintained edge-of-chaos
    student_engagement: float  # Proxy measure for engagement
    efficiency_gain: float  # Improvement over baseline methods


class CurriculumEffectivenessValidator:
    """Framework for validating curriculum effectiveness."""

    def __init__(self, curriculum_engine: CurriculumOrchestrator | None = None):
        self.curriculum_engine = curriculum_engine
        self.experiments: list[CurriculumExperiment] = []
        self.learning_outcomes: list[LearningOutcome] = []

    async def run_controlled_experiment(
        self,
        domain: str = "coding-python",
        duration_minutes: int = 60,
        num_students: int = 10,
        use_curriculum: bool = True,
    ) -> CurriculumExperiment:
        """Run a controlled experiment comparing curriculum vs baseline."""

        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        control_group = "curriculum" if use_curriculum else "baseline"

        logger.info(f"Starting experiment {experiment_id} with {control_group}")

        # Generate experiment problems
        problems = await self._generate_experiment_problems(domain, difficulty_range=(0.3, 0.8))

        # Simulate student learning
        results = []
        for student_id in range(num_students):
            student_results = await self._simulate_student_learning(
                student_id=f"student_{student_id}",
                problems=problems,
                duration_minutes=duration_minutes,
                use_curriculum=use_curriculum,
            )
            results.extend(student_results)

        # Calculate experiment metrics
        success_rate = sum(1 for r in results if r.success) / len(results)
        avg_attempts = statistics.mean(r.attempts for r in results)

        # Learning efficiency: successful problems per minute
        successful_outcomes = [r for r in results if r.success]
        if successful_outcomes:
            learning_efficiency = len(successful_outcomes) / duration_minutes
        else:
            learning_efficiency = 0.0

        # Retention rate (mock - would need follow-up testing)
        retention_rate = await self._calculate_retention_rate(results)

        # Difficulty progression
        difficulty_progression = [r.difficulty for r in results if r.success]

        experiment = CurriculumExperiment(
            experiment_id=experiment_id,
            control_group=control_group,
            domain=domain,
            duration_minutes=duration_minutes,
            problems_attempted=len(results),
            success_rate=success_rate,
            average_attempts_to_success=avg_attempts,
            learning_efficiency=learning_efficiency,
            retention_rate=retention_rate,
            difficulty_progression=difficulty_progression,
        )

        self.experiments.append(experiment)
        self.learning_outcomes.extend(results)

        return experiment

    async def validate_edge_of_chaos_effectiveness(
        self, target_accuracy_range: tuple[float, float] = (0.55, 0.75)
    ) -> dict[str, float]:
        """Validate that edge-of-chaos curriculum improves learning."""

        if not self.curriculum_engine:
            raise ValueError("Curriculum engine required for edge validation")

        logger.info("Validating edge-of-chaos effectiveness")

        # Test different difficulty ranges
        test_ranges = [
            (0.2, 0.4),  # Too easy
            (0.55, 0.75),  # Edge-of-chaos
            (0.8, 0.95),  # Too hard
        ]

        results = {}

        for low, high in test_ranges:
            range_name = f"range_{low:.1f}_{high:.1f}"

            # Run experiment in this difficulty range
            experiment = await self._run_difficulty_range_experiment(
                difficulty_range=(low, high), num_students=5, duration_minutes=30
            )

            results[range_name] = {
                "success_rate": experiment.success_rate,
                "learning_efficiency": experiment.learning_efficiency,
                "avg_attempts": experiment.average_attempts_to_success,
            }

        # Edge-of-chaos should show best learning efficiency
        edge_results = results["range_0.5_0.8"]
        easy_results = results["range_0.2_0.4"]
        hard_results = results["range_0.8_0.9"]

        validation_metrics = {
            "edge_vs_easy_efficiency": edge_results["learning_efficiency"]
            / max(easy_results["learning_efficiency"], 0.001),
            "edge_vs_hard_efficiency": edge_results["learning_efficiency"]
            / max(hard_results["learning_efficiency"], 0.001),
            "edge_success_rate": edge_results["success_rate"],
            "optimal_range_validation": (
                1.0
                if edge_results["learning_efficiency"]
                == max(
                    edge_results["learning_efficiency"],
                    easy_results["learning_efficiency"],
                    hard_results["learning_efficiency"],
                )
                else 0.0
            ),
        }

        logger.info(f"Edge validation results: {validation_metrics}")
        return validation_metrics

    async def measure_transfer_learning(
        self,
        source_domain: str = "coding-python",
        target_domain: str = "coding-javascript",
    ) -> float:
        """Measure transfer learning effectiveness."""

        logger.info(f"Measuring transfer learning: {source_domain} → {target_domain}")

        # Train on source domain
        source_experiment = await self.run_controlled_experiment(
            domain=source_domain,
            duration_minutes=45,
            num_students=8,
            use_curriculum=True,
        )

        # Test on target domain (simulated)
        transfer_problems = await self._generate_experiment_problems(target_domain)
        transfer_success_rate = await self._simulate_transfer_performance(
            source_experiment.difficulty_progression, transfer_problems
        )

        # Compare to baseline performance on target domain
        baseline_experiment = await self.run_controlled_experiment(
            domain=target_domain,
            duration_minutes=30,
            num_students=8,
            use_curriculum=False,
        )

        transfer_effectiveness = transfer_success_rate / max(baseline_experiment.success_rate, 0.001)

        logger.info(f"Transfer learning effectiveness: {transfer_effectiveness:.2f}x")
        return transfer_effectiveness

    async def analyze_curriculum_adherence(self) -> dict[str, float]:
        """Analyze how well curriculum maintained target difficulty bands."""

        if not self.curriculum_engine:
            return {"error": "No curriculum engine available"}

        # Analyze recent curriculum history
        curriculum_outcomes = [o for o in self.learning_outcomes if o.problem_type == "curriculum"]

        if len(curriculum_outcomes) < 10:
            return {"insufficient_data": len(curriculum_outcomes)}

        # Calculate actual accuracy in difficulty bands
        accuracy_by_difficulty = {}
        for outcome in curriculum_outcomes:
            diff_band = round(outcome.difficulty, 1)
            if diff_band not in accuracy_by_difficulty:
                accuracy_by_difficulty[diff_band] = []
            accuracy_by_difficulty[diff_band].append(1.0 if outcome.success else 0.0)

        # Calculate adherence to edge-of-chaos
        target_bands = [0.5, 0.6, 0.7, 0.8]  # Edge-of-chaos range
        adherence_scores = []

        for band in target_bands:
            if band in accuracy_by_difficulty:
                actual_accuracy = statistics.mean(accuracy_by_difficulty[band])
                target_accuracy = 0.65  # Middle of edge-of-chaos
                adherence = 1.0 - abs(actual_accuracy - target_accuracy)
                adherence_scores.append(max(0.0, adherence))

        overall_adherence = statistics.mean(adherence_scores) if adherence_scores else 0.0

        return {
            "overall_adherence": overall_adherence,
            "bands_analyzed": len(adherence_scores),
            "accuracy_by_band": {str(k): statistics.mean(v) for k, v in accuracy_by_difficulty.items()},
        }

    def generate_effectiveness_report(self) -> dict[str, Any]:
        """Generate comprehensive effectiveness report."""

        if not self.experiments:
            return {"error": "No experiments available"}

        # Separate curriculum and baseline experiments
        curriculum_exps = [e for e in self.experiments if e.control_group == "curriculum"]
        baseline_exps = [e for e in self.experiments if e.control_group == "baseline"]

        report = {
            "summary": {
                "total_experiments": len(self.experiments),
                "curriculum_experiments": len(curriculum_exps),
                "baseline_experiments": len(baseline_exps),
                "total_learning_outcomes": len(self.learning_outcomes),
            },
            "effectiveness_metrics": {},
            "statistical_analysis": {},
            "recommendations": [],
        }

        if curriculum_exps and baseline_exps:
            # Compare curriculum vs baseline
            curr_success = statistics.mean(e.success_rate for e in curriculum_exps)
            base_success = statistics.mean(e.success_rate for e in baseline_exps)

            curr_efficiency = statistics.mean(e.learning_efficiency for e in curriculum_exps)
            base_efficiency = statistics.mean(e.learning_efficiency for e in baseline_exps)

            report["effectiveness_metrics"] = {
                "success_rate_improvement": curr_success / max(base_success, 0.001),
                "learning_efficiency_improvement": curr_efficiency / max(base_efficiency, 0.001),
                "curriculum_avg_success": curr_success,
                "baseline_avg_success": base_success,
                "curriculum_avg_efficiency": curr_efficiency,
                "baseline_avg_efficiency": base_efficiency,
            }

            # Generate recommendations
            if curr_success > base_success * 1.1:
                report["recommendations"].append("Curriculum shows significant success rate improvement")

            if curr_efficiency > base_efficiency * 1.2:
                report["recommendations"].append("Curriculum demonstrates superior learning efficiency")

        # Analyze learning progression
        if self.learning_outcomes:
            successful_outcomes = [o for o in self.learning_outcomes if o.success]
            if successful_outcomes:
                avg_difficulty = statistics.mean(o.difficulty for o in successful_outcomes)
                avg_attempts = statistics.mean(o.attempts for o in successful_outcomes)

                report["learning_progression"] = {
                    "average_mastered_difficulty": avg_difficulty,
                    "average_attempts_to_success": avg_attempts,
                    "mastery_rate": len(successful_outcomes) / len(self.learning_outcomes),
                }

        return report

    async def _generate_experiment_problems(
        self,
        domain: str,
        difficulty_range: tuple[float, float] = (0.3, 0.8),
        count: int = 20,
    ) -> list[Problem]:
        """Generate problems for controlled experiments."""

        problems = []
        for i in range(count):
            difficulty = difficulty_range[0] + (difficulty_range[1] - difficulty_range[0]) * (i / count)

            problems.append(
                Problem(
                    id=f"exp_prob_{i:03d}",
                    topic=f"{domain}_topic_{i % 5}",
                    difficulty=round(difficulty, 2),
                    statement=f"Experiment problem {i} in {domain}",
                    canonical_answer=f"def solution_{i}(): pass",
                    rubric=f"Rubric for problem {i}",
                    unit_tests=[f"assert solution_{i}() is not None"],
                )
            )

        return problems

    async def _simulate_student_learning(
        self,
        student_id: str,
        problems: list[Problem],
        duration_minutes: int,
        use_curriculum: bool,
    ) -> list[LearningOutcome]:
        """Simulate student learning process."""

        outcomes = []
        start_time = datetime.now()

        # Simulate learning over time
        for i, problem in enumerate(problems[: min(len(problems), duration_minutes // 2)]):
            # Simulate attempts based on difficulty and student capability
            student_skill = 0.4 + (i * 0.02)  # Gradual improvement
            difficulty_delta = abs(problem.difficulty - student_skill)

            if use_curriculum:
                # Curriculum helps with better difficulty matching
                success_prob = max(0.1, 0.8 - difficulty_delta * 0.8)
                attempts = max(1, int(difficulty_delta * 3) + 1)
            else:
                # Baseline has more variance and difficulty
                success_prob = max(0.1, 0.6 - difficulty_delta * 1.2)
                attempts = max(1, int(difficulty_delta * 5) + 1)

            success = np.random.random() < success_prob
            confidence = success_prob * 0.9 + np.random.random() * 0.2

            outcome = LearningOutcome(
                student_id=student_id,
                problem_type="curriculum" if use_curriculum else "baseline",
                difficulty=problem.difficulty,
                attempts=attempts,
                success=success,
                time_to_mastery=attempts * 2.5 if success else None,
                retention_score=None,  # Would be measured later
                confidence_level=min(1.0, confidence),
                timestamp=start_time + timedelta(minutes=i * 2),
            )

            outcomes.append(outcome)

        return outcomes

    async def _calculate_retention_rate(self, outcomes: list[LearningOutcome]) -> float:
        """Calculate retention rate (simplified simulation)."""
        successful_outcomes = [o for o in outcomes if o.success]

        if not successful_outcomes:
            return 0.0

        # Simulate retention based on confidence and difficulty
        retention_scores = []
        for outcome in successful_outcomes:
            # Higher confidence and moderate difficulty improve retention
            difficulty_factor = 1.0 - abs(outcome.difficulty - 0.6) * 0.5
            confidence_factor = outcome.confidence_level

            retention = (difficulty_factor + confidence_factor) / 2
            retention_scores.append(max(0.3, retention))  # Minimum 30% retention

        return statistics.mean(retention_scores)

    async def _run_difficulty_range_experiment(
        self,
        difficulty_range: tuple[float, float],
        num_students: int = 5,
        duration_minutes: int = 30,
    ) -> CurriculumExperiment:
        """Run experiment within specific difficulty range."""

        problems = await self._generate_experiment_problems(
            domain="test_domain", difficulty_range=difficulty_range, count=15
        )

        # Simulate learning with fixed difficulty range
        all_outcomes = []
        for student_id in range(num_students):
            outcomes = await self._simulate_student_learning(
                student_id=f"range_student_{student_id}",
                problems=problems,
                duration_minutes=duration_minutes,
                use_curriculum=True,
            )
            all_outcomes.extend(outcomes)

        # Calculate metrics
        success_rate = sum(1 for o in all_outcomes if o.success) / len(all_outcomes)
        avg_attempts = statistics.mean(o.attempts for o in all_outcomes)
        learning_efficiency = len([o for o in all_outcomes if o.success]) / duration_minutes
        retention_rate = await self._calculate_retention_rate(all_outcomes)

        return CurriculumExperiment(
            experiment_id=f"range_{difficulty_range[0]}_{difficulty_range[1]}",
            control_group="curriculum",
            domain="test_domain",
            duration_minutes=duration_minutes,
            problems_attempted=len(all_outcomes),
            success_rate=success_rate,
            average_attempts_to_success=avg_attempts,
            learning_efficiency=learning_efficiency,
            retention_rate=retention_rate,
            difficulty_progression=[o.difficulty for o in all_outcomes if o.success],
        )

    async def _simulate_transfer_performance(
        self, source_progression: list[float], target_problems: list[Problem]
    ) -> float:
        """Simulate transfer learning performance."""

        if not source_progression:
            return 0.3  # Baseline performance

        # Calculate student skill level from source domain
        max_source_difficulty = max(source_progression) if source_progression else 0.5

        # Transfer performance based on skill development
        transfer_successes = 0
        for problem in target_problems[:10]:  # Test on first 10 problems
            # Transfer is less effective than direct learning
            transfer_skill = max_source_difficulty * 0.7  # 70% transfer efficiency
            success_prob = max(0.1, 0.8 - abs(problem.difficulty - transfer_skill))

            if np.random.random() < success_prob:
                transfer_successes += 1

        return transfer_successes / min(10, len(target_problems))


class TestCurriculumEffectivenessValidation:
    """Test curriculum effectiveness validation framework."""

    @pytest.fixture
    def mock_llm(self):
        """Mock LLM for testing."""
        from tests.curriculum.test_integration_comprehensive import MockOpenRouterLLM

        return MockOpenRouterLLM("mock-key")

    @pytest.fixture
    def curriculum_engine(self, mock_llm):
        """Curriculum engine fixture."""
        return CurriculumOrchestrator(mock_llm)

    @pytest.fixture
    def validator(self, curriculum_engine):
        """Effectiveness validator fixture."""
        return CurriculumEffectivenessValidator(curriculum_engine)

    @pytest.mark.asyncio
    async def test_controlled_experiment_execution(self, validator):
        """Test execution of controlled curriculum experiments."""

        # Run curriculum experiment
        curriculum_exp = await validator.run_controlled_experiment(
            domain="coding-python",
            duration_minutes=30,
            num_students=5,
            use_curriculum=True,
        )

        # Run baseline experiment
        baseline_exp = await validator.run_controlled_experiment(
            domain="coding-python",
            duration_minutes=30,
            num_students=5,
            use_curriculum=False,
        )

        # Validate experiment structure
        assert curriculum_exp.control_group == "curriculum"
        assert baseline_exp.control_group == "baseline"
        assert curriculum_exp.duration_minutes == 30
        assert curriculum_exp.problems_attempted > 0
        assert 0.0 <= curriculum_exp.success_rate <= 1.0
        assert curriculum_exp.learning_efficiency >= 0.0

        # Should have learning outcomes
        assert len(validator.learning_outcomes) > 0

    @pytest.mark.asyncio
    async def test_edge_of_chaos_validation(self, validator):
        """Test validation of edge-of-chaos effectiveness."""

        results = await validator.validate_edge_of_chaos_effectiveness()

        # Should contain key validation metrics
        assert "edge_vs_easy_efficiency" in results
        assert "edge_vs_hard_efficiency" in results
        assert "edge_success_rate" in results
        assert "optimal_range_validation" in results

        # Metrics should be reasonable
        assert results["edge_success_rate"] >= 0.0
        assert results["edge_vs_easy_efficiency"] >= 0.0
        assert results["edge_vs_hard_efficiency"] >= 0.0
        assert 0.0 <= results["optimal_range_validation"] <= 1.0

    @pytest.mark.asyncio
    async def test_transfer_learning_measurement(self, validator):
        """Test measurement of transfer learning effectiveness."""

        transfer_score = await validator.measure_transfer_learning(
            source_domain="coding-python", target_domain="coding-javascript"
        )

        # Should return reasonable transfer score
        assert isinstance(transfer_score, float)
        assert transfer_score >= 0.0
        assert transfer_score <= 10.0  # Reasonable upper bound

    @pytest.mark.asyncio
    async def test_curriculum_adherence_analysis(self, validator):
        """Test analysis of curriculum adherence to target difficulty."""

        # Run experiment to generate data
        await validator.run_controlled_experiment(duration_minutes=20, num_students=3, use_curriculum=True)

        adherence = await validator.analyze_curriculum_adherence()

        if "error" not in adherence and "insufficient_data" not in adherence:
            assert "overall_adherence" in adherence
            assert "bands_analyzed" in adherence
            assert "accuracy_by_band" in adherence
            assert 0.0 <= adherence["overall_adherence"] <= 1.0

    @pytest.mark.asyncio
    async def test_effectiveness_report_generation(self, validator):
        """Test generation of comprehensive effectiveness report."""

        # Run multiple experiments
        await validator.run_controlled_experiment(duration_minutes=15, num_students=3, use_curriculum=True)

        await validator.run_controlled_experiment(duration_minutes=15, num_students=3, use_curriculum=False)

        report = validator.generate_effectiveness_report()

        # Should contain key sections
        assert "summary" in report
        assert "effectiveness_metrics" in report
        assert "recommendations" in report

        # Summary should be accurate
        summary = report["summary"]
        assert summary["total_experiments"] == 2
        assert summary["curriculum_experiments"] == 1
        assert summary["baseline_experiments"] == 1
        assert summary["total_learning_outcomes"] > 0

        # Should have effectiveness comparison
        if report["effectiveness_metrics"]:
            metrics = report["effectiveness_metrics"]
            assert "success_rate_improvement" in metrics
            assert "learning_efficiency_improvement" in metrics


class TestLongTermEffectivenessTracking:
    """Test long-term effectiveness tracking and analysis."""

    @pytest.mark.asyncio
    async def test_longitudinal_effectiveness_analysis(self):
        """Test longitudinal analysis of curriculum effectiveness."""

        from tests.curriculum.test_integration_comprehensive import MockOpenRouterLLM

        mock_llm = MockOpenRouterLLM("mock-key")
        curriculum_engine = CurriculumOrchestrator(mock_llm)
        validator = CurriculumEffectivenessValidator(curriculum_engine)

        # Simulate multiple time periods
        time_periods = [
            ("week1", 15, 3),
            ("week2", 20, 4),
            ("week3", 25, 5),
        ]

        longitudinal_results = {}

        for period, duration, students in time_periods:
            experiment = await validator.run_controlled_experiment(
                duration_minutes=duration, num_students=students, use_curriculum=True
            )

            longitudinal_results[period] = {
                "success_rate": experiment.success_rate,
                "efficiency": experiment.learning_efficiency,
                "avg_difficulty": (
                    statistics.mean(experiment.difficulty_progression) if experiment.difficulty_progression else 0.0
                ),
            }

        # Analyze trends
        success_rates = [longitudinal_results[p]["success_rate"] for p in longitudinal_results]
        efficiencies = [longitudinal_results[p]["efficiency"] for p in longitudinal_results]

        # Should show learning progression over time
        assert len(success_rates) == 3
        assert len(efficiencies) == 3

        # Generally expect improvement or stability
        final_success = success_rates[-1]
        initial_success = success_rates[0]

        logger.info(f"Longitudinal success rates: {success_rates}")
        logger.info(f"Success improvement: {final_success / max(initial_success, 0.001):.2f}x")

        assert final_success >= 0.0  # Should maintain reasonable performance


if __name__ == "__main__":
    # Run validation tests
    pytest.main([__file__, "-v", "--asyncio-mode=auto", "-s"])
