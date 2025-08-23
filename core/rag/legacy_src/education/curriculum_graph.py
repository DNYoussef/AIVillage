"""Hypergraph-based Curriculum Graph Implementation - Refactored
Sprint R-4+AF1: Education Core System - Task A.1.

This refactored version breaks down the monolithic CurriculumGraph into
focused, manageable components following single responsibility principle.
"""

import asyncio
import hashlib
import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import wandb

# Import from hyperag components
from ..core.hypergraph_kg import Hyperedge, HypergraphKG, Node

logger = logging.getLogger(__name__)


@dataclass
class ConceptNode:
    """Educational concept with metadata."""

    concept_id: str
    name: str
    subject: str
    grade: int
    difficulty_level: float
    content: dict[str, str]  # language -> content
    learning_objectives: list[str]
    assessment_criteria: list[str]
    estimated_time_minutes: int
    prerequisites: list[str]
    follow_up_concepts: list[str]
    cultural_adaptations: dict[str, Any]  # region -> adaptations
    created_at: str = ""
    updated_at: str = ""


@dataclass
class LearningPath:
    """Sequence of concepts forming a learning path."""

    path_id: str
    name: str
    subject: str
    grade_range: tuple[int, int]
    concepts: list[str]  # concept_ids in order
    estimated_duration_hours: float
    difficulty_progression: list[float]
    cultural_region: str
    language: str
    completion_rate: float = 0.0
    effectiveness_score: float = 0.0


class ConceptManager:
    """Manages educational concepts and their relationships."""

    def __init__(self):
        self.concepts = {}  # concept_id -> ConceptNode
        self.subject_taxonomies = {}  # subject -> hierarchy

    def generate_concept_id(self, subject: str, grade: int, concept: str) -> str:
        """Generate unique concept identifier."""
        combined = f"{subject}_{grade}_{concept}".lower()
        return hashlib.md5(combined.encode()).hexdigest()[:12]

    def estimate_difficulty(self, prerequisites: list[str], grade: int) -> float:
        """Estimate concept difficulty based on prerequisites and grade."""
        base_difficulty = grade * 0.1
        prerequisite_difficulty = len(prerequisites) * 0.05
        difficulty = min(base_difficulty + prerequisite_difficulty, 1.0)

        adjustment_factors = {"mathematics": 1.2, "science": 1.1, "language_arts": 0.9, "social_studies": 0.8}

        return min(difficulty, 1.0)

    def generate_assessment_criteria(self, concept: str, grade: int) -> list[str]:
        """Generate age-appropriate assessment criteria."""
        base_criteria = [
            f"Student can explain the concept of {concept}",
            f"Student can apply {concept} to solve basic problems",
            f"Student can identify examples of {concept} in real situations",
        ]

        if grade >= 3:
            base_criteria.extend(
                [
                    f"Student can compare {concept} with related concepts",
                    f"Student can create original examples using {concept}",
                ]
            )

        if grade >= 6:
            base_criteria.extend(
                [
                    f"Student can analyze the impact of {concept}",
                    f"Student can evaluate different approaches to {concept}",
                ]
            )

        return base_criteria

    def add_concept(self, concept: ConceptNode) -> bool:
        """Add a new concept to the curriculum."""
        if concept.concept_id in self.concepts:
            logger.warning(f"Concept {concept.concept_id} already exists")
            return False

        self.concepts[concept.concept_id] = concept

        # Update subject taxonomy
        if concept.subject not in self.subject_taxonomies:
            self.subject_taxonomies[concept.subject] = defaultdict(list)

        self.subject_taxonomies[concept.subject][concept.grade].append(concept.concept_id)

        logger.info(f"Added concept: {concept.name} ({concept.concept_id})")
        return True

    def get_concept(self, concept_id: str) -> ConceptNode | None:
        """Retrieve a concept by ID."""
        return self.concepts.get(concept_id)

    def get_concepts_by_subject(self, subject: str, grade: int | None = None) -> list[ConceptNode]:
        """Get all concepts for a subject, optionally filtered by grade."""
        concepts = [c for c in self.concepts.values() if c.subject == subject]
        if grade is not None:
            concepts = [c for c in concepts if c.grade == grade]
        return concepts


class PathBuilder:
    """Builds and manages learning paths."""

    def __init__(self, concept_manager: ConceptManager):
        self.concept_manager = concept_manager
        self.learning_paths = {}  # path_id -> LearningPath

    def create_learning_path(
        self, subject: str, grade_range: tuple[int, int], cultural_region: str = "global", language: str = "en"
    ) -> str:
        """Create an optimal learning path for given parameters."""
        path_id = self._generate_path_id(subject, grade_range, cultural_region, language)

        # Get relevant concepts
        concepts = []
        for grade in range(grade_range[0], grade_range[1] + 1):
            grade_concepts = self.concept_manager.get_concepts_by_subject(subject, grade)
            concepts.extend(grade_concepts)

        # Sort by prerequisites and difficulty
        ordered_concepts = self._order_concepts_by_dependencies(concepts)

        # Calculate path metrics
        total_duration = sum(c.estimated_time_minutes for c in ordered_concepts) / 60
        difficulty_progression = [c.difficulty_level for c in ordered_concepts]

        learning_path = LearningPath(
            path_id=path_id,
            name=f"{subject.title()} ({grade_range[0]}-{grade_range[1]})",
            subject=subject,
            grade_range=grade_range,
            concepts=[c.concept_id for c in ordered_concepts],
            estimated_duration_hours=total_duration,
            difficulty_progression=difficulty_progression,
            cultural_region=cultural_region,
            language=language,
        )

        self.learning_paths[path_id] = learning_path
        logger.info(f"Created learning path: {learning_path.name}")
        return path_id

    def _generate_path_id(self, subject: str, grade_range: tuple[int, int], cultural_region: str, language: str) -> str:
        """Generate unique path identifier."""
        path_string = f"{subject}_{grade_range[0]}_{grade_range[1]}_{cultural_region}_{language}"
        return hashlib.md5(path_string.encode()).hexdigest()[:12]

    def _order_concepts_by_dependencies(self, concepts: list[ConceptNode]) -> list[ConceptNode]:
        """Order concepts based on prerequisite dependencies."""
        concept_map = {c.concept_id: c for c in concepts}
        visited = set()
        ordered = []

        def visit(concept_id: str):
            if concept_id in visited or concept_id not in concept_map:
                return

            concept = concept_map[concept_id]

            # Visit prerequisites first
            for prereq in concept.prerequisites:
                if prereq in concept_map:
                    visit(prereq)

            visited.add(concept_id)
            ordered.append(concept)

        # Visit all concepts
        for concept in concepts:
            visit(concept.concept_id)

        return ordered

    def optimize_path_difficulty(self, path_id: str) -> bool:
        """Optimize the difficulty progression of a learning path."""
        path = self.learning_paths.get(path_id)
        if not path:
            return False

        # Smooth difficulty progression
        optimized_concepts = []
        current_difficulty = 0.0

        for concept_id in path.concepts:
            concept = self.concept_manager.get_concept(concept_id)
            if concept:
                # Ensure gradual difficulty increase
                if concept.difficulty_level > current_difficulty + 0.15:
                    # Insert intermediate concepts if needed
                    intermediate = self._find_intermediate_concept(
                        current_difficulty, concept.difficulty_level, concept.subject, concept.grade
                    )
                    if intermediate:
                        optimized_concepts.append(intermediate.concept_id)
                        current_difficulty = intermediate.difficulty_level

                optimized_concepts.append(concept_id)
                current_difficulty = concept.difficulty_level

        path.concepts = optimized_concepts
        path.difficulty_progression = [
            self.concept_manager.get_concept(cid).difficulty_level
            for cid in optimized_concepts
            if self.concept_manager.get_concept(cid)
        ]

        return True

    def _find_intermediate_concept(
        self, current_diff: float, target_diff: float, subject: str, grade: int
    ) -> ConceptNode | None:
        """Find an intermediate concept to bridge difficulty gap."""
        target_difficulty = (current_diff + target_diff) / 2

        subject_concepts = self.concept_manager.get_concepts_by_subject(subject, grade)

        best_match = None
        best_difference = float("inf")

        for concept in subject_concepts:
            diff = abs(concept.difficulty_level - target_difficulty)
            if diff < best_difference:
                best_difference = diff
                best_match = concept

        return best_match if best_difference < 0.1 else None


class GraphAnalyzer:
    """Analyzes curriculum graph patterns and relationships."""

    def __init__(self, graph: HypergraphKG, concept_manager: ConceptManager):
        self.graph = graph
        self.concept_manager = concept_manager
        self.concept_effectiveness = defaultdict(float)

    def analyze_concept_dependencies(self) -> dict[str, list[str]]:
        """Analyze dependency patterns across concepts."""
        dependencies = {}

        for concept_id, concept in self.concept_manager.concepts.items():
            dependencies[concept_id] = {
                "prerequisites": concept.prerequisites,
                "dependents": [
                    cid for cid, c in self.concept_manager.concepts.items() if concept_id in c.prerequisites
                ],
                "grade_level": concept.grade,
                "difficulty": concept.difficulty_level,
            }

        return dependencies

    def find_learning_bottlenecks(self) -> list[dict[str, Any]]:
        """Identify concepts that create learning bottlenecks."""
        dependencies = self.analyze_concept_dependencies()
        bottlenecks = []

        for concept_id, deps in dependencies.items():
            if len(deps["dependents"]) > 5:  # High fan-out
                concept = self.concept_manager.get_concept(concept_id)
                if concept and concept.difficulty_level > 0.7:  # High difficulty
                    bottlenecks.append(
                        {
                            "concept_id": concept_id,
                            "concept_name": concept.name,
                            "dependent_count": len(deps["dependents"]),
                            "difficulty": concept.difficulty_level,
                            "grade": concept.grade,
                            "bottleneck_severity": len(deps["dependents"]) * concept.difficulty_level,
                        }
                    )

        return sorted(bottlenecks, key=lambda x: x["bottleneck_severity"], reverse=True)

    def calculate_concept_effectiveness(
        self, concept_id: str, completion_rates: list[float], assessment_scores: list[float]
    ) -> float:
        """Calculate effectiveness score for a concept."""
        if not completion_rates or not assessment_scores:
            return 0.0

        avg_completion = sum(completion_rates) / len(completion_rates)
        avg_assessment = sum(assessment_scores) / len(assessment_scores)

        # Weight completion rate and assessment scores
        effectiveness = (avg_completion * 0.4) + (avg_assessment * 0.6)

        self.concept_effectiveness[concept_id] = effectiveness
        return effectiveness

    def recommend_curriculum_improvements(self) -> list[dict[str, Any]]:
        """Generate recommendations for curriculum improvements."""
        recommendations = []

        # Find low-effectiveness concepts
        low_effectiveness = [(cid, score) for cid, score in self.concept_effectiveness.items() if score < 0.6]

        for concept_id, score in low_effectiveness:
            concept = self.concept_manager.get_concept(concept_id)
            if concept:
                recommendations.append(
                    {
                        "type": "low_effectiveness",
                        "concept_id": concept_id,
                        "concept_name": concept.name,
                        "current_score": score,
                        "recommendation": f"Review and improve content for {concept.name}",
                        "suggested_actions": [
                            "Add more interactive examples",
                            "Reduce cognitive load",
                            "Improve prerequisite preparation",
                            "Add cultural relevance",
                        ],
                    }
                )

        # Find bottlenecks
        bottlenecks = self.find_learning_bottlenecks()
        for bottleneck in bottlenecks[:5]:  # Top 5 bottlenecks
            recommendations.append(
                {
                    "type": "bottleneck",
                    "concept_id": bottleneck["concept_id"],
                    "concept_name": bottleneck["concept_name"],
                    "severity": bottleneck["bottleneck_severity"],
                    "recommendation": f"Address learning bottleneck in {bottleneck['concept_name']}",
                    "suggested_actions": [
                        "Break into smaller concepts",
                        "Add alternative learning paths",
                        "Provide additional practice materials",
                        "Create prerequisite bridge concepts",
                    ],
                }
            )

        return recommendations


class CurriculumOptimizer:
    """Optimizes curriculum using machine learning and analytics."""

    def __init__(self, path_builder: PathBuilder, graph_analyzer: GraphAnalyzer):
        self.path_builder = path_builder
        self.graph_analyzer = graph_analyzer
        self.learning_analytics = defaultdict(list)
        self.cultural_examples = defaultdict(dict)
        self.regional_contexts = {}

    def optimize_for_cultural_context(self, path_id: str, cultural_region: str) -> bool:
        """Optimize a learning path for specific cultural context."""
        path = self.path_builder.learning_paths.get(path_id)
        if not path:
            return False

        # Apply cultural adaptations
        for concept_id in path.concepts:
            concept = self.path_builder.concept_manager.get_concept(concept_id)
            if concept and cultural_region in concept.cultural_adaptations:
                adaptations = concept.cultural_adaptations[cultural_region]

                # Apply content adaptations
                for lang, content in concept.content.items():
                    if "examples" in adaptations:
                        concept.content[lang] = self._adapt_content_examples(content, adaptations["examples"])

        path.cultural_region = cultural_region
        return True

    def _adapt_content_examples(self, content: str, cultural_examples: list[str]) -> str:
        """Adapt content with culturally relevant examples."""
        # Simple placeholder replacement for cultural adaptation
        adapted_content = content
        for i, example in enumerate(cultural_examples[:3]):
            adapted_content = adapted_content.replace(f"[EXAMPLE_{i+1}]", example)
        return adapted_content

    def track_learning_progress(self, student_id: str, concept_id: str, completion_time: float, score: float) -> None:
        """Track individual learning progress for optimization."""
        self.learning_analytics[concept_id].append(
            {
                "student_id": student_id,
                "completion_time": completion_time,
                "score": score,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

    def generate_adaptive_recommendations(self, student_id: str, current_concept: str) -> list[str]:
        """Generate adaptive learning recommendations for a student."""
        # Analyze student's performance pattern
        student_performance = self._analyze_student_performance(student_id)

        recommendations = []

        if student_performance["avg_score"] < 0.7:
            recommendations.extend(
                [
                    "Review prerequisite concepts",
                    "Try alternative explanation approaches",
                    "Use additional practice materials",
                ]
            )

        if student_performance["avg_completion_time"] > 1.5 * student_performance["expected_time"]:
            recommendations.extend(
                ["Break down complex concepts", "Add scaffolding materials", "Consider peer collaboration"]
            )

        return recommendations

    def _analyze_student_performance(self, student_id: str) -> dict[str, float]:
        """Analyze a student's overall performance pattern."""
        student_data = []

        for concept_analytics in self.learning_analytics.values():
            student_records = [record for record in concept_analytics if record["student_id"] == student_id]
            student_data.extend(student_records)

        if not student_data:
            return {"avg_score": 0.0, "avg_completion_time": 0.0, "expected_time": 0.0}

        avg_score = sum(record["score"] for record in student_data) / len(student_data)
        avg_completion_time = sum(record["completion_time"] for record in student_data) / len(student_data)

        return {
            "avg_score": avg_score,
            "avg_completion_time": avg_completion_time,
            "expected_time": avg_completion_time * 0.8,  # Expected time is 80% of average
            "total_concepts": len(student_data),
        }


class CurriculumGraph:
    """Main curriculum management system - refactored facade."""

    def __init__(self, project_name: str = "aivillage-education") -> None:
        self.project_name = project_name
        self.graph = HypergraphKG()

        # Initialize component managers
        self.concept_manager = ConceptManager()
        self.path_builder = PathBuilder(self.concept_manager)
        self.graph_analyzer = GraphAnalyzer(self.graph, self.concept_manager)
        self.optimizer = CurriculumOptimizer(self.path_builder, self.graph_analyzer)

        # Initialize W&B tracking
        self.initialize_wandb_tracking()

        # Load base curriculum data
        asyncio.create_task(self.initialize_base_curriculum())

    def initialize_wandb_tracking(self) -> None:
        """Initialize W&B tracking for curriculum development."""
        try:
            wandb.init(
                project=self.project_name,
                job_type="curriculum_development",
                config={
                    "curriculum_version": "2.0.0",
                    "grade_range": "K-8",
                    "subjects": ["mathematics", "science", "language_arts", "social_studies"],
                    "languages": ["en", "es", "hi", "fr", "ar", "pt", "sw"],
                    "cultural_regions": ["north_america", "latin_america", "europe", "africa", "asia"],
                    "optimization_enabled": True,
                    "adaptive_learning": True,
                },
            )
            logger.info("W&B tracking initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize W&B tracking: {e}")

    async def initialize_base_curriculum(self) -> None:
        """Initialize base curriculum with core concepts."""
        logger.info("Initializing base curriculum...")

        # Sample base concepts for mathematics
        math_concepts = [
            ("counting", 0, ["Number recognition", "Basic counting skills"]),
            ("addition", 1, ["Single digit addition", "Understanding plus symbol"]),
            ("subtraction", 1, ["Single digit subtraction", "Understanding minus symbol"]),
            ("multiplication", 2, ["Times tables", "Understanding multiplication concept"]),
            ("division", 3, ["Basic division", "Understanding division concept"]),
        ]

        for concept_name, grade, objectives in math_concepts:
            concept_id = self.concept_manager.generate_concept_id("mathematics", grade, concept_name)

            concept = ConceptNode(
                concept_id=concept_id,
                name=concept_name.title(),
                subject="mathematics",
                grade=grade,
                difficulty_level=self.concept_manager.estimate_difficulty([], grade),
                content={"en": f"Learn about {concept_name} in mathematics"},
                learning_objectives=objectives,
                assessment_criteria=self.concept_manager.generate_assessment_criteria(concept_name, grade),
                estimated_time_minutes=45,
                prerequisites=[],
                follow_up_concepts=[],
                cultural_adaptations={},
                created_at=datetime.now(UTC).isoformat(),
                updated_at=datetime.now(UTC).isoformat(),
            )

            self.concept_manager.add_concept(concept)

        logger.info("Base curriculum initialized with sample concepts")

    def create_comprehensive_curriculum(self, subject: str, grade_range: tuple[int, int]) -> str:
        """Create a comprehensive curriculum for subject and grade range."""
        path_id = self.path_builder.create_learning_path(subject, grade_range)

        # Optimize the path
        self.path_builder.optimize_path_difficulty(path_id)

        # Log to W&B
        if wandb.run:
            wandb.log(
                {
                    f"curriculum_created/{subject}": 1,
                    f"grade_range": f"{grade_range[0]}-{grade_range[1]}",
                    f"path_concepts_count": len(self.path_builder.learning_paths[path_id].concepts),
                }
            )

        return path_id

    def analyze_curriculum_effectiveness(self) -> dict[str, Any]:
        """Analyze overall curriculum effectiveness."""
        analysis = {
            "total_concepts": len(self.concept_manager.concepts),
            "total_paths": len(self.path_builder.learning_paths),
            "bottlenecks": self.graph_analyzer.find_learning_bottlenecks(),
            "recommendations": self.graph_analyzer.recommend_curriculum_improvements(),
            "effectiveness_scores": dict(self.graph_analyzer.concept_effectiveness),
        }

        # Log to W&B
        if wandb.run:
            wandb.log(
                {
                    "curriculum_analysis/total_concepts": analysis["total_concepts"],
                    "curriculum_analysis/total_paths": analysis["total_paths"],
                    "curriculum_analysis/bottlenecks_count": len(analysis["bottlenecks"]),
                }
            )

        return analysis

    def export_curriculum_data(self) -> dict[str, Any]:
        """Export comprehensive curriculum data."""
        return {
            "concepts": {
                cid: {
                    "name": concept.name,
                    "subject": concept.subject,
                    "grade": concept.grade,
                    "difficulty": concept.difficulty_level,
                    "prerequisites": concept.prerequisites,
                }
                for cid, concept in self.concept_manager.concepts.items()
            },
            "learning_paths": {
                pid: {
                    "name": path.name,
                    "subject": path.subject,
                    "grade_range": path.grade_range,
                    "concepts": path.concepts,
                    "duration_hours": path.estimated_duration_hours,
                }
                for pid, path in self.path_builder.learning_paths.items()
            },
            "analytics": {
                "effectiveness": dict(self.graph_analyzer.concept_effectiveness),
                "bottlenecks": self.graph_analyzer.find_learning_bottlenecks(),
            },
        }


# Maintain backward compatibility with simplified interface
async def create_curriculum_graph(project_name: str = "aivillage-education") -> CurriculumGraph:
    """Factory function to create and initialize curriculum graph."""
    curriculum = CurriculumGraph(project_name)
    await curriculum.initialize_base_curriculum()
    return curriculum
