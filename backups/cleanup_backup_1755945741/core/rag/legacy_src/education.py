"""HyperAG Education System - Advanced Learning Components

Implements educational AI components including:
- ELI5 (Explain Like I'm 5) chain reasoning
- Curriculum graph construction and navigation
- Adaptive learning path generation
- Knowledge dependency mapping
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import hashlib
import json
import logging
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


class DifficultyLevel(Enum):
    """Learning difficulty levels."""

    BEGINNER = "beginner"
    ELEMENTARY = "elementary"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class LearningStyle(Enum):
    """Different learning modalities."""

    VISUAL = "visual"
    AUDITORY = "auditory"
    KINESTHETIC = "kinesthetic"
    READING = "reading"
    MULTIMODAL = "multimodal"


@dataclass
class ConceptNode:
    """Represents a learning concept in the curriculum graph."""

    concept_id: str
    name: str
    description: str
    difficulty: DifficultyLevel
    prerequisites: list[str]
    learning_objectives: list[str]
    estimated_time_minutes: int
    tags: list[str]
    content_types: list[str]  # text, video, interactive, etc.

    def __post_init__(self):
        """Validate concept node after initialization."""
        if not self.concept_id:
            self.concept_id = hashlib.md5(self.name.encode()).hexdigest()[:12]


@dataclass
class LearningPath:
    """Represents a personalized learning path."""

    path_id: str
    learner_id: str
    concepts: list[ConceptNode]
    estimated_duration_hours: float
    difficulty_progression: list[DifficultyLevel]
    learning_style: LearningStyle
    completion_percentage: float = 0.0
    created_at: datetime = None

    def __post_init__(self):
        """Initialize learning path metadata."""
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class ELI5Explanation:
    """ELI5 (Explain Like I'm 5) explanation structure."""

    concept: str
    explanation: str
    analogies: list[str]
    examples: list[str]
    key_points: list[str]
    follow_up_questions: list[str]
    difficulty_level: DifficultyLevel
    age_appropriate: bool


class CurriculumGraph:
    """
    Knowledge graph for curriculum planning and learning path generation.

    Uses NetworkX for graph operations and implements adaptive learning
    algorithms for personalized education paths.
    """

    def __init__(self):
        """Initialize curriculum graph."""
        self.graph = nx.DiGraph()
        self.concepts: dict[str, ConceptNode] = {}
        self.learning_paths: dict[str, LearningPath] = {}
        self.learner_profiles: dict[str, dict[str, Any]] = {}

        # Graph statistics
        self.stats = {
            "total_concepts": 0,
            "total_connections": 0,
            "average_depth": 0.0,
            "learning_paths_generated": 0,
        }

        logger.info("Curriculum graph initialized")

    def add_concept(self, concept: ConceptNode) -> bool:
        """
        Add a learning concept to the curriculum graph.

        Args:
            concept: ConceptNode to add

        Returns:
            bool: True if successfully added
        """
        try:
            # Add concept to storage
            self.concepts[concept.concept_id] = concept

            # Add node to graph
            self.graph.add_node(
                concept.concept_id,
                name=concept.name,
                difficulty=concept.difficulty.value,
                estimated_time=concept.estimated_time_minutes,
                tags=concept.tags,
            )

            # Add prerequisite edges
            for prereq_id in concept.prerequisites:
                if prereq_id in self.concepts:
                    self.graph.add_edge(prereq_id, concept.concept_id)
                else:
                    logger.warning(f"Prerequisite {prereq_id} not found for concept {concept.concept_id}")

            self._update_stats()
            logger.debug(f"Added concept: {concept.name} ({concept.concept_id})")
            return True

        except Exception as e:
            logger.error(f"Failed to add concept {concept.name}: {e}")
            return False

    def remove_concept(self, concept_id: str) -> bool:
        """
        Remove a concept from the curriculum graph.

        Args:
            concept_id: ID of concept to remove

        Returns:
            bool: True if successfully removed
        """
        try:
            if concept_id not in self.concepts:
                logger.warning(f"Concept {concept_id} not found")
                return False

            # Remove from storage
            del self.concepts[concept_id]

            # Remove from graph
            self.graph.remove_node(concept_id)

            self._update_stats()
            logger.debug(f"Removed concept: {concept_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to remove concept {concept_id}: {e}")
            return False

    def get_prerequisites(self, concept_id: str) -> list[ConceptNode]:
        """
        Get all prerequisite concepts for a given concept.

        Args:
            concept_id: Target concept ID

        Returns:
            List of prerequisite ConceptNodes
        """
        if concept_id not in self.graph:
            return []

        prerequisite_ids = list(self.graph.predecessors(concept_id))
        return [self.concepts[pid] for pid in prerequisite_ids if pid in self.concepts]

    def get_dependents(self, concept_id: str) -> list[ConceptNode]:
        """
        Get all concepts that depend on the given concept.

        Args:
            concept_id: Source concept ID

        Returns:
            List of dependent ConceptNodes
        """
        if concept_id not in self.graph:
            return []

        dependent_ids = list(self.graph.successors(concept_id))
        return [self.concepts[did] for did in dependent_ids if did in self.concepts]

    def find_learning_path(
        self,
        start_concepts: list[str],
        target_concepts: list[str],
        learner_id: str,
        max_concepts: int = 20,
    ) -> LearningPath | None:
        """
        Generate optimal learning path from start to target concepts.

        Args:
            start_concepts: List of already mastered concept IDs
            target_concepts: List of target concept IDs to learn
            learner_id: Unique learner identifier
            max_concepts: Maximum concepts in learning path

        Returns:
            LearningPath or None if no path found
        """
        try:
            # Get learner profile for personalization
            learner_profile = self.learner_profiles.get(learner_id, self._create_default_profile())
            preferred_style = LearningStyle(learner_profile.get("learning_style", "multimodal"))

            # Find shortest paths to all targets
            all_concepts = set()
            path_concepts = []

            for target_id in target_concepts:
                if target_id not in self.graph:
                    logger.warning(f"Target concept {target_id} not found")
                    continue

                # Find path from any start concept to this target
                best_path = None
                best_length = float("inf")

                for start_id in start_concepts:
                    if start_id not in self.graph:
                        continue

                    try:
                        path = nx.shortest_path(self.graph, start_id, target_id)
                        if len(path) < best_length:
                            best_path = path[1:]  # Exclude start concept
                            best_length = len(path)
                    except nx.NetworkXNoPath:
                        continue

                if best_path:
                    all_concepts.update(best_path)
                    path_concepts.extend(best_path)

            if not all_concepts:
                logger.warning("No learning path found to target concepts")
                return None

            # Remove duplicates and sort by prerequisites
            unique_concepts = list(all_concepts)
            sorted_concepts = self._topological_sort_concepts(unique_concepts)

            # Limit to max_concepts
            if len(sorted_concepts) > max_concepts:
                sorted_concepts = sorted_concepts[:max_concepts]

            # Create ConceptNode list
            concept_list = [self.concepts[cid] for cid in sorted_concepts if cid in self.concepts]

            # Calculate estimated duration
            total_minutes = sum(concept.estimated_time_minutes for concept in concept_list)
            estimated_hours = total_minutes / 60.0

            # Determine difficulty progression
            difficulty_progression = [concept.difficulty for concept in concept_list]

            # Generate unique path ID
            path_id = hashlib.md5(f"{learner_id}_{target_concepts}_{datetime.now()}".encode()).hexdigest()[:12]

            # Create learning path
            learning_path = LearningPath(
                path_id=path_id,
                learner_id=learner_id,
                concepts=concept_list,
                estimated_duration_hours=estimated_hours,
                difficulty_progression=difficulty_progression,
                learning_style=preferred_style,
            )

            # Store learning path
            self.learning_paths[path_id] = learning_path
            self.stats["learning_paths_generated"] += 1

            logger.info(f"Generated learning path {path_id} with {len(concept_list)} concepts")
            return learning_path

        except Exception as e:
            logger.error(f"Failed to find learning path: {e}")
            return None

    def update_learner_progress(self, learner_id: str, path_id: str, completed_concepts: list[str]) -> bool:
        """
        Update learner progress on a learning path.

        Args:
            learner_id: Learner identifier
            path_id: Learning path identifier
            completed_concepts: List of completed concept IDs

        Returns:
            bool: True if successfully updated
        """
        try:
            if path_id not in self.learning_paths:
                logger.warning(f"Learning path {path_id} not found")
                return False

            learning_path = self.learning_paths[path_id]

            if learning_path.learner_id != learner_id:
                logger.warning(f"Learning path {path_id} does not belong to learner {learner_id}")
                return False

            # Calculate completion percentage
            total_concepts = len(learning_path.concepts)
            completed_count = len([c for c in learning_path.concepts if c.concept_id in completed_concepts])
            completion_percentage = (completed_count / total_concepts) * 100 if total_concepts > 0 else 0

            # Update learning path
            learning_path.completion_percentage = completion_percentage

            # Update learner profile
            if learner_id not in self.learner_profiles:
                self.learner_profiles[learner_id] = self._create_default_profile()

            profile = self.learner_profiles[learner_id]
            profile["completed_concepts"] = profile.get("completed_concepts", []) + completed_concepts
            profile["total_learning_time"] = profile.get("total_learning_time", 0) + sum(
                c.estimated_time_minutes for c in learning_path.concepts if c.concept_id in completed_concepts
            )

            logger.info(f"Updated progress for learner {learner_id}: {completion_percentage:.1f}% complete")
            return True

        except Exception as e:
            logger.error(f"Failed to update learner progress: {e}")
            return False

    def get_recommended_next_concepts(self, learner_id: str, max_recommendations: int = 5) -> list[ConceptNode]:
        """
        Get recommended next concepts for a learner based on their progress.

        Args:
            learner_id: Learner identifier
            max_recommendations: Maximum number of recommendations

        Returns:
            List of recommended ConceptNodes
        """
        try:
            if learner_id not in self.learner_profiles:
                return []

            profile = self.learner_profiles[learner_id]
            completed_concepts = set(profile.get("completed_concepts", []))

            # Find concepts where all prerequisites are completed
            available_concepts = []

            for concept_id, concept in self.concepts.items():
                if concept_id in completed_concepts:
                    continue

                # Check if all prerequisites are completed
                prerequisites = set(concept.prerequisites)
                if prerequisites.issubset(completed_concepts):
                    available_concepts.append(concept)

            # Sort by difficulty and estimated time
            available_concepts.sort(key=lambda c: (c.difficulty.value, c.estimated_time_minutes))

            # Return top recommendations
            return available_concepts[:max_recommendations]

        except Exception as e:
            logger.error(f"Failed to get recommendations for learner {learner_id}: {e}")
            return []

    def _topological_sort_concepts(self, concept_ids: list[str]) -> list[str]:
        """
        Sort concepts in topological order based on prerequisites.

        Args:
            concept_ids: List of concept IDs to sort

        Returns:
            List of concept IDs in dependency order
        """
        # Create subgraph with only the specified concepts
        subgraph = self.graph.subgraph(concept_ids)

        try:
            # Topological sort
            sorted_ids = list(nx.topological_sort(subgraph))
            return sorted_ids
        except nx.NetworkXError:
            # Fallback: sort by difficulty if graph has cycles
            concept_difficulty = [
                (cid, self.concepts[cid].difficulty.value) for cid in concept_ids if cid in self.concepts
            ]
            concept_difficulty.sort(key=lambda x: x[1])
            return [cid for cid, _ in concept_difficulty]

    def _create_default_profile(self) -> dict[str, Any]:
        """Create default learner profile."""
        return {
            "learning_style": "multimodal",
            "preferred_difficulty": "intermediate",
            "completed_concepts": [],
            "total_learning_time": 0,
            "created_at": datetime.now().isoformat(),
        }

    def _update_stats(self):
        """Update graph statistics."""
        self.stats["total_concepts"] = len(self.concepts)
        self.stats["total_connections"] = self.graph.number_of_edges()

        if self.graph.number_of_nodes() > 0:
            try:
                # Calculate average path length as proxy for depth
                path_lengths = []
                for source in self.graph.nodes():
                    for target in self.graph.nodes():
                        if source != target:
                            try:
                                length = nx.shortest_path_length(self.graph, source, target)
                                path_lengths.append(length)
                            except nx.NetworkXNoPath:
                                pass

                self.stats["average_depth"] = sum(path_lengths) / len(path_lengths) if path_lengths else 0.0
            except Exception:
                self.stats["average_depth"] = 0.0

    def get_graph_stats(self) -> dict[str, Any]:
        """Get curriculum graph statistics."""
        return dict(self.stats)

    def export_graph(self, format_type: str = "json") -> str:
        """
        Export curriculum graph in specified format.

        Args:
            format_type: Export format ("json", "gml", "graphml")

        Returns:
            Serialized graph data
        """
        try:
            if format_type == "json":
                # Export as JSON with concept metadata
                graph_data = {
                    "concepts": {
                        cid: {
                            "name": concept.name,
                            "description": concept.description,
                            "difficulty": concept.difficulty.value,
                            "prerequisites": concept.prerequisites,
                            "estimated_time": concept.estimated_time_minutes,
                            "tags": concept.tags,
                        }
                        for cid, concept in self.concepts.items()
                    },
                    "edges": list(self.graph.edges()),
                    "statistics": self.stats,
                }
                return json.dumps(graph_data, indent=2)

            else:
                # Fallback to JSON format for unsupported formats
                logger.warning(f"Export format {format_type} not supported, using JSON format")
                return json.dumps({"message": f"Format {format_type} not supported", "data": {}}, indent=2)

        except Exception as e:
            logger.error(f"Failed to export graph: {e}")
            return ""


class ELI5Chain:
    """
    ELI5 (Explain Like I'm 5) explanation generation system.

    Generates age-appropriate explanations for complex concepts using
    analogies, examples, and progressive complexity levels.
    """

    def __init__(self):
        """Initialize ELI5 explanation system."""
        self.explanation_cache: dict[str, ELI5Explanation] = {}
        self.analogy_templates: dict[str, list[str]] = self._load_analogy_templates()
        self.age_guidelines: dict[str, dict[str, Any]] = self._load_age_guidelines()

        logger.info("ELI5 explanation system initialized")

    def generate_explanation(
        self,
        concept: str,
        difficulty_level: DifficultyLevel = DifficultyLevel.ELEMENTARY,
        target_age: int = 8,
    ) -> ELI5Explanation:
        """
        Generate ELI5 explanation for a concept.

        Args:
            concept: Concept to explain
            difficulty_level: Target difficulty level
            target_age: Target age for explanation

        Returns:
            ELI5Explanation object
        """
        # Check cache first
        cache_key = f"{concept}_{difficulty_level.value}_{target_age}"
        if cache_key in self.explanation_cache:
            return self.explanation_cache[cache_key]

        try:
            # Generate explanation components
            explanation = self._generate_core_explanation(concept, difficulty_level, target_age)
            analogies = self._generate_analogies(concept, target_age)
            examples = self._generate_examples(concept, target_age)
            key_points = self._extract_key_points(concept, difficulty_level)
            follow_up_questions = self._generate_follow_up_questions(concept, target_age)

            # Check age appropriateness
            age_appropriate = self._check_age_appropriateness(explanation, target_age)

            # Create ELI5 explanation
            eli5_explanation = ELI5Explanation(
                concept=concept,
                explanation=explanation,
                analogies=analogies,
                examples=examples,
                key_points=key_points,
                follow_up_questions=follow_up_questions,
                difficulty_level=difficulty_level,
                age_appropriate=age_appropriate,
            )

            # Cache explanation
            self.explanation_cache[cache_key] = eli5_explanation

            logger.debug(f"Generated ELI5 explanation for: {concept}")
            return eli5_explanation

        except Exception as e:
            logger.error(f"Failed to generate ELI5 explanation for {concept}: {e}")
            # Return basic explanation as fallback
            return ELI5Explanation(
                concept=concept,
                explanation=f"This is about {concept}. It's something we can learn more about!",
                analogies=["Like learning about something new and interesting"],
                examples=[f"An example of {concept}"],
                key_points=[f"The main thing about {concept}"],
                follow_up_questions=["What would you like to know more about?"],
                difficulty_level=difficulty_level,
                age_appropriate=True,
            )

    def generate_progressive_explanations(self, concept: str, levels: list[DifficultyLevel]) -> list[ELI5Explanation]:
        """
        Generate explanations at multiple difficulty levels.

        Args:
            concept: Concept to explain
            levels: List of difficulty levels

        Returns:
            List of ELI5Explanation objects
        """
        explanations = []

        age_mapping = {
            DifficultyLevel.BEGINNER: 5,
            DifficultyLevel.ELEMENTARY: 8,
            DifficultyLevel.INTERMEDIATE: 12,
            DifficultyLevel.ADVANCED: 16,
            DifficultyLevel.EXPERT: 20,
        }

        for level in levels:
            target_age = age_mapping.get(level, 10)
            explanation = self.generate_explanation(concept, level, target_age)
            explanations.append(explanation)

        return explanations

    def _generate_core_explanation(self, concept: str, difficulty_level: DifficultyLevel, target_age: int) -> str:
        """Generate the core explanation text."""
        # This would be replaced with actual AI/NLP explanation generation
        # For now, provide structured explanation templates

        templates = {
            DifficultyLevel.BEGINNER: f"{concept} is like a special thing that helps us understand the world better. It's really cool and interesting!",
            DifficultyLevel.ELEMENTARY: f"{concept} is an important idea that helps us learn and solve problems. When we understand {concept}, we can do amazing things!",
            DifficultyLevel.INTERMEDIATE: f"{concept} is a concept that involves several important parts working together. Understanding {concept} helps us see how different things connect and influence each other.",
            DifficultyLevel.ADVANCED: f"{concept} represents a complex system with multiple interconnected components and relationships. Mastering {concept} requires understanding both the individual elements and their interactions.",
            DifficultyLevel.EXPERT: f"{concept} encompasses a sophisticated framework of interdependent principles, mechanisms, and applications that form the foundation for advanced analysis and implementation.",
        }

        return templates.get(difficulty_level, templates[DifficultyLevel.ELEMENTARY])

    def _generate_analogies(self, concept: str, target_age: int) -> list[str]:
        """Generate age-appropriate analogies."""
        # Simple analogy generation based on concept and age
        if target_age <= 6:
            analogies = [
                f"{concept} is like a puzzle where each piece fits perfectly",
                f"Think of {concept} like building blocks that stack together",
                "It's like a story where each part helps tell the whole tale",
            ]
        elif target_age <= 10:
            analogies = [
                f"{concept} works like a team where everyone has a special job",
                f"Imagine {concept} as a recipe - you need all the ingredients to make it work",
                "It's like a machine with different parts that work together",
            ]
        else:
            analogies = [
                f"{concept} functions like an ecosystem where all parts are interconnected",
                f"Think of {concept} as a symphony where different instruments create harmony",
                "It's like a complex network where information flows between nodes",
            ]

        return analogies[:2]  # Return top 2 analogies

    def _generate_examples(self, concept: str, target_age: int) -> list[str]:
        """Generate age-appropriate examples."""
        # Example generation based on concept and age
        examples = [
            f"When you see {concept} in everyday life, like...",
            f"A simple example of {concept} would be...",
            f"You might encounter {concept} when...",
        ]

        return examples[:3]

    def _extract_key_points(self, concept: str, difficulty_level: DifficultyLevel) -> list[str]:
        """Extract key points for the concept."""
        point_counts = {
            DifficultyLevel.BEGINNER: 2,
            DifficultyLevel.ELEMENTARY: 3,
            DifficultyLevel.INTERMEDIATE: 4,
            DifficultyLevel.ADVANCED: 5,
            DifficultyLevel.EXPERT: 6,
        }

        num_points = point_counts.get(difficulty_level, 3)

        key_points = [f"Key point {i + 1} about {concept}" for i in range(num_points)]

        return key_points

    def _generate_follow_up_questions(self, concept: str, target_age: int) -> list[str]:
        """Generate follow-up questions to encourage deeper thinking."""
        if target_age <= 8:
            questions = [
                f"What do you think is the most interesting thing about {concept}?",
                f"Can you think of where you might see {concept} in your daily life?",
                f"What questions do you have about {concept}?",
            ]
        else:
            questions = [
                f"How do you think {concept} connects to other things you've learned?",
                f"What would happen if we didn't have {concept}?",
                f"Can you think of ways to apply {concept} to solve problems?",
                f"What aspects of {concept} would you like to explore further?",
            ]

        return questions[:3]

    def _check_age_appropriateness(self, explanation: str, target_age: int) -> bool:
        """Check if explanation is age-appropriate."""
        # Simple heuristics for age appropriateness
        guidelines = self.age_guidelines.get(str(target_age), self.age_guidelines.get("default", {}))

        max_sentence_length = guidelines.get("max_sentence_length", 20)
        forbidden_words = guidelines.get("forbidden_words", [])

        # Check sentence length
        sentences = explanation.split(".")
        for sentence in sentences:
            if len(sentence.split()) > max_sentence_length:
                return False

        # Check for forbidden words
        explanation_lower = explanation.lower()
        for word in forbidden_words:
            if word in explanation_lower:
                return False

        return True

    def _load_analogy_templates(self) -> dict[str, list[str]]:
        """Load analogy templates for different concepts."""
        return {
            "science": [
                "like an experiment where we discover new things",
                "like exploring a mystery with clues to find",
                "like building something amazing with special tools",
            ],
            "math": [
                "like solving puzzles with numbers",
                "like using a special language to describe patterns",
                "like following recipes that always work the same way",
            ],
            "technology": [
                "like having a super smart helper",
                "like magic that we can understand and control",
                "like tools that make our lives easier and more fun",
            ],
            "default": [
                "like something we can explore and understand",
                "like a new adventure waiting to be discovered",
                "like a skill we can learn and practice",
            ],
        }

    def _load_age_guidelines(self) -> dict[str, dict[str, Any]]:
        """Load age-appropriate language guidelines."""
        return {
            "5": {
                "max_sentence_length": 8,
                "forbidden_words": ["complex", "sophisticated", "paradigm"],
                "preferred_words": ["fun", "cool", "amazing", "special"],
            },
            "8": {
                "max_sentence_length": 12,
                "forbidden_words": ["paradigm", "methodology", "implementation"],
                "preferred_words": ["interesting", "important", "helpful", "useful"],
            },
            "12": {
                "max_sentence_length": 16,
                "forbidden_words": ["paradigm", "epistemological"],
                "preferred_words": ["understand", "learn", "discover", "explore"],
            },
            "default": {
                "max_sentence_length": 20,
                "forbidden_words": [],
                "preferred_words": ["comprehend", "analyze", "investigate"],
            },
        }


# Global instances for easy access
curriculum_graph = CurriculumGraph()
eli5_chain = ELI5Chain()


# Utility functions for common operations
def add_concept_to_curriculum(
    name: str,
    description: str,
    difficulty: str = "intermediate",
    prerequisites: list[str] = None,
    estimated_minutes: int = 60,
) -> str:
    """
    Utility function to easily add concepts to the curriculum graph.

    Args:
        name: Concept name
        description: Concept description
        difficulty: Difficulty level string
        prerequisites: List of prerequisite concept IDs
        estimated_minutes: Estimated learning time

    Returns:
        Generated concept ID
    """
    concept = ConceptNode(
        concept_id="",  # Will be auto-generated
        name=name,
        description=description,
        difficulty=DifficultyLevel(difficulty),
        prerequisites=prerequisites or [],
        learning_objectives=[f"Understand {name}", f"Apply {name} concepts"],
        estimated_time_minutes=estimated_minutes,
        tags=[difficulty, "curriculum"],
        content_types=["text", "examples"],
    )

    if curriculum_graph.add_concept(concept):
        return concept.concept_id
    else:
        return ""


def generate_learning_path_for_topic(
    topic: str, learner_id: str, current_knowledge: list[str] = None
) -> LearningPath | None:
    """
    Generate a learning path for a specific topic.

    Args:
        topic: Topic or concept to learn
        learner_id: Unique learner identifier
        current_knowledge: List of already known concept IDs

    Returns:
        LearningPath or None
    """
    # Find concepts related to the topic
    related_concepts = []
    for concept_id, concept in curriculum_graph.concepts.items():
        if topic.lower() in concept.name.lower() or topic.lower() in concept.description.lower():
            related_concepts.append(concept_id)

    if not related_concepts:
        logger.warning(f"No concepts found for topic: {topic}")
        return None

    # Generate learning path
    return curriculum_graph.find_learning_path(
        start_concepts=current_knowledge or [],
        target_concepts=related_concepts,
        learner_id=learner_id,
    )


def explain_concept_eli5(concept: str, age: int = 8) -> ELI5Explanation:
    """
    Generate an ELI5 explanation for a concept.

    Args:
        concept: Concept to explain
        age: Target age for explanation

    Returns:
        ELI5Explanation object
    """
    difficulty_mapping = {
        5: DifficultyLevel.BEGINNER,
        8: DifficultyLevel.ELEMENTARY,
        12: DifficultyLevel.INTERMEDIATE,
        16: DifficultyLevel.ADVANCED,
    }

    difficulty = difficulty_mapping.get(age, DifficultyLevel.ELEMENTARY)
    return eli5_chain.generate_explanation(concept, difficulty, age)
