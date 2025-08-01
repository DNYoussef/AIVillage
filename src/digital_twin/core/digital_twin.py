"""Digital Twin Core System for Personalized Learning
Sprint R-5: Digital Twin MVP - Task A.1.
"""

import asyncio
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
import sqlite3
from typing import Any
import uuid

from cryptography.fernet import Fernet
import numpy as np

import wandb

logger = logging.getLogger(__name__)


@dataclass
class LearningProfile:
    """Individual learner's profile and preferences."""

    student_id: str
    name: str
    age: int
    grade_level: int
    language: str
    region: str
    learning_style: str  # visual, auditory, kinesthetic, reading
    strengths: list[str]  # Subject areas where student excels
    challenges: list[str]  # Areas needing improvement
    interests: list[str]  # Topics of interest for engagement
    attention_span_minutes: int
    preferred_session_times: list[str]  # "morning", "afternoon", "evening"
    parent_constraints: dict[str, Any]  # Screen time limits, content filters
    accessibility_needs: list[str]  # Special accommodations
    motivation_triggers: list[str]  # What motivates this student
    created_at: str = ""
    last_updated: str = ""


@dataclass
class LearningSession:
    """Individual learning session record."""

    session_id: str
    student_id: str
    tutor_model_id: str
    start_time: str
    end_time: str
    duration_minutes: int
    concepts_covered: list[str]
    questions_asked: int
    questions_correct: int
    engagement_score: float
    difficulty_level: float
    adaptations_made: list[str]
    parent_feedback: str | None = None
    student_mood: str = "neutral"
    session_notes: str = ""


@dataclass
class KnowledgeState:
    """Student's current knowledge state."""

    student_id: str
    subject: str
    concept: str
    mastery_level: float  # 0.0 to 1.0
    confidence_score: float
    last_practiced: str
    practice_count: int
    mistake_patterns: list[str]
    prerequisite_gaps: list[str]
    next_recommended: list[str]
    estimated_study_time: int  # minutes to achieve mastery
    retention_decay_rate: float  # How quickly student forgets


@dataclass
class PersonalizationVector:
    """Multi-dimensional representation of student's learning needs."""

    cognitive_load_preference: float  # 0=simple, 1=complex
    explanation_depth: float  # 0=brief, 1=detailed
    visual_learning_weight: float
    kinesthetic_learning_weight: float
    social_learning_preference: float
    gamification_response: float
    cultural_context_importance: float
    language_complexity_comfort: float
    pacing_preference: float  # 0=slow, 1=fast
    challenge_seeking: float  # Risk tolerance for difficult problems


class DigitalTwin:
    """Comprehensive digital twin for personalized math tutoring."""

    def __init__(self, project_name: str = "aivillage-digital-twin") -> None:
        self.project_name = project_name
        self.students = {}  # student_id -> LearningProfile
        self.knowledge_states = defaultdict(
            dict
        )  # student_id -> {concept: KnowledgeState}
        self.session_history = defaultdict(list)  # student_id -> List[LearningSession]
        self.personalization_vectors = {}  # student_id -> PersonalizationVector

        # Learning analytics
        self.learning_patterns = defaultdict(dict)
        self.adaptation_rules = {}
        self.performance_predictors = {}

        # Recommendation engine
        self.recommendation_cache = {}
        self.content_difficulty_map = {}

        # Privacy and security
        self.encryption_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)

        # Database for persistence
        self.db_path = "digital_twin.db"
        self.init_database()

        # Background tasks
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.background_tasks = []

        # Initialize W&B tracking
        self.initialize_wandb_tracking()

        # Start background analytics
        asyncio.create_task(self.start_background_analytics())

    def initialize_wandb_tracking(self) -> None:
        """Initialize W&B tracking for digital twin."""
        try:
            wandb.init(
                project=self.project_name,
                job_type="digital_twin_personalization",
                config={
                    "digital_twin_version": "1.0.0",
                    "personalization_dimensions": 10,
                    "knowledge_tracking": "concept_level",
                    "adaptation_frequency": "real_time",
                    "privacy_level": "encrypted_local",
                    "supported_learning_styles": [
                        "visual",
                        "auditory",
                        "kinesthetic",
                        "reading",
                    ],
                    "supported_languages": ["en", "es", "hi", "fr", "ar"],
                    "grade_range": "K-8",
                    "real_time_adaptation": True,
                },
            )

            logger.info("Digital Twin W&B tracking initialized")

        except Exception as e:
            logger.exception(f"Failed to initialize W&B tracking: {e}")

    def init_database(self) -> None:
        """Initialize SQLite database for persistent storage."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Students table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS students (
                    student_id TEXT PRIMARY KEY,
                    profile_data TEXT,  -- Encrypted JSON
                    created_at TEXT,
                    last_updated TEXT
                )
            """)

            # Knowledge states table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_states (
                    student_id TEXT,
                    subject TEXT,
                    concept TEXT,
                    state_data TEXT,  -- Encrypted JSON
                    last_updated TEXT,
                    PRIMARY KEY (student_id, subject, concept)
                )
            """)

            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS learning_sessions (
                    session_id TEXT PRIMARY KEY,
                    student_id TEXT,
                    session_data TEXT,  -- Encrypted JSON
                    start_time TEXT,
                    end_time TEXT,
                    created_at TEXT
                )
            """)

            # Personalization vectors table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS personalization_vectors (
                    student_id TEXT PRIMARY KEY,
                    vector_data TEXT,  -- Encrypted JSON
                    last_updated TEXT
                )
            """)

            conn.commit()
            conn.close()

            logger.info("Digital twin database initialized")

        except Exception as e:
            logger.exception(f"Failed to initialize database: {e}")

    async def create_student_profile(
        self,
        name: str,
        age: int,
        grade_level: int,
        language: str = "en",
        region: str = "north_america",
        parent_email: str | None = None,
        initial_assessment: dict[str, Any] | None = None,
    ) -> str:
        """Create new student profile with initial assessment."""
        student_id = str(uuid.uuid4())

        # Run initial assessment if provided
        if initial_assessment:
            learning_profile = await self.analyze_initial_assessment(
                student_id, age, grade_level, initial_assessment
            )
        else:
            # Create default profile
            learning_profile = LearningProfile(
                student_id=student_id,
                name=name,
                age=age,
                grade_level=grade_level,
                language=language,
                region=region,
                learning_style="balanced",  # Will be refined through observation
                strengths=[],
                challenges=[],
                interests=[],
                attention_span_minutes=max(15, min(45, age * 3)),  # Age-appropriate
                preferred_session_times=["afternoon"],
                parent_constraints={
                    "max_daily_minutes": age * 10,
                    "content_filter": "age_appropriate",
                    "progress_notifications": True,
                },
                accessibility_needs=[],
                motivation_triggers=["praise", "progress_visualization"],
                created_at=datetime.now(timezone.utc).isoformat(),
                last_updated=datetime.now(timezone.utc).isoformat(),
            )

        # Initialize personalization vector
        personalization_vector = PersonalizationVector(
            cognitive_load_preference=0.5,  # Start balanced
            explanation_depth=0.6 if age >= 10 else 0.4,
            visual_learning_weight=0.5,
            kinesthetic_learning_weight=0.5,
            social_learning_preference=0.3,
            gamification_response=0.8 if age <= 12 else 0.5,
            cultural_context_importance=0.7,
            language_complexity_comfort=min(1.0, age / 16.0),
            pacing_preference=0.5,
            challenge_seeking=0.4,
        )

        # Store profile
        self.students[student_id] = learning_profile
        self.personalization_vectors[student_id] = personalization_vector

        # Initialize knowledge states for grade-appropriate concepts
        await self.initialize_knowledge_states(student_id, grade_level)

        # Save to database
        await self.save_student_profile(student_id)

        # Log creation
        wandb.log(
            {
                "digital_twin/student_created": True,
                "digital_twin/age": age,
                "digital_twin/grade_level": grade_level,
                "digital_twin/language": language,
                "digital_twin/region": region,
                "students/total_count": len(self.students),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        logger.info(f"Created student profile for {name} (ID: {student_id[:8]})")

        return student_id

    async def analyze_initial_assessment(
        self, student_id: str, age: int, grade_level: int, assessment: dict[str, Any]
    ) -> LearningProfile:
        """Analyze initial assessment to create personalized profile."""
        # Extract learning style from assessment responses
        learning_style = self.detect_learning_style(assessment)

        # Identify strengths and challenges
        strengths, challenges = self.analyze_math_abilities(assessment, grade_level)

        # Extract interests from responses
        interests = self.extract_interests(assessment)

        # Estimate attention span from response patterns
        attention_span = self.estimate_attention_span(assessment, age)

        # Determine motivation triggers
        motivation_triggers = self.identify_motivation_triggers(assessment)

        profile = LearningProfile(
            student_id=student_id,
            name=assessment.get("name", "Student"),
            age=age,
            grade_level=grade_level,
            language=assessment.get("language", "en"),
            region=assessment.get("region", "north_america"),
            learning_style=learning_style,
            strengths=strengths,
            challenges=challenges,
            interests=interests,
            attention_span_minutes=attention_span,
            preferred_session_times=assessment.get("preferred_times", ["afternoon"]),
            parent_constraints=assessment.get(
                "parent_constraints",
                {"max_daily_minutes": age * 10, "content_filter": "age_appropriate"},
            ),
            accessibility_needs=assessment.get("accessibility_needs", []),
            motivation_triggers=motivation_triggers,
            created_at=datetime.now(timezone.utc).isoformat(),
            last_updated=datetime.now(timezone.utc).isoformat(),
        )

        return profile

    def detect_learning_style(self, assessment: dict[str, Any]) -> str:
        """Detect primary learning style from assessment."""
        style_scores = {"visual": 0, "auditory": 0, "kinesthetic": 0, "reading": 0}

        # Analyze response patterns
        for response in assessment.get("responses", {}).values():
            if "picture" in response.lower() or "see" in response.lower():
                style_scores["visual"] += 1
            if "hear" in response.lower() or "listen" in response.lower():
                style_scores["auditory"] += 1
            if "hands" in response.lower() or "touch" in response.lower():
                style_scores["kinesthetic"] += 1
            if "read" in response.lower() or "write" in response.lower():
                style_scores["reading"] += 1

        # Return dominant style or balanced
        max_score = max(style_scores.values())
        if max_score == 0:
            return "balanced"

        return max(style_scores, key=style_scores.get)

    def analyze_math_abilities(
        self, assessment: dict[str, Any], grade_level: int
    ) -> tuple[list[str], list[str]]:
        """Analyze mathematical strengths and challenges."""
        strengths = []
        challenges = []

        # Analyze performance by concept
        concept_scores = assessment.get("concept_scores", {})

        for concept, score in concept_scores.items():
            if score >= 0.8:
                strengths.append(concept)
            elif score <= 0.4:
                challenges.append(concept)

        # Default grade-appropriate analysis
        if not concept_scores:
            grade_concepts = {
                1: (["counting", "addition"], ["subtraction"]),
                2: (["addition"], ["multiplication", "fractions"]),
                3: (["multiplication"], ["division", "fractions"]),
                4: (["basic_operations"], ["fractions", "decimals"]),
                5: (["fractions"], ["decimals", "geometry"]),
                6: (["decimals"], ["algebra", "ratios"]),
                7: (["algebra"], ["geometry", "statistics"]),
                8: (["algebra"], ["geometry", "functions"]),
            }

            if grade_level in grade_concepts:
                default_strengths, default_challenges = grade_concepts[grade_level]
                strengths.extend(default_strengths)
                challenges.extend(default_challenges)

        return strengths, challenges

    def extract_interests(self, assessment: dict[str, Any]) -> list[str]:
        """Extract student interests from assessment."""
        interests = assessment.get("interests", [])

        # Analyze free-text responses for interest keywords
        interest_keywords = {
            "sports": ["football", "soccer", "basketball", "baseball", "tennis"],
            "art": ["drawing", "painting", "music", "creative"],
            "science": ["experiments", "nature", "animals", "space"],
            "technology": ["computers", "games", "coding", "robots"],
            "reading": ["books", "stories", "reading"],
            "cooking": ["food", "cooking", "baking"],
            "travel": ["places", "countries", "travel"],
        }

        responses_text = " ".join(assessment.get("responses", {}).values()).lower()

        for interest, keywords in interest_keywords.items():
            if any(keyword in responses_text for keyword in keywords):
                interests.append(interest)

        return list(set(interests))

    def estimate_attention_span(self, assessment: dict[str, Any], age: int) -> int:
        """Estimate attention span from assessment behavior."""
        # Base estimate by age
        base_minutes = max(15, min(45, age * 3))

        # Adjust based on assessment completion time and quality
        completion_time = assessment.get("completion_time_minutes", base_minutes)
        response_quality = assessment.get("response_quality_score", 0.5)

        # If they completed assessment quickly with good quality, higher attention span
        if completion_time < base_minutes and response_quality > 0.7:
            return min(60, int(base_minutes * 1.2))
        if completion_time > base_minutes * 1.5:
            return max(10, int(base_minutes * 0.8))

        return base_minutes

    def identify_motivation_triggers(self, assessment: dict[str, Any]) -> list[str]:
        """Identify what motivates the student."""
        triggers = []

        # Analyze responses for motivation indicators
        responses_text = " ".join(assessment.get("responses", {}).values()).lower()

        motivation_patterns = {
            "achievement": ["best", "win", "first", "perfect", "succeed"],
            "praise": ["good job", "proud", "praise", "compliment"],
            "progress": ["improve", "better", "progress", "grow"],
            "social": ["friends", "team", "together", "help others"],
            "competition": ["compete", "race", "challenge", "beat"],
            "curiosity": ["why", "how", "wonder", "curious", "explore"],
            "autonomy": ["choose", "decide", "own way", "independent"],
        }

        for trigger, keywords in motivation_patterns.items():
            if any(keyword in responses_text for keyword in keywords):
                triggers.append(trigger)

        # Default motivators by age
        if not triggers:
            if age <= 8:
                triggers = ["praise", "visual_progress", "gamification"]
            elif age <= 12:
                triggers = ["achievement", "progress", "social"]
            else:
                triggers = ["autonomy", "mastery", "purpose"]

        return triggers

    async def initialize_knowledge_states(self, student_id: str, grade_level: int) -> None:
        """Initialize knowledge states for grade-appropriate concepts."""
        # Import curriculum graph
        from hyperag.education.curriculum_graph import curriculum_graph

        # Get concepts for grade level
        grade_concepts = await curriculum_graph.get_concepts_by_grade(grade_level)

        for subject, concepts in grade_concepts.items():
            for concept in concepts:
                knowledge_state = KnowledgeState(
                    student_id=student_id,
                    subject=subject,
                    concept=concept,
                    mastery_level=0.0,  # Unknown initially
                    confidence_score=0.0,
                    last_practiced="never",
                    practice_count=0,
                    mistake_patterns=[],
                    prerequisite_gaps=[],
                    next_recommended=[],
                    estimated_study_time=30,  # Default estimate
                    retention_decay_rate=0.1,  # Will be personalized
                )

                self.knowledge_states[student_id][concept] = knowledge_state

        logger.info(
            f"Initialized knowledge states for student {student_id[:8]} with {len(self.knowledge_states[student_id])} concepts"
        )

    async def record_learning_session(self, session: LearningSession) -> None:
        """Record and analyze a learning session."""
        # Store session
        self.session_history[session.student_id].append(session)

        # Update knowledge states based on session
        await self.update_knowledge_states_from_session(session)

        # Update personalization vector
        await self.update_personalization_vector(session)

        # Update learning patterns
        self.update_learning_patterns(session)

        # Save to database
        await self.save_learning_session(session)

        # Log to W&B
        wandb.log(
            {
                "session/duration_minutes": session.duration_minutes,
                "session/engagement_score": session.engagement_score,
                "session/accuracy": session.questions_correct
                / max(session.questions_asked, 1),
                "session/concepts_covered": len(session.concepts_covered),
                "session/student_age": self.students[session.student_id].age,
                "session/difficulty_level": session.difficulty_level,
                "session/adaptations_made": len(session.adaptations_made),
                "sessions/total_count": len(self.session_history[session.student_id]),
                "timestamp": session.end_time,
            }
        )

        logger.info(
            f"Recorded learning session {session.session_id[:8]} for student {session.student_id[:8]}"
        )

    async def update_knowledge_states_from_session(self, session: LearningSession) -> None:
        """Update knowledge states based on session performance."""
        accuracy = session.questions_correct / max(session.questions_asked, 1)

        for concept in session.concepts_covered:
            if concept in self.knowledge_states[session.student_id]:
                state = self.knowledge_states[session.student_id][concept]

                # Update mastery level using exponential moving average
                alpha = 0.3  # Learning rate
                state.mastery_level = (
                    1 - alpha
                ) * state.mastery_level + alpha * accuracy

                # Update confidence based on consistency
                confidence_adjustment = 0.1 if accuracy > 0.7 else -0.05
                state.confidence_score = max(
                    0, min(1, state.confidence_score + confidence_adjustment)
                )

                # Update practice count and time
                state.practice_count += 1
                state.last_practiced = session.end_time

                # Estimate retention decay based on performance
                if accuracy < 0.5:
                    state.retention_decay_rate = min(
                        0.3, state.retention_decay_rate + 0.02
                    )
                else:
                    state.retention_decay_rate = max(
                        0.05, state.retention_decay_rate - 0.01
                    )

                # Update estimated study time
                if accuracy > 0.8:
                    state.estimated_study_time = max(
                        10, int(state.estimated_study_time * 0.9)
                    )
                elif accuracy < 0.5:
                    state.estimated_study_time = min(
                        60, int(state.estimated_study_time * 1.2)
                    )

    async def update_personalization_vector(self, session: LearningSession) -> None:
        """Update personalization vector based on session outcomes."""
        if session.student_id not in self.personalization_vectors:
            return

        vector = self.personalization_vectors[session.student_id]
        learning_rate = 0.1

        # Update based on engagement and performance
        engagement = session.engagement_score
        accuracy = session.questions_correct / max(session.questions_asked, 1)

        # Adjust pacing preference based on session duration vs planned
        planned_duration = self.students[session.student_id].attention_span_minutes
        actual_duration = session.duration_minutes

        if actual_duration > planned_duration and engagement > 0.7:
            # Student can handle longer sessions
            vector.pacing_preference = min(
                1.0, vector.pacing_preference + learning_rate * 0.1
            )
        elif actual_duration < planned_duration * 0.7:
            # Student needs shorter sessions
            vector.pacing_preference = max(
                0.0, vector.pacing_preference - learning_rate * 0.1
            )

        # Adjust challenge seeking based on difficulty vs performance
        if session.difficulty_level > 0.7 and accuracy > 0.6:
            vector.challenge_seeking = min(
                1.0, vector.challenge_seeking + learning_rate * 0.2
            )
        elif session.difficulty_level < 0.4 and engagement < 0.5:
            vector.challenge_seeking = min(
                1.0, vector.challenge_seeking + learning_rate * 0.1
            )

        # Adjust explanation depth based on adaptations made
        explanation_adaptations = [
            a for a in session.adaptations_made if "explanation" in a.lower()
        ]
        if explanation_adaptations and engagement > 0.7:
            if "more_detail" in " ".join(explanation_adaptations):
                vector.explanation_depth = min(
                    1.0, vector.explanation_depth + learning_rate * 0.1
                )
            elif "simpler" in " ".join(explanation_adaptations):
                vector.explanation_depth = max(
                    0.0, vector.explanation_depth - learning_rate * 0.1
                )

        # Update gamification response
        gamification_elements = [
            a
            for a in session.adaptations_made
            if any(word in a.lower() for word in ["game", "points", "badge", "reward"])
        ]
        if gamification_elements:
            if engagement > 0.8:
                vector.gamification_response = min(
                    1.0, vector.gamification_response + learning_rate * 0.15
                )
            elif engagement < 0.4:
                vector.gamification_response = max(
                    0.0, vector.gamification_response - learning_rate * 0.1
                )

    def update_learning_patterns(self, session: LearningSession) -> None:
        """Update detected learning patterns."""
        student_id = session.student_id

        if student_id not in self.learning_patterns:
            self.learning_patterns[student_id] = {
                "peak_performance_times": [],
                "optimal_session_length": [],
                "difficulty_progression": [],
                "engagement_factors": [],
                "common_mistakes": [],
            }

        patterns = self.learning_patterns[student_id]

        # Track peak performance times
        session_start = datetime.fromisoformat(session.start_time)
        hour_of_day = session_start.hour
        accuracy = session.questions_correct / max(session.questions_asked, 1)

        patterns["peak_performance_times"].append((hour_of_day, accuracy))

        # Track optimal session length
        patterns["optimal_session_length"].append(
            (session.duration_minutes, session.engagement_score)
        )

        # Track difficulty progression
        patterns["difficulty_progression"].append((session.difficulty_level, accuracy))

        # Track engagement factors
        patterns["engagement_factors"].append(
            {
                "adaptations": session.adaptations_made,
                "engagement": session.engagement_score,
                "concepts": session.concepts_covered,
            }
        )

        # Limit pattern history to recent data
        max_history = 50
        for pattern_type in patterns:
            patterns[pattern_type] = patterns[pattern_type][-max_history:]

    async def get_personalized_recommendations(self, student_id: str) -> dict[str, Any]:
        """Generate personalized learning recommendations."""
        if student_id not in self.students:
            return {"error": "Student not found"}

        student = self.students[student_id]
        vector = self.personalization_vectors.get(student_id)
        knowledge = self.knowledge_states.get(student_id, {})

        recommendations = {
            "next_concepts": [],
            "difficulty_level": 0.5,
            "session_duration": student.attention_span_minutes,
            "learning_style_adaptations": [],
            "content_format": "mixed",
            "gamification_elements": [],
            "cultural_context": student.region,
            "motivation_strategy": student.motivation_triggers,
            "parent_notification": None,
        }

        # Find concepts ready for learning
        ready_concepts = []
        struggling_concepts = []

        for concept, state in knowledge.items():
            if state.mastery_level < 0.3:
                if not state.prerequisite_gaps:
                    ready_concepts.append((concept, state.mastery_level))
                else:
                    struggling_concepts.append((concept, state.mastery_level))
            elif 0.3 <= state.mastery_level < 0.7:
                ready_concepts.append((concept, state.mastery_level))

        # Sort by mastery level (focus on partially learned concepts first)
        ready_concepts.sort(key=lambda x: x[1], reverse=True)
        recommendations["next_concepts"] = [
            concept for concept, _ in ready_concepts[:3]
        ]

        # Determine optimal difficulty based on recent performance
        recent_sessions = self.session_history[student_id][-5:]  # Last 5 sessions
        if recent_sessions:
            avg_accuracy = np.mean(
                [
                    s.questions_correct / max(s.questions_asked, 1)
                    for s in recent_sessions
                ]
            )

            if avg_accuracy > 0.8:
                recommendations["difficulty_level"] = min(
                    1.0, 0.6 + vector.challenge_seeking * 0.3
                )
            elif avg_accuracy < 0.5:
                recommendations["difficulty_level"] = max(
                    0.2, 0.4 - (0.5 - avg_accuracy) * 0.5
                )
            else:
                recommendations["difficulty_level"] = 0.5

        # Adapt to learning style
        if student.learning_style == "visual" or vector.visual_learning_weight > 0.7:
            recommendations["learning_style_adaptations"].extend(
                ["use_diagrams", "color_coding", "visual_analogies"]
            )
            recommendations["content_format"] = "visual_heavy"

        if (
            student.learning_style == "kinesthetic"
            or vector.kinesthetic_learning_weight > 0.7
        ):
            recommendations["learning_style_adaptations"].extend(
                ["hands_on_activities", "physical_manipulatives", "movement_breaks"]
            )

        # Gamification recommendations
        if vector.gamification_response > 0.6:
            recommendations["gamification_elements"] = [
                "progress_badges",
                "point_system",
                "mini_challenges",
            ]
            if student.age <= 10:
                recommendations["gamification_elements"].append("avatar_customization")

        # Session duration optimization
        patterns = self.learning_patterns.get(student_id, {})
        if patterns.get("optimal_session_length"):
            # Find session length that maximizes engagement
            session_data = patterns["optimal_session_length"]
            best_duration = max(session_data, key=lambda x: x[1])[0]
            recommendations["session_duration"] = min(
                student.parent_constraints.get("max_session_minutes", 60), best_duration
            )

        # Parent notification strategy
        if len(recent_sessions) >= 3:
            avg_engagement = np.mean([s.engagement_score for s in recent_sessions])
            if avg_engagement < 0.4:
                recommendations["parent_notification"] = {
                    "type": "support_needed",
                    "message": f"{student.name} seems to be struggling with engagement. Consider shorter sessions or different topics.",
                }
            elif avg_engagement > 0.8:
                recommendations["parent_notification"] = {
                    "type": "progress_celebration",
                    "message": f"{student.name} is doing great! High engagement across recent sessions.",
                }

        return recommendations

    async def adapt_in_real_time(
        self,
        student_id: str,
        current_engagement: float,
        current_accuracy: float,
        time_in_session: int,
    ) -> dict[str, Any]:
        """Provide real-time adaptations during a session."""
        adaptations = {
            "difficulty_adjustment": 0,  # -1 easier, 0 same, 1 harder
            "explanation_style": "maintain",  # simpler, maintain, detailed
            "encouragement_level": "normal",  # high, normal, low
            "break_recommendation": False,
            "content_switch": None,  # Switch to different concept/activity
            "gamification_boost": False,
        }

        student = self.students.get(student_id)
        vector = self.personalization_vectors.get(student_id)

        if not student or not vector:
            return adaptations

        # Engagement-based adaptations
        if current_engagement < 0.3:
            # Low engagement - need immediate intervention
            adaptations["encouragement_level"] = "high"
            adaptations["gamification_boost"] = vector.gamification_response > 0.5

            if time_in_session > student.attention_span_minutes * 0.8:
                adaptations["break_recommendation"] = True
            elif current_accuracy < 0.4:
                adaptations["difficulty_adjustment"] = -1
                adaptations["explanation_style"] = "simpler"
            else:
                # Engagement low but accuracy OK - try different content
                adaptations["content_switch"] = "more_interactive"

        elif current_engagement > 0.8 and current_accuracy > 0.7:
            # High engagement and performance - can challenge more
            if vector.challenge_seeking > 0.6:
                adaptations["difficulty_adjustment"] = 1
            if vector.explanation_depth > 0.7:
                adaptations["explanation_style"] = "detailed"

        # Accuracy-based adaptations
        if current_accuracy < 0.4:
            adaptations["difficulty_adjustment"] = min(
                -1, adaptations["difficulty_adjustment"] - 1
            )
            adaptations["explanation_style"] = "simpler"
            adaptations["encouragement_level"] = "high"
        elif current_accuracy > 0.9:
            adaptations["difficulty_adjustment"] = max(
                1, adaptations["difficulty_adjustment"] + 1
            )

        # Time-based adaptations
        if time_in_session > student.attention_span_minutes:
            if current_engagement > 0.6:
                # Good engagement despite long session - minor break
                adaptations["break_recommendation"] = (
                    time_in_session > student.attention_span_minutes * 1.2
                )
            else:
                # Poor engagement and long session - definitely break
                adaptations["break_recommendation"] = True

        return adaptations

    async def save_student_profile(self, student_id: str) -> None:
        """Save student profile to encrypted database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Encrypt profile data
            profile_json = json.dumps(asdict(self.students[student_id]))
            encrypted_data = self.cipher_suite.encrypt(profile_json.encode()).decode()

            cursor.execute(
                """
                INSERT OR REPLACE INTO students (student_id, profile_data, created_at, last_updated)
                VALUES (?, ?, ?, ?)
            """,
                (
                    student_id,
                    encrypted_data,
                    self.students[student_id].created_at,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

            # Save personalization vector
            if student_id in self.personalization_vectors:
                vector_json = json.dumps(
                    asdict(self.personalization_vectors[student_id])
                )
                encrypted_vector = self.cipher_suite.encrypt(
                    vector_json.encode()
                ).decode()

                cursor.execute(
                    """
                    INSERT OR REPLACE INTO personalization_vectors (student_id, vector_data, last_updated)
                    VALUES (?, ?, ?)
                """,
                    (
                        student_id,
                        encrypted_vector,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )

            conn.commit()
            conn.close()

        except Exception as e:
            logger.exception(f"Failed to save student profile: {e}")

    async def save_learning_session(self, session: LearningSession) -> None:
        """Save learning session to encrypted database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Encrypt session data
            session_json = json.dumps(asdict(session))
            encrypted_data = self.cipher_suite.encrypt(session_json.encode()).decode()

            cursor.execute(
                """
                INSERT OR REPLACE INTO learning_sessions
                (session_id, student_id, session_data, start_time, end_time, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    session.session_id,
                    session.student_id,
                    encrypted_data,
                    session.start_time,
                    session.end_time,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            logger.exception(f"Failed to save learning session: {e}")

    async def start_background_analytics(self) -> None:
        """Start background analytics and pattern detection."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes

                # Update retention decay for all students
                await self.update_retention_decay()

                # Detect learning patterns
                await self.detect_learning_patterns()

                # Generate insights
                await self.generate_learning_insights()

            except Exception as e:
                logger.exception(f"Error in background analytics: {e}")
                await asyncio.sleep(60)

    async def update_retention_decay(self) -> None:
        """Update knowledge retention based on time decay."""
        current_time = datetime.now(timezone.utc)

        for knowledge_map in self.knowledge_states.values():
            for state in knowledge_map.values():
                if state.last_practiced != "never":
                    last_practice = datetime.fromisoformat(state.last_practiced)
                    days_since = (current_time - last_practice).days

                    if days_since > 0:
                        # Apply exponential decay
                        decay_factor = np.exp(-state.retention_decay_rate * days_since)
                        state.mastery_level *= decay_factor
                        state.confidence_score *= decay_factor

    async def detect_learning_patterns(self) -> None:
        """Detect and update learning patterns for all students."""
        for student_id in self.students:
            if student_id not in self.learning_patterns:
                continue

            patterns = self.learning_patterns[student_id]

            # Analyze peak performance times
            if patterns.get("peak_performance_times"):
                time_performance = patterns["peak_performance_times"]

                # Group by hour and calculate average performance
                hour_performance = defaultdict(list)
                for hour, accuracy in time_performance:
                    hour_performance[hour].append(accuracy)

                best_hours = []
                for hour, accuracies in hour_performance.items():
                    avg_accuracy = np.mean(accuracies)
                    if avg_accuracy > 0.7:
                        best_hours.append(hour)

                # Update student's preferred session times
                if best_hours:
                    preferred_times = []
                    for hour in best_hours:
                        if 6 <= hour < 12:
                            preferred_times.append("morning")
                        elif 12 <= hour < 17:
                            preferred_times.append("afternoon")
                        elif 17 <= hour < 21:
                            preferred_times.append("evening")

                    self.students[student_id].preferred_session_times = list(
                        set(preferred_times)
                    )

    async def generate_learning_insights(self) -> None:
        """Generate learning insights and log to W&B."""
        total_students = len(self.students)
        if total_students == 0:
            return

        # Aggregate insights across all students
        insights = {
            "total_students": total_students,
            "avg_session_duration": 0,
            "avg_engagement": 0,
            "avg_accuracy": 0,
            "learning_style_distribution": defaultdict(int),
            "concept_difficulty_ranking": defaultdict(list),
            "retention_patterns": {},
            "adaptation_effectiveness": {},
        }

        all_sessions = []
        for sessions in self.session_history.values():
            all_sessions.extend(sessions)

        if all_sessions:
            insights["avg_session_duration"] = np.mean(
                [s.duration_minutes for s in all_sessions]
            )
            insights["avg_engagement"] = np.mean(
                [s.engagement_score for s in all_sessions]
            )
            insights["avg_accuracy"] = np.mean(
                [s.questions_correct / max(s.questions_asked, 1) for s in all_sessions]
            )

        # Learning style distribution
        for student in self.students.values():
            insights["learning_style_distribution"][student.learning_style] += 1

        # Log insights to W&B
        wandb.log(
            {
                "insights/total_students": insights["total_students"],
                "insights/avg_session_duration": insights["avg_session_duration"],
                "insights/avg_engagement": insights["avg_engagement"],
                "insights/avg_accuracy": insights["avg_accuracy"],
                "insights/visual_learners": insights["learning_style_distribution"][
                    "visual"
                ],
                "insights/kinesthetic_learners": insights[
                    "learning_style_distribution"
                ]["kinesthetic"],
                "insights/auditory_learners": insights["learning_style_distribution"][
                    "auditory"
                ],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def get_student_dashboard(self, student_id: str) -> dict[str, Any]:
        """Generate comprehensive student dashboard data."""
        if student_id not in self.students:
            return {"error": "Student not found"}

        student = self.students[student_id]
        sessions = self.session_history.get(student_id, [])
        knowledge = self.knowledge_states.get(student_id, {})

        # Calculate progress metrics
        total_concepts = len(knowledge)
        mastered_concepts = len(
            [k for k in knowledge.values() if k.mastery_level >= 0.8]
        )
        in_progress_concepts = len(
            [k for k in knowledge.values() if 0.3 <= k.mastery_level < 0.8]
        )

        # Recent performance
        recent_sessions = sessions[-10:] if len(sessions) >= 10 else sessions
        avg_recent_engagement = (
            np.mean([s.engagement_score for s in recent_sessions])
            if recent_sessions
            else 0
        )
        avg_recent_accuracy = (
            np.mean(
                [
                    s.questions_correct / max(s.questions_asked, 1)
                    for s in recent_sessions
                ]
            )
            if recent_sessions
            else 0
        )

        # Study time analysis
        total_study_time = sum(s.duration_minutes for s in sessions)
        sessions_this_week = [
            s
            for s in sessions
            if (datetime.now(timezone.utc) - datetime.fromisoformat(s.start_time)).days
            <= 7
        ]
        study_time_this_week = sum(s.duration_minutes for s in sessions_this_week)

        dashboard = {
            "student_info": {
                "name": student.name,
                "age": student.age,
                "grade_level": student.grade_level,
                "learning_style": student.learning_style,
                "strengths": student.strengths,
                "interests": student.interests,
            },
            "progress_summary": {
                "total_concepts": total_concepts,
                "mastered_concepts": mastered_concepts,
                "in_progress_concepts": in_progress_concepts,
                "mastery_percentage": (mastered_concepts / max(total_concepts, 1))
                * 100,
            },
            "recent_performance": {
                "sessions_completed": len(sessions),
                "avg_engagement": avg_recent_engagement,
                "avg_accuracy": avg_recent_accuracy,
                "improvement_trend": self.calculate_improvement_trend(sessions),
            },
            "study_habits": {
                "total_study_time_minutes": total_study_time,
                "study_time_this_week": study_time_this_week,
                "avg_session_length": np.mean([s.duration_minutes for s in sessions])
                if sessions
                else 0,
                "preferred_times": student.preferred_session_times,
                "consistency_score": self.calculate_consistency_score(sessions),
            },
            "next_recommendations": asyncio.create_task(
                self.get_personalized_recommendations(student_id)
            ),
            "achievements": self.get_student_achievements(student_id),
            "parent_insights": self.get_parent_insights(student_id),
        }

        return dashboard

    def calculate_improvement_trend(self, sessions: list[LearningSession]) -> str:
        """Calculate improvement trend from session history."""
        if len(sessions) < 3:
            return "insufficient_data"

        # Compare first half vs second half of recent sessions
        mid_point = len(sessions) // 2
        first_half = sessions[:mid_point]
        second_half = sessions[mid_point:]

        first_half_accuracy = np.mean(
            [s.questions_correct / max(s.questions_asked, 1) for s in first_half]
        )
        second_half_accuracy = np.mean(
            [s.questions_correct / max(s.questions_asked, 1) for s in second_half]
        )

        improvement = second_half_accuracy - first_half_accuracy

        if improvement > 0.1:
            return "improving"
        if improvement < -0.1:
            return "declining"
        return "stable"

    def calculate_consistency_score(self, sessions: list[LearningSession]) -> float:
        """Calculate consistency score based on regular study patterns."""
        if len(sessions) < 5:
            return 0.5

        # Calculate days between sessions
        session_dates = [
            datetime.fromisoformat(s.start_time).date() for s in sessions[-10:]
        ]
        session_dates.sort()

        day_gaps = [
            (session_dates[i] - session_dates[i - 1]).days
            for i in range(1, len(session_dates))
        ]

        if not day_gaps:
            return 0.5

        # Consistency is higher when gaps are more regular
        gap_variance = np.var(day_gaps)
        consistency = max(0, 1 - gap_variance / 10)  # Normalize variance

        return min(1.0, consistency)

    def get_student_achievements(self, student_id: str) -> list[dict[str, Any]]:
        """Get student achievements and badges."""
        achievements = []
        sessions = self.session_history.get(student_id, [])
        knowledge = self.knowledge_states.get(student_id, {})

        # Mastery achievements
        mastered_count = len([k for k in knowledge.values() if k.mastery_level >= 0.8])
        if mastered_count >= 5:
            achievements.append(
                {
                    "type": "mastery",
                    "title": "Concept Master",
                    "description": f"Mastered {mastered_count} concepts!",
                    "earned_date": datetime.now().isoformat(),
                }
            )

        # Consistency achievements
        if len(sessions) >= 7:
            recent_week = [
                s
                for s in sessions
                if (
                    datetime.now(timezone.utc) - datetime.fromisoformat(s.start_time)
                ).days
                <= 7
            ]
            if len(recent_week) >= 5:
                achievements.append(
                    {
                        "type": "consistency",
                        "title": "Daily Learner",
                        "description": "Studied 5 days this week!",
                        "earned_date": datetime.now().isoformat(),
                    }
                )

        # Improvement achievements
        if len(sessions) >= 10:
            trend = self.calculate_improvement_trend(sessions)
            if trend == "improving":
                achievements.append(
                    {
                        "type": "improvement",
                        "title": "Rising Star",
                        "description": "Showing great improvement!",
                        "earned_date": datetime.now().isoformat(),
                    }
                )

        return achievements

    def get_parent_insights(self, student_id: str) -> dict[str, Any]:
        """Generate insights for parents."""
        student = self.students[student_id]
        sessions = self.session_history.get(student_id, [])

        insights = {
            "weekly_summary": {
                "sessions_completed": 0,
                "total_study_time": 0,
                "avg_engagement": 0,
                "concepts_practiced": [],
            },
            "strengths_observed": [],
            "areas_for_support": [],
            "recommended_activities": [],
            "screen_time_usage": {
                "daily_average": 0,
                "weekly_total": 0,
                "within_limits": True,
            },
        }

        # Weekly summary
        week_sessions = [
            s
            for s in sessions
            if (datetime.now(timezone.utc) - datetime.fromisoformat(s.start_time)).days
            <= 7
        ]

        if week_sessions:
            insights["weekly_summary"]["sessions_completed"] = len(week_sessions)
            insights["weekly_summary"]["total_study_time"] = sum(
                s.duration_minutes for s in week_sessions
            )
            insights["weekly_summary"]["avg_engagement"] = np.mean(
                [s.engagement_score for s in week_sessions]
            )

            all_concepts = []
            for s in week_sessions:
                all_concepts.extend(s.concepts_covered)
            insights["weekly_summary"]["concepts_practiced"] = list(set(all_concepts))

        # Screen time analysis
        daily_limit = student.parent_constraints.get("max_daily_minutes", 60)
        weekly_total = insights["weekly_summary"]["total_study_time"]
        daily_average = weekly_total / 7

        insights["screen_time_usage"] = {
            "daily_average": daily_average,
            "weekly_total": weekly_total,
            "within_limits": daily_average <= daily_limit,
            "daily_limit": daily_limit,
        }

        # Strengths and support areas
        if week_sessions:
            avg_accuracy = np.mean(
                [s.questions_correct / max(s.questions_asked, 1) for s in week_sessions]
            )

            if avg_accuracy > 0.8:
                insights["strengths_observed"].append(
                    "High accuracy in problem solving"
                )
            if insights["weekly_summary"]["avg_engagement"] > 0.7:
                insights["strengths_observed"].append(
                    "Strong engagement with learning material"
                )

            if avg_accuracy < 0.5:
                insights["areas_for_support"].append(
                    "May need additional practice with current concepts"
                )
            if insights["weekly_summary"]["avg_engagement"] < 0.4:
                insights["areas_for_support"].append(
                    "Consider varying activity types to maintain interest"
                )

        return insights


# Global digital twin instance - initialize only when running directly
if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        DigitalTwin()
        print("✅ Digital Twin initialized successfully")

    asyncio.run(main())
else:
    # For imports, create factory function
    def create_digital_twin():
        return DigitalTwin()
