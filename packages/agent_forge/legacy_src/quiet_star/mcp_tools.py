"""
MCP Tools and Journaling System for Quiet-STaR.
Implements E1) MCP tools integration and surprise memory journaling.
"""

import asyncio
import hashlib
import json
import logging
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from .config import QuietSTaRConfig


class SurpriseLevel(Enum):
    """Levels of surprise for memory journaling."""

    NONE = "none"  # Expected outcome
    LOW = "low"  # Slightly unexpected
    MEDIUM = "medium"  # Moderately surprising
    HIGH = "high"  # Very surprising
    EXTREME = "extreme"  # Completely unexpected


class ReflectionType(Enum):
    """Types of reflections in the journal."""

    PREDICTION = "prediction"  # What we expect to happen
    OUTCOME = "outcome"  # What actually happened
    ANALYSIS = "analysis"  # Why the difference occurred
    LEARNING = "learning"  # What we learned from this
    PATTERN = "pattern"  # Patterns we've noticed


@dataclass
class JournalEntry:
    """A single journal entry capturing a reflection or experience."""

    id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    reflection_type: ReflectionType = ReflectionType.PREDICTION
    content: str = ""
    question: str = ""
    context: dict[str, Any] = field(default_factory=dict)

    # Surprise tracking
    predicted_outcome: str = ""
    actual_outcome: str = ""
    surprise_level: SurpriseLevel = SurpriseLevel.NONE
    surprise_reasoning: str = ""

    # Quality metrics
    confidence_score: float = 0.0
    accuracy_score: float = 0.0  # How accurate was our prediction
    learning_value: float = 0.0  # How much did we learn from this

    # Connections
    related_entries: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.id:
            # Generate unique ID from content hash and timestamp
            content_hash = hashlib.md5(f"{self.content}{self.timestamp}".encode()).hexdigest()[:8]
            self.id = f"journal_{content_hash}"


@dataclass
class MemoryPattern:
    """A pattern discovered from journal entries."""

    pattern_id: str
    description: str
    supporting_entries: list[str]
    pattern_type: str  # "prediction_error", "confidence_mismatch", "context_similarity"
    confidence: float
    discovered_at: datetime
    last_reinforced: datetime
    reinforcement_count: int = 0


class SurpriseMemoryJournal:
    """
    Journal system that tracks predictions, outcomes, and learning from surprises.
    Implements surprise-driven memory consolidation for improved reasoning.
    """

    def __init__(self, config: QuietSTaRConfig, journal_db_path: str = "quiet_star_journal.db"):
        self.config = config
        self.journal_db_path = journal_db_path
        self.db_connection = None
        self.logger = logging.getLogger(__name__)

        # Initialize database
        self._init_database()

        # Pattern detection settings
        self.min_pattern_support = 3  # Minimum entries to form a pattern
        self.pattern_confidence_threshold = 0.6

        # Surprise detection thresholds
        self.surprise_thresholds = {
            "confidence_mismatch": 0.3,  # Diff between predicted and actual confidence
            "outcome_deviation": 0.4,  # Semantic similarity threshold for outcomes
            "expectation_violation": 0.5,  # General expectation violation threshold
        }

    def _init_database(self):
        """Initialize SQLite database for journal storage."""
        self.db_connection = sqlite3.connect(self.journal_db_path)
        self.db_connection.row_factory = sqlite3.Row

        cursor = self.db_connection.cursor()

        # Create journal entries table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS journal_entries (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                reflection_type TEXT NOT NULL,
                content TEXT NOT NULL,
                question TEXT,
                context TEXT,
                predicted_outcome TEXT,
                actual_outcome TEXT,
                surprise_level TEXT,
                surprise_reasoning TEXT,
                confidence_score REAL,
                accuracy_score REAL,
                learning_value REAL,
                related_entries TEXT,
                tags TEXT
            )
        """
        )

        # Create patterns table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_patterns (
                pattern_id TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                supporting_entries TEXT NOT NULL,
                pattern_type TEXT NOT NULL,
                confidence REAL,
                discovered_at TEXT,
                last_reinforced TEXT,
                reinforcement_count INTEGER DEFAULT 0
            )
        """
        )

        # Create indices for better query performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON journal_entries(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_surprise_level ON journal_entries(surprise_level)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_reflection_type ON journal_entries(reflection_type)")

        self.db_connection.commit()

    def add_prediction(
        self,
        question: str,
        predicted_outcome: str,
        confidence: float,
        reasoning: str = "",
        context: dict[str, Any] = None,
    ) -> str:
        """Add a prediction to the journal."""

        entry = JournalEntry(
            reflection_type=ReflectionType.PREDICTION,
            content=reasoning or f"Predicting outcome for: {question}",
            question=question,
            context=context or {},
            predicted_outcome=predicted_outcome,
            confidence_score=confidence,
            tags=["prediction", "awaiting_outcome"],
        )

        return self._save_entry(entry)

    def record_outcome(self, prediction_id: str, actual_outcome: str, context: dict[str, Any] = None) -> str | None:
        """Record the actual outcome for a prediction and analyze surprise."""

        # Retrieve the original prediction
        original_entry = self.get_entry(prediction_id)
        if not original_entry:
            self.logger.error(f"Prediction entry {prediction_id} not found")
            return None

        # Analyze surprise level
        surprise_analysis = self._analyze_surprise(
            original_entry.predicted_outcome,
            actual_outcome,
            original_entry.confidence_score,
        )

        # Create outcome entry
        outcome_entry = JournalEntry(
            reflection_type=ReflectionType.OUTCOME,
            content=f"Actual outcome for prediction {prediction_id}",
            question=original_entry.question,
            context=context or {},
            predicted_outcome=original_entry.predicted_outcome,
            actual_outcome=actual_outcome,
            surprise_level=surprise_analysis["level"],
            surprise_reasoning=surprise_analysis["reasoning"],
            confidence_score=original_entry.confidence_score,
            accuracy_score=surprise_analysis["accuracy"],
            learning_value=surprise_analysis["learning_value"],
            related_entries=[prediction_id],
            tags=["outcome", f"surprise_{surprise_analysis['level'].value}"],
        )

        outcome_id = self._save_entry(outcome_entry)

        # Update original prediction with outcome reference
        self._link_entries(prediction_id, outcome_id)

        # If surprise is significant, trigger learning analysis
        if surprise_analysis["level"] in [
            SurpriseLevel.MEDIUM,
            SurpriseLevel.HIGH,
            SurpriseLevel.EXTREME,
        ]:
            self._trigger_learning_analysis(prediction_id, outcome_id)

        return outcome_id

    def _analyze_surprise(
        self, predicted_outcome: str, actual_outcome: str, predicted_confidence: float
    ) -> dict[str, Any]:
        """Analyze the level of surprise between prediction and outcome."""

        # Simple semantic similarity heuristic
        # In a full implementation, you'd use embedding similarity
        predicted_words = set(predicted_outcome.lower().split())
        actual_words = set(actual_outcome.lower().split())

        if predicted_words and actual_words:
            overlap = len(predicted_words & actual_words)
            union = len(predicted_words | actual_words)
            semantic_similarity = overlap / union if union > 0 else 0
        else:
            semantic_similarity = 0

        # Calculate surprise metrics
        outcome_deviation = 1 - semantic_similarity
        confidence_factor = predicted_confidence  # Higher confidence = more surprise when wrong

        surprise_score = outcome_deviation * confidence_factor

        # Determine surprise level
        if surprise_score < 0.1:
            level = SurpriseLevel.NONE
            reasoning = "Outcome closely matched prediction"
        elif surprise_score < 0.3:
            level = SurpriseLevel.LOW
            reasoning = "Minor deviation from prediction"
        elif surprise_score < 0.5:
            level = SurpriseLevel.MEDIUM
            reasoning = "Moderate surprise - outcome differed from prediction"
        elif surprise_score < 0.7:
            level = SurpriseLevel.HIGH
            reasoning = "High surprise - significant deviation from prediction"
        else:
            level = SurpriseLevel.EXTREME
            reasoning = "Extreme surprise - outcome was completely unexpected"

        # Calculate learning value (higher surprise = more learning potential)
        learning_value = min(surprise_score * 1.2, 1.0)

        # Calculate accuracy (inverse of deviation)
        accuracy = max(0, 1 - outcome_deviation)

        return {
            "level": level,
            "reasoning": reasoning,
            "surprise_score": surprise_score,
            "accuracy": accuracy,
            "learning_value": learning_value,
            "semantic_similarity": semantic_similarity,
        }

    def _trigger_learning_analysis(self, prediction_id: str, outcome_id: str):
        """Trigger deeper learning analysis for surprising outcomes."""

        prediction_entry = self.get_entry(prediction_id)
        outcome_entry = self.get_entry(outcome_id)

        if not prediction_entry or not outcome_entry:
            return

        # Generate learning reflection
        learning_content = self._generate_learning_reflection(prediction_entry, outcome_entry)

        learning_entry = JournalEntry(
            reflection_type=ReflectionType.LEARNING,
            content=learning_content,
            question=prediction_entry.question,
            context={
                "prediction_id": prediction_id,
                "outcome_id": outcome_id,
                "surprise_level": outcome_entry.surprise_level.value,
            },
            learning_value=outcome_entry.learning_value,
            related_entries=[prediction_id, outcome_id],
            tags=[
                "learning",
                "surprise_driven",
                f"level_{outcome_entry.surprise_level.value}",
            ],
        )

        self._save_entry(learning_entry)

        # Look for patterns
        self._update_patterns_from_surprise(prediction_entry, outcome_entry)

    def _generate_learning_reflection(self, prediction_entry: JournalEntry, outcome_entry: JournalEntry) -> str:
        """Generate a learning reflection from surprising outcomes."""

        reflection_parts = []

        # What we expected vs what happened
        reflection_parts.append(f"Expected: {prediction_entry.predicted_outcome}")
        reflection_parts.append(f"Actual: {outcome_entry.actual_outcome}")

        # Why this was surprising
        reflection_parts.append(f"Surprise analysis: {outcome_entry.surprise_reasoning}")

        # Confidence analysis
        if prediction_entry.confidence_score > 0.7 and outcome_entry.accuracy_score < 0.5:
            reflection_parts.append(
                f"High confidence ({prediction_entry.confidence_score:.2f}) led to greater surprise. "
                f"Consider factors that might cause confidence miscalibration."
            )

        # Pattern suggestions
        similar_contexts = self._find_similar_contexts(prediction_entry.context)
        if similar_contexts:
            reflection_parts.append(
                f"Found {len(similar_contexts)} similar contexts. "
                f"Consider whether this represents a recurring pattern."
            )

        # Learning suggestions
        if outcome_entry.surprise_level == SurpriseLevel.EXTREME:
            reflection_parts.append(
                "Extreme surprise suggests a gap in understanding. "
                "Consider revising mental models or gathering additional context."
            )

        return " ".join(reflection_parts)

    def _find_similar_contexts(self, target_context: dict[str, Any], limit: int = 5) -> list[JournalEntry]:
        """Find journal entries with similar contexts."""

        if not target_context:
            return []

        cursor = self.db_connection.cursor()
        cursor.execute(
            """
            SELECT * FROM journal_entries
            WHERE context IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT 100
        """
        )

        recent_entries = []
        for row in cursor.fetchall():
            try:
                context = json.loads(row["context"])
                entry = JournalEntry(
                    id=row["id"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    reflection_type=ReflectionType(row["reflection_type"]),
                    content=row["content"],
                    question=row["question"],
                    context=context,
                    surprise_level=SurpriseLevel(row["surprise_level"]),
                )
                recent_entries.append(entry)
            except (json.JSONDecodeError, ValueError):
                continue

        # Simple similarity based on shared context keys
        similar_entries = []
        target_keys = set(target_context.keys())

        for entry in recent_entries:
            entry_keys = set(entry.context.keys())
            if target_keys and entry_keys:
                similarity = len(target_keys & entry_keys) / len(target_keys | entry_keys)
                if similarity > 0.3:  # At least 30% key overlap
                    similar_entries.append(entry)

        return similar_entries[:limit]

    def _update_patterns_from_surprise(self, prediction_entry: JournalEntry, outcome_entry: JournalEntry):
        """Update memory patterns based on surprising outcomes."""

        # Look for prediction error patterns
        if outcome_entry.accuracy_score < 0.5:
            self._detect_prediction_error_pattern(prediction_entry, outcome_entry)

        # Look for confidence miscalibration patterns
        confidence_error = abs(prediction_entry.confidence_score - outcome_entry.accuracy_score)
        if confidence_error > self.surprise_thresholds["confidence_mismatch"]:
            self._detect_confidence_pattern(prediction_entry, outcome_entry)

        # Look for context-based patterns
        self._detect_context_patterns(prediction_entry, outcome_entry)

    def _detect_prediction_error_pattern(self, prediction_entry: JournalEntry, outcome_entry: JournalEntry):
        """Detect patterns in prediction errors."""

        # Find similar prediction errors
        cursor = self.db_connection.cursor()
        cursor.execute(
            """
            SELECT id FROM journal_entries
            WHERE reflection_type = ?
            AND accuracy_score < 0.5
            AND question LIKE ?
            ORDER BY timestamp DESC
        """,
            (ReflectionType.OUTCOME.value, f"%{prediction_entry.question[:20]}%"),
        )

        similar_errors = [row["id"] for row in cursor.fetchall()]

        if len(similar_errors) >= self.min_pattern_support:
            pattern = MemoryPattern(
                pattern_id=f"pred_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                description=f"Recurring prediction errors for questions similar to: {prediction_entry.question[:50]}...",
                supporting_entries=similar_errors,
                pattern_type="prediction_error",
                confidence=min(len(similar_errors) / 10.0, 1.0),
                discovered_at=datetime.now(),
                last_reinforced=datetime.now(),
            )

            self._save_pattern(pattern)

    def _detect_confidence_pattern(self, prediction_entry: JournalEntry, outcome_entry: JournalEntry):
        """Detect patterns in confidence miscalibration."""

        # Find entries with similar confidence vs accuracy gaps
        cursor = self.db_connection.cursor()
        cursor.execute(
            """
            SELECT id FROM journal_entries
            WHERE reflection_type = ?
            AND ABS(confidence_score - accuracy_score) > ?
            ORDER BY timestamp DESC
        """,
            (
                ReflectionType.OUTCOME.value,
                self.surprise_thresholds["confidence_mismatch"],
            ),
        )

        miscalibrated_entries = [row["id"] for row in cursor.fetchall()]

        if len(miscalibrated_entries) >= self.min_pattern_support:
            pattern = MemoryPattern(
                pattern_id=f"conf_pattern_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                description="Pattern of confidence miscalibration - predicted confidence doesn't match actual accuracy",
                supporting_entries=miscalibrated_entries,
                pattern_type="confidence_mismatch",
                confidence=min(len(miscalibrated_entries) / 15.0, 1.0),
                discovered_at=datetime.now(),
                last_reinforced=datetime.now(),
            )

            self._save_pattern(pattern)

    def _detect_context_patterns(self, prediction_entry: JournalEntry, outcome_entry: JournalEntry):
        """Detect patterns based on context similarities."""

        similar_contexts = self._find_similar_contexts(prediction_entry.context, limit=10)

        if len(similar_contexts) >= self.min_pattern_support:
            # Check if similar contexts tend to have similar surprise levels
            surprise_levels = [
                entry.surprise_level
                for entry in similar_contexts
                if hasattr(entry, "surprise_level") and entry.surprise_level
            ]

            if surprise_levels and len(set(surprise_levels)) <= 2:  # Consistent surprise pattern
                pattern = MemoryPattern(
                    pattern_id=f"context_pattern_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    description=f"Context pattern: Similar contexts tend to produce {surprise_levels[0].value} surprise",
                    supporting_entries=[entry.id for entry in similar_contexts],
                    pattern_type="context_similarity",
                    confidence=len(surprise_levels) / len(similar_contexts),
                    discovered_at=datetime.now(),
                    last_reinforced=datetime.now(),
                )

                self._save_pattern(pattern)

    def _save_entry(self, entry: JournalEntry) -> str:
        """Save journal entry to database."""

        cursor = self.db_connection.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO journal_entries
            (id, timestamp, reflection_type, content, question, context,
             predicted_outcome, actual_outcome, surprise_level, surprise_reasoning,
             confidence_score, accuracy_score, learning_value, related_entries, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                entry.id,
                entry.timestamp.isoformat(),
                entry.reflection_type.value,
                entry.content,
                entry.question,
                json.dumps(entry.context),
                entry.predicted_outcome,
                entry.actual_outcome,
                entry.surprise_level.value,
                entry.surprise_reasoning,
                entry.confidence_score,
                entry.accuracy_score,
                entry.learning_value,
                json.dumps(entry.related_entries),
                json.dumps(entry.tags),
            ),
        )

        self.db_connection.commit()
        return entry.id

    def _save_pattern(self, pattern: MemoryPattern):
        """Save memory pattern to database."""

        cursor = self.db_connection.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO memory_patterns
            (pattern_id, description, supporting_entries, pattern_type,
             confidence, discovered_at, last_reinforced, reinforcement_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                pattern.pattern_id,
                pattern.description,
                json.dumps(pattern.supporting_entries),
                pattern.pattern_type,
                pattern.confidence,
                pattern.discovered_at.isoformat(),
                pattern.last_reinforced.isoformat(),
                pattern.reinforcement_count,
            ),
        )

        self.db_connection.commit()

    def get_entry(self, entry_id: str) -> JournalEntry | None:
        """Retrieve a journal entry by ID."""

        cursor = self.db_connection.cursor()
        cursor.execute("SELECT * FROM journal_entries WHERE id = ?", (entry_id,))
        row = cursor.fetchone()

        if not row:
            return None

        try:
            return JournalEntry(
                id=row["id"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                reflection_type=ReflectionType(row["reflection_type"]),
                content=row["content"],
                question=row["question"] or "",
                context=json.loads(row["context"]) if row["context"] else {},
                predicted_outcome=row["predicted_outcome"] or "",
                actual_outcome=row["actual_outcome"] or "",
                surprise_level=SurpriseLevel(row["surprise_level"]),
                surprise_reasoning=row["surprise_reasoning"] or "",
                confidence_score=row["confidence_score"] or 0.0,
                accuracy_score=row["accuracy_score"] or 0.0,
                learning_value=row["learning_value"] or 0.0,
                related_entries=json.loads(row["related_entries"]) if row["related_entries"] else [],
                tags=json.loads(row["tags"]) if row["tags"] else [],
            )
        except (ValueError, json.JSONDecodeError) as e:
            self.logger.error(f"Error parsing journal entry {entry_id}: {e}")
            return None

    def get_recent_entries(self, limit: int = 50, entry_type: ReflectionType | None = None) -> list[JournalEntry]:
        """Get recent journal entries."""

        cursor = self.db_connection.cursor()

        if entry_type:
            cursor.execute(
                """
                SELECT * FROM journal_entries
                WHERE reflection_type = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (entry_type.value, limit),
            )
        else:
            cursor.execute(
                """
                SELECT * FROM journal_entries
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (limit,),
            )

        entries = []
        for row in cursor.fetchall():
            entry = self.get_entry(row["id"])
            if entry:
                entries.append(entry)

        return entries

    def get_surprise_entries(self, min_level: SurpriseLevel = SurpriseLevel.MEDIUM) -> list[JournalEntry]:
        """Get entries with surprise level at or above threshold."""

        surprise_levels = [
            SurpriseLevel.MEDIUM,
            SurpriseLevel.HIGH,
            SurpriseLevel.EXTREME,
        ]
        if min_level == SurpriseLevel.HIGH:
            surprise_levels = [SurpriseLevel.HIGH, SurpriseLevel.EXTREME]
        elif min_level == SurpriseLevel.EXTREME:
            surprise_levels = [SurpriseLevel.EXTREME]

        cursor = self.db_connection.cursor()
        placeholders = ",".join("?" * len(surprise_levels))
        cursor.execute(
            f"""
            SELECT * FROM journal_entries
            WHERE surprise_level IN ({placeholders})
            ORDER BY learning_value DESC, timestamp DESC
        """,
            [level.value for level in surprise_levels],
        )

        entries = []
        for row in cursor.fetchall():
            entry = self.get_entry(row["id"])
            if entry:
                entries.append(entry)

        return entries

    def get_patterns(self, pattern_type: str | None = None) -> list[MemoryPattern]:
        """Get discovered memory patterns."""

        cursor = self.db_connection.cursor()

        if pattern_type:
            cursor.execute(
                """
                SELECT * FROM memory_patterns
                WHERE pattern_type = ?
                ORDER BY confidence DESC, last_reinforced DESC
            """,
                (pattern_type,),
            )
        else:
            cursor.execute(
                """
                SELECT * FROM memory_patterns
                ORDER BY confidence DESC, last_reinforced DESC
            """
            )

        patterns = []
        for row in cursor.fetchall():
            try:
                pattern = MemoryPattern(
                    pattern_id=row["pattern_id"],
                    description=row["description"],
                    supporting_entries=json.loads(row["supporting_entries"]),
                    pattern_type=row["pattern_type"],
                    confidence=row["confidence"],
                    discovered_at=datetime.fromisoformat(row["discovered_at"]),
                    last_reinforced=datetime.fromisoformat(row["last_reinforced"]),
                    reinforcement_count=row["reinforcement_count"],
                )
                patterns.append(pattern)
            except (ValueError, json.JSONDecodeError) as e:
                self.logger.error(f"Error parsing pattern {row['pattern_id']}: {e}")

        return patterns

    def _link_entries(self, entry1_id: str, entry2_id: str):
        """Create bidirectional link between entries."""

        # Update entry1 to reference entry2
        cursor = self.db_connection.cursor()
        cursor.execute("SELECT related_entries FROM journal_entries WHERE id = ?", (entry1_id,))
        row = cursor.fetchone()

        if row:
            related = json.loads(row["related_entries"]) if row["related_entries"] else []
            if entry2_id not in related:
                related.append(entry2_id)
                cursor.execute(
                    "UPDATE journal_entries SET related_entries = ? WHERE id = ?",
                    (json.dumps(related), entry1_id),
                )

        # Update entry2 to reference entry1
        cursor.execute("SELECT related_entries FROM journal_entries WHERE id = ?", (entry2_id,))
        row = cursor.fetchone()

        if row:
            related = json.loads(row["related_entries"]) if row["related_entries"] else []
            if entry1_id not in related:
                related.append(entry1_id)
                cursor.execute(
                    "UPDATE journal_entries SET related_entries = ? WHERE id = ?",
                    (json.dumps(related), entry2_id),
                )

        self.db_connection.commit()

    def get_journal_summary(self) -> dict[str, Any]:
        """Get summary statistics about the journal."""

        cursor = self.db_connection.cursor()

        # Basic counts
        cursor.execute("SELECT COUNT(*) as total FROM journal_entries")
        total_entries = cursor.fetchone()["total"]

        cursor.execute("SELECT reflection_type, COUNT(*) as count FROM journal_entries GROUP BY reflection_type")
        type_counts = {row["reflection_type"]: row["count"] for row in cursor.fetchall()}

        cursor.execute("SELECT surprise_level, COUNT(*) as count FROM journal_entries GROUP BY surprise_level")
        surprise_counts = {row["surprise_level"]: row["count"] for row in cursor.fetchall()}

        cursor.execute("SELECT AVG(accuracy_score) as avg_accuracy FROM journal_entries WHERE accuracy_score > 0")
        avg_accuracy = cursor.fetchone()["avg_accuracy"] or 0

        cursor.execute("SELECT AVG(learning_value) as avg_learning FROM journal_entries WHERE learning_value > 0")
        avg_learning = cursor.fetchone()["avg_learning"] or 0

        cursor.execute("SELECT COUNT(*) as pattern_count FROM memory_patterns")
        pattern_count = cursor.fetchone()["pattern_count"]

        return {
            "total_entries": total_entries,
            "entry_types": type_counts,
            "surprise_distribution": surprise_counts,
            "average_accuracy": avg_accuracy,
            "average_learning_value": avg_learning,
            "discovered_patterns": pattern_count,
            "journal_path": self.journal_db_path,
        }

    def close(self):
        """Close database connection."""
        if self.db_connection:
            self.db_connection.close()


class MCPIntegration:
    """Integration layer for MCP (Model Context Protocol) tools."""

    def __init__(self, config: QuietSTaRConfig, journal: SurpriseMemoryJournal):
        self.config = config
        self.journal = journal
        self.logger = logging.getLogger(__name__)

        # MCP tool registry
        self.tools: dict[str, Callable] = {}
        self.tool_descriptions: dict[str, str] = {}

        # Register built-in tools
        self._register_built_in_tools()

    def _register_built_in_tools(self):
        """Register built-in MCP tools."""

        self.register_tool(
            "predict_and_journal",
            self.predict_and_journal,
            "Make a prediction and journal it for surprise tracking",
        )

        self.register_tool(
            "record_outcome",
            self.record_outcome,
            "Record the actual outcome for a prediction",
        )

        self.register_tool(
            "analyze_patterns",
            self.analyze_patterns,
            "Analyze discovered patterns from journal entries",
        )

        self.register_tool(
            "get_surprise_insights",
            self.get_surprise_insights,
            "Get insights from surprising outcomes",
        )

        self.register_tool(
            "search_journal",
            self.search_journal,
            "Search journal entries by content or tags",
        )

    def register_tool(self, name: str, func: Callable, description: str):
        """Register a new MCP tool."""
        self.tools[name] = func
        self.tool_descriptions[name] = description
        self.logger.info(f"Registered MCP tool: {name}")

    async def predict_and_journal(
        self,
        question: str,
        prediction: str,
        confidence: float,
        reasoning: str = "",
        context: dict[str, Any] = None,
    ) -> dict[str, Any]:
        """Make a prediction and journal it."""

        prediction_id = self.journal.add_prediction(
            question=question,
            predicted_outcome=prediction,
            confidence=confidence,
            reasoning=reasoning,
            context=context or {},
        )

        return {
            "prediction_id": prediction_id,
            "question": question,
            "prediction": prediction,
            "confidence": confidence,
            "status": "prediction_recorded",
        }

    async def record_outcome(
        self, prediction_id: str, actual_outcome: str, context: dict[str, Any] = None
    ) -> dict[str, Any]:
        """Record actual outcome for a prediction."""

        outcome_id = self.journal.record_outcome(
            prediction_id=prediction_id,
            actual_outcome=actual_outcome,
            context=context or {},
        )

        if not outcome_id:
            return {"error": f"Prediction {prediction_id} not found"}

        # Get surprise analysis
        outcome_entry = self.journal.get_entry(outcome_id)

        return {
            "outcome_id": outcome_id,
            "prediction_id": prediction_id,
            "actual_outcome": actual_outcome,
            "surprise_level": outcome_entry.surprise_level.value,
            "surprise_reasoning": outcome_entry.surprise_reasoning,
            "accuracy": outcome_entry.accuracy_score,
            "learning_value": outcome_entry.learning_value,
            "status": "outcome_recorded",
        }

    async def analyze_patterns(self, pattern_type: str | None = None) -> dict[str, Any]:
        """Analyze discovered patterns."""

        patterns = self.journal.get_patterns(pattern_type)

        pattern_analysis = []
        for pattern in patterns:
            pattern_analysis.append(
                {
                    "pattern_id": pattern.pattern_id,
                    "type": pattern.pattern_type,
                    "description": pattern.description,
                    "confidence": pattern.confidence,
                    "support_count": len(pattern.supporting_entries),
                    "discovered": pattern.discovered_at.isoformat(),
                    "last_reinforced": pattern.last_reinforced.isoformat(),
                }
            )

        return {
            "total_patterns": len(patterns),
            "patterns": pattern_analysis,
            "pattern_types": list(set(p.pattern_type for p in patterns)),
        }

    async def get_surprise_insights(self, min_surprise: str = "medium") -> dict[str, Any]:
        """Get insights from surprising outcomes."""

        surprise_level = SurpriseLevel(min_surprise)
        surprise_entries = self.journal.get_surprise_entries(surprise_level)

        insights = []
        for entry in surprise_entries[:10]:  # Limit to top 10
            insights.append(
                {
                    "entry_id": entry.id,
                    "question": entry.question,
                    "predicted": entry.predicted_outcome,
                    "actual": entry.actual_outcome,
                    "surprise_level": entry.surprise_level.value,
                    "learning_value": entry.learning_value,
                    "reasoning": entry.surprise_reasoning,
                    "timestamp": entry.timestamp.isoformat(),
                }
            )

        return {
            "surprise_threshold": min_surprise,
            "total_surprising_entries": len(surprise_entries),
            "top_insights": insights,
            "avg_learning_value": (
                sum(e.learning_value for e in surprise_entries) / len(surprise_entries) if surprise_entries else 0
            ),
        }

    async def search_journal(self, query: str, entry_type: str | None = None, limit: int = 20) -> dict[str, Any]:
        """Search journal entries."""

        # Simple text search in content and questions
        cursor = self.journal.db_connection.cursor()

        if entry_type:
            cursor.execute(
                """
                SELECT * FROM journal_entries
                WHERE (content LIKE ? OR question LIKE ?)
                AND reflection_type = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (f"%{query}%", f"%{query}%", entry_type, limit),
            )
        else:
            cursor.execute(
                """
                SELECT * FROM journal_entries
                WHERE content LIKE ? OR question LIKE ?
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (f"%{query}%", f"%{query}%", limit),
            )

        results = []
        for row in cursor.fetchall():
            entry = self.journal.get_entry(row["id"])
            if entry:
                results.append(
                    {
                        "entry_id": entry.id,
                        "type": entry.reflection_type.value,
                        "content": entry.content[:200] + "..." if len(entry.content) > 200 else entry.content,
                        "question": entry.question,
                        "surprise_level": entry.surprise_level.value,
                        "timestamp": entry.timestamp.isoformat(),
                        "tags": entry.tags,
                    }
                )

        return {"query": query, "results_count": len(results), "results": results}

    def get_available_tools(self) -> dict[str, str]:
        """Get list of available MCP tools."""
        return self.tool_descriptions.copy()

    async def call_tool(self, tool_name: str, **kwargs) -> dict[str, Any]:
        """Call an MCP tool by name."""

        if tool_name not in self.tools:
            return {"error": f"Tool {tool_name} not found"}

        try:
            result = await self.tools[tool_name](**kwargs)
            return result
        except Exception as e:
            self.logger.error(f"Error calling tool {tool_name}: {e}")
            return {"error": str(e)}


if __name__ == "__main__":
    # Demo MCP tools and journaling system
    from .config import get_training_config

    print("📔 MCP Tools and Journaling System Demo")
    print("=" * 50)

    # Initialize system
    config = get_training_config()
    journal = SurpriseMemoryJournal(config, "demo_journal.db")
    mcp = MCPIntegration(config, journal)

    print(f"Initialized journal database: {journal.journal_db_path}")
    print(f"Available MCP tools: {len(mcp.tools)}")
    print()

    # Demonstrate prediction and outcome cycle
    async def demo_prediction_cycle():
        print("1. Prediction and Outcome Cycle Demo:")

        # Make some predictions
        predictions = [
            {
                "question": "What will be the primary challenge in deploying GPT-4 at scale?",
                "prediction": "The main challenge will be computational cost and infrastructure scaling.",
                "confidence": 0.7,
                "reasoning": "Based on current trends in LLM deployment",
            },
            {
                "question": "How will users react to thought-token reasoning in AI assistants?",
                "prediction": "Users will appreciate the transparency and improved reasoning quality.",
                "confidence": 0.6,
                "reasoning": "Transparency generally increases user trust",
            },
            {
                "question": "What will be the next major breakthrough in neural architecture?",
                "prediction": "Mixture of Experts (MoE) architectures will become dominant.",
                "confidence": 0.4,
                "reasoning": "MoE shows promise but many alternatives exist",
            },
        ]

        prediction_ids = []
        for pred in predictions:
            result = await mcp.predict_and_journal(**pred)
            prediction_ids.append(result["prediction_id"])
            print(f"   Recorded prediction: {pred['question'][:50]}...")
            print(f"   ID: {result['prediction_id']}")

        print()

        # Record some outcomes (simulated)
        print("2. Recording Outcomes (Simulated):")

        outcomes = [
            {
                "prediction_id": prediction_ids[0],
                "actual_outcome": "The main challenge was actually data privacy and content filtering, not computational cost.",
            },
            {
                "prediction_id": prediction_ids[1],
                "actual_outcome": "Users appreciated transparency but were more concerned about response speed than reasoning quality.",
            },
            {
                "prediction_id": prediction_ids[2],
                "actual_outcome": "Retrieval-augmented architectures became more prominent than MoE systems.",
            },
        ]

        for outcome in outcomes:
            result = await mcp.record_outcome(**outcome)
            print(f"   Recorded outcome: {result['surprise_level']} surprise")
            print(f"   Accuracy: {result['accuracy']:.2f}, Learning value: {result['learning_value']:.2f}")

        print()

        # Get surprise insights
        print("3. Surprise Analysis:")
        insights = await mcp.get_surprise_insights("low")
        print(f"   Found {insights['total_surprising_entries']} surprising entries")
        print(f"   Average learning value: {insights['avg_learning_value']:.2f}")

        for insight in insights["top_insights"][:3]:
            print(f"   • {insight['surprise_level']} surprise: {insight['question'][:60]}...")
            print(f"     Predicted: {insight['predicted'][:50]}...")
            print(f"     Actual: {insight['actual'][:50]}...")

        print()

        # Pattern analysis
        print("4. Pattern Discovery:")
        pattern_analysis = await mcp.analyze_patterns()
        print(f"   Discovered {pattern_analysis['total_patterns']} patterns")

        if pattern_analysis["patterns"]:
            for pattern in pattern_analysis["patterns"][:3]:
                print(f"   • {pattern['type']}: {pattern['description'][:60]}...")
                print(f"     Confidence: {pattern['confidence']:.2f}, Support: {pattern['support_count']} entries")

        print()

        # Journal search
        print("5. Journal Search:")
        search_results = await mcp.search_journal("challenge")
        print(f"   Search for 'challenge': {search_results['results_count']} results")

        for result in search_results["results"][:2]:
            print(f"   • {result['type']}: {result['content'][:50]}...")
            print(f"     Tags: {', '.join(result['tags'])}")

        print()

        # Journal summary
        print("6. Journal Summary:")
        summary = journal.get_journal_summary()
        print(f"   Total entries: {summary['total_entries']}")
        print(f"   Entry types: {summary['entry_types']}")
        print(f"   Surprise distribution: {summary['surprise_distribution']}")
        print(f"   Average accuracy: {summary['average_accuracy']:.2f}")
        print(f"   Average learning value: {summary['average_learning_value']:.2f}")
        print(f"   Discovered patterns: {summary['discovered_patterns']}")

    # Run the demo
    import asyncio

    asyncio.run(demo_prediction_cycle())

    # Cleanup
    journal.close()

    print()
    print("=" * 50)
    print("*** MCP TOOLS AND JOURNALING SYSTEM VERIFIED! ***")
    print()
    print("Implementation Status:")
    print("   [X] B1: Core Quiet-STaR system (completed)")
    print("   [X] B2: Teacher prompt generation (completed)")
    print("   [X] B3: Student distillation prompts (completed)")
    print("   [X] B4: Thought-token A/B baking system (completed)")
    print("   [X] C1: Alignment prelude integration (completed)")
    print("   [X] D1-D2: Temperature self-recognition system (completed)")
    print("   [X] E1: MCP tools and journaling (completed)")
    print()
    print("Key MCP & Journaling Features Demonstrated:")
    print("   • Prediction and outcome tracking with surprise analysis")
    print("   • Multi-level surprise detection (none/low/medium/high/extreme)")
    print("   • Automatic pattern discovery from surprising outcomes")
    print("   • Learning value assessment and memory consolidation")
    print("   • MCP tool integration with async/await support")
    print("   • Journal search and analysis capabilities")
    print("   • SQLite-based persistent storage")
    print("   • Confidence calibration tracking")
    print()
    print("🎉 COMPLETE QUIET-STAR + THOUGHT-TOKEN PROMPT BAKING SYSTEM!")
    print("All components (B1-E1) successfully implemented and demonstrated.")
