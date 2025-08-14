"""Mastery Tracker - Track learning progress with 3-variant-pass rule.

Implements mastery tracking based on distinct variant success patterns,
determining when students have truly understood a concept.
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .openrouter import OpenRouterLLM
from .schemas import (
    AttemptRecord,
    LastResult,
    MasteryAction,
    MasteryRequest,
    MasteryResponse,
    MasteryStatus,
)

logger = logging.getLogger(__name__)


class MasteryTracker:
    """Tracks student mastery using the 3-variant-pass rule."""

    def __init__(
        self,
        llm_client: OpenRouterLLM,
        model: str = "openai/gpt-4o-mini",
        temperature: float = 0.2,
        storage_path: str = "mastery.db",
        mastery_threshold: int = 3,
        stall_threshold: int = 5,
    ):
        """Initialize MasteryTracker.

        Args:
            llm_client: OpenRouter client for LLM calls
            model: Model to use for mastery policy decisions
            temperature: Low temperature for consistent decisions
            storage_path: Path to SQLite database for persistence
            mastery_threshold: Number of distinct variants needed for mastery
            stall_threshold: Number of attempts before considering stalled
        """
        self.llm_client = llm_client
        self.model = model
        self.temperature = temperature
        self.mastery_threshold = mastery_threshold
        self.stall_threshold = stall_threshold
        self.storage_path = storage_path

        # Load template
        template_path = Path(__file__).parent / "templates" / "mastery_policy.jinja"
        with open(template_path, encoding="utf-8") as f:
            self.template = f.read()

        # Initialize database
        self._init_database()

        logger.info(f"MasteryTracker initialized with {mastery_threshold}-variant threshold")

    def _init_database(self) -> None:
        """Initialize SQLite database for mastery tracking."""

        with sqlite3.connect(self.storage_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS mastery_records (
                    student_id TEXT NOT NULL,
                    problem_id TEXT NOT NULL,
                    attempts INTEGER DEFAULT 0,
                    correct_count INTEGER DEFAULT 0,
                    variant_ids_seen TEXT DEFAULT '[]',
                    variant_ids_correct TEXT DEFAULT '[]',
                    last_variant_id TEXT,
                    last_correct BOOLEAN,
                    last_updated TEXT,
                    current_status TEXT DEFAULT 'learning',
                    next_action TEXT DEFAULT 'reshuffle',
                    needs_hint BOOLEAN DEFAULT FALSE,
                    created_at TEXT,
                    PRIMARY KEY (student_id, problem_id)
                )
            """
            )

            # Index for efficient queries
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_student_status
                ON mastery_records(student_id, current_status)
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_problem_status
                ON mastery_records(problem_id, current_status)
            """
            )

    def _serialize_list(self, lst: list[str]) -> str:
        """Serialize list to JSON string for database storage."""
        return json.dumps(lst)

    def _deserialize_list(self, json_str: str) -> list[str]:
        """Deserialize JSON string to list."""
        try:
            return json.loads(json_str) if json_str else []
        except:
            return []

    def get_attempt_record(self, student_id: str, problem_id: str) -> AttemptRecord:
        """Get current attempt record for student-problem pair.

        Args:
            student_id: Unique student identifier
            problem_id: Unique problem identifier

        Returns:
            AttemptRecord with current attempt history
        """
        with sqlite3.connect(self.storage_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM mastery_records
                WHERE student_id = ? AND problem_id = ?
            """,
                (student_id, problem_id),
            )

            row = cursor.fetchone()

            if row:
                return AttemptRecord(
                    problem_id=row["problem_id"],
                    attempts=row["attempts"],
                    correct_count=row["correct_count"],
                    variant_ids_seen=self._deserialize_list(row["variant_ids_seen"]),
                    variant_ids_correct=self._deserialize_list(row["variant_ids_correct"]),
                )
            else:
                # Create new record
                return AttemptRecord(
                    problem_id=problem_id,
                    attempts=0,
                    correct_count=0,
                    variant_ids_seen=[],
                    variant_ids_correct=[],
                )

    def record_attempt(self, student_id: str, problem_id: str, variant_id: str, correct: bool) -> None:
        """Record a new attempt for tracking.

        Args:
            student_id: Unique student identifier
            problem_id: Unique problem identifier
            variant_id: Unique variant identifier
            correct: Whether the attempt was correct
        """
        now = datetime.now(timezone.utc).isoformat()

        with sqlite3.connect(self.storage_path) as conn:
            # Get current record
            cursor = conn.execute(
                """
                SELECT variant_ids_seen, variant_ids_correct, attempts, correct_count
                FROM mastery_records
                WHERE student_id = ? AND problem_id = ?
            """,
                (student_id, problem_id),
            )

            row = cursor.fetchone()

            if row:
                # Update existing record
                seen = self._deserialize_list(row[0])
                correct_variants = self._deserialize_list(row[1])
                attempts = row[2]
                correct_count = row[3]

                # Add variant to seen list if not already there
                if variant_id not in seen:
                    seen.append(variant_id)

                # Update correct variants and count
                if correct and variant_id not in correct_variants:
                    correct_variants.append(variant_id)
                    correct_count += 1

                attempts += 1

                conn.execute(
                    """
                    UPDATE mastery_records
                    SET attempts = ?, correct_count = ?,
                        variant_ids_seen = ?, variant_ids_correct = ?,
                        last_variant_id = ?, last_correct = ?, last_updated = ?
                    WHERE student_id = ? AND problem_id = ?
                """,
                    (
                        attempts,
                        correct_count,
                        self._serialize_list(seen),
                        self._serialize_list(correct_variants),
                        variant_id,
                        correct,
                        now,
                        student_id,
                        problem_id,
                    ),
                )
            else:
                # Create new record
                seen = [variant_id]
                correct_variants = [variant_id] if correct else []
                attempts = 1
                correct_count = 1 if correct else 0

                conn.execute(
                    """
                    INSERT INTO mastery_records
                    (student_id, problem_id, attempts, correct_count,
                     variant_ids_seen, variant_ids_correct, last_variant_id,
                     last_correct, last_updated, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        student_id,
                        problem_id,
                        attempts,
                        correct_count,
                        self._serialize_list(seen),
                        self._serialize_list(correct_variants),
                        variant_id,
                        correct,
                        now,
                        now,
                    ),
                )

        logger.debug(f"Recorded attempt: {student_id}/{problem_id}/{variant_id} = {correct}")

    def _evaluate_mastery_locally(self, record: AttemptRecord, last_result: LastResult) -> MasteryResponse:
        """Evaluate mastery using local rules without LLM."""

        # Check for mastery (3+ distinct variants correct)
        distinct_correct = len(record.variant_ids_correct)

        if distinct_correct >= self.mastery_threshold:
            return MasteryResponse(
                ok=True,
                msg="mastery achieved",
                status=MasteryStatus.UNDERSTOOD,
                next_action=MasteryAction.PROMOTE,
                needs_hint=False,
            )

        # Check for stalling (too many attempts with low success)
        if record.attempts >= self.stall_threshold:
            success_rate = record.correct_count / record.attempts

            if success_rate < 0.3:  # Less than 30% success
                return MasteryResponse(
                    ok=True,
                    msg="stalled - needs intervention",
                    status=MasteryStatus.STALLED,
                    next_action=MasteryAction.INJECT_HINT_VARIANT,
                    needs_hint=True,
                )

        # Check for repeated failures on same concept
        if not last_result.correct and record.attempts >= 3 and record.correct_count == 0:
            return MasteryResponse(
                ok=True,
                msg="struggling - needs hint",
                status=MasteryStatus.STALLED,
                next_action=MasteryAction.INJECT_HINT_VARIANT,
                needs_hint=True,
            )

        # Still learning - continue with current approach
        return MasteryResponse(
            ok=True,
            msg="continue learning",
            status=MasteryStatus.LEARNING,
            next_action=MasteryAction.RESHUFFLE,
            needs_hint=not last_result.correct and record.attempts > 1,
        )

    async def evaluate_mastery(
        self,
        student_id: str,
        problem_id: str,
        variant_id: str,
        correct: bool,
        use_llm_policy: bool = True,
        use_local_fallback: bool = True,
    ) -> MasteryResponse:
        """Evaluate mastery status after an attempt.

        Args:
            student_id: Unique student identifier
            problem_id: Unique problem identifier
            variant_id: Variant attempted
            correct: Whether attempt was correct
            use_llm_policy: Whether to use LLM for policy decisions
            use_local_fallback: Use local rules if LLM fails

        Returns:
            MasteryResponse with status and next action
        """
        # Record the attempt first
        self.record_attempt(student_id, problem_id, variant_id, correct)

        # Get updated record
        record = self.get_attempt_record(student_id, problem_id)
        last_result = LastResult(variant_id=variant_id, correct=correct)

        logger.info(
            f"Evaluating mastery: {student_id}/{problem_id} - {record.correct_count}/{len(record.variant_ids_correct)} variants correct"
        )

        # Try LLM policy evaluation
        if use_llm_policy:
            try:
                request = MasteryRequest(record=record, last_result=last_result)

                # Render prompt
                prompt = self.llm_client.render_template(
                    self.template,
                    record=request.record,
                    last_result=request.last_result,
                )

                response = await self.llm_client.invoke_with_schema(
                    prompt=prompt,
                    schema_class=MasteryResponse,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=512,
                    max_schema_retries=2,
                )

                # Update database with new status
                self._update_mastery_status(student_id, problem_id, response)

                logger.info(f"LLM mastery evaluation: {response.status.value} -> {response.next_action.value}")
                return response

            except Exception as e:
                logger.error(f"LLM mastery evaluation failed: {e}")

        # Fallback to local rules
        if use_local_fallback:
            logger.info("Using local mastery evaluation")
            response = self._evaluate_mastery_locally(record, last_result)

            # Update database
            self._update_mastery_status(student_id, problem_id, response)

            logger.info(f"Local mastery evaluation: {response.status.value} -> {response.next_action.value}")
            return response

        # Ultimate fallback
        return MasteryResponse(
            ok=False,
            msg="evaluation failed",
            status=MasteryStatus.LEARNING,
            next_action=MasteryAction.RESHUFFLE,
            needs_hint=False,
        )

    def _update_mastery_status(self, student_id: str, problem_id: str, response: MasteryResponse) -> None:
        """Update mastery status in database."""

        now = datetime.now(timezone.utc).isoformat()

        with sqlite3.connect(self.storage_path) as conn:
            conn.execute(
                """
                UPDATE mastery_records
                SET current_status = ?, next_action = ?,
                    needs_hint = ?, last_updated = ?
                WHERE student_id = ? AND problem_id = ?
            """,
                (
                    response.status.value,
                    response.next_action.value,
                    response.needs_hint,
                    now,
                    student_id,
                    problem_id,
                ),
            )

    def get_student_mastery_summary(self, student_id: str) -> dict[str, Any]:
        """Get comprehensive mastery summary for a student.

        Args:
            student_id: Unique student identifier

        Returns:
            Dictionary with mastery statistics and status
        """
        with sqlite3.connect(self.storage_path) as conn:
            conn.row_factory = sqlite3.Row

            # Overall stats
            cursor = conn.execute(
                """
                SELECT
                    COUNT(*) as total_problems,
                    SUM(CASE WHEN current_status = 'understood' THEN 1 ELSE 0 END) as mastered,
                    SUM(CASE WHEN current_status = 'learning' THEN 1 ELSE 0 END) as learning,
                    SUM(CASE WHEN current_status = 'stalled' THEN 1 ELSE 0 END) as stalled,
                    SUM(attempts) as total_attempts,
                    SUM(correct_count) as total_correct
                FROM mastery_records
                WHERE student_id = ?
            """,
                (student_id,),
            )

            stats = cursor.fetchone()

            if not stats or stats["total_problems"] == 0:
                return {
                    "student_id": student_id,
                    "total_problems": 0,
                    "mastery_rate": 0.0,
                    "overall_accuracy": 0.0,
                    "status_distribution": {
                        "understood": 0,
                        "learning": 0,
                        "stalled": 0,
                    },
                    "problems": [],
                }

            # Individual problem details
            cursor = conn.execute(
                """
                SELECT problem_id, current_status, attempts, correct_count,
                       variant_ids_correct, needs_hint, last_updated
                FROM mastery_records
                WHERE student_id = ?
                ORDER BY last_updated DESC
            """,
                (student_id,),
            )

            problems = []
            for row in cursor.fetchall():
                correct_variants = self._deserialize_list(row["variant_ids_correct"])
                problems.append(
                    {
                        "problem_id": row["problem_id"],
                        "status": row["current_status"],
                        "attempts": row["attempts"],
                        "correct_count": row["correct_count"],
                        "distinct_variants_correct": len(correct_variants),
                        "needs_hint": row["needs_hint"],
                        "last_updated": row["last_updated"],
                    }
                )

            return {
                "student_id": student_id,
                "total_problems": stats["total_problems"],
                "mastery_rate": stats["mastered"] / stats["total_problems"],
                "overall_accuracy": stats["total_correct"] / stats["total_attempts"]
                if stats["total_attempts"] > 0
                else 0,
                "status_distribution": {
                    "understood": stats["mastered"],
                    "learning": stats["learning"],
                    "stalled": stats["stalled"],
                },
                "total_attempts": stats["total_attempts"],
                "total_correct": stats["total_correct"],
                "problems": problems,
            }

    def get_problem_mastery_summary(self, problem_id: str) -> dict[str, Any]:
        """Get mastery summary across all students for a problem.

        Args:
            problem_id: Unique problem identifier

        Returns:
            Dictionary with cross-student mastery statistics
        """
        with sqlite3.connect(self.storage_path) as conn:
            conn.row_factory = sqlite3.Row

            cursor = conn.execute(
                """
                SELECT
                    COUNT(*) as total_students,
                    SUM(CASE WHEN current_status = 'understood' THEN 1 ELSE 0 END) as mastered,
                    SUM(CASE WHEN current_status = 'learning' THEN 1 ELSE 0 END) as learning,
                    SUM(CASE WHEN current_status = 'stalled' THEN 1 ELSE 0 END) as stalled,
                    AVG(attempts) as avg_attempts,
                    AVG(CAST(correct_count AS FLOAT) / attempts) as avg_accuracy
                FROM mastery_records
                WHERE problem_id = ?
            """,
                (problem_id,),
            )

            stats = cursor.fetchone()

            if not stats or stats["total_students"] == 0:
                return {
                    "problem_id": problem_id,
                    "total_students": 0,
                    "mastery_rate": 0.0,
                    "difficulty_estimate": 0.5,
                    "students": [],
                }

            # Individual student performance
            cursor = conn.execute(
                """
                SELECT student_id, current_status, attempts, correct_count, last_updated
                FROM mastery_records
                WHERE problem_id = ?
                ORDER BY last_updated DESC
            """,
                (problem_id,),
            )

            students = []
            for row in cursor.fetchall():
                accuracy = row["correct_count"] / row["attempts"] if row["attempts"] > 0 else 0
                students.append(
                    {
                        "student_id": row["student_id"],
                        "status": row["current_status"],
                        "attempts": row["attempts"],
                        "accuracy": accuracy,
                        "last_updated": row["last_updated"],
                    }
                )

            # Estimate difficulty (inverse of mastery rate)
            mastery_rate = stats["mastered"] / stats["total_students"]
            difficulty_estimate = 1.0 - mastery_rate

            return {
                "problem_id": problem_id,
                "total_students": stats["total_students"],
                "mastery_rate": mastery_rate,
                "difficulty_estimate": difficulty_estimate,
                "status_distribution": {
                    "understood": stats["mastered"],
                    "learning": stats["learning"],
                    "stalled": stats["stalled"],
                },
                "avg_attempts": stats["avg_attempts"] or 0,
                "avg_accuracy": stats["avg_accuracy"] or 0,
                "students": students,
            }

    def get_students_needing_intervention(self) -> list[dict[str, Any]]:
        """Get list of students who need immediate intervention.

        Returns:
            List of students with stalled status or high hint needs
        """
        with sqlite3.connect(self.storage_path) as conn:
            conn.row_factory = sqlite3.Row

            cursor = conn.execute(
                """
                SELECT student_id, problem_id, current_status, attempts,
                       correct_count, needs_hint, last_updated
                FROM mastery_records
                WHERE current_status = 'stalled' OR needs_hint = TRUE
                ORDER BY last_updated ASC
            """
            )

            interventions = []
            for row in cursor.fetchall():
                interventions.append(
                    {
                        "student_id": row["student_id"],
                        "problem_id": row["problem_id"],
                        "status": row["current_status"],
                        "attempts": row["attempts"],
                        "success_rate": row["correct_count"] / row["attempts"] if row["attempts"] > 0 else 0,
                        "needs_hint": bool(row["needs_hint"]),
                        "priority": "high" if row["current_status"] == "stalled" else "medium",
                        "last_updated": row["last_updated"],
                    }
                )

            return interventions

    def reset_student_progress(self, student_id: str, problem_id: str | None = None) -> int:
        """Reset progress for student (useful for testing or fresh starts).

        Args:
            student_id: Student to reset
            problem_id: Specific problem to reset (or all if None)

        Returns:
            Number of records reset
        """
        with sqlite3.connect(self.storage_path) as conn:
            if problem_id:
                cursor = conn.execute(
                    """
                    DELETE FROM mastery_records
                    WHERE student_id = ? AND problem_id = ?
                """,
                    (student_id, problem_id),
                )
            else:
                cursor = conn.execute(
                    """
                    DELETE FROM mastery_records
                    WHERE student_id = ?
                """,
                    (student_id,),
                )

            return cursor.rowcount


async def track_student_mastery(
    api_key: str,
    student_id: str,
    problem_id: str,
    variant_id: str,
    correct: bool,
    storage_path: str = "mastery.db",
    model: str = "openai/gpt-4o-mini",
    **kwargs,
) -> MasteryResponse:
    """Convenience function for mastery tracking with minimal setup.

    Args:
        api_key: OpenRouter API key
        student_id: Unique student identifier
        problem_id: Unique problem identifier
        variant_id: Variant attempted
        correct: Whether attempt was correct
        storage_path: Path to mastery database
        model: Model to use for mastery decisions
        **kwargs: Additional arguments for MasteryTracker

    Returns:
        MasteryResponse with mastery evaluation
    """
    async with OpenRouterLLM(api_key=api_key) as client:
        tracker = MasteryTracker(client, model=model, storage_path=storage_path)
        return await tracker.evaluate_mastery(student_id, problem_id, variant_id, correct, **kwargs)


if __name__ == "__main__":
    # Demo usage
    import asyncio
    import os
    import tempfile

    async def demo():
        # Create temporary database for demo
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_db = f.name

        api_key = os.getenv("OPENROUTER_API_KEY", "demo-key")

        try:
            if api_key == "demo-key":
                print("🔧 Demo mode: Testing local mastery tracking")

                # Local mastery tracking demo
                dummy_client = OpenRouterLLM(api_key="dummy")
                tracker = MasteryTracker(dummy_client, storage_path=temp_db)

                student_id = "demo_student"
                problem_id = "test_problem_001"

                # Simulate learning progression
                attempts = [
                    ("variant_1", False),  # First attempt fails
                    ("variant_1", True),  # Second attempt succeeds
                    ("variant_2", True),  # New variant succeeds
                    ("variant_3", False),  # Third variant fails
                    ("variant_3", True),  # Third variant succeeds (MASTERY!)
                ]

                for i, (variant_id, correct) in enumerate(attempts):
                    print(f"\n📝 Attempt {i + 1}: {variant_id} = {'✅' if correct else '❌'}")

                    # Use local evaluation
                    response = await tracker.evaluate_mastery(
                        student_id,
                        problem_id,
                        variant_id,
                        correct,
                        use_llm_policy=False,
                    )

                    print(f"   Status: {response.status.value}")
                    print(f"   Action: {response.next_action.value}")
                    print(f"   Needs hint: {response.needs_hint}")

                    # Check if mastered
                    if response.status == MasteryStatus.UNDERSTOOD:
                        print("   🎉 MASTERY ACHIEVED!")
                        break

                # Show final summary
                summary = tracker.get_student_mastery_summary(student_id)
                print("\n📊 Student Summary:")
                print(f"   Mastery rate: {summary['mastery_rate']:.1%}")
                print(f"   Overall accuracy: {summary['overall_accuracy']:.1%}")
                print(f"   Status distribution: {summary['status_distribution']}")

                return

            # Live API test
            print("🎯 Testing live mastery tracking...")

            result = await track_student_mastery(
                api_key=api_key,
                student_id="test_student",
                problem_id="live_test_001",
                variant_id="variant_a",
                correct=True,
                storage_path=temp_db,
            )

            print("✅ Mastery evaluation complete")
            print(f"   Status: {result.status.value}")
            print(f"   Next action: {result.next_action.value}")
            print(f"   Needs hint: {result.needs_hint}")

        except Exception as e:
            print(f"❌ Demo failed: {e}")

        finally:
            # Cleanup temp file
            try:
                os.unlink(temp_db)
            except:
                pass

    asyncio.run(demo())
