"""Parent Progress Tracker - Comprehensive Monitoring Dashboard
Sprint R-5: Digital Twin MVP - Task A.4.
"""

import asyncio
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
import hashlib
import json
import logging
import sqlite3
from typing import Any

import numpy as np
import wandb

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    CONCERN = "concern"
    URGENT = "urgent"


class NotificationChannel(Enum):
    EMAIL = "email"
    APP = "app"
    SMS = "sms"
    PUSH = "push"


@dataclass
class ProgressMilestone:
    """Educational milestone tracking."""

    milestone_id: str
    student_id: str
    subject: str
    concept: str
    description: str
    target_mastery_level: float
    current_mastery_level: float
    achieved: bool
    achieved_date: str | None
    estimated_completion: str
    difficulty_level: str
    importance: str  # critical, important, helpful
    celebration_message: str
    next_milestone: str | None = None


@dataclass
class LearningAlert:
    """Alert for parents about learning status."""

    alert_id: str
    student_id: str
    alert_type: str  # milestone, concern, celebration, reminder
    level: AlertLevel
    title: str
    message: str
    created_at: str
    read: bool = False
    acknowledged: bool = False
    action_required: bool = False
    related_concepts: list[str] = None
    suggested_actions: list[str] = None
    expires_at: str | None = None


@dataclass
class WeeklyReport:
    """Weekly progress report."""

    report_id: str
    student_id: str
    week_start: str
    week_end: str
    total_study_time_minutes: int
    sessions_completed: int
    concepts_practiced: list[str]
    concepts_mastered: list[str]
    avg_engagement_score: float
    avg_accuracy: float
    achievements_earned: list[str]
    areas_of_strength: list[str]
    areas_for_improvement: list[str]
    recommended_focus: list[str]
    parent_notes: str = ""
    generated_at: str = ""


@dataclass
class ParentInsight:
    """Actionable insight for parents."""

    insight_id: str
    student_id: str
    category: str  # motivation, schedule, difficulty, engagement
    title: str
    description: str
    evidence: list[str]
    recommended_actions: list[str]
    priority: str  # high, medium, low
    confidence_score: float
    created_at: str
    implemented: bool = False


class ParentProgressTracker:
    """Comprehensive progress tracking and parent communication system."""

    def __init__(self, project_name: str = "aivillage-parent-tracker") -> None:
        """Initialize ParentProgressTracker."""
        self.project_name = project_name
        self.student_progress = {}  # student_id -> progress data
        self.parent_alerts = defaultdict(list)  # student_id -> List[LearningAlert]
        self.weekly_reports = defaultdict(list)  # student_id -> List[WeeklyReport]
        self.milestones = defaultdict(list)  # student_id -> List[ProgressMilestone]
        self.parent_insights = defaultdict(list)  # student_id -> List[ParentInsight]

        # Parent preferences and settings
        self.parent_settings = {}  # student_id -> parent settings
        self.notification_preferences = {}  # student_id -> notification prefs

        # Analytics and trends
        self.progress_trends = defaultdict(dict)  # student_id -> trend data
        self.engagement_patterns = defaultdict(list)  # student_id -> engagement
        self.learning_velocity = defaultdict(deque)  # student_id -> velocity

        # Communication channels
        self.email_config = {
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "username": "",  # Would be configured
            "password": "",  # Would be configured securely
        }

        # Database for persistence
        self.db_path = "parent_tracker.db"
        self.init_database()

        # Background tasks
        self.monitoring_active = True

        # Initialize W&B tracking
        self.initialize_wandb_tracking()

        # Start background monitoring
        asyncio.create_task(self.start_progress_monitoring())

        logger.info("Parent Progress Tracker initialized")

    def initialize_wandb_tracking(self) -> None:
        """Initialize W&B tracking for parent monitoring."""
        try:
            wandb.init(
                project=self.project_name,
                job_type="parent_progress_tracking",
                config={
                    "tracker_version": "1.0.0",
                    "monitoring_features": [
                        "real_time_progress",
                        "milestone_tracking",
                        "alert_system",
                        "weekly_reports",
                        "parent_insights",
                        "engagement_analytics",
                    ],
                    "notification_channels": ["email", "app", "sms", "push"],
                    "report_frequency": "weekly",
                    "alert_levels": ["info", "success", "warning", "concern", "urgent"],
                    "privacy_compliant": True,
                    "parental_controls": True,
                },
            )

            logger.info("Parent tracker W&B tracking initialized")

        except Exception as e:
            logger.exception("Failed to initialize W&B tracking: %s", e)

    def init_database(self) -> None:
        """Initialize database for parent tracking data."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Progress milestones table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS progress_milestones (
                    milestone_id TEXT PRIMARY KEY,
                    student_id TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    concept TEXT NOT NULL,
                    description TEXT NOT NULL,
                    target_mastery_level REAL NOT NULL,
                    current_mastery_level REAL NOT NULL,
                    achieved INTEGER DEFAULT 0,
                    achieved_date TEXT,
                    estimated_completion TEXT,
                    difficulty_level TEXT,
                    importance TEXT,
                    celebration_message TEXT,
                    created_at TEXT NOT NULL
                )
            """)

            # Learning alerts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS learning_alerts (
                    alert_id TEXT PRIMARY KEY,
                    student_id TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    level TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    read INTEGER DEFAULT 0,
                    acknowledged INTEGER DEFAULT 0,
                    action_required INTEGER DEFAULT 0,
                    related_concepts TEXT,  -- JSON
                    suggested_actions TEXT, -- JSON
                    expires_at TEXT
                )
            """)

            # Weekly reports table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS weekly_reports (
                    report_id TEXT PRIMARY KEY,
                    student_id TEXT NOT NULL,
                    week_start TEXT NOT NULL,
                    week_end TEXT NOT NULL,
                    report_data TEXT NOT NULL,  -- JSON
                    generated_at TEXT NOT NULL
                )
            """)

            # Parent insights table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS parent_insights (
                    insight_id TEXT PRIMARY KEY,
                    student_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    evidence TEXT NOT NULL,  -- JSON
                    recommended_actions TEXT NOT NULL,  -- JSON
                    priority TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    implemented INTEGER DEFAULT 0
                )
            """)

            # Parent settings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS parent_settings (
                    student_id TEXT PRIMARY KEY,
                    parent_email TEXT,
                    parent_name TEXT,
                    notification_preferences TEXT,  -- JSON
                    report_frequency TEXT DEFAULT 'weekly',
                    alert_threshold TEXT DEFAULT 'normal',
                    privacy_settings TEXT,  -- JSON
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # Create indexes
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_milestones_student "
                "ON progress_milestones(student_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_alerts_student "
                "ON learning_alerts(student_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_reports_student "
                "ON weekly_reports(student_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_insights_student "
                "ON parent_insights(student_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_alerts_created "
                "ON learning_alerts(created_at)"
            )

            conn.commit()
            conn.close()

            logger.info("Parent tracker database initialized")

        except Exception as e:
            logger.exception("Failed to initialize database: %s", e)

    async def register_parent(
        self,
        student_id: str,
        parent_email: str,
        parent_name: str,
        notification_preferences: dict[str, Any] | None = None,
    ) -> bool:
        """Register parent for progress tracking."""
        if notification_preferences is None:
            notification_preferences = {
                "email_enabled": True,
                "app_notifications": True,
                "weekly_reports": True,
                "milestone_alerts": True,
                "concern_alerts": True,
                "celebration_alerts": True,
                "study_reminders": False,
                "alert_quiet_hours": {"start": "22:00", "end": "07:00"},
                "max_daily_alerts": 5,
            }

        parent_settings = {
            "student_id": student_id,
            "parent_email": parent_email,
            "parent_name": parent_name,
            "notification_preferences": notification_preferences,
            "report_frequency": "weekly",
            "alert_threshold": "normal",
            "privacy_settings": {
                "share_detailed_analytics": True,
                "include_session_details": True,
                "anonymize_reports": False,
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        # Store settings
        self.parent_settings[student_id] = parent_settings
        self.notification_preferences[student_id] = notification_preferences

        # Save to database
        await self._save_parent_settings(student_id, parent_settings)

        # Initialize milestone tracking
        await self.initialize_student_milestones(student_id)

        # Send welcome message
        await self.create_alert(
            student_id=student_id,
            alert_type="welcome",
            level=AlertLevel.INFO,
            title="Welcome to AI Village Progress Tracking!",
            message=(
                f"Hi {parent_name}! You'll now receive regular updates about "
                "your child's learning progress, achievements, and areas where "
                "they might need support."
            ),
            suggested_actions=[
                "Explore the parent dashboard",
                "Set your notification preferences",
            ],
        )

        # Log to W&B
        wandb.log(
            {
                "parent_tracker/parent_registered": True,
                "parent_tracker/student_id": student_id,
                "parent_tracker/notifications_enabled": (
                    notification_preferences.get("email_enabled", False)
                ),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        logger.info("Registered parent %s for student %s", parent_name, student_id[:8])

        return True

    async def initialize_student_milestones(self, student_id: str) -> None:
        """Initialize learning milestones for a student."""
        # Get student's current grade/age from digital twin
        try:
            from digital_twin.core.digital_twin import digital_twin

            if student_id in digital_twin.students:
                student = digital_twin.students[student_id]
                grade_level = student.grade_level

                # Create grade-appropriate milestones
                milestones = await self._generate_grade_milestones(
                    student_id, grade_level
                )

                for milestone in milestones:
                    self.milestones[student_id].append(milestone)
                    await self._save_milestone(milestone)

                logger.info(
                    "Initialized %d milestones for student %s",
                    len(milestones),
                    student_id[:8],
                )

        except Exception as e:
            logger.warning("Could not initialize milestones: %s", e)

    async def _generate_grade_milestones(
        self, student_id: str, grade_level: int
    ) -> list[ProgressMilestone]:
        """Generate appropriate milestones for grade level."""
        milestones = []

        # Grade-specific mathematical milestones
        if grade_level <= 2:
            milestone_concepts = [
                ("counting", "Count to 100 with confidence", 0.8, "critical"),
                ("addition", "Add single-digit numbers fluently", 0.8, "critical"),
                (
                    "subtraction",
                    "Subtract single-digit numbers fluently",
                    0.8,
                    "critical",
                ),
                ("shapes", "Recognize basic geometric shapes", 0.7, "important"),
            ]
        elif grade_level <= 4:
            milestone_concepts = [
                (
                    "multiplication",
                    "Master multiplication tables 1-10",
                    0.8,
                    "critical",
                ),
                ("division", "Understand division with remainders", 0.7, "critical"),
                ("fractions", "Compare and order simple fractions", 0.7, "important"),
                ("word_problems", "Solve multi-step word problems", 0.6, "important"),
                ("geometry", "Calculate area and perimeter", 0.6, "helpful"),
            ]
        elif grade_level <= 6:
            milestone_concepts = [
                ("decimals", "Add and subtract decimals confidently", 0.8, "critical"),
                ("fractions", "Multiply and divide fractions", 0.7, "critical"),
                ("ratios", "Understand ratios and proportions", 0.7, "important"),
                ("geometry", "Work with angles and triangles", 0.6, "important"),
                ("statistics", "Interpret graphs and data", 0.6, "helpful"),
            ]
        else:  # Grade 7-8
            milestone_concepts = [
                ("algebra", "Solve linear equations", 0.8, "critical"),
                ("functions", "Understand function concepts", 0.7, "critical"),
                ("geometry", "Apply Pythagorean theorem", 0.7, "important"),
                ("statistics", "Calculate measures of center", 0.6, "important"),
                ("probability", "Understand basic probability", 0.6, "helpful"),
            ]

        for i, (concept, description, target_mastery, importance) in enumerate(
            milestone_concepts
        ):
            milestone_id = f"milestone_{student_id[:8]}_{concept}_{i}"

            # Estimate completion based on current progress
            estimated_weeks = (
                4 if importance == "critical" else 6 if importance == "important" else 8
            )
            estimated_completion = (
                datetime.now(timezone.utc) + timedelta(weeks=estimated_weeks)
            ).isoformat()

            milestone = ProgressMilestone(
                milestone_id=milestone_id,
                student_id=student_id,
                subject="mathematics",
                concept=concept,
                description=description,
                target_mastery_level=target_mastery,
                current_mastery_level=0.0,  # Updated from knowledge states
                achieved=False,
                achieved_date=None,
                estimated_completion=estimated_completion,
                difficulty_level="grade_appropriate",
                importance=importance,
                celebration_message=(
                    f"🎉 Amazing work! You've mastered {description.lower()}!"
                ),
                next_milestone=None,
            )

            milestones.append(milestone)

        return milestones

    async def update_student_progress(
        self, student_id: str, session_data: dict[str, Any]
    ) -> None:
        """Update student progress and check for alerts/milestones."""
        try:
            # Update milestone progress
            await self._update_milestone_progress(student_id, session_data)

            # Check for new alerts
            await self._check_progress_alerts(student_id, session_data)

            # Update learning velocity
            self._update_learning_velocity(student_id, session_data)

            # Update engagement patterns
            self._update_engagement_patterns(student_id, session_data)

            # Generate insights if needed
            await self._generate_parent_insights(student_id)

            # Log progress update
            wandb.log(
                {
                    "parent_tracker/progress_updated": True,
                    "parent_tracker/student_id": student_id,
                    "parent_tracker/session_engagement": (
                        session_data.get("engagement_score", 0)
                    ),
                    "parent_tracker/session_accuracy": (
                        session_data.get("accuracy", 0)
                    ),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

        except Exception as e:
            logger.exception("Error updating student progress: %s", e)

    async def _update_milestone_progress(
        self, student_id: str, session_data: dict[str, Any]
    ) -> None:
        """Update milestone progress based on session data."""
        concepts_practiced = session_data.get("concepts_covered", [])
        accuracy = session_data.get("accuracy", 0.0)

        for milestone in self.milestones[student_id]:
            if milestone.achieved:
                continue

            # Check if this session practiced the milestone concept
            if milestone.concept in concepts_practiced:
                # Update mastery level (simplified calculation)
                learning_rate = 0.1
                milestone.current_mastery_level = min(
                    1.0, milestone.current_mastery_level + learning_rate * accuracy
                )

                # Check if milestone is achieved
                if milestone.current_mastery_level >= milestone.target_mastery_level:
                    milestone.achieved = True
                    milestone.achieved_date = datetime.now(timezone.utc).isoformat()

                    # Create celebration alert
                    await self.create_alert(
                        student_id=student_id,
                        alert_type="milestone",
                        level=AlertLevel.SUCCESS,
                        title="Milestone Achieved! 🎉",
                        message=milestone.celebration_message,
                        suggested_actions=[
                            "Celebrate this achievement!",
                            "Move on to the next challenge",
                        ],
                    )

                    logger.info(
                        "Milestone achieved: %s for student %s",
                        milestone.description,
                        student_id[:8],
                    )

                # Update in database
                await self._save_milestone(milestone)

    async def _check_progress_alerts(
        self, student_id: str, session_data: dict[str, Any]
    ) -> None:
        """Check for progress-based alerts."""
        session_data.get("engagement_score", 0.5)
        session_data.get("accuracy", 0.5)
        session_data.get("duration_minutes", 0)

        # Get recent session history for trend analysis
        recent_sessions = self._get_recent_sessions(student_id, days=7)

        if len(recent_sessions) >= 3:
            avg_engagement = np.mean(
                [s.get("engagement_score", 0.5) for s in recent_sessions]
            )
            avg_accuracy = np.mean([s.get("accuracy", 0.5) for s in recent_sessions])

            # Low engagement alert
            if avg_engagement < 0.3:
                await self.create_alert(
                    student_id=student_id,
                    alert_type="concern",
                    level=AlertLevel.CONCERN,
                    title="Low Engagement Detected",
                    message=(
                        "Your child's engagement has been lower than usual over "
                        "the past week. They might need a break or a change in "
                        "learning activities."
                    ),
                    suggested_actions=[
                        "Try shorter learning sessions",
                        "Introduce more interactive activities",
                        "Take a day break and return refreshed",
                        "Ask your child what topics interest them most",
                    ],
                    action_required=True,
                )

            # Accuracy concerns
            elif avg_accuracy < 0.4:
                await self.create_alert(
                    student_id=student_id,
                    alert_type="concern",
                    level=AlertLevel.WARNING,
                    title="Learning Challenge Identified",
                    message=(
                        "Your child is working hard but finding the current "
                        "material challenging. Additional support might be helpful."
                    ),
                    suggested_actions=[
                        "Review previous concepts that build up to current work",
                        "Practice with simpler problems first",
                        "Consider working together on homework",
                        "Celebrate effort, not just correct answers",
                    ],
                    action_required=True,
                )

            # Positive trends
            elif avg_engagement > 0.8 and avg_accuracy > 0.7:
                improvement_trend = self._calculate_improvement_trend(recent_sessions)
                if improvement_trend > 0.1:
                    await self.create_alert(
                        student_id=student_id,
                        alert_type="celebration",
                        level=AlertLevel.SUCCESS,
                        title="Excellent Progress! 🌟",
                        message=(
                            "Your child is doing fantastic! High engagement and "
                            "accuracy show they're really connecting with the material."
                        ),
                        suggested_actions=[
                            "Celebrate this great progress!",
                            "Consider introducing slightly more challenging material",
                            "Ask them to teach you what they've learned",
                        ],
                    )

    def _get_recent_sessions(
        self, student_id: str, days: int = 7
    ) -> list[dict[str, Any]]:
        """Get recent session data for analysis."""
        # This would integrate with the digital twin session history
        # For now, return mock data structure
        return []

    def _calculate_improvement_trend(self, sessions: list[dict[str, Any]]) -> float:
        """Calculate improvement trend from session data."""
        if len(sessions) < 2:
            return 0.0

        # Simple linear trend calculation
        accuracies = [s.get("accuracy", 0.5) for s in sessions]
        n = len(accuracies)
        x = list(range(n))

        # Calculate slope
        x_mean = np.mean(x)
        y_mean = np.mean(accuracies)

        numerator = sum((x[i] - x_mean) * (accuracies[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0

        slope = numerator / denominator
        return slope

    def _update_learning_velocity(self, student_id: str, session_data: dict[str, Any]) -> None:
        """Update learning velocity tracking."""
        concepts_learned = len(session_data.get("concepts_covered", []))
        session_time = session_data.get("duration_minutes", 0)

        if session_time > 0:
            velocity = concepts_learned / (session_time / 60)  # Per hour

            self.learning_velocity[student_id].append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "velocity": velocity,
                    "concepts": concepts_learned,
                    "time_hours": session_time / 60,
                }
            )

            # Keep only recent data (last 30 days)
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=30)
            self.learning_velocity[student_id] = deque(
                [
                    entry
                    for entry in self.learning_velocity[student_id]
                    if datetime.fromisoformat(entry["timestamp"]) > cutoff_date
                ],
                maxlen=100,
            )

    def _update_engagement_patterns(
        self, student_id: str, session_data: dict[str, Any]
    ) -> None:
        """Update engagement pattern analysis."""
        session_start = session_data.get("start_time")
        if session_start:
            try:
                start_dt = datetime.fromisoformat(session_start)
                hour_of_day = start_dt.hour
                day_of_week = start_dt.weekday()

                engagement_data = {
                    "timestamp": session_start,
                    "hour_of_day": hour_of_day,
                    "day_of_week": day_of_week,
                    "engagement_score": session_data.get("engagement_score", 0.5),
                    "duration_minutes": session_data.get("duration_minutes", 0),
                }

                self.engagement_patterns[student_id].append(engagement_data)

                # Keep only recent data
                if len(self.engagement_patterns[student_id]) > 100:
                    patterns = self.engagement_patterns[student_id]
                    self.engagement_patterns[student_id] = patterns[-50:]

            except Exception as e:
                logger.warning("Error updating engagement patterns: %s", e)

    async def _generate_parent_insights(self, student_id: str) -> None:
        """Generate actionable insights for parents."""
        # Check if we have enough data
        if len(self.engagement_patterns[student_id]) < 10:
            return

        insights = []

        # Analyze optimal study times
        engagement_by_hour = defaultdict(list)
        for session in self.engagement_patterns[student_id]:
            hour = session["hour_of_day"]
            engagement_by_hour[hour].append(session["engagement_score"])

        # Find best and worst performance times
        hour_averages = {
            hour: np.mean(scores)
            for hour, scores in engagement_by_hour.items()
            if len(scores) >= 3
        }

        if hour_averages:
            best_hour = max(hour_averages, key=hour_averages.get)
            worst_hour = min(hour_averages, key=hour_averages.get)

            if hour_averages[best_hour] - hour_averages[worst_hour] > 0.2:
                insights.append(
                    ParentInsight(
                        insight_id=(
                            f"timing_{student_id}_{datetime.now().strftime('%Y%m%d')}"
                        ),
                        student_id=student_id,
                        category="schedule",
                        title="Optimal Study Time Identified",
                        description=(
                            f"Your child shows highest engagement around "
                            f"{best_hour}:00 and lowest around {worst_hour}:00."
                        ),
                        evidence=[
                            f"Best performance: {best_hour}:00 "
                            f"(engagement: {hour_averages[best_hour]:.2f})",
                            f"Challenging time: {worst_hour}:00 "
                            f"(engagement: {hour_averages[worst_hour]:.2f})",
                        ],
                        recommended_actions=[
                            f"Schedule important learning sessions around "
                            f"{best_hour}:00",
                            f"Use {worst_hour}:00 for lighter review or break time",
                            "Be consistent with timing to build routine",
                        ],
                        priority="medium",
                        confidence_score=0.8,
                        created_at=datetime.now(timezone.utc).isoformat(),
                    )
                )

        # Analyze learning velocity trends
        if len(self.learning_velocity[student_id]) >= 5:
            velocities = [
                entry["velocity"] for entry in self.learning_velocity[student_id]
            ]
            velocity_trend = self._calculate_improvement_trend(
                [{"accuracy": v} for v in velocities]
            )

            if velocity_trend > 0.1:
                insights.append(
                    ParentInsight(
                        insight_id=(
                            f"velocity_{student_id}_{datetime.now().strftime('%Y%m%d')}"
                        ),
                        student_id=student_id,
                        category="motivation",
                        title="Learning Speed is Increasing",
                        description=(
                            "Your child is learning concepts faster over time, "
                            "showing great progress!"
                        ),
                        evidence=[
                            f"Average learning velocity: "
                            f"{np.mean(velocities):.2f} concepts/hour",
                            f"Positive trend: +{velocity_trend:.3f} improvement rate",
                        ],
                        recommended_actions=[
                            "Acknowledge and celebrate this improvement",
                            "Consider introducing more challenging material",
                            "Maintain current study routine as it's working well",
                        ],
                        priority="high",
                        confidence_score=0.9,
                        created_at=datetime.now(timezone.utc).isoformat(),
                    )
                )

        # Store new insights
        for insight in insights:
            self.parent_insights[student_id].append(insight)
            await self._save_insight(insight)

    async def create_alert(
        self,
        student_id: str,
        alert_type: str,
        level: AlertLevel,
        title: str,
        message: str,
        suggested_actions: list[str] | None = None,
        action_required: bool = False,
        related_concepts: list[str] | None = None,
        expires_hours: int = 168,  # Default 7 days
    ) -> str:
        """Create alert for parents."""
        message_to_hash = f"{student_id}_{title}_{datetime.now().isoformat()}"
        alert_id = f"alert_{hashlib.md5(message_to_hash.encode()).hexdigest()[:12]}"

        expires_at = (
            datetime.now(timezone.utc) + timedelta(hours=expires_hours)
        ).isoformat()

        alert = LearningAlert(
            alert_id=alert_id,
            student_id=student_id,
            alert_type=alert_type,
            level=level,
            title=title,
            message=message,
            created_at=datetime.now(timezone.utc).isoformat(),
            read=False,
            acknowledged=False,
            action_required=action_required,
            related_concepts=related_concepts or [],
            suggested_actions=suggested_actions or [],
            expires_at=expires_at,
        )

        # Store alert
        self.parent_alerts[student_id].append(alert)
        await self._save_alert(alert)

        # Send notification if enabled
        await self._send_notification(student_id, alert)

        # Log to W&B
        wandb.log(
            {
                "parent_tracker/alert_created": True,
                "parent_tracker/alert_type": alert_type,
                "parent_tracker/alert_level": level.value,
                "parent_tracker/action_required": action_required,
                "timestamp": alert.created_at,
            }
        )

        logger.info(
            "Created %s alert for student %s: %s", level.value, student_id[:8], title
        )

        return alert_id

    async def _send_notification(self, student_id: str, alert: LearningAlert) -> None:
        """Send notification to parent."""
        parent_settings = self.parent_settings.get(student_id)
        if not parent_settings:
            return

        notification_prefs = parent_settings.get("notification_preferences", {})

        # Check if notifications are enabled for this alert type
        alert_pref_key = f"{alert.alert_type}_alerts"
        if not notification_prefs.get(alert_pref_key, True):
            return

        # Check quiet hours
        quiet_hours = notification_prefs.get("alert_quiet_hours", {})
        if self._is_quiet_hours(quiet_hours):
            logger.info("Delaying notification due to quiet hours: %s", alert.alert_id)
            return

        # Check daily limit
        daily_alerts = len(
            [
                a
                for a in self.parent_alerts[student_id]
                if (
                    datetime.now(timezone.utc) - datetime.fromisoformat(a.created_at)
                ).days
                == 0
            ]
        )

        max_daily = notification_prefs.get("max_daily_alerts", 5)
        if daily_alerts > max_daily:
            logger.info("Daily alert limit reached for student %s", student_id[:8])
            return

        # Send email notification
        if notification_prefs.get("email_enabled", True):
            await self._send_email_notification(parent_settings, alert)

        # Other notification channels would be implemented here
        # (app push, SMS, etc.)

    def _is_quiet_hours(self, quiet_hours: dict[str, str]) -> bool:
        """Check if current time is in quiet hours."""
        if not quiet_hours:
            return False

        try:
            now = datetime.now()
            start_time = datetime.strptime(quiet_hours["start"], "%H:%M").time()
            end_time = datetime.strptime(quiet_hours["end"], "%H:%M").time()
            current_time = now.time()

            if start_time <= end_time:
                return start_time <= current_time <= end_time
            # Crosses midnight
            return current_time >= start_time or current_time <= end_time
        except Exception:
            return False

    async def _send_email_notification(
        self, parent_settings: dict[str, Any], alert: LearningAlert
    ) -> None:
        """Send email notification to parent."""
        try:
            parent_email = parent_settings.get("parent_email")
            parent_name = parent_settings.get("parent_name", "Parent")

            if not parent_email or not self.email_config.get("username"):
                return

            # Create email content
            subject = f"AI Village Update: {alert.title}"

            # HTML email template
            html_content = f"""
            <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; \
color: #333;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                    <h2 style="color: #2c3e50;">Hello {parent_name}!</h2>

                    <div style="background-color: #f8f9fa; padding: 20px; \
border-radius: 8px; margin: 20px 0;">
                        <h3 style="color: #e74c3c; margin-top: 0;">{alert.title}</h3>
                        <p>{alert.message}</p>
                    </div>

                    {
                f'''
                    <div style="background-color: #e8f5e8; padding: 15px;
border-radius: 8px; margin: 20px 0;">
                        <h4 style="color: #27ae60; margin-top: 0;">
Suggested Actions:</h4>
                        <ul>
                            {
                    "".join(
                        [f"<li>{action}</li>" for action in alert.suggested_actions]
                    )
                }
                        </ul>
                    </div>
                    '''
                if alert.suggested_actions
                else ""
            }

                    <div style="text-align: center; margin: 30px 0;">
                        <a href="#" style="background-color: #3498db; color: white; \
padding: 12px 24px; text-decoration: none; border-radius: 4px;">
                            View Full Dashboard
                        </a>
                    </div>

                    <div style="border-top: 1px solid #eee; padding-top: 20px; \
margin-top: 30px; font-size: 14px; color: #666;">
                        <p>Best regards,<br>The AI Village Team</p>
                        <p style="font-size: 12px;">
                            You received this because you're registered for \
progress updates.
                            <a href="#">Update your notification preferences</a>
                        </p>
                    </div>
                </div>
            </body>
            </html>
            """

            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.email_config["username"]
            msg["To"] = parent_email

            # Attach HTML content
            html_part = MIMEText(html_content, "html")
            msg.attach(html_part)

            # Send email (would need proper SMTP configuration)
            # This is a placeholder for the actual email sending logic
            logger.info(
                "Email notification prepared for %s: %s", parent_email, alert.title
            )

        except Exception as e:
            logger.exception("Failed to send email notification: %s", e)

    async def generate_weekly_report(self, student_id: str) -> WeeklyReport:
        """Generate comprehensive weekly progress report."""
        # Calculate week boundaries
        today = datetime.now(timezone.utc)
        week_start = today - timedelta(days=today.weekday())
        week_end = week_start + timedelta(days=6)

        report_id = f"weekly_{student_id}_{week_start.strftime('%Y%m%d')}"

        # Gather week's data (this would integrate with actual session data)
        week_sessions = self._get_week_sessions(student_id, week_start, week_end)

        # Calculate summary statistics
        total_study_time = sum(s.get("duration_minutes", 0) for s in week_sessions)
        sessions_completed = len(week_sessions)

        all_concepts = []
        for session in week_sessions:
            all_concepts.extend(session.get("concepts_covered", []))
        concepts_practiced = list(set(all_concepts))

        # Get concepts mastered this week
        concepts_mastered = self._get_concepts_mastered_this_week(
            student_id, week_start, week_end
        )

        # Calculate averages
        avg_engagement = (
            np.mean([s.get("engagement_score", 0) for s in week_sessions])
            if week_sessions
            else 0
        )
        avg_accuracy = (
            np.mean([s.get("accuracy", 0) for s in week_sessions])
            if week_sessions
            else 0
        )

        # Get achievements
        achievements_earned = self._get_achievements_this_week(
            student_id, week_start, week_end
        )

        # Analyze strengths and areas for improvement
        areas_of_strength, areas_for_improvement = self._analyze_performance_areas(
            week_sessions
        )

        # Generate recommendations
        recommended_focus = self._generate_focus_recommendations(
            student_id, week_sessions, concepts_mastered
        )

        report = WeeklyReport(
            report_id=report_id,
            student_id=student_id,
            week_start=week_start.isoformat(),
            week_end=week_end.isoformat(),
            total_study_time_minutes=total_study_time,
            sessions_completed=sessions_completed,
            concepts_practiced=concepts_practiced,
            concepts_mastered=concepts_mastered,
            avg_engagement_score=avg_engagement,
            avg_accuracy=avg_accuracy,
            achievements_earned=achievements_earned,
            areas_of_strength=areas_of_strength,
            areas_for_improvement=areas_for_improvement,
            recommended_focus=recommended_focus,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

        # Store report
        self.weekly_reports[student_id].append(report)
        await self._save_weekly_report(report)

        # Send report to parent
        await self._send_weekly_report_notification(student_id, report)

        # Log to W&B
        wandb.log(
            {
                "parent_tracker/weekly_report_generated": True,
                "parent_tracker/study_time_minutes": total_study_time,
                "parent_tracker/sessions_completed": sessions_completed,
                "parent_tracker/avg_engagement": avg_engagement,
                "parent_tracker/avg_accuracy": avg_accuracy,
                "parent_tracker/concepts_mastered": len(concepts_mastered),
                "timestamp": report.generated_at,
            }
        )

        logger.info("Generated weekly report for student %s", student_id[:8])

        return report

    def _get_week_sessions(
        self, student_id: str, week_start: datetime, week_end: datetime
    ) -> list[dict[str, Any]]:
        """Get sessions for the specified week."""
        # This would integrate with actual session data
        return []

    def _get_concepts_mastered_this_week(
        self, student_id: str, week_start: datetime, week_end: datetime
    ) -> list[str]:
        """Get concepts that were mastered this week."""
        mastered = []

        for milestone in self.milestones[student_id]:
            if milestone.achieved and milestone.achieved_date:
                achieved_date = datetime.fromisoformat(milestone.achieved_date)
                if week_start <= achieved_date <= week_end:
                    mastered.append(milestone.concept)

        return mastered

    def _get_achievements_this_week(
        self, student_id: str, week_start: datetime, week_end: datetime
    ) -> list[str]:
        """Get achievements earned this week."""
        achievements = []

        # Check milestone achievements
        for milestone in self.milestones[student_id]:
            if milestone.achieved and milestone.achieved_date:
                achieved_date = datetime.fromisoformat(milestone.achieved_date)
                if week_start <= achieved_date <= week_end:
                    achievements.append(f"Mastered {milestone.description}")

        return achievements

    def _analyze_performance_areas(
        self, sessions: list[dict[str, Any]]
    ) -> tuple[list[str], list[str]]:
        """Analyze performance to identify strengths and improvement areas."""
        if not sessions:
            return [], []

        # Analyze by concept performance
        concept_performance = defaultdict(list)
        for session in sessions:
            for concept in session.get("concepts_covered", []):
                concept_performance[concept].append(session.get("accuracy", 0))

        strengths = []
        improvements = []

        for concept, accuracies in concept_performance.items():
            avg_accuracy = np.mean(accuracies)

            if avg_accuracy >= 0.8:
                strengths.append(f"Strong performance in {concept}")
            elif avg_accuracy <= 0.5:
                improvements.append(f"Additional practice needed in {concept}")

        # Overall engagement analysis
        avg_engagement = np.mean([s.get("engagement_score", 0) for s in sessions])
        if avg_engagement >= 0.8:
            strengths.append("High engagement and motivation")
        elif avg_engagement <= 0.4:
            improvements.append("Consider ways to increase engagement")

        return strengths, improvements

    def _generate_focus_recommendations(
        self,
        student_id: str,
        sessions: list[dict[str, Any]],
        concepts_mastered: list[str],
    ) -> list[str]:
        """Generate focus recommendations for next week."""
        recommendations = []

        if not sessions:
            recommendations.append("Continue regular learning sessions")
            return recommendations

        # Check for struggling concepts
        concept_struggles = defaultdict(int)
        for session in sessions:
            if session.get("accuracy", 0) < 0.5:
                for concept in session.get("concepts_covered", []):
                    concept_struggles[concept] += 1

        if concept_struggles:
            most_challenging = max(concept_struggles, key=concept_struggles.get)
            recommendations.append(f"Focus extra attention on {most_challenging}")

        # Check for mastered concepts - can advance
        if concepts_mastered:
            recommendations.append("Ready to explore more advanced topics")

        # Engagement-based recommendations
        avg_engagement = np.mean([s.get("engagement_score", 0) for s in sessions])
        if avg_engagement < 0.5:
            recommendations.append(
                "Try incorporating more interactive or hands-on activities"
            )

        # Session length optimization
        session_lengths = [s.get("duration_minutes", 0) for s in sessions]
        if session_lengths:
            avg_length = np.mean(session_lengths)
            if avg_length < 15:
                recommendations.append(
                    "Consider slightly longer sessions for deeper learning"
                )
            elif avg_length > 45:
                recommendations.append("Consider shorter, more frequent sessions")

        return recommendations or ["Continue current learning approach"]

    async def _send_weekly_report_notification(
        self, student_id: str, report: WeeklyReport
    ) -> None:
        """Send weekly report to parent."""
        parent_settings = self.parent_settings.get(student_id)
        if not parent_settings:
            return

        notification_prefs = parent_settings.get("notification_preferences", {})
        if not notification_prefs.get("weekly_reports", True):
            return

        # Create summary alert
        achievements_text = (
            "🎉 " + ", ".join(report.achievements_earned)
            if report.achievements_earned
            else "Keep up the great work!"
        )
        summary_message = f"""
        This week your child completed {report.sessions_completed} learning
sessions with {report.total_study_time_minutes} minutes of study time.

        🌟 Highlights:
        • {len(report.concepts_mastered)} concepts mastered
        • {report.avg_engagement_score:.0%} average engagement
        • {report.avg_accuracy:.0%} average accuracy

        {achievements_text}
        """

        await self.create_alert(
            student_id=student_id,
            alert_type="weekly_report",
            level=AlertLevel.INFO,
            title="Weekly Progress Report",
            message=summary_message.strip(),
            suggested_actions=[
                "Review the full weekly report",
                "Discuss progress with your child",
            ],
            expires_hours=168,  # 7 days
        )

    async def start_progress_monitoring(self) -> None:
        """Start background progress monitoring."""
        while self.monitoring_active:
            try:
                await asyncio.sleep(3600)  # Check every hour

                # Generate weekly reports if it's the right time
                await self._check_weekly_report_schedule()

                # Clean up expired alerts
                await self._cleanup_expired_alerts()

                # Update progress trends
                await self._update_progress_trends()

            except Exception as e:
                logger.exception(f"Error in progress monitoring: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry

    async def _check_weekly_report_schedule(self) -> None:
        """Check if it's time to generate weekly reports."""
        now = datetime.now(timezone.utc)

        # Generate reports on Sunday evenings
        if now.weekday() == 6 and now.hour == 20:  # Sunday 8 PM
            for student_id in self.parent_settings:
                try:
                    await self.generate_weekly_report(student_id)
                except Exception as e:
                    logger.exception(
                        "Error generating weekly report for %s: %s", student_id, e
                    )

    async def _cleanup_expired_alerts(self) -> None:
        """Clean up expired alerts."""
        current_time = datetime.now(timezone.utc)

        for student_id in list(self.parent_alerts.keys()):
            alerts = self.parent_alerts[student_id]

            # Filter out expired alerts
            active_alerts = []
            for alert in alerts:
                if alert.expires_at:
                    expires_at = datetime.fromisoformat(alert.expires_at)
                    if current_time < expires_at:
                        active_alerts.append(alert)
                else:
                    active_alerts.append(alert)

            self.parent_alerts[student_id] = active_alerts

    async def _update_progress_trends(self) -> None:
        """Update long-term progress trends."""
        for student_id in self.parent_settings:
            # Update various trend analyses
            self._calculate_learning_trends(student_id)

    def _calculate_learning_trends(self, student_id: str) -> None:
        """Calculate learning trends for analytics."""
        # This would perform sophisticated trend analysis
        # For now, store basic placeholder data

        self.progress_trends[student_id] = {
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "learning_velocity_trend": "stable",
            "engagement_trend": "improving",
            "accuracy_trend": "stable",
        }

    # Database helper methods
    async def _save_milestone(self, milestone: ProgressMilestone) -> None:
        """Save milestone to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO progress_milestones
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    milestone.milestone_id,
                    milestone.student_id,
                    milestone.subject,
                    milestone.concept,
                    milestone.description,
                    milestone.target_mastery_level,
                    milestone.current_mastery_level,
                    1 if milestone.achieved else 0,
                    milestone.achieved_date,
                    milestone.estimated_completion,
                    milestone.difficulty_level,
                    milestone.importance,
                    milestone.celebration_message,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            logger.exception("Failed to save milestone: %s", e)

    async def _save_alert(self, alert: LearningAlert) -> None:
        """Save alert to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO learning_alerts VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    alert.alert_id,
                    alert.student_id,
                    alert.alert_type,
                    alert.level.value,
                    alert.title,
                    alert.message,
                    alert.created_at,
                    1 if alert.read else 0,
                    1 if alert.acknowledged else 0,
                    1 if alert.action_required else 0,
                    json.dumps(alert.related_concepts),
                    json.dumps(alert.suggested_actions),
                    alert.expires_at,
                ),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            logger.exception("Failed to save alert: %s", e)

    async def _save_weekly_report(self, report: WeeklyReport) -> None:
        """Save weekly report to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO weekly_reports VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    report.report_id,
                    report.student_id,
                    report.week_start,
                    report.week_end,
                    json.dumps(asdict(report)),
                    report.generated_at,
                ),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            logger.exception("Failed to save weekly report: %s", e)

    async def _save_insight(self, insight: ParentInsight) -> None:
        """Save parent insight to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO parent_insights VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    insight.insight_id,
                    insight.student_id,
                    insight.category,
                    insight.title,
                    insight.description,
                    json.dumps(insight.evidence),
                    json.dumps(insight.recommended_actions),
                    insight.priority,
                    insight.confidence_score,
                    insight.created_at,
                    1 if insight.implemented else 0,
                ),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            logger.exception("Failed to save insight: %s", e)

    async def _save_parent_settings(self, student_id: str, settings: dict[str, Any]) -> None:
        """Save parent settings to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO parent_settings VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    student_id,
                    settings["parent_email"],
                    settings["parent_name"],
                    json.dumps(settings["notification_preferences"]),
                    settings["report_frequency"],
                    settings["alert_threshold"],
                    json.dumps(settings["privacy_settings"]),
                    settings["created_at"],
                    settings["updated_at"],
                ),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            logger.exception("Failed to save parent settings: %s", e)

    def get_parent_dashboard_data(self, student_id: str) -> dict[str, Any]:
        """Get comprehensive dashboard data for parents."""
        # Get recent alerts
        recent_alerts = [
            asdict(alert) for alert in self.parent_alerts[student_id][-10:]
        ]

        # Get milestone progress
        milestone_progress = [
            {
                "concept": m.concept,
                "description": m.description,
                "progress": m.current_mastery_level,
                "target": m.target_mastery_level,
                "achieved": m.achieved,
            }
            for m in self.milestones[student_id]
        ]

        # Get recent insights
        recent_insights = [
            asdict(insight) for insight in self.parent_insights[student_id][-5:]
        ]

        # Get latest weekly report
        latest_report = None
        if self.weekly_reports[student_id]:
            latest_report = asdict(self.weekly_reports[student_id][-1])

        dashboard = {
            "student_id": student_id,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "alerts": {
                "recent": recent_alerts,
                "unread_count": len(
                    [a for a in self.parent_alerts[student_id] if not a.read]
                ),
                "action_required_count": len(
                    [
                        a
                        for a in self.parent_alerts[student_id]
                        if a.action_required and not a.acknowledged
                    ]
                ),
            },
            "milestones": {
                "progress": milestone_progress,
                "completed_count": len(
                    [m for m in self.milestones[student_id] if m.achieved]
                ),
                "total_count": len(self.milestones[student_id]),
            },
            "insights": {
                "recent": recent_insights,
                "high_priority_count": len(
                    [
                        i
                        for i in self.parent_insights[student_id]
                        if i.priority == "high" and not i.implemented
                    ]
                ),
            },
            "latest_report": latest_report,
            "quick_stats": {
                "learning_velocity": (
                    list(self.learning_velocity[student_id])[-1]["velocity"]
                    if self.learning_velocity[student_id]
                    else 0
                ),
                "engagement_trend": self.progress_trends.get(student_id, {}).get(
                    "engagement_trend", "unknown"
                ),
                "accuracy_trend": self.progress_trends.get(student_id, {}).get(
                    "accuracy_trend", "unknown"
                ),
            },
        }

        return dashboard


# Global parent progress tracker instance
parent_progress_tracker = ParentProgressTracker()
