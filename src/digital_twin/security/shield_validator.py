"""Shield Validator - AI Safety and Content Validation System
Sprint R-5: Digital Twin MVP - Task A.3.
"""

import asyncio
import hashlib
import json
import logging
import re
import sqlite3
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
import torch
from transformers import pipeline

import wandb

# Content filtering imports
try:
    import spacy

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning(
        "spaCy not available - some content analysis features will be limited"
    )

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationCategory(Enum):
    SAFETY = "safety"
    PRIVACY = "privacy"
    CONTENT = "content"
    EDUCATIONAL = "educational"
    TECHNICAL = "technical"
    COMPLIANCE = "compliance"


@dataclass
class ValidationRule:
    """Validation rule definition."""

    rule_id: str
    category: ValidationCategory
    severity: ValidationSeverity
    description: str
    pattern: str | None = None  # Regex pattern
    keywords: list[str] = None  # Keywords to check
    ml_classifier: str | None = None  # ML model for validation
    age_specific: bool = False
    min_age: int = 0
    max_age: int = 18
    enabled: bool = True


@dataclass
class ValidationResult:
    """Result of content validation."""

    validation_id: str
    content_hash: str
    student_id: str
    content_type: str  # tutor_response, user_input, system_message
    timestamp: str
    passed: bool
    violations: list[dict[str, Any]]
    warnings: list[dict[str, Any]]
    safety_score: float  # 0.0 to 1.0
    educational_value: float  # 0.0 to 1.0
    age_appropriateness: float  # 0.0 to 1.0
    privacy_compliant: bool
    recommendations: list[str]
    processing_time_ms: float


@dataclass
class ShieldMetrics:
    """Shield validation metrics."""

    total_validations: int = 0
    blocked_content: int = 0
    warnings_issued: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    avg_processing_time: float = 0.0
    safety_score_distribution: dict[str, int] = None
    common_violations: dict[str, int] = None


class ShieldValidator:
    """Comprehensive AI safety and content validation system."""

    def __init__(self, project_name: str = "aivillage-shield") -> None:
        self.project_name = project_name
        self.validation_rules = {}
        self.validation_history = []
        self.metrics = ShieldMetrics()
        self.blocked_patterns = set()

        # ML models for content analysis
        self.safety_classifier = None
        self.toxicity_classifier = None
        self.educational_classifier = None

        # Language processing
        self.nlp_processor = None

        # Content filters
        self.profanity_filter = set()
        self.educational_keywords = set()
        self.safety_keywords = set()

        # Age-specific content guidelines
        self.age_guidelines = {
            (0, 5): {
                "complexity": "very_simple",
                "topics": ["basic_counting", "colors", "shapes"],
            },
            (6, 8): {
                "complexity": "simple",
                "topics": ["basic_math", "reading", "nature"],
            },
            (9, 11): {
                "complexity": "moderate",
                "topics": ["multiplication", "science", "history"],
            },
            (12, 14): {
                "complexity": "advanced",
                "topics": ["algebra", "biology", "literature"],
            },
            (15, 18): {
                "complexity": "complex",
                "topics": ["calculus", "chemistry", "philosophy"],
            },
        }

        # Database for validation logs
        self.db_path = "shield_validation.db"
        self.init_database()

        # Performance monitoring
        self.performance_cache = {}
        self.validation_queue = asyncio.Queue(maxsize=1000)

        # Initialize W&B tracking
        self.initialize_wandb_tracking()

        # Load validation rules and models (only if event loop is available)
        try:
            asyncio.create_task(self.initialize_shield_system())
        except RuntimeError:
            # No event loop available - shield system can be initialized later
            logger.info(
                "No event loop available, shield system can be initialized manually"
            )

    def initialize_wandb_tracking(self) -> None:
        """Initialize W&B tracking for Shield validation."""
        try:
            wandb.init(
                project=self.project_name,
                job_type="shield_validation",
                config={
                    "shield_version": "2.0.0-enterprise",
                    "safety_models": [
                        "toxicity",
                        "educational_value",
                        "age_appropriateness",
                    ],
                    "compliance_standards": ["COPPA", "CIPA", "FERPA", "GDPR"],
                    "validation_categories": [cat.value for cat in ValidationCategory],
                    "severity_levels": [sev.value for sev in ValidationSeverity],
                    "real_time_validation": True,
                    "ml_enhanced": True,
                    "privacy_first": True,
                },
            )

            logger.info("Shield Validator W&B tracking initialized")

        except Exception as e:
            logger.exception(f"Failed to initialize W&B tracking: {e}")

    def init_database(self) -> None:
        """Initialize database for validation logs."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Validation results table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS validation_results (
                    validation_id TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    student_id TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    passed INTEGER NOT NULL,
                    violations TEXT,  -- JSON
                    warnings TEXT,    -- JSON
                    safety_score REAL,
                    educational_value REAL,
                    age_appropriateness REAL,
                    privacy_compliant INTEGER,
                    processing_time_ms REAL
                )
            """
            )

            # Validation rules table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS validation_rules (
                    rule_id TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    description TEXT NOT NULL,
                    rule_data TEXT NOT NULL,  -- JSON
                    enabled INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """
            )

            # Shield metrics table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS shield_metrics (
                    metric_date TEXT PRIMARY KEY,
                    total_validations INTEGER,
                    blocked_content INTEGER,
                    warnings_issued INTEGER,
                    avg_safety_score REAL,
                    common_violations TEXT  -- JSON
                )
            """
            )

            # Create indexes
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_validation_student ON validation_results(student_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_validation_timestamp ON validation_results(timestamp)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_validation_passed ON validation_results(passed)"
            )

            conn.commit()
            conn.close()

            logger.info("Shield validation database initialized")

        except Exception as e:
            logger.exception(f"Failed to initialize database: {e}")

    async def initialize_shield_system(self) -> None:
        """Initialize shield system with rules and models."""
        logger.info("Initializing Shield validation system...")

        # Load validation rules
        await self.load_validation_rules()

        # Initialize ML models
        await self.initialize_ml_models()

        # Load content filters
        await self.load_content_filters()

        # Initialize language processor
        await self.initialize_nlp_processor()

        # Start background validation processing
        asyncio.create_task(self.process_validation_queue())

        logger.info("Shield validation system initialized")

    async def load_validation_rules(self) -> None:
        """Load comprehensive validation rules."""
        # Safety rules
        safety_rules = [
            ValidationRule(
                rule_id="safety_001",
                category=ValidationCategory.SAFETY,
                severity=ValidationSeverity.CRITICAL,
                description="Detect harmful or dangerous content",
                keywords=[
                    "violence",
                    "weapon",
                    "hurt",
                    "kill",
                    "die",
                    "suicide",
                    "self-harm",
                ],
                age_specific=True,
                min_age=0,
                max_age=18,
            ),
            ValidationRule(
                rule_id="safety_002",
                category=ValidationCategory.SAFETY,
                severity=ValidationSeverity.ERROR,
                description="Detect inappropriate social content",
                keywords=["drugs", "alcohol", "smoking", "gambling", "dating"],
                age_specific=True,
                min_age=0,
                max_age=12,
            ),
            ValidationRule(
                rule_id="safety_003",
                category=ValidationCategory.SAFETY,
                severity=ValidationSeverity.WARNING,
                description="Detect potentially scary or disturbing content",
                keywords=["scary", "monster", "ghost", "nightmare", "fear", "terror"],
                age_specific=True,
                min_age=0,
                max_age=8,
            ),
        ]

        # Privacy rules
        privacy_rules = [
            ValidationRule(
                rule_id="privacy_001",
                category=ValidationCategory.PRIVACY,
                severity=ValidationSeverity.CRITICAL,
                description="Detect personal information requests",
                pattern=r"(what.*your.*(name|address|phone|email|school))|(tell.*me.*(where.*live|your.*address))",
                keywords=[
                    "full name",
                    "home address",
                    "phone number",
                    "email address",
                    "school name",
                ],
            ),
            ValidationRule(
                rule_id="privacy_002",
                category=ValidationCategory.PRIVACY,
                severity=ValidationSeverity.ERROR,
                description="Detect attempts to share personal info",
                pattern=r"(\d{3}-\d{3}-\d{4})|(\d{3}\.\d{3}\.\d{4})|([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
                keywords=["my phone", "my email", "my address", "I live at"],
            ),
        ]

        # Content appropriateness rules
        content_rules = [
            ValidationRule(
                rule_id="content_001",
                category=ValidationCategory.CONTENT,
                severity=ValidationSeverity.ERROR,
                description="Detect profanity and inappropriate language",
                ml_classifier="toxicity",
            ),
            ValidationRule(
                rule_id="content_002",
                category=ValidationCategory.CONTENT,
                severity=ValidationSeverity.WARNING,
                description="Detect complex language for young learners",
                age_specific=True,
                min_age=0,
                max_age=10,
            ),
        ]

        # Educational appropriateness rules
        educational_rules = [
            ValidationRule(
                rule_id="edu_001",
                category=ValidationCategory.EDUCATIONAL,
                severity=ValidationSeverity.WARNING,
                description="Ensure educational value in responses",
                ml_classifier="educational_value",
            ),
            ValidationRule(
                rule_id="edu_002",
                category=ValidationCategory.EDUCATIONAL,
                severity=ValidationSeverity.INFO,
                description="Check age-appropriate complexity",
                age_specific=True,
            ),
        ]

        # Technical validation rules
        technical_rules = [
            ValidationRule(
                rule_id="tech_001",
                category=ValidationCategory.TECHNICAL,
                severity=ValidationSeverity.WARNING,
                description="Detect potential prompt injection",
                pattern=r"(ignore.*instructions|system.*prompt|jailbreak|DAN|pretend.*you.*are)",
                keywords=[
                    "ignore previous",
                    "new instructions",
                    "role play",
                    "pretend you are",
                ],
            )
        ]

        # Compliance rules
        compliance_rules = [
            ValidationRule(
                rule_id="comp_001",
                category=ValidationCategory.COMPLIANCE,
                severity=ValidationSeverity.CRITICAL,
                description="COPPA compliance - no personal data collection from under 13",
                age_specific=True,
                min_age=0,
                max_age=12,
            )
        ]

        # Store all rules
        all_rules = (
            safety_rules
            + privacy_rules
            + content_rules
            + educational_rules
            + technical_rules
            + compliance_rules
        )

        for rule in all_rules:
            self.validation_rules[rule.rule_id] = rule

        logger.info(f"Loaded {len(all_rules)} validation rules")

    async def initialize_ml_models(self) -> None:
        """Initialize ML models for content analysis."""
        try:
            # Toxicity classifier
            self.toxicity_classifier = pipeline(
                "text-classification",
                model="unitary/toxic-bert",
                device=0 if torch.cuda.is_available() else -1,
            )

            # Educational value classifier (using a general sentiment model as proxy)
            self.educational_classifier = pipeline(
                "text-classification",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if torch.cuda.is_available() else -1,
            )

            logger.info("ML models initialized successfully")

        except Exception as e:
            logger.warning(f"Failed to initialize some ML models: {e}")

    async def load_content_filters(self) -> None:
        """Load content filter lists."""
        # Basic profanity filter (placeholder - in production would load from secure source)
        self.profanity_filter = {
            "damn",
            "hell",
            "crap",
            "stupid",
            "dumb",
            "idiot",
            "hate",
            "kill",
            "die",
        }

        # Educational keywords (positive indicators)
        self.educational_keywords = {
            "learn",
            "understand",
            "explain",
            "teach",
            "practice",
            "study",
            "explore",
            "discover",
            "solve",
            "calculate",
            "analyze",
            "think",
            "reason",
            "example",
            "step by step",
            "because",
            "therefore",
            "however",
            "in conclusion",
        }

        # Safety keywords (require careful handling)
        self.safety_keywords = {
            "safe",
            "careful",
            "appropriate",
            "suitable",
            "proper",
            "correct",
            "positive",
            "helpful",
            "educational",
            "learning",
            "family-friendly",
        }

        logger.info("Content filters loaded")

    async def initialize_nlp_processor(self) -> None:
        """Initialize NLP processor for advanced text analysis."""
        if SPACY_AVAILABLE:
            try:
                import spacy

                self.nlp_processor = spacy.load("en_core_web_sm")
                logger.info("spaCy NLP processor initialized")
            except OSError:
                logger.warning(
                    "spaCy English model not found - install with: python -m spacy download en_core_web_sm"
                )
                self.nlp_processor = None
        else:
            logger.warning("spaCy not available - advanced NLP features disabled")

    async def validate_content(
        self,
        content: str,
        student_id: str,
        content_type: str = "tutor_response",
        student_age: int = 10,
        context: dict[str, Any] | None = None,
    ) -> ValidationResult:
        """Comprehensive content validation."""
        start_time = asyncio.get_event_loop().time()

        # Generate content hash for caching and tracking
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        # Check cache first
        cache_key = f"{content_hash}_{student_age}_{content_type}"
        if cache_key in self.performance_cache:
            return self.performance_cache[cache_key]

        # Initialize validation result
        validation_id = f"val_{hashlib.md5(f'{student_id}_{content_hash}_{datetime.now().isoformat()}'.encode()).hexdigest()[:12]}"

        violations = []
        warnings = []

        # Run all validation checks
        safety_score = await self._check_safety(
            content, student_age, violations, warnings
        )
        privacy_score = await self._check_privacy(content, violations, warnings)
        await self._check_content_appropriateness(
            content, student_age, violations, warnings
        )
        educational_score = await self._check_educational_value(
            content, student_age, warnings
        )
        age_score = await self._check_age_appropriateness(
            content, student_age, warnings
        )
        technical_score = await self._check_technical_safety(
            content, violations, warnings
        )

        # Calculate overall scores
        overall_safety_score = min(safety_score, privacy_score, technical_score)
        educational_value = educational_score
        age_appropriateness = age_score
        privacy_compliant = privacy_score >= 0.8

        # Determine if content passes validation
        passed = (
            len([v for v in violations if v["severity"] in ["critical", "error"]]) == 0
            and overall_safety_score >= 0.7
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            violations, warnings, overall_safety_score, educational_value
        )

        # Calculate processing time
        processing_time = (asyncio.get_event_loop().time() - start_time) * 1000

        # Create validation result
        result = ValidationResult(
            validation_id=validation_id,
            content_hash=content_hash,
            student_id=student_id,
            content_type=content_type,
            timestamp=datetime.now(UTC).isoformat(),
            passed=passed,
            violations=violations,
            warnings=warnings,
            safety_score=overall_safety_score,
            educational_value=educational_value,
            age_appropriateness=age_appropriateness,
            privacy_compliant=privacy_compliant,
            recommendations=recommendations,
            processing_time_ms=processing_time,
        )

        # Cache result
        self.performance_cache[cache_key] = result

        # Limit cache size
        if len(self.performance_cache) > 1000:
            # Remove oldest entries
            keys_to_remove = list(self.performance_cache.keys())[:200]
            for key in keys_to_remove:
                del self.performance_cache[key]

        # Store validation result
        self.validation_history.append(result)
        await self._save_validation_result(result)

        # Update metrics
        self._update_metrics(result)

        # Log to W&B
        wandb.log(
            {
                "shield/validation_completed": True,
                "shield/passed": passed,
                "shield/safety_score": overall_safety_score,
                "shield/educational_value": educational_value,
                "shield/age_appropriateness": age_appropriateness,
                "shield/violations": len(violations),
                "shield/warnings": len(warnings),
                "shield/processing_time_ms": processing_time,
                "shield/content_type": content_type,
                "shield/student_age": student_age,
                "timestamp": result.timestamp,
            }
        )

        logger.info(
            f"Validated content {content_hash} for student {student_id[:8]} - {'PASSED' if passed else 'BLOCKED'}"
        )

        return result

    async def _check_safety(
        self, content: str, student_age: int, violations: list, warnings: list
    ) -> float:
        """Check content safety."""
        safety_score = 1.0
        content_lower = content.lower()

        # Check safety rules
        for rule in self.validation_rules.values():
            if rule.category != ValidationCategory.SAFETY or not rule.enabled:
                continue

            # Age-specific rules
            if rule.age_specific and not (rule.min_age <= student_age <= rule.max_age):
                continue

            # Check keywords
            if rule.keywords:
                found_keywords = [
                    kw for kw in rule.keywords if kw.lower() in content_lower
                ]
                if found_keywords:
                    violation = {
                        "rule_id": rule.rule_id,
                        "severity": rule.severity.value,
                        "description": rule.description,
                        "found_keywords": found_keywords,
                        "category": "safety",
                    }

                    if rule.severity in [
                        ValidationSeverity.CRITICAL,
                        ValidationSeverity.ERROR,
                    ]:
                        violations.append(violation)
                        safety_score -= 0.3
                    else:
                        warnings.append(violation)
                        safety_score -= 0.1

            # Check patterns
            if rule.pattern:
                matches = re.findall(rule.pattern, content_lower, re.IGNORECASE)
                if matches:
                    violation = {
                        "rule_id": rule.rule_id,
                        "severity": rule.severity.value,
                        "description": rule.description,
                        "pattern_matches": matches,
                        "category": "safety",
                    }

                    if rule.severity in [
                        ValidationSeverity.CRITICAL,
                        ValidationSeverity.ERROR,
                    ]:
                        violations.append(violation)
                        safety_score -= 0.4
                    else:
                        warnings.append(violation)
                        safety_score -= 0.1

        # ML-based toxicity detection
        if self.toxicity_classifier:
            try:
                result = self.toxicity_classifier(
                    content[:512]
                )  # Limit length for performance
                if result[0]["label"] == "TOXIC" and result[0]["score"] > 0.8:
                    violations.append(
                        {
                            "rule_id": "ml_toxicity",
                            "severity": "error",
                            "description": "Content flagged as potentially toxic by ML model",
                            "confidence": result[0]["score"],
                            "category": "safety",
                        }
                    )
                    safety_score -= 0.5
                elif result[0]["label"] == "TOXIC" and result[0]["score"] > 0.6:
                    warnings.append(
                        {
                            "rule_id": "ml_toxicity_warning",
                            "severity": "warning",
                            "description": "Content may contain inappropriate language",
                            "confidence": result[0]["score"],
                            "category": "safety",
                        }
                    )
                    safety_score -= 0.2
            except Exception as e:
                logger.warning(f"Toxicity classification failed: {e}")

        return max(0.0, safety_score)

    async def _check_privacy(
        self, content: str, violations: list, warnings: list
    ) -> float:
        """Check privacy compliance."""
        privacy_score = 1.0

        # Check privacy rules
        for rule in self.validation_rules.values():
            if rule.category != ValidationCategory.PRIVACY or not rule.enabled:
                continue

            # Check patterns (email, phone, etc.)
            if rule.pattern:
                matches = re.findall(rule.pattern, content, re.IGNORECASE)
                if matches:
                    violation = {
                        "rule_id": rule.rule_id,
                        "severity": rule.severity.value,
                        "description": rule.description,
                        "detected_info": [
                            match if isinstance(match, str) else match[0]
                            for match in matches
                        ],
                        "category": "privacy",
                    }

                    if rule.severity in [
                        ValidationSeverity.CRITICAL,
                        ValidationSeverity.ERROR,
                    ]:
                        violations.append(violation)
                        privacy_score -= 0.6
                    else:
                        warnings.append(violation)
                        privacy_score -= 0.2

            # Check keywords
            if rule.keywords:
                found_keywords = [
                    kw for kw in rule.keywords if kw.lower() in content.lower()
                ]
                if found_keywords:
                    violation = {
                        "rule_id": rule.rule_id,
                        "severity": rule.severity.value,
                        "description": rule.description,
                        "found_keywords": found_keywords,
                        "category": "privacy",
                    }

                    if rule.severity in [
                        ValidationSeverity.CRITICAL,
                        ValidationSeverity.ERROR,
                    ]:
                        violations.append(violation)
                        privacy_score -= 0.4
                    else:
                        warnings.append(violation)
                        privacy_score -= 0.1

        return max(0.0, privacy_score)

    async def _check_content_appropriateness(
        self, content: str, student_age: int, violations: list, warnings: list
    ) -> float:
        """Check content appropriateness."""
        content_score = 1.0
        content_lower = content.lower()

        # Check for profanity
        found_profanity = [
            word for word in self.profanity_filter if word in content_lower
        ]
        if found_profanity:
            violation = {
                "rule_id": "profanity_filter",
                "severity": "error",
                "description": "Content contains inappropriate language",
                "found_words": found_profanity,
                "category": "content",
            }
            violations.append(violation)
            content_score -= 0.5

        # Check reading level for young students
        if student_age <= 8:
            # Simple readability check
            words = content.split()
            avg_word_length = np.mean([len(word) for word in words]) if words else 0
            sentences = content.count(".") + content.count("!") + content.count("?")
            avg_sentence_length = len(words) / max(sentences, 1)

            if avg_word_length > 6 or avg_sentence_length > 15:
                warnings.append(
                    {
                        "rule_id": "reading_level",
                        "severity": "warning",
                        "description": "Content may be too complex for student's age",
                        "avg_word_length": avg_word_length,
                        "avg_sentence_length": avg_sentence_length,
                        "category": "content",
                    }
                )
                content_score -= 0.2

        return max(0.0, content_score)

    async def _check_educational_value(
        self, content: str, student_age: int, warnings: list
    ) -> float:
        """Check educational value of content."""
        educational_score = 0.5  # Start neutral
        content_lower = content.lower()

        # Check for educational keywords
        found_educational = [
            kw for kw in self.educational_keywords if kw in content_lower
        ]
        educational_score += len(found_educational) * 0.1

        # Check for explanation patterns
        explanation_patterns = [
            r"because\s+",
            r"this\s+means\s+",
            r"for\s+example\s+",
            r"step\s+by\s+step",
            r"let\s+me\s+explain",
            r"here\s+is\s+why",
        ]

        explanation_count = sum(
            len(re.findall(pattern, content_lower)) for pattern in explanation_patterns
        )
        educational_score += explanation_count * 0.1

        # Use ML classifier if available
        if self.educational_classifier:
            try:
                # Use sentiment as proxy for educational positivity
                result = self.educational_classifier(content[:512])
                if result[0]["label"] == "LABEL_2":  # Positive
                    educational_score += 0.2
                elif result[0]["label"] == "LABEL_0":  # Negative
                    educational_score -= 0.1
            except Exception as e:
                logger.warning(f"Educational classification failed: {e}")

        # Check for mathematical content (for math tutoring)
        math_indicators = [
            "equation",
            "solve",
            "calculate",
            "answer",
            "result",
            "+",
            "-",
            "×",
            "÷",
            "=",
        ]
        math_count = sum(
            1 for indicator in math_indicators if indicator in content_lower
        )
        if math_count > 0:
            educational_score += 0.2

        return min(1.0, max(0.0, educational_score))

    async def _check_age_appropriateness(
        self, content: str, student_age: int, warnings: list
    ) -> float:
        """Check age appropriateness of content."""
        age_score = 1.0

        # Get age guidelines
        age_guideline = None
        for (min_age, max_age), guideline in self.age_guidelines.items():
            if min_age <= student_age <= max_age:
                age_guideline = guideline
                break

        if not age_guideline:
            return 0.8  # Default if no specific guideline

        # Check complexity
        words = content.split()
        if not words:
            return age_score

        avg_word_length = np.mean([len(word) for word in words])
        sentences = max(1, content.count(".") + content.count("!") + content.count("?"))
        avg_sentence_length = len(words) / sentences

        # Age-specific complexity thresholds
        complexity_thresholds = {
            "very_simple": {"word_length": 4, "sentence_length": 8},
            "simple": {"word_length": 5, "sentence_length": 12},
            "moderate": {"word_length": 6, "sentence_length": 16},
            "advanced": {"word_length": 7, "sentence_length": 20},
            "complex": {"word_length": 8, "sentence_length": 25},
        }

        threshold = complexity_thresholds.get(
            age_guideline["complexity"], complexity_thresholds["moderate"]
        )

        if avg_word_length > threshold["word_length"] * 1.5:
            warnings.append(
                {
                    "rule_id": "age_complexity_words",
                    "severity": "warning",
                    "description": f"Word complexity may be too high for age {student_age}",
                    "avg_word_length": avg_word_length,
                    "recommended_max": threshold["word_length"],
                    "category": "educational",
                }
            )
            age_score -= 0.2

        if avg_sentence_length > threshold["sentence_length"] * 1.5:
            warnings.append(
                {
                    "rule_id": "age_complexity_sentences",
                    "severity": "warning",
                    "description": f"Sentence complexity may be too high for age {student_age}",
                    "avg_sentence_length": avg_sentence_length,
                    "recommended_max": threshold["sentence_length"],
                    "category": "educational",
                }
            )
            age_score -= 0.2

        return max(0.0, age_score)

    async def _check_technical_safety(
        self, content: str, violations: list, warnings: list
    ) -> float:
        """Check for technical safety issues like prompt injection."""
        technical_score = 1.0

        # Check technical rules
        for rule in self.validation_rules.values():
            if rule.category != ValidationCategory.TECHNICAL or not rule.enabled:
                continue

            # Check patterns
            if rule.pattern:
                matches = re.findall(rule.pattern, content, re.IGNORECASE)
                if matches:
                    violation = {
                        "rule_id": rule.rule_id,
                        "severity": rule.severity.value,
                        "description": rule.description,
                        "pattern_matches": matches,
                        "category": "technical",
                    }

                    if rule.severity in [
                        ValidationSeverity.CRITICAL,
                        ValidationSeverity.ERROR,
                    ]:
                        violations.append(violation)
                        technical_score -= 0.5
                    else:
                        warnings.append(violation)
                        technical_score -= 0.2

            # Check keywords
            if rule.keywords:
                found_keywords = [
                    kw for kw in rule.keywords if kw.lower() in content.lower()
                ]
                if found_keywords:
                    violation = {
                        "rule_id": rule.rule_id,
                        "severity": rule.severity.value,
                        "description": rule.description,
                        "found_keywords": found_keywords,
                        "category": "technical",
                    }

                    if rule.severity in [
                        ValidationSeverity.CRITICAL,
                        ValidationSeverity.ERROR,
                    ]:
                        violations.append(violation)
                        technical_score -= 0.4
                    else:
                        warnings.append(violation)
                        technical_score -= 0.1

        return max(0.0, technical_score)

    def _generate_recommendations(
        self,
        violations: list,
        warnings: list,
        safety_score: float,
        educational_value: float,
    ) -> list[str]:
        """Generate content improvement recommendations."""
        recommendations = []

        # Safety recommendations
        if safety_score < 0.7:
            recommendations.append(
                "Review content for potentially harmful or inappropriate material"
            )

        # Educational recommendations
        if educational_value < 0.5:
            recommendations.append(
                "Consider adding more educational elements like explanations or examples"
            )

        # Specific violation recommendations
        violation_types = set()
        for violation in violations + warnings:
            violation_types.add(violation.get("category", "general"))

        if "privacy" in violation_types:
            recommendations.append(
                "Remove or mask any personal information requests or sharing"
            )

        if "content" in violation_types:
            recommendations.append("Use more age-appropriate language and concepts")

        if "technical" in violation_types:
            recommendations.append(
                "Ensure content follows proper AI interaction guidelines"
            )

        # General recommendations if no specific issues
        if not recommendations:
            if educational_value > 0.8:
                recommendations.append(
                    "Content meets all safety and educational standards"
                )
            else:
                recommendations.append(
                    "Consider enhancing educational value with more detailed explanations"
                )

        return recommendations

    async def _save_validation_result(self, result: ValidationResult) -> None:
        """Save validation result to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO validation_results
                (validation_id, content_hash, student_id, content_type, timestamp, passed,
                 violations, warnings, safety_score, educational_value, age_appropriateness,
                 privacy_compliant, processing_time_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    result.validation_id,
                    result.content_hash,
                    result.student_id,
                    result.content_type,
                    result.timestamp,
                    1 if result.passed else 0,
                    json.dumps(result.violations),
                    json.dumps(result.warnings),
                    result.safety_score,
                    result.educational_value,
                    result.age_appropriateness,
                    1 if result.privacy_compliant else 0,
                    result.processing_time_ms,
                ),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            logger.exception(f"Failed to save validation result: {e}")

    def _update_metrics(self, result: ValidationResult) -> None:
        """Update shield metrics."""
        self.metrics.total_validations += 1

        if not result.passed:
            self.metrics.blocked_content += 1

        if result.warnings:
            self.metrics.warnings_issued += 1

        # Update average processing time
        total_time = self.metrics.avg_processing_time * (
            self.metrics.total_validations - 1
        )
        self.metrics.avg_processing_time = (
            total_time + result.processing_time_ms
        ) / self.metrics.total_validations

        # Update safety score distribution
        if self.metrics.safety_score_distribution is None:
            self.metrics.safety_score_distribution = defaultdict(int)

        score_bucket = f"{int(result.safety_score * 10) * 10}-{int(result.safety_score * 10) * 10 + 10}"
        self.metrics.safety_score_distribution[score_bucket] += 1

        # Update common violations
        if self.metrics.common_violations is None:
            self.metrics.common_violations = defaultdict(int)

        for violation in result.violations:
            self.metrics.common_violations[violation.get("rule_id", "unknown")] += 1

    async def process_validation_queue(self) -> None:
        """Process validation requests from queue."""
        while True:
            try:
                # Get validation request from queue
                validation_request = await self.validation_queue.get()

                # Process validation
                await self.validate_content(**validation_request)

                # Mark task as done
                self.validation_queue.task_done()

            except Exception as e:
                logger.exception(f"Error processing validation queue: {e}")
                await asyncio.sleep(1)

    async def batch_validate(
        self, content_list: list[dict[str, Any]]
    ) -> list[ValidationResult]:
        """Validate multiple content items efficiently."""
        results = []

        # Process in parallel with semaphore to limit concurrency
        semaphore = asyncio.Semaphore(10)  # Max 10 concurrent validations

        async def validate_single(content_item):
            async with semaphore:
                return await self.validate_content(**content_item)

        # Create tasks for all validations
        tasks = [validate_single(item) for item in content_list]

        # Wait for all validations to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and log them
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Validation failed for item {i}: {result}")
            else:
                valid_results.append(result)

        return valid_results

    async def get_validation_analytics(
        self, student_id: str | None = None, days: int = 7
    ) -> dict[str, Any]:
        """Get validation analytics."""
        # Filter results by student and time period
        cutoff_date = datetime.now(UTC) - timedelta(days=days)

        filtered_results = [
            result
            for result in self.validation_history
            if (student_id is None or result.student_id == student_id)
            and datetime.fromisoformat(result.timestamp) > cutoff_date
        ]

        if not filtered_results:
            return {"message": "No validation data available"}

        analytics = {
            "period_days": days,
            "total_validations": len(filtered_results),
            "passed_validations": len([r for r in filtered_results if r.passed]),
            "blocked_content": len([r for r in filtered_results if not r.passed]),
            "pass_rate": len([r for r in filtered_results if r.passed])
            / len(filtered_results),
            "avg_safety_score": np.mean([r.safety_score for r in filtered_results]),
            "avg_educational_value": np.mean(
                [r.educational_value for r in filtered_results]
            ),
            "avg_age_appropriateness": np.mean(
                [r.age_appropriateness for r in filtered_results]
            ),
            "avg_processing_time_ms": np.mean(
                [r.processing_time_ms for r in filtered_results]
            ),
            "violation_categories": defaultdict(int),
            "content_type_breakdown": defaultdict(int),
            "safety_trends": [],
        }

        # Analyze violations by category
        for result in filtered_results:
            for violation in result.violations:
                category = violation.get("category", "unknown")
                analytics["violation_categories"][category] += 1

            # Content type breakdown
            analytics["content_type_breakdown"][result.content_type] += 1

        # Safety score trends (daily averages)
        daily_scores = defaultdict(list)
        for result in filtered_results:
            date = result.timestamp[:10]  # YYYY-MM-DD
            daily_scores[date].append(result.safety_score)

        for date, scores in daily_scores.items():
            analytics["safety_trends"].append(
                {
                    "date": date,
                    "avg_safety_score": np.mean(scores),
                    "validations": len(scores),
                }
            )

        # Sort trends by date
        analytics["safety_trends"].sort(key=lambda x: x["date"])

        return analytics

    async def export_validation_report(
        self, student_id: str, format: str = "json"
    ) -> str:
        """Export detailed validation report."""
        # Get student's validation history
        student_results = [
            r for r in self.validation_history if r.student_id == student_id
        ]

        report = {
            "student_id": student_id,
            "report_generated": datetime.now(UTC).isoformat(),
            "total_validations": len(student_results),
            "validation_history": [
                asdict(result) for result in student_results[-50:]
            ],  # Last 50
            "analytics": await self.get_validation_analytics(student_id, days=30),
        }

        if format == "json":
            return json.dumps(report, indent=2, default=str)
        return str(report)

    def get_shield_status(self) -> dict[str, Any]:
        """Get current shield system status."""
        return {
            "status": "active",
            "version": "2.0.0-enterprise",
            "rules_loaded": len(self.validation_rules),
            "rules_enabled": len(
                [r for r in self.validation_rules.values() if r.enabled]
            ),
            "ml_models_available": {
                "toxicity_classifier": self.toxicity_classifier is not None,
                "educational_classifier": self.educational_classifier is not None,
                "nlp_processor": self.nlp_processor is not None,
            },
            "performance": {
                "total_validations": self.metrics.total_validations,
                "blocked_content": self.metrics.blocked_content,
                "avg_processing_time_ms": self.metrics.avg_processing_time,
                "cache_size": len(self.performance_cache),
            },
            "last_updated": datetime.now(UTC).isoformat(),
        }


# Global shield validator instance
shield_validator = ShieldValidator()
