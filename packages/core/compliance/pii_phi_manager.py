"""
Comprehensive PII/PHI Management and Data Retention System for AIVillage.

This module provides complete Privacy and Healthcare data compliance management including:
- Automated PII/PHI discovery and classification
- GDPR/HIPAA/CCPA compliance monitoring
- Automated data retention and deletion
- Privacy impact assessments
- Data lineage tracking
- Breach detection and reporting
"""

import asyncio
import hashlib
import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DataClassification(Enum):
    """Data classification levels for privacy compliance."""

    PUBLIC = "public"  # No privacy concerns
    INTERNAL = "internal"  # Internal use only
    PII = "pii"  # Personally Identifiable Information
    PHI = "phi"  # Protected Health Information
    FINANCIAL = "financial"  # Financial/payment data
    BIOMETRIC = "biometric"  # Biometric identifiers
    SENSITIVE = "sensitive"  # Other sensitive data
    CONFIDENTIAL = "confidential"  # Highest sensitivity


class RetentionPolicy(Enum):
    """Data retention policy types."""

    IMMEDIATE = "immediate"  # Delete immediately after use
    SHORT_TERM = "short_term"  # 30 days retention
    STANDARD = "standard"  # 1 year retention
    LONG_TERM = "long_term"  # 3 years retention
    HEALTHCARE = "healthcare"  # 7 years (HIPAA requirement)
    FINANCIAL = "financial"  # 5 years (financial regulations)
    LEGAL_HOLD = "legal_hold"  # Indefinite retention
    CUSTOM = "custom"  # Custom retention period


class ComplianceRegulation(Enum):
    """Supported privacy regulations."""

    GDPR = "gdpr"  # EU General Data Protection Regulation
    HIPAA = "hipaa"  # US Health Insurance Portability and Accountability Act
    CCPA = "ccpa"  # California Consumer Privacy Act
    PIPEDA = "pipeda"  # Personal Information Protection and Electronic Documents Act
    SOX = "sox"  # Sarbanes-Oxley Act
    FERPA = "ferpa"  # Family Educational Rights and Privacy Act


@dataclass
class PIIDetectionRule:
    """Rule for detecting PII/PHI in data."""

    rule_id: str
    name: str
    description: str
    classification: DataClassification
    pattern: str  # Regex pattern for detection
    field_names: list[str] = field(default_factory=list)  # Common field names
    confidence_threshold: float = 0.8  # Confidence threshold for match
    regulation: ComplianceRegulation | None = None
    examples: list[str] = field(default_factory=list)
    false_positive_patterns: list[str] = field(default_factory=list)


@dataclass
class DataLocation:
    """Location where PII/PHI data is found."""

    location_id: str
    source_type: str  # database, file, api_endpoint, etc.
    path: str  # Full path or identifier
    table_name: str | None = None  # Database table
    column_name: str | None = None  # Database column
    field_name: str | None = None  # JSON/document field
    tenant_id: str | None = None  # Associated tenant
    classification: DataClassification = DataClassification.INTERNAL
    retention_policy: RetentionPolicy = RetentionPolicy.STANDARD
    regulations: list[ComplianceRegulation] = field(default_factory=list)

    # Discovery metadata
    discovered_at: datetime = field(default_factory=datetime.utcnow)
    last_verified: datetime | None = None
    sample_count: int = 0
    confidence_score: float = 0.0

    # Compliance status
    compliant: bool = True
    last_audit: datetime | None = None
    violations: list[str] = field(default_factory=list)

    # Data characteristics
    estimated_records: int = 0
    data_size_bytes: int = 0
    access_frequency: int = 0
    last_accessed: datetime | None = None


@dataclass
class RetentionJob:
    """Automated data retention job."""

    job_id: str
    name: str
    description: str
    location_ids: list[str]  # Data locations to process
    retention_policy: RetentionPolicy
    custom_retention_days: int | None = None

    # Scheduling
    enabled: bool = True
    schedule_cron: str = "0 2 * * 0"  # Weekly at 2 AM Sunday
    last_run: datetime | None = None
    next_run: datetime | None = None

    # Execution results
    run_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_records_processed: int = 0
    total_records_deleted: int = 0
    last_error: str | None = None

    # Compliance
    requires_approval: bool = True
    approval_status: str = "pending"  # pending, approved, denied
    approver: str | None = None
    approved_at: datetime | None = None


class PIIDetectionEngine:
    """Engine for detecting PII/PHI in various data sources."""

    def __init__(self):
        """Initialize detection engine with built-in rules."""
        self.detection_rules: dict[str, PIIDetectionRule] = {}
        self.compiled_patterns: dict[str, re.Pattern] = {}
        self._initialize_builtin_rules()

    def _initialize_builtin_rules(self):
        """Initialize built-in detection rules for common PII/PHI."""

        rules = [
            # Personal Identifiers
            PIIDetectionRule(
                rule_id="ssn",
                name="Social Security Number",
                description="US Social Security Number",
                classification=DataClassification.PII,
                pattern=r"\b\d{3}-?\d{2}-?\d{4}\b",
                field_names=["ssn", "social_security", "social_security_number"],
                regulation=ComplianceRegulation.HIPAA,
                examples=["123-45-6789", "987654321"],
                false_positive_patterns=[r"\b000-?\d{2}-?\d{4}\b"],  # Invalid SSNs
            ),
            PIIDetectionRule(
                rule_id="email",
                name="Email Address",
                description="Email address",
                classification=DataClassification.PII,
                pattern=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                field_names=["email", "email_address", "contact_email"],
                regulation=ComplianceRegulation.GDPR,
                examples=["user@example.com", "test.email+tag@domain.co.uk"],
            ),
            PIIDetectionRule(
                rule_id="phone_us",
                name="US Phone Number",
                description="US phone number in various formats",
                classification=DataClassification.PII,
                pattern=r"\b(?:\+?1[-.\s]?)?\(?([2-9][0-8][0-9])\)?[-.\s]?([2-9][0-9]{2})[-.\s]?([0-9]{4})\b",
                field_names=["phone", "phone_number", "telephone", "mobile"],
                regulation=ComplianceRegulation.CCPA,
                examples=["(555) 123-4567", "555.123.4567", "+1-555-123-4567"],
            ),
            PIIDetectionRule(
                rule_id="credit_card",
                name="Credit Card Number",
                description="Credit card number (Visa, MasterCard, AmEx, Discover)",
                classification=DataClassification.FINANCIAL,
                pattern=r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b",
                field_names=["credit_card", "card_number", "payment_method"],
                regulation=ComplianceRegulation.SOX,
                examples=["4532015112830366", "5555555555554444"],
            ),
            # Health Information
            PIIDetectionRule(
                rule_id="medical_record",
                name="Medical Record Number",
                description="Medical record number",
                classification=DataClassification.PHI,
                pattern=r"\b(?:MRN|MR|MED|MEDICAL)[-.\s]?#?[\s]?([A-Z0-9]{6,12})\b",
                field_names=["mrn", "medical_record", "medical_id", "patient_id"],
                regulation=ComplianceRegulation.HIPAA,
                examples=["MRN-123456789", "MED#A1B2C3D4"],
            ),
            PIIDetectionRule(
                rule_id="npi",
                name="National Provider Identifier",
                description="Healthcare provider NPI number",
                classification=DataClassification.PHI,
                pattern=r"\b[1-9][0-9]{9}\b",
                field_names=["npi", "provider_id", "national_provider"],
                regulation=ComplianceRegulation.HIPAA,
                examples=["1234567890", "9876543210"],
            ),
            PIIDetectionRule(
                rule_id="diagnosis_code",
                name="Medical Diagnosis Code",
                description="ICD-10 or ICD-9 diagnosis codes",
                classification=DataClassification.PHI,
                pattern=r"\b(?:[A-Z]\d{2}(?:\.\d{1,4})?|[0-9]{3}(?:\.[0-9]{1,2})?)\b",
                field_names=["icd", "diagnosis", "diagnosis_code", "medical_code"],
                regulation=ComplianceRegulation.HIPAA,
                examples=["Z51.11", "250.00", "R06.02"],
            ),
            # Biometric Data
            PIIDetectionRule(
                rule_id="fingerprint",
                name="Fingerprint Hash",
                description="Fingerprint biometric hash",
                classification=DataClassification.BIOMETRIC,
                pattern=r"\b[A-Fa-f0-9]{40,64}\b",  # SHA-1 or SHA-256 style
                field_names=["fingerprint", "biometric", "finger_hash"],
                regulation=ComplianceRegulation.GDPR,
                examples=["a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3"],
            ),
            # Government Identifiers
            PIIDetectionRule(
                rule_id="passport",
                name="Passport Number",
                description="Passport number",
                classification=DataClassification.PII,
                pattern=r"\b[A-Z]{1,2}[0-9]{6,9}\b",
                field_names=["passport", "passport_number", "passport_id"],
                regulation=ComplianceRegulation.GDPR,
                examples=["A12345678", "AB1234567"],
            ),
            PIIDetectionRule(
                rule_id="drivers_license",
                name="Driver's License",
                description="Driver's license number",
                classification=DataClassification.PII,
                pattern=r"\b[A-Z]{1,2}[0-9]{6,8}\b",
                field_names=["license", "drivers_license", "dl_number"],
                regulation=ComplianceRegulation.CCPA,
                examples=["D12345678", "CA9876543"],
            ),
            # IP Addresses (can be PII under GDPR)
            PIIDetectionRule(
                rule_id="ip_address",
                name="IP Address",
                description="IPv4 or IPv6 address",
                classification=DataClassification.PII,
                pattern=r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b|(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b",
                field_names=["ip", "ip_address", "client_ip", "remote_addr"],
                regulation=ComplianceRegulation.GDPR,
                examples=["192.168.1.1", "2001:0db8:85a3:0000:0000:8a2e:0370:7334"],
            ),
            # Financial Information
            PIIDetectionRule(
                rule_id="bank_account",
                name="Bank Account Number",
                description="Bank account number",
                classification=DataClassification.FINANCIAL,
                pattern=r"\b[0-9]{8,17}\b",
                field_names=["account", "account_number", "bank_account"],
                regulation=ComplianceRegulation.SOX,
                examples=["12345678901234", "987654321"],
            ),
            PIIDetectionRule(
                rule_id="routing_number",
                name="Bank Routing Number",
                description="US bank routing number (ABA)",
                classification=DataClassification.FINANCIAL,
                pattern=r"\b[0-9]{9}\b",
                field_names=["routing", "routing_number", "aba"],
                regulation=ComplianceRegulation.SOX,
                examples=["011401533", "124003116"],
            ),
        ]

        for rule in rules:
            self.add_detection_rule(rule)

    def add_detection_rule(self, rule: PIIDetectionRule):
        """Add a detection rule to the engine."""
        self.detection_rules[rule.rule_id] = rule
        try:
            self.compiled_patterns[rule.rule_id] = re.compile(rule.pattern, re.IGNORECASE)
        except re.error as e:
            logger.error(f"Invalid regex pattern for rule {rule.rule_id}: {e}")

    def detect_in_text(self, text: str) -> list[tuple[PIIDetectionRule, list[str]]]:
        """Detect PII/PHI in text content."""
        detections = []

        for rule_id, rule in self.detection_rules.items():
            if rule_id not in self.compiled_patterns:
                continue

            pattern = self.compiled_patterns[rule_id]
            matches = pattern.findall(text)

            if matches:
                # Filter out false positives
                filtered_matches = []
                for match in matches:
                    match_str = match if isinstance(match, str) else "".join(match)

                    # Check false positive patterns
                    is_false_positive = False
                    for fp_pattern in rule.false_positive_patterns:
                        if re.match(fp_pattern, match_str, re.IGNORECASE):
                            is_false_positive = True
                            break

                    if not is_false_positive:
                        filtered_matches.append(match_str)

                if filtered_matches:
                    detections.append((rule, filtered_matches))

        return detections

    def detect_in_field_name(self, field_name: str) -> list[PIIDetectionRule]:
        """Detect potential PII/PHI based on field names."""
        matches = []
        field_lower = field_name.lower()

        for rule in self.detection_rules.values():
            for known_field in rule.field_names:
                if known_field in field_lower:
                    matches.append(rule)
                    break

        return matches

    def calculate_confidence(
        self, rule: PIIDetectionRule, matches: list[str], field_name: str = "", total_samples: int = 1
    ) -> float:
        """Calculate confidence score for a detection."""
        base_confidence = 0.5

        # Boost confidence for field name matches
        if field_name:
            field_matches = self.detect_in_field_name(field_name)
            if rule in field_matches:
                base_confidence += 0.3

        # Boost confidence based on match quality
        if matches:
            # Higher confidence for more matches (up to a point)
            match_ratio = min(len(matches) / total_samples, 0.5)
            base_confidence += match_ratio * 0.3

            # Boost confidence for well-formed matches
            valid_matches = 0
            for match in matches[:10]:  # Check first 10 matches
                if self._validate_match(rule, match):
                    valid_matches += 1

            if matches:
                validation_ratio = valid_matches / min(len(matches), 10)
                base_confidence += validation_ratio * 0.2

        return min(base_confidence, 1.0)

    def _validate_match(self, rule: PIIDetectionRule, match: str) -> bool:
        """Validate a match using additional checks."""
        # Add specific validation logic for different data types
        if rule.rule_id == "ssn":
            # Basic SSN validation
            digits = re.sub(r"[^0-9]", "", match)
            if len(digits) != 9:
                return False
            if digits[:3] == "000" or digits[3:5] == "00" or digits[5:] == "0000":
                return False
            return True

        elif rule.rule_id == "credit_card":
            # Luhn algorithm validation
            return self._luhn_check(re.sub(r"[^0-9]", "", match))

        elif rule.rule_id == "email":
            # Basic email validation beyond regex
            parts = match.split("@")
            if len(parts) != 2:
                return False
            local, domain = parts
            return len(local) <= 64 and len(domain) <= 253

        # Default validation - just check pattern match
        return True

    def _luhn_check(self, card_number: str) -> bool:
        """Validate credit card number using Luhn algorithm."""

        def luhn_checksum(card_num):
            def digits_of(n):
                return [int(d) for d in str(n)]

            digits = digits_of(card_num)
            odd_digits = digits[-1::-2]
            even_digits = digits[-2::-2]
            checksum = sum(odd_digits)
            for d in even_digits:
                checksum += sum(digits_of(d * 2))
            return checksum % 10

        return luhn_checksum(card_number) == 0


class PIIPHIManager:
    """Main PII/PHI management and compliance system."""

    def __init__(self, config_path: Path | None = None):
        """Initialize PII/PHI manager."""
        self.config_path = config_path or Path("config/compliance/pii_phi_config.json")
        self.detection_engine = PIIDetectionEngine()

        # Database paths
        self.compliance_db = Path("data/compliance/pii_phi.db")
        self.compliance_db.parent.mkdir(parents=True, exist_ok=True)

        # Data locations and retention jobs
        self.data_locations: dict[str, DataLocation] = {}
        self.retention_jobs: dict[str, RetentionJob] = {}

        # Configuration
        self.config = self._load_config()

        # Initialize database
        self._init_database()
        self._load_existing_data()

        logger.info("PII/PHI Manager initialized")

    def _load_config(self) -> dict[str, Any]:
        """Load configuration file."""
        default_config = {
            "scanning": {
                "enabled": True,
                "auto_discovery": True,
                "scan_schedule": "0 3 * * 0",  # Weekly at 3 AM Sunday
                "batch_size": 1000,
                "sample_size": 100,
                "confidence_threshold": 0.7,
            },
            "retention": {
                "enabled": True,
                "default_policy": "standard",
                "require_approval": True,
                "grace_period_days": 30,
                "notification_days": [30, 7, 1],  # Days before deletion to notify
            },
            "compliance": {
                "default_regulations": ["gdpr"],
                "audit_schedule": "0 4 1 * *",  # Monthly audit
                "report_retention_days": 2555,  # 7 years
                "auto_classification": True,
            },
            "monitoring": {
                "enabled": True,
                "alert_on_new_pii": True,
                "alert_on_violations": True,
                "performance_monitoring": True,
            },
        }

        if self.config_path.exists():
            with open(self.config_path) as f:
                user_config = json.load(f)
                default_config.update(user_config)
        else:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, "w") as f:
                json.dump(default_config, f, indent=2)

        return default_config

    def _init_database(self):
        """Initialize compliance database."""
        conn = sqlite3.connect(self.compliance_db)
        cursor = conn.cursor()

        # Data locations table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS data_locations (
                location_id TEXT PRIMARY KEY,
                source_type TEXT NOT NULL,
                path TEXT NOT NULL,
                table_name TEXT,
                column_name TEXT,
                field_name TEXT,
                tenant_id TEXT,
                classification TEXT NOT NULL,
                retention_policy TEXT NOT NULL,
                regulations TEXT,  -- JSON array
                discovered_at TIMESTAMP NOT NULL,
                last_verified TIMESTAMP,
                sample_count INTEGER DEFAULT 0,
                confidence_score REAL DEFAULT 0.0,
                compliant BOOLEAN DEFAULT 1,
                last_audit TIMESTAMP,
                violations TEXT,  -- JSON array
                estimated_records INTEGER DEFAULT 0,
                data_size_bytes INTEGER DEFAULT 0,
                access_frequency INTEGER DEFAULT 0,
                last_accessed TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Retention jobs table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS retention_jobs (
                job_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                location_ids TEXT,  -- JSON array
                retention_policy TEXT NOT NULL,
                custom_retention_days INTEGER,
                enabled BOOLEAN DEFAULT 1,
                schedule_cron TEXT,
                last_run TIMESTAMP,
                next_run TIMESTAMP,
                run_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                total_records_processed INTEGER DEFAULT 0,
                total_records_deleted INTEGER DEFAULT 0,
                last_error TEXT,
                requires_approval BOOLEAN DEFAULT 1,
                approval_status TEXT DEFAULT 'pending',
                approver TEXT,
                approved_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Audit log table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                audit_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_id TEXT,
                tenant_id TEXT,
                location_id TEXT,
                job_id TEXT,
                details TEXT,  -- JSON object
                classification TEXT,
                regulation TEXT,
                compliance_status TEXT,
                risk_level TEXT
            )
        """
        )

        # Detection history table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS detection_history (
                detection_id TEXT PRIMARY KEY,
                location_id TEXT NOT NULL,
                rule_id TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                matches_found INTEGER DEFAULT 0,
                confidence_score REAL DEFAULT 0.0,
                sample_data TEXT,  -- Hashed/anonymized sample
                false_positive BOOLEAN DEFAULT 0,
                verified_by TEXT,
                verified_at TIMESTAMP
            )
        """
        )

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_locations_classification ON data_locations(classification)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_locations_tenant ON data_locations(tenant_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_locations_compliance ON data_locations(compliant)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_enabled ON retention_jobs(enabled)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_next_run ON retention_jobs(next_run)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_event_type ON audit_log(event_type)")

        conn.commit()
        conn.close()

    def _load_existing_data(self):
        """Load existing data locations and retention jobs from database."""
        conn = sqlite3.connect(self.compliance_db)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Load data locations
        cursor.execute("SELECT * FROM data_locations")
        for row in cursor.fetchall():
            location = DataLocation(
                location_id=row["location_id"],
                source_type=row["source_type"],
                path=row["path"],
                table_name=row["table_name"],
                column_name=row["column_name"],
                field_name=row["field_name"],
                tenant_id=row["tenant_id"],
                classification=DataClassification(row["classification"]),
                retention_policy=RetentionPolicy(row["retention_policy"]),
                regulations=[ComplianceRegulation(r) for r in json.loads(row["regulations"] or "[]")],
                discovered_at=datetime.fromisoformat(row["discovered_at"]),
                last_verified=datetime.fromisoformat(row["last_verified"]) if row["last_verified"] else None,
                sample_count=row["sample_count"],
                confidence_score=row["confidence_score"],
                compliant=bool(row["compliant"]),
                last_audit=datetime.fromisoformat(row["last_audit"]) if row["last_audit"] else None,
                violations=json.loads(row["violations"] or "[]"),
                estimated_records=row["estimated_records"],
                data_size_bytes=row["data_size_bytes"],
                access_frequency=row["access_frequency"],
                last_accessed=datetime.fromisoformat(row["last_accessed"]) if row["last_accessed"] else None,
            )
            self.data_locations[location.location_id] = location

        # Load retention jobs
        cursor.execute("SELECT * FROM retention_jobs")
        for row in cursor.fetchall():
            job = RetentionJob(
                job_id=row["job_id"],
                name=row["name"],
                description=row["description"],
                location_ids=json.loads(row["location_ids"] or "[]"),
                retention_policy=RetentionPolicy(row["retention_policy"]),
                custom_retention_days=row["custom_retention_days"],
                enabled=bool(row["enabled"]),
                schedule_cron=row["schedule_cron"],
                last_run=datetime.fromisoformat(row["last_run"]) if row["last_run"] else None,
                next_run=datetime.fromisoformat(row["next_run"]) if row["next_run"] else None,
                run_count=row["run_count"],
                success_count=row["success_count"],
                failure_count=row["failure_count"],
                total_records_processed=row["total_records_processed"],
                total_records_deleted=row["total_records_deleted"],
                last_error=row["last_error"],
                requires_approval=bool(row["requires_approval"]),
                approval_status=row["approval_status"],
                approver=row["approver"],
                approved_at=datetime.fromisoformat(row["approved_at"]) if row["approved_at"] else None,
            )
            self.retention_jobs[job.job_id] = job

        conn.close()

        logger.info(f"Loaded {len(self.data_locations)} data locations and {len(self.retention_jobs)} retention jobs")

    # Discovery methods

    async def discover_pii_phi_in_database(self, db_path: str, tenant_id: str | None = None) -> list[DataLocation]:
        """Discover PII/PHI in database tables."""
        discoveries = []

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            for table_name in tables:
                # Skip system tables
                if table_name.startswith("sqlite_"):
                    continue

                # Get table schema
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()

                for col_info in columns:
                    col_name = col_info[1]

                    # Check field name for PII indicators
                    field_rules = self.detection_engine.detect_in_field_name(col_name)

                    if field_rules:
                        # Sample data from column
                        cursor.execute(
                            f"SELECT DISTINCT {col_name} FROM {table_name} WHERE {col_name} IS NOT NULL LIMIT 100"
                        )
                        samples = [row[0] for row in cursor.fetchall() if row[0]]

                        # Analyze samples for PII content
                        all_detections = []
                        for sample in samples:
                            if isinstance(sample, str):
                                detections = self.detection_engine.detect_in_text(sample)
                                all_detections.extend(detections)

                        # Create data location if PII found
                        if all_detections or field_rules:
                            best_rule = field_rules[0] if field_rules else all_detections[0][0]
                            confidence = self.detection_engine.calculate_confidence(
                                best_rule, [d[1] for d in all_detections], col_name, len(samples)
                            )

                            if confidence >= self.config["scanning"]["confidence_threshold"]:
                                location_id = (
                                    f"db_{hashlib.md5(f'{db_path}:{table_name}:{col_name}'.encode()).hexdigest()[:16]}"
                                )

                                # Get record count
                                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                                record_count = cursor.fetchone()[0]

                                location = DataLocation(
                                    location_id=location_id,
                                    source_type="database",
                                    path=db_path,
                                    table_name=table_name,
                                    column_name=col_name,
                                    tenant_id=tenant_id,
                                    classification=best_rule.classification,
                                    retention_policy=self._determine_retention_policy(best_rule),
                                    regulations=[best_rule.regulation] if best_rule.regulation else [],
                                    sample_count=len(samples),
                                    confidence_score=confidence,
                                    estimated_records=record_count,
                                )

                                discoveries.append(location)
                                self.data_locations[location_id] = location
                                await self._save_data_location(location)

                                # Log discovery
                                await self._log_audit_event(
                                    "pii_discovered",
                                    location_id=location_id,
                                    tenant_id=tenant_id,
                                    details={
                                        "source": db_path,
                                        "table": table_name,
                                        "column": col_name,
                                        "classification": best_rule.classification.value,
                                        "confidence": confidence,
                                        "record_count": record_count,
                                    },
                                )

            conn.close()

        except Exception as e:
            logger.error(f"Error discovering PII/PHI in database {db_path}: {e}")
            await self._log_audit_event("discovery_error", details={"source": db_path, "error": str(e)})

        logger.info(f"Discovered {len(discoveries)} PII/PHI locations in {db_path}")
        return discoveries

    async def discover_pii_phi_in_files(
        self, directory: Path, tenant_id: str | None = None, file_patterns: list[str] = None
    ) -> list[DataLocation]:
        """Discover PII/PHI in files."""
        if file_patterns is None:
            file_patterns = ["*.json", "*.txt", "*.log", "*.csv"]

        discoveries = []

        for pattern in file_patterns:
            for file_path in directory.rglob(pattern):
                if file_path.is_file():
                    try:
                        # Read file content (sample for large files)
                        content = ""
                        with open(file_path, encoding="utf-8", errors="ignore") as f:
                            if file_path.stat().st_size > 1024 * 1024:  # 1MB limit for sampling
                                content = f.read(1024 * 100)  # Read first 100KB
                            else:
                                content = f.read()

                        # Detect PII in content
                        detections = self.detection_engine.detect_in_text(content)

                        if detections:
                            for rule, matches in detections:
                                confidence = self.detection_engine.calculate_confidence(
                                    rule, matches, file_path.name, 1
                                )

                                if confidence >= self.config["scanning"]["confidence_threshold"]:
                                    location_id = f"file_{hashlib.md5(str(file_path).encode()).hexdigest()[:16]}"

                                    location = DataLocation(
                                        location_id=location_id,
                                        source_type="file",
                                        path=str(file_path),
                                        tenant_id=tenant_id,
                                        classification=rule.classification,
                                        retention_policy=self._determine_retention_policy(rule),
                                        regulations=[rule.regulation] if rule.regulation else [],
                                        sample_count=len(matches),
                                        confidence_score=confidence,
                                        estimated_records=len(matches),
                                        data_size_bytes=file_path.stat().st_size,
                                    )

                                    discoveries.append(location)
                                    self.data_locations[location_id] = location
                                    await self._save_data_location(location)

                    except Exception as e:
                        logger.warning(f"Error scanning file {file_path}: {e}")

        logger.info(f"Discovered {len(discoveries)} PII/PHI locations in files under {directory}")
        return discoveries

    async def scan_all_known_locations(self) -> dict[str, list[DataLocation]]:
        """Scan all known AIVillage data locations for PII/PHI."""
        all_discoveries = {"databases": [], "files": [], "configurations": []}

        # Scan databases
        database_paths = [
            "data/rbac.db",
            "data/tenants.db",
            "data/vector_db/qdrant.db",
            "packages/core/legacy/database/evolution.db",
        ]

        for db_path in database_paths:
            if Path(db_path).exists():
                discoveries = await self.discover_pii_phi_in_database(db_path)
                all_discoveries["databases"].extend(discoveries)

        # Scan tenant-specific databases
        tenant_dirs = Path("data/tenants")
        if tenant_dirs.exists():
            for tenant_dir in tenant_dirs.iterdir():
                if tenant_dir.is_dir():
                    tenant_id = tenant_dir.name

                    # Scan tenant databases
                    for db_file in tenant_dir.rglob("*.db"):
                        discoveries = await self.discover_pii_phi_in_database(str(db_file), tenant_id)
                        all_discoveries["databases"].extend(discoveries)

                    # Scan tenant files
                    discoveries = await self.discover_pii_phi_in_files(tenant_dir, tenant_id)
                    all_discoveries["files"].extend(discoveries)

        # Scan log files
        log_dirs = [Path("logs"), Path("data/logs")]
        for log_dir in log_dirs:
            if log_dir.exists():
                discoveries = await self.discover_pii_phi_in_files(log_dir, file_patterns=["*.log", "*.txt"])
                all_discoveries["files"].extend(discoveries)

        # Scan configuration files
        config_dirs = [Path("config"), Path(".env*")]
        for config_dir in config_dirs:
            if config_dir.exists():
                discoveries = await self.discover_pii_phi_in_files(
                    config_dir, file_patterns=["*.json", "*.yaml", "*.yml", "*.env*"]
                )
                all_discoveries["configurations"].extend(discoveries)

        # Log comprehensive scan results
        total_discoveries = sum(len(discoveries) for discoveries in all_discoveries.values())
        await self._log_audit_event(
            "comprehensive_scan",
            details={
                "total_discoveries": total_discoveries,
                "databases": len(all_discoveries["databases"]),
                "files": len(all_discoveries["files"]),
                "configurations": len(all_discoveries["configurations"]),
            },
        )

        logger.info(f"Comprehensive scan completed: {total_discoveries} PII/PHI locations found")
        return all_discoveries

    def _determine_retention_policy(self, rule: PIIDetectionRule) -> RetentionPolicy:
        """Determine retention policy based on data classification and regulation."""
        if rule.classification == DataClassification.PHI:
            return RetentionPolicy.HEALTHCARE
        elif rule.classification == DataClassification.FINANCIAL:
            return RetentionPolicy.FINANCIAL
        elif rule.regulation == ComplianceRegulation.HIPAA:
            return RetentionPolicy.HEALTHCARE
        elif rule.regulation == ComplianceRegulation.SOX:
            return RetentionPolicy.FINANCIAL
        else:
            return RetentionPolicy.STANDARD

    # Retention management methods

    async def create_retention_job(
        self,
        name: str,
        description: str,
        location_ids: list[str],
        retention_policy: RetentionPolicy,
        custom_retention_days: int | None = None,
        schedule_cron: str = "0 2 * * 0",
    ) -> str:
        """Create a new data retention job."""
        job_id = f"retention_{int(time.time())}_{hashlib.md5(name.encode()).hexdigest()[:8]}"

        job = RetentionJob(
            job_id=job_id,
            name=name,
            description=description,
            location_ids=location_ids,
            retention_policy=retention_policy,
            custom_retention_days=custom_retention_days,
            schedule_cron=schedule_cron,
        )

        self.retention_jobs[job_id] = job
        await self._save_retention_job(job)

        await self._log_audit_event(
            "retention_job_created",
            job_id=job_id,
            details={
                "name": name,
                "policy": retention_policy.value,
                "location_count": len(location_ids),
                "custom_days": custom_retention_days,
            },
        )

        logger.info(f"Created retention job: {job_id} ({name})")
        return job_id

    async def execute_retention_job(self, job_id: str) -> dict[str, Any]:
        """Execute a retention job."""
        if job_id not in self.retention_jobs:
            raise ValueError(f"Retention job {job_id} not found")

        job = self.retention_jobs[job_id]

        if not job.enabled:
            raise ValueError(f"Retention job {job_id} is disabled")

        if job.requires_approval and job.approval_status != "approved":
            raise ValueError(f"Retention job {job_id} requires approval")

        job.last_run = datetime.utcnow()
        job.run_count += 1

        results = {
            "job_id": job_id,
            "started_at": job.last_run.isoformat(),
            "locations_processed": 0,
            "records_deleted": 0,
            "errors": [],
        }

        try:
            # Calculate retention cutoff date
            if job.retention_policy == RetentionPolicy.CUSTOM:
                retention_days = job.custom_retention_days or 365
            else:
                retention_days = self._get_retention_days(job.retention_policy)

            cutoff_date = datetime.utcnow() - timedelta(days=retention_days)

            # Process each location
            for location_id in job.location_ids:
                if location_id not in self.data_locations:
                    results["errors"].append(f"Location {location_id} not found")
                    continue

                location = self.data_locations[location_id]

                try:
                    deleted_count = await self._delete_expired_data(location, cutoff_date)
                    results["records_deleted"] += deleted_count
                    results["locations_processed"] += 1

                    # Update location access info
                    location.last_accessed = datetime.utcnow()
                    await self._save_data_location(location)

                except Exception as e:
                    error_msg = f"Failed to process location {location_id}: {e}"
                    results["errors"].append(error_msg)
                    logger.error(error_msg)

            # Update job statistics
            if results["errors"]:
                job.failure_count += 1
                job.last_error = "; ".join(results["errors"])
            else:
                job.success_count += 1
                job.last_error = None

            job.total_records_processed += results["locations_processed"]
            job.total_records_deleted += results["records_deleted"]

            await self._save_retention_job(job)

            # Log completion
            results["completed_at"] = datetime.utcnow().isoformat()
            await self._log_audit_event("retention_job_executed", job_id=job_id, details=results)

            logger.info(f"Retention job {job_id} completed: {results['records_deleted']} records deleted")

        except Exception as e:
            job.failure_count += 1
            job.last_error = str(e)
            await self._save_retention_job(job)

            results["errors"].append(str(e))
            results["completed_at"] = datetime.utcnow().isoformat()

            await self._log_audit_event("retention_job_failed", job_id=job_id, details=results)

            logger.error(f"Retention job {job_id} failed: {e}")

        return results

    def _get_retention_days(self, policy: RetentionPolicy) -> int:
        """Get retention days for a policy."""
        policy_days = {
            RetentionPolicy.IMMEDIATE: 0,
            RetentionPolicy.SHORT_TERM: 30,
            RetentionPolicy.STANDARD: 365,
            RetentionPolicy.LONG_TERM: 1095,  # 3 years
            RetentionPolicy.HEALTHCARE: 2555,  # 7 years
            RetentionPolicy.FINANCIAL: 1825,  # 5 years
            RetentionPolicy.LEGAL_HOLD: 36500,  # 100 years (effectively permanent)
        }
        return policy_days.get(policy, 365)

    async def _delete_expired_data(self, location: DataLocation, cutoff_date: datetime) -> int:
        """Delete expired data from a specific location."""
        deleted_count = 0

        if location.source_type == "database":
            deleted_count = await self._delete_from_database(location, cutoff_date)
        elif location.source_type == "file":
            deleted_count = await self._delete_from_file(location, cutoff_date)

        return deleted_count

    async def _delete_from_database(self, location: DataLocation, cutoff_date: datetime) -> int:
        """Delete expired data from database."""
        try:
            conn = sqlite3.connect(location.path)
            cursor = conn.cursor()

            # For this implementation, we'll use a generic approach
            # In production, this would be customized per table structure

            # Try to find a timestamp column
            cursor.execute(f"PRAGMA table_info({location.table_name})")
            columns = cursor.fetchall()

            timestamp_columns = []
            for col_info in columns:
                col_name = col_info[1].lower()
                if any(ts in col_name for ts in ["created", "timestamp", "date", "time"]):
                    timestamp_columns.append(col_info[1])

            if timestamp_columns:
                # Delete records older than cutoff date
                timestamp_col = timestamp_columns[0]
                cursor.execute(
                    f"""
                    DELETE FROM {location.table_name}
                    WHERE {timestamp_col} < ?
                """,
                    (cutoff_date.isoformat(),),
                )

                deleted_count = cursor.rowcount
                conn.commit()

                logger.info(f"Deleted {deleted_count} records from {location.table_name}")
            else:
                logger.warning(f"No timestamp column found in {location.table_name}")
                deleted_count = 0

            conn.close()
            return deleted_count

        except Exception as e:
            logger.error(f"Error deleting from database {location.path}: {e}")
            raise

    async def _delete_from_file(self, location: DataLocation, cutoff_date: datetime) -> int:
        """Delete expired data from file."""
        try:
            file_path = Path(location.path)

            # Check file modification time
            if file_path.exists():
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)

                if file_mtime < cutoff_date:
                    file_path.unlink()
                    logger.info(f"Deleted expired file: {file_path}")
                    return 1

            return 0

        except Exception as e:
            logger.error(f"Error deleting file {location.path}: {e}")
            raise

    # Utility methods

    async def _save_data_location(self, location: DataLocation):
        """Save data location to database."""
        conn = sqlite3.connect(self.compliance_db)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO data_locations
            (location_id, source_type, path, table_name, column_name, field_name, tenant_id,
             classification, retention_policy, regulations, discovered_at, last_verified,
             sample_count, confidence_score, compliant, last_audit, violations,
             estimated_records, data_size_bytes, access_frequency, last_accessed, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
            (
                location.location_id,
                location.source_type,
                location.path,
                location.table_name,
                location.column_name,
                location.field_name,
                location.tenant_id,
                location.classification.value,
                location.retention_policy.value,
                json.dumps([r.value for r in location.regulations]),
                location.discovered_at.isoformat(),
                location.last_verified.isoformat() if location.last_verified else None,
                location.sample_count,
                location.confidence_score,
                location.compliant,
                location.last_audit.isoformat() if location.last_audit else None,
                json.dumps(location.violations),
                location.estimated_records,
                location.data_size_bytes,
                location.access_frequency,
                location.last_accessed.isoformat() if location.last_accessed else None,
            ),
        )

        conn.commit()
        conn.close()

    async def _save_retention_job(self, job: RetentionJob):
        """Save retention job to database."""
        conn = sqlite3.connect(self.compliance_db)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO retention_jobs
            (job_id, name, description, location_ids, retention_policy, custom_retention_days,
             enabled, schedule_cron, last_run, next_run, run_count, success_count, failure_count,
             total_records_processed, total_records_deleted, last_error, requires_approval,
             approval_status, approver, approved_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
            (
                job.job_id,
                job.name,
                job.description,
                json.dumps(job.location_ids),
                job.retention_policy.value,
                job.custom_retention_days,
                job.enabled,
                job.schedule_cron,
                job.last_run.isoformat() if job.last_run else None,
                job.next_run.isoformat() if job.next_run else None,
                job.run_count,
                job.success_count,
                job.failure_count,
                job.total_records_processed,
                job.total_records_deleted,
                job.last_error,
                job.requires_approval,
                job.approval_status,
                job.approver,
                job.approved_at.isoformat() if job.approved_at else None,
            ),
        )

        conn.commit()
        conn.close()

    async def _log_audit_event(
        self,
        event_type: str,
        user_id: str | None = None,
        tenant_id: str | None = None,
        location_id: str | None = None,
        job_id: str | None = None,
        details: dict | None = None,
        classification: DataClassification | None = None,
        regulation: ComplianceRegulation | None = None,
    ):
        """Log audit event."""
        audit_id = f"audit_{int(time.time())}_{hashlib.md5(f'{event_type}{location_id}'.encode()).hexdigest()[:8]}"

        conn = sqlite3.connect(self.compliance_db)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO audit_log
            (audit_id, event_type, timestamp, user_id, tenant_id, location_id, job_id,
             details, classification, regulation, compliance_status, risk_level)
            VALUES (?, ?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                audit_id,
                event_type,
                user_id,
                tenant_id,
                location_id,
                job_id,
                json.dumps(details) if details else None,
                classification.value if classification else None,
                regulation.value if regulation else None,
                "compliant",  # Default status
                self._calculate_risk_level(classification, details),
            ),
        )

        conn.commit()
        conn.close()

    def _calculate_risk_level(self, classification: DataClassification | None, details: dict | None) -> str:
        """Calculate risk level for audit event."""
        if classification == DataClassification.PHI:
            return "high"
        elif classification in [DataClassification.PII, DataClassification.FINANCIAL]:
            return "medium"
        else:
            return "low"

    # Query methods

    async def scan_job_inputs_for_compliance(
        self, job_id: str, job_inputs: dict[str, Any], namespace: str = "default", user_id: str | None = None
    ) -> dict[str, Any]:
        """
        Scan fog computing job inputs for PII/PHI violations

        Args:
            job_id: Unique job identifier
            job_inputs: Job input data including payload, env vars, args
            namespace: Job namespace
            user_id: User submitting the job

        Returns:
            dict: Scan results with violations and recommendations
        """

        scan_result = {
            "job_id": job_id,
            "namespace": namespace,
            "scan_timestamp": datetime.utcnow().isoformat(),
            "violations": [],
            "compliance_issues": [],
            "recommendations": [],
            "risk_level": "LOW",
            "action_required": False,
            "audit_trail": [],
        }

        try:
            # Log audit event for scan
            await self._log_audit_event(
                "job_input_scan_started",
                job_id=job_id,
                details={"namespace": namespace, "user_id": user_id, "input_keys": list(job_inputs.keys())},
            )

            # Scan different input types
            violations_found = []

            # 1. Scan job payload/code
            if "payload" in job_inputs:
                payload_violations = await self._scan_job_payload(job_inputs["payload"], job_id)
                violations_found.extend(payload_violations)

            # 2. Scan environment variables
            if "env" in job_inputs:
                env_violations = await self._scan_environment_variables(job_inputs["env"], job_id)
                violations_found.extend(env_violations)

            # 3. Scan command line arguments
            if "args" in job_inputs:
                args_violations = await self._scan_command_arguments(job_inputs["args"], job_id)
                violations_found.extend(args_violations)

            # 4. Scan input data files
            if "input_data" in job_inputs:
                data_violations = await self._scan_input_data(job_inputs["input_data"], job_id)
                violations_found.extend(data_violations)

            # 5. Scan metadata and labels
            if "metadata" in job_inputs:
                metadata_violations = await self._scan_metadata(job_inputs["metadata"], job_id)
                violations_found.extend(metadata_violations)

            # Process all violations
            scan_result["violations"] = violations_found

            if violations_found:
                scan_result["action_required"] = True

                # Categorize violations by severity
                high_severity = [v for v in violations_found if v["severity"] == "HIGH"]
                medium_severity = [v for v in violations_found if v["severity"] == "MEDIUM"]

                if high_severity:
                    scan_result["risk_level"] = "HIGH"
                    scan_result["compliance_issues"].extend(
                        [
                            f"Found {len(high_severity)} high-severity PII/PHI violations that must be resolved",
                            "Job execution should be blocked until violations are addressed",
                        ]
                    )
                elif medium_severity:
                    scan_result["risk_level"] = "MEDIUM"
                    scan_result["compliance_issues"].extend(
                        [
                            f"Found {len(medium_severity)} medium-severity privacy violations",
                            "Review and sanitize input data before proceeding",
                        ]
                    )
                else:
                    scan_result["risk_level"] = "LOW"

                # Generate recommendations
                scan_result["recommendations"] = await self._generate_remediation_recommendations(violations_found)

            # Log completion
            await self._log_audit_event(
                "job_input_scan_completed",
                job_id=job_id,
                details={
                    "violations_count": len(violations_found),
                    "risk_level": scan_result["risk_level"],
                    "action_required": scan_result["action_required"],
                },
            )

            logger.info(
                f"Job input scan for {job_id}: {len(violations_found)} violations, risk: {scan_result['risk_level']}"
            )

        except Exception as e:
            error_msg = f"Job input scan failed for {job_id}: {e}"
            scan_result["violations"].append(
                {"type": "SCAN_ERROR", "severity": "HIGH", "message": error_msg, "location": "scan_engine"}
            )

            await self._log_audit_event("job_input_scan_failed", job_id=job_id, details={"error": str(e)})

            logger.error(error_msg)

        return scan_result

    async def _scan_job_payload(self, payload: bytes | str, job_id: str) -> list[dict[str, Any]]:
        """Scan job payload for PII/PHI"""
        violations = []

        try:
            # Convert bytes to string for analysis
            if isinstance(payload, bytes):
                try:
                    text_content = payload.decode("utf-8")
                except UnicodeDecodeError:
                    # Binary content - limited scanning
                    text_content = str(payload[:1000])  # Sample first 1KB
            else:
                text_content = str(payload)

            # Detect PII/PHI in content
            detections = self.detection_engine.detect_in_text(text_content)

            for rule, matches in detections:
                for match in matches:
                    violation = {
                        "type": "PII_IN_PAYLOAD",
                        "rule_id": rule.rule_id,
                        "classification": rule.classification.value,
                        "severity": self._get_violation_severity(rule.classification),
                        "message": f"Found {rule.name} in job payload",
                        "location": "job_payload",
                        "match": match[:50] + "..." if len(match) > 50 else match,
                        "regulation": rule.regulation.value if rule.regulation else None,
                        "confidence": 0.9,  # High confidence for content matches
                    }
                    violations.append(violation)

        except Exception as e:
            logger.warning(f"Error scanning job payload for {job_id}: {e}")

        return violations

    async def _scan_environment_variables(self, env_vars: dict[str, str], job_id: str) -> list[dict[str, Any]]:
        """Scan environment variables for sensitive data"""
        violations = []

        for key, value in env_vars.items():
            # Check variable name for sensitive indicators
            key_detections = self.detection_engine.detect_in_field_name(key.lower())
            if key_detections:
                for rule in key_detections:
                    violation = {
                        "type": "SENSITIVE_ENV_VAR_NAME",
                        "rule_id": rule.rule_id,
                        "classification": rule.classification.value,
                        "severity": self._get_violation_severity(rule.classification),
                        "message": f"Environment variable name '{key}' suggests sensitive data",
                        "location": f"env.{key}",
                        "regulation": rule.regulation.value if rule.regulation else None,
                        "confidence": 0.8,
                    }
                    violations.append(violation)

            # Check variable value for PII/PHI
            if isinstance(value, str):
                value_detections = self.detection_engine.detect_in_text(value)
                for rule, matches in value_detections:
                    for match in matches:
                        violation = {
                            "type": "PII_IN_ENV_VAR",
                            "rule_id": rule.rule_id,
                            "classification": rule.classification.value,
                            "severity": "HIGH",  # Environment variables are high risk
                            "message": f"Found {rule.name} in environment variable '{key}'",
                            "location": f"env.{key}",
                            "match": match[:20] + "..." if len(match) > 20 else match,
                            "regulation": rule.regulation.value if rule.regulation else None,
                            "confidence": 0.95,
                        }
                        violations.append(violation)

        return violations

    async def _scan_command_arguments(self, args: list[str], job_id: str) -> list[dict[str, Any]]:
        """Scan command line arguments for sensitive data"""
        violations = []

        for i, arg in enumerate(args):
            if isinstance(arg, str):
                # Check for common sensitive argument patterns
                if any(sensitive in arg.lower() for sensitive in ["password", "token", "key", "secret"]):
                    violation = {
                        "type": "SENSITIVE_COMMAND_ARG",
                        "classification": "sensitive",
                        "severity": "MEDIUM",
                        "message": f"Command argument {i} appears to contain sensitive data",
                        "location": f"args[{i}]",
                        "match": arg[:30] + "..." if len(arg) > 30 else arg,
                        "confidence": 0.7,
                    }
                    violations.append(violation)

                # Check for PII/PHI patterns
                detections = self.detection_engine.detect_in_text(arg)
                for rule, matches in detections:
                    for match in matches:
                        violation = {
                            "type": "PII_IN_COMMAND_ARG",
                            "rule_id": rule.rule_id,
                            "classification": rule.classification.value,
                            "severity": self._get_violation_severity(rule.classification),
                            "message": f"Found {rule.name} in command argument {i}",
                            "location": f"args[{i}]",
                            "match": match[:30] + "..." if len(match) > 30 else match,
                            "regulation": rule.regulation.value if rule.regulation else None,
                            "confidence": 0.9,
                        }
                        violations.append(violation)

        return violations

    async def _scan_input_data(self, input_data: Any, job_id: str) -> list[dict[str, Any]]:
        """Scan input data for PII/PHI"""
        violations = []

        try:
            # Convert input data to string for analysis
            if isinstance(input_data, dict | list):
                text_content = json.dumps(input_data, default=str)
            elif isinstance(input_data, bytes):
                try:
                    text_content = input_data.decode("utf-8")
                except UnicodeDecodeError:
                    text_content = str(input_data[:1000])
            else:
                text_content = str(input_data)

            # Detect PII/PHI patterns
            detections = self.detection_engine.detect_in_text(text_content)

            for rule, matches in detections:
                for match in matches:
                    violation = {
                        "type": "PII_IN_INPUT_DATA",
                        "rule_id": rule.rule_id,
                        "classification": rule.classification.value,
                        "severity": self._get_violation_severity(rule.classification),
                        "message": f"Found {rule.name} in input data",
                        "location": "input_data",
                        "match": match[:50] + "..." if len(match) > 50 else match,
                        "regulation": rule.regulation.value if rule.regulation else None,
                        "confidence": 0.9,
                    }
                    violations.append(violation)

        except Exception as e:
            logger.warning(f"Error scanning input data for {job_id}: {e}")

        return violations

    async def _scan_metadata(self, metadata: dict[str, Any], job_id: str) -> list[dict[str, Any]]:
        """Scan job metadata and labels for sensitive information"""
        violations = []

        for key, value in metadata.items():
            # Check metadata key names
            key_detections = self.detection_engine.detect_in_field_name(key.lower())
            if key_detections:
                for rule in key_detections:
                    violation = {
                        "type": "SENSITIVE_METADATA_KEY",
                        "rule_id": rule.rule_id,
                        "classification": rule.classification.value,
                        "severity": "MEDIUM",
                        "message": f"Metadata key '{key}' suggests sensitive data",
                        "location": f"metadata.{key}",
                        "regulation": rule.regulation.value if rule.regulation else None,
                        "confidence": 0.7,
                    }
                    violations.append(violation)

            # Check metadata values
            if isinstance(value, str):
                value_detections = self.detection_engine.detect_in_text(value)
                for rule, matches in value_detections:
                    for match in matches:
                        violation = {
                            "type": "PII_IN_METADATA",
                            "rule_id": rule.rule_id,
                            "classification": rule.classification.value,
                            "severity": self._get_violation_severity(rule.classification),
                            "message": f"Found {rule.name} in metadata '{key}'",
                            "location": f"metadata.{key}",
                            "match": match[:30] + "..." if len(match) > 30 else match,
                            "regulation": rule.regulation.value if rule.regulation else None,
                            "confidence": 0.9,
                        }
                        violations.append(violation)

        return violations

    def _get_violation_severity(self, classification: DataClassification) -> str:
        """Get violation severity based on data classification"""
        severity_map = {
            DataClassification.PUBLIC: "LOW",
            DataClassification.INTERNAL: "LOW",
            DataClassification.PII: "HIGH",
            DataClassification.PHI: "HIGH",
            DataClassification.FINANCIAL: "HIGH",
            DataClassification.BIOMETRIC: "HIGH",
            DataClassification.SENSITIVE: "MEDIUM",
            DataClassification.CONFIDENTIAL: "HIGH",
        }
        return severity_map.get(classification, "MEDIUM")

    async def _generate_remediation_recommendations(self, violations: list[dict[str, Any]]) -> list[str]:
        """Generate recommendations for fixing violations"""
        recommendations = []

        violation_types = set(v["type"] for v in violations)

        if "PII_IN_PAYLOAD" in violation_types:
            recommendations.append("Remove or encrypt PII data in job payload before submission")
            recommendations.append("Consider using data anonymization techniques for training data")

        if "PII_IN_ENV_VAR" in violation_types:
            recommendations.append("Use secure secret management instead of environment variables for sensitive data")
            recommendations.append("Move credentials to a secure vault system")

        if "SENSITIVE_ENV_VAR_NAME" in violation_types:
            recommendations.append("Rename environment variables to avoid indicating sensitive data types")

        if "PII_IN_COMMAND_ARG" in violation_types:
            recommendations.append("Pass sensitive arguments through secure input channels instead of command line")

        if "PII_IN_INPUT_DATA" in violation_types:
            recommendations.append("Sanitize input data to remove PII before processing")
            recommendations.append("Implement data masking or tokenization for sensitive fields")

        if "PII_IN_METADATA" in violation_types:
            recommendations.append("Review metadata and labels to ensure no sensitive information is exposed")

        # Add regulatory recommendations
        regulations = set(v.get("regulation") for v in violations if v.get("regulation"))
        if "gdpr" in regulations:
            recommendations.append("Ensure GDPR compliance: obtain consent for processing personal data")
        if "hipaa" in regulations:
            recommendations.append("Ensure HIPAA compliance: implement appropriate safeguards for PHI")

        return recommendations

    async def validate_job_compliance(
        self,
        job_id: str,
        job_inputs: dict[str, Any],
        namespace: str = "default",
        user_id: str | None = None,
        strict_mode: bool = True,
    ) -> dict[str, Any]:
        """
        Validate job compliance and determine if job should be allowed

        Returns:
            dict: {"allowed": bool, "violations": list, "reason": str}
        """

        scan_result = await self.scan_job_inputs_for_compliance(job_id, job_inputs, namespace, user_id)

        violations = scan_result["violations"]
        high_severity = [v for v in violations if v["severity"] == "HIGH"]

        if strict_mode and high_severity:
            return {
                "allowed": False,
                "violations": violations,
                "reason": f"Job blocked due to {len(high_severity)} high-severity PII/PHI violations",
                "scan_result": scan_result,
            }
        elif violations:
            return {
                "allowed": True,
                "violations": violations,
                "reason": f"Job allowed with {len(violations)} compliance warnings",
                "scan_result": scan_result,
            }
        else:
            return {
                "allowed": True,
                "violations": [],
                "reason": "No compliance violations detected",
                "scan_result": scan_result,
            }

    async def get_compliance_summary(self) -> dict[str, Any]:
        """Get comprehensive compliance summary."""
        summary = {
            "total_locations": len(self.data_locations),
            "by_classification": {},
            "by_regulation": {},
            "by_tenant": {},
            "compliance_status": {"compliant": 0, "violations": 0, "unaudited": 0},
            "retention_jobs": {"total": len(self.retention_jobs), "enabled": 0, "pending_approval": 0},
        }

        # Analyze data locations
        for location in self.data_locations.values():
            # By classification
            cls = location.classification.value
            summary["by_classification"][cls] = summary["by_classification"].get(cls, 0) + 1

            # By regulation
            for reg in location.regulations:
                reg_name = reg.value
                summary["by_regulation"][reg_name] = summary["by_regulation"].get(reg_name, 0) + 1

            # By tenant
            if location.tenant_id:
                summary["by_tenant"][location.tenant_id] = summary["by_tenant"].get(location.tenant_id, 0) + 1

            # Compliance status
            if location.compliant:
                summary["compliance_status"]["compliant"] += 1
            else:
                summary["compliance_status"]["violations"] += 1

            if not location.last_audit:
                summary["compliance_status"]["unaudited"] += 1

        # Analyze retention jobs
        for job in self.retention_jobs.values():
            if job.enabled:
                summary["retention_jobs"]["enabled"] += 1
            if job.approval_status == "pending":
                summary["retention_jobs"]["pending_approval"] += 1

        return summary

    async def get_locations_by_classification(self, classification: DataClassification) -> list[DataLocation]:
        """Get all locations with specific classification."""
        return [loc for loc in self.data_locations.values() if loc.classification == classification]

    async def get_locations_by_tenant(self, tenant_id: str) -> list[DataLocation]:
        """Get all locations for specific tenant."""
        return [loc for loc in self.data_locations.values() if loc.tenant_id == tenant_id]

    async def get_retention_jobs_needing_approval(self) -> list[RetentionJob]:
        """Get all retention jobs needing approval."""
        return [
            job for job in self.retention_jobs.values() if job.requires_approval and job.approval_status == "pending"
        ]


if __name__ == "__main__":
    # Example usage
    async def main():
        # Initialize PII/PHI manager
        pii_manager = PIIPHIManager()

        # Run comprehensive scan
        print("Running comprehensive PII/PHI scan...")
        discoveries = await pii_manager.scan_all_known_locations()

        for category, locations in discoveries.items():
            print(f"\n{category.upper()}: {len(locations)} locations")
            for loc in locations[:3]:  # Show first 3
                print(f"  - {loc.path} ({loc.classification.value}, confidence: {loc.confidence_score:.2f})")

        # Get compliance summary
        summary = await pii_manager.get_compliance_summary()
        print("\nCompliance Summary:")
        print(f"Total locations: {summary['total_locations']}")
        print(f"By classification: {summary['by_classification']}")
        print(f"Compliance status: {summary['compliance_status']}")

        # Create sample retention job
        if pii_manager.data_locations:
            pii_locations = [
                loc.location_id
                for loc in pii_manager.data_locations.values()
                if loc.classification == DataClassification.PII
            ]

            if pii_locations:
                job_id = await pii_manager.create_retention_job(
                    name="PII Cleanup",
                    description="Remove PII data after standard retention period",
                    location_ids=pii_locations[:5],
                    retention_policy=RetentionPolicy.STANDARD,
                )
                print(f"\nCreated retention job: {job_id}")

    asyncio.run(main())
