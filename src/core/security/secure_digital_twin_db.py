#!/usr/bin/env python3
"""Secure Digital Twin Database Integration.

Provides encrypted database operations for Digital Twin system with full
CODEX compliance including GDPR, COPPA, and FERPA requirements.
"""

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .digital_twin_encryption import ComplianceViolationError, DigitalTwinEncryption

logger = logging.getLogger(__name__)


class SecureDigitalTwinDB:
    """Secure database operations for Digital Twin with encryption and compliance."""

    def __init__(self, db_path: str | None = None) -> None:
        """Initialize secure Digital Twin database.

        Args:
            db_path: Path to Digital Twin SQLite database
        """
        self.db_path = db_path or os.getenv(
            "DIGITAL_TWIN_DB_PATH", "./data/digital_twin.db"
        )
        self.enable_wal = os.getenv("DIGITAL_TWIN_SQLITE_WAL", "true").lower() == "true"

        # Initialize encryption system
        self.encryption = DigitalTwinEncryption()

        # Ensure database directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize database schema
        self.init_database()

        logger.info(f"Secure Digital Twin database initialized: {self.db_path}")

    @contextmanager
    def get_connection(self):
        """Get database connection with WAL mode and security settings."""
        conn = sqlite3.connect(
            self.db_path,
            timeout=30.0,
            isolation_level=None,  # Autocommit mode
        )

        try:
            # Configure SQLite for security and performance
            if self.enable_wal:
                conn.execute("PRAGMA journal_mode=WAL")

            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA mmap_size=268435456")  # 256MB

            # Enable foreign key constraints
            conn.execute("PRAGMA foreign_keys=ON")

            conn.row_factory = sqlite3.Row
            yield conn

        finally:
            conn.close()

    def init_database(self) -> None:
        """Initialize database schema with encryption and compliance features."""
        with self.get_connection() as conn:
            # Learning profiles with encrypted sensitive data
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS learning_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    profile_id TEXT UNIQUE NOT NULL,
                    user_id_hash TEXT NOT NULL,

                    -- Non-sensitive profile data (plaintext)
                    preferred_difficulty TEXT DEFAULT 'medium',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                    -- Encrypted sensitive data (BLOB fields)
                    learning_style_encrypted BLOB,
                    knowledge_domains_encrypted BLOB,
                    learning_goals_encrypted BLOB,
                    privacy_settings_encrypted BLOB,

                    -- Compliance and retention
                    ttl_expires_at TIMESTAMP,
                    compliance_flags TEXT,  -- JSON with COPPA/FERPA/GDPR flags
                    audit_log_enabled BOOLEAN DEFAULT 1,

                    -- Retention policy compliance
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 0,

                    CONSTRAINT chk_profile_id CHECK (length(profile_id) > 0),
                    CONSTRAINT chk_user_hash CHECK (length(user_id_hash) = 64)
                )
            """
            )

            # Learning sessions with privacy protection
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS learning_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    profile_id TEXT NOT NULL,
                    session_type TEXT NOT NULL,

                    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    end_time TIMESTAMP,
                    duration_minutes REAL,

                    -- Encrypted session data
                    topics_covered_encrypted BLOB,
                    performance_metrics_encrypted BLOB,
                    engagement_score_encrypted BLOB,

                    completion_status TEXT DEFAULT 'in_progress',

                    -- Privacy and compliance
                    privacy_level TEXT DEFAULT 'standard',
                    requires_parental_consent BOOLEAN DEFAULT 0,

                    FOREIGN KEY (profile_id) REFERENCES learning_profiles (profile_id)
                        ON DELETE CASCADE ON UPDATE CASCADE,

                    CONSTRAINT chk_session_type CHECK (
                        session_type IN ('lesson', 'assessment', 'practice', 'review')
                    ),
                    CONSTRAINT chk_completion CHECK (
                        completion_status IN ('in_progress', 'completed', 'abandoned', 'expired')
                    )
                )
            """
            )

            # Knowledge states with domain protection
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    profile_id TEXT NOT NULL,
                    knowledge_domain TEXT NOT NULL,
                    topic TEXT NOT NULL,

                    mastery_level REAL DEFAULT 0.0,
                    confidence_score REAL DEFAULT 0.0,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                    -- Encrypted learning trajectory
                    learning_trajectory_encrypted BLOB,

                    prerequisites_met BOOLEAN DEFAULT FALSE,

                    FOREIGN KEY (profile_id) REFERENCES learning_profiles (profile_id)
                        ON DELETE CASCADE ON UPDATE CASCADE,

                    UNIQUE(profile_id, knowledge_domain, topic),

                    CONSTRAINT chk_mastery CHECK (mastery_level >= 0.0 AND mastery_level <= 1.0),
                    CONSTRAINT chk_confidence CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0)
                )
            """
            )

            # Compliance audit log
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS compliance_audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                    action TEXT NOT NULL,
                    profile_id TEXT,
                    user_id_hash TEXT,
                    field_name TEXT,

                    compliance_type TEXT,  -- COPPA, FERPA, GDPR
                    compliance_status TEXT,

                    ip_address TEXT,
                    user_agent TEXT,
                    session_id TEXT,

                    additional_data TEXT,  -- JSON for extra context

                    CONSTRAINT chk_action CHECK (
                        action IN ('create', 'read', 'update', 'delete', 'encrypt', 'decrypt',
                                  'consent_granted', 'consent_revoked', 'data_exported', 'data_deleted')
                    ),
                    CONSTRAINT chk_compliance_type CHECK (
                        compliance_type IN ('COPPA', 'FERPA', 'GDPR', 'GENERAL')
                    )
                )
            """
            )

            # Data retention tracking
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS data_retention_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    profile_id TEXT NOT NULL,

                    created_at TIMESTAMP NOT NULL,
                    scheduled_deletion_at TIMESTAMP NOT NULL,
                    retention_policy TEXT NOT NULL,

                    last_review_at TIMESTAMP,
                    review_status TEXT DEFAULT 'pending',

                    deletion_requested BOOLEAN DEFAULT 0,
                    deletion_requested_at TIMESTAMP,
                    deletion_reason TEXT,

                    FOREIGN KEY (profile_id) REFERENCES learning_profiles (profile_id)
                        ON DELETE CASCADE ON UPDATE CASCADE,

                    CONSTRAINT chk_review_status CHECK (
                        review_status IN ('pending', 'reviewed', 'approved', 'rejected')
                    )
                )
            """
            )

            # Create indexes for performance
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_profiles_user_hash ON learning_profiles (user_id_hash)",
                "CREATE INDEX IF NOT EXISTS idx_profiles_expires ON learning_profiles (ttl_expires_at)",
                "CREATE INDEX IF NOT EXISTS idx_profiles_updated ON learning_profiles (updated_at)",
                "CREATE INDEX IF NOT EXISTS idx_sessions_profile ON learning_sessions (profile_id)",
                "CREATE INDEX IF NOT EXISTS idx_sessions_start ON learning_sessions (start_time)",
                "CREATE INDEX IF NOT EXISTS idx_knowledge_profile ON knowledge_states (profile_id)",
                "CREATE INDEX IF NOT EXISTS idx_knowledge_domain ON knowledge_states (knowledge_domain)",
                "CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON compliance_audit_log (timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_audit_profile ON compliance_audit_log (profile_id)",
                "CREATE INDEX IF NOT EXISTS idx_retention_scheduled ON data_retention_tracking (scheduled_deletion_at)",
            ]

            for index_sql in indexes:
                conn.execute(index_sql)

            conn.execute("COMMIT")
            logger.info("Database schema initialized successfully")

    def create_learning_profile(self, profile_data: dict[str, Any]) -> str:
        """Create new learning profile with encryption.

        Args:
            profile_data: Profile data including sensitive fields

        Returns:
            Created profile_id
        """
        with self.get_connection() as conn:
            # Generate profile ID and hash user ID
            profile_id = (
                profile_data.get("profile_id") or f"profile_{os.urandom(8).hex()}"
            )
            user_id = profile_data.get("user_id", "")
            user_id_hash = self.encryption.hash_user_id(user_id)

            # Calculate TTL expiration
            ttl_expires_at = datetime.utcnow() + timedelta(
                days=self.encryption.profile_ttl_days
            )

            # Prepare compliance flags
            compliance_flags = json.dumps(
                {
                    "coppa_compliant": self.encryption.coppa_compliant,
                    "ferpa_compliant": self.encryption.ferpa_compliant,
                    "gdpr_compliant": self.encryption.gdpr_compliant,
                    "requires_parental_consent": profile_data.get("age", 18) < 13,
                    "educational_record": True,
                }
            )

            # Encrypt sensitive fields
            encrypted_data = {}
            sensitive_fields = [
                "learning_style",
                "knowledge_domains",
                "learning_goals",
                "privacy_settings",
            ]

            for field in sensitive_fields:
                if field in profile_data:
                    encrypted_data[
                        f"{field}_encrypted"
                    ] = self.encryption.encrypt_sensitive_field(
                        profile_data[field], field
                    )

            # Insert profile
            conn.execute(
                """
                INSERT INTO learning_profiles (
                    profile_id, user_id_hash, preferred_difficulty,
                    learning_style_encrypted, knowledge_domains_encrypted,
                    learning_goals_encrypted, privacy_settings_encrypted,
                    ttl_expires_at, compliance_flags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    profile_id,
                    user_id_hash,
                    profile_data.get("preferred_difficulty", "medium"),
                    encrypted_data.get("learning_style_encrypted"),
                    encrypted_data.get("knowledge_domains_encrypted"),
                    encrypted_data.get("learning_goals_encrypted"),
                    encrypted_data.get("privacy_settings_encrypted"),
                    ttl_expires_at,
                    compliance_flags,
                ),
            )

            # Create retention tracking record
            conn.execute(
                """
                INSERT INTO data_retention_tracking (
                    profile_id, created_at, scheduled_deletion_at, retention_policy
                ) VALUES (?, ?, ?, ?)
            """,
                (
                    profile_id,
                    datetime.utcnow(),
                    ttl_expires_at,
                    f"TTL_{self.encryption.profile_ttl_days}_DAYS",
                ),
            )

            # Log compliance audit
            self._log_compliance_audit(
                conn,
                "create",
                profile_id,
                user_id_hash,
                compliance_type="GENERAL",
                additional_data=json.dumps(
                    {
                        "profile_created": True,
                        "encryption_enabled": True,
                        "compliance_flags": json.loads(compliance_flags),
                    }
                ),
            )

            conn.execute("COMMIT")
            logger.info(f"Created encrypted learning profile: {profile_id}")
            return profile_id

    def get_learning_profile(
        self, profile_id: str, decrypt: bool = True
    ) -> dict[str, Any] | None:
        """Retrieve learning profile with optional decryption.

        Args:
            profile_id: Profile identifier
            decrypt: Whether to decrypt sensitive fields

        Returns:
            Profile data or None if not found
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM learning_profiles WHERE profile_id = ?
            """,
                (profile_id,),
            )

            row = cursor.fetchone()
            if not row:
                return None

            profile = dict(row)

            # Update access tracking
            conn.execute(
                """
                UPDATE learning_profiles
                SET last_accessed = CURRENT_TIMESTAMP, access_count = access_count + 1
                WHERE profile_id = ?
            """,
                (profile_id,),
            )

            # Check compliance and retention
            compliance_status = self._check_compliance_access(profile)
            if not compliance_status["access_allowed"]:
                raise ComplianceViolationError(compliance_status["reason"])

            if decrypt:
                # Decrypt sensitive fields
                encrypted_fields = [
                    "learning_style_encrypted",
                    "knowledge_domains_encrypted",
                    "learning_goals_encrypted",
                    "privacy_settings_encrypted",
                ]

                for encrypted_field in encrypted_fields:
                    if profile[encrypted_field]:
                        original_field = encrypted_field.replace("_encrypted", "")
                        try:
                            decrypted_value = self.encryption.decrypt_sensitive_field(
                                profile[encrypted_field], original_field
                            )
                            profile[original_field] = decrypted_value
                        except Exception as e:
                            logger.exception(f"Failed to decrypt {original_field}: {e}")
                            profile[original_field] = None

                        # Remove encrypted version from response
                        del profile[encrypted_field]

            # Log access
            self._log_compliance_audit(
                conn,
                "read",
                profile_id,
                profile["user_id_hash"],
                field_name="profile",
                compliance_type="GENERAL",
            )

            conn.execute("COMMIT")
            return profile

    def update_learning_profile(self, profile_id: str, updates: dict[str, Any]) -> bool:
        """Update learning profile with encryption.

        Args:
            profile_id: Profile to update
            updates: Fields to update

        Returns:
            True if successful
        """
        with self.get_connection() as conn:
            # Check if profile exists and compliance
            existing = self.get_learning_profile(profile_id, decrypt=False)
            if not existing:
                return False

            # Prepare update fields
            set_clauses = []
            values = []

            # Handle non-encrypted fields
            simple_fields = ["preferred_difficulty"]
            for field in simple_fields:
                if field in updates:
                    set_clauses.append(f"{field} = ?")
                    values.append(updates[field])

            # Handle encrypted fields
            sensitive_fields = [
                "learning_style",
                "knowledge_domains",
                "learning_goals",
                "privacy_settings",
            ]
            for field in sensitive_fields:
                if field in updates:
                    encrypted_field = f"{field}_encrypted"
                    encrypted_data = self.encryption.encrypt_sensitive_field(
                        updates[field], field
                    )
                    set_clauses.append(f"{encrypted_field} = ?")
                    values.append(encrypted_data)

            if not set_clauses:
                return True  # Nothing to update

            # Add updated timestamp
            set_clauses.append("updated_at = CURRENT_TIMESTAMP")
            values.append(profile_id)

            # Execute update
            conn.execute(
                f"""
                UPDATE learning_profiles
                SET {", ".join(set_clauses)}
                WHERE profile_id = ?
            """,
                values,
            )

            # Log compliance audit
            self._log_compliance_audit(
                conn,
                "update",
                profile_id,
                existing["user_id_hash"],
                compliance_type="GENERAL",
                additional_data=json.dumps(
                    {
                        "fields_updated": list(updates.keys()),
                        "encrypted_fields": [
                            f for f in updates if f in sensitive_fields
                        ],
                    }
                ),
            )

            conn.execute("COMMIT")
            logger.info(f"Updated encrypted learning profile: {profile_id}")
            return True

    def delete_learning_profile(
        self, profile_id: str, reason: str = "user_request"
    ) -> bool:
        """Delete learning profile with GDPR compliance.

        Args:
            profile_id: Profile to delete
            reason: Reason for deletion

        Returns:
            True if successful
        """
        with self.get_connection() as conn:
            # Get profile for audit logging
            cursor = conn.execute(
                """
                SELECT user_id_hash FROM learning_profiles WHERE profile_id = ?
            """,
                (profile_id,),
            )
            row = cursor.fetchone()
            if not row:
                return False

            user_id_hash = row[0]

            # Delete profile (cascades to sessions and knowledge states)
            conn.execute(
                "DELETE FROM learning_profiles WHERE profile_id = ?", (profile_id,)
            )

            # Update retention tracking
            conn.execute(
                """
                DELETE FROM data_retention_tracking WHERE profile_id = ?
            """,
                (profile_id,),
            )

            # Log compliance audit
            self._log_compliance_audit(
                conn,
                "delete",
                profile_id,
                user_id_hash,
                compliance_type="GDPR",
                additional_data=json.dumps(
                    {
                        "deletion_reason": reason,
                        "gdpr_right_to_erasure": True,
                        "cascade_deletion": True,
                    }
                ),
            )

            conn.execute("COMMIT")
            logger.info(f"Deleted learning profile with GDPR compliance: {profile_id}")
            return True

    def _check_compliance_access(self, profile: dict[str, Any]) -> dict[str, Any]:
        """Check if profile access complies with regulations.

        Args:
            profile: Profile data

        Returns:
            Compliance status and details
        """
        compliance_flags = json.loads(profile.get("compliance_flags", "{}"))

        # Check TTL expiration
        ttl_expires_at = profile.get("ttl_expires_at")
        if ttl_expires_at:
            expires_dt = datetime.fromisoformat(ttl_expires_at.replace("Z", "+00:00"))
            if datetime.utcnow() > expires_dt:
                return {
                    "access_allowed": False,
                    "reason": "Profile has expired per data retention policy",
                    "requires_action": "DELETE_EXPIRED_DATA",
                }

        # Check COPPA compliance for minors
        if (
            compliance_flags.get("requires_parental_consent")
            and not self.encryption.coppa_compliant
        ):
            return {
                "access_allowed": False,
                "reason": "COPPA compliance required for minor profiles",
                "requires_action": "ENABLE_COPPA_COMPLIANCE",
            }

        # Check FERPA compliance for educational records
        if (
            compliance_flags.get("educational_record")
            and not self.encryption.ferpa_compliant
        ):
            return {
                "access_allowed": False,
                "reason": "FERPA compliance required for educational records",
                "requires_action": "ENABLE_FERPA_COMPLIANCE",
            }

        return {"access_allowed": True, "reason": "Compliance checks passed"}

    def _log_compliance_audit(
        self,
        conn,
        action: str,
        profile_id: str,
        user_id_hash: str,
        field_name: str | None = None,
        compliance_type: str = "GENERAL",
        additional_data: str | None = None,
    ) -> None:
        """Log compliance audit entry.

        Args:
            conn: Database connection
            action: Action performed
            profile_id: Profile ID
            user_id_hash: Hashed user ID
            field_name: Field accessed (optional)
            compliance_type: Type of compliance
            additional_data: Extra context as JSON
        """
        conn.execute(
            """
            INSERT INTO compliance_audit_log (
                action, profile_id, user_id_hash, field_name,
                compliance_type, compliance_status, session_id, additional_data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                action,
                profile_id,
                user_id_hash,
                field_name,
                compliance_type,
                "compliant",
                os.urandom(8).hex(),
                additional_data,
            ),
        )

    def get_expired_profiles(self) -> list[dict[str, Any]]:
        """Get profiles that have exceeded retention policy.

        Returns:
            List of expired profiles requiring deletion
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT profile_id, user_id_hash, ttl_expires_at, compliance_flags
                FROM learning_profiles
                WHERE ttl_expires_at < CURRENT_TIMESTAMP
                ORDER BY ttl_expires_at ASC
            """
            )

            expired = []
            for row in cursor:
                expired.append(
                    {
                        "profile_id": row[0],
                        "user_id_hash": row[1],
                        "expired_at": row[2],
                        "compliance_flags": json.loads(row[3] or "{}"),
                        "requires_gdpr_deletion": self.encryption.gdpr_compliant,
                    }
                )

            return expired

    def cleanup_expired_data(self) -> int:
        """Clean up expired profiles per retention policy.

        Returns:
            Number of profiles deleted
        """
        expired_profiles = self.get_expired_profiles()
        deleted_count = 0

        for profile in expired_profiles:
            success = self.delete_learning_profile(
                profile["profile_id"], reason="retention_policy_expiry"
            )
            if success:
                deleted_count += 1

        logger.info(f"Cleaned up {deleted_count} expired profiles")
        return deleted_count

    def export_user_data(self, user_id_hash: str) -> dict[str, Any]:
        """Export all user data for GDPR compliance.

        Args:
            user_id_hash: Hashed user identifier

        Returns:
            Complete user data export
        """
        with self.get_connection() as conn:
            # Get all profiles for user
            cursor = conn.execute(
                """
                SELECT * FROM learning_profiles WHERE user_id_hash = ?
            """,
                (user_id_hash,),
            )

            profiles = []
            for row in cursor:
                profile = self.get_learning_profile(row["profile_id"], decrypt=True)
                if profile:
                    profiles.append(profile)

            # Get all sessions
            sessions = []
            for profile in profiles:
                session_cursor = conn.execute(
                    """
                    SELECT * FROM learning_sessions WHERE profile_id = ?
                """,
                    (profile["profile_id"],),
                )

                for session_row in session_cursor:
                    sessions.append(dict(session_row))

            # Get audit log
            audit_cursor = conn.execute(
                """
                SELECT * FROM compliance_audit_log
                WHERE user_id_hash = ?
                ORDER BY timestamp DESC
            """,
                (user_id_hash,),
            )

            audit_log = [dict(row) for row in audit_cursor]

            export_data = {
                "export_timestamp": datetime.utcnow().isoformat(),
                "user_id_hash": user_id_hash,
                "profiles": profiles,
                "learning_sessions": sessions,
                "audit_log": audit_log,
                "compliance_status": {
                    "gdpr_compliant": self.encryption.gdpr_compliant,
                    "coppa_compliant": self.encryption.coppa_compliant,
                    "ferpa_compliant": self.encryption.ferpa_compliant,
                },
            }

            # Log the export
            self._log_compliance_audit(
                conn,
                "data_exported",
                None,
                user_id_hash,
                compliance_type="GDPR",
                additional_data=json.dumps(
                    {
                        "export_generated": True,
                        "profiles_count": len(profiles),
                        "sessions_count": len(sessions),
                        "audit_entries": len(audit_log),
                    }
                ),
            )

            conn.execute("COMMIT")
            return export_data

    def get_compliance_stats(self) -> dict[str, Any]:
        """Get compliance and security statistics.

        Returns:
            Security and compliance metrics
        """
        with self.get_connection() as conn:
            stats = {}

            # Profile statistics
            cursor = conn.execute(
                """
                SELECT
                    COUNT(*) as total_profiles,
                    COUNT(CASE WHEN ttl_expires_at > CURRENT_TIMESTAMP THEN 1 END) as active_profiles,
                    COUNT(CASE WHEN ttl_expires_at <= CURRENT_TIMESTAMP THEN 1 END) as expired_profiles,
                    AVG(access_count) as avg_access_count
                FROM learning_profiles
            """
            )
            profile_stats = dict(cursor.fetchone())
            stats["profiles"] = profile_stats

            # Compliance audit statistics
            cursor = conn.execute(
                """
                SELECT
                    compliance_type,
                    COUNT(*) as audit_entries,
                    MAX(timestamp) as last_audit_entry
                FROM compliance_audit_log
                GROUP BY compliance_type
            """
            )
            audit_stats = {
                row[0]: {"count": row[1], "last_entry": row[2]} for row in cursor
            }
            stats["audit"] = audit_stats

            # Encryption status
            stats["encryption"] = {
                "encryption_enabled": True,
                "compliance_flags": {
                    "coppa_compliant": self.encryption.coppa_compliant,
                    "ferpa_compliant": self.encryption.ferpa_compliant,
                    "gdpr_compliant": self.encryption.gdpr_compliant,
                },
                "retention_policy_days": self.encryption.profile_ttl_days,
            }

            # Database health
            cursor = conn.execute("PRAGMA integrity_check")
            integrity_result = cursor.fetchone()[0]
            stats["database_health"] = {
                "integrity_check": integrity_result,
                "wal_enabled": self.enable_wal,
                "foreign_keys_enabled": True,
            }

            return stats


# Example usage
if __name__ == "__main__":
    import tempfile

    # Set test environment
    os.environ[
        "DIGITAL_TWIN_ENCRYPTION_KEY"
    ] = "dGVzdF9rZXlfMzJfYnl0ZXNfZm9yX2VuY3J5cHRpb25fdGVzdGluZ19wdXJwb3Nlcw=="
    os.environ["DIGITAL_TWIN_COPPA_COMPLIANT"] = "true"
    os.environ["DIGITAL_TWIN_FERPA_COMPLIANT"] = "true"
    os.environ["DIGITAL_TWIN_GDPR_COMPLIANT"] = "true"

    # Test with temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    os.environ["DIGITAL_TWIN_DB_PATH"] = db_path

    # Test secure database operations
    secure_db = SecureDigitalTwinDB()

    # Create test profile
    profile_data = {
        "user_id": "test_user_123",
        "learning_style": "visual",
        "knowledge_domains": ["mathematics", "science"],
        "learning_goals": ["improve_problem_solving", "master_algebra"],
        "preferred_difficulty": "intermediate",
        "privacy_settings": {"share_progress": False, "parent_notifications": True},
    }

    profile_id = secure_db.create_learning_profile(profile_data)
    print(f"Created profile: {profile_id}")

    # Retrieve and decrypt profile
    retrieved_profile = secure_db.get_learning_profile(profile_id)
    print(f"Retrieved profile: {retrieved_profile['learning_style']}")

    # Get compliance stats
    stats = secure_db.get_compliance_stats()
    print(f"Compliance stats: {stats}")

    # Cleanup
    os.unlink(db_path)
