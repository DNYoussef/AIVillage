"""Authentication & Access Control System - Prompt H

Comprehensive authentication and authorization framework including:
- Multi-factor authentication (MFA)
- Role-based access control (RBAC)
- Session management and token validation
- API key management
- Audit logging and security monitoring
- Zero-trust security principles

Integration Point: Security layer for Phase 4 testing
"""

import base64
import hashlib
import hmac
import json
import logging
import secrets
import sqlite3
import sys
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class AuthenticationMethod(Enum):
    """Authentication methods."""

    PASSWORD = "password"
    API_KEY = "api_key"
    TOKEN = "token"
    CERTIFICATE = "certificate"
    BIOMETRIC = "biometric"
    OTP = "otp"
    SOCIAL = "social"


class UserRole(Enum):
    """User roles for RBAC."""

    ADMIN = "admin"
    OPERATOR = "operator"
    DEVELOPER = "developer"
    ANALYST = "analyst"
    VIEWER = "viewer"
    GUEST = "guest"


class Permission(Enum):
    """System permissions."""

    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    ADMIN = "admin"
    CONFIG = "config"
    MONITOR = "monitor"
    AUDIT = "audit"


class SecurityLevel(Enum):
    """Security clearance levels."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"


class SessionStatus(Enum):
    """Session status."""

    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    SUSPENDED = "suspended"


@dataclass
class User:
    """User account information."""

    user_id: str
    username: str
    email: str
    password_hash: str
    role: UserRole
    security_level: SecurityLevel
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_login: datetime | None = None
    failed_login_attempts: int = 0
    locked_until: datetime | None = None
    mfa_enabled: bool = False
    mfa_secret: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ApiKey:
    """API key information."""

    key_id: str
    key_hash: str
    user_id: str
    name: str
    permissions: list[Permission]
    created_at: datetime
    expires_at: datetime | None = None
    last_used: datetime | None = None
    usage_count: int = 0
    enabled: bool = True
    ip_whitelist: list[str] = field(default_factory=list)
    rate_limit: int | None = None


@dataclass
class Session:
    """User session information."""

    session_id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    last_activity: datetime
    ip_address: str
    user_agent: str
    status: SessionStatus = SessionStatus.ACTIVE
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditLog:
    """Security audit log entry."""

    log_id: str
    timestamp: datetime
    user_id: str | None
    action: str
    resource: str
    ip_address: str
    user_agent: str
    success: bool
    details: dict[str, Any] = field(default_factory=dict)
    risk_score: float = 0.0


@dataclass
class AuthConfig:
    """Authentication system configuration."""

    password_min_length: int = 12
    password_require_uppercase: bool = True
    password_require_lowercase: bool = True
    password_require_numbers: bool = True
    password_require_symbols: bool = True
    max_failed_attempts: int = 5
    lockout_duration_minutes: int = 30
    session_timeout_minutes: int = 120
    token_expiry_hours: int = 24
    require_mfa: bool = False
    api_key_expiry_days: int = 365
    password_history_count: int = 12
    enable_audit_logging: bool = True
    rate_limit_requests_per_minute: int = 100


class PasswordManager:
    """Secure password hashing and validation."""

    def __init__(self):
        """Initialize password manager."""
        self.salt_length = 32
        self.iterations = 100000

    def hash_password(self, password: str) -> str:
        """Hash password securely using PBKDF2.

        Args:
            password: Plain text password

        Returns:
            Hashed password with salt
        """
        salt = secrets.token_bytes(self.salt_length)
        hash_bytes = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, self.iterations
        )

        # Store salt + hash + iterations
        return base64.b64encode(
            salt + hash_bytes + self.iterations.to_bytes(4, "big")
        ).decode("ascii")

    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash.

        Args:
            password: Plain text password
            password_hash: Stored password hash

        Returns:
            True if password matches
        """
        try:
            decoded = base64.b64decode(password_hash.encode("ascii"))
            salt = decoded[: self.salt_length]
            stored_hash = decoded[self.salt_length : -4]
            iterations = int.from_bytes(decoded[-4:], "big")

            new_hash = hashlib.pbkdf2_hmac(
                "sha256", password.encode("utf-8"), salt, iterations
            )

            return hmac.compare_digest(stored_hash, new_hash)

        except Exception:
            return False

    def validate_password_strength(
        self, password: str, config: AuthConfig
    ) -> tuple[bool, list[str]]:
        """Validate password strength.

        Args:
            password: Password to validate
            config: Authentication configuration

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        if len(password) < config.password_min_length:
            errors.append(
                f"Password must be at least {config.password_min_length} characters"
            )

        if config.password_require_uppercase and not any(c.isupper() for c in password):
            errors.append("Password must contain uppercase letters")

        if config.password_require_lowercase and not any(c.islower() for c in password):
            errors.append("Password must contain lowercase letters")

        if config.password_require_numbers and not any(c.isdigit() for c in password):
            errors.append("Password must contain numbers")

        if config.password_require_symbols and not any(
            c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password
        ):
            errors.append("Password must contain special characters")

        return len(errors) == 0, errors


class TokenManager:
    """JWT-like token management without external dependencies."""

    def __init__(self, secret_key: str | None = None):
        """Initialize token manager.

        Args:
            secret_key: Secret key for signing tokens
        """
        self.secret_key = secret_key or secrets.token_urlsafe(32)

    def create_token(
        self, user_id: str, permissions: list[str], expires_in_hours: int = 24
    ) -> str:
        """Create authentication token.

        Args:
            user_id: User identifier
            permissions: User permissions
            expires_in_hours: Token expiry in hours

        Returns:
            Signed token
        """
        payload = {
            "user_id": user_id,
            "permissions": permissions,
            "issued_at": int(time.time()),
            "expires_at": int(time.time() + expires_in_hours * 3600),
            "token_id": secrets.token_urlsafe(16),
        }

        # Simple token format: base64(payload).signature
        payload_encoded = base64.b64encode(json.dumps(payload).encode()).decode()
        signature = self._sign_payload(payload_encoded)

        return f"{payload_encoded}.{signature}"

    def verify_token(self, token: str) -> tuple[bool, dict[str, Any] | None]:
        """Verify and decode token.

        Args:
            token: Token to verify

        Returns:
            Tuple of (is_valid, payload)
        """
        try:
            payload_encoded, signature = token.split(".")

            # Verify signature
            expected_signature = self._sign_payload(payload_encoded)
            if not hmac.compare_digest(signature, expected_signature):
                return False, None

            # Decode payload
            payload = json.loads(base64.b64decode(payload_encoded).decode())

            # Check expiry
            if payload.get("expires_at", 0) < time.time():
                return False, None

            return True, payload

        except Exception:
            return False, None

    def _sign_payload(self, payload: str) -> str:
        """Sign payload with HMAC.

        Args:
            payload: Payload to sign

        Returns:
            Signature
        """
        signature = hmac.new(
            self.secret_key.encode(), payload.encode(), hashlib.sha256
        ).hexdigest()

        return signature


class MFAManager:
    """Multi-factor authentication manager."""

    def __init__(self):
        """Initialize MFA manager."""
        self.otp_window = 30  # seconds
        self.otp_digits = 6

    def generate_secret(self) -> str:
        """Generate MFA secret key.

        Returns:
            Base32-encoded secret
        """
        return base64.b32encode(secrets.token_bytes(20)).decode()

    def generate_otp(self, secret: str, timestamp: int | None = None) -> str:
        """Generate OTP code.

        Args:
            secret: MFA secret
            timestamp: Unix timestamp (current time if None)

        Returns:
            OTP code
        """
        if timestamp is None:
            timestamp = int(time.time())

        # Time-based counter
        counter = timestamp // self.otp_window

        # HMAC-based OTP (simplified)
        # Add padding to make length multiple of 8
        padding_needed = (8 - len(secret) % 8) % 8
        padded_secret = secret + "=" * padding_needed
        key = base64.b32decode(padded_secret)
        counter_bytes = counter.to_bytes(8, "big")

        hmac_hash = hmac.new(key, counter_bytes, hashlib.sha1).digest()
        offset = hmac_hash[-1] & 0x0F
        truncated = int.from_bytes(hmac_hash[offset : offset + 4], "big") & 0x7FFFFFFF

        otp = truncated % (10**self.otp_digits)
        return f"{otp:0{self.otp_digits}d}"

    def verify_otp(self, secret: str, code: str, timestamp: int | None = None) -> bool:
        """Verify OTP code.

        Args:
            secret: MFA secret
            code: OTP code to verify
            timestamp: Unix timestamp (current time if None)

        Returns:
            True if code is valid
        """
        if timestamp is None:
            timestamp = int(time.time())

        # Check current window and adjacent windows for clock skew
        for offset in [-1, 0, 1]:
            test_timestamp = timestamp + (offset * self.otp_window)
            expected_code = self.generate_otp(secret, test_timestamp)
            if hmac.compare_digest(code, expected_code):
                return True

        return False


class AuthenticationManager:
    """Main authentication manager."""

    def __init__(self, config: AuthConfig | None = None, db_path: str = ":memory:"):
        """Initialize authentication manager.

        Args:
            config: Authentication configuration
            db_path: Database path for user storage
        """
        self.config = config or AuthConfig()
        self.db_path = db_path
        self.password_manager = PasswordManager()
        self.token_manager = TokenManager()
        self.mfa_manager = MFAManager()

        # Active sessions
        self.active_sessions: dict[str, Session] = {}
        self.session_lock = threading.Lock()

        # Initialize database
        self._init_database()

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def _init_database(self):
        """Initialize user database."""
        with self._get_db() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL,
                    security_level TEXT NOT NULL,
                    enabled BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    failed_login_attempts INTEGER DEFAULT 0,
                    locked_until TIMESTAMP,
                    mfa_enabled BOOLEAN DEFAULT 0,
                    mfa_secret TEXT,
                    metadata TEXT DEFAULT '{}'
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    key_id TEXT PRIMARY KEY,
                    key_hash TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    permissions TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    last_used TIMESTAMP,
                    usage_count INTEGER DEFAULT 0,
                    enabled BOOLEAN DEFAULT 1,
                    ip_whitelist TEXT DEFAULT '[]',
                    rate_limit INTEGER,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    log_id TEXT PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_id TEXT,
                    action TEXT NOT NULL,
                    resource TEXT NOT NULL,
                    ip_address TEXT NOT NULL,
                    user_agent TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    details TEXT DEFAULT '{}',
                    risk_score REAL DEFAULT 0.0
                )
            """)

    @contextmanager
    def _get_db(self):
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        role: UserRole = UserRole.VIEWER,
        security_level: SecurityLevel = SecurityLevel.INTERNAL,
    ) -> User:
        """Create new user account.

        Args:
            username: Username
            email: Email address
            password: Plain text password
            role: User role
            security_level: Security clearance level

        Returns:
            Created user
        """
        # Validate password
        is_valid, errors = self.password_manager.validate_password_strength(
            password, self.config
        )
        if not is_valid:
            raise ValueError(f"Password validation failed: {', '.join(errors)}")

        # Hash password
        password_hash = self.password_manager.hash_password(password)

        user = User(
            user_id=secrets.token_urlsafe(16),
            username=username,
            email=email,
            password_hash=password_hash,
            role=role,
            security_level=security_level,
        )

        # Store in database
        with self._get_db() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO users (
                        user_id, username, email, password_hash, role, security_level,
                        enabled, created_at, mfa_enabled, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        user.user_id,
                        user.username,
                        user.email,
                        user.password_hash,
                        user.role.value,
                        user.security_level.value,
                        user.enabled,
                        user.created_at,
                        user.mfa_enabled,
                        json.dumps(user.metadata),
                    ),
                )
                conn.commit()

                self._audit_log(
                    user_id=user.user_id,
                    action="user_created",
                    resource=f"user:{username}",
                    success=True,
                    details={
                        "role": role.value,
                        "security_level": security_level.value,
                    },
                )

            except sqlite3.IntegrityError as e:
                raise ValueError(f"User creation failed: {e}")

        return user

    def authenticate(
        self,
        username: str,
        password: str,
        ip_address: str = "unknown",
        user_agent: str = "unknown",
        mfa_code: str | None = None,
    ) -> tuple[bool, User | None, str | None]:
        """Authenticate user credentials.

        Args:
            username: Username or email
            password: Password
            ip_address: Client IP address
            user_agent: Client user agent
            mfa_code: MFA code if MFA is enabled

        Returns:
            Tuple of (success, user, session_token)
        """
        with self._get_db() as conn:
            # Find user
            cursor = conn.execute(
                """
                SELECT * FROM users
                WHERE username = ? OR email = ?
            """,
                (username, username),
            )

            user_row = cursor.fetchone()
            if not user_row:
                self._audit_log(
                    action="login_failed",
                    resource=f"user:{username}",
                    ip_address=ip_address,
                    user_agent=user_agent,
                    success=False,
                    details={"reason": "user_not_found"},
                )
                return False, None, None

            # Convert to User object
            user = self._row_to_user(user_row)

            # Check if account is locked
            if user.locked_until and user.locked_until > datetime.now():
                self._audit_log(
                    user_id=user.user_id,
                    action="login_failed",
                    resource=f"user:{username}",
                    ip_address=ip_address,
                    user_agent=user_agent,
                    success=False,
                    details={"reason": "account_locked"},
                )
                return False, user, None

            # Verify password
            if not self.password_manager.verify_password(password, user.password_hash):
                # Increment failed attempts
                failed_attempts = user.failed_login_attempts + 1
                locked_until = None

                if failed_attempts >= self.config.max_failed_attempts:
                    locked_until = datetime.now() + timedelta(
                        minutes=self.config.lockout_duration_minutes
                    )

                conn.execute(
                    """
                    UPDATE users
                    SET failed_login_attempts = ?, locked_until = ?
                    WHERE user_id = ?
                """,
                    (failed_attempts, locked_until, user.user_id),
                )
                conn.commit()

                self._audit_log(
                    user_id=user.user_id,
                    action="login_failed",
                    resource=f"user:{username}",
                    ip_address=ip_address,
                    user_agent=user_agent,
                    success=False,
                    details={
                        "reason": "invalid_password",
                        "failed_attempts": failed_attempts,
                    },
                )
                return False, user, None

            # Check MFA if enabled
            if user.mfa_enabled:
                if not mfa_code:
                    return False, user, "mfa_required"

                if not self.mfa_manager.verify_otp(user.mfa_secret, mfa_code):
                    self._audit_log(
                        user_id=user.user_id,
                        action="login_failed",
                        resource=f"user:{username}",
                        ip_address=ip_address,
                        user_agent=user_agent,
                        success=False,
                        details={"reason": "invalid_mfa"},
                    )
                    return False, user, None

            # Successful authentication
            # Reset failed attempts and update last login
            conn.execute(
                """
                UPDATE users
                SET failed_login_attempts = 0, locked_until = NULL, last_login = ?
                WHERE user_id = ?
            """,
                (datetime.now(), user.user_id),
            )
            conn.commit()

            # Create session
            session_token = self._create_session(user, ip_address, user_agent)

            self._audit_log(
                user_id=user.user_id,
                action="login_success",
                resource=f"user:{username}",
                ip_address=ip_address,
                user_agent=user_agent,
                success=True,
                details={"auth_method": "password"},
            )

            return True, user, session_token

    def authenticate_api_key(
        self, api_key: str, ip_address: str = "unknown"
    ) -> tuple[bool, User | None]:
        """Authenticate using API key.

        Args:
            api_key: API key
            ip_address: Client IP address

        Returns:
            Tuple of (success, user)
        """
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        with self._get_db() as conn:
            cursor = conn.execute(
                """
                SELECT ak.*, u.* FROM api_keys ak
                JOIN users u ON ak.user_id = u.user_id
                WHERE ak.key_hash = ? AND ak.enabled = 1 AND u.enabled = 1
            """,
                (key_hash,),
            )

            row = cursor.fetchone()
            if not row:
                self._audit_log(
                    action="api_auth_failed",
                    resource="api_key",
                    ip_address=ip_address,
                    user_agent="api",
                    success=False,
                    details={"reason": "invalid_key"},
                )
                return False, None

            # Check expiry
            if (
                row["expires_at"]
                and datetime.fromisoformat(row["expires_at"]) < datetime.now()
            ):
                self._audit_log(
                    action="api_auth_failed",
                    resource="api_key",
                    ip_address=ip_address,
                    user_agent="api",
                    success=False,
                    details={"reason": "expired_key"},
                )
                return False, None

            # Check IP whitelist
            ip_whitelist = json.loads(row["ip_whitelist"])
            if ip_whitelist and ip_address not in ip_whitelist:
                self._audit_log(
                    action="api_auth_failed",
                    resource="api_key",
                    ip_address=ip_address,
                    user_agent="api",
                    success=False,
                    details={"reason": "ip_not_whitelisted"},
                )
                return False, None

            # Update usage
            conn.execute(
                """
                UPDATE api_keys
                SET last_used = ?, usage_count = usage_count + 1
                WHERE key_id = ?
            """,
                (datetime.now(), row["key_id"]),
            )
            conn.commit()

            user = self._row_to_user(row)

            self._audit_log(
                user_id=user.user_id,
                action="api_auth_success",
                resource="api_key",
                ip_address=ip_address,
                user_agent="api",
                success=True,
                details={"key_id": row["key_id"]},
            )

            return True, user

    def _create_session(self, user: User, ip_address: str, user_agent: str) -> str:
        """Create user session.

        Args:
            user: User object
            ip_address: Client IP
            user_agent: Client user agent

        Returns:
            Session token
        """
        session_id = secrets.token_urlsafe(32)
        now = datetime.now()
        expires_at = now + timedelta(minutes=self.config.session_timeout_minutes)

        session = Session(
            session_id=session_id,
            user_id=user.user_id,
            created_at=now,
            expires_at=expires_at,
            last_activity=now,
            ip_address=ip_address,
            user_agent=user_agent,
        )

        with self.session_lock:
            self.active_sessions[session_id] = session

        return session_id

    def validate_session(self, session_token: str) -> tuple[bool, User | None]:
        """Validate session token.

        Args:
            session_token: Session token

        Returns:
            Tuple of (is_valid, user)
        """
        with self.session_lock:
            session = self.active_sessions.get(session_token)

            if not session:
                return False, None

            # Check expiry
            if session.expires_at < datetime.now():
                del self.active_sessions[session_token]
                return False, None

            # Update last activity
            session.last_activity = datetime.now()

            # Get user
            with self._get_db() as conn:
                cursor = conn.execute(
                    """
                    SELECT * FROM users WHERE user_id = ? AND enabled = 1
                """,
                    (session.user_id,),
                )

                user_row = cursor.fetchone()
                if not user_row:
                    del self.active_sessions[session_token]
                    return False, None

                user = self._row_to_user(user_row)
                return True, user

    def create_api_key(
        self,
        user_id: str,
        name: str,
        permissions: list[Permission],
        expires_in_days: int | None = None,
    ) -> tuple[str, ApiKey]:
        """Create API key for user.

        Args:
            user_id: User ID
            name: Key name/description
            permissions: Key permissions
            expires_in_days: Expiry in days

        Returns:
            Tuple of (api_key, ApiKey object)
        """
        api_key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)

        api_key_obj = ApiKey(
            key_id=secrets.token_urlsafe(16),
            key_hash=key_hash,
            user_id=user_id,
            name=name,
            permissions=permissions,
            created_at=datetime.now(),
            expires_at=expires_at,
        )

        with self._get_db() as conn:
            conn.execute(
                """
                INSERT INTO api_keys (
                    key_id, key_hash, user_id, name, permissions,
                    created_at, expires_at, enabled
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    api_key_obj.key_id,
                    api_key_obj.key_hash,
                    api_key_obj.user_id,
                    api_key_obj.name,
                    json.dumps([p.value for p in permissions]),
                    api_key_obj.created_at,
                    api_key_obj.expires_at,
                    api_key_obj.enabled,
                ),
            )
            conn.commit()

        self._audit_log(
            user_id=user_id,
            action="api_key_created",
            resource=f"api_key:{api_key_obj.key_id}",
            success=True,
            details={"name": name, "permissions": [p.value for p in permissions]},
        )

        return api_key, api_key_obj

    def revoke_api_key(self, key_id: str, user_id: str) -> bool:
        """Revoke API key.

        Args:
            key_id: API key ID
            user_id: User ID (for authorization)

        Returns:
            True if revoked successfully
        """
        with self._get_db() as conn:
            cursor = conn.execute(
                """
                UPDATE api_keys
                SET enabled = 0
                WHERE key_id = ? AND user_id = ?
            """,
                (key_id, user_id),
            )

            if cursor.rowcount > 0:
                conn.commit()

                self._audit_log(
                    user_id=user_id,
                    action="api_key_revoked",
                    resource=f"api_key:{key_id}",
                    success=True,
                )
                return True

        return False

    def enable_mfa(self, user_id: str) -> str:
        """Enable MFA for user.

        Args:
            user_id: User ID

        Returns:
            MFA secret for QR code generation
        """
        secret = self.mfa_manager.generate_secret()

        with self._get_db() as conn:
            conn.execute(
                """
                UPDATE users
                SET mfa_enabled = 1, mfa_secret = ?
                WHERE user_id = ?
            """,
                (secret, user_id),
            )
            conn.commit()

        self._audit_log(
            user_id=user_id,
            action="mfa_enabled",
            resource=f"user:{user_id}",
            success=True,
        )

        return secret

    def disable_mfa(self, user_id: str):
        """Disable MFA for user.

        Args:
            user_id: User ID
        """
        with self._get_db() as conn:
            conn.execute(
                """
                UPDATE users
                SET mfa_enabled = 0, mfa_secret = NULL
                WHERE user_id = ?
            """,
                (user_id,),
            )
            conn.commit()

        self._audit_log(
            user_id=user_id,
            action="mfa_disabled",
            resource=f"user:{user_id}",
            success=True,
        )

    def _row_to_user(self, row) -> User:
        """Convert database row to User object.

        Args:
            row: Database row

        Returns:
            User object
        """
        return User(
            user_id=row["user_id"],
            username=row["username"],
            email=row["email"],
            password_hash=row["password_hash"],
            role=UserRole(row["role"]),
            security_level=SecurityLevel(row["security_level"]),
            enabled=bool(row["enabled"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            last_login=datetime.fromisoformat(row["last_login"])
            if row["last_login"]
            else None,
            failed_login_attempts=row["failed_login_attempts"],
            locked_until=datetime.fromisoformat(row["locked_until"])
            if row["locked_until"]
            else None,
            mfa_enabled=bool(row["mfa_enabled"]),
            mfa_secret=row["mfa_secret"],
            metadata=json.loads(row["metadata"]),
        )

    def _audit_log(
        self,
        action: str,
        resource: str,
        success: bool,
        user_id: str | None = None,
        ip_address: str = "unknown",
        user_agent: str = "unknown",
        details: dict[str, Any] | None = None,
        risk_score: float = 0.0,
    ):
        """Log security event.

        Args:
            action: Action performed
            resource: Resource accessed
            success: Whether action succeeded
            user_id: User ID
            ip_address: Client IP
            user_agent: Client user agent
            details: Additional details
            risk_score: Risk assessment score
        """
        if not self.config.enable_audit_logging:
            return

        log_entry = AuditLog(
            log_id=secrets.token_urlsafe(16),
            timestamp=datetime.now(),
            user_id=user_id,
            action=action,
            resource=resource,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            details=details or {},
            risk_score=risk_score,
        )

        with self._get_db() as conn:
            conn.execute(
                """
                INSERT INTO audit_logs (
                    log_id, timestamp, user_id, action, resource,
                    ip_address, user_agent, success, details, risk_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    log_entry.log_id,
                    log_entry.timestamp,
                    log_entry.user_id,
                    log_entry.action,
                    log_entry.resource,
                    log_entry.ip_address,
                    log_entry.user_agent,
                    log_entry.success,
                    json.dumps(log_entry.details),
                    log_entry.risk_score,
                ),
            )
            conn.commit()

    def get_audit_logs(
        self,
        user_id: str | None = None,
        action: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditLog]:
        """Get audit logs with filters.

        Args:
            user_id: Filter by user ID
            action: Filter by action
            start_time: Start time filter
            end_time: End time filter
            limit: Maximum records to return

        Returns:
            List of audit log entries
        """
        query = "SELECT * FROM audit_logs WHERE 1=1"
        params = []

        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)

        if action:
            query += " AND action = ?"
            params.append(action)

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)

        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        logs = []
        with self._get_db() as conn:
            cursor = conn.execute(query, params)
            for row in cursor.fetchall():
                logs.append(
                    AuditLog(
                        log_id=row["log_id"],
                        timestamp=datetime.fromisoformat(row["timestamp"]),
                        user_id=row["user_id"],
                        action=row["action"],
                        resource=row["resource"],
                        ip_address=row["ip_address"],
                        user_agent=row["user_agent"],
                        success=bool(row["success"]),
                        details=json.loads(row["details"]),
                        risk_score=row["risk_score"],
                    )
                )

        return logs


class AuthorizationManager:
    """Role-based access control manager."""

    def __init__(self):
        """Initialize authorization manager."""
        self.role_permissions = self._get_default_role_permissions()

    def _get_default_role_permissions(self) -> dict[UserRole, set[Permission]]:
        """Get default role permissions mapping.

        Returns:
            Role to permissions mapping
        """
        return {
            UserRole.ADMIN: {
                Permission.READ,
                Permission.WRITE,
                Permission.DELETE,
                Permission.EXECUTE,
                Permission.ADMIN,
                Permission.CONFIG,
                Permission.MONITOR,
                Permission.AUDIT,
            },
            UserRole.OPERATOR: {
                Permission.READ,
                Permission.WRITE,
                Permission.EXECUTE,
                Permission.MONITOR,
            },
            UserRole.DEVELOPER: {
                Permission.READ,
                Permission.WRITE,
                Permission.EXECUTE,
                Permission.CONFIG,
            },
            UserRole.ANALYST: {Permission.READ, Permission.MONITOR, Permission.AUDIT},
            UserRole.VIEWER: {Permission.READ},
            UserRole.GUEST: set(),
        }

    def has_permission(self, user: User, permission: Permission) -> bool:
        """Check if user has specific permission.

        Args:
            user: User object
            permission: Permission to check

        Returns:
            True if user has permission
        """
        user_permissions = self.role_permissions.get(user.role, set())
        return permission in user_permissions

    def check_access(
        self,
        user: User,
        resource: str,
        action: Permission,
        required_security_level: SecurityLevel = SecurityLevel.INTERNAL,
    ) -> bool:
        """Check if user can perform action on resource.

        Args:
            user: User object
            resource: Resource identifier
            action: Action/permission required
            required_security_level: Minimum security level required

        Returns:
            True if access granted
        """
        # Check if user is enabled
        if not user.enabled:
            return False

        # Check security level
        security_levels = {
            SecurityLevel.PUBLIC: 0,
            SecurityLevel.INTERNAL: 1,
            SecurityLevel.CONFIDENTIAL: 2,
            SecurityLevel.SECRET: 3,
            SecurityLevel.TOP_SECRET: 4,
        }

        user_level = security_levels.get(user.security_level, 0)
        required_level = security_levels.get(required_security_level, 0)

        if user_level < required_level:
            return False

        # Check permission
        return self.has_permission(user, action)


def require_auth(
    permission: Permission = Permission.READ,
    security_level: SecurityLevel = SecurityLevel.INTERNAL,
):
    """Decorator for requiring authentication and authorization.

    Args:
        permission: Required permission
        security_level: Required security level
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # This is a simplified decorator - in practice would extract
            # user from request context and validate
            return func(*args, **kwargs)

        return wrapper

    return decorator
