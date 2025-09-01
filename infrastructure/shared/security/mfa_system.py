"""Multi-Factor Authentication System for AIVillage Security.

Provides TOTP, SMS, Email verification with backup codes and recovery mechanisms.
"""

import base64
from datetime import datetime, timedelta
import hashlib
import hmac
from io import BytesIO
import logging
import secrets
from typing import Any

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import pyotp
import qrcode

logger = logging.getLogger(__name__)


class MFAError(Exception):
    """MFA-related errors."""

    pass


class MFAMethodType:
    """MFA method types."""

    TOTP = "totp"
    SMS = "sms"
    EMAIL = "email"
    BACKUP_CODES = "backup_codes"


class MFABackupCodes:
    """Manages backup codes for MFA recovery."""

    def __init__(self):
        self.backend = default_backend()

    def generate_backup_codes(self, count: int = 10) -> list[str]:
        """Generate cryptographically secure backup codes."""
        codes = []
        for _ in range(count):
            # Generate 8-character alphanumeric code
            code = secrets.token_hex(4).upper()
            codes.append(f"{code[:4]}-{code[4:]}")
        return codes

    def hash_backup_code(self, code: str) -> str:
        """Hash backup code for secure storage."""
        # Remove formatting
        clean_code = code.replace("-", "").upper()

        # Use PBKDF2 with random salt
        salt = secrets.token_bytes(32)
        kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=100000, backend=self.backend)

        key = kdf.derive(clean_code.encode())

        # Return salt + hash as base64
        return base64.b64encode(salt + key).decode()

    def verify_backup_code(self, code: str, stored_hash: str) -> bool:
        """Verify backup code against stored hash."""
        try:
            clean_code = code.replace("-", "").upper()

            # Decode stored hash
            decoded = base64.b64decode(stored_hash.encode())
            salt = decoded[:32]
            expected_key = decoded[32:]

            # Derive key from provided code
            kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=100000, backend=self.backend)

            derived_key = kdf.derive(clean_code.encode())

            return hmac.compare_digest(derived_key, expected_key)

        except Exception as e:
            logger.error(f"Backup code verification failed: {e}")
            return False


class TOTPManager:
    """TOTP (Time-based One-Time Password) manager."""

    def __init__(self, issuer_name: str = "AIVillage"):
        self.issuer_name = issuer_name

    def generate_secret(self) -> str:
        """Generate new TOTP secret."""
        return pyotp.random_base32()

    def generate_qr_code(self, secret: str, user_email: str) -> bytes:
        """Generate QR code for TOTP setup."""
        try:
            totp = pyotp.TOTP(secret)
            provisioning_uri = totp.provisioning_uri(name=user_email, issuer_name=self.issuer_name)

            # Generate QR code
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=4,
            )
            qr.add_data(provisioning_uri)
            qr.make(fit=True)

            img = qr.make_image(fill_color="black", back_color="white")

            # Convert to bytes
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            return buffer.getvalue()

        except Exception as e:
            logger.error(f"QR code generation failed: {e}")
            raise MFAError(f"Failed to generate QR code: {e}")

    def verify_totp(self, secret: str, token: str, window: int = 1) -> bool:
        """Verify TOTP token with time window tolerance."""
        try:
            totp = pyotp.TOTP(secret)
            return totp.verify(token, valid_window=window)
        except Exception as e:
            logger.error(f"TOTP verification failed: {e}")
            return False

    def get_current_token(self, secret: str) -> str:
        """Get current TOTP token (for testing/debugging)."""
        totp = pyotp.TOTP(secret)
        return totp.now()


class SMSProvider:
    """SMS provider interface (mock implementation)."""

    def send_sms(self, phone_number: str, message: str) -> bool:
        """Send SMS verification code."""
        # Mock implementation - in production, integrate with Twilio, AWS SNS, etc.
        logger.info(f"SMS sent to {phone_number}: {message}")
        return True


class EmailProvider:
    """Email provider interface (mock implementation)."""

    def send_email(self, email: str, subject: str, message: str) -> bool:
        """Send email verification code."""
        # Mock implementation - in production, integrate with SendGrid, AWS SES, etc.
        logger.info(f"Email sent to {email}: {subject}")
        return True


class MFASystem:
    """Comprehensive Multi-Factor Authentication system."""

    def __init__(self):
        self.totp_manager = TOTPManager()
        self.backup_codes = MFABackupCodes()
        self.sms_provider = SMSProvider()
        self.email_provider = EmailProvider()

        # In-memory storage for verification codes (use Redis in production)
        self.verification_codes = {}
        self.rate_limits = {}

    def setup_totp(self, user_id: str, user_email: str) -> dict[str, Any]:
        """Set up TOTP for a user."""
        try:
            secret = self.totp_manager.generate_secret()
            qr_code = self.totp_manager.generate_qr_code(secret, user_email)

            return {
                "secret": secret,
                "qr_code": base64.b64encode(qr_code).decode(),
                "backup_codes": self.generate_backup_codes(user_id),
                "method": MFAMethodType.TOTP,
            }

        except Exception as e:
            logger.error(f"TOTP setup failed for user {user_id}: {e}")
            raise MFAError(f"Failed to set up TOTP: {e}")

    def generate_backup_codes(self, user_id: str) -> list[str]:
        """Generate and store backup codes for user."""
        codes = self.backup_codes.generate_backup_codes()

        # Hash codes for storage
        hashed_codes = [self.backup_codes.hash_backup_code(code) for code in codes]

        # Store hashed codes (in production, store in database)
        self.store_backup_codes(user_id, hashed_codes)

        return codes

    def store_backup_codes(self, user_id: str, hashed_codes: list[str]):
        """Store backup codes securely."""
        # In production, store in secure database
        logger.info(f"Backup codes stored for user {user_id}")

    def verify_totp(self, user_id: str, token: str, secret: str) -> bool:
        """Verify TOTP token for user."""
        if not self.check_rate_limit(user_id, MFAMethodType.TOTP):
            return False

        return self.totp_manager.verify_totp(secret, token)

    def send_sms_verification(self, user_id: str, phone_number: str) -> bool:
        """Send SMS verification code."""
        if not self.check_rate_limit(user_id, MFAMethodType.SMS):
            return False

        # Generate 6-digit code
        code = f"{secrets.randbelow(900000) + 100000:06d}"

        # Store code with expiration
        self.store_verification_code(user_id, MFAMethodType.SMS, code)

        # Send SMS
        message = f"Your AIVillage verification code is: {code}"
        return self.sms_provider.send_sms(phone_number, message)

    def send_email_verification(self, user_id: str, email: str) -> bool:
        """Send email verification code."""
        if not self.check_rate_limit(user_id, MFAMethodType.EMAIL):
            return False

        # Generate 6-digit code
        code = f"{secrets.randbelow(900000) + 100000:06d}"

        # Store code with expiration
        self.store_verification_code(user_id, MFAMethodType.EMAIL, code)

        # Send email
        subject = "AIVillage Security Verification"
        message = f"Your verification code is: {code}\n\nThis code expires in 10 minutes."
        return self.email_provider.send_email(email, subject, message)

    def verify_sms_code(self, user_id: str, code: str) -> bool:
        """Verify SMS verification code."""
        return self.verify_stored_code(user_id, MFAMethodType.SMS, code)

    def verify_email_code(self, user_id: str, code: str) -> bool:
        """Verify email verification code."""
        return self.verify_stored_code(user_id, MFAMethodType.EMAIL, code)

    def verify_backup_code(self, user_id: str, code: str) -> bool:
        """Verify backup code and mark as used."""
        # Get stored backup codes for user
        stored_codes = self.get_backup_codes(user_id)

        for i, stored_hash in enumerate(stored_codes):
            if self.backup_codes.verify_backup_code(code, stored_hash):
                # Mark code as used by removing it
                self.remove_backup_code(user_id, i)
                return True

        return False

    def store_verification_code(self, user_id: str, method: str, code: str):
        """Store verification code with expiration."""
        key = f"{user_id}:{method}"
        expiration = datetime.utcnow() + timedelta(minutes=10)

        # Hash the code for storage
        code_hash = hashlib.sha256(code.encode()).hexdigest()

        self.verification_codes[key] = {"hash": code_hash, "expires_at": expiration, "attempts": 0}

    def verify_stored_code(self, user_id: str, method: str, code: str) -> bool:
        """Verify stored verification code."""
        key = f"{user_id}:{method}"

        if key not in self.verification_codes:
            return False

        stored_data = self.verification_codes[key]

        # Check expiration
        if datetime.utcnow() > stored_data["expires_at"]:
            del self.verification_codes[key]
            return False

        # Check attempts limit
        if stored_data["attempts"] >= 3:
            del self.verification_codes[key]
            return False

        # Verify code
        code_hash = hashlib.sha256(code.encode()).hexdigest()

        stored_data["attempts"] += 1

        if hmac.compare_digest(code_hash, stored_data["hash"]):
            del self.verification_codes[key]
            return True

        return False

    def check_rate_limit(self, user_id: str, method: str) -> bool:
        """Check rate limiting for MFA attempts."""
        key = f"{user_id}:{method}"
        now = datetime.utcnow()

        if key not in self.rate_limits:
            self.rate_limits[key] = []

        # Clean old attempts
        self.rate_limits[key] = [
            attempt_time for attempt_time in self.rate_limits[key] if now - attempt_time < timedelta(hours=1)
        ]

        # Check limit (5 attempts per hour)
        if len(self.rate_limits[key]) >= 5:
            return False

        self.rate_limits[key].append(now)
        return True

    def get_backup_codes(self, user_id: str) -> list[str]:
        """Get stored backup codes for user."""
        # In production, retrieve from secure database
        return []

    def remove_backup_code(self, user_id: str, index: int):
        """Remove used backup code."""
        # In production, mark code as used in database
        logger.info(f"Backup code {index} used for user {user_id}")

    def verify_mfa(self, user_id: str, method: str, token: str, **kwargs) -> bool:
        """Unified MFA verification method."""
        try:
            if method == MFAMethodType.TOTP:
                secret = kwargs.get("secret")
                if not secret:
                    raise MFAError("TOTP secret required")
                return self.verify_totp(user_id, token, secret)

            elif method == MFAMethodType.SMS:
                return self.verify_sms_code(user_id, token)

            elif method == MFAMethodType.EMAIL:
                return self.verify_email_code(user_id, token)

            elif method == MFAMethodType.BACKUP_CODES:
                return self.verify_backup_code(user_id, token)

            else:
                raise MFAError(f"Unsupported MFA method: {method}")

        except Exception as e:
            logger.error(f"MFA verification failed: {e}")
            return False

    def get_user_mfa_status(self, user_id: str) -> dict[str, Any]:
        """Get MFA status for user."""
        # In production, retrieve from database
        return {
            "totp_enabled": False,
            "sms_enabled": False,
            "email_enabled": False,
            "backup_codes_count": 0,
            "methods_available": [
                MFAMethodType.TOTP,
                MFAMethodType.SMS,
                MFAMethodType.EMAIL,
                MFAMethodType.BACKUP_CODES,
            ],
        }

    def disable_mfa(self, user_id: str, method: str) -> bool:
        """Disable specific MFA method for user."""
        try:
            # In production, update database
            logger.info(f"MFA method {method} disabled for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to disable MFA: {e}")
            return False
