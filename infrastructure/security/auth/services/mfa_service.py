"""Multi-Factor Authentication Service.

Provides MFA functionality including TOTP, SMS, email, and backup codes.
Extracted from the EnhancedSecureAPIServer God class for better modularity.
"""

import logging
import secrets
from typing import List, Optional
import hashlib
import time
import base64

try:
    import pyotp
    import qrcode
    from io import BytesIO
    TOTP_AVAILABLE = True
except ImportError:
    TOTP_AVAILABLE = False

from ..interfaces import IMFAService, MFAMethodType, MFASetupResult, MFAStatus

logger = logging.getLogger(__name__)


class MFAError(Exception):
    """MFA-related error."""
    pass


class MFAService(IMFAService):
    """Multi-factor authentication service.
    
    Provides TOTP, SMS, email, and backup code authentication methods.
    Extracted from the God class to follow Single Responsibility Principle.
    """

    def __init__(self, config: dict = None):
        """Initialize MFA service."""
        self.config = config or {}
        
        # MFA configuration
        self.totp_window = self.config.get("totp_window", 1)  # Allow 30s window
        self.backup_code_length = self.config.get("backup_code_length", 8)
        self.backup_code_count = self.config.get("backup_code_count", 10)
        
        # Storage for MFA data (in production, use database)
        self.user_mfa_data: dict = {}
        self.backup_codes: dict = {}
        
        if not TOTP_AVAILABLE:
            logger.warning("TOTP libraries not available, TOTP functionality disabled")

    def setup_totp(self, user_id: str, email: str) -> MFASetupResult:
        """Setup TOTP for user."""
        try:
            if not TOTP_AVAILABLE:
                return MFASetupResult(
                    success=False,
                    error_message="TOTP libraries not available"
                )
            
            # Generate secret
            secret = pyotp.random_base32()
            
            # Create TOTP instance
            totp = pyotp.TOTP(secret)
            
            # Generate provisioning URI
            provisioning_uri = totp.provisioning_uri(
                name=email,
                issuer_name="AIVillage"
            )
            
            # Generate QR code
            qr_code_data = self._generate_qr_code(provisioning_uri)
            
            # Generate backup codes
            backup_codes = self.generate_backup_codes(user_id)
            
            # Store secret (in production, encrypt this)
            if user_id not in self.user_mfa_data:
                self.user_mfa_data[user_id] = {}
            
            self.user_mfa_data[user_id]["totp_secret"] = secret
            self.user_mfa_data[user_id]["totp_enabled"] = False  # Not enabled until verified
            
            return MFASetupResult(
                success=True,
                qr_code=qr_code_data,
                backup_codes=backup_codes,
                secret=secret
            )
            
        except Exception as e:
            logger.error(f"TOTP setup failed for user {user_id}: {e}")
            return MFASetupResult(
                success=False,
                error_message=f"TOTP setup failed: {e}"
            )

    def verify_mfa(
        self,
        user_id: str,
        method: MFAMethodType,
        token: str,
        secret: Optional[str] = None
    ) -> bool:
        """Verify MFA token."""
        try:
            if method == MFAMethodType.TOTP:
                return self._verify_totp(user_id, token, secret)
            elif method == MFAMethodType.BACKUP_CODE:
                return self.verify_backup_code(user_id, token)
            elif method == MFAMethodType.SMS:
                return self._verify_sms(user_id, token)
            elif method == MFAMethodType.EMAIL:
                return self._verify_email(user_id, token)
            else:
                logger.warning(f"Unsupported MFA method: {method}")
                return False
                
        except Exception as e:
            logger.error(f"MFA verification failed for user {user_id}: {e}")
            return False

    def get_user_mfa_status(self, user_id: str) -> MFAStatus:
        """Get MFA status for user."""
        try:
            user_data = self.user_mfa_data.get(user_id, {})
            backup_codes_count = len(self.backup_codes.get(user_id, []))
            
            methods_available = []
            if user_data.get("totp_enabled", False):
                methods_available.append("TOTP")
            if user_data.get("sms_enabled", False):
                methods_available.append("SMS")
            if user_data.get("email_enabled", False):
                methods_available.append("Email")
            if backup_codes_count > 0:
                methods_available.append("Backup Codes")
            
            return MFAStatus(
                totp_enabled=user_data.get("totp_enabled", False),
                sms_enabled=user_data.get("sms_enabled", False),
                email_enabled=user_data.get("email_enabled", False),
                backup_codes_available=backup_codes_count,
                methods_available=methods_available
            )
            
        except Exception as e:
            logger.error(f"Failed to get MFA status for user {user_id}: {e}")
            return MFAStatus()

    def generate_backup_codes(self, user_id: str, count: int = 10) -> List[str]:
        """Generate backup codes for user."""
        try:
            codes = []
            for _ in range(count):
                # Generate 8-character alphanumeric code
                code = ''.join(secrets.choice('ABCDEFGHJKMNPQRSTUVWXYZ23456789') 
                              for _ in range(self.backup_code_length))
                codes.append(code)
            
            # Store hashed versions
            hashed_codes = []
            for code in codes:
                hashed = hashlib.sha256(code.encode()).hexdigest()
                hashed_codes.append(hashed)
            
            self.backup_codes[user_id] = hashed_codes
            
            logger.info(f"Generated {count} backup codes for user {user_id}")
            return codes
            
        except Exception as e:
            logger.error(f"Failed to generate backup codes for user {user_id}: {e}")
            return []

    def verify_backup_code(self, user_id: str, code: str) -> bool:
        """Verify backup code."""
        try:
            user_codes = self.backup_codes.get(user_id, [])
            if not user_codes:
                return False
            
            # Hash the provided code
            code_hash = hashlib.sha256(code.upper().encode()).hexdigest()
            
            # Check if code exists and remove it (one-time use)
            if code_hash in user_codes:
                user_codes.remove(code_hash)
                logger.info(f"Backup code used for user {user_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Backup code verification failed for user {user_id}: {e}")
            return False

    def disable_mfa(self, user_id: str, method: MFAMethodType) -> bool:
        """Disable MFA method for user."""
        try:
            if user_id not in self.user_mfa_data:
                return False
            
            if method == MFAMethodType.TOTP:
                self.user_mfa_data[user_id]["totp_enabled"] = False
                if "totp_secret" in self.user_mfa_data[user_id]:
                    del self.user_mfa_data[user_id]["totp_secret"]
            elif method == MFAMethodType.SMS:
                self.user_mfa_data[user_id]["sms_enabled"] = False
            elif method == MFAMethodType.EMAIL:
                self.user_mfa_data[user_id]["email_enabled"] = False
            elif method == MFAMethodType.BACKUP_CODE:
                if user_id in self.backup_codes:
                    del self.backup_codes[user_id]
            
            logger.info(f"Disabled {method.value} MFA for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to disable MFA for user {user_id}: {e}")
            return False

    def enable_totp(self, user_id: str) -> bool:
        """Enable TOTP after successful verification."""
        try:
            if user_id in self.user_mfa_data and "totp_secret" in self.user_mfa_data[user_id]:
                self.user_mfa_data[user_id]["totp_enabled"] = True
                logger.info(f"TOTP enabled for user {user_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to enable TOTP for user {user_id}: {e}")
            return False

    def _verify_totp(self, user_id: str, token: str, secret: Optional[str] = None) -> bool:
        """Verify TOTP token."""
        try:
            if not TOTP_AVAILABLE:
                return False
            
            # Get secret from storage or parameter
            if secret:
                totp_secret = secret
            else:
                user_data = self.user_mfa_data.get(user_id, {})
                totp_secret = user_data.get("totp_secret")
            
            if not totp_secret:
                return False
            
            # Create TOTP instance and verify
            totp = pyotp.TOTP(totp_secret)
            
            # Verify with time window
            for i in range(-self.totp_window, self.totp_window + 1):
                if totp.verify(token, valid_window=i):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"TOTP verification failed for user {user_id}: {e}")
            return False

    def _verify_sms(self, user_id: str, token: str) -> bool:
        """Verify SMS token."""
        # In a real implementation, this would verify against stored SMS tokens
        # For now, return False as SMS is not implemented
        logger.warning("SMS MFA not implemented")
        return False

    def _verify_email(self, user_id: str, token: str) -> bool:
        """Verify email token."""
        # In a real implementation, this would verify against stored email tokens
        # For now, return False as email is not implemented
        logger.warning("Email MFA not implemented")
        return False

    def _generate_qr_code(self, provisioning_uri: str) -> str:
        """Generate QR code for TOTP setup."""
        try:
            if not TOTP_AVAILABLE:
                return ""
            
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=4,
            )
            qr.add_data(provisioning_uri)
            qr.make(fit=True)
            
            img = qr.make_image(fill_color="black", back_color="white")
            
            # Convert to base64
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            img_data = buffer.getvalue()
            
            return base64.b64encode(img_data).decode()
            
        except Exception as e:
            logger.error(f"QR code generation failed: {e}")
            return ""

    def get_user_totp_secret(self, user_id: str) -> Optional[str]:
        """Get TOTP secret for user (for internal use)."""
        user_data = self.user_mfa_data.get(user_id, {})
        return user_data.get("totp_secret")

    def is_mfa_required(self, user_id: str) -> bool:
        """Check if MFA is required for user."""
        status = self.get_user_mfa_status(user_id)
        return (status.totp_enabled or status.sms_enabled or 
                status.email_enabled or status.backup_codes_available > 0)

    def cleanup_expired_data(self) -> int:
        """Clean up expired MFA data."""
        # In a real implementation, this would clean up expired SMS/email tokens
        # For now, just return 0
        return 0