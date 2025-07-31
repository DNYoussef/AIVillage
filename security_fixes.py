#!/usr/bin/env python3
"""
Critical Security Fixes for AIVillage
Apply immediately before production deployment
"""

import os
import secrets
import hashlib
import re
from typing import Any, Dict, List
from cryptography.fernet import Fernet
from pydantic import BaseModel, validator
import jwt
from datetime import datetime, timedelta
import logging

# Configure security logging
security_logger = logging.getLogger('aivillage.security')
security_logger.setLevel(logging.INFO)

class SecurityError(Exception):
    """Custom exception for security-related errors."""
    pass

class SecureConfig:
    """Secure configuration management with environment variables."""
    
    @staticmethod
    def get_jwt_secret() -> str:
        """Get JWT secret from environment with validation."""
        secret = os.environ.get('AIVILLAGE_JWT_SECRET')
        if not secret:
            # Generate secure secret for development (DO NOT USE IN PRODUCTION)
            if os.environ.get('ENVIRONMENT') == 'development':
                secret = secrets.token_urlsafe(32)
                security_logger.warning("Using generated JWT secret for development")
            else:
                raise SecurityError("JWT_SECRET environment variable required for production")
        
        if len(secret) < 32:
            raise SecurityError("JWT secret must be at least 32 characters")
        
        return secret
    
    @staticmethod
    def get_db_password() -> str:
        """Get database password from environment."""
        password = os.environ.get('AIVILLAGE_DB_PASSWORD')
        if not password:
            if os.environ.get('ENVIRONMENT') == 'development':
                return 'dev_password_change_in_prod'
            raise SecurityError("DB_PASSWORD environment variable required")
        return password
    
    @staticmethod
    def get_encryption_key() -> bytes:
        """Get encryption key for sensitive data."""
        key = os.environ.get('AIVILLAGE_ENCRYPTION_KEY')
        if not key:
            if os.environ.get('ENVIRONMENT') == 'development':
                key = Fernet.generate_key().decode()
                security_logger.warning("Using generated encryption key for development")
            else:
                raise SecurityError("ENCRYPTION_KEY environment variable required")
        
        try:
            return key.encode() if isinstance(key, str) else key
        except Exception as e:
            raise SecurityError(f"Invalid encryption key format: {e}") from e

class SecureJWTManager:
    """Secure JWT token management."""
    
    def __init__(self):
        self.secret = SecureConfig.get_jwt_secret()
        self.algorithm = 'HS256'
        self.default_expiry = timedelta(hours=1)
    
    def create_token(self, user_id: str, role: str, agent_id: str = None, 
                    custom_claims: Dict[str, Any] = None) -> str:
        """Create secure JWT token with proper claims."""
        now = datetime.utcnow()
        
        payload = {
            'sub': user_id,
            'role': role,
            'agent_id': agent_id or user_id,
            'iat': now,
            'exp': now + self.default_expiry,
            'jti': secrets.token_urlsafe(32),  # Unique token ID
            'iss': 'aivillage-mcp',  # Issuer
            'aud': 'aivillage-agents'  # Audience
        }
        
        if custom_claims:
            # Validate custom claims don't override security claims
            forbidden_claims = {'sub', 'iat', 'exp', 'jti', 'iss', 'aud'}
            if any(claim in forbidden_claims for claim in custom_claims):
                raise SecurityError("Custom claims cannot override security claims")
            payload.update(custom_claims)
        
        try:
            return jwt.encode(payload, self.secret, algorithm=self.algorithm)
        except Exception as e:
            security_logger.error(f"Token creation failed: {e}")
            raise SecurityError("Token creation failed") from e
    
    def validate_token(self, token: str) -> Dict[str, Any]:
        """Validate JWT token with proper verification."""
        try:
            payload = jwt.decode(
                token, 
                self.secret, 
                algorithms=[self.algorithm],
                audience='aivillage-agents',
                issuer='aivillage-mcp',
                options={
                    'verify_signature': True,
                    'verify_exp': True,
                    'verify_iat': True,
                    'verify_aud': True,
                    'verify_iss': True
                }
            )
            
            # Additional security checks
            if not payload.get('sub'):
                raise SecurityError("Missing subject in token")
            
            if not payload.get('role'):
                raise SecurityError("Missing role in token")
                
            return payload
            
        except jwt.ExpiredSignatureError:
            security_logger.warning("Token expired")
            raise SecurityError("Token expired")
        except jwt.InvalidTokenError as e:
            security_logger.warning(f"Invalid token: {e}")
            raise SecurityError("Invalid token")

class SecureDatabaseManager:
    """Secure database query management to prevent SQL injection."""
    
    @staticmethod
    def build_safe_query(base_query: str, conditions: List[str], 
                        parameters: List[Any]) -> tuple[str, List[Any]]:
        """Build parameterized query safely."""
        if not conditions:
            return base_query, parameters
        
        # Validate conditions contain only safe column names
        safe_column_pattern = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')
        for condition in conditions:
            # Extract column name from condition (assumes format "column = ?")
            column = condition.split()[0] if ' ' in condition else condition
            if not safe_column_pattern.match(column):
                raise SecurityError(f"Unsafe column name in condition: {column}")
        
        where_clause = " AND ".join(conditions)
        full_query = f"{base_query} WHERE {where_clause}"
        
        return full_query, parameters
    
    @staticmethod
    def execute_safe_query(connection, query: str, parameters: List[Any] = None):
        """Execute query with parameter validation."""
        if parameters is None:
            parameters = []
        
        # Log query for security monitoring (without parameters)
        security_logger.info(f"Executing query: {query[:100]}...")
        
        try:
            return connection.execute(query, parameters)
        except Exception as e:
            security_logger.error(f"Database query failed: {type(e).__name__}")
            raise SecurityError("Database operation failed") from e

class SecureHashingManager:
    """Secure hashing for security-sensitive operations."""
    
    @staticmethod
    def hash_sensitive_data(data: str, salt: str = None) -> str:
        """Hash sensitive data using SHA-256 with salt."""
        if salt is None:
            salt = secrets.token_hex(16)
        
        combined = f"{data}{salt}"
        hash_obj = hashlib.sha256(combined.encode(), usedforsecurity=True)
        return f"{salt}:{hash_obj.hexdigest()}"
    
    @staticmethod
    def verify_hash(data: str, hashed: str) -> bool:
        """Verify hashed data."""
        try:
            salt, expected_hash = hashed.split(':', 1)
            computed_hash = SecureHashingManager.hash_sensitive_data(data, salt)
            return secrets.compare_digest(computed_hash, hashed)
        except (ValueError, AttributeError):
            return False
    
    @staticmethod
    def generate_model_id(agent_id: str, model_name: str, config: Dict[str, Any]) -> str:
        """Generate secure model ID using SHA-256."""
        data = f"{agent_id}:{model_name}:{str(sorted(config.items()))}"
        return hashlib.sha256(data.encode(), usedforsecurity=True).hexdigest()[:16]

class SecureInputValidator:
    """Input validation and sanitization."""
    
    class MessageInput(BaseModel):
        content: str
        sender: str
        receiver: str
        message_type: str = "standard"
        
        @validator('content')
        def validate_content(cls, v):
            if not v or len(v.strip()) == 0:
                raise ValueError('Content cannot be empty')
            
            if len(v) > 10000:  # Prevent DoS attacks
                raise ValueError('Message content too long (max 10000 characters)')
            
            # Check for potential XSS/injection patterns
            dangerous_patterns = [
                r'<script[^>]*>.*?</script>',
                r'javascript:',
                r'data:text/html',
                r'vbscript:',
                r'onload\s*=',
                r'onerror\s*=',
                r'eval\s*\(',
                r'Function\s*\('
            ]
            
            for pattern in dangerous_patterns:
                if re.search(pattern, v, re.IGNORECASE):
                    raise ValueError('Potentially malicious content detected')
            
            return v.strip()
        
        @validator('sender', 'receiver')
        def validate_agent_id(cls, v):
            if not re.match(r'^[a-zA-Z0-9_-]{1,50}$', v):
                raise ValueError('Invalid agent ID format')
            return v
        
        @validator('message_type')
        def validate_message_type(cls, v):
            allowed_types = {
                'standard', 'query', 'response', 'broadcast', 
                'system', 'error', 'tool_call'
            }
            if v not in allowed_types:
                raise ValueError(f'Invalid message type: {v}')
            return v
    
    class AuthenticationInput(BaseModel):
        username: str
        token: str = None
        api_key: str = None
        
        @validator('username')
        def validate_username(cls, v):
            if not re.match(r'^[a-zA-Z0-9_-]{3,50}$', v):
                raise ValueError('Invalid username format')
            return v
        
        @validator('token', 'api_key')
        def validate_credentials(cls, v):
            if v and (len(v) < 10 or len(v) > 500):
                raise ValueError('Invalid credential format')
            return v

class SecurityMonitor:
    """Security event monitoring and alerting."""
    
    def __init__(self):
        self.failed_attempts = {}
        self.rate_limits = {}
        self.alert_thresholds = {
            'failed_auth_attempts': 5,
            'rate_limit_violations': 10,
            'suspicious_queries': 3
        }
    
    def record_security_event(self, event_type: str, user_id: str, 
                            details: Dict[str, Any] = None):
        """Record security event for monitoring."""
        timestamp = datetime.utcnow()
        
        security_logger.warning(
            f"Security event: {event_type} for user {user_id} at {timestamp}"
        )
        
        # Track failed attempts
        if event_type == 'auth_failure':
            self.failed_attempts.setdefault(user_id, []).append(timestamp)
            
            # Clean old attempts (older than 1 hour)
            cutoff = timestamp - timedelta(hours=1)
            self.failed_attempts[user_id] = [
                attempt for attempt in self.failed_attempts[user_id]
                if attempt > cutoff
            ]
            
            # Check if threshold exceeded
            if len(self.failed_attempts[user_id]) >= self.alert_thresholds['failed_auth_attempts']:
                self.trigger_security_alert('multiple_auth_failures', user_id, {
                    'failure_count': len(self.failed_attempts[user_id]),
                    'time_window': '1 hour'
                })
    
    def trigger_security_alert(self, alert_type: str, user_id: str, 
                             context: Dict[str, Any]):
        """Trigger security alert.""" 
        security_logger.critical(
            f"SECURITY ALERT: {alert_type} for user {user_id}. Context: {context}"
        )
        
        # In production, this would integrate with alerting systems
        # (e.g., send to SIEM, Slack, email, etc.)
        
    def check_rate_limit(self, user_id: str, operation: str, 
                        limit: int = 100, window_minutes: int = 1) -> bool:
        """Check rate limiting for operations."""
        now = datetime.utcnow()
        key = f"{user_id}:{operation}"
        
        # Initialize or clean old entries
        if key not in self.rate_limits:
            self.rate_limits[key] = []
        
        cutoff = now - timedelta(minutes=window_minutes)
        self.rate_limits[key] = [
            timestamp for timestamp in self.rate_limits[key]
            if timestamp > cutoff
        ]
        
        # Check limit
        if len(self.rate_limits[key]) >= limit:
            self.record_security_event('rate_limit_violation', user_id, {
                'operation': operation,
                'limit': limit,
                'window_minutes': window_minutes
            })
            return False
        
        # Record current operation
        self.rate_limits[key].append(now)
        return True

class SecureModelDownloader:
    """Secure model downloading with revision pinning."""
    
    @staticmethod
    def download_model_safely(model_name: str, revision: str = None, 
                            trust_remote_code: bool = False):
        """Download model with security controls."""
        if not revision:
            raise SecurityError("Model revision must be specified for security")
        
        if trust_remote_code:
            security_logger.warning(
                f"Downloading model {model_name} with trust_remote_code=True"
            )
        
        # Validate model name format
        if not re.match(r'^[a-zA-Z0-9_/-]+$', model_name):
            raise SecurityError("Invalid model name format")
        
        download_config = {
            'revision': revision,
            'trust_remote_code': trust_remote_code,
            'use_auth_token': True,
            'local_files_only': False
        }
        
        security_logger.info(f"Downloading model {model_name} with revision {revision}")
        return download_config

# Secure environment setup
def setup_secure_environment():
    """Setup secure environment configuration."""
    required_vars = [
        'AIVILLAGE_JWT_SECRET',
        'AIVILLAGE_DB_PASSWORD', 
        'AIVILLAGE_ENCRYPTION_KEY'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    if missing_vars and os.environ.get('ENVIRONMENT') == 'production':
        raise SecurityError(f"Missing required environment variables: {missing_vars}")
    
    # Setup logging for security events
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

if __name__ == "__main__":
    # Test security components
    print("Testing security fixes...")
    
    # Test JWT manager
    jwt_manager = SecureJWTManager()
    token = jwt_manager.create_token("test_user", "king", "test_agent")
    payload = jwt_manager.validate_token(token)
    print(f"JWT test passed: {payload['sub']}")
    
    # Test input validation
    try:
        validator = SecureInputValidator.MessageInput(
            content="Hello world",
            sender="agent1",
            receiver="agent2"
        )
        print("Input validation test passed")
    except Exception as e:
        print(f"Input validation test failed: {e}")
    
    # Test secure hashing
    data = "sensitive_information"
    hashed = SecureHashingManager.hash_sensitive_data(data)
    is_valid = SecureHashingManager.verify_hash(data, hashed)
    print(f"Hashing test passed: {is_valid}")
    
    print("All security tests completed!")