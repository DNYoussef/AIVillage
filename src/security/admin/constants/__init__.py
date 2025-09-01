"""
Security Admin Constants Module
Centralized constants to reduce magic literals in security administration.
"""

# Session and Authentication Constants
SESSION_TIMEOUT_MINUTES = 30
MAX_SESSIONS_PER_USER = 3
MFA_TOKEN_LENGTH = 6
SESSION_ID_LENGTH = 32

# Rate Limiting and Security
MAX_BLOCKED_ATTEMPTS = 5
BLOCKING_DURATION_MINUTES = 15

# HTTP Status Codes
HTTP_STATUS_OK = 200
HTTP_STATUS_UNAUTHORIZED = 401
HTTP_STATUS_FORBIDDEN = 403
HTTP_STATUS_TOO_MANY_REQUESTS = 429

# Security Headers
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY", 
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
}

# Allowed IPs and Network
LOCALHOST_IPS = {'127.0.0.1', '::1'}
DEFAULT_BIND_INTERFACE = "127.0.0.1"
DEFAULT_ADMIN_PORT = 3006

# CORS Settings
CORS_ALLOWED_ORIGINS = [
    "http://127.0.0.1:3000",
    "https://127.0.0.1:3000", 
    "http://localhost:3000",
    "https://localhost:3000"
]

CORS_ALLOWED_METHODS = ["GET", "POST", "PUT", "DELETE"]
CORS_ALLOWED_HEADERS = ["Content-Type", "Authorization", "X-Admin-Session"]

# Audit Logging
DEFAULT_AUDIT_LOG_PATH = "logs/admin_audit.log"

# User Roles and Permissions
ROLE_PERMISSIONS = {
    "super_admin": {"admin:read", "admin:write", "admin:system", "admin:users", "admin:security"},
    "admin": {"admin:read", "admin:write", "admin:system"},
    "operator": {"admin:read", "admin:system"},
    "viewer": {"admin:read"}
}

# Emergency Shutdown
EMERGENCY_CONFIRMATION_TEXT = "EMERGENCY_SHUTDOWN_CONFIRMED"

__all__ = [
    'SESSION_TIMEOUT_MINUTES', 'MAX_SESSIONS_PER_USER', 'MFA_TOKEN_LENGTH', 
    'SESSION_ID_LENGTH', 'MAX_BLOCKED_ATTEMPTS', 'BLOCKING_DURATION_MINUTES',
    'HTTP_STATUS_OK', 'HTTP_STATUS_UNAUTHORIZED', 'HTTP_STATUS_FORBIDDEN', 
    'HTTP_STATUS_TOO_MANY_REQUESTS', 'SECURITY_HEADERS', 'LOCALHOST_IPS',
    'DEFAULT_BIND_INTERFACE', 'DEFAULT_ADMIN_PORT', 'CORS_ALLOWED_ORIGINS',
    'CORS_ALLOWED_METHODS', 'CORS_ALLOWED_HEADERS', 'DEFAULT_AUDIT_LOG_PATH',
    'ROLE_PERMISSIONS', 'EMERGENCY_CONFIRMATION_TEXT'
]