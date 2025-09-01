"""
Secure CORS Configuration for AIVillage
=======================================

Provides environment-specific CORS policies to replace wildcard configurations.
"""

import os
from typing import List


def get_secure_cors_origins() -> List[str]:
    """
    Get secure CORS origins based on environment.
    
    Returns:
        List of allowed origins (no wildcards)
    """
    # Environment-specific origins
    env = os.getenv("AIVILLAGE_ENV", "development")
    
    if env == "production":
        # Production: Only specific domains
        return [
            "https://aivillage.app",
            "https://www.aivillage.app",
            "https://api.aivillage.app"
        ]
    elif env == "staging":
        # Staging: Staging domains + localhost for testing
        return [
            "https://staging.aivillage.app",
            "https://test.aivillage.app",
            "http://localhost:3000",
            "http://localhost:8080"
        ]
    else:
        # Development: Localhost only with common dev ports
        return [
            "http://localhost:3000",
            "http://localhost:3001", 
            "http://localhost:3002",
            "http://localhost:8080",
            "http://localhost:8083",
            "http://localhost:8084",
            "http://localhost:8085",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8080"
        ]


def get_admin_cors_origins() -> List[str]:
    """
    Get CORS origins for admin interfaces (more restrictive).
    
    Returns:
        List of allowed origins for admin endpoints
    """
    env = os.getenv("AIVILLAGE_ENV", "development")
    
    if env == "production":
        # Production: Admin subdomain only
        return [
            "https://admin.aivillage.app"
        ]
    elif env == "staging":
        # Staging: Staging admin only
        return [
            "https://admin-staging.aivillage.app",
            "http://localhost:3000"  # For testing
        ]
    else:
        # Development: Localhost only
        return [
            "http://localhost:3000",
            "http://localhost:8080",
            "http://127.0.0.1:3000", 
            "http://127.0.0.1:8080"
        ]


def get_websocket_cors_origins() -> List[str]:
    """
    Get CORS origins for WebSocket connections.
    
    Returns:
        List of allowed origins for WebSocket endpoints
    """
    # WebSockets use same origins as regular API
    return get_secure_cors_origins()


# Default secure configuration
SECURE_CORS_CONFIG = {
    "allow_origins": get_secure_cors_origins(),
    "allow_credentials": True,
    "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    "allow_headers": [
        "Accept",
        "Accept-Language", 
        "Content-Language",
        "Content-Type",
        "Authorization",
        "X-Requested-With",
        "X-API-Key"
    ],
    "expose_headers": ["X-Total-Count", "X-Rate-Limit-Remaining"],
    "max_age": 86400  # 24 hours
}

# Admin interface configuration (more restrictive)
ADMIN_CORS_CONFIG = {
    "allow_origins": get_admin_cors_origins(),
    "allow_credentials": True,
    "allow_methods": ["GET", "POST", "PUT", "DELETE"],  # No OPTIONS preflight
    "allow_headers": [
        "Content-Type",
        "Authorization",
        "X-Admin-Token"
    ],
    "max_age": 3600  # 1 hour
}

# WebSocket configuration
WEBSOCKET_CORS_CONFIG = {
    "allow_origins": get_websocket_cors_origins(),
    "allow_credentials": True
}