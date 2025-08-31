"""Authentication Module.

Provides comprehensive authentication, session management, and MFA services.
This module was extracted from the EnhancedSecureAPIServer God class to follow
Single Responsibility Principle and enable better testing and maintainability.

## Architecture

The authentication module follows clean architecture principles with clear
separation of concerns:

- **Interfaces**: Define contracts for all services
- **Services**: Core business logic for auth, sessions, and MFA  
- **Handlers**: HTTP request/response handling
- **Container**: Dependency injection and service lifecycle management

## Usage

```python
from infrastructure.security.auth import AuthContainer

# Initialize container
container = AuthContainer(config)
await container.initialize(rbac_system, encryption_service)

# Register routes
container.register_routes(app, "/auth")

# Use services
auth_service = container.get_auth_service()
session_manager = container.get_session_manager()
mfa_service = container.get_mfa_service()
```

## Services

- **AuthenticationService**: JWT token management, user authentication
- **SessionService**: Redis-backed session management with fallback
- **MFAService**: TOTP, SMS, email, and backup code authentication
"""

from .interfaces import (
    IAuthenticationService,
    ISessionManager, 
    IMFAService,
    AuthCredentials,
    AuthResult,
    TokenValidationResult,
    SessionData,
    DeviceInfo,
    MFASetupResult,
    MFAStatus,
    MFAMethodType
)

from .services import (
    AuthenticationService,
    SessionService,
    MFAService
)

from .handlers import (
    AuthHandlers,
    MFAHandlers,
    SessionHandlers
)

from .container import AuthContainer

# Version info
__version__ = "1.0.0"
__author__ = "AIVillage Security Team"

# Public API
__all__ = [
    # Main container
    "AuthContainer",
    
    # Interfaces
    "IAuthenticationService",
    "ISessionManager",
    "IMFAService",
    
    # Services
    "AuthenticationService", 
    "SessionService",
    "MFAService",
    
    # Handlers
    "AuthHandlers",
    "MFAHandlers", 
    "SessionHandlers",
    
    # Data classes and types
    "AuthCredentials",
    "AuthResult",
    "TokenValidationResult",
    "SessionData",
    "DeviceInfo",
    "MFASetupResult",
    "MFAStatus",
    "MFAMethodType"
]