# Interface Specifications and Contracts

## Interface Design Principles

### Contract Design Rules
1. **Interface Segregation**: Clients should not depend on methods they don't use
2. **Dependency Inversion**: Depend on abstractions, not concretions
3. **Stable Abstractions**: Interfaces should be more stable than implementations
4. **Liskov Substitution**: Implementations must be substitutable for their interfaces

### Security Contract Rules
1. **Fail-Safe Defaults**: All security operations default to denial
2. **Input Validation**: All inputs validated at interface boundaries
3. **Output Sanitization**: All outputs sanitized before transmission
4. **Audit Trail**: All security-relevant operations logged

## Core Interface Definitions

### Authentication Module Interfaces

#### ISessionManager
```python
from abc import ABC, abstractmethod
from typing import Optional, Set, Dict, Any
from datetime import datetime, timedelta

class ISessionManager(ABC):
    """Session management interface for secure admin sessions"""
    
    @abstractmethod
    async def create_session(
        self, 
        user_id: str, 
        user_roles: Set[str], 
        client_ip: str,
        user_agent: str
    ) -> SessionToken:
        """Create a new secure session
        
        Args:
            user_id: Authenticated user identifier
            user_roles: User's assigned roles
            client_ip: Client IP address for binding
            user_agent: Client user agent for binding
            
        Returns:
            SessionToken: Secure session token with metadata
            
        Raises:
            SessionCreationError: If session cannot be created
            SecurityViolationError: If security constraints violated
        """
        pass
    
    @abstractmethod
    async def validate_session(
        self, 
        session_id: str, 
        client_ip: str,
        user_agent: str
    ) -> Optional[SessionContext]:
        """Validate existing session
        
        Args:
            session_id: Session identifier to validate
            client_ip: Client IP for consistency check
            user_agent: Client user agent for consistency check
            
        Returns:
            Optional[SessionContext]: Session context if valid, None if invalid
        """
        pass
    
    @abstractmethod
    async def destroy_session(self, session_id: str) -> bool:
        """Destroy session and cleanup resources
        
        Args:
            session_id: Session to destroy
            
        Returns:
            bool: True if session was destroyed, False if not found
        """
        pass
    
    @abstractmethod
    async def require_mfa_verification(self, session_id: str) -> bool:
        """Mark session as requiring MFA re-verification
        
        Args:
            session_id: Session to mark
            
        Returns:
            bool: True if marked successfully
        """
        pass
```

#### IAuthenticationService
```python
class IAuthenticationService(ABC):
    """Authentication service interface"""
    
    @abstractmethod
    async def authenticate_user(
        self, 
        username: str, 
        password: str,
        client_context: ClientContext
    ) -> AuthenticationResult:
        """Authenticate user credentials
        
        Args:
            username: User identifier
            password: User password
            client_context: Client request context
            
        Returns:
            AuthenticationResult: Authentication outcome with user data
            
        Raises:
            AuthenticationError: If authentication fails
            RateLimitExceededError: If too many attempts
        """
        pass
    
    @abstractmethod
    async def get_user_roles(self, user_id: str) -> Set[str]:
        """Get user's assigned roles
        
        Args:
            user_id: User identifier
            
        Returns:
            Set[str]: User's roles
        """
        pass
    
    @abstractmethod
    async def get_user_permissions(self, roles: Set[str]) -> Set[str]:
        """Resolve permissions from roles
        
        Args:
            roles: User roles
            
        Returns:
            Set[str]: Resolved permissions
        """
        pass
```

#### IMFAService
```python
class IMFAService(ABC):
    """Multi-factor authentication service interface"""
    
    @abstractmethod
    async def generate_mfa_challenge(self, user_id: str) -> MFAChallenge:
        """Generate MFA challenge for user
        
        Args:
            user_id: User requesting MFA challenge
            
        Returns:
            MFAChallenge: Challenge data and expiration
        """
        pass
    
    @abstractmethod
    async def verify_mfa_response(
        self, 
        user_id: str, 
        challenge_id: str,
        response: str
    ) -> MFAResult:
        """Verify MFA response
        
        Args:
            user_id: User providing response
            challenge_id: Challenge being responded to
            response: User's MFA response (TOTP, SMS code, etc.)
            
        Returns:
            MFAResult: Verification result with metadata
        """
        pass
```

### Security Module Interfaces

#### ISecurityContext
```python
class ISecurityContext(ABC):
    """Security context management interface"""
    
    @abstractmethod
    async def create_context(
        self,
        session: SessionContext,
        request: RequestContext
    ) -> SecurityContext:
        """Create security context from session and request
        
        Args:
            session: Validated session context
            request: HTTP request context
            
        Returns:
            SecurityContext: Complete security context
        """
        pass
    
    @abstractmethod
    async def validate_context(self, context: SecurityContext) -> ValidationResult:
        """Validate security context integrity
        
        Args:
            context: Security context to validate
            
        Returns:
            ValidationResult: Validation outcome with details
        """
        pass
    
    @abstractmethod
    async def enrich_context(
        self,
        context: SecurityContext,
        additional_data: Dict[str, Any]
    ) -> SecurityContext:
        """Enrich context with additional security data
        
        Args:
            context: Base security context
            additional_data: Additional context data
            
        Returns:
            SecurityContext: Enriched context
        """
        pass
```

#### IPolicyEngine
```python
class IPolicyEngine(ABC):
    """Security policy evaluation interface"""
    
    @abstractmethod
    async def evaluate_access_policy(
        self,
        context: SecurityContext,
        resource: str,
        action: str
    ) -> PolicyDecision:
        """Evaluate access policy for resource and action
        
        Args:
            context: Security context
            resource: Resource being accessed
            action: Action being performed
            
        Returns:
            PolicyDecision: Allow/Deny decision with reasoning
        """
        pass
    
    @abstractmethod
    async def evaluate_risk_policy(
        self,
        context: SecurityContext,
        request: RequestContext
    ) -> RiskAssessment:
        """Evaluate risk-based policies
        
        Args:
            context: Security context
            request: Request context
            
        Returns:
            RiskAssessment: Risk level and mitigation requirements
        """
        pass
```

### Middleware Module Interfaces

#### ISecurityMiddleware
```python
class ISecurityMiddleware(ABC):
    """Base security middleware interface"""
    
    @abstractmethod
    async def process_request(
        self,
        request: Request,
        context: Optional[SecurityContext]
    ) -> MiddlewareResult:
        """Process incoming request
        
        Args:
            request: HTTP request
            context: Optional security context
            
        Returns:
            MiddlewareResult: Processing result (continue, block, modify)
        """
        pass
    
    @abstractmethod
    async def process_response(
        self,
        response: Response,
        context: Optional[SecurityContext]
    ) -> Response:
        """Process outgoing response
        
        Args:
            response: HTTP response
            context: Optional security context
            
        Returns:
            Response: Modified response
        """
        pass
```

#### IMiddlewareFactory
```python
class IMiddlewareFactory(ABC):
    """Middleware factory interface"""
    
    @abstractmethod
    def create_security_headers_middleware(
        self,
        config: SecurityHeadersConfig
    ) -> ISecurityMiddleware:
        """Create security headers middleware"""
        pass
    
    @abstractmethod
    def create_localhost_guard_middleware(
        self,
        config: LocalhostGuardConfig
    ) -> ISecurityMiddleware:
        """Create localhost guard middleware"""
        pass
    
    @abstractmethod
    def create_audit_logging_middleware(
        self,
        config: AuditLoggingConfig
    ) -> ISecurityMiddleware:
        """Create audit logging middleware"""
        pass
```

### Handlers Module Interfaces

#### IRequestHandler
```python
class IRequestHandler(ABC):
    """Base request handler interface"""
    
    @abstractmethod
    async def handle(
        self,
        request: Request,
        context: SecurityContext
    ) -> Response:
        """Handle HTTP request
        
        Args:
            request: HTTP request
            context: Security context
            
        Returns:
            Response: HTTP response
            
        Raises:
            HandlerError: If request cannot be handled
            SecurityViolationError: If security constraints violated
        """
        pass
    
    @abstractmethod
    def get_required_permissions(self) -> Set[str]:
        """Get permissions required by this handler
        
        Returns:
            Set[str]: Required permissions
        """
        pass
```

#### IHandlerFactory
```python
class IHandlerFactory(ABC):
    """Handler factory with dependency injection"""
    
    @abstractmethod
    def create_login_handler(self) -> IRequestHandler:
        """Create login handler with dependencies"""
        pass
    
    @abstractmethod
    def create_mfa_handler(self) -> IRequestHandler:
        """Create MFA handler with dependencies"""
        pass
    
    @abstractmethod
    def create_system_status_handler(self) -> IRequestHandler:
        """Create system status handler with dependencies"""
        pass
```

## Data Models and DTOs

### Authentication Models

```python
@dataclass(frozen=True)
class SessionToken:
    """Secure session token"""
    session_id: str
    user_id: str
    expires_at: datetime
    created_at: datetime
    token_hash: str  # Never store raw tokens

@dataclass(frozen=True)
class SessionContext:
    """Session context data"""
    session_id: str
    user_id: str
    roles: frozenset[str]
    permissions: frozenset[str]
    client_ip: str
    user_agent: str
    mfa_verified: bool
    created_at: datetime
    last_activity: datetime

@dataclass(frozen=True)
class AuthenticationResult:
    """Authentication operation result"""
    success: bool
    user_id: Optional[str]
    user_roles: Optional[Set[str]]
    error_code: Optional[str]
    requires_mfa: bool
    lockout_remaining: Optional[timedelta]
```

### Security Models

```python
@dataclass(frozen=True)
class SecurityContext:
    """Complete security context"""
    user_id: str
    session_id: str
    roles: frozenset[str]
    permissions: frozenset[str]
    security_level: SecurityLevel
    source_ip: str
    user_agent: str
    request_id: str
    timestamp: datetime
    risk_score: float
    additional_attributes: Dict[str, Any]

@dataclass(frozen=True)
class PolicyDecision:
    """Policy evaluation decision"""
    decision: PolicyDecisionType  # ALLOW, DENY, CONDITIONAL
    resource: str
    action: str
    reasoning: List[str]
    conditions: Optional[Dict[str, Any]]
    expires_at: Optional[datetime]

@dataclass(frozen=True)
class RiskAssessment:
    """Security risk assessment"""
    risk_level: RiskLevel  # LOW, MEDIUM, HIGH, CRITICAL
    risk_score: float  # 0.0 - 1.0
    risk_factors: List[RiskFactor]
    mitigation_required: List[str]
    monitoring_required: bool
```

### Middleware Models

```python
@dataclass(frozen=True)
class MiddlewareResult:
    """Middleware processing result"""
    action: MiddlewareAction  # CONTINUE, BLOCK, REDIRECT
    modified_request: Optional[Request]
    error_response: Optional[Response]
    metadata: Dict[str, Any]

@dataclass
class AuditLogEntry:
    """Audit log entry"""
    timestamp: datetime
    request_id: str
    user_id: Optional[str]
    session_id: Optional[str]
    event_type: AuditEventType
    resource: str
    action: str
    outcome: AuditOutcome
    client_ip: str
    user_agent: str
    additional_data: Dict[str, Any]
```

## Error Handling Contracts

### Exception Hierarchy

```python
class AdminServerError(Exception):
    """Base exception for admin server errors"""
    pass

class SecurityViolationError(AdminServerError):
    """Security constraint violation"""
    def __init__(self, message: str, violation_type: SecurityViolationType):
        super().__init__(message)
        self.violation_type = violation_type

class AuthenticationError(AdminServerError):
    """Authentication failure"""
    def __init__(self, message: str, error_code: str):
        super().__init__(message)
        self.error_code = error_code

class AuthorizationError(AdminServerError):
    """Authorization failure"""
    def __init__(self, message: str, required_permissions: Set[str]):
        super().__init__(message)
        self.required_permissions = required_permissions

class SessionError(AdminServerError):
    """Session management error"""
    pass

class PolicyViolationError(AdminServerError):
    """Security policy violation"""
    def __init__(self, message: str, policy: str, context: Dict[str, Any]):
        super().__init__(message)
        self.policy = policy
        self.context = context
```

## Interface Composition Patterns

### Service Locator Pattern
```python
class ServiceLocator:
    """Service locator for dependency resolution"""
    
    def __init__(self):
        self._services: Dict[Type, Any] = {}
    
    def register(self, interface: Type[T], implementation: T):
        """Register service implementation"""
        self._services[interface] = implementation
    
    def get(self, interface: Type[T]) -> T:
        """Get service implementation"""
        return self._services.get(interface)
```

### Factory Method Pattern
```python
class SecurityServiceFactory:
    """Factory for security-related services"""
    
    @staticmethod
    def create_session_manager(config: SessionConfig) -> ISessionManager:
        """Create configured session manager"""
        return SessionManager(
            timeout=config.timeout,
            max_sessions=config.max_sessions,
            storage=config.storage_backend
        )
    
    @staticmethod
    def create_policy_engine(config: PolicyConfig) -> IPolicyEngine:
        """Create configured policy engine"""
        return PolicyEngine(
            policies=config.policies,
            risk_threshold=config.risk_threshold
        )
```

## Quality Assurance Contracts

### Testing Interfaces
```python
class ITestDouble(ABC):
    """Base interface for test doubles"""
    
    @abstractmethod
    def reset(self) -> None:
        """Reset test double state"""
        pass

class IMockSessionManager(ISessionManager, ITestDouble):
    """Mock session manager for testing"""
    pass
```

### Performance Contracts
```python
class IPerformanceMonitor(ABC):
    """Performance monitoring interface"""
    
    @abstractmethod
    async def record_operation_time(
        self,
        operation: str,
        duration: timedelta,
        context: Dict[str, Any]
    ) -> None:
        """Record operation timing"""
        pass
```

## Contract Validation

### Runtime Validation
```python
def validate_security_context(context: SecurityContext) -> bool:
    """Validate security context contract compliance"""
    return (
        context.user_id and 
        context.session_id and
        context.roles and
        context.timestamp > datetime.utcnow() - timedelta(hours=1)
    )

def validate_session_token(token: SessionToken) -> bool:
    """Validate session token contract compliance"""
    return (
        token.session_id and
        token.user_id and
        token.expires_at > datetime.utcnow() and
        len(token.token_hash) >= 32
    )
```

These interfaces provide a solid foundation for the modular architecture, ensuring loose coupling, high testability, and clear separation of concerns while maintaining strong security guarantees.