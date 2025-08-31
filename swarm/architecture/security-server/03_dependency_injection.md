# Dependency Injection Framework Design

## Dependency Injection Principles

### Core DI Principles
1. **Inversion of Control**: Dependencies flow from outside-in
2. **Dependency Inversion**: Depend on abstractions, not concretions
3. **Single Responsibility**: Each class has one reason to change
4. **Open/Closed**: Open for extension, closed for modification
5. **Composition over Inheritance**: Favor composition relationships

### Security DI Principles
1. **Secure by Default**: All injected dependencies configured securely
2. **Principle of Least Privilege**: Minimal required permissions
3. **Fail-Safe Defaults**: Security failures default to denial
4. **Auditability**: All dependency creation and injection logged

## Container Architecture

### Core Container Interface

```python
from typing import TypeVar, Type, Callable, Dict, Any, Optional
from abc import ABC, abstractmethod

T = TypeVar('T')

class IDependencyContainer(ABC):
    """Dependency injection container interface"""
    
    @abstractmethod
    def register_singleton(
        self, 
        interface: Type[T], 
        implementation: Type[T],
        *args, **kwargs
    ) -> None:
        """Register singleton service"""
        pass
    
    @abstractmethod
    def register_transient(
        self, 
        interface: Type[T], 
        implementation: Type[T]
    ) -> None:
        """Register transient service"""
        pass
    
    @abstractmethod
    def register_factory(
        self, 
        interface: Type[T], 
        factory: Callable[..., T]
    ) -> None:
        """Register factory function"""
        pass
    
    @abstractmethod
    def resolve(self, interface: Type[T]) -> T:
        """Resolve service instance"""
        pass
    
    @abstractmethod
    def configure(self, configuration: 'ContainerConfiguration') -> None:
        """Configure container with configuration"""
        pass
```

### Container Implementation

```python
from dataclasses import dataclass, field
from enum import Enum
import inspect
from typing import get_type_hints

class ServiceLifetime(Enum):
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"

@dataclass
class ServiceRegistration:
    """Service registration metadata"""
    interface: Type
    implementation: Type
    lifetime: ServiceLifetime
    factory: Optional[Callable] = None
    instance: Optional[Any] = None
    dependencies: List[Type] = field(default_factory=list)

class AdminServerContainer(IDependencyContainer):
    """Dependency injection container for admin server"""
    
    def __init__(self):
        self._registrations: Dict[Type, ServiceRegistration] = {}
        self._singletons: Dict[Type, Any] = {}
        self._audit_logger = self._create_audit_logger()
    
    def register_singleton(
        self, 
        interface: Type[T], 
        implementation: Type[T],
        *args, **kwargs
    ) -> None:
        """Register singleton service"""
        self._audit_logger.info(f"Registering singleton: {interface.__name__} -> {implementation.__name__}")
        
        dependencies = self._extract_dependencies(implementation)
        
        registration = ServiceRegistration(
            interface=interface,
            implementation=implementation,
            lifetime=ServiceLifetime.SINGLETON,
            dependencies=dependencies
        )
        
        self._registrations[interface] = registration
    
    def register_transient(
        self, 
        interface: Type[T], 
        implementation: Type[T]
    ) -> None:
        """Register transient service"""
        self._audit_logger.info(f"Registering transient: {interface.__name__} -> {implementation.__name__}")
        
        dependencies = self._extract_dependencies(implementation)
        
        registration = ServiceRegistration(
            interface=interface,
            implementation=implementation,
            lifetime=ServiceLifetime.TRANSIENT,
            dependencies=dependencies
        )
        
        self._registrations[interface] = registration
    
    def register_factory(
        self, 
        interface: Type[T], 
        factory: Callable[..., T]
    ) -> None:
        """Register factory function"""
        self._audit_logger.info(f"Registering factory for: {interface.__name__}")
        
        registration = ServiceRegistration(
            interface=interface,
            implementation=type(None),  # Placeholder
            lifetime=ServiceLifetime.SINGLETON,
            factory=factory
        )
        
        self._registrations[interface] = registration
    
    def resolve(self, interface: Type[T]) -> T:
        """Resolve service instance with dependency injection"""
        if interface not in self._registrations:
            raise DependencyResolutionError(f"Service not registered: {interface.__name__}")
        
        registration = self._registrations[interface]
        
        # Use factory if available
        if registration.factory:
            return registration.factory()
        
        # Return singleton instance if exists
        if registration.lifetime == ServiceLifetime.SINGLETON:
            if interface in self._singletons:
                return self._singletons[interface]
        
        # Create new instance with dependency injection
        instance = self._create_instance(registration)
        
        # Cache singleton
        if registration.lifetime == ServiceLifetime.SINGLETON:
            self._singletons[interface] = instance
        
        self._audit_logger.debug(f"Resolved service: {interface.__name__}")
        return instance
    
    def _create_instance(self, registration: ServiceRegistration) -> Any:
        """Create service instance with dependency injection"""
        implementation = registration.implementation
        
        # Resolve constructor dependencies
        constructor_args = {}
        constructor_signature = inspect.signature(implementation.__init__)
        
        for param_name, param in constructor_signature.parameters.items():
            if param_name == 'self':
                continue
                
            if param.annotation in self._registrations:
                constructor_args[param_name] = self.resolve(param.annotation)
            elif param.default != inspect.Parameter.empty:
                # Use default value
                pass
            else:
                raise DependencyResolutionError(
                    f"Cannot resolve dependency {param.annotation} for {implementation.__name__}"
                )
        
        return implementation(**constructor_args)
    
    def _extract_dependencies(self, implementation: Type) -> List[Type]:
        """Extract constructor dependencies from type hints"""
        try:
            type_hints = get_type_hints(implementation.__init__)
            return [hint for name, hint in type_hints.items() if name != 'return']
        except Exception:
            return []
    
    def _create_audit_logger(self):
        """Create audit logger for container operations"""
        import logging
        logger = logging.getLogger("admin_server.container")
        return logger
```

## Configuration System

### Container Configuration

```python
@dataclass
class SecurityConfiguration:
    """Security-related configuration"""
    session_timeout_minutes: int = 30
    max_sessions_per_user: int = 3
    mfa_required: bool = True
    localhost_only: bool = True
    audit_enabled: bool = True
    rate_limit_per_minute: int = 60

@dataclass
class DatabaseConfiguration:
    """Database configuration"""
    connection_string: str
    pool_size: int = 10
    timeout_seconds: int = 30

@dataclass
class ContainerConfiguration:
    """Complete container configuration"""
    security: SecurityConfiguration = field(default_factory=SecurityConfiguration)
    database: DatabaseConfiguration = field(default_factory=DatabaseConfiguration)
    environment: str = "development"
    debug_enabled: bool = False

class ConfigurationBuilder:
    """Builder for container configuration"""
    
    def __init__(self):
        self._config = ContainerConfiguration()
    
    def with_security_config(self, security: SecurityConfiguration) -> 'ConfigurationBuilder':
        """Set security configuration"""
        self._config.security = security
        return self
    
    def with_database_config(self, database: DatabaseConfiguration) -> 'ConfigurationBuilder':
        """Set database configuration"""
        self._config.database = database
        return self
    
    def for_environment(self, environment: str) -> 'ConfigurationBuilder':
        """Set environment"""
        self._config.environment = environment
        return self
    
    def build(self) -> ContainerConfiguration:
        """Build final configuration"""
        return self._config
```

## Service Registration Patterns

### Module Registration

```python
class AuthenticationModule:
    """Authentication module registration"""
    
    @staticmethod
    def register(container: IDependencyContainer, config: ContainerConfiguration):
        """Register authentication services"""
        
        # Register session manager
        container.register_singleton(
            ISessionManager,
            SessionManager,
            timeout=timedelta(minutes=config.security.session_timeout_minutes),
            max_sessions=config.security.max_sessions_per_user
        )
        
        # Register authentication service
        container.register_singleton(
            IAuthenticationService,
            CredentialValidator
        )
        
        # Register MFA service with factory
        if config.security.mfa_required:
            container.register_factory(
                IMFAService,
                lambda: TOTPMFAService(config.security)
            )
        else:
            container.register_factory(
                IMFAService,
                lambda: NoOpMFAService()
            )

class SecurityModule:
    """Security module registration"""
    
    @staticmethod
    def register(container: IDependencyContainer, config: ContainerConfiguration):
        """Register security services"""
        
        # Register security context service
        container.register_singleton(
            ISecurityContext,
            SecurityContextService
        )
        
        # Register policy engine
        container.register_singleton(
            IPolicyEngine,
            AdminPolicyEngine
        )
        
        # Register access controller
        container.register_transient(
            IAccessController,
            RoleBasedAccessController
        )

class MiddlewareModule:
    """Middleware module registration"""
    
    @staticmethod
    def register(container: IDependencyContainer, config: ContainerConfiguration):
        """Register middleware services"""
        
        # Register middleware factory
        container.register_singleton(
            IMiddlewareFactory,
            MiddlewareFactory
        )
        
        # Register individual middleware as singletons
        container.register_singleton(
            ISecurityHeadersMiddleware,
            SecurityHeadersMiddleware
        )
        
        if config.security.localhost_only:
            container.register_singleton(
                ILocalhostGuardMiddleware,
                LocalhostGuardMiddleware
            )
        
        if config.security.audit_enabled:
            container.register_singleton(
                IAuditLoggingMiddleware,
                AuditLoggingMiddleware
            )

class HandlersModule:
    """Handlers module registration"""
    
    @staticmethod
    def register(container: IDependencyContainer, config: ContainerConfiguration):
        """Register handler services"""
        
        # Register handler factory
        container.register_singleton(
            IHandlerFactory,
            HandlerFactory
        )
        
        # Register individual handlers
        container.register_transient(ILoginHandler, LoginHandler)
        container.register_transient(IMFAHandler, MFAHandler)
        container.register_transient(ISystemStatusHandler, SystemStatusHandler)
        container.register_transient(IAuditHandler, AuditHandler)
        container.register_transient(IEmergencyHandler, EmergencyHandler)
```

## Composition Root

### Application Assembly

```python
class AdminServerCompositionRoot:
    """Composition root for admin server application"""
    
    def __init__(self, config: ContainerConfiguration):
        self.config = config
        self.container = AdminServerContainer()
        self._configure_container()
    
    def _configure_container(self):
        """Configure dependency injection container"""
        
        # Register core infrastructure
        self._register_infrastructure()
        
        # Register modules
        AuthenticationModule.register(self.container, self.config)
        SecurityModule.register(self.container, self.config)
        MiddlewareModule.register(self.container, self.config)
        HandlersModule.register(self.container, self.config)
        
        # Validate registrations
        self._validate_container()
    
    def _register_infrastructure(self):
        """Register infrastructure services"""
        
        # Register database connections
        if self.config.database.connection_string:
            self.container.register_factory(
                IDatabaseConnection,
                lambda: DatabaseConnection(self.config.database.connection_string)
            )
        
        # Register logging
        self.container.register_singleton(
            ILogger,
            StructuredLogger
        )
        
        # Register metrics
        self.container.register_singleton(
            IMetricsCollector,
            PrometheusMetricsCollector
        )
    
    def create_application(self) -> 'SecureAdminServerApp':
        """Create fully configured application"""
        
        # Resolve main application components
        session_manager = self.container.resolve(ISessionManager)
        security_context = self.container.resolve(ISecurityContext)
        middleware_factory = self.container.resolve(IMiddlewareFactory)
        handler_factory = self.container.resolve(IHandlerFactory)
        
        # Create application
        app = SecureAdminServerApp(
            session_manager=session_manager,
            security_context=security_context,
            middleware_factory=middleware_factory,
            handler_factory=handler_factory,
            config=self.config
        )
        
        return app
    
    def _validate_container(self):
        """Validate container configuration"""
        required_services = [
            ISessionManager,
            IAuthenticationService,
            IMFAService,
            ISecurityContext,
            IPolicyEngine,
            IMiddlewareFactory,
            IHandlerFactory
        ]
        
        for service in required_services:
            try:
                self.container.resolve(service)
            except DependencyResolutionError as e:
                raise ContainerValidationError(f"Required service not configured: {service.__name__}") from e
```

## Scoped Dependency Management

### Request Scoped Services

```python
class RequestScope:
    """Request-scoped dependency management"""
    
    def __init__(self, container: IDependencyContainer):
        self.container = container
        self._scoped_instances: Dict[Type, Any] = {}
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup_scoped_instances()
    
    def resolve(self, interface: Type[T]) -> T:
        """Resolve service within request scope"""
        
        # Check if already resolved in this scope
        if interface in self._scoped_instances:
            return self._scoped_instances[interface]
        
        # Resolve from container
        instance = self.container.resolve(interface)
        
        # Cache in scope
        self._scoped_instances[interface] = instance
        
        return instance
    
    def _cleanup_scoped_instances(self):
        """Cleanup scoped instances"""
        for instance in self._scoped_instances.values():
            if hasattr(instance, 'dispose'):
                instance.dispose()
        
        self._scoped_instances.clear()

class RequestScopedMiddleware:
    """Middleware to create request scope"""
    
    def __init__(self, container: IDependencyContainer):
        self.container = container
    
    async def __call__(self, request: Request, call_next):
        """Process request with scoped dependencies"""
        
        with RequestScope(self.container) as scope:
            # Attach scope to request
            request.state.dependency_scope = scope
            
            # Process request
            response = await call_next(request)
            
            return response
```

## Security and Validation

### Secure Service Creation

```python
class SecureServiceFactory:
    """Factory with security validation"""
    
    def __init__(self, container: IDependencyContainer):
        self.container = container
        self.security_validator = SecurityServiceValidator()
    
    def create_secure_service(self, interface: Type[T]) -> T:
        """Create service with security validation"""
        
        # Validate service security requirements
        self.security_validator.validate_service_security(interface)
        
        # Resolve service
        service = self.container.resolve(interface)
        
        # Apply security wrapper if needed
        if self._requires_security_wrapper(interface):
            service = SecurityWrapper(service, interface)
        
        return service
    
    def _requires_security_wrapper(self, interface: Type) -> bool:
        """Check if service requires security wrapper"""
        return hasattr(interface, '__security_sensitive__')

class SecurityServiceValidator:
    """Validator for service security compliance"""
    
    def validate_service_security(self, interface: Type) -> None:
        """Validate service meets security requirements"""
        
        # Check for required security attributes
        if hasattr(interface, '__requires_authentication__'):
            if not hasattr(interface, '__authentication_method__'):
                raise SecurityValidationError(f"Service {interface.__name__} requires authentication but no method specified")
        
        # Check for audit requirements
        if hasattr(interface, '__requires_audit__'):
            if not hasattr(interface, '__audit_events__'):
                raise SecurityValidationError(f"Service {interface.__name__} requires audit but no events specified")
```

## Testing Support

### Mock Container

```python
class MockContainer(IDependencyContainer):
    """Mock container for testing"""
    
    def __init__(self):
        self._mocks: Dict[Type, Any] = {}
    
    def register_mock(self, interface: Type[T], mock: T):
        """Register mock implementation"""
        self._mocks[interface] = mock
    
    def resolve(self, interface: Type[T]) -> T:
        """Resolve mock service"""
        if interface not in self._mocks:
            raise DependencyResolutionError(f"Mock not registered: {interface.__name__}")
        
        return self._mocks[interface]
    
    # Implement other abstract methods as no-ops for testing
    def register_singleton(self, interface, implementation, *args, **kwargs): pass
    def register_transient(self, interface, implementation): pass
    def register_factory(self, interface, factory): pass
    def configure(self, configuration): pass

class ContainerTestFixture:
    """Test fixture for container testing"""
    
    def __init__(self):
        self.container = MockContainer()
        self._setup_default_mocks()
    
    def _setup_default_mocks(self):
        """Setup commonly used mocks"""
        from unittest.mock import AsyncMock, Mock
        
        # Mock authentication services
        self.container.register_mock(ISessionManager, AsyncMock())
        self.container.register_mock(IAuthenticationService, AsyncMock())
        self.container.register_mock(IMFAService, AsyncMock())
        
        # Mock security services
        self.container.register_mock(ISecurityContext, Mock())
        self.container.register_mock(IPolicyEngine, AsyncMock())
```

## Error Handling

### DI-Specific Exceptions

```python
class DependencyInjectionError(Exception):
    """Base exception for dependency injection errors"""
    pass

class DependencyResolutionError(DependencyInjectionError):
    """Service resolution error"""
    pass

class CircularDependencyError(DependencyInjectionError):
    """Circular dependency detected"""
    def __init__(self, dependency_chain: List[str]):
        self.dependency_chain = dependency_chain
        super().__init__(f"Circular dependency detected: {' -> '.join(dependency_chain)}")

class ContainerValidationError(DependencyInjectionError):
    """Container validation error"""
    pass

class SecurityValidationError(DependencyInjectionError):
    """Service security validation error"""
    pass
```

This dependency injection framework provides a robust foundation for the modular architecture, ensuring loose coupling, testability, and security compliance throughout the admin server system.