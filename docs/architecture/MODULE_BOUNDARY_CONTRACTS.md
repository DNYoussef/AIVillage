# Module Boundary Contracts

This document defines the interface contracts that govern communication between layers in the AIVillage clean architecture.

## Core Principles

1. **Interface-First Design**: All cross-layer communication through well-defined interfaces
2. **Dependency Inversion**: High-level modules don't depend on low-level modules; both depend on abstractions
3. **Single Responsibility**: Each interface has one clear purpose
4. **Connascence Management**: Minimize coupling through weak connascence forms

## Layer Interface Contracts

### 1. Core Layer Interfaces

#### Agent Interface Contract
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, AsyncGenerator
from dataclasses import dataclass
from enum import Enum

class AgentCapability(Enum):
    REASONING = "reasoning"
    CREATIVITY = "creativity"
    KNOWLEDGE = "knowledge"
    GOVERNANCE = "governance"
    INFRASTRUCTURE = "infrastructure"

@dataclass
class AgentRequest:
    """Standard agent request format"""
    id: str
    type: str
    payload: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class AgentResponse:
    """Standard agent response format"""
    request_id: str
    status: str  # "success", "error", "partial"
    result: Any
    metadata: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

class IAgent(ABC):
    """Core agent interface - all agents must implement"""

    @abstractmethod
    async def process_request(self, request: AgentRequest) -> AgentResponse:
        """Process incoming request with standardized format"""
        pass

    @abstractmethod
    async def get_capabilities(self) -> List[AgentCapability]:
        """Return agent capabilities"""
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Health check with detailed status"""
        pass

    @abstractmethod
    async def get_metrics(self) -> Dict[str, Any]:
        """Performance and usage metrics"""
        pass

class IAgentCoordinator(ABC):
    """Agent coordination interface"""

    @abstractmethod
    async def route_request(self, request: AgentRequest) -> str:
        """Route request to appropriate agent, return agent_id"""
        pass

    @abstractmethod
    async def orchestrate_workflow(self, workflow: Dict[str, Any]) -> AsyncGenerator[AgentResponse, None]:
        """Orchestrate multi-agent workflow"""
        pass

    @abstractmethod
    async def get_agent_registry(self) -> Dict[str, IAgent]:
        """Get available agents"""
        pass
```

#### RAG System Interface
```python
@dataclass
class Document:
    """Document representation"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

@dataclass
class RetrievalRequest:
    """RAG retrieval request"""
    query: str
    filters: Optional[Dict[str, Any]] = None
    max_results: int = 10
    threshold: float = 0.7

@dataclass
class RetrievalResult:
    """RAG retrieval result"""
    documents: List[Document]
    scores: List[float]
    total_found: int
    query_embedding: List[float]

class IIngestionService(ABC):
    """Document ingestion interface"""

    @abstractmethod
    async def ingest_document(self, document: Document) -> str:
        """Ingest single document, return document_id"""
        pass

    @abstractmethod
    async def ingest_batch(self, documents: List[Document]) -> List[str]:
        """Ingest multiple documents"""
        pass

    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        """Delete document from index"""
        pass

class IRetrievalService(ABC):
    """Information retrieval interface"""

    @abstractmethod
    async def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """Retrieve relevant documents"""
        pass

    @abstractmethod
    async def get_similar_documents(self, document_id: str, limit: int = 10) -> List[Document]:
        """Find similar documents"""
        pass

class IGenerationService(ABC):
    """Response generation interface"""

    @abstractmethod
    async def generate_response(self, query: str, context: List[Document]) -> str:
        """Generate response using retrieved context"""
        pass

    @abstractmethod
    async def synthesize_information(self, documents: List[Document]) -> str:
        """Synthesize information from multiple documents"""
        pass
```

#### Domain Services Interface
```python
class ITokenomicsService(ABC):
    """Tokenomics and governance interface"""

    @abstractmethod
    async def calculate_rewards(self, contribution: Dict[str, Any]) -> float:
        """Calculate rewards for contribution"""
        pass

    @abstractmethod
    async def process_governance_proposal(self, proposal: Dict[str, Any]) -> str:
        """Process governance proposal, return proposal_id"""
        pass

    @abstractmethod
    async def execute_vote(self, vote: Dict[str, Any]) -> bool:
        """Execute governance vote"""
        pass

class IIdentityService(ABC):
    """Identity management interface"""

    @abstractmethod
    async def authenticate_user(self, credentials: Dict[str, Any]) -> Optional[str]:
        """Authenticate user, return user_id if successful"""
        pass

    @abstractmethod
    async def authorize_action(self, user_id: str, action: str, resource: str) -> bool:
        """Check if user is authorized for action"""
        pass
```

### 2. Infrastructure Layer Interfaces

#### Gateway Interface
```python
class IAPIGateway(ABC):
    """API Gateway interface"""

    @abstractmethod
    async def register_route(self, route: str, handler: Any) -> None:
        """Register API route with handler"""
        pass

    @abstractmethod
    async def apply_rate_limit(self, user_id: str, endpoint: str) -> bool:
        """Apply rate limiting rules"""
        pass

    @abstractmethod
    async def authenticate_request(self, request: Any) -> Optional[str]:
        """Authenticate incoming request"""
        pass

class ILoadBalancer(ABC):
    """Load balancing interface"""

    @abstractmethod
    async def route_request(self, request: Any) -> str:
        """Route request to available service instance"""
        pass

    @abstractmethod
    async def health_check_services(self) -> Dict[str, bool]:
        """Check health of all service instances"""
        pass
```

#### P2P Network Interface
```python
class IP2PNetwork(ABC):
    """P2P Network interface"""

    @abstractmethod
    async def join_network(self, node_config: Dict[str, Any]) -> str:
        """Join P2P network, return node_id"""
        pass

    @abstractmethod
    async def send_message(self, target_node: str, message: Dict[str, Any]) -> bool:
        """Send message to target node"""
        pass

    @abstractmethod
    async def broadcast_message(self, message: Dict[str, Any]) -> int:
        """Broadcast message to network, return delivery count"""
        pass

    @abstractmethod
    async def discover_peers(self, capability: str) -> List[str]:
        """Discover peers with specific capability"""
        pass

class IMessagingService(ABC):
    """Messaging service interface"""

    @abstractmethod
    async def publish(self, topic: str, message: Dict[str, Any]) -> None:
        """Publish message to topic"""
        pass

    @abstractmethod
    async def subscribe(self, topic: str, handler: Any) -> str:
        """Subscribe to topic, return subscription_id"""
        pass

    @abstractmethod
    async def create_queue(self, queue_name: str, config: Dict[str, Any]) -> None:
        """Create message queue"""
        pass
```

#### Data Services Interface
```python
class IDataStorage(ABC):
    """Data storage interface"""

    @abstractmethod
    async def store(self, key: str, data: Any, ttl: Optional[int] = None) -> bool:
        """Store data with optional TTL"""
        pass

    @abstractmethod
    async def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve data by key"""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete data by key"""
        pass

    @abstractmethod
    async def query(self, filters: Dict[str, Any]) -> List[Any]:
        """Query data with filters"""
        pass

class ICacheService(ABC):
    """Caching service interface"""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int) -> bool:
        """Set cached value with TTL"""
        pass

    @abstractmethod
    async def invalidate(self, pattern: str) -> int:
        """Invalidate cached values matching pattern"""
        pass
```

### 3. Apps Layer Interfaces

#### UI Service Interface
```python
class IUIService(ABC):
    """UI service interface for backend communication"""

    @abstractmethod
    async def render_component(self, component: str, props: Dict[str, Any]) -> str:
        """Render UI component with props"""
        pass

    @abstractmethod
    async def handle_user_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Handle user interaction"""
        pass

    @abstractmethod
    async def get_ui_state(self, user_id: str) -> Dict[str, Any]:
        """Get current UI state for user"""
        pass

class IMobileService(ABC):
    """Mobile-specific service interface"""

    @abstractmethod
    async def optimize_for_device(self, content: Any, device_info: Dict[str, Any]) -> Any:
        """Optimize content for mobile device"""
        pass

    @abstractmethod
    async def handle_offline_sync(self, user_id: str) -> Dict[str, Any]:
        """Handle offline synchronization"""
        pass
```

### 4. Cross-Layer Communication Patterns

#### Event-Driven Communication
```python
@dataclass
class SystemEvent:
    """Standard system event format"""
    id: str
    type: str
    source: str
    timestamp: float
    payload: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

class IEventBus(ABC):
    """Event bus for cross-layer communication"""

    @abstractmethod
    async def publish_event(self, event: SystemEvent) -> None:
        """Publish system event"""
        pass

    @abstractmethod
    async def subscribe_to_events(self, event_types: List[str], handler: Any) -> str:
        """Subscribe to event types, return subscription_id"""
        pass

    @abstractmethod
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events"""
        pass
```

#### Configuration Interface
```python
class IConfiguration(ABC):
    """Configuration management interface"""

    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        pass

    @abstractmethod
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get configuration section"""
        pass

    @abstractmethod
    def validate_config(self) -> List[str]:
        """Validate configuration, return list of errors"""
        pass
```

## Implementation Guidelines

### 1. Interface Implementation
```python
# Example: Core agent implementation
from core.agents.contracts import IAgent, AgentRequest, AgentResponse

class KingAgent(IAgent):
    """King agent implementation"""

    def __init__(self, config: IConfiguration):
        self.config = config
        # Initialize dependencies through dependency injection

    async def process_request(self, request: AgentRequest) -> AgentResponse:
        """Implement agent-specific logic"""
        try:
            # Process request
            result = await self._handle_governance_request(request)
            return AgentResponse(
                request_id=request.id,
                status="success",
                result=result
            )
        except Exception as e:
            return AgentResponse(
                request_id=request.id,
                status="error",
                result=None,
                error_message=str(e)
            )
```

### 2. Dependency Injection
```python
# Dependency injection container
class DIContainer:
    """Simple dependency injection container"""

    def __init__(self):
        self._services = {}
        self._singletons = {}

    def register(self, interface: type, implementation: type, singleton: bool = False):
        """Register service implementation"""
        self._services[interface] = (implementation, singleton)

    def get(self, interface: type):
        """Get service instance"""
        if interface in self._singletons:
            return self._singletons[interface]

        if interface not in self._services:
            raise ValueError(f"Service {interface} not registered")

        implementation, is_singleton = self._services[interface]
        instance = implementation()

        if is_singleton:
            self._singletons[interface] = instance

        return instance

# Usage in main application
def setup_dependencies():
    container = DIContainer()

    # Register core services
    container.register(IAgent, KingAgent, singleton=True)
    container.register(IRAGService, HyperRAGService, singleton=True)

    # Register infrastructure services
    container.register(IP2PNetwork, BetaNetService, singleton=True)
    container.register(IDataStorage, PostgresStorage, singleton=True)

    return container
```

### 3. Error Handling
```python
class ArchitecturalError(Exception):
    """Base exception for architectural violations"""
    pass

class LayerViolationError(ArchitecturalError):
    """Raised when layer boundaries are violated"""
    pass

class InterfaceViolationError(ArchitecturalError):
    """Raised when interface contracts are violated"""
    pass

# Usage in layer boundary validation
def validate_import(import_path: str, current_layer: str) -> None:
    """Validate that import doesn't violate layer boundaries"""
    forbidden_imports = LAYER_RULES[current_layer].get('forbidden_dependencies', [])

    for forbidden in forbidden_imports:
        if import_path.startswith(forbidden):
            raise LayerViolationError(
                f"Layer '{current_layer}' cannot import from '{forbidden}'"
            )
```

## Validation and Testing

### 1. Interface Compliance Testing
```python
import pytest
from abc import ABC

def test_interface_compliance():
    """Test that all implementations comply with interfaces"""

    # Get all interface implementations
    implementations = discover_implementations()

    for interface, implementation in implementations.items():
        # Verify implementation follows interface
        assert issubclass(implementation, interface)

        # Verify all abstract methods are implemented
        abstract_methods = get_abstract_methods(interface)
        implementation_methods = get_methods(implementation)

        for method in abstract_methods:
            assert method in implementation_methods
```

### 2. Layer Boundary Testing
```python
def test_layer_boundaries():
    """Test that no layer boundary violations exist"""

    violations = []

    for file_path in get_all_python_files():
        layer = determine_layer(file_path)
        imports = extract_imports(file_path)

        for import_path in imports:
            if violates_layer_rules(layer, import_path):
                violations.append((file_path, import_path))

    assert len(violations) == 0, f"Layer violations found: {violations}"
```

### 3. Connascence Validation
```python
def test_connascence_compliance():
    """Test connascence rules are followed"""

    violations = []

    # Check position connascence
    for function in get_all_functions():
        if get_parameter_count(function) > 3:
            violations.append(f"Position connascence: {function} has >3 parameters")

    # Check algorithm connascence
    duplicate_algorithms = find_duplicate_algorithms()
    for algorithm in duplicate_algorithms:
        violations.append(f"Algorithm connascence: {algorithm} duplicated")

    assert len(violations) == 0, f"Connascence violations: {violations}"
```

## Documentation Requirements

### 1. Interface Documentation
- Each interface must have comprehensive docstrings
- Include usage examples and error handling
- Document performance characteristics
- Specify thread safety guarantees

### 2. Implementation Documentation
- Document design decisions and trade-offs
- Include architecture decision records (ADRs)
- Maintain change logs for interface modifications
- Provide migration guides for breaking changes

### 3. Integration Documentation
- Document cross-layer integration patterns
- Provide troubleshooting guides
- Include performance tuning guidelines
- Maintain service dependency maps

This contract system ensures that the AIVillage clean architecture maintains clear boundaries while enabling flexible, testable, and maintainable code across all layers.
