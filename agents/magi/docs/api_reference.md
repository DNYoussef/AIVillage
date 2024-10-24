# MAGI API Reference

## Core Module

### MagiAgent

```python
class MagiAgent:
    def __init__(
        self,
        config: MagiAgentConfig,
        communication_protocol: Optional[CommunicationProtocol] = None,
        rag_config: Optional[RAGConfig] = None,
        vector_store: Optional[VectorStore] = None
    ):
        """Initialize MAGI agent.
        
        Args:
            config: Agent configuration
            communication_protocol: Optional communication protocol
            rag_config: Optional RAG system configuration
            vector_store: Optional vector store for embeddings
        """

    async def execute_task(
        self,
        task: str,
        timeout: Optional[float] = None,
        **kwargs
    ) -> TaskResult:
        """Execute a task using appropriate techniques.
        
        Args:
            task: Task description
            timeout: Optional execution timeout
            **kwargs: Additional task parameters
            
        Returns:
            TaskResult containing execution results
            
        Raises:
            ExecutionError: If task execution fails
            TimeoutError: If execution exceeds timeout
        """

    def get_state(self) -> Dict[str, Any]:
        """Get current agent state.
        
        Returns:
            Dictionary containing agent state
        """

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load agent state.
        
        Args:
            state: State dictionary to load
        """
```

### Config

```python
class MagiAgentConfig:
    """Agent configuration."""
    name: str
    description: str
    capabilities: List[str]
    development_capabilities: Optional[List[str]]
    model: str
    instructions: Optional[str]
```

## Techniques Module

### Base Technique

```python
class BaseTechnique:
    """Base class for reasoning techniques."""
    
    @property
    def name(self) -> str:
        """Technique name."""
        
    @property
    def description(self) -> str:
        """Technique description."""
        
    async def apply(
        self,
        agent: MagiAgent,
        task: str,
        **kwargs
    ) -> TechniqueResult:
        """Apply technique to task.
        
        Args:
            agent: MAGI agent instance
            task: Task description
            **kwargs: Additional parameters
            
        Returns:
            TechniqueResult containing results
        """
```

### Technique Registry

```python
class TechniqueRegistry:
    """Registry for reasoning techniques."""
    
    def register(
        self,
        name: str,
        technique: BaseTechnique
    ) -> None:
        """Register a technique.
        
        Args:
            name: Technique name
            technique: Technique instance
        """
        
    def get(self, name: str) -> Optional[BaseTechnique]:
        """Get technique by name.
        
        Args:
            name: Technique name
            
        Returns:
            Technique instance if found, None otherwise
        """
```

## Tools Module

### Tool Creator

```python
class ToolCreator:
    """Dynamic tool creation system."""
    
    async def create_tool(
        self,
        name: str,
        description: str,
        parameters: Dict[str, str],
        code: Optional[str] = None,
        examples: Optional[List[Tuple]] = None
    ) -> Tool:
        """Create a new tool.
        
        Args:
            name: Tool name
            description: Tool description
            parameters: Tool parameters
            code: Optional implementation code
            examples: Optional usage examples
            
        Returns:
            Created tool instance
            
        Raises:
            ToolError: If tool creation fails
        """
```

### Tool Manager

```python
class ToolManager:
    """Tool management system."""
    
    def register_tool(self, tool: Tool) -> None:
        """Register a tool.
        
        Args:
            tool: Tool instance
        """
        
    def get_tool(
        self,
        name: str,
        version: Optional[str] = None
    ) -> Optional[Tool]:
        """Get tool by name and version.
        
        Args:
            name: Tool name
            version: Optional version
            
        Returns:
            Tool instance if found
        """
```

## Utils Module

### Helpers

```python
def format_error_message(
    error: Exception,
    context: Dict[str, Any]
) -> str:
    """Format error message with context.
    
    Args:
        error: Exception instance
        context: Error context
        
    Returns:
        Formatted error message
    """

def validate_input(
    value: Any,
    expected_type: Type,
    **kwargs
) -> bool:
    """Validate input value.
    
    Args:
        value: Input value
        expected_type: Expected type
        **kwargs: Validation parameters
        
    Returns:
        True if valid, False otherwise
    """
```

### Logging

```python
def setup_logger(
    name: str,
    level: Optional[int] = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    """Set up logger instance.
    
    Args:
        name: Logger name
        level: Optional log level
        log_file: Optional log file path
        
    Returns:
        Configured logger instance
    """
```

## Feedback Module

### Analysis

```python
class FeedbackAnalyzer:
    """Performance analysis system."""
    
    def analyze_technique_performance(
        self,
        technique_name: str
    ) -> TechniqueMetrics:
        """Analyze technique performance.
        
        Args:
            technique_name: Technique name
            
        Returns:
            Performance metrics
        """
        
    def analyze_system_performance(self) -> SystemMetrics:
        """Analyze overall system performance.
        
        Returns:
            System metrics
        """
```

### Improvement

```python
class ImprovementManager:
    """System improvement manager."""
    
    async def generate_improvement_plans(self) -> List[ImprovementPlan]:
        """Generate improvement plans.
        
        Returns:
            List of improvement plans
        """
        
    async def implement_improvements(
        self,
        plans: List[ImprovementPlan]
    ) -> List[ImprovementResult]:
        """Implement improvement plans.
        
        Args:
            plans: List of plans to implement
            
        Returns:
            Implementation results
        """
```

## Integrations Module

### Database

```python
class DatabaseManager:
    """Database integration manager."""
    
    async def store_tool(self, tool_data: Dict[str, Any]) -> None:
        """Store tool data.
        
        Args:
            tool_data: Tool data to store
        """
        
    async def get_tool(
        self,
        name: str,
        version: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get tool data.
        
        Args:
            name: Tool name
            version: Optional version
            
        Returns:
            Tool data if found
        """
```

### API

```python
class APIManager:
    """API integration manager."""
    
    async def get(
        self,
        api_name: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make GET request.
        
        Args:
            api_name: API name
            endpoint: API endpoint
            params: Optional parameters
            
        Returns:
            Response data
            
        Raises:
            APIError: If request fails
        """
        
    async def post(
        self,
        api_name: str,
        endpoint: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Make POST request.
        
        Args:
            api_name: API name
            endpoint: API endpoint
            data: Request data
            
        Returns:
            Response data
            
        Raises:
            APIError: If request fails
        """
```

## Data Types

### Results

```python
class TaskResult:
    """Task execution result."""
    result: Any
    confidence: float
    technique_results: List[TechniqueResult]
    execution_time: float
    resources_used: ResourceUsage

class TechniqueResult:
    """Technique execution result."""
    technique: str
    result: Any
    confidence: float
    steps: List[ReasoningStep]
    execution_time: float

class ImprovementResult:
    """Improvement implementation result."""
    plan: ImprovementPlan
    success: bool
    changes_made: List[str]
    metrics_before: Dict[str, float]
    metrics_after: Dict[str, float]
```

### Metrics

```python
class TechniqueMetrics:
    """Technique performance metrics."""
    success_rate: float
    average_execution_time: float
    average_confidence: float
    common_patterns: List[str]

class SystemMetrics:
    """System performance metrics."""
    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    average_response_time: float
    peak_memory_usage: int
    active_techniques: List[str]
    active_tools: List[str]
```

### Plans

```python
class ImprovementPlan:
    """System improvement plan."""
    target: str
    target_name: Optional[str]
    improvements: List[Dict[str, Any]]
    priority: int
    estimated_impact: float
    dependencies: List[str]
    implementation_steps: List[str]
```

## Exceptions

```python
class MagiError(Exception):
    """Base exception for MAGI system."""

class ConfigurationError(MagiError):
    """Configuration error."""

class ExecutionError(MagiError):
    """Task execution error."""

class ValidationError(MagiError):
    """Input validation error."""

class ToolError(MagiError):
    """Tool-related error."""

class APIError(MagiError):
    """API integration error."""

class DatabaseError(MagiError):
    """Database integration error."""
