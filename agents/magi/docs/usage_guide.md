# MAGI Usage Guide

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/magi.git

# Install dependencies
pip install -r requirements.txt
```

### Basic Configuration

```python
from magi.core.config import MagiAgentConfig
from magi.core.magi_agent import MagiAgent

# Create configuration
config = MagiAgentConfig(
    name="my_magi",
    description="My MAGI instance",
    capabilities=["coding", "debugging", "analysis"],
    model="gpt-4"  # Or your preferred model
)

# Initialize agent
agent = MagiAgent(config)
```

## Common Use Cases

### 1. Code Generation

```python
# Generate a new function
result = await agent.execute_task("""
Create a function to calculate Fibonacci numbers using dynamic programming.
Requirements:
- Efficient implementation
- Type hints
- Docstring
- Error handling
""")

print(result.result)  # Generated code
```

Example output:
```python
def fibonacci(n: int) -> int:
    """Calculate nth Fibonacci number using dynamic programming.
    
    Args:
        n: Position in Fibonacci sequence (0-based)
        
    Returns:
        nth Fibonacci number
        
    Raises:
        ValueError: If n is negative
    """
    if n < 0:
        raise ValueError("Position must be non-negative")
        
    if n <= 1:
        return n
        
    fib = [0] * (n + 1)
    fib[1] = 1
    
    for i in range(2, n + 1):
        fib[i] = fib[i-1] + fib[i-2]
        
    return fib[n]
```

### 2. Code Analysis

```python
# Analyze existing code
result = await agent.execute_task("""
Analyze this code for potential improvements:

def process_data(data):
    results = []
    for item in data:
        if item > 0:
            results.append(item * 2)
    return results
""")

print(result.result)  # Analysis and suggestions
```

Example output:
```
Analysis:
1. Missing type hints
2. No input validation
3. Could use list comprehension
4. No docstring

Improved version:
def process_data(data: List[float]) -> List[float]:
    """Process data by doubling positive values.
    
    Args:
        data: List of numbers to process
        
    Returns:
        List of processed values
    """
    return [item * 2 for item in data if item > 0]
```

### 3. Problem Solving

```python
# Solve complex problem
result = await agent.execute_task("""
Design a caching system with:
- LRU eviction
- Size limit
- Thread safety
- Persistence
""")

print(result.result)  # Solution design and implementation
```

### 4. Tool Creation

```python
from magi.tools.tool_creator import ToolCreator

# Create custom tool
creator = ToolCreator()
tool = await creator.create_tool(
    name="data_processor",
    description="Process data with custom logic",
    parameters={
        "data": "List[float]",
        "threshold": "float"
    },
    examples=[
        ([1.0, 2.0, 3.0], 2.0, [3.0])  # Input1, Input2, Output
    ]
)

# Use created tool
result = await tool.execute([1.5, 2.5, 3.5], 2.0)
```

## Advanced Usage

### 1. Custom Techniques

```python
from magi.techniques.base import BaseTechnique

class CustomTechnique(BaseTechnique):
    """Custom reasoning technique."""
    
    @property
    def name(self) -> str:
        return "Custom-Technique"
    
    async def apply(
        self,
        agent: MagiAgent,
        task: str,
        **kwargs
    ) -> TechniqueResult:
        # Implement custom logic
        pass

# Register technique
agent.register_technique(CustomTechnique())
```

### 2. Feedback Integration

```python
from magi.feedback.analysis import FeedbackAnalyzer
from magi.feedback.improvement import ImprovementManager

# Set up feedback system
analyzer = FeedbackAnalyzer()
improvement_manager = ImprovementManager(analyzer)

# Execute task with feedback
result = await agent.execute_task("Implement feature")
analyzer.record_execution(result)

# Generate improvements
improvements = await improvement_manager.generate_improvement_plans()
await improvement_manager.implement_improvements(improvements)
```

### 3. Database Integration

```python
from magi.integrations.database import DatabaseManager

# Set up database
db_manager = DatabaseManager()

# Store tool data
await db_manager.store_tool({
    "name": "my_tool",
    "version": "1.0.0",
    "code": "...",
    "parameters": {}
})

# Retrieve tool
tool_data = await db_manager.get_tool("my_tool")
```

### 4. API Integration

```python
from magi.integrations.api import APIManager

# Set up API manager
api_manager = APIManager()

# Make API requests
async with api_manager:
    # GET request
    data = await api_manager.get(
        "github",
        "repos/user/repo"
    )
    
    # POST request
    result = await api_manager.post(
        "api",
        "endpoint",
        {"data": "value"}
    )
```

## Best Practices

### 1. Task Description

Write clear, specific task descriptions:

```python
# Good
result = await agent.execute_task("""
Create a function to parse CSV files with these requirements:
- Handle quoted fields
- Skip empty lines
- Custom delimiter support
- Error handling for malformed lines
- Return list of dictionaries
""")

# Not as good
result = await agent.execute_task(
    "Write code to parse CSV files"
)
```

### 2. Error Handling

Always handle potential errors:

```python
try:
    result = await agent.execute_task("Complex task")
except ExecutionError as e:
    logger.error(f"Execution failed: {e}")
    # Handle error
except TimeoutError as e:
    logger.error(f"Execution timed out: {e}")
    # Handle timeout
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    # Handle other errors
```

### 3. Resource Management

Monitor and manage resources:

```python
# Get current usage
usage = agent.get_resource_usage()

# Set limits
agent.set_resource_limits(
    memory_limit=1024 * 1024 * 1024,  # 1GB
    cpu_limit=80  # 80% CPU
)

# Clean up
await agent.cleanup()
```

### 4. Performance Optimization

```python
# Use caching
agent.enable_caching()

# Batch similar tasks
tasks = [
    "Task 1",
    "Task 2",
    "Task 3"
]
results = await asyncio.gather(*[
    agent.execute_task(task)
    for task in tasks
])

# Profile execution
with agent.profile():
    result = await agent.execute_task("Task")
```

## Common Patterns

### 1. Iterative Development

```python
# Initial implementation
result = await agent.execute_task("Implement basic version")

# Analyze and improve
analysis = await agent.execute_task(f"""
Analyze this code for improvements:
{result.result}
""")

# Implement improvements
final = await agent.execute_task(f"""
Improve the code based on this analysis:
{analysis.result}
""")
```

### 2. Test Generation

```python
# Generate implementation
impl = await agent.execute_task("Implement feature")

# Generate tests
tests = await agent.execute_task(f"""
Create comprehensive tests for this implementation:
{impl.result}
""")

# Verify implementation
verification = await agent.execute_task(f"""
Verify the implementation against these tests:
Implementation:
{impl.result}

Tests:
{tests.result}
""")
```

### 3. Documentation Generation

```python
# Generate implementation
impl = await agent.execute_task("Implement feature")

# Generate documentation
docs = await agent.execute_task(f"""
Create comprehensive documentation for this code:
{impl.result}

Include:
- Overview
- API reference
- Examples
- Best practices
""")
```

## Troubleshooting

### 1. Common Issues

- **High Memory Usage**: Enable resource limits and cleanup
- **Slow Execution**: Use technique caching and batching
- **Poor Results**: Provide more detailed task descriptions
- **Integration Errors**: Check configuration and connectivity

### 2. Debugging

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable execution tracing
agent.enable_tracing()

# Monitor execution
agent.add_execution_callback(lambda e: print(f"Event: {e}"))
```

### 3. Performance Monitoring

```python
# Monitor metrics
metrics = agent.get_performance_metrics()
print(f"Average response time: {metrics.avg_response_time}")
print(f"Success rate: {metrics.success_rate}")

# Profile execution
profile = agent.get_execution_profile()
print(f"Hot spots: {profile.hot_spots}")
```

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines on:
- Code style
- Testing requirements
- Documentation standards
- Pull request process
