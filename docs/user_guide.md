# AI Village User Guide

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/AIVillage.git
cd AIVillage

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

1. Set up your OpenRouter API key:
```bash
# On Linux/Mac
export OPENROUTER_API_KEY=your_api_key

# On Windows
set OPENROUTER_API_KEY=your_api_key
```

2. Configure agents in `config/openrouter_agents.yaml` if needed (default configuration should work out of the box)

### 3. Basic Usage

```python
from agent_forge.main import AIVillage

# Initialize the system
village = AIVillage()

# Process a task
import asyncio
result = asyncio.run(village.process_task(
    task="Analyze the impact of AI on healthcare",
    agent_type="sage"  # Optional, system can determine automatically
))

print(result["response"])
```

## Understanding the Agents

### King Agent
- **Purpose**: Complex problem-solving and strategic thinking
- **Best for**: 
  - Strategic planning
  - Decision making
  - Complex analysis
  - Multi-step tasks
- **Example tasks**:
  ```python
  "Develop a strategic plan for AI implementation in healthcare"
  "Analyze the pros and cons of remote work policies"
  "Create a roadmap for digital transformation"
  ```

### Sage Agent
- **Purpose**: Research, analysis, and knowledge synthesis
- **Best for**:
  - Research questions
  - Data analysis
  - Literature review
  - Trend analysis
- **Example tasks**:
  ```python
  "Research the latest developments in quantum computing"
  "Analyze the impact of climate change on agriculture"
  "Synthesize current AI safety research"
  ```

### Magi Agent
- **Purpose**: Code generation and technical problem-solving
- **Best for**:
  - Code writing
  - Algorithm design
  - Technical documentation
  - Debugging
- **Example tasks**:
  ```python
  "Write a Python function to implement merge sort"
  "Create a React component for a login form"
  "Debug this JavaScript code snippet"
  ```

## Advanced Usage

### Batch Processing

```python
async def process_batch():
    # Add multiple tasks
    tasks = [
        "Task 1",
        "Task 2",
        "Task 3"
    ]
    
    for task in tasks:
        await village.add_task(task)
    
    # Process all tasks
    await village.task_queue.join()

# Run batch processing
asyncio.run(process_batch())
```

### Custom Task Parameters

```python
# For research tasks
result = await village.process_task(
    task="Research quantum computing",
    agent_type="sage",
    context="Focus on recent developments",
    depth="deep"  # "quick", "standard", or "deep"
)

# For coding tasks
result = await village.process_task(
    task="Write a sorting function",
    agent_type="magi",
    language="python",
    requirements=["Must be efficient", "Include documentation"]
)
```

### System Monitoring

```python
# Get system status
status = village.get_system_status()
print(status)

# Example output:
{
    "queue_size": 2,
    "agent_metrics": {
        "king": {...},
        "sage": {...},
        "magi": {...}
    },
    "complexity_thresholds": {...},
    "training_data_counts": {...}
}
```

## Performance Optimization

### Model Selection

The system automatically chooses between frontier and local models based on:
1. Task complexity
2. Local model performance
3. Resource availability

You can influence this by:

```python
# Force use of frontier model
result = await village.process_task(
    task="Your task",
    force_frontier=True  # Optional parameter
)

# Force use of local model
result = await village.process_task(
    task="Your task",
    force_local=True  # Optional parameter
)
```

### Resource Management

Monitor and manage system resources:

```python
from agent_forge.utils.logging_setup import LogAnalyzer

# Initialize analyzer
analyzer = LogAnalyzer("logs")

# Get performance metrics
metrics = analyzer.analyze_performance(days=7)
print(metrics)
```

## Troubleshooting

### Common Issues

1. **API Rate Limits**
   ```python
   # System automatically handles rate limiting, but you can adjust:
   village.agent_manager.openrouter_agents["king"].requests_per_minute = 30
   ```

2. **Memory Usage**
   ```python
   # Use smaller local models
   # In config/openrouter_agents.yaml:
   agents:
     king:
       local_model: "smaller-model-name"
   ```

3. **Slow Response Times**
   ```python
   # Adjust complexity thresholds
   village.complexity_evaluator.thresholds["king"]["base_complexity_threshold"] = 0.7
   ```

### Error Handling

The system provides detailed error logging:

```python
from agent_forge.utils.logging_setup import log_error

try:
    result = await village.process_task(...)
except Exception as e:
    log_error(e, context={"task": "task description"})
```

## Best Practices

1. **Task Formulation**
   - Be specific and clear
   - Provide relevant context
   - Specify constraints if any

2. **Resource Usage**
   - Monitor system metrics
   - Use batch processing for multiple tasks
   - Clean up old logs regularly

3. **Performance Optimization**
   - Let the system learn from interactions
   - Monitor local model performance
   - Adjust thresholds as needed

4. **Data Management**
   - Export data periodically
   - Monitor training data quality
   - Review performance metrics

## System Maintenance

### Regular Tasks

1. **Log Management**
```python
from agent_forge.utils.logging_setup import compress_old_logs

# Compress logs older than 30 days
compress_old_logs("logs", days_old=30)
```

2. **Data Export**
```python
# Export collected data
export_paths = village.data_collector.export_data()
print(f"Data exported to: {export_paths}")
```

3. **Performance Analysis**
```python
# Analyze system performance
analysis = analyzer.analyze_performance(days=7)
print(analysis)
```

## Getting Help

1. Check the logs:
   ```bash
   tail -f logs/ai_village.log
   ```

2. Review error logs:
   ```bash
   tail -f logs/error.log
   ```

3. Check system status:
   ```python
   print(village.get_system_status())
   ```

4. Generate performance report:
   ```python
   metrics = analyzer.analyze_performance()
   print(json.dumps(metrics, indent=2))
   ```

For more detailed information, refer to the [Architecture Documentation](architecture.md).
