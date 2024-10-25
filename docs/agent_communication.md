# Agent Communication System

## Overview

The agent communication system enables seamless interaction between King, Sage, and Magi agents through a standardized protocol. It supports both synchronous and asynchronous communication patterns.

## Architecture

### Components

1. **Communication Protocol**
   - Location: `communications/protocol.py`
   - Purpose: Defines message formats and routing rules
   - Features:
     - Message serialization
     - Priority handling
     - Delivery guarantees

2. **Message Router**
   - Location: `communications/message.py`
   - Purpose: Routes messages between agents
   - Features:
     - Priority queues
     - Load balancing
     - Message tracking

3. **Community Hub**
   - Location: `communications/community_hub.py`
   - Purpose: Central communication management
   - Features:
     - Agent registration
     - Message broadcasting
     - Status monitoring

## Message Types

### Task Messages
```python
{
    "type": "task",
    "sender": "king",
    "receiver": "magi",
    "priority": 1,
    "content": {
        "task_type": "code_generation",
        "requirements": [...],
        "deadline": "2024-01-01T00:00:00Z"
    }
}
```

### Knowledge Messages
```python
{
    "type": "knowledge",
    "sender": "sage",
    "receiver": "all",
    "content": {
        "concept": "task_planning",
        "knowledge": {...},
        "confidence": 0.95
    }
}
```

### Status Messages
```python
{
    "type": "status",
    "sender": "magi",
    "receiver": "king",
    "content": {
        "status": "task_completed",
        "task_id": "123",
        "result": {...}
    }
}
```

## Communication Patterns

### Direct Communication
```python
from communications.protocol import send_message

await send_message(
    sender="king",
    receiver="magi",
    content=task_data
)
```

### Broadcast
```python
from communications.community_hub import broadcast

await broadcast(
    sender="sage",
    message=knowledge_update
)
```

### Request-Response
```python
from communications.protocol import request_response

response = await request_response(
    sender="king",
    receiver="sage",
    request=query,
    timeout=30
)
```

## Agent Interactions

### King → Sage
1. Research requests
2. Knowledge validation
3. Decision support

### King → Magi
1. Tool creation requests
2. Code generation tasks
3. System optimization

### Sage → Magi
1. Code pattern requests
2. Tool improvement suggestions
3. Research implementations

## Error Handling

### Retry Logic
```python
from communications.error_handling import retry_send

@retry_send(max_retries=3)
async def reliable_send(message):
    return await send_message(message)
```

### Circuit Breaking
```python
from communications.error_handling import circuit_breaker

@circuit_breaker(failure_threshold=5)
async def protected_send(message):
    return await send_message(message)
```

## Monitoring

### Message Tracking
```python
from communications.monitoring import MessageTracker

tracker = MessageTracker()
stats = await tracker.get_stats()
print(f"Messages processed: {stats['processed']}")
```

### Performance Metrics
```python
from communications.monitoring import PerformanceMonitor

monitor = PerformanceMonitor()
latency = await monitor.get_average_latency()
print(f"Average latency: {latency}ms")
```

## Configuration

### System Settings
```yaml
communication:
  protocol: "async"
  max_retries: 3
  timeout: 30
  batch_size: 100
  
queues:
  high_priority:
    max_size: 1000
    timeout: 10
  normal:
    max_size: 5000
    timeout: 30
```

## Best Practices

1. Message Design
   - Keep messages focused
   - Include necessary context
   - Use appropriate priority levels

2. Error Handling
   - Implement retries
   - Use circuit breakers
   - Log failures

3. Performance
   - Batch when possible
   - Monitor queue sizes
   - Use appropriate timeouts

4. Security
   - Validate messages
   - Authenticate senders
   - Encrypt sensitive data

## Common Issues

1. Message Delivery
   - Check network connectivity
   - Verify queue status
   - Monitor timeouts

2. Performance
   - Monitor queue lengths
   - Check message sizes
   - Optimize batch processing

3. Integration
   - Verify protocol compatibility
   - Check authentication
   - Monitor system resources

## Testing

### Unit Tests
```python
def test_message_sending():
    result = send_message(test_message)
    assert result.status == "delivered"
```

### Integration Tests
```python
async def test_agent_communication():
    response = await king_agent.send_to_magi(task)
    assert response.status == "accepted"
```

## Maintenance

### Queue Cleanup
```python
from communications.maintenance import QueueMaintainer

maintainer = QueueMaintainer()
await maintainer.cleanup_old_messages(hours=24)
```

### Performance Optimization
```python
from communications.optimization import Optimizer

optimizer = Optimizer()
await optimizer.optimize_queues()
```

## Security

### Message Encryption
```python
from communications.security import encrypt_message

encrypted = await encrypt_message(
    message,
    recipient_key
)
```

### Authentication
```python
from communications.security import authenticate_sender

is_valid = await authenticate_sender(
    message,
    sender_id
)
```

## Debugging

### Message Tracing
```python
from communications.debugging import MessageTracer

tracer = MessageTracer()
path = await tracer.trace_message(message_id)
```

### Performance Analysis
```python
from communications.debugging import PerformanceAnalyzer

analyzer = PerformanceAnalyzer()
bottlenecks = await analyzer.find_bottlenecks()
