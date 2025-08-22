# Resilient Chat Engine Architecture

## Overview

The Digital Twin Chat Engine has been redesigned with a resilient architecture that eliminates the hardcoded dependency on the external twin:8001 service. The system now supports graceful degradation with meaningful offline functionality, ensuring users always receive helpful responses even when remote services are unavailable.

## Architecture Pattern: Circuit Breaker with Local Fallback

The solution implements the **Circuit Breaker pattern** combined with **Local Fallback processing** to create a robust, fault-tolerant chat system.

### Key Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Chat Engine   │───▶│ Circuit Breaker  │───▶│ Twin Service    │
│    (Main)       │    │   & Retry        │    │  (twin:8001)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
          │                       │
          │ (on failure)          │ (when open)
          ▼                       ▼
┌─────────────────┐    ┌──────────────────┐
│ Local Chat      │    │ Health Monitor   │
│   Processor     │    │ & Status Checks  │
└─────────────────┘    └──────────────────┘
```

## Operation Modes

The system supports three configurable operation modes:

### 1. **Remote Mode** (`CHAT_MODE=remote`)
- Always attempts to use Twin service
- Falls back to local processing only if `OFFLINE_RESPONSES_ENABLED=1`
- Best for production when Twin service is expected to be available

### 2. **Local Mode** (`CHAT_MODE=local`)
- Always uses local chat processing
- No network calls to Twin service
- Ideal for development, testing, or offline scenarios

### 3. **Hybrid Mode** (`CHAT_MODE=hybrid`) - **Default**
- Intelligently chooses between remote and local processing
- Uses health checks and circuit breaker state to make decisions
- Provides the best user experience with automatic failover

## Circuit Breaker Implementation

### States
- **CLOSED**: Service is healthy, requests flow normally
- **OPEN**: Service has failed, all requests use local fallback
- **HALF-OPEN**: Testing service recovery, limited requests allowed

### Configuration
```python
CIRCUIT_BREAKER_FAILURE_THRESHOLD=5     # Failures before opening
CIRCUIT_BREAKER_TIMEOUT_MS=60000        # Time before retry (1 minute)
CIRCUIT_BREAKER_SUCCESS_THRESHOLD=3     # Successes to close circuit
```

### State Transitions
```
CLOSED ──[5+ failures]──▶ OPEN ──[60s timeout]──▶ HALF_OPEN
  ▲                         │                         │
  └────[3+ successes]───────┘                         │
  ▲                                                   │
  └─────────────────[1+ failure]──────────────────────┘
```

## Local Chat Processor Features

When operating offline or in fallback mode, the Local Chat Processor provides:

### Core Functionality
- **Conversation History**: Maintains context across messages
- **Message Categorization**: Greetings, questions, help requests, status checks
- **Special Commands**: `/status`, `/help`, `/echo [text]`
- **Contextual Responses**: Varied responses based on message type and history

### Response Templates
```python
response_types = {
    "greeting": ["Hello! I'm running in local mode...", ...],
    "question": ["That's an interesting question...", ...],
    "help": ["I'm operating in local mode with basic functionality...", ...],
    "status": ["Status: Local chat mode active...", ...],
    "default": ["I received your message...", ...]
}
```

### Mock Confidence Scoring
- Generates deterministic but varied confidence scores (0.70-1.00)
- Based on message hash for consistency
- No calibration in local mode

## Health Monitoring

### Service Health Checks
```python
def health_check_twin_service() -> bool:
    # Try dedicated health endpoint first
    try:
        response = requests.get(f"{TWIN_URL.replace('/v1/chat', '/health')}", timeout=5)
        return response.status_code == 200
    except:
        # Fallback: test main endpoint
        try:
            response = requests.post(TWIN_URL, json=test_payload, timeout=5)
            return response.status_code in (200, 400)  # Service is up
        except:
            return False
```

### Health Check Interval
- Configurable via `SERVICE_HEALTH_CHECK_INTERVAL` (default: 30 seconds)
- Prevents excessive health check overhead
- Updates service status: `HEALTHY`, `DEGRADED`, `OFFLINE`

## Configuration System

### Environment Variables
```bash
# Core Settings
TWIN_URL=https://twin:8001/v1/chat          # Twin service endpoint
CHAT_MODE=hybrid                            # remote|local|hybrid
OFFLINE_RESPONSES_ENABLED=1                 # Enable local fallback

# Health Monitoring
SERVICE_HEALTH_CHECK_INTERVAL=30            # Seconds between health checks

# Circuit Breaker
CIRCUIT_BREAKER_FAILURE_THRESHOLD=5         # Failures before opening
CIRCUIT_BREAKER_TIMEOUT_MS=60000            # Milliseconds before retry

# Legacy Settings (still supported)
CALIBRATION_ENABLED=0                       # Enable confidence calibration
```

### Configuration Precedence
1. Environment variables (highest priority)
2. Default values in code
3. Fallback constants

## Error Handling & Resilience

### Retry Logic
- **Exponential backoff** with jitter
- **Configurable retry attempts** (default: 3)
- **Retryable exceptions**: `ConnectionError`, `Timeout`
- **Maximum delay**: 5 seconds

### Graceful Degradation
- Clear user feedback about operating mode
- Meaningful responses in offline scenarios
- Error context preserved for debugging
- No hard failures for network issues

### Error Categories
```python
class ServiceStatus(Enum):
    HEALTHY = "healthy"      # All systems operational
    DEGRADED = "degraded"    # Some issues, partial functionality
    OFFLINE = "offline"      # Remote service unavailable
```

## Response Format

### Enhanced Response Schema
```python
{
    "response": str,                    # Chat response text
    "conversation_id": str,             # Conversation identifier
    "raw_prob": float,                  # Confidence score (0.0-1.0)
    "calibrated_prob": float|None,      # Calibrated confidence (if enabled)
    "timestamp": str,                   # ISO 8601 timestamp
    "mode": str,                        # "remote"|"local"|"fallback"
    "service_status": str,              # "healthy"|"degraded"|"offline"
    "processing_time_ms": int,          # Response generation time

    # Optional fields in fallback mode
    "fallback_reason": str,             # Why fallback was used
    "notice": str,                      # User notification about limited capabilities
    "conversation_length": int          # Number of messages in conversation
}
```

## Usage Examples

### Basic Usage
```python
from infrastructure.twin.chat_engine import ChatEngine

# Initialize with default hybrid mode
engine = ChatEngine()

# Process chat message
response = engine.process_chat("Hello, how are you?", "user123")

print(f"Response: {response['response']}")
print(f"Mode: {response['mode']} | Status: {response['service_status']}")
```

### Configuration Override
```python
import os

# Force local mode for testing
os.environ["CHAT_MODE"] = "local"
os.environ["OFFLINE_RESPONSES_ENABLED"] = "1"

engine = ChatEngine()
response = engine.process_chat("Test message", "test_conv")
# Will always use local processing
```

### System Status Monitoring
```python
# Get comprehensive system status
status = engine.get_system_status()

print(f"Circuit Breaker State: {status['circuit_breaker_state']}")
print(f"Service Status: {status['service_status']}")
print(f"Total Requests: {status['circuit_breaker_stats']['total_calls']}")
```

### Administrative Operations
```python
# Force mode change (for testing/admin)
engine.force_mode_change("local")

# Reset circuit breaker (after fixing service issues)
engine.reset_circuit_breaker()

# Get conversation history (local mode only)
history = engine.get_conversation_history("user123")
```

## Performance Characteristics

### Response Times
- **Local mode**: ~50ms (simulated processing)
- **Remote mode**: ~200-2000ms (depends on Twin service)
- **Fallback mode**: ~50-100ms (local processing + error handling)

### Resource Usage
- **Memory**: Minimal overhead for conversation history
- **Network**: Health checks every 30s (configurable)
- **CPU**: Low impact from message categorization and templating

### Scalability
- **Stateless design**: Each ChatEngine instance is independent
- **Conversation storage**: In-memory only (suitable for development)
- **Production considerations**: Consider external conversation storage

## Testing & Validation

### Unit Tests
```python
# Test offline functionality
def test_local_mode_responses():
    os.environ["CHAT_MODE"] = "local"
    engine = ChatEngine()
    response = engine.process_chat("Hello", "test")
    assert response["mode"] == "local"
    assert response["service_status"] == "offline"

# Test circuit breaker behavior
def test_circuit_breaker_opens():
    engine = ChatEngine()
    # Simulate failures...
    assert engine._circuit_breaker.state == CircuitBreakerState.OPEN
```

### Integration Testing
- Health check endpoint validation
- Network failure simulation
- Circuit breaker state transitions
- Conversation history persistence

### Demo Script
Run the comprehensive demonstration:
```bash
cd /path/to/AIVillage
python examples/resilient_chat_demo.py
```

## Migration from Legacy System

### Breaking Changes
- Response format includes new fields (`mode`, `service_status`)
- `ChatEngine` constructor now requires no parameters
- Configuration is environment-driven

### Backward Compatibility
- Core `process_chat(message, conversation_id)` API unchanged
- Existing response fields maintained
- Legacy environment variables still supported

### Migration Steps
1. Update environment configuration
2. Handle new response fields in client code
3. Test offline scenarios thoroughly
4. Update monitoring to check `service_status`

## Production Deployment

### Recommended Configuration
```bash
# Production settings
CHAT_MODE=hybrid
OFFLINE_RESPONSES_ENABLED=1
SERVICE_HEALTH_CHECK_INTERVAL=60
CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
CIRCUIT_BREAKER_TIMEOUT_MS=120000
```

### Monitoring & Alerting
- Monitor `service_status` and `circuit_breaker_state`
- Alert on sustained `OFFLINE` or `DEGRADED` status
- Track failure rates and response times
- Monitor conversation volume in local vs remote modes

### Load Balancing Considerations
- Circuit breaker state is per-instance
- Consider shared state for multi-instance deployments
- Health check intervals should be coordinated

## Troubleshooting Guide

### Common Issues

**Q: Chat engine always uses local mode even when Twin service is available**
- Check `CHAT_MODE` environment variable
- Verify `TWIN_URL` is correct and accessible
- Check network connectivity and firewall rules

**Q: Circuit breaker opens too frequently**
- Increase `CIRCUIT_BREAKER_FAILURE_THRESHOLD`
- Check Twin service stability and response times
- Review network infrastructure for intermittent issues

**Q: Local responses are too generic**
- Customize response templates in `LocalChatProcessor`
- Add more message categorization rules
- Implement domain-specific response generation

**Q: Health checks failing but service is available**
- Verify Twin service has `/health` endpoint
- Check timeout settings vs actual response times
- Review authentication requirements for health endpoint

### Debug Information
```python
# Enable detailed logging
import logging
logging.getLogger('infrastructure.twin.chat_engine').setLevel(logging.DEBUG)

# Get circuit breaker statistics
status = engine.get_system_status()
print(status['circuit_breaker_stats'])

# Force manual health check
is_healthy = engine.health_check_twin_service()
print(f"Manual health check: {is_healthy}")
```

## Security Considerations

### Network Security
- HTTPS enforced for Twin service communication
- Timeout limits prevent resource exhaustion
- No sensitive data in local conversation history

### Error Information Disclosure
- Error messages sanitized in user responses
- Detailed errors logged server-side only
- Fallback responses don't reveal infrastructure details

### Input Validation
- Message length limits should be enforced by caller
- Conversation ID validation prevents injection attacks
- Special command parsing is basic and safe

## Future Enhancements

### Potential Improvements
1. **Persistent Conversation Storage**: Redis/database backend for conversation history
2. **Advanced Response Generation**: Integration with local language models
3. **Adaptive Health Checking**: Dynamic intervals based on service stability
4. **Metrics & Observability**: Prometheus metrics, distributed tracing
5. **Multi-Service Support**: Circuit breakers for multiple backend services
6. **Response Caching**: Cache remote responses for improved offline experience

### Architectural Evolution
- Event-driven architecture for service status changes
- Pluggable response processors for different domains
- Machine learning for local response quality improvement
- Integration with centralized resilience frameworks

## Conclusion

The Resilient Chat Engine transforms a brittle, hardcoded system into a robust, fault-tolerant architecture that gracefully handles service outages while maintaining meaningful user interactions. The circuit breaker pattern with local fallback ensures high availability, while the flexible configuration system allows adaptation to different deployment scenarios.

**Key Benefits:**
- ✅ **Zero downtime**: System works even when Twin service is offline
- ✅ **Graceful degradation**: Users get helpful responses in all scenarios
- ✅ **Operational visibility**: Comprehensive health monitoring and status reporting
- ✅ **Flexible deployment**: Support for remote, local, and hybrid modes
- ✅ **Production ready**: Circuit breaker prevents cascade failures

The system successfully eliminates the original `ConnectionError: HTTPSConnectionPool(host='twin', port=8001): Max retries exceeded` failure mode and provides a foundation for building resilient, user-focused chat experiences.
