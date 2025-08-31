# Service Dependency Flow and Integration Points

## Service Dependency Graph

```
┌─────────────────┐
│   API Service   │ (Entry Point)
└─────────┬───────┘
          │ coordinates
          ▼
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ Training Service│◄───┤  Model Service   │◄───┤ WebSocket Service│
│                 │    │                  │    │                  │
│ • PyTorch       │    │ • File Storage   │    │ • Real-time      │
│ • GrokFast      │    │ • Metadata       │    │ • Broadcasting   │
│ • Dataset Mgmt  │    │ • Version Ctrl   │    │ • Subscriptions  │
└─────────┬───────┘    └──────────┬───────┘    └──────────┬───────┘
          │                       │                       │
          │ publishes events       │ publishes events      │ receives events
          ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Event Bus / Message Broker                   │
│                         (Redis/RabbitMQ)                        │
└─────────────────────────┬───────────────────────────────────────┘
                          │ aggregates events
                          ▼
                ┌─────────────────┐
                │ Monitoring      │
                │ Service         │
                │                 │
                │ • Health Checks │
                │ • Metrics       │
                │ • Alerting      │
                └─────────────────┘
```

## Service Communication Patterns

### 1. Synchronous Communication (REST APIs)

#### API Service → Training Service
```python
# API Service calls Training Service
training_job = TrainingJob(phase=ModelPhase.COGNATE, config=config)
job_id = await training_service.start_training_job(training_job)
```

#### API Service → Model Service
```python
# API Service calls Model Service  
models = await model_service.list_models(phase=ModelPhase.COGNATE)
export_path = await model_service.export_models(export_request)
```

#### API Service → WebSocket Service
```python
# API Service calls WebSocket Service
connections = await websocket_service.get_active_connections()
await websocket_service.broadcast_message(message)
```

### 2. Asynchronous Communication (Event-Driven)

#### Training Service → Event Bus
```python
# Training Service publishes training events
event = TrainingProgressEvent(
    source_service="training_service",
    data={"job_id": job_id, "progress": 0.5, "message": "Training in progress"}
)
await event_publisher.publish(event)
```

#### Model Service → Event Bus  
```python
# Model Service publishes model events
event = ModelSavedEvent(
    source_service="model_service", 
    data={"model_id": model_id, "phase": "cognate"}
)
await event_publisher.publish(event)
```

#### Event Bus → WebSocket Service
```python
# WebSocket Service subscribes to events and broadcasts
async def handle_training_progress_event(event: TrainingProgressEvent):
    message = WebSocketMessage(
        type="training_progress",
        source_service="websocket_service",
        data=event.data
    )
    await websocket_service.broadcast_to_topic("training_updates", message)
```

## Data Flow Scenarios

### Training Workflow Data Flow

```
1. Client Request
   └─ POST /phases/cognate/start
      └─ API Service receives request
         └─ Creates TrainingJob
            └─ Calls Training Service
               └─ Training Service starts job
                  └─ Publishes TrainingStartedEvent
                     └─ WebSocket Service broadcasts to clients
                        └─ Training progresses with periodic updates
                           └─ Training Service publishes ProgressEvents
                              └─ WebSocket Service broadcasts progress
                                 └─ Training completes
                                    └─ Training Service saves to Model Service
                                       └─ Model Service publishes ModelSavedEvent
                                          └─ Training Service publishes CompletedEvent
                                             └─ WebSocket Service broadcasts completion
```

### Model Export Data Flow

```
1. Client Request
   └─ POST /models/export
      └─ API Service receives request  
         └─ Validates model IDs
            └─ Calls Model Service.export_models()
               └─ Model Service loads model files
                  └─ Creates export package
                     └─ Returns export path to API Service
                        └─ API Service returns path to client
```

### Real-time Update Data Flow

```
1. Training Progress Update
   └─ Training Service generates progress
      └─ Publishes ProgressEvent to Event Bus
         └─ WebSocket Service subscribes to events
            └─ Transforms event to WebSocketMessage
               └─ Broadcasts to subscribed clients
                  └─ Client receives real-time update
                     
2. System Health Update
   └─ Monitoring Service detects health change
      └─ Publishes HealthChangeEvent
         └─ WebSocket Service broadcasts health status
            └─ Admin clients receive health updates
```

## Integration Points

### 1. Event Publishing Interface

```python
class EventPublisher:
    async def publish(self, event: Event) -> bool:
        """Publish event to message broker"""
        
    async def subscribe(self, event_type: str, handler: callable):
        """Subscribe to specific event types"""
```

### 2. Service Discovery Interface

```python
class ServiceDiscovery:
    async def register_service(self, name: str, endpoint: str) -> bool:
        """Register service endpoint"""
        
    async def discover_service(self, name: str) -> Optional[str]:
        """Find service endpoint by name"""
        
    async def health_check(self, name: str) -> bool:
        """Check if service is healthy"""
```

### 3. Configuration Management

```python
class ConfigManager:
    def get_service_config(self, service_name: str) -> Dict[str, Any]:
        """Get configuration for a service"""
        
    def get_database_config(self) -> Dict[str, Any]:
        """Get database connection configuration"""
        
    def get_message_broker_config(self) -> Dict[str, Any]:
        """Get message broker configuration"""
```

## Error Handling and Resilience

### Circuit Breaker Pattern
```python
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    async def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise CircuitBreakerOpenError()
        
        try:
            result = await func(*args, **kwargs)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise
```

### Retry Mechanism
```python
async def retry_with_backoff(func, max_retries: int = 3, base_delay: float = 1.0):
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            await asyncio.sleep(delay)
```

## Service Startup Dependencies

### Dependency Order
1. **Message Broker** (Redis/RabbitMQ) - Must be available first
2. **Database** (PostgreSQL) - Required for metadata storage
3. **Model Service** - File system and database dependent
4. **Training Service** - Depends on Model Service for persistence
5. **Monitoring Service** - Can start independently, monitors others
6. **WebSocket Service** - Depends on Message Broker for events
7. **API Service** - Depends on all other services, starts last

### Health Check Dependencies
```python
service_dependencies = {
    "api_service": ["training_service", "model_service", "websocket_service"],
    "training_service": ["model_service", "message_broker"],
    "model_service": ["database", "file_system"],
    "websocket_service": ["message_broker"],
    "monitoring_service": []  # Independent
}
```

## Performance Considerations

### Caching Strategy
- **API Service**: Cache frequently accessed data (model metadata, phase status)
- **Model Service**: Cache model metadata in memory, lazy-load binary data
- **WebSocket Service**: Cache connection state, use connection pooling
- **Training Service**: Cache dataset downloads, reuse prepared data
- **Monitoring Service**: Cache health status, aggregate metrics periodically

### Load Balancing
- **API Service**: Stateless, can be load balanced with round-robin
- **Training Service**: Stateful (running jobs), use consistent hashing
- **Model Service**: Stateless for metadata, file access can be load balanced
- **WebSocket Service**: Sticky sessions required for connections
- **Monitoring Service**: Single instance sufficient, can replicate for redundancy

This dependency flow ensures clean separation of concerns while maintaining efficient communication between services.