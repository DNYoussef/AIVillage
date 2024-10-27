# AI Village Architecture

## Overview

AI Village is a sophisticated system that combines frontier models (via OpenRouter) with local models (via HuggingFace) to provide efficient and scalable AI capabilities. The system uses a dynamic learning approach where frontier models help train and improve local models over time.

## Core Components

### 1. Agent System

#### OpenRouter Integration
- Provides access to frontier models:
  - King: nvidia/llama-3.1-nemotron-70b-instruct
  - Sage: anthropic/claude-3.5-sonnet
  - Magi: openai/o1-mini-2024-09-12
- Handles rate limiting and error management
- Tracks interactions for training data

#### Local Models
- Manages HuggingFace models:
  - King: Qwen/Qwen2.5-3B-Instruct
  - Sage: deepseek-ai/Janus-1.3B
  - Magi: ibm-granite/granite-3b-code-instruct-128k
- Provides efficient inference for simpler tasks
- Learns from frontier model interactions

#### Specialized Agents
- **KingAgent**: Complex problem-solving and strategic thinking
- **SageAgent**: Research, analysis, and knowledge synthesis
- **MagiAgent**: Code generation and technical problem-solving

### 2. Data Management

#### DataCollector
- Stores API outputs and performance metrics
- Manages training data collection
- Provides data export capabilities
- Uses SQLite for efficient storage

#### ComplexityEvaluator
- Determines task complexity
- Manages model selection thresholds
- Adapts based on performance metrics
- Provides complexity analysis

### 3. System Management

#### AgentManager
- Orchestrates agent interactions
- Handles task routing
- Manages agent configurations
- Tracks performance metrics

#### AIVillage
- Main system orchestrator
- Manages task queue
- Coordinates components
- Provides system monitoring

## Data Flow

1. **Task Ingestion**
   ```
   User Task → Task Queue → AIVillage
   ```

2. **Task Processing**
   ```
   AIVillage → ComplexityEvaluator → AgentManager → Appropriate Agent
   ```

3. **Model Selection**
   ```
   Agent → Complexity Check → Local/Frontier Model → Response
   ```

4. **Data Collection**
   ```
   Response → DataCollector → Training Data → Local Model Improvement
   ```

## Performance Optimization

### Complexity-Based Routing
- Tasks are evaluated for complexity
- Simple tasks use local models
- Complex tasks use frontier models
- Thresholds adjust automatically

### Learning Pipeline
1. Frontier models handle complex tasks
2. Interactions are recorded
3. Data is used to train local models
4. Local models gradually handle more tasks

### Performance Monitoring
- Response times tracked
- Model usage monitored
- Error rates analyzed
- System metrics collected

## System Requirements

### Hardware
- CPU: 4+ cores recommended
- RAM: 16GB minimum, 32GB recommended
- Storage: 50GB minimum for models and data

### Software
- Python 3.8+
- PyTorch 1.8+
- SQLite 3
- Required Python packages in requirements.txt

## Configuration

### Environment Variables
```bash
OPENROUTER_API_KEY=your_api_key
```

### Configuration Files
- `config/openrouter_agents.yaml`: Agent configurations
- `config/default.yaml`: System settings

## Logging System

### Log Types
1. **System Logs**: General operation logs
2. **Performance Logs**: Metrics and timing data
3. **Error Logs**: Detailed error tracking

### Log Management
- Automatic rotation
- Compression of old logs
- Analysis tools provided

## Security Considerations

### API Security
- API keys stored securely
- Rate limiting implemented
- Request validation

### Data Security
- Local data encryption
- Secure model storage
- Access control

## Scaling Considerations

### Horizontal Scaling
- Multiple instances possible
- Shared database support
- Load balancing ready

### Vertical Scaling
- Model compression options
- Memory optimization
- Performance tuning

## Error Handling

### Recovery Mechanisms
- Automatic retries
- Fallback options
- Error logging

### Monitoring
- Real-time alerts
- Performance tracking
- Error pattern analysis

## Future Expansion

### Planned Features
1. Additional model support
2. Enhanced learning capabilities
3. Advanced monitoring tools
4. Multi-modal support

### Integration Points
- API endpoints
- Plugin system
- Custom model support

## Best Practices

### Development
1. Follow PEP 8 style guide
2. Write comprehensive tests
3. Document all changes
4. Use type hints

### Deployment
1. Use virtual environments
2. Monitor resource usage
3. Regular backups
4. Update dependencies

### Maintenance
1. Regular log analysis
2. Performance optimization
3. Model updates
4. Security patches
