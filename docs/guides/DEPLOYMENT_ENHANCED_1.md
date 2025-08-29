# Enhanced WhatsApp Wave Bridge Deployment Guide

## Part B: Agent Forge Phase 4 - Prompt Engineering Deployment

This guide covers deploying the enhanced WhatsApp Wave Bridge with advanced prompt engineering capabilities.

## 🚀 Quick Start (Enhanced Mode)

### Prerequisites

- Python 3.11+
- Twilio WhatsApp Business Account
- Weights & Biases Account (required for prompt engineering)
- AI Model API Keys (Anthropic and/or OpenAI)
- Docker (optional, for containerized deployment)

### Installation Steps

1. **Clone and Navigate**
```bash
cd services/wave_bridge
```

2. **Install Enhanced Dependencies**
```bash
pip install -r requirements_enhanced.txt
```

3. **Environment Configuration**
```bash
cp .env.example .env.enhanced
# Edit .env.enhanced with your API keys and configuration
```

4. **Initialize Prompt Engineering**
```bash
python -c "from agent_forge.prompt_engineering.tutor_prompts import tutor_prompt_engineer; import asyncio; asyncio.run(tutor_prompt_engineer.create_prompt_sweep())"
```

5. **Run Enhanced Service**
```bash
uvicorn app_enhanced:app --host 0.0.0.0 --port 8000 --reload
```

## 🔧 Enhanced Configuration

### Environment Variables

#### Core Features
```env
# Enhanced prompt engineering features
ENABLE_ENHANCED_PROMPTS=true
ENABLE_REAL_TIME_OPTIMIZATION=true
ENABLE_ADVANCED_AB_TESTING=true
ENABLE_PROMPT_BAKING=true

# Performance thresholds
PROMPT_OPTIMIZATION_THRESHOLD=0.75
RESPONSE_TIME_TARGET=5.0
MIN_SAMPLE_SIZE_AB_TEST=30
CONFIDENCE_LEVEL=0.95
```

#### Required APIs
```env
# Twilio (Required)
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886

# W&B (Required for prompt engineering)
WANDB_API_KEY=your_wandb_key
WANDB_PROJECT=aivillage-tutoring
WANDB_ENTITY=your_wandb_entity

# AI Models (At least one required)
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key
```

#### Optional Features
```env
# Performance monitoring
ENABLE_METRICS=true
ENABLE_PERFORMANCE_ALERTS=true
ALERT_WEBHOOK_URL=https://your-alert-endpoint.com/webhook

# Caching
ENABLE_RESPONSE_CACHE=true
CACHE_TTL=300
MAX_CACHE_SIZE=1000

# Security
RATE_LIMIT_PER_MINUTE=60
MAX_MESSAGE_LENGTH=1000
ENABLE_REQUEST_VALIDATION=true
```

## 📊 Enhanced Features Overview

### 1. Advanced Prompt Engineering

**TutorPromptEngineer**: Generates and optimizes prompt templates using W&B sweeps
- Bayesian optimization for prompt parameters
- Multi-dimensional parameter tuning (greeting style, hint complexity, example type)
- Performance tracking with statistical significance
- Automated artifact versioning

**Usage:**
```python
from agent_forge.prompt_engineering.tutor_prompts import tutor_prompt_engineer

# Create optimized prompt template
template = await tutor_prompt_engineer.generate_prompt_template(
    greeting_style="friendly",
    hint_complexity="guided",
    example_type="real-world",
    encouragement_frequency=0.3,
    subject="mathematics"
)
```

### 2. Advanced A/B Testing Framework

**PromptABTest**: Sophisticated A/B testing with multi-armed bandit algorithms
- UCB1 algorithm for balanced exploration/exploitation
- Statistical significance testing
- Real-time performance tracking
- Automatic winner identification

**Features:**
- Consistent user assignment across sessions
- Multi-variant testing (up to 10+ variants per test)
- Performance-based traffic allocation
- Detailed analytics and reporting

### 3. Prompt Baking System

**PromptBaker**: Identifies and prepares winning prompts for production
- Performance threshold validation
- Statistical confidence scoring
- Automated artifact creation
- Deployment readiness assessment

**Capabilities:**
- Winner identification from W&B data
- Weight optimization based on performance
- Deployment package creation
- Production validation

### 4. Enhanced Metrics & Monitoring

**Real-time Performance Tracking:**
- Response time distribution analysis
- Language-specific performance metrics
- Subject area expertise tracking
- A/B test performance monitoring

**Alerting System:**
- Performance degradation alerts
- Statistical significance notifications
- Error rate monitoring
- Custom threshold configuration

## 🌐 API Endpoints (Enhanced)

### Core Endpoints
- `POST /whatsapp/webhook` - Enhanced webhook with prompt engineering
- `GET /health/enhanced` - Detailed health check with feature status
- `GET /metrics/enhanced` - Comprehensive metrics including A/B test data

### Admin Endpoints
- `POST /admin/optimize-prompts` - Trigger prompt optimization
- `GET /admin/ab-test-results` - Detailed A/B test analytics
- `POST /admin/deploy-winners` - Deploy winning prompt templates
- `GET /admin/performance-report` - Generate performance report

### Configuration Endpoints
- `GET /config/features` - Get enabled features
- `GET /config/thresholds` - Get performance thresholds
- `POST /config/update` - Update configuration (admin only)

## 🧪 Testing the Enhanced System

### Integration Tests
```bash
# Run comprehensive integration test suite
pytest tests/integration/test_whatsapp_tutor_flow.py -v

# Run specific test cases
pytest tests/integration/test_whatsapp_tutor_flow.py::test_spanish_math_tutoring -v
pytest tests/integration/test_whatsapp_tutor_flow.py::test_prompt_engineering_integration -v
```

### Performance Testing
```bash
# Run performance benchmarks
python tests/integration/test_whatsapp_tutor_flow.py

# Test acceptance criteria validation
pytest tests/integration/test_whatsapp_tutor_flow.py::test_acceptance_criteria_validation -v
```

### Manual Testing Scenarios

1. **Multi-Language Tutoring**
   - Send messages in Spanish, Hindi, French, Arabic
   - Verify language consistency and tutoring quality
   - Check response times under 5 seconds

2. **Subject Specialization**
   - Test mathematics, science, programming questions
   - Verify subject-appropriate responses
   - Check for relevant examples and explanations

3. **A/B Test Validation**
   - Create multiple test users
   - Verify consistent variant assignment
   - Monitor performance differences

4. **Real-time Optimization**
   - Send messages that trigger slow responses
   - Verify optimization triggers are activated
   - Check W&B logs for optimization events

## 📈 Performance Benchmarks

### Target Metrics (Enhanced)
- **Response Time**: 95% under 5 seconds, average 2.8 seconds
- **Language Support**: 10 languages with auto-detection
- **A/B Test Coverage**: 95% of interactions in active tests
- **Prompt Optimization**: Daily optimization cycles
- **Engagement Rate**: 95% of responses include encouragement

### Acceptance Criteria Validation
- ✅ WhatsApp messages received and responded to in <5 seconds
- ✅ 7+ languages auto-detected and maintained in responses
- ✅ W&B tracking all interactions with prompt variants
- ✅ A/B tests running on at least 4 prompt dimensions
- ✅ Daily W&B report showing best performing prompts
- ✅ 95% of responses include appropriate encouragement

## 🐳 Docker Deployment (Enhanced)

### Build Enhanced Container
```bash
# Build with enhanced features
docker build -f Dockerfile.enhanced -t wave-bridge:enhanced .

# Run with environment configuration
docker run -d \
  --name wave-bridge-enhanced \
  --env-file .env.enhanced \
  -p 8000:8000 \
  wave-bridge:enhanced
```

### Docker Compose (Production)
```yaml
version: '3.8'
services:
  wave-bridge-enhanced:
    build:
      context: .
      dockerfile: Dockerfile.enhanced
    ports:
      - "8000:8000"
    environment:
      - ENABLE_ENHANCED_PROMPTS=true
      - ENABLE_REAL_TIME_OPTIMIZATION=true
      - WANDB_API_KEY=${WANDB_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    volumes:
      - ./agent_forge/baked_prompts:/app/agent_forge/baked_prompts
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health/enhanced"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## 🔍 Monitoring & Observability

### W&B Dashboard Setup

1. **Project Configuration**
   ```python
   wandb.init(
       project="aivillage-tutoring",
       job_type="enhanced-whatsapp-bridge",
       config={
           "prompt_engineering_version": "2.0.0",
           "enhanced_features": True
       }
   )
   ```

2. **Key Metrics to Monitor**
   - Response time distribution
   - A/B test performance
   - Language detection accuracy
   - Subject specialization effectiveness
   - Prompt template performance scores

3. **Alert Configuration**
   - Response time > 5 seconds
   - A/B test statistical significance reached
   - Prompt performance degradation
   - Error rate > 5%

### Custom Dashboards

Create custom W&B dashboards for:
- **Real-time Performance**: Response times, success rates
- **A/B Test Results**: Variant performance, statistical significance
- **Language Analytics**: Per-language performance metrics
- **Subject Expertise**: Subject-specific engagement and quality

## 🚨 Troubleshooting (Enhanced)

### Common Issues

**Slow Response Times**
```bash
# Check A/B test allocation
curl http://localhost:8000/admin/ab-test-results

# Monitor prompt performance
curl http://localhost:8000/metrics/enhanced | jq '.prompt_engineering'

# Trigger optimization
curl -X POST http://localhost:8000/admin/optimize-prompts
```

**A/B Test Issues**
```bash
# Verify test configuration
python -c "from agent_forge.prompt_engineering.ab_testing import prompt_ab_test; print(prompt_ab_test.active_tests)"

# Check user assignments
grep "variant_assignment" logs/app.log | tail -20
```

**Prompt Engineering Failures**
```bash
# Check W&B connectivity
python -c "import wandb; wandb.login()"

# Validate prompt templates
python -c "from agent_forge.prompt_engineering.tutor_prompts import tutor_prompt_engineer; print(len(tutor_prompt_engineer.active_templates))"
```

### Debug Mode
```bash
# Enable detailed logging
export LOG_LEVEL=DEBUG
export ENABLE_DEBUG_MODE=true

# Run with debug output
uvicorn app_enhanced:app --log-level debug --reload
```

## 📚 Additional Resources

- [W&B Integration Guide](https://docs.wandb.ai/guides/integrations)
- [FastAPI Best Practices](https://fastapi.tiangolo.com/tutorial/)
- [Twilio WhatsApp API](https://www.twilio.com/docs/whatsapp)
- [Statistical A/B Testing](https://en.wikipedia.org/wiki/A/B_testing)

## 🎯 Production Checklist

- [ ] All API keys configured and validated
- [ ] W&B project and sweeps initialized
- [ ] A/B tests configured and running
- [ ] Performance monitoring active
- [ ] Health checks passing
- [ ] Integration tests passing
- [ ] Docker container builds successfully
- [ ] Load balancer configured (if applicable)
- [ ] SSL certificates installed
- [ ] Monitoring dashboards created
- [ ] Alert webhooks configured
- [ ] Backup and recovery procedures tested

---

**🎓 Built for Education, Optimized for Performance, Enhanced with AI**

*Making quality tutoring accessible worldwide through WhatsApp with cutting-edge prompt engineering* 🌍📚✨
