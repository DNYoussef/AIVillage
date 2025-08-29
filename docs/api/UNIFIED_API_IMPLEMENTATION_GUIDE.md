# AIVillage Unified API Gateway - Implementation Guide

## Overview

The AIVillage Unified API Gateway is a production-ready REST API that consolidates all AIVillage services into a single, secure, and scalable endpoint. This implementation addresses the API reality gap identified in the documentation audit and provides a complete, working solution.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Unified API Gateway (Port 8000)             │
├─────────────────────────────────────────────────────────────────┤
│  • JWT Authentication with MFA support                         │
│  • Rate Limiting (Standard/Premium/Enterprise tiers)           │
│  • Comprehensive Error Handling                                │
│  • CORS, GZip, Security Middleware                             │
│  • OpenAPI 3.0 Documentation                                   │
│  • Real-time WebSocket Updates                                 │
├─────────────────┬─────────────────┬─────────────────────────────┤
│  Agent Forge    │  P2P/Fog APIs   │  Utility Services           │
│  • /v1/models/  │  • /v1/p2p/     │  • /v1/query               │
│  • /v1/chat     │  • /v1/fog/     │  • /v1/upload              │
│  • Training     │  • /v1/tokens   │  • Health checks           │
└─────────────────┴─────────────────┴─────────────────────────────┘
```

## Key Features Implemented

### ✅ Authentication & Security
- **JWT Authentication**: Full implementation with configurable MFA
- **API Key Support**: Alternative authentication method
- **Rate Limiting**: Tiered rate limits (Standard/Premium/Enterprise)
- **Security Headers**: CORS, content-type validation, request sanitization
- **Token Management**: Creation, validation, refresh, and revocation

### ✅ Agent Forge Integration
- **Training Endpoints**: `/v1/models/train` with 7-phase pipeline support
- **Model Management**: `/v1/models` for listing trained models
- **Chat Interface**: `/v1/chat` for model interaction
- **Real-time Updates**: WebSocket notifications for training progress
- **Background Processing**: Async training with progress tracking

### ✅ P2P/Fog Computing APIs
- **Network Status**: `/v1/p2p/status` for BitChat/BetaNet connectivity
- **Fog Nodes**: `/v1/fog/nodes` for computing node management
- **Token Economics**: `/v1/tokens` for FOG token system
- **Resource Monitoring**: Real-time resource utilization tracking

### ✅ Production Features
- **Health Monitoring**: Comprehensive health checks with service status
- **Error Handling**: Standardized error responses with request tracking
- **API Versioning**: `/v1/` prefix with backward compatibility planning
- **Documentation**: Auto-generated OpenAPI docs at `/docs` and `/redoc`
- **WebSocket Support**: Real-time updates at `/ws`

## Quick Start

### 1. Installation

```bash
# Navigate to gateway directory
cd infrastructure/gateway/

# Install dependencies
pip install fastapi uvicorn pyjwt python-multipart

# Set environment variables
export JWT_SECRET_KEY="your-secret-key-here"
export PORT=8000
export DEBUG=false
```

### 2. Start the Gateway

```bash
# Using the startup script (recommended)
python start_unified_gateway.py

# Or directly with uvicorn
uvicorn unified_api_gateway:app --host 0.0.0.0 --port 8000
```

### 3. Access the API

- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health
- **WebSocket**: ws://localhost:8000/ws

## Authentication Setup

### JWT Authentication

```python
# Create JWT token
from infrastructure.gateway.auth import JWTHandler

jwt_handler = JWTHandler(secret_key="your-secret")
token = jwt_handler.create_token(
    user_id="user123",
    scopes=["read", "write"],
    mfa_verified=True,
    rate_limit_tier="premium"
)

# Use token in requests
headers = {"Authorization": f"Bearer {token}"}
```

### API Key Authentication

```bash
# Set API keys in environment
export API_KEY_1="key123:user1:read,write:premium"
export API_KEY_2="key456:user2:read:standard"

# Use in requests
curl -H "X-API-Key: key123" "http://localhost:8000/v1/models"
```

## API Usage Examples

### Start Model Training

```bash
curl -X POST "http://localhost:8000/v1/models/train" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "phase_name": "cognate",
    "real_training": true,
    "max_steps": 2000,
    "batch_size": 2
  }'
```

### Chat with Model

```bash
curl -X POST "http://localhost:8000/v1/chat" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "cognate-model-123",
    "message": "Hello, how can you help me?",
    "conversation_id": "conv-456"
  }'
```

### Check P2P Status

```bash
curl -X GET "http://localhost:8000/v1/p2p/status" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Upload File

```bash
curl -X POST "http://localhost:8000/v1/upload" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@document.pdf"
```

## WebSocket Integration

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    if (data.type === 'training_progress') {
        console.log(`Training progress: ${data.progress * 100}%`);
    } else if (data.type === 'training_complete') {
        console.log('Training completed successfully!');
    }
};

// Send ping
ws.send(JSON.stringify({type: 'ping'}));
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | 0.0.0.0 | Server host |
| `PORT` | 8000 | Server port |
| `JWT_SECRET_KEY` | (required) | JWT signing secret |
| `REQUIRE_MFA` | false | Require multi-factor auth |
| `DEBUG` | false | Enable debug mode |
| `LOG_LEVEL` | info | Logging level |
| `WORKERS` | 1 | Number of worker processes |

### Rate Limiting Tiers

| Tier | Requests/Minute | Requests/Hour | Features |
|------|-----------------|---------------|----------|
| **Standard** | 60 | 1,000 | Basic access |
| **Premium** | 200 | 5,000 | Priority support |
| **Enterprise** | 500 | 10,000 | Custom features |

## Error Handling

All errors follow a standardized format:

```json
{
  "success": false,
  "error_code": "RATE_LIMIT_EXCEEDED",
  "message": "Rate limit exceeded for tier standard",
  "timestamp": "2025-08-27T14:30:00Z",
  "request_id": "req-12345-abcde"
}
```

### Common Error Codes

- `AUTHENTICATION_REQUIRED`: Missing or invalid authentication
- `INSUFFICIENT_PERMISSIONS`: User lacks required permissions
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `RESOURCE_NOT_FOUND`: Requested resource doesn't exist
- `VALIDATION_ERROR`: Invalid request data
- `SERVICE_UNAVAILABLE`: Backend service unavailable
- `INTERNAL_ERROR`: Server-side error

## Monitoring and Health Checks

### Health Check Endpoint

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2025-08-27T14:30:00Z",
  "services": {
    "agent_forge": {
      "status": "running",
      "available": true
    },
    "p2p_fog": {
      "status": "running", 
      "available": true
    },
    "websocket": {
      "status": "running",
      "active_connections": 5
    }
  },
  "version": "1.0.0"
}
```

### Metrics and Logging

- **Request Logging**: All requests logged with timing and status
- **Error Tracking**: Detailed error logs with request IDs
- **Performance Metrics**: Response times and throughput
- **Security Events**: Authentication failures and suspicious activity

## Testing

### Run Integration Tests

```bash
cd tests/integration/
python -m pytest test_unified_api.py -v
```

### Test Coverage

- ✅ Authentication and authorization
- ✅ All API endpoints
- ✅ Rate limiting
- ✅ Error handling
- ✅ WebSocket functionality
- ✅ Data validation
- ✅ Performance benchmarks

## Deployment

### Development Deployment

```bash
# Start with auto-reload
python start_unified_gateway.py
```

### Production Deployment

```bash
# Set production environment
export DEBUG=false
export WORKERS=4
export JWT_SECRET_KEY="production-secret-key"

# Start production server
python start_unified_gateway.py
```

### Docker Deployment

```dockerfile
# Dockerfile example
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "infrastructure/gateway/start_unified_gateway.py"]
```

## Migration from Development Server

The unified gateway replaces the deprecated development server (`server.py`). Key differences:

### Old Development Server Issues:
- ❌ Deprecation warnings
- ❌ Optional authentication
- ❌ Limited error handling
- ❌ No rate limiting
- ❌ Basic validation only

### New Unified Gateway Benefits:
- ✅ Production-ready security
- ✅ Comprehensive authentication
- ✅ Standardized error responses
- ✅ Advanced rate limiting
- ✅ Full service integration
- ✅ Real-time capabilities

### Migration Steps:

1. **Update Client Code**: Change endpoints from `/query` to `/v1/query`
2. **Add Authentication**: Implement JWT or API key authentication
3. **Handle Rate Limits**: Implement retry logic for rate limit responses
4. **Update Error Handling**: Handle new standardized error format
5. **Test Thoroughly**: Run integration tests before production deployment

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   - Check JWT_SECRET_KEY is set
   - Verify token is not expired
   - Ensure required scopes are included

2. **Rate Limit Exceeded**
   - Check current tier limits
   - Implement exponential backoff
   - Consider upgrading tier

3. **Service Unavailable**
   - Check service dependencies
   - Review health check endpoint
   - Verify environment configuration

4. **WebSocket Connection Issues**
   - Ensure WebSocket support in client
   - Check firewall settings
   - Verify correct protocol (ws/wss)

### Debug Mode

```bash
export DEBUG=true
python start_unified_gateway.py
```

Debug mode provides:
- Detailed error traces
- Auto-reload on code changes
- Verbose logging
- Development-friendly settings

## Support and Contributing

- **Documentation**: See `docs/api/` for complete API reference
- **Issues**: Report issues at GitHub repository
- **Testing**: Run `pytest tests/integration/` before submitting PRs
- **Code Style**: Follow existing patterns and include type hints

---

## Summary

The AIVillage Unified API Gateway provides a production-ready solution that:

✅ **Resolves API Reality Gap**: Implements all documented features with actual working code
✅ **Unifies Services**: Single endpoint for Agent Forge, P2P/Fog, and utilities  
✅ **Production Security**: JWT authentication, rate limiting, comprehensive error handling
✅ **Complete Integration**: 7-phase pipeline, WebSocket updates, health monitoring
✅ **Standards Compliant**: OpenAPI 3.0, REST principles, semantic versioning
✅ **Test Coverage**: Comprehensive integration tests with performance benchmarks

This implementation transforms AIVillage from a development prototype into a production-ready API platform ready for real-world deployment.