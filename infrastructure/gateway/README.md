# AIVillage Unified API Gateway

Production-ready REST API gateway that unifies all AIVillage services into a single, secure, and scalable endpoint.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export JWT_SECRET_KEY="your-secret-key-here"
export PORT=8000

# Start the gateway
python start_unified_gateway.py
```

**Access Points:**
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health
- WebSocket: ws://localhost:8000/ws

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Unified API Gateway                         │
├─────────────────────────────────────────────────────────────────┤
│  🔐 JWT Authentication  │  ⚡ Rate Limiting  │  🛡️ Security     │
├─────────────────────────┼─────────────────────┼─────────────────┤
│  Agent Forge APIs      │  P2P/Fog APIs      │  Utility APIs    │
│  • Model Training      │  • Network Status   │  • File Upload   │
│  • Chat Interface      │  • Fog Computing    │  • RAG Queries   │
│  • Real-time Updates   │  • Token Economics  │  • Health Checks │
└─────────────────────────┴─────────────────────┴─────────────────┘
```

## ✨ Features

### 🔐 Authentication & Security
- JWT authentication with MFA support
- API key alternative authentication
- Tiered rate limiting (Standard/Premium/Enterprise)
- Comprehensive security middleware
- Token management (create, validate, refresh, revoke)

### 🤖 Agent Forge Integration
- `/v1/models/train` - Start 7-phase training pipeline
- `/v1/models` - List trained models
- `/v1/chat` - Chat with trained models
- Real-time training progress via WebSocket
- Background processing with progress tracking

### 🌐 P2P/Fog Computing
- `/v1/p2p/status` - P2P network connectivity
- `/v1/fog/nodes` - Fog computing nodes
- `/v1/tokens` - FOG token economics
- BitChat/BetaNet integration
- Decentralized resource monitoring

### 📊 Production Features
- Comprehensive health monitoring
- Standardized error handling
- OpenAPI 3.0 documentation
- Real-time WebSocket updates
- Request tracking and metrics
- Graceful shutdown handling

## 📖 API Documentation

### Authentication

**JWT Token:**
```bash
curl -H "Authorization: Bearer YOUR_TOKEN" "http://localhost:8000/v1/models"
```

**API Key:**
```bash
curl -H "X-API-Key: YOUR_API_KEY" "http://localhost:8000/v1/models"
```

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health status |
| `/v1/models/train` | POST | Start model training |
| `/v1/models` | GET | List models |
| `/v1/chat` | POST | Chat with model |
| `/v1/p2p/status` | GET | P2P network status |
| `/v1/fog/nodes` | GET | Fog computing nodes |
| `/v1/tokens` | GET | Token economics |
| `/v1/query` | POST | RAG query processing |
| `/v1/upload` | POST | File upload |
| `/ws` | WS | Real-time updates |

### Example Usage

**Start Training:**
```bash
curl -X POST "http://localhost:8000/v1/models/train" \
  -H "Authorization: Bearer TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"phase_name": "cognate", "real_training": true}'
```

**Chat with Model:**
```bash
curl -X POST "http://localhost:8000/v1/chat" \
  -H "Authorization: Bearer TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model_id": "model-123", "message": "Hello!"}'
```

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | 0.0.0.0 | Server host |
| `PORT` | 8000 | Server port |
| `JWT_SECRET_KEY` | (required) | JWT signing secret |
| `REQUIRE_MFA` | false | Require multi-factor auth |
| `DEBUG` | false | Enable debug mode |
| `WORKERS` | 1 | Worker processes |

### Rate Limiting

| Tier | Per Minute | Per Hour | Features |
|------|------------|----------|----------|
| Standard | 60 | 1,000 | Basic access |
| Premium | 200 | 5,000 | Priority support |
| Enterprise | 500 | 10,000 | Custom features |

## 🧪 Testing

```bash
# Run integration tests
cd tests/integration/
python -m pytest test_unified_api.py -v

# Test specific functionality
pytest test_unified_api.py::TestAuthentication -v
```

## 🚢 Deployment

### Development
```bash
export DEBUG=true
python start_unified_gateway.py
```

### Production
```bash
export DEBUG=false
export WORKERS=4
export JWT_SECRET_KEY="production-secret"
python start_unified_gateway.py
```

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "start_unified_gateway.py"]
```

## 🔧 Monitoring

### Health Check
```bash
curl http://localhost:8000/health
```

Returns service status for:
- Agent Forge training pipeline
- P2P/Fog computing services
- WebSocket connections
- Overall system health

### WebSocket Monitoring
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Real-time update:', data);
};
```

## 🛠️ Development

### Project Structure
```
infrastructure/gateway/
├── unified_api_gateway.py      # Main FastAPI application
├── start_unified_gateway.py    # Production startup script
├── auth/                       # Authentication components
│   ├── __init__.py
│   └── jwt_handler.py          # JWT authentication
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

### Code Style
- Type hints for all functions
- Async/await for I/O operations
- Comprehensive error handling
- Structured logging
- OpenAPI documentation

## 📚 Documentation

- **Implementation Guide**: `docs/api/UNIFIED_API_IMPLEMENTATION_GUIDE.md`
- **OpenAPI Spec**: `docs/api/UNIFIED_API_SPECIFICATION.yaml`
- **Architecture Docs**: `docs/architecture/`
- **Integration Tests**: `tests/integration/test_unified_api.py`

## 🆘 Troubleshooting

### Common Issues

1. **Authentication Errors**
   - Verify JWT_SECRET_KEY is set
   - Check token expiration
   - Ensure required scopes

2. **Rate Limiting**
   - Check tier limits
   - Implement retry logic
   - Consider upgrading tier

3. **Service Unavailable**
   - Check dependencies
   - Review health endpoint
   - Verify configuration

### Debug Mode
```bash
export DEBUG=true
python start_unified_gateway.py
```

## 🤝 Contributing

1. Run tests: `pytest tests/integration/`
2. Format code: `black . && isort .`
3. Type checking: `mypy .`
4. Update documentation for API changes

## 📄 License

MIT License - see project root for details

---

**AIVillage Unified API Gateway** - Production-ready API integration for the AIVillage ecosystem.
