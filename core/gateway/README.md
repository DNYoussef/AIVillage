# AIVillage Unified Production Gateway

This is the consolidated, production-ready HTTP API gateway that unifies all HTTP API capabilities from across the AIVillage codebase into a single, high-performance FastAPI application.

## Architecture

```
User Request → API Gateway → Agent Controller → Knowledge System → Response
```

## Key Features

### 🚀 Performance
- **<100ms health check response time** (performance target)
- Async/await throughout for maximum concurrency
- Connection pooling and keepalive optimization
- Prometheus metrics for monitoring and optimization

### 🔒 Security
- Comprehensive security headers (CSP, HSTS, etc.)
- Multi-tier rate limiting with IP-based tracking
- Input validation with XSS/SQL injection prevention
- API key authentication with configurable policies
- CORS configuration with environment-specific policies

### 🛠 Production Ready
- Structured error handling with proper HTTP status codes
- Health check cascade probing of downstream services
- Graceful degradation when services are unavailable
- Environment-based configuration with validation
- Comprehensive logging and observability

### 📊 Monitoring
- Prometheus metrics integration
- Request/response duration histograms
- Rate limiting and security event tracking
- Downstream service health monitoring
- Performance bottleneck identification

## Installation

1. Install dependencies:
```bash
cd core/gateway
pip install -r requirements.txt
```

2. Configure environment (copy and modify):
```bash
cp config.yaml config.production.yaml
# Edit config.production.yaml for your environment
```

3. Set environment variables:
```bash
export ENVIRONMENT=production
export API_KEY=your-secure-api-key
export SECRET_KEY=your-secure-secret-key
export TWIN_URL=http://your-twin-service:8001
export CORS_ORIGINS=https://your-domain.com,https://api.your-domain.com
```

## Running

### Development
```bash
python server.py
```

### Production
```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker
```bash
docker build -t aivillage-gateway .
docker run -p 8000:8000 -e ENVIRONMENT=production aivillage-gateway
```

## API Endpoints

### Core Endpoints

| Endpoint | Method | Description | Auth Required |
|----------|--------|-------------|---------------|
| `/healthz` | GET | Health check with cascade probing | No |
| `/v1/query` | POST | Main query endpoint | Yes |
| `/v1/chat` | POST | Chat interface (proxies to Twin) | Yes |
| `/v1/upload` | POST | File upload processing | Yes |
| `/metrics` | GET | Prometheus metrics | No |
| `/` | GET | Gateway information | No |

### Health Check Response
```json
{
  "status": "healthy|degraded|unhealthy",
  "timestamp": "2025-01-22T10:30:00Z",
  "services": {
    "twin": {
      "status": "healthy",
      "status_code": 200,
      "response_time_ms": 45.2
    },
    "agent_controller": {
      "status": "healthy",
      "status_code": 200
    }
  },
  "gateway": {
    "uptime_seconds": 3600,
    "environment": "production",
    "response_time_ms": 67.8
  }
}
```

### Query Request Format
```json
{
  "query": "What is the capital of France?",
  "session_id": "optional-session-id",
  "options": {
    "max_tokens": 1000,
    "temperature": 0.7
  }
}
```

## Configuration

The gateway uses a hierarchical configuration system:

1. **Default values** (in code)
2. **YAML configuration** (config.yaml)
3. **Environment variables** (highest priority)

### Key Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ENVIRONMENT` | Deployment environment | `development` |
| `GATEWAY_HOST` | Server bind address | `0.0.0.0` |
| `GATEWAY_PORT` | Server port | `8000` |
| `API_KEY` | Authentication API key | `dev-gateway-key-12345` |
| `SECRET_KEY` | JWT/session secret | `dev-secret-key-change-in-production` |
| `RATE_LIMIT_REQUESTS` | Requests per window | `100` |
| `RATE_LIMIT_WINDOW` | Rate limit window (seconds) | `60` |
| `CORS_ORIGINS` | Allowed CORS origins | `*` |
| `TWIN_URL` | Twin service URL | `http://localhost:8001` |
| `AGENT_CONTROLLER_URL` | Agent controller URL | `http://localhost:8002` |
| `KNOWLEDGE_SYSTEM_URL` | Knowledge system URL | `http://localhost:8003` |
| `ENABLE_METRICS` | Enable Prometheus metrics | `true` |
| `ENABLE_FILE_UPLOAD` | Enable file upload | `true` |
| `MAX_FILE_SIZE` | Max upload size (bytes) | `10485760` |

## Security Considerations

### Production Checklist

- [ ] Set strong `API_KEY` and `SECRET_KEY` environment variables
- [ ] Configure specific `CORS_ORIGINS` (not wildcard)
- [ ] Set `ENVIRONMENT=production`
- [ ] Disable debug mode (`DEBUG=false`)
- [ ] Configure proper SSL/TLS termination (reverse proxy)
- [ ] Set up rate limiting monitoring and alerting
- [ ] Review and adjust rate limits for your use case
- [ ] Implement API key rotation strategy
- [ ] Set up log aggregation and monitoring
- [ ] Configure health check alerts

### Security Headers Applied

- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Strict-Transport-Security: max-age=63072000; includeSubDomains; preload`
- `Content-Security-Policy: [restrictive policy]`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Permissions-Policy: [minimal permissions]`

## Performance Targets

### Health Check Performance
- **Target**: <100ms response time
- **Monitoring**: Prometheus `gateway_health_check_duration_seconds` histogram
- **Alerting**: Set up alerts if P95 > 100ms

### Request Performance
- **Target**: P95 < 2s for query requests
- **Monitoring**: Prometheus `gateway_request_duration_seconds` histogram
- **Optimization**: Connection pooling, async processing

## Monitoring & Observability

### Prometheus Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `gateway_requests_total` | Counter | Total HTTP requests by method/endpoint/status |
| `gateway_request_duration_seconds` | Histogram | Request duration distribution |
| `gateway_health_check_duration_seconds` | Histogram | Health check response time |

### Log Levels
- `ERROR`: Service failures, security events
- `WARN`: Performance issues, rate limiting
- `INFO`: Service lifecycle, configuration
- `DEBUG`: Request/response details (dev only)

## Migration from Legacy Gateways

This consolidated gateway replaces multiple existing FastAPI implementations:

### Deprecated Services
1. `infrastructure/gateway/server.py` - Development server (deprecated)
2. `infrastructure/shared/bin/server.py` - Duplicate of above
3. `experiments/services/services/gateway/app.py` - Atlantis Gateway (merged)
4. Various other FastAPI servers throughout the codebase

### Migration Steps

1. **Stop legacy services**:
   ```bash
   # Stop all existing FastAPI servers
   pkill -f "uvicorn.*gateway"
   pkill -f "python.*server.py"
   ```

2. **Update client configurations**:
   ```bash
   # Update all client code to point to unified gateway
   # Old: http://localhost:8080/query
   # New: http://localhost:8000/v1/query
   ```

3. **Migrate environment variables**:
   ```bash
   # Rename/consolidate environment variables
   # See configuration section above
   ```

4. **Test thoroughly**:
   ```bash
   # Run integration tests
   pytest core/gateway/tests/

   # Load test health check performance
   curl -w "@curl-format.txt" http://localhost:8000/healthz
   ```

5. **Deploy gradually**:
   - Deploy to staging environment first
   - Validate all endpoints and performance
   - Monitor health checks and metrics
   - Roll out to production with blue-green deployment

### Import Path Updates

Update your Python imports:

```python
# OLD - Multiple different imports
from infrastructure.gateway.server import app
from experiments.services.services.gateway.app import app

# NEW - Unified import
from core.gateway.server import app
```

## Troubleshooting

### Common Issues

1. **Health check timeouts**:
   - Check downstream service availability
   - Adjust `HEALTH_CHECK_TIMEOUT` environment variable
   - Monitor `gateway_health_check_duration_seconds` metric

2. **Rate limiting false positives**:
   - Review rate limit configuration
   - Check for proxy/load balancer IP forwarding
   - Monitor `X-Forwarded-For` and `X-Real-IP` headers

3. **CORS errors**:
   - Verify `CORS_ORIGINS` configuration
   - Check client request headers
   - Review browser developer tools

4. **Authentication failures**:
   - Verify `API_KEY` environment variable
   - Check `Authorization: Bearer <token>` header format
   - Review gateway logs for details

### Debug Mode

Enable detailed logging:
```bash
export DEBUG=true
export LOG_LEVEL=DEBUG
python server.py
```

### Performance Analysis

Monitor key metrics:
```bash
# Check health check performance
curl -w "Response time: %{time_total}s\n" http://localhost:8000/healthz

# View metrics
curl http://localhost:8000/metrics | grep gateway_
```

## Contributing

When adding new endpoints or modifying existing ones:

1. Follow FastAPI best practices
2. Add comprehensive type hints
3. Implement proper error handling
4. Add Prometheus metrics
5. Update this documentation
6. Add integration tests
7. Verify security implications

## Support

For issues and questions:
1. Check this README and configuration
2. Review application logs
3. Check Prometheus metrics
4. File an issue with full context and logs
