# AIVillage Client SDKs

This directory contains officially supported client libraries for the AIVillage API in multiple programming languages.

## Available SDKs

| Language | Status | Directory | Package |
|----------|--------|-----------|---------|
| **TypeScript/JavaScript** | ✅ Production | [`typescript/`](typescript/) | `aivillage-client` |
| **Python** | ✅ Production | [`python/`](python/) | `aivillage-client` |
| **Java** | ✅ Production | [`java/`](java/) | `io.aivillage:aivillage-client` |
| **Swift** | ✅ Production | [`swift/`](swift/) | `AIVillageClient` |
| **Kotlin** | ✅ Production | [`kotlin/`](kotlin/) | `io.aivillage:aivillage-client-kotlin` |
| **Go** | ✅ Production | [`go/`](go/) | `github.com/DNYoussef/AIVillage/clients/go` |
| **Rust** | ✅ Production | [`rust/`](rust/) | `aivillage-client` |
| **Web Client** | ✅ Production | [`web/`](web/) | Vue 3 Progressive Web App |

## Quick Start

### 1. Choose Your SDK

Pick the client library for your preferred programming language:

```bash
# TypeScript/JavaScript
npm install aivillage-client

# Python
pip install aivillage-client

# Java (Maven)
# <dependency>
#   <groupId>io.aivillage</groupId>
#   <artifactId>aivillage-client</artifactId>
#   <version>1.0.0</version>
# </dependency>

# Go
go get github.com/DNYoussef/AIVillage/clients/go

# Rust
cargo add aivillage-client
```

### 2. Get API Key

1. Visit [AIVillage Dashboard](https://dashboard.aivillage.io)
2. Sign up for an account
3. Generate an API key
4. Store securely (never commit to version control)

### 3. Basic Usage

**TypeScript/JavaScript:**
```typescript
import { AIVillageClient } from 'aivillage-client';

const client = AIVillageClient.withBearerToken('your-api-key');

const response = await client.chat.chat({
  message: 'Hello! How can I optimize my mobile app?',
  agent_preference: 'magi',
  mode: 'balanced'
});

console.log(response.response);
```

**Python:**
```python
import aivillage_client

configuration = aivillage_client.Configuration(
    host="https://api.aivillage.io/v1",
    access_token="your-api-key"
)

async with aivillage_client.ApiClient(configuration) as api_client:
    chat_api = aivillage_client.ChatApi(api_client)

    response = await chat_api.chat({
        "message": "Hello! How can I optimize my mobile app?",
        "agent_preference": "magi",
        "mode": "balanced"
    })

    print(response.response)
```

**Go:**
```go
import aivillage "github.com/DNYoussef/AIVillage/clients/go"

config := aivillage.NewConfiguration()
config.Host = "api.aivillage.io"
client := aivillage.NewAPIClient(config)

auth := context.WithValue(context.Background(),
    aivillage.ContextAccessToken, "your-api-key")

response, _, err := client.ChatApi.Chat(auth).ChatRequest(
    aivillage.ChatRequest{
        Message: "Hello! How can I optimize my mobile app?",
        AgentPreference: aivillage.PtrString("magi"),
        Mode: aivillage.PtrString("balanced"),
    },
).Execute()
```

## SDK Features

All SDKs include:

### 🤖 **Core AI Features**
- **Multi-Agent Chat**: Interact with specialized AI agents (King, Magi, Sage, Oracle, Navigator)
- **Advanced RAG**: Retrieval-augmented generation with Bayesian trust networks
- **Agent Management**: List, monitor, and assign tasks to AI agents
- **P2P Network**: Monitor peer-to-peer mesh network status
- **Digital Twin**: Privacy-preserving personal AI assistant

### ⚡ **Reliability & Performance**
- **Automatic Retries**: Exponential backoff for failed requests
- **Circuit Breakers**: Fail-fast behavior during service outages
- **Rate Limiting**: Built-in rate limit awareness and handling
- **Idempotency**: Safe retry of mutating operations
- **Request Deduplication**: Automatic deduplication of identical requests

### 🔒 **Security & Privacy**
- **Bearer Token Auth**: Secure API key authentication
- **HTTPS Only**: All communication over secure connections
- **Input Validation**: Client-side validation before API calls
- **Error Handling**: Structured error responses with detailed context

### 📊 **Developer Experience**
- **Type Safety**: Full type definitions in supported languages
- **Comprehensive Docs**: Complete API documentation and examples
- **Error Context**: Detailed error messages and debugging information
- **Consistent APIs**: Same patterns across all language SDKs

## Architecture

### Generation Process

All SDKs are automatically generated from the [OpenAPI 3.0 specification](../docs/api/openapi.yaml) using OpenAPI Generator:

```bash
# Generate all SDKs
python tools/generate_client_sdk.py

# Generate specific language
python tools/generate_client_sdk.py --languages typescript python java
```

### SDK Structure

Each SDK follows the same architectural pattern:

```
<language>/
├── src/                    # Generated source code
├── docs/                   # API documentation
├── examples/              # Usage examples
├── tests/                 # Unit and integration tests
├── README.md              # Language-specific guide
└── package.<ext>          # Package manifest
```

### API Client Pattern

All SDKs use a consistent client pattern:

1. **Configuration**: Set base URL, authentication, timeouts
2. **API Clients**: Specialized clients for each API category
3. **Models**: Type-safe request/response models
4. **Error Handling**: Structured error types with context

**Example Structure:**
```
AIVillageClient
├── health: HealthAPI       # /health endpoints
├── chat: ChatAPI          # /chat endpoints
├── rag: RAGAPI           # /query endpoints
├── agents: AgentsAPI      # /agents endpoints
├── p2p: P2PAPI           # /p2p endpoints
└── digitalTwin: DigitalTwinAPI  # /digital-twin endpoints
```

## Web Client

The [`web/`](web/) directory contains a full-featured Progressive Web Application built with Vue 3:

### Features
- **Accessibility First**: WCAG 2.1 AA compliant with screen reader support
- **Multi-Language**: English, Spanish, French, German, Japanese, Chinese
- **Offline Support**: Service worker with intelligent caching
- **Mobile Optimized**: Responsive design with touch-friendly interface
- **Real-time Updates**: WebSocket support for live agent interactions

### Quick Start
```bash
cd web/
npm install
npm run dev
```

Visit [web/README.md](web/README.md) for detailed documentation.

## Error Handling

All SDKs implement consistent error handling:

### HTTP Status Codes

| Status | Type | Retry? | Description |
|--------|------|--------|-------------|
| 400 | `ValidationError` | No | Invalid request format |
| 401 | `AuthenticationError` | No | Invalid or missing API key |
| 403 | `AuthorizationError` | No | Insufficient permissions |
| 404 | `NotFoundError` | No | Resource not found |
| 429 | `RateLimitError` | Yes | Rate limit exceeded |
| 500 | `ServerError` | Yes | Internal server error |
| 503 | `ServiceUnavailableError` | Yes | Service temporarily unavailable |

### Retry Logic

SDKs automatically retry requests for:
- **Rate Limits (429)**: Wait for `Retry-After` seconds
- **Server Errors (5xx)**: Exponential backoff up to 32 seconds
- **Network Errors**: Exponential backoff up to 32 seconds

Client errors (4xx) are never retried.

### Example Error Handling

**TypeScript:**
```typescript
import { APIError, RateLimitError } from 'aivillage-client';

try {
  const response = await client.chat.chat(request);
} catch (error) {
  if (error instanceof RateLimitError) {
    console.log(`Rate limited. Retry after ${error.response.retry_after}s`);
  } else if (error instanceof APIError) {
    console.log(`API Error ${error.status}: ${error.response.detail}`);
  } else {
    console.log(`Network Error: ${error.message}`);
  }
}
```

## Rate Limiting

All API endpoints have rate limits:

### Limits
- **Unauthenticated**: 100 requests per 60 seconds
- **Authenticated**: 200 requests per 60 seconds
- **Premium**: 500 requests per 60 seconds

### Headers
Rate limit information is provided in response headers:
- `X-RateLimit-Limit`: Total requests allowed in window
- `X-RateLimit-Remaining`: Requests remaining in current window
- `X-RateLimit-Reset`: Unix timestamp when window resets
- `Retry-After`: Seconds to wait before retrying (on 429 responses)

### Best Practices
1. **Check remaining requests** before making calls
2. **Implement exponential backoff** for 429 responses
3. **Cache responses** when appropriate
4. **Use idempotency keys** for safe retries
5. **Batch requests** when possible

## Idempotency

POST, PUT, and PATCH requests support idempotency keys to prevent duplicate operations:

```http
POST /v1/chat
Idempotency-Key: chat-2025-08-19-uuid-123
Content-Type: application/json

{
  "message": "Hello world"
}
```

**Benefits:**
- Safe to retry failed requests
- Prevents accidental duplicates
- Returns cached response for repeated requests
- Essential for reliable distributed systems

**Key Requirements:**
- Must be unique per request
- Minimum 16 characters
- Maximum 64 characters
- Alphanumeric and hyphens only

## Performance

### Caching Strategy

SDKs implement intelligent caching:

1. **Response Caching**: Cache GET responses based on TTL headers
2. **Request Deduplication**: Merge identical concurrent requests
3. **Connection Pooling**: Reuse HTTP connections when possible
4. **Compression**: Support gzip/deflate response compression

### Optimization Tips

1. **Reuse Client Instances**: Create once, use many times
2. **Configure Timeouts**: Set appropriate connection and read timeouts
3. **Enable Compression**: Use gzip for large responses
4. **Pool Connections**: Configure HTTP client connection pooling
5. **Monitor Performance**: Track response times and error rates

## Testing

### Unit Testing

Each SDK includes comprehensive unit tests:

```bash
# TypeScript/JavaScript
npm test

# Python
pytest

# Java
mvn test

# Go
go test ./...

# Rust
cargo test
```

### Integration Testing

Use the staging environment for integration tests:

```bash
export AIVILLAGE_API_URL=https://staging-api.aivillage.io/v1
export AIVILLAGE_API_KEY=staging-api-key
```

### Mock Testing

All SDKs support mocking for offline testing. See individual SDK documentation for language-specific examples.

## Development

### Regenerating SDKs

SDKs are automatically generated from the OpenAPI specification. To regenerate:

1. **Update OpenAPI spec**: Edit [`docs/api/openapi.yaml`](../docs/api/openapi.yaml)
2. **Validate spec**: `python tools/generate_client_sdk.py --validate-only`
3. **Generate SDKs**: `python tools/generate_client_sdk.py`
4. **Test changes**: Run SDK test suites
5. **Update docs**: Update this README and SDK guides

### Custom Modifications

If you need to customize generated SDKs:

1. **Use templates**: Create custom OpenAPI Generator templates
2. **Post-process**: Add build scripts to modify generated code
3. **Wrapper libraries**: Create thin wrappers around generated SDKs
4. **Contribute upstream**: Submit improvements to OpenAPI Generator

### Contribution Guidelines

1. **Test thoroughly**: All changes must pass existing tests
2. **Update documentation**: Keep docs synchronized with changes
3. **Follow conventions**: Match existing code style and patterns
4. **Backward compatibility**: Avoid breaking changes when possible
5. **Version appropriately**: Use semantic versioning for releases

## Versioning

SDKs follow [Semantic Versioning](https://semver.org/):

- **Major**: Breaking changes to API or SDK interface
- **Minor**: New features, backward compatible
- **Patch**: Bug fixes, backward compatible

### Version Compatibility

| SDK Version | API Version | Status |
|------------|-------------|---------|
| 1.x.x | v1 | ✅ Supported |
| 0.x.x | v1-beta | ⚠️ Deprecated |

## Support

### Documentation
- **API Reference**: [docs.aivillage.io/api](https://docs.aivillage.io/api)
- **SDK Guide**: [SDK_GUIDE.md](../docs/api/SDK_GUIDE.md)
- **OpenAPI Spec**: [openapi.yaml](../docs/api/openapi.yaml)
- **Examples**: [github.com/DNYoussef/AIVillage/tree/main/examples](https://github.com/DNYoussef/AIVillage/tree/main/examples)

### Community
- **GitHub Issues**: [github.com/DNYoussef/AIVillage/issues](https://github.com/DNYoussef/AIVillage/issues)
- **Discussions**: [github.com/DNYoussef/AIVillage/discussions](https://github.com/DNYoussef/AIVillage/discussions)
- **Discord**: [discord.gg/aivillage](https://discord.gg/aivillage)

### Contact
- **General Support**: [support@aivillage.io](mailto:support@aivillage.io)
- **API Issues**: [api-support@aivillage.io](mailto:api-support@aivillage.io)
- **Security Issues**: [security@aivillage.io](mailto:security@aivillage.io)
- **SDK Feedback**: [sdk-feedback@aivillage.io](mailto:sdk-feedback@aivillage.io)

## License

All client SDKs are released under the MIT License. See [LICENSE](../LICENSE) for details.

---

**Ready to get started?** Choose your language above and dive into the [SDK Guide](../docs/api/SDK_GUIDE.md) for detailed examples and best practices!
