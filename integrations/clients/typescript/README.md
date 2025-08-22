# AIVillage TypeScript/JavaScript SDK

A type-safe, production-ready client for the AIVillage API with built-in reliability patterns and comprehensive error handling.

## Features

- **Type Safety**: Full TypeScript definitions for all API endpoints
- **Reliability**: Automatic retries with exponential backoff and circuit breakers
- **Idempotency**: Safe retry of mutating operations with idempotency keys
- **Rate Limiting**: Built-in rate limit awareness with automatic backoff
- **Error Handling**: Structured error types with detailed context
- **Authentication**: Support for Bearer token and API key authentication

## Installation

```bash
# npm
npm install aivillage-client

# yarn
yarn add aivillage-client

# pnpm
pnpm add aivillage-client
```

## Quick Start

```typescript
import { AIVillageClient } from 'aivillage-client';

// Initialize with API key
const client = AIVillageClient.withBearerToken('your-api-key');

// Chat with AI agents
const response = await client.chat.chat({
  message: 'Hello! How can I optimize my mobile app?',
  agent_preference: 'magi',
  mode: 'balanced'
});

console.log(response.response);
```

## Configuration

### Basic Configuration

```typescript
import { AIVillageClient, Configuration } from 'aivillage-client';

const config: Configuration = {
  basePath: 'https://api.aivillage.io/v1',
  accessToken: 'your-api-key',
  timeout: 30000,
  headers: {
    'User-Agent': 'MyApp/1.0.0'
  }
};

const client = new AIVillageClient(config);
```

### Authentication Methods

```typescript
// Method 1: Bearer token (recommended)
const client = AIVillageClient.withBearerToken('your-api-key');

// Method 2: API key header
const client = AIVillageClient.withApiKey('your-api-key');

// Method 3: Custom configuration
const client = new AIVillageClient({
  basePath: 'https://api.aivillage.io/v1',
  accessToken: 'your-api-key'
});
```

## API Reference

### Chat API

Interact with AIVillage's specialized AI agents.

```typescript
// Basic chat
const response = await client.chat.chat({
  message: 'Explain federated learning',
  agent_preference: 'magi', // Research specialist
  mode: 'comprehensive'
});

// Chat with context
const contextualResponse = await client.chat.chat({
  message: 'How can I optimize this further?',
  conversation_id: response.conversation_id,
  agent_preference: 'navigator', // Routing specialist
  mode: 'balanced',
  user_context: {
    device_type: 'mobile',
    battery_level: 75,
    network_type: 'wifi'
  }
});
```

**Available Agents:**
- `king`: Coordination and oversight
- `magi`: Research and analysis
- `sage`: Deep knowledge and wisdom
- `oracle`: Predictions and forecasting
- `navigator`: Routing and optimization
- `any`: Auto-select best agent (default)

**Response Modes:**
- `fast`: Quick responses with minimal processing
- `balanced`: Good balance of speed and thoroughness (default)
- `comprehensive`: Detailed analysis with full context
- `creative`: Innovative and creative responses

### RAG API

Advanced knowledge retrieval with Bayesian trust networks.

```typescript
// Basic query
const result = await client.rag.processQuery({
  query: 'What are the latest trends in federated learning?',
  mode: 'analytical',
  include_sources: true,
  max_results: 10
});

console.log('Response:', result.response);
console.log('Confidence:', result.metadata.bayesian_confidence);

// Process sources
result.sources.forEach(source => {
  console.log(`${source.title}: ${source.confidence}`);
});
```

**Query Modes:**
- `fast`: Quick retrieval with basic processing
- `balanced`: Standard retrieval with moderate analysis
- `comprehensive`: Deep analysis with full context
- `creative`: Creative connections and insights
- `analytical`: Systematic analysis and reasoning

### Agents API

Manage and monitor AI agents.

```typescript
// List available agents
const agents = await client.agents.listAgents({
  category: 'knowledge',
  available_only: true
});

agents.agents.forEach(agent => {
  console.log(`${agent.name}: ${agent.status} (${agent.current_load}% load)`);
});

// Assign task to specific agent
const taskResult = await client.agents.assignAgentTask('magi-001', {
  task_description: 'Research quantum computing applications in AI',
  priority: 'high',
  timeout_seconds: 600,
  context: {
    domain: 'quantum_ai',
    depth: 'comprehensive'
  }
});
```

### P2P API

Monitor peer-to-peer mesh network status.

```typescript
// Get network status
const status = await client.p2p.getP2PStatus();
console.log(`Network: ${status.status}, Peers: ${status.peer_count}`);

// List connected peers
const peers = await client.p2p.listPeers('bitchat');
peers.peers.forEach(peer => {
  console.log(`${peer.id}: ${peer.transport} (${peer.status})`);
});
```

### Digital Twin API

Privacy-preserving personal AI assistant.

```typescript
// Get profile
const profile = await client.digitalTwin.getDigitalTwinProfile();
console.log(`Model size: ${profile.model_size_mb}MB`);
console.log(`Accuracy: ${profile.learning_stats.accuracy_score}`);

// Update with new data
await client.digitalTwin.updateDigitalTwinData({
  data_type: 'interaction',
  data_points: [{
    timestamp: new Date().toISOString(),
    content: {
      interaction_type: 'chat',
      user_satisfaction: 0.85
    },
    prediction_accuracy: 0.92
  }]
});
```

## Error Handling

### Error Types

```typescript
import { APIError, RateLimitError } from 'aivillage-client';

try {
  const response = await client.chat.chat(request);
} catch (error) {
  if (error instanceof RateLimitError) {
    console.log(`Rate limited. Retry after ${error.response.retry_after}s`);
    // Wait and retry
    await new Promise(resolve =>
      setTimeout(resolve, error.response.retry_after * 1000)
    );
  } else if (error instanceof APIError) {
    console.log(`API Error ${error.status}: ${error.response.detail}`);
    console.log(`Request ID: ${error.response.request_id}`);
  } else {
    console.log(`Network Error: ${error.message}`);
  }
}
```

### Retry Pattern

```typescript
async function resilientRequest<T>(
  requestFn: () => Promise<T>,
  maxRetries: number = 3
): Promise<T> {
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      return await requestFn();
    } catch (error) {
      if (error instanceof RateLimitError) {
        // Wait for rate limit reset
        await new Promise(resolve =>
          setTimeout(resolve, error.response.retry_after * 1000)
        );
      } else if (error instanceof APIError && error.status >= 500) {
        // Exponential backoff for server errors
        const backoffMs = Math.min(1000 * Math.pow(2, attempt), 10000);
        await new Promise(resolve => setTimeout(resolve, backoffMs));
      } else {
        // Don't retry client errors
        throw error;
      }
    }
  }

  throw new Error('Max retries exceeded');
}

// Usage
const response = await resilientRequest(() =>
  client.chat.chat({
    message: 'Hello world'
  })
);
```

## Advanced Usage

### Idempotency

```typescript
// Generate idempotency key
const idempotencyKey = `chat-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

const response = await client.chat.chat({
  message: 'Process this important request',
  mode: 'comprehensive'
}, { idempotencyKey });

// Repeating the same request will return cached response
const cachedResponse = await client.chat.chat({
  message: 'Process this important request',
  mode: 'comprehensive'
}, { idempotencyKey });

console.log(response.response === cachedResponse.response); // true
```

### Custom Headers

```typescript
const response = await client.chat.chat({
  message: 'Hello'
}, {
  headers: {
    'X-User-ID': 'user123',
    'X-Session-ID': 'session456'
  }
});
```

### Request Timeout

```typescript
// Override timeout for specific request
const client = new AIVillageClient({
  basePath: 'https://api.aivillage.io/v1',
  accessToken: 'your-api-key',
  timeout: 60000 // 60 seconds for long-running operations
});
```

## TypeScript Support

### Type Definitions

```typescript
import type {
  ChatRequest,
  ChatResponse,
  QueryRequest,
  QueryResponse,
  Agent,
  AgentTaskRequest,
  HealthResponse
} from 'aivillage-client';

// Strongly typed request
const request: ChatRequest = {
  message: 'Hello',
  agent_preference: 'magi', // Type-checked
  mode: 'balanced', // Type-checked
  user_context: {
    device_type: 'mobile', // Type-checked
    battery_level: 75,
    network_type: 'wifi' // Type-checked
  }
};

// Strongly typed response
const response: ChatResponse = await client.chat.chat(request);
```

### Generic Helper Types

```typescript
// Custom response handler type
type ResponseHandler<T> = (response: T) => void | Promise<void>;

async function handleChatResponse(
  message: string,
  handler: ResponseHandler<ChatResponse>
): Promise<void> {
  const response = await client.chat.chat({ message });
  await handler(response);
}

// Usage with type inference
await handleChatResponse('Hello', (response) => {
  // response is automatically typed as ChatResponse
  console.log(response.agent_used);
});
```

## Browser Usage

### ES Modules

```typescript
// Modern browsers with ES module support
import { AIVillageClient } from 'aivillage-client';

const client = AIVillageClient.withBearerToken('your-api-key');
```

### CommonJS

```javascript
// Node.js or bundlers
const { AIVillageClient } = require('aivillage-client');

const client = AIVillageClient.withBearerToken('your-api-key');
```

### CDN Usage

```html
<!DOCTYPE html>
<html>
<head>
  <script src="https://unpkg.com/aivillage-client@1.0.0/dist/aivillage-client.min.js"></script>
</head>
<body>
  <script>
    const client = AIVillageClient.withBearerToken('your-api-key');

    client.chat.chat({
      message: 'Hello from browser!'
    }).then(response => {
      console.log(response.response);
    });
  </script>
</body>
</html>
```

## Node.js Considerations

### Environment Variables

```typescript
// Use environment variables for API keys
const client = AIVillageClient.withBearerToken(process.env.AIVILLAGE_API_KEY!);

// Support multiple environments
const apiUrl = process.env.NODE_ENV === 'production'
  ? 'https://api.aivillage.io/v1'
  : 'https://staging-api.aivillage.io/v1';

const client = new AIVillageClient({
  basePath: apiUrl,
  accessToken: process.env.AIVILLAGE_API_KEY!
});
```

### HTTP Agent Configuration

```typescript
import https from 'https';

// Custom HTTPS agent for Node.js
const client = new AIVillageClient({
  basePath: 'https://api.aivillage.io/v1',
  accessToken: process.env.AIVILLAGE_API_KEY!
}, {
  // Custom fetch configuration for Node.js
  agent: new https.Agent({
    keepAlive: true,
    maxSockets: 10
  })
});
```

## Testing

### Mock Client

```typescript
import { jest } from '@jest/globals';

// Mock the entire client
jest.mock('aivillage-client', () => ({
  AIVillageClient: {
    withBearerToken: jest.fn(() => ({
      chat: {
        chat: jest.fn().mockResolvedValue({
          response: 'Mocked response',
          agent_used: 'mock-agent',
          processing_time_ms: 100,
          conversation_id: 'mock-conv-123',
          metadata: {
            confidence: 0.95,
            features_used: ['mock-feature']
          }
        })
      },
      rag: {
        processQuery: jest.fn().mockResolvedValue({
          query_id: 'mock-query-123',
          response: 'Mocked query response',
          sources: [],
          metadata: {
            processing_time_ms: 150,
            mode: 'balanced',
            bayesian_confidence: 0.88
          }
        })
      }
    }))
  }
}));
```

### Integration Testing

```typescript
import { AIVillageClient } from 'aivillage-client';

describe('AIVillage Integration Tests', () => {
  let client: AIVillageClient;

  beforeAll(() => {
    // Use test API key and staging environment
    client = new AIVillageClient({
      basePath: 'https://staging-api.aivillage.io/v1',
      accessToken: process.env.AIVILLAGE_TEST_API_KEY!
    });
  });

  test('chat with agents', async () => {
    const response = await client.chat.chat({
      message: 'Test message',
      mode: 'fast'
    });

    expect(response.response).toBeDefined();
    expect(response.agent_used).toBeDefined();
    expect(response.processing_time_ms).toBeGreaterThan(0);
  });

  test('handle rate limiting', async () => {
    // Test rate limiting behavior
    const requests = Array(10).fill(null).map(() =>
      client.chat.chat({ message: 'Rate limit test' })
    );

    const responses = await Promise.allSettled(requests);

    // Some requests should succeed, others may be rate limited
    expect(responses.some(r => r.status === 'fulfilled')).toBe(true);
  });
});
```

## Performance

### Connection Pooling

The SDK automatically handles HTTP connection pooling for optimal performance. In Node.js environments, you can configure the underlying HTTP agent:

```typescript
import https from 'https';

const client = new AIVillageClient({
  basePath: 'https://api.aivillage.io/v1',
  accessToken: process.env.AIVILLAGE_API_KEY!
}, {
  // Configure HTTP agent
  agent: new https.Agent({
    keepAlive: true,
    maxSockets: 20,
    timeout: 30000
  })
});
```

### Batch Operations

```typescript
// Process multiple queries concurrently
const queries = [
  'Explain machine learning',
  'What is federated learning?',
  'How do neural networks work?'
];

const responses = await Promise.all(
  queries.map(query =>
    client.rag.processQuery({
      query,
      mode: 'fast',
      max_results: 3
    })
  )
);

responses.forEach((response, index) => {
  console.log(`Query ${index + 1}: ${response.response}`);
});
```

## Troubleshooting

### Common Issues

**Network Timeouts:**
```typescript
// Increase timeout for slow networks
const client = new AIVillageClient({
  basePath: 'https://api.aivillage.io/v1',
  accessToken: process.env.AIVILLAGE_API_KEY!,
  timeout: 60000 // 60 seconds
});
```

**CORS in Browser:**
```typescript
// Ensure your domain is allowlisted in AIVillage dashboard
// or use a proxy server for development
const client = new AIVillageClient({
  basePath: '/api/proxy', // Proxy to avoid CORS
  accessToken: process.env.AIVILLAGE_API_KEY!
});
```

**Rate Limiting:**
```typescript
// Implement exponential backoff
async function rateLimitSafeRequest<T>(requestFn: () => Promise<T>): Promise<T> {
  let delay = 1000; // Start with 1 second

  while (true) {
    try {
      return await requestFn();
    } catch (error) {
      if (error instanceof RateLimitError) {
        await new Promise(resolve => setTimeout(resolve, delay));
        delay = Math.min(delay * 2, 60000); // Cap at 1 minute
        continue;
      }
      throw error;
    }
  }
}
```

### Debug Logging

```typescript
// Enable debug logging
const client = new AIVillageClient({
  basePath: 'https://api.aivillage.io/v1',
  accessToken: process.env.AIVILLAGE_API_KEY!
}, {
  // Add request/response logging
  fetch: async (input, init) => {
    console.log('Request:', { url: input, options: init });
    const response = await fetch(input, init);
    console.log('Response:', { status: response.status, headers: Object.fromEntries(response.headers) });
    return response;
  }
});
```

## Support

- **Documentation**: [docs.aivillage.io](https://docs.aivillage.io)
- **API Reference**: [docs.aivillage.io/api](https://docs.aivillage.io/api)
- **GitHub Issues**: [github.com/DNYoussef/AIVillage/issues](https://github.com/DNYoussef/AIVillage/issues)
- **Discord**: [discord.gg/aivillage](https://discord.gg/aivillage)

## License

MIT License - see [LICENSE](../../LICENSE) for details.
