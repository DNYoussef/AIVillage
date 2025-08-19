# AIVillage Client SDK Guide

## Overview

This guide covers all officially supported client SDKs for the AIVillage API, including installation, configuration, and usage examples. All SDKs are automatically generated from the [OpenAPI 3.0 specification](./openapi.yaml) to ensure consistency and type safety.

**Key Benefits:**
- **Type Safety**: Full type definitions in supported languages
- **Reliability**: Built-in retries, circuit breakers, and idempotency
- **Consistency**: Same patterns and features across all languages
- **Production Ready**: Comprehensive error handling and monitoring

## Available SDKs

| Language | Package Name | Version | Documentation |
|----------|--------------|---------|---------------|
| **TypeScript/JavaScript** | `aivillage-client` | 1.0.0 | [TypeScript Guide](#typescript--javascript) |
| **Python** | `aivillage-client` | 1.0.0 | [Python Guide](#python) |
| **Java** | `io.aivillage:aivillage-client` | 1.0.0 | [Java Guide](#java) |
| **Swift** | `AIVillageClient` | 1.0.0 | [Swift Guide](#swift--ios) |
| **Kotlin** | `io.aivillage:aivillage-client-kotlin` | 1.0.0 | [Kotlin Guide](#kotlin--android) |
| **Go** | `github.com/DNYoussef/AIVillage/clients/go` | 1.0.0 | [Go Guide](#go) |
| **Rust** | `aivillage-client` | 1.0.0 | [Rust Guide](#rust) |

## Quick Start

### 1. Get Your API Key

Visit the [AIVillage Dashboard](https://dashboard.aivillage.io) to obtain your API key:

1. Sign up for an account
2. Navigate to **API Keys** section
3. Generate a new API key
4. Store it securely (never commit to version control)

### 2. Choose Your SDK

Pick the SDK for your preferred language and follow the installation guide below.

## TypeScript / JavaScript

### Installation

```bash
# npm
npm install aivillage-client

# yarn
yarn add aivillage-client

# pnpm
pnpm add aivillage-client
```

### Basic Usage

```typescript
import { AIVillageClient, Configuration } from 'aivillage-client';

// Initialize client with API key
const client = AIVillageClient.withBearerToken('your-api-key', {
  basePath: 'https://api.aivillage.io/v1'
});

// Chat with AI agents
async function chatExample() {
  try {
    const response = await client.chat.chat({
      message: 'Hello! Can you help me with mobile app optimization?',
      agent_preference: 'magi', // Research specialist
      mode: 'comprehensive',
      user_context: {
        device_type: 'mobile',
        battery_level: 75,
        network_type: 'wifi'
      }
    });

    console.log(`${response.agent_used}: ${response.response}`);
    console.log(`Processing time: ${response.processing_time_ms}ms`);
  } catch (error) {
    if (error instanceof RateLimitError) {
      console.log(`Rate limited. Retry after ${error.response.retry_after}s`);
    } else {
      console.error('Chat error:', error.message);
    }
  }
}

// RAG query processing
async function ragExample() {
  const result = await client.rag.processQuery({
    query: 'What are the latest trends in federated learning?',
    mode: 'analytical',
    include_sources: true,
    max_results: 5
  });

  console.log('Response:', result.response);
  console.log('Sources:', result.sources);
  console.log('Confidence:', result.metadata.bayesian_confidence);
}
```

### Advanced Features

```typescript
// Error handling with retry logic
import { APIError, RateLimitError } from 'aivillage-client';

async function resilientRequest() {
  let retries = 3;

  while (retries > 0) {
    try {
      return await client.agents.listAgents({
        category: 'knowledge',
        available_only: true
      });
    } catch (error) {
      if (error instanceof RateLimitError) {
        await new Promise(resolve =>
          setTimeout(resolve, error.response.retry_after * 1000)
        );
        retries--;
      } else if (error instanceof APIError && error.status >= 500) {
        await new Promise(resolve => setTimeout(resolve, 1000 * (4 - retries)));
        retries--;
      } else {
        throw error;
      }
    }
  }
}

// Idempotency for safe retries
const idempotencyKey = `task-${Date.now()}-${Math.random()}`;
await client.agents.assignAgentTask('magi-001', {
  task_description: 'Research quantum computing applications',
  priority: 'high',
  timeout_seconds: 600
}, { idempotencyKey });
```

## Python

### Installation

```bash
# pip
pip install aivillage-client

# pipenv
pipenv install aivillage-client

# poetry
poetry add aivillage-client
```

### Basic Usage

```python
import aivillage_client
from aivillage_client.rest import ApiException
from aivillage_client.models import ChatRequest, QueryRequest

# Configure API client
configuration = aivillage_client.Configuration(
    host="https://api.aivillage.io/v1",
    access_token="your-api-key"
)

async def chat_example():
    async with aivillage_client.ApiClient(configuration) as api_client:
        chat_api = aivillage_client.ChatApi(api_client)

        try:
            request = ChatRequest(
                message="How can I implement federated learning on mobile devices?",
                agent_preference="navigator",  # Routing specialist
                mode="balanced",
                user_context={
                    "device_type": "mobile",
                    "battery_level": 60,
                    "network_type": "cellular"
                }
            )

            response = await chat_api.chat(request)

            print(f"{response.agent_used}: {response.response}")
            print(f"Confidence: {response.metadata.confidence:.2f}")

        except ApiException as e:
            print(f"API Exception: {e}")

# RAG query processing
async def rag_example():
    async with aivillage_client.ApiClient(configuration) as api_client:
        rag_api = aivillage_client.RAGApi(api_client)

        query = QueryRequest(
            query="Explain differential privacy in federated learning",
            mode="comprehensive",
            include_sources=True,
            max_results=10
        )

        result = await rag_api.process_query(query)

        print("Response:", result.response)
        for source in result.sources:
            print(f"Source: {source.title} (confidence: {source.confidence:.2f})")

# Run async examples
import asyncio
asyncio.run(chat_example())
asyncio.run(rag_example())
```

### Context Manager Pattern

```python
# Recommended pattern for resource management
from contextlib import asynccontextmanager

@asynccontextmanager
async def aivillage_client_context():
    async with aivillage_client.ApiClient(configuration) as api_client:
        yield {
            'chat': aivillage_client.ChatApi(api_client),
            'rag': aivillage_client.RAGApi(api_client),
            'agents': aivillage_client.AgentsApi(api_client),
            'p2p': aivillage_client.P2PApi(api_client),
        }

async def use_multiple_apis():
    async with aivillage_client_context() as apis:
        # Get available agents
        agents = await apis['agents'].list_agents()

        # Process a query
        result = await apis['rag'].process_query(QueryRequest(
            query="Compare different AI architectures"
        ))

        # Chat with preferred agent
        chat_response = await apis['chat'].chat(ChatRequest(
            message="Can you elaborate on the architectures you mentioned?",
            conversation_id=result.query_id  # Continue conversation
        ))
```

## Java

### Installation

**Maven**
```xml
<dependency>
    <groupId>io.aivillage</groupId>
    <artifactId>aivillage-client</artifactId>
    <version>1.0.0</version>
</dependency>
```

**Gradle**
```gradle
implementation 'io.aivillage:aivillage-client:1.0.0'
```

### Basic Usage

```java
import io.aivillage.client.*;
import io.aivillage.client.api.*;
import io.aivillage.client.model.*;

public class AIVillageExample {
    public static void main(String[] args) {
        // Configure client
        ApiClient defaultClient = new ApiClient();
        defaultClient.setBasePath("https://api.aivillage.io/v1");
        defaultClient.setBearerToken("your-api-key");

        // Chat example
        chatExample(defaultClient);

        // RAG example
        ragExample(defaultClient);
    }

    private static void chatExample(ApiClient client) {
        ChatApi chatApi = new ChatApi(client);

        try {
            ChatRequest request = new ChatRequest()
                .message("What are the security considerations for mobile AI?")
                .agentPreference(ChatRequest.AgentPreferenceEnum.SAGE)
                .mode(ChatRequest.ModeEnum.COMPREHENSIVE)
                .userContext(new ChatRequestUserContext()
                    .deviceType(ChatRequestUserContext.DeviceTypeEnum.MOBILE)
                    .batteryLevel(80)
                    .networkType(ChatRequestUserContext.NetworkTypeEnum.WIFI));

            ChatResponse response = chatApi.chat(request);

            System.out.println("Agent: " + response.getAgentUsed());
            System.out.println("Response: " + response.getResponse());
            System.out.println("Processing time: " + response.getProcessingTimeMs() + "ms");

        } catch (ApiException e) {
            System.err.println("Chat API error: " + e.getResponseBody());
        }
    }

    private static void ragExample(ApiClient client) {
        RAGApi ragApi = new RAGApi(client);

        try {
            QueryRequest request = new QueryRequest()
                .query("How do I optimize model inference on edge devices?")
                .mode(QueryRequest.ModeEnum.ANALYTICAL)
                .includeSources(true)
                .maxResults(5);

            QueryResponse response = ragApi.processQuery(request);

            System.out.println("Response: " + response.getResponse());
            System.out.println("Bayesian confidence: " + response.getMetadata().getBayesianConfidence());

            for (QueryResponseSourcesInner source : response.getSources()) {
                System.out.println("Source: " + source.getTitle() +
                    " (confidence: " + source.getConfidence() + ")");
            }

        } catch (ApiException e) {
            System.err.println("RAG API error: " + e.getResponseBody());
        }
    }
}
```

### Async Usage with CompletableFuture

```java
import java.util.concurrent.CompletableFuture;

public class AsyncExample {
    public static void asyncChatExample(ApiClient client) {
        ChatApi chatApi = new ChatApi(client);

        CompletableFuture.supplyAsync(() -> {
            try {
                ChatRequest request = new ChatRequest()
                    .message("Explain federated learning benefits")
                    .mode(ChatRequest.ModeEnum.BALANCED);

                return chatApi.chat(request);
            } catch (ApiException e) {
                throw new RuntimeException(e);
            }
        })
        .thenAccept(response -> {
            System.out.println("Async response: " + response.getResponse());
        })
        .exceptionally(throwable -> {
            System.err.println("Async error: " + throwable.getMessage());
            return null;
        });
    }
}
```

## Swift / iOS

### Installation

**Swift Package Manager**
```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/DNYoussef/AIVillage", from: "1.0.0")
]
```

**CocoaPods**
```ruby
# Podfile
pod 'AIVillageClient', '~> 1.0.0'
```

### Basic Usage

```swift
import AIVillageClient
import Foundation

class AIVillageManager {
    private let client: DefaultAPI

    init(apiKey: String) {
        let config = URLSessionConfiguration.default
        config.httpAdditionalHeaders = [
            "Authorization": "Bearer \(apiKey)",
            "Content-Type": "application/json"
        ]

        client = DefaultAPI(basePath: "https://api.aivillage.io/v1")
    }

    func chatWithAgent(message: String) async throws -> ChatResponse {
        let request = ChatRequest(
            message: message,
            agentPreference: .magi,
            mode: .balanced,
            userContext: ChatRequest.UserContext(
                deviceType: .mobile,
                batteryLevel: UIDevice.current.batteryLevel * 100,
                networkType: .wifi
            )
        )

        return try await client.chat(chatRequest: request)
    }

    func processQuery(_ query: String) async throws -> QueryResponse {
        let request = QueryRequest(
            query: query,
            mode: .comprehensive,
            includeSources: true,
            maxResults: 10
        )

        return try await client.processQuery(queryRequest: request)
    }
}

// Usage example
class ViewController: UIViewController {
    private let aiManager = AIVillageManager(apiKey: "your-api-key")

    @IBAction func sendMessageTapped() {
        Task {
            do {
                let response = try await aiManager.chatWithAgent(
                    message: "How can I optimize my iOS app's ML model?"
                )

                DispatchQueue.main.async {
                    self.displayResponse(response.response)
                }
            } catch {
                print("Error: \(error)")
            }
        }
    }

    private func displayResponse(_ text: String) {
        // Update UI with response
    }
}
```

### Error Handling

```swift
extension AIVillageManager {
    func chatWithRetry(message: String, maxRetries: Int = 3) async throws -> ChatResponse {
        for attempt in 1...maxRetries {
            do {
                return try await chatWithAgent(message: message)
            } catch let error as APIError where error.statusCode == 429 {
                // Rate limited - wait and retry
                let retryAfter = error.retryAfter ?? 60
                try await Task.sleep(nanoseconds: UInt64(retryAfter * 1_000_000_000))

                if attempt == maxRetries {
                    throw error
                }
            } catch let error as APIError where error.statusCode >= 500 {
                // Server error - exponential backoff
                let backoffSeconds = min(pow(2.0, Double(attempt)), 32.0)
                try await Task.sleep(nanoseconds: UInt64(backoffSeconds * 1_000_000_000))

                if attempt == maxRetries {
                    throw error
                }
            } catch {
                // Client error or network error - don't retry
                throw error
            }
        }

        fatalError("Unreachable")
    }
}
```

## Kotlin / Android

### Installation

**Gradle (app-level)**
```kotlin
dependencies {
    implementation 'io.aivillage:aivillage-client-kotlin:1.0.0'
    implementation 'com.squareup.okhttp3:okhttp:4.11.0'
    implementation 'com.google.code.gson:gson:2.10.1'
}
```

### Basic Usage

```kotlin
import io.aivillage.client.*
import io.aivillage.client.apis.*
import io.aivillage.client.models.*
import kotlinx.coroutines.*

class AIVillageClient(private val apiKey: String) {
    private val client = ApiClient().apply {
        basePath = "https://api.aivillage.io/v1"
        setBearerToken(apiKey)
    }

    private val chatApi = ChatApi(client)
    private val ragApi = RAGApi(client)
    private val agentsApi = AgentsApi(client)

    suspend fun sendMessage(
        message: String,
        agentPreference: ChatRequest.AgentPreference = ChatRequest.AgentPreference.any,
        mode: ChatRequest.Mode = ChatRequest.Mode.balanced
    ): ChatResponse = withContext(Dispatchers.IO) {
        val request = ChatRequest(
            message = message,
            agentPreference = agentPreference,
            mode = mode,
            userContext = ChatRequest.UserContext(
                deviceType = ChatRequest.UserContext.DeviceType.mobile,
                batteryLevel = getBatteryLevel(),
                networkType = getNetworkType()
            )
        )

        chatApi.chat(request)
    }

    suspend fun queryKnowledge(
        query: String,
        mode: QueryRequest.Mode = QueryRequest.Mode.balanced
    ): QueryResponse = withContext(Dispatchers.IO) {
        val request = QueryRequest(
            query = query,
            mode = mode,
            includeSources = true,
            maxResults = 10
        )

        ragApi.processQuery(request)
    }

    private fun getBatteryLevel(): Int {
        // Android battery level detection
        return 75 // Placeholder
    }

    private fun getNetworkType(): ChatRequest.UserContext.NetworkType {
        // Android network detection
        return ChatRequest.UserContext.NetworkType.wifi
    }
}

// Android Activity usage
class ChatActivity : AppCompatActivity() {
    private lateinit var aiClient: AIVillageClient

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        aiClient = AIVillageClient(BuildConfig.AIVILLAGE_API_KEY)

        setupChat()
    }

    private fun setupChat() {
        lifecycleScope.launch {
            try {
                val response = aiClient.sendMessage(
                    message = "How can I optimize battery usage in my Android app?",
                    agentPreference = ChatRequest.AgentPreference.navigator
                )

                displayMessage(response.agentUsed, response.response)

            } catch (e: ApiException) {
                when (e.code) {
                    429 -> showRateLimitError(e)
                    else -> showGenericError(e)
                }
            }
        }
    }

    private fun displayMessage(agent: String, message: String) {
        // Update UI
    }

    private fun showRateLimitError(e: ApiException) {
        // Handle rate limiting
    }

    private fun showGenericError(e: ApiException) {
        // Handle other errors
    }
}
```

## Go

### Installation

```bash
go mod init your-project
go get github.com/DNYoussef/AIVillage/clients/go
```

### Basic Usage

```go
package main

import (
    "context"
    "fmt"
    "log"

    aivillage "github.com/DNYoussef/AIVillage/clients/go"
)

func main() {
    // Initialize client
    config := aivillage.NewConfiguration()
    config.Host = "api.aivillage.io"
    config.Scheme = "https"

    client := aivillage.NewAPIClient(config)

    // Set authentication
    auth := context.WithValue(
        context.Background(),
        aivillage.ContextAccessToken,
        "your-api-key",
    )

    // Chat example
    chatExample(client, auth)

    // RAG example
    ragExample(client, auth)
}

func chatExample(client *aivillage.APIClient, ctx context.Context) {
    request := aivillage.ChatRequest{
        Message:         "What are best practices for Go microservices?",
        AgentPreference: aivillage.PtrString("sage"),
        Mode:            aivillage.PtrString("comprehensive"),
        UserContext: &aivillage.ChatRequestUserContext{
            DeviceType:  aivillage.PtrString("desktop"),
            NetworkType: aivillage.PtrString("wifi"),
        },
    }

    response, httpResponse, err := client.ChatApi.Chat(ctx).ChatRequest(request).Execute()
    if err != nil {
        log.Printf("Chat API error: %v", err)
        return
    }
    defer httpResponse.Body.Close()

    fmt.Printf("Agent: %s\n", response.GetAgentUsed())
    fmt.Printf("Response: %s\n", response.GetResponse())
    fmt.Printf("Processing time: %dms\n", response.GetProcessingTimeMs())
}

func ragExample(client *aivillage.APIClient, ctx context.Context) {
    request := aivillage.QueryRequest{
        Query:          "Explain microservice communication patterns",
        Mode:           aivillage.PtrString("analytical"),
        IncludeSources: aivillage.PtrBool(true),
        MaxResults:     aivillage.PtrInt32(5),
    }

    response, httpResponse, err := client.RAGApi.ProcessQuery(ctx).QueryRequest(request).Execute()
    if err != nil {
        log.Printf("RAG API error: %v", err)
        return
    }
    defer httpResponse.Body.Close()

    fmt.Printf("Query ID: %s\n", response.GetQueryId())
    fmt.Printf("Response: %s\n", response.GetResponse())

    for _, source := range response.GetSources() {
        fmt.Printf("Source: %s (confidence: %.2f)\n",
            source.GetTitle(),
            source.GetConfidence())
    }
}
```

### Error Handling and Retries

```go
package main

import (
    "context"
    "fmt"
    "net/http"
    "time"

    aivillage "github.com/DNYoussef/AIVillage/clients/go"
)

func chatWithRetry(client *aivillage.APIClient, ctx context.Context, request aivillage.ChatRequest, maxRetries int) (*aivillage.ChatResponse, error) {
    for attempt := 0; attempt < maxRetries; attempt++ {
        response, httpResponse, err := client.ChatApi.Chat(ctx).ChatRequest(request).Execute()

        if err == nil {
            httpResponse.Body.Close()
            return response, nil
        }

        if httpResponse != nil {
            defer httpResponse.Body.Close()

            switch httpResponse.StatusCode {
            case http.StatusTooManyRequests:
                // Rate limited - check Retry-After header
                retryAfter := httpResponse.Header.Get("Retry-After")
                if retryAfter != "" {
                    if duration, parseErr := time.ParseDuration(retryAfter + "s"); parseErr == nil {
                        time.Sleep(duration)
                        continue
                    }
                }
                time.Sleep(time.Minute) // Default wait

            case http.StatusInternalServerError, http.StatusBadGateway, http.StatusServiceUnavailable:
                // Server error - exponential backoff
                backoff := time.Duration(1<<attempt) * time.Second
                if backoff > 32*time.Second {
                    backoff = 32 * time.Second
                }
                time.Sleep(backoff)

            default:
                // Client error - don't retry
                return nil, fmt.Errorf("API error: %v", err)
            }
        }

        if attempt == maxRetries-1 {
            return nil, fmt.Errorf("max retries exceeded: %v", err)
        }
    }

    return nil, fmt.Errorf("unexpected error")
}
```

## Rust

### Installation

**Cargo.toml**
```toml
[dependencies]
aivillage-client = "1.0.0"
tokio = { version = "1.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
```

### Basic Usage

```rust
use aivillage_client::{Client, Configuration};
use aivillage_client::apis::chat_api;
use aivillage_client::models::{ChatRequest, ChatRequestUserContext};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure client
    let config = Configuration {
        base_path: "https://api.aivillage.io/v1".to_string(),
        bearer_access_token: Some("your-api-key".to_string()),
        ..Default::default()
    };

    let client = Client::new(config);

    // Chat example
    chat_example(&client).await?;

    // RAG example
    rag_example(&client).await?;

    Ok(())
}

async fn chat_example(client: &Client) -> Result<(), Box<dyn std::error::Error>> {
    let request = ChatRequest {
        message: "How can I optimize Rust performance for machine learning?".to_string(),
        agent_preference: Some("magi".to_string()),
        mode: Some("comprehensive".to_string()),
        conversation_id: None,
        user_context: Some(ChatRequestUserContext {
            device_type: Some("desktop".to_string()),
            battery_level: None,
            network_type: Some("wifi".to_string()),
        }),
    };

    match chat_api::chat(&client.configuration, request).await {
        Ok(response) => {
            println!("Agent: {}", response.agent_used);
            println!("Response: {}", response.response);
            println!("Processing time: {}ms", response.processing_time_ms);
        }
        Err(e) => {
            eprintln!("Chat error: {}", e);
        }
    }

    Ok(())
}

async fn rag_example(client: &Client) -> Result<(), Box<dyn std::error::Error>> {
    use aivillage_client::apis::rag_api;
    use aivillage_client::models::QueryRequest;

    let request = QueryRequest {
        query: "Explain zero-copy optimizations in Rust".to_string(),
        mode: Some("analytical".to_string()),
        include_sources: Some(true),
        max_results: Some(5),
        user_id: None,
    };

    match rag_api::process_query(&client.configuration, request).await {
        Ok(response) => {
            println!("Query ID: {}", response.query_id);
            println!("Response: {}", response.response);

            for source in response.sources {
                println!("Source: {} (confidence: {:.2})",
                    source.title, source.confidence);
            }
        }
        Err(e) => {
            eprintln!("RAG error: {}", e);
        }
    }

    Ok(())
}
```

### Advanced Error Handling

```rust
use aivillage_client::apis::Error as ApiError;
use std::time::Duration;
use tokio::time::sleep;

pub async fn chat_with_retry(
    client: &Client,
    request: ChatRequest,
    max_retries: usize,
) -> Result<ChatResponse, Box<dyn std::error::Error>> {
    for attempt in 0..max_retries {
        match chat_api::chat(&client.configuration, request.clone()).await {
            Ok(response) => return Ok(response),
            Err(ApiError::ResponseError(ref response_content)) => {
                match response_content.status {
                    429 => {
                        // Rate limited
                        let retry_after = response_content
                            .headers
                            .get("retry-after")
                            .and_then(|v| v.parse::<u64>().ok())
                            .unwrap_or(60);

                        sleep(Duration::from_secs(retry_after)).await;
                    }
                    500..=599 => {
                        // Server error - exponential backoff
                        let backoff_secs = std::cmp::min(1 << attempt, 32);
                        sleep(Duration::from_secs(backoff_secs)).await;
                    }
                    _ => {
                        // Client error - don't retry
                        return Err(Box::new(ApiError::ResponseError(response_content.clone())));
                    }
                }
            }
            Err(e) => {
                if attempt == max_retries - 1 {
                    return Err(Box::new(e));
                }

                // Network error - exponential backoff
                let backoff_secs = std::cmp::min(1 << attempt, 32);
                sleep(Duration::from_secs(backoff_secs)).await;
            }
        }
    }

    Err("Max retries exceeded".into())
}
```

## Common Patterns

### Authentication

All SDKs support two authentication methods:

1. **Bearer Token** (recommended):
   ```
   Authorization: Bearer your-api-key
   ```

2. **API Key Header**:
   ```
   x-api-key: your-api-key
   ```

### Error Handling

Common HTTP status codes and their meanings:

| Status | Meaning | Retry? | Action |
|--------|---------|--------|--------|
| 400 | Bad Request | No | Fix request format |
| 401 | Unauthorized | No | Check API key |
| 403 | Forbidden | No | Check permissions |
| 404 | Not Found | No | Check endpoint URL |
| 429 | Rate Limited | Yes | Wait for `Retry-After` seconds |
| 500 | Server Error | Yes | Exponential backoff |
| 503 | Service Unavailable | Yes | Exponential backoff |

### Rate Limiting

All endpoints are rate limited:
- **Default**: 100 requests per 60 seconds
- **Authenticated**: 200 requests per 60 seconds
- **Premium**: 500 requests per 60 seconds

Rate limit headers in responses:
- `X-RateLimit-Limit`: Total requests allowed
- `X-RateLimit-Remaining`: Requests remaining
- `X-RateLimit-Reset`: Reset timestamp
- `Retry-After`: Seconds to wait (on 429 response)

### Idempotency

POST, PUT, and PATCH requests support idempotency keys:

```http
POST /v1/chat
Idempotency-Key: chat-2025-08-19-uuid-123
Content-Type: application/json

{
  "message": "Hello world"
}
```

Benefits:
- Safe to retry failed requests
- Prevents duplicate operations
- Cached responses for repeated requests

### Timeouts

Recommended timeout settings:
- **Connection**: 10 seconds
- **Read**: 30 seconds
- **Total**: 60 seconds for complex operations

### Logging

Enable debug logging to troubleshoot issues:

**TypeScript/JavaScript:**
```typescript
const client = new AIVillageClient({
  basePath: 'https://api.aivillage.io/v1',
  accessToken: 'your-api-key',
  // Enable debug logging
  debugLog: console.log
});
```

**Python:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Java:**
```java
// Enable OkHttp logging
HttpLoggingInterceptor logging = new HttpLoggingInterceptor();
logging.setLevel(HttpLoggingInterceptor.Level.BODY);
client.getHttpClient().newBuilder()
    .addInterceptor(logging)
    .build();
```

## Testing

### Mock Responses

For testing, use mock responses instead of real API calls:

**TypeScript/JavaScript:**
```typescript
// Jest example
jest.mock('aivillage-client', () => ({
  AIVillageClient: jest.fn().mockImplementation(() => ({
    chat: {
      chat: jest.fn().mockResolvedValue({
        response: 'Mocked response',
        agent_used: 'mock-agent',
        processing_time_ms: 100
      })
    }
  }))
}));
```

**Python:**
```python
# pytest example
from unittest.mock import AsyncMock, patch

@patch('aivillage_client.ChatApi')
async def test_chat(mock_chat_api):
    mock_chat_api.return_value.chat = AsyncMock(return_value=ChatResponse(
        response="Mocked response",
        agent_used="mock-agent",
        processing_time_ms=100
    ))

    # Your test code here
```

### Integration Testing

For integration tests, use the staging environment:

```bash
AIVILLAGE_API_URL=https://staging-api.aivillage.io/v1
```

## Support

### Documentation
- **API Reference**: [docs.aivillage.io/api](https://docs.aivillage.io/api)
- **OpenAPI Spec**: [docs/api/openapi.yaml](./openapi.yaml)
- **Examples**: [github.com/DNYoussef/AIVillage/tree/main/examples](https://github.com/DNYoussef/AIVillage/tree/main/examples)

### Community
- **GitHub Issues**: [github.com/DNYoussef/AIVillage/issues](https://github.com/DNYoussef/AIVillage/issues)
- **Discussions**: [github.com/DNYoussef/AIVillage/discussions](https://github.com/DNYoussef/AIVillage/discussions)

### Contact
- **General**: [support@aivillage.io](mailto:support@aivillage.io)
- **API Issues**: [api-support@aivillage.io](mailto:api-support@aivillage.io)
- **Security**: [security@aivillage.io](mailto:security@aivillage.io)

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines on:
- Reporting bugs
- Requesting features
- Contributing code
- SDK development
