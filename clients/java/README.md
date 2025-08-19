# AIVillage Java SDK

A comprehensive, production-ready Java client for the AIVillage API with full async support, type safety, and built-in reliability patterns.

## Features

- **Type Safety**: Full type definitions with comprehensive model classes
- **Async Support**: CompletableFuture-based asynchronous operations
- **Reliability**: Automatic retries with exponential backoff and circuit breaker patterns
- **Idempotency**: Safe retry of mutating operations with idempotency keys
- **Rate Limiting**: Built-in rate limit awareness with automatic backoff
- **Error Handling**: Rich exception types with detailed error context
- **Authentication**: Bearer token and API key authentication methods
- **Connection Pooling**: Optimized HTTP client with connection reuse

## Installation

### Maven

```xml
<dependency>
    <groupId>io.aivillage</groupId>
    <artifactId>aivillage-client</artifactId>
    <version>1.0.0</version>
</dependency>
```

### Gradle

```gradle
implementation 'io.aivillage:aivillage-client:1.0.0'

// For reactive support (optional)
implementation 'io.reactivex.rxjava3:rxjava:3.1.6'
```

### SBT

```scala
libraryDependencies += "io.aivillage" % "aivillage-client" % "1.0.0"
```

## Quick Start

```java
import io.aivillage.client.*;
import io.aivillage.client.api.*;
import io.aivillage.client.model.*;
import java.util.concurrent.CompletableFuture;

public class AIVillageExample {
    public static void main(String[] args) {
        // Configure client
        ApiClient defaultClient = new ApiClient();
        defaultClient.setBasePath("https://api.aivillage.io/v1");
        defaultClient.setBearerToken("your-api-key");

        // Create API instance
        ChatApi chatApi = new ChatApi(defaultClient);

        try {
            // Basic chat request
            ChatRequest request = new ChatRequest()
                .message("How can I optimize machine learning model performance on mobile devices?")
                .agentPreference(ChatRequest.AgentPreferenceEnum.MAGI)
                .mode(ChatRequest.ModeEnum.COMPREHENSIVE)
                .userContext(new ChatRequestUserContext()
                    .deviceType(ChatRequestUserContext.DeviceTypeEnum.MOBILE)
                    .batteryLevel(75)
                    .networkType(ChatRequestUserContext.NetworkTypeEnum.WIFI));

            ChatResponse response = chatApi.chat(request);

            System.out.println("Agent: " + response.getAgentUsed());
            System.out.println("Response: " + response.getResponse());
            System.out.println("Processing time: " + response.getProcessingTimeMs() + "ms");

        } catch (ApiException e) {
            System.err.println("API error: " + e.getMessage());
            System.err.println("Status: " + e.getCode());
            System.err.println("Response body: " + e.getResponseBody());
        }
    }
}
```

## Configuration

### Basic Configuration

```java
import io.aivillage.client.ApiClient;
import io.aivillage.client.Configuration;

// Method 1: Direct configuration
ApiClient client = new ApiClient();
client.setBasePath("https://api.aivillage.io/v1");
client.setBearerToken("your-api-key");
client.setConnectTimeout(30000); // 30 seconds
client.setReadTimeout(60000);    // 60 seconds

// Method 2: Using Configuration object
Configuration config = new Configuration();
config.setDefaultApiClient(client);

// Method 3: Environment-based configuration
String apiUrl = System.getenv().getOrDefault("AIVILLAGE_API_URL", "https://api.aivillage.io/v1");
String apiKey = System.getenv("AIVILLAGE_API_KEY");

ApiClient envClient = new ApiClient();
envClient.setBasePath(apiUrl);
envClient.setBearerToken(apiKey);
```

### Advanced Configuration

```java
import okhttp3.*;
import java.time.Duration;
import java.util.concurrent.TimeUnit;

public class AdvancedConfiguration {
    public static ApiClient createProductionClient(String apiKey) {
        // Custom OkHttp client with connection pooling
        OkHttpClient httpClient = new OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(60, TimeUnit.SECONDS)
            .writeTimeout(60, TimeUnit.SECONDS)
            .connectionPool(new ConnectionPool(10, 5, TimeUnit.MINUTES))
            .retryOnConnectionFailure(true)
            .addInterceptor(new HttpLoggingInterceptor().setLevel(HttpLoggingInterceptor.Level.BASIC))
            .build();

        ApiClient client = new ApiClient(httpClient);
        client.setBasePath("https://api.aivillage.io/v1");
        client.setBearerToken(apiKey);

        // Configure serialization
        client.setLenientOnJson(true);
        client.setDebugging(false); // Disable for production

        return client;
    }
}
```

## API Reference

### Chat API

Interact with AIVillage's specialized AI agents.

```java
import io.aivillage.client.api.ChatApi;
import io.aivillage.client.model.*;
import java.util.concurrent.CompletableFuture;

public class ChatExample {
    private final ChatApi chatApi;

    public ChatExample(ApiClient apiClient) {
        this.chatApi = new ChatApi(apiClient);
    }

    public ChatResponse basicChat(String message) throws ApiException {
        ChatRequest request = new ChatRequest()
            .message(message)
            .agentPreference(ChatRequest.AgentPreferenceEnum.SAGE)
            .mode(ChatRequest.ModeEnum.BALANCED);

        return chatApi.chat(request);
    }

    public ChatResponse contextualChat(String message, String conversationId) throws ApiException {
        ChatRequest request = new ChatRequest()
            .message(message)
            .conversationId(conversationId)
            .agentPreference(ChatRequest.AgentPreferenceEnum.NAVIGATOR)
            .mode(ChatRequest.ModeEnum.COMPREHENSIVE)
            .userContext(new ChatRequestUserContext()
                .deviceType(ChatRequestUserContext.DeviceTypeEnum.MOBILE)
                .batteryLevel(60)
                .networkType(ChatRequestUserContext.NetworkTypeEnum.CELLULAR)
                .dataBudgetMb(50));

        return chatApi.chat(request);
    }

    // Async version with CompletableFuture
    public CompletableFuture<ChatResponse> asyncChat(String message) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                return basicChat(message);
            } catch (ApiException e) {
                throw new RuntimeException(e);
            }
        });
    }
}
```

**Available Agents:**
- `KING`: Coordination and oversight with public thought bubbles
- `MAGI`: Research and comprehensive analysis
- `SAGE`: Deep knowledge and wisdom
- `ORACLE`: Predictions and forecasting
- `NAVIGATOR`: Routing and mobile optimization
- `ANY`: Auto-select best agent (default)

**Response Modes:**
- `FAST`: Quick responses with minimal processing
- `BALANCED`: Good balance of speed and thoroughness (default)
- `COMPREHENSIVE`: Detailed analysis with full context
- `CREATIVE`: Innovative and creative insights
- `ANALYTICAL`: Systematic analysis and reasoning

### RAG API

Advanced knowledge retrieval with Bayesian trust networks.

```java
import io.aivillage.client.api.RAGApi;
import io.aivillage.client.model.*;
import java.util.List;

public class RAGExample {
    private final RAGApi ragApi;

    public RAGExample(ApiClient apiClient) {
        this.ragApi = new RAGApi(apiClient);
    }

    public QueryResponse processKnowledgeQuery(String query) throws ApiException {
        QueryRequest request = new QueryRequest()
            .query(query)
            .mode(QueryRequest.ModeEnum.COMPREHENSIVE)
            .includeSources(true)
            .maxResults(10);

        QueryResponse response = ragApi.processQuery(request);

        System.out.println("Query ID: " + response.getQueryId());
        System.out.println("Response: " + response.getResponse());
        System.out.println("Confidence: " + response.getMetadata().getBayesianConfidence());

        // Process sources
        List<QueryResponseSourcesInner> sources = response.getSources();
        for (QueryResponseSourcesInner source : sources) {
            System.out.printf("Source: %s (confidence: %.3f)%n",
                source.getTitle(), source.getConfidence());
        }

        return response;
    }

    public List<QueryResponse> batchProcessQueries(List<String> queries) {
        return queries.parallelStream()
            .map(query -> {
                try {
                    return ragApi.processQuery(
                        new QueryRequest()
                            .query(query)
                            .mode(QueryRequest.ModeEnum.FAST)
                            .maxResults(5)
                    );
                } catch (ApiException e) {
                    throw new RuntimeException("Query failed: " + query, e);
                }
            })
            .toList();
    }
}
```

### Agents API

Manage and monitor AI agents.

```java
import io.aivillage.client.api.AgentsApi;
import io.aivillage.client.model.*;
import java.util.List;
import java.util.Map;

public class AgentsExample {
    private final AgentsApi agentsApi;

    public AgentsExample(ApiClient apiClient) {
        this.agentsApi = new AgentsApi(apiClient);
    }

    public void manageAgents() throws ApiException {
        // List available agents
        ListAgentsResponse agentsList = agentsApi.listAgents()
            .category("knowledge")
            .availableOnly(true)
            .execute();

        System.out.println("Available agents:");
        for (Agent agent : agentsList.getAgents()) {
            System.out.printf("  %s: %s (load: %d%%)%n",
                agent.getName(), agent.getStatus(), agent.getCurrentLoad());
        }

        // Assign task to specific agent
        if (!agentsList.getAgents().isEmpty()) {
            Agent firstAgent = agentsList.getAgents().get(0);

            AgentTaskRequest taskRequest = new AgentTaskRequest()
                .taskDescription("Research quantum computing applications in federated learning")
                .priority(AgentTaskRequest.PriorityEnum.HIGH)
                .timeoutSeconds(600)
                .context(Map.of(
                    "domain", "quantum_ai",
                    "depth", "comprehensive",
                    "include_implementation", true
                ));

            AgentTaskResponse taskResponse = agentsApi.assignAgentTask(firstAgent.getId(), taskRequest);

            System.out.println("Task assigned: " + taskResponse.getTaskId());
            System.out.println("Estimated completion: " + taskResponse.getEstimatedCompletionTime());
        }
    }
}
```

### P2P API

Monitor peer-to-peer mesh network status.

```java
import io.aivillage.client.api.P2PApi;
import io.aivillage.client.model.*;

public class P2PExample {
    private final P2PApi p2pApi;

    public P2PExample(ApiClient apiClient) {
        this.p2pApi = new P2PApi(apiClient);
    }

    public void monitorNetwork() throws ApiException {
        // Get overall network status
        P2PStatusResponse status = p2pApi.getP2PStatus();
        System.out.println("Network status: " + status.getStatus());
        System.out.println("Connected peers: " + status.getPeerCount());
        System.out.println("Network health: " + status.getHealthScore());

        // List peers by transport type
        String[] transports = {"bitchat", "betanet"};

        for (String transport : transports) {
            ListPeersResponse peers = p2pApi.listPeers(transport);

            System.out.printf("\n%s peers (%d total):\n",
                transport.toUpperCase(), peers.getPeers().size());

            for (P2PPeer peer : peers.getPeers()) {
                System.out.printf("  %s... - %s (%dms latency)\n",
                    peer.getId().substring(0, 8),
                    peer.getStatus(),
                    peer.getLatencyMs());
            }
        }
    }
}
```

### Digital Twin API

Privacy-preserving personal AI assistant.

```java
import io.aivillage.client.api.DigitalTwinApi;
import io.aivillage.client.model.*;
import java.time.OffsetDateTime;
import java.util.List;
import java.util.Map;

public class DigitalTwinExample {
    private final DigitalTwinApi twinApi;

    public DigitalTwinExample(ApiClient apiClient) {
        this.twinApi = new DigitalTwinApi(apiClient);
    }

    public void manageDigitalTwin() throws ApiException {
        // Get current profile
        DigitalTwinProfileResponse profile = twinApi.getDigitalTwinProfile();
        System.out.println("Model size: " + profile.getModelSizeMb() + "MB");
        System.out.println("Accuracy: " + profile.getLearningStats().getAccuracyScore());
        System.out.println("Privacy level: " + profile.getPrivacySettings().getLevel());

        // Update with new interaction data
        DigitalTwinDataUpdateRequest updateRequest = new DigitalTwinDataUpdateRequest()
            .dataType("interaction")
            .dataPoints(List.of(
                Map.of(
                    "timestamp", OffsetDateTime.now().toString(),
                    "content", Map.of(
                        "interaction_type", "chat",
                        "user_satisfaction", 0.9,
                        "context", "mobile_optimization_help"
                    ),
                    "prediction_accuracy", 0.87
                )
            ));

        twinApi.updateDigitalTwinData(updateRequest);
        System.out.println("Digital twin updated with new interaction data");
    }
}
```

## Error Handling

### Exception Types

```java
import io.aivillage.client.ApiException;

public class ErrorHandlingExample {

    public void handleApiErrors(ChatApi chatApi, ChatRequest request) {
        try {
            ChatResponse response = chatApi.chat(request);
            System.out.println("Success: " + response.getResponse());

        } catch (ApiException e) {
            switch (e.getCode()) {
                case 400:
                    System.err.println("Bad Request: " + e.getResponseBody());
                    // Handle validation errors
                    break;

                case 401:
                    System.err.println("Unauthorized: Check your API key");
                    break;

                case 403:
                    System.err.println("Forbidden: Insufficient permissions");
                    break;

                case 404:
                    System.err.println("Not Found: " + e.getResponseBody());
                    break;

                case 429:
                    System.err.println("Rate Limited: " + e.getResponseHeaders().get("Retry-After"));
                    // Implement retry logic
                    handleRateLimit(e, chatApi, request);
                    break;

                case 500:
                case 502:
                case 503:
                    System.err.println("Server Error: " + e.getMessage());
                    // Implement exponential backoff retry
                    handleServerError(e, chatApi, request);
                    break;

                default:
                    System.err.println("Unexpected error: " + e.getMessage());
            }
        }
    }

    private void handleRateLimit(ApiException e, ChatApi chatApi, ChatRequest request) {
        try {
            String retryAfter = e.getResponseHeaders().get("Retry-After").get(0);
            int waitSeconds = Integer.parseInt(retryAfter);

            System.out.println("Waiting " + waitSeconds + " seconds before retry...");
            Thread.sleep(waitSeconds * 1000);

            // Retry the request
            ChatResponse response = chatApi.chat(request);
            System.out.println("Retry successful: " + response.getResponse());

        } catch (InterruptedException | ApiException retryException) {
            System.err.println("Retry failed: " + retryException.getMessage());
        }
    }

    private void handleServerError(ApiException e, ChatApi chatApi, ChatRequest request) {
        int maxRetries = 3;
        int baseDelay = 1000; // 1 second

        for (int attempt = 0; attempt < maxRetries; attempt++) {
            try {
                int delay = baseDelay * (int) Math.pow(2, attempt);
                System.out.println("Retrying in " + delay + "ms...");
                Thread.sleep(delay);

                ChatResponse response = chatApi.chat(request);
                System.out.println("Retry successful: " + response.getResponse());
                return;

            } catch (InterruptedException | ApiException retryException) {
                if (attempt == maxRetries - 1) {
                    System.err.println("All retries exhausted: " + retryException.getMessage());
                }
            }
        }
    }
}
```

### Circuit Breaker Pattern

```java
import java.time.Duration;
import java.time.Instant;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Supplier;

public class CircuitBreaker<T> {
    private enum State { CLOSED, OPEN, HALF_OPEN }

    private final int failureThreshold;
    private final Duration timeout;
    private final AtomicInteger failureCount = new AtomicInteger(0);
    private final AtomicReference<State> state = new AtomicReference<>(State.CLOSED);
    private volatile Instant lastFailureTime;

    public CircuitBreaker(int failureThreshold, Duration timeout) {
        this.failureThreshold = failureThreshold;
        this.timeout = timeout;
    }

    public T execute(Supplier<T> operation) throws Exception {
        if (state.get() == State.OPEN) {
            if (Instant.now().isAfter(lastFailureTime.plus(timeout))) {
                state.set(State.HALF_OPEN);
            } else {
                throw new Exception("Circuit breaker is OPEN");
            }
        }

        try {
            T result = operation.get();
            onSuccess();
            return result;
        } catch (Exception e) {
            onFailure();
            throw e;
        }
    }

    private void onSuccess() {
        failureCount.set(0);
        state.set(State.CLOSED);
    }

    private void onFailure() {
        failureCount.incrementAndGet();
        lastFailureTime = Instant.now();

        if (failureCount.get() >= failureThreshold) {
            state.set(State.OPEN);
        }
    }
}

// Usage example
public class ResilientChatService {
    private final ChatApi chatApi;
    private final CircuitBreaker<ChatResponse> circuitBreaker;

    public ResilientChatService(ChatApi chatApi) {
        this.chatApi = chatApi;
        this.circuitBreaker = new CircuitBreaker<>(3, Duration.ofMinutes(1));
    }

    public ChatResponse sendMessage(String message) throws Exception {
        return circuitBreaker.execute(() -> {
            try {
                return chatApi.chat(new ChatRequest().message(message));
            } catch (ApiException e) {
                throw new RuntimeException(e);
            }
        });
    }
}
```

## Advanced Usage

### Async Operations with CompletableFuture

```java
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.List;
import java.util.stream.Collectors;

public class AsyncAIVillageClient {
    private final ChatApi chatApi;
    private final RAGApi ragApi;
    private final ExecutorService executor;

    public AsyncAIVillageClient(ApiClient apiClient) {
        this.chatApi = new ChatApi(apiClient);
        this.ragApi = new RAGApi(apiClient);
        this.executor = Executors.newFixedThreadPool(10);
    }

    public CompletableFuture<ChatResponse> chatAsync(String message) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                return chatApi.chat(new ChatRequest().message(message));
            } catch (ApiException e) {
                throw new RuntimeException(e);
            }
        }, executor);
    }

    public CompletableFuture<QueryResponse> queryAsync(String query) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                return ragApi.processQuery(
                    new QueryRequest().query(query).mode(QueryRequest.ModeEnum.FAST)
                );
            } catch (ApiException e) {
                throw new RuntimeException(e);
            }
        }, executor);
    }

    // Complex async workflow
    public CompletableFuture<String> processComplexRequest(String userMessage) {
        return queryAsync("Background information about: " + userMessage)
            .thenCompose(ragResponse -> {
                String context = "Based on: " + ragResponse.getResponse() + "\n\nUser question: " + userMessage;
                return chatAsync(context);
            })
            .thenApply(chatResponse -> {
                return "Final answer: " + chatResponse.getResponse();
            })
            .exceptionally(throwable -> {
                return "Error processing request: " + throwable.getMessage();
            });
    }

    // Batch operations
    public CompletableFuture<List<ChatResponse>> batchChat(List<String> messages) {
        List<CompletableFuture<ChatResponse>> futures = messages.stream()
            .map(this::chatAsync)
            .collect(Collectors.toList());

        return CompletableFuture.allOf(futures.toArray(new CompletableFuture[0]))
            .thenApply(v -> futures.stream()
                .map(CompletableFuture::join)
                .collect(Collectors.toList()));
    }

    public void shutdown() {
        executor.shutdown();
    }
}
```

### Idempotency Keys

```java
import java.util.UUID;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

public class IdempotencyManager {

    public static String generateIdempotencyKey(String operation, String context) {
        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss"));
        String uniqueId = UUID.randomUUID().toString().substring(0, 8);

        if (context != null && !context.isEmpty()) {
            return String.format("%s-%s-%s-%s", operation, context, timestamp, uniqueId);
        }
        return String.format("%s-%s-%s", operation, timestamp, uniqueId);
    }

    public static ChatResponse idempotentChat(ChatApi chatApi, String message, String agentPreference)
            throws ApiException {
        String idempotencyKey = generateIdempotencyKey("chat", agentPreference);

        ChatRequest request = new ChatRequest()
            .message(message)
            .agentPreference(ChatRequest.AgentPreferenceEnum.fromValue(agentPreference.toUpperCase()));

        // Add idempotency key to headers
        ApiClient client = chatApi.getApiClient();
        client.addDefaultHeader("Idempotency-Key", idempotencyKey);

        return chatApi.chat(request);
    }
}
```

### Custom Interceptors

```java
import okhttp3.*;
import java.io.IOException;

// Request logging interceptor
public class LoggingInterceptor implements Interceptor {
    @Override
    public Response intercept(Chain chain) throws IOException {
        Request request = chain.request();

        long startTime = System.currentTimeMillis();
        System.out.printf("→ %s %s%n", request.method(), request.url());

        Response response = chain.proceed(request);

        long endTime = System.currentTimeMillis();
        System.out.printf("← %d %s (%dms)%n",
            response.code(), request.url(), endTime - startTime);

        return response;
    }
}

// Retry interceptor
public class RetryInterceptor implements Interceptor {
    private final int maxRetries;

    public RetryInterceptor(int maxRetries) {
        this.maxRetries = maxRetries;
    }

    @Override
    public Response intercept(Chain chain) throws IOException {
        Request request = chain.request();
        Response response = null;
        IOException exception = null;

        for (int i = 0; i <= maxRetries; i++) {
            try {
                response = chain.proceed(request);

                // Retry on server errors
                if (response.code() >= 500 && i < maxRetries) {
                    response.close();
                    Thread.sleep(1000 * (long) Math.pow(2, i)); // Exponential backoff
                    continue;
                }

                return response;
            } catch (IOException e) {
                exception = e;
                if (i == maxRetries) {
                    break;
                }

                try {
                    Thread.sleep(1000 * (long) Math.pow(2, i));
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    break;
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }

        if (exception != null) {
            throw exception;
        }

        return response;
    }
}

// Usage with custom client
public static ApiClient createCustomClient(String apiKey) {
    OkHttpClient httpClient = new OkHttpClient.Builder()
        .addInterceptor(new LoggingInterceptor())
        .addInterceptor(new RetryInterceptor(3))
        .build();

    ApiClient client = new ApiClient(httpClient);
    client.setBasePath("https://api.aivillage.io/v1");
    client.setBearerToken(apiKey);

    return client;
}
```

## Testing

### Unit Testing with JUnit 5

```java
import org.junit.jupiter.api.*;
import org.mockito.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import io.aivillage.client.api.ChatApi;
import io.aivillage.client.model.*;

class ChatApiTest {

    @Mock
    private ApiClient mockApiClient;

    @Mock
    private ChatApi chatApi;

    private ChatResponse expectedResponse;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);

        expectedResponse = new ChatResponse()
            .response("Mocked response for testing")
            .agentUsed("test-agent")
            .processingTimeMs(100);
    }

    @Test
    @DisplayName("Should send chat message successfully")
    void testChatMessageSuccess() throws ApiException {
        // Given
        ChatRequest request = new ChatRequest()
            .message("Test message")
            .mode(ChatRequest.ModeEnum.FAST);

        when(chatApi.chat(request)).thenReturn(expectedResponse);

        // When
        ChatResponse response = chatApi.chat(request);

        // Then
        assertNotNull(response);
        assertEquals("Mocked response for testing", response.getResponse());
        assertEquals("test-agent", response.getAgentUsed());
        assertEquals(100, response.getProcessingTimeMs());

        verify(chatApi).chat(request);
    }

    @Test
    @DisplayName("Should handle API exception gracefully")
    void testApiExceptionHandling() throws ApiException {
        // Given
        ChatRequest request = new ChatRequest().message("Test");
        ApiException expectedException = new ApiException(429, "Rate limited");

        when(chatApi.chat(request)).thenThrow(expectedException);

        // When & Then
        ApiException thrown = assertThrows(ApiException.class, () -> {
            chatApi.chat(request);
        });

        assertEquals(429, thrown.getCode());
        assertEquals("Rate limited", thrown.getMessage());
    }

    @Test
    @DisplayName("Should validate request parameters")
    void testRequestValidation() {
        // Test empty message
        assertThrows(IllegalArgumentException.class, () -> {
            new ChatRequest().message("").mode(ChatRequest.ModeEnum.FAST);
        });

        // Test null agent preference (should use default)
        ChatRequest request = new ChatRequest()
            .message("Test")
            .agentPreference(null);

        // Should default to ANY
        assertNull(request.getAgentPreference());
    }
}
```

### Integration Testing

```java
import org.junit.jupiter.api.*;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import io.aivillage.client.*;
import io.aivillage.client.api.*;
import io.aivillage.client.model.*;

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class AIVillageIntegrationTest {

    private ApiClient apiClient;
    private ChatApi chatApi;
    private RAGApi ragApi;

    @BeforeAll
    void setUp() {
        String apiKey = System.getenv("AIVILLAGE_TEST_API_KEY");
        String apiUrl = System.getenv("AIVILLAGE_TEST_API_URL");

        if (apiKey == null) {
            fail("AIVILLAGE_TEST_API_KEY environment variable is required");
        }

        apiClient = new ApiClient();
        apiClient.setBasePath(apiUrl != null ? apiUrl : "https://staging-api.aivillage.io/v1");
        apiClient.setBearerToken(apiKey);

        chatApi = new ChatApi(apiClient);
        ragApi = new RAGApi(apiClient);
    }

    @Test
    @DisplayName("Integration test: Chat with agents")
    @EnabledIfEnvironmentVariable(named = "AIVILLAGE_TEST_API_KEY", matches = ".*")
    void testChatIntegration() throws ApiException {
        ChatRequest request = new ChatRequest()
            .message("Hello, this is an integration test message")
            .mode(ChatRequest.ModeEnum.FAST);

        ChatResponse response = chatApi.chat(request);

        assertNotNull(response);
        assertNotNull(response.getResponse());
        assertNotNull(response.getAgentUsed());
        assertTrue(response.getProcessingTimeMs() > 0);

        System.out.println("Integration test response: " + response.getResponse());
    }

    @Test
    @DisplayName("Integration test: RAG query processing")
    @EnabledIfEnvironmentVariable(named = "AIVILLAGE_TEST_API_KEY", matches = ".*")
    void testRAGIntegration() throws ApiException {
        QueryRequest request = new QueryRequest()
            .query("What is machine learning?")
            .mode(QueryRequest.ModeEnum.FAST)
            .maxResults(3);

        QueryResponse response = ragApi.processQuery(request);

        assertNotNull(response);
        assertNotNull(response.getResponse());
        assertNotNull(response.getQueryId());
        assertTrue(response.getMetadata().getProcessingTimeMs() > 0);

        System.out.println("RAG integration test response: " + response.getResponse());
    }

    @Test
    @DisplayName("Integration test: Rate limiting behavior")
    @EnabledIfEnvironmentVariable(named = "AIVILLAGE_TEST_API_KEY", matches = ".*")
    void testRateLimitingBehavior() {
        // Send multiple requests quickly to test rate limiting
        ChatRequest request = new ChatRequest()
            .message("Rate limit test")
            .mode(ChatRequest.ModeEnum.FAST);

        int successCount = 0;
        int rateLimitCount = 0;

        for (int i = 0; i < 10; i++) {
            try {
                ChatResponse response = chatApi.chat(request);
                successCount++;
            } catch (ApiException e) {
                if (e.getCode() == 429) {
                    rateLimitCount++;
                } else {
                    fail("Unexpected API exception: " + e.getMessage());
                }
            }
        }

        assertTrue(successCount > 0, "At least some requests should succeed");
        System.out.printf("Rate limiting test: %d successful, %d rate limited%n",
            successCount, rateLimitCount);
    }
}
```

### Performance Testing

```java
import org.junit.jupiter.api.Test;
import java.time.Duration;
import java.time.Instant;
import java.util.concurrent.*;
import java.util.List;
import java.util.ArrayList;

public class PerformanceTest {

    @Test
    @DisplayName("Performance test: Concurrent chat requests")
    void testConcurrentChatPerformance() throws Exception {
        String apiKey = System.getenv("AIVILLAGE_TEST_API_KEY");
        if (apiKey == null) {
            System.out.println("Skipping performance test - no API key");
            return;
        }

        ApiClient client = new ApiClient();
        client.setBasePath("https://staging-api.aivillage.io/v1");
        client.setBearerToken(apiKey);

        ChatApi chatApi = new ChatApi(client);
        ExecutorService executor = Executors.newFixedThreadPool(10);

        int numRequests = 20;
        List<CompletableFuture<Long>> futures = new ArrayList<>();

        Instant startTime = Instant.now();

        for (int i = 0; i < numRequests; i++) {
            final int requestId = i;
            CompletableFuture<Long> future = CompletableFuture.supplyAsync(() -> {
                try {
                    Instant requestStart = Instant.now();

                    ChatRequest request = new ChatRequest()
                        .message("Performance test message " + requestId)
                        .mode(ChatRequest.ModeEnum.FAST);

                    ChatResponse response = chatApi.chat(request);

                    long duration = Duration.between(requestStart, Instant.now()).toMillis();
                    System.out.printf("Request %d completed in %dms%n", requestId, duration);

                    return duration;
                } catch (ApiException e) {
                    System.err.printf("Request %d failed: %s%n", requestId, e.getMessage());
                    return -1L;
                }
            }, executor);

            futures.add(future);
        }

        // Wait for all requests to complete
        CompletableFuture<Void> allOf = CompletableFuture.allOf(
            futures.toArray(new CompletableFuture[0])
        );

        allOf.get(2, TimeUnit.MINUTES); // 2 minute timeout

        Duration totalTime = Duration.between(startTime, Instant.now());

        // Calculate statistics
        List<Long> durations = futures.stream()
            .map(CompletableFuture::join)
            .filter(duration -> duration > 0)
            .toList();

        double avgDuration = durations.stream().mapToLong(Long::longValue).average().orElse(0);
        long minDuration = durations.stream().mapToLong(Long::longValue).min().orElse(0);
        long maxDuration = durations.stream().mapToLong(Long::longValue).max().orElse(0);

        System.out.printf("\nPerformance Results:\n");
        System.out.printf("Total requests: %d\n", numRequests);
        System.out.printf("Successful: %d\n", durations.size());
        System.out.printf("Total time: %dms\n", totalTime.toMillis());
        System.out.printf("Requests/second: %.2f\n", numRequests / (totalTime.toMillis() / 1000.0));
        System.out.printf("Average duration: %.1fms\n", avgDuration);
        System.out.printf("Min duration: %dms\n", minDuration);
        System.out.printf("Max duration: %dms\n", maxDuration);

        executor.shutdown();
    }
}
```

## Deployment

### Production Configuration

```java
import io.aivillage.client.ApiClient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ProductionClientFactory {
    private static final Logger logger = LoggerFactory.getLogger(ProductionClientFactory.class);

    public static ApiClient createProductionClient() {
        String apiUrl = System.getenv("AIVILLAGE_API_URL");
        String apiKey = System.getenv("AIVILLAGE_API_KEY");

        if (apiKey == null) {
            throw new IllegalArgumentException("AIVILLAGE_API_KEY environment variable is required");
        }

        ApiClient client = new ApiClient();
        client.setBasePath(apiUrl != null ? apiUrl : "https://api.aivillage.io/v1");
        client.setBearerToken(apiKey);

        // Production timeouts
        client.setConnectTimeout(30000); // 30 seconds
        client.setReadTimeout(60000);    // 60 seconds
        client.setWriteTimeout(60000);   // 60 seconds

        // Disable debugging in production
        client.setDebugging(false);

        logger.info("AIVillage client configured for production: {}", client.getBasePath());

        return client;
    }
}
```

### Spring Boot Integration

```java
// Configuration class
@Configuration
@EnableConfigurationProperties(AIVillageProperties.class)
public class AIVillageConfig {

    @Bean
    @ConditionalOnMissingBean
    public ApiClient aivillageApiClient(AIVillageProperties properties) {
        ApiClient client = new ApiClient();
        client.setBasePath(properties.getApiUrl());
        client.setBearerToken(properties.getApiKey());
        client.setConnectTimeout((int) properties.getConnectTimeout().toMillis());
        client.setReadTimeout((int) properties.getReadTimeout().toMillis());

        return client;
    }

    @Bean
    public ChatApi chatApi(ApiClient apiClient) {
        return new ChatApi(apiClient);
    }

    @Bean
    public RAGApi ragApi(ApiClient apiClient) {
        return new RAGApi(apiClient);
    }
}

// Properties class
@ConfigurationProperties(prefix = "aivillage")
@Data
public class AIVillageProperties {
    private String apiUrl = "https://api.aivillage.io/v1";
    private String apiKey;
    private Duration connectTimeout = Duration.ofSeconds(30);
    private Duration readTimeout = Duration.ofMinutes(1);
}

// Service class
@Service
public class AIVillageService {
    private final ChatApi chatApi;
    private final RAGApi ragApi;

    public AIVillageService(ChatApi chatApi, RAGApi ragApi) {
        this.chatApi = chatApi;
        this.ragApi = ragApi;
    }

    public CompletableFuture<String> processUserQuery(String query) {
        return CompletableFuture
            .supplyAsync(() -> {
                try {
                    ChatRequest request = new ChatRequest()
                        .message(query)
                        .mode(ChatRequest.ModeEnum.BALANCED);
                    return chatApi.chat(request);
                } catch (ApiException e) {
                    throw new RuntimeException("Chat API error", e);
                }
            })
            .thenApply(ChatResponse::getResponse);
    }
}
```

### Docker Configuration

```dockerfile
FROM openjdk:21-jre-slim

# Install curl for health checks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy application JAR
COPY target/aivillage-app-*.jar app.jar

# Environment variables
ENV AIVILLAGE_API_URL=https://api.aivillage.io/v1
ENV JAVA_OPTS="-Xmx512m -Xms256m"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/actuator/health || exit 1

# Run application
ENTRYPOINT ["sh", "-c", "java $JAVA_OPTS -jar /app.jar"]

EXPOSE 8080
```

## Troubleshooting

### Common Issues

**SSL/TLS Certificate Issues:**
```java
// Disable SSL verification for development (NOT recommended for production)
ApiClient client = new ApiClient();
client.setVerifyingSsl(false);
client.setBasePath("https://api.aivillage.io/v1");
```

**Connection Pool Exhausted:**
```java
// Increase connection pool size
OkHttpClient httpClient = new OkHttpClient.Builder()
    .connectionPool(new ConnectionPool(20, 5, TimeUnit.MINUTES))
    .build();

ApiClient client = new ApiClient(httpClient);
```

**Memory Issues with Large Responses:**
```java
// Configure streaming for large responses
ApiClient client = new ApiClient();
client.setLenientOnJson(true);
// Set reasonable timeout values
client.setReadTimeout(120000); // 2 minutes
```

### Debug Logging

```java
import okhttp3.logging.HttpLoggingInterceptor;

// Enable HTTP logging
HttpLoggingInterceptor logging = new HttpLoggingInterceptor();
logging.setLevel(HttpLoggingInterceptor.Level.BODY);

OkHttpClient httpClient = new OkHttpClient.Builder()
    .addInterceptor(logging)
    .build();

ApiClient client = new ApiClient(httpClient);
client.setDebugging(true);
```

## Support

- **Documentation**: [docs.aivillage.io](https://docs.aivillage.io)
- **API Reference**: [docs.aivillage.io/api](https://docs.aivillage.io/api)
- **GitHub Issues**: [github.com/DNYoussef/AIVillage/issues](https://github.com/DNYoussef/AIVillage/issues)
- **Maven Central**: [search.maven.org/artifact/io.aivillage/aivillage-client](https://search.maven.org/artifact/io.aivillage/aivillage-client)

## License

MIT License - see [LICENSE](../../LICENSE) for details.
