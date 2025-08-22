# AIVillage Go SDK

A comprehensive, production-ready Go client for the AIVillage API with full context support, structured error handling, and built-in reliability patterns.

## Features

- **Type Safety**: Full Go struct definitions with comprehensive JSON tags
- **Context Support**: Native context.Context integration for cancellation and timeouts
- **Reliability**: Automatic retries with exponential backoff and circuit breaker patterns
- **Idempotency**: Safe retry of mutating operations with idempotency keys
- **Rate Limiting**: Built-in rate limit awareness with automatic backoff
- **Error Handling**: Rich error types with detailed error context
- **Authentication**: Bearer token and API key authentication methods
- **HTTP/2 Support**: Efficient HTTP/2 client with connection reuse

## Installation

```bash
# Initialize Go module (if not already done)
go mod init your-project

# Install AIVillage Go SDK
go get github.com/DNYoussef/AIVillage/clients/go
```

## Quick Start

```go
package main

import (
    "context"
    "fmt"
    "log"

    aivillage "github.com/DNYoussef/AIVillage/clients/go"
)

func main() {
    // Configure client
    config := aivillage.NewConfiguration()
    config.Host = "api.aivillage.io"
    config.Scheme = "https"

    client := aivillage.NewAPIClient(config)

    // Set authentication context
    auth := context.WithValue(
        context.Background(),
        aivillage.ContextAccessToken,
        "your-api-key",
    )

    // Create chat request
    request := aivillage.ChatRequest{
        Message:         "How can I optimize Go applications for mobile deployment?",
        AgentPreference: aivillage.PtrString("magi"),
        Mode:            aivillage.PtrString("comprehensive"),
        UserContext: &aivillage.ChatRequestUserContext{
            DeviceType:  aivillage.PtrString("mobile"),
            NetworkType: aivillage.PtrString("wifi"),
        },
    }

    // Send request
    response, httpResponse, err := client.ChatApi.Chat(auth).ChatRequest(request).Execute()
    if err != nil {
        log.Fatalf("Chat API error: %v", err)
    }
    defer httpResponse.Body.Close()

    fmt.Printf("Agent: %s\n", response.GetAgentUsed())
    fmt.Printf("Response: %s\n", response.GetResponse())
    fmt.Printf("Processing time: %dms\n", response.GetProcessingTimeMs())
}
```

## Configuration

### Basic Configuration

```go
package main

import (
    "context"
    "os"
    "time"

    aivillage "github.com/DNYoussef/AIVillage/clients/go"
)

// Basic configuration
func createBasicClient() *aivillage.APIClient {
    config := aivillage.NewConfiguration()
    config.Host = "api.aivillage.io"
    config.Scheme = "https"

    return aivillage.NewAPIClient(config)
}

// Environment-based configuration
func createClientFromEnv() *aivillage.APIClient {
    config := aivillage.NewConfiguration()

    apiURL := os.Getenv("AIVILLAGE_API_URL")
    if apiURL != "" {
        config.Host = apiURL
    } else {
        config.Host = "api.aivillage.io"
    }

    config.Scheme = "https"

    return aivillage.NewAPIClient(config)
}

// Create authenticated context
func createAuthContext(apiKey string) context.Context {
    return context.WithValue(
        context.Background(),
        aivillage.ContextAccessToken,
        apiKey,
    )
}

// Context with timeout
func createTimeoutContext(apiKey string, timeout time.Duration) (context.Context, context.CancelFunc) {
    auth := createAuthContext(apiKey)
    return context.WithTimeout(auth, timeout)
}
```

### Advanced Configuration

```go
import (
    "crypto/tls"
    "net/http"
    "time"
)

// Custom HTTP client with advanced configuration
func createAdvancedClient(apiKey string) *aivillage.APIClient {
    // Custom transport for production use
    transport := &http.Transport{
        MaxIdleConns:        100,
        MaxIdleConnsPerHost: 10,
        IdleConnTimeout:     90 * time.Second,
        DisableCompression:  false,
        TLSClientConfig: &tls.Config{
            MinVersion: tls.VersionTLS12,
        },
    }

    // Custom HTTP client
    httpClient := &http.Client{
        Timeout:   60 * time.Second,
        Transport: transport,
    }

    config := aivillage.NewConfiguration()
    config.Host = "api.aivillage.io"
    config.Scheme = "https"
    config.HTTPClient = httpClient

    // Add default headers
    config.DefaultHeader = map[string]string{
        "User-Agent": "AIVillage-Go-SDK/1.0.0",
    }

    return aivillage.NewAPIClient(config)
}
```

## API Reference

### Chat API

Interact with AIVillage's specialized AI agents.

```go
import (
    "context"
    "fmt"
    aivillage "github.com/DNYoussef/AIVillage/clients/go"
)

type ChatService struct {
    client *aivillage.APIClient
    auth   context.Context
}

func NewChatService(client *aivillage.APIClient, apiKey string) *ChatService {
    auth := context.WithValue(context.Background(), aivillage.ContextAccessToken, apiKey)
    return &ChatService{
        client: client,
        auth:   auth,
    }
}

func (s *ChatService) BasicChat(message string) (*aivillage.ChatResponse, error) {
    request := aivillage.ChatRequest{
        Message:         message,
        AgentPreference: aivillage.PtrString("sage"),
        Mode:            aivillage.PtrString("balanced"),
    }

    response, httpResp, err := s.client.ChatApi.Chat(s.auth).ChatRequest(request).Execute()
    if httpResp != nil {
        defer httpResp.Body.Close()
    }

    return response, err
}

func (s *ChatService) ContextualChat(message, conversationID string, userContext *aivillage.ChatRequestUserContext) (*aivillage.ChatResponse, error) {
    request := aivillage.ChatRequest{
        Message:        message,
        ConversationId: aivillage.PtrString(conversationID),
        AgentPreference: aivillage.PtrString("navigator"),
        Mode:           aivillage.PtrString("comprehensive"),
        UserContext:    userContext,
    }

    response, httpResp, err := s.client.ChatApi.Chat(s.auth).ChatRequest(request).Execute()
    if httpResp != nil {
        defer httpResp.Body.Close()
    }

    return response, err
}

func (s *ChatService) MobileChatWithContext(message string, batteryLevel int32, networkType string) (*aivillage.ChatResponse, error) {
    userContext := &aivillage.ChatRequestUserContext{
        DeviceType:   aivillage.PtrString("mobile"),
        BatteryLevel: aivillage.PtrInt32(batteryLevel),
        NetworkType:  aivillage.PtrString(networkType),
    }

    request := aivillage.ChatRequest{
        Message:         message,
        AgentPreference: aivillage.PtrString("navigator"), // Mobile optimization specialist
        Mode:            aivillage.PtrString("balanced"),
        UserContext:     userContext,
    }

    response, httpResp, err := s.client.ChatApi.Chat(s.auth).ChatRequest(request).Execute()
    if httpResp != nil {
        defer httpResp.Body.Close()
    }

    return response, err
}

// Example usage
func ExampleChatUsage() {
    config := aivillage.NewConfiguration()
    config.Host = "api.aivillage.io"
    client := aivillage.NewAPIClient(config)

    chatService := NewChatService(client, "your-api-key")

    // Basic chat
    response, err := chatService.BasicChat("Explain the benefits of Go for microservices")
    if err != nil {
        fmt.Printf("Error: %v\n", err)
        return
    }

    fmt.Printf("Agent: %s\n", response.GetAgentUsed())
    fmt.Printf("Response: %s\n", response.GetResponse())

    // Mobile-optimized chat
    mobileResponse, err := chatService.MobileChatWithContext(
        "How can I deploy this on mobile devices?",
        45, // 45% battery
        "cellular",
    )
    if err != nil {
        fmt.Printf("Error: %v\n", err)
        return
    }

    fmt.Printf("Mobile response: %s\n", mobileResponse.GetResponse())
}
```

**Available Agents:**
- `king`: Coordination and oversight with public thought bubbles
- `magi`: Research and comprehensive analysis
- `sage`: Deep knowledge and wisdom
- `oracle`: Predictions and forecasting
- `navigator`: Routing and mobile optimization
- `any`: Auto-select best agent (default)

**Response Modes:**
- `fast`: Quick responses with minimal processing
- `balanced`: Good balance of speed and thoroughness (default)
- `comprehensive`: Detailed analysis with full context
- `creative`: Innovative and creative insights
- `analytical`: Systematic analysis and reasoning

### RAG API

Advanced knowledge retrieval with Bayesian trust networks.

```go
type RAGService struct {
    client *aivillage.APIClient
    auth   context.Context
}

func NewRAGService(client *aivillage.APIClient, apiKey string) *RAGService {
    auth := context.WithValue(context.Background(), aivillage.ContextAccessToken, apiKey)
    return &RAGService{
        client: client,
        auth:   auth,
    }
}

func (s *RAGService) ProcessQuery(query string, mode string, maxResults int32) (*aivillage.QueryResponse, error) {
    request := aivillage.QueryRequest{
        Query:          query,
        Mode:           aivillage.PtrString(mode),
        IncludeSources: aivillage.PtrBool(true),
        MaxResults:     aivillage.PtrInt32(maxResults),
    }

    response, httpResp, err := s.client.RAGApi.ProcessQuery(s.auth).QueryRequest(request).Execute()
    if httpResp != nil {
        defer httpResp.Body.Close()
    }

    return response, err
}

func (s *RAGService) BatchProcessQueries(queries []string) ([]*aivillage.QueryResponse, error) {
    type result struct {
        response *aivillage.QueryResponse
        err      error
        index    int
    }

    resultChan := make(chan result, len(queries))

    // Process queries concurrently
    for i, query := range queries {
        go func(idx int, q string) {
            resp, err := s.ProcessQuery(q, "fast", 5)
            resultChan <- result{response: resp, err: err, index: idx}
        }(i, query)
    }

    // Collect results in order
    results := make([]*aivillage.QueryResponse, len(queries))
    var firstError error

    for i := 0; i < len(queries); i++ {
        res := <-resultChan
        if res.err != nil && firstError == nil {
            firstError = res.err
        }
        results[res.index] = res.response
    }

    return results, firstError
}

func (s *RAGService) ProcessComprehensiveQuery(query string) (*aivillage.QueryResponse, error) {
    request := aivillage.QueryRequest{
        Query:          query,
        Mode:           aivillage.PtrString("comprehensive"),
        IncludeSources: aivillage.PtrBool(true),
        MaxResults:     aivillage.PtrInt32(15),
    }

    response, httpResp, err := s.client.RAGApi.ProcessQuery(s.auth).QueryRequest(request).Execute()
    if httpResp != nil {
        defer httpResp.Body.Close()
    }

    if err != nil {
        return nil, err
    }

    // Print detailed results
    fmt.Printf("Query ID: %s\n", response.GetQueryId())
    fmt.Printf("Response: %s\n", response.GetResponse())
    fmt.Printf("Bayesian confidence: %.3f\n", response.GetMetadata().GetBayesianConfidence())

    for _, source := range response.GetSources() {
        fmt.Printf("Source: %s (confidence: %.3f)\n",
            source.GetTitle(), source.GetConfidence())
    }

    return response, nil
}
```

### Agents API

Manage and monitor AI agents.

```go
type AgentsService struct {
    client *aivillage.APIClient
    auth   context.Context
}

func NewAgentsService(client *aivillage.APIClient, apiKey string) *AgentsService {
    auth := context.WithValue(context.Background(), aivillage.ContextAccessToken, apiKey)
    return &AgentsService{
        client: client,
        auth:   auth,
    }
}

func (s *AgentsService) ListAgents(category string, availableOnly bool) (*aivillage.ListAgentsResponse, error) {
    request := s.client.AgentsApi.ListAgents(s.auth)

    if category != "" {
        request = request.Category(category)
    }
    if availableOnly {
        request = request.AvailableOnly(availableOnly)
    }

    response, httpResp, err := request.Execute()
    if httpResp != nil {
        defer httpResp.Body.Close()
    }

    return response, err
}

func (s *AgentsService) AssignTask(agentID, taskDescription string, priority string, timeout int32) (*aivillage.AgentTaskResponse, error) {
    taskRequest := aivillage.AgentTaskRequest{
        TaskDescription: taskDescription,
        Priority:        aivillage.PtrString(priority),
        TimeoutSeconds:  aivillage.PtrInt32(timeout),
        Context: map[string]interface{}{
            "request_time": time.Now().Unix(),
            "source":       "go-sdk",
        },
    }

    response, httpResp, err := s.client.AgentsApi.AssignAgentTask(s.auth, agentID).
        AgentTaskRequest(taskRequest).Execute()
    if httpResp != nil {
        defer httpResp.Body.Close()
    }

    return response, err
}

func (s *AgentsService) MonitorAgents() {
    for {
        agents, err := s.ListAgents("", true)
        if err != nil {
            fmt.Printf("Error listing agents: %v\n", err)
            time.Sleep(30 * time.Second)
            continue
        }

        fmt.Printf("\n=== Agent Status (%s) ===\n", time.Now().Format("15:04:05"))
        for _, agent := range agents.GetAgents() {
            fmt.Printf("  %s: %s (load: %d%%)\n",
                agent.GetName(), agent.GetStatus(), agent.GetCurrentLoad())
        }

        time.Sleep(30 * time.Second)
    }
}
```

### P2P API

Monitor peer-to-peer mesh network status.

```go
type P2PService struct {
    client *aivillage.APIClient
    auth   context.Context
}

func NewP2PService(client *aivillage.APIClient, apiKey string) *P2PService {
    auth := context.WithValue(context.Background(), aivillage.ContextAccessToken, apiKey)
    return &P2PService{
        client: client,
        auth:   auth,
    }
}

func (s *P2PService) GetNetworkStatus() (*aivillage.P2PStatusResponse, error) {
    response, httpResp, err := s.client.P2PApi.GetP2PStatus(s.auth).Execute()
    if httpResp != nil {
        defer httpResp.Body.Close()
    }

    return response, err
}

func (s *P2PService) ListPeers(transportType string) (*aivillage.ListPeersResponse, error) {
    response, httpResp, err := s.client.P2PApi.ListPeers(s.auth, transportType).Execute()
    if httpResp != nil {
        defer httpResp.Body.Close()
    }

    return response, err
}

func (s *P2PService) MonitorNetwork() {
    ticker := time.NewTicker(15 * time.Second)
    defer ticker.Stop()

    for {
        select {
        case <-ticker.C:
            status, err := s.GetNetworkStatus()
            if err != nil {
                fmt.Printf("Error getting network status: %v\n", err)
                continue
            }

            fmt.Printf("\n=== P2P Network Status ===\n")
            fmt.Printf("Status: %s\n", status.GetStatus())
            fmt.Printf("Peers: %d\n", status.GetPeerCount())
            fmt.Printf("Health: %.2f\n", status.GetHealthScore())

            // List BitChat peers
            bitchatPeers, err := s.ListPeers("bitchat")
            if err == nil {
                fmt.Printf("BitChat peers: %d\n", len(bitchatPeers.GetPeers()))
            }

            // List BetaNet peers
            betanetPeers, err := s.ListPeers("betanet")
            if err == nil {
                fmt.Printf("BetaNet peers: %d\n", len(betanetPeers.GetPeers()))
            }
        }
    }
}
```

### Digital Twin API

Privacy-preserving personal AI assistant.

```go
type DigitalTwinService struct {
    client *aivillage.APIClient
    auth   context.Context
}

func NewDigitalTwinService(client *aivillage.APIClient, apiKey string) *DigitalTwinService {
    auth := context.WithValue(context.Background(), aivillage.ContextAccessToken, apiKey)
    return &DigitalTwinService{
        client: client,
        auth:   auth,
    }
}

func (s *DigitalTwinService) GetProfile() (*aivillage.DigitalTwinProfileResponse, error) {
    response, httpResp, err := s.client.DigitalTwinApi.GetDigitalTwinProfile(s.auth).Execute()
    if httpResp != nil {
        defer httpResp.Body.Close()
    }

    return response, err
}

func (s *DigitalTwinService) UpdateInteractionData(interactionType string, satisfaction, accuracy float64) error {
    updateRequest := aivillage.DigitalTwinDataUpdateRequest{
        DataType: "interaction",
        DataPoints: []map[string]interface{}{
            {
                "timestamp": time.Now().Format(time.RFC3339),
                "content": map[string]interface{}{
                    "interaction_type":  interactionType,
                    "user_satisfaction": satisfaction,
                    "context":          "go_sdk_interaction",
                },
                "prediction_accuracy": accuracy,
            },
        },
    }

    _, httpResp, err := s.client.DigitalTwinApi.UpdateDigitalTwinData(s.auth).
        DigitalTwinDataUpdateRequest(updateRequest).Execute()
    if httpResp != nil {
        defer httpResp.Body.Close()
    }

    return err
}

func (s *DigitalTwinService) PrintProfile() error {
    profile, err := s.GetProfile()
    if err != nil {
        return err
    }

    fmt.Printf("=== Digital Twin Profile ===\n")
    fmt.Printf("Model size: %.1fMB\n", profile.GetModelSizeMb())
    fmt.Printf("Accuracy: %.3f\n", profile.GetLearningStats().GetAccuracyScore())
    fmt.Printf("Privacy level: %s\n", profile.GetPrivacySettings().GetLevel())
    fmt.Printf("Total interactions: %d\n", profile.GetLearningStats().GetTotalInteractions())

    return nil
}
```

## Error Handling

### Structured Error Handling

```go
import (
    "net/http"
    "fmt"
    "strings"
    aivillage "github.com/DNYoussef/AIVillage/clients/go"
)

type APIError struct {
    StatusCode int
    Message    string
    RequestID  string
    RetryAfter int
}

func (e *APIError) Error() string {
    return fmt.Sprintf("API error %d: %s", e.StatusCode, e.Message)
}

func HandleAPIError(err error, httpResp *http.Response) *APIError {
    if httpResp == nil {
        return &APIError{
            StatusCode: 0,
            Message:    err.Error(),
        }
    }

    apiError := &APIError{
        StatusCode: httpResp.StatusCode,
        Message:    err.Error(),
        RequestID:  httpResp.Header.Get("X-Request-ID"),
    }

    // Handle rate limiting
    if httpResp.StatusCode == 429 {
        if retryAfter := httpResp.Header.Get("Retry-After"); retryAfter != "" {
            if seconds, parseErr := strconv.Atoi(retryAfter); parseErr == nil {
                apiError.RetryAfter = seconds
            }
        }
    }

    return apiError
}

func ExampleErrorHandling() {
    config := aivillage.NewConfiguration()
    client := aivillage.NewAPIClient(config)
    auth := context.WithValue(context.Background(), aivillage.ContextAccessToken, "invalid-key")

    request := aivillage.ChatRequest{Message: "Test message"}

    response, httpResp, err := client.ChatApi.Chat(auth).ChatRequest(request).Execute()
    if err != nil {
        apiError := HandleAPIError(err, httpResp)

        switch apiError.StatusCode {
        case 400:
            fmt.Printf("Bad Request: %s\n", apiError.Message)
        case 401:
            fmt.Printf("Unauthorized: Check your API key\n")
        case 403:
            fmt.Printf("Forbidden: Insufficient permissions\n")
        case 404:
            fmt.Printf("Not Found: %s\n", apiError.Message)
        case 429:
            fmt.Printf("Rate Limited. Retry after %d seconds\n", apiError.RetryAfter)
        case 500, 502, 503:
            fmt.Printf("Server Error: %s (Request ID: %s)\n", apiError.Message, apiError.RequestID)
        default:
            fmt.Printf("Unexpected error: %s\n", apiError.Message)
        }

        return
    }

    fmt.Printf("Success: %s\n", response.GetResponse())
}
```

### Retry Logic with Circuit Breaker

```go
import (
    "time"
    "errors"
    "sync"
)

type CircuitState int

const (
    Closed CircuitState = iota
    Open
    HalfOpen
)

type CircuitBreaker struct {
    mu                sync.RWMutex
    state             CircuitState
    failureCount      int
    failureThreshold  int
    timeout           time.Duration
    lastFailureTime   time.Time
    successCount      int
}

func NewCircuitBreaker(failureThreshold int, timeout time.Duration) *CircuitBreaker {
    return &CircuitBreaker{
        failureThreshold: failureThreshold,
        timeout:         timeout,
        state:           Closed,
    }
}

func (cb *CircuitBreaker) Execute(fn func() error) error {
    cb.mu.RLock()
    state := cb.state
    cb.mu.RUnlock()

    if state == Open {
        cb.mu.Lock()
        if time.Since(cb.lastFailureTime) > cb.timeout {
            cb.state = HalfOpen
            cb.successCount = 0
        } else {
            cb.mu.Unlock()
            return errors.New("circuit breaker is open")
        }
        cb.mu.Unlock()
    }

    err := fn()

    cb.mu.Lock()
    defer cb.mu.Unlock()

    if err != nil {
        cb.failureCount++
        cb.lastFailureTime = time.Now()

        if cb.failureCount >= cb.failureThreshold {
            cb.state = Open
        }

        return err
    }

    // Success
    if cb.state == HalfOpen {
        cb.successCount++
        if cb.successCount >= 3 { // Require 3 successes to close
            cb.state = Closed
            cb.failureCount = 0
        }
    } else {
        cb.failureCount = 0
    }

    return nil
}

// Resilient chat function with retry and circuit breaker
func ResilientChat(client *aivillage.APIClient, auth context.Context, request aivillage.ChatRequest) (*aivillage.ChatResponse, error) {
    cb := NewCircuitBreaker(3, 60*time.Second)

    maxRetries := 3
    baseDelay := 1 * time.Second

    for attempt := 0; attempt < maxRetries; attempt++ {
        var response *aivillage.ChatResponse
        var httpResp *http.Response
        var err error

        circuitErr := cb.Execute(func() error {
            response, httpResp, err = client.ChatApi.Chat(auth).ChatRequest(request).Execute()
            return err
        })

        if circuitErr != nil {
            if strings.Contains(circuitErr.Error(), "circuit breaker is open") {
                return nil, circuitErr
            }
        }

        if httpResp != nil {
            defer httpResp.Body.Close()
        }

        if err == nil {
            return response, nil
        }

        // Handle specific error types
        if httpResp != nil {
            switch httpResp.StatusCode {
            case 429:
                // Rate limited - wait for retry-after
                if retryAfter := httpResp.Header.Get("Retry-After"); retryAfter != "" {
                    if seconds, parseErr := strconv.Atoi(retryAfter); parseErr == nil {
                        time.Sleep(time.Duration(seconds) * time.Second)
                        continue
                    }
                }
                time.Sleep(60 * time.Second) // Default rate limit wait

            case 500, 502, 503:
                // Server errors - exponential backoff
                delay := baseDelay * time.Duration(1<<attempt)
                if delay > 32*time.Second {
                    delay = 32 * time.Second
                }
                time.Sleep(delay)

            default:
                // Client errors - don't retry
                return nil, err
            }
        } else {
            // Network errors - exponential backoff
            delay := baseDelay * time.Duration(1<<attempt)
            time.Sleep(delay)
        }

        if attempt == maxRetries-1 {
            return nil, fmt.Errorf("max retries exceeded: %w", err)
        }
    }

    return nil, errors.New("unexpected retry loop exit")
}
```

## Advanced Usage

### Concurrent Operations

```go
import (
    "sync"
    "context"
    "time"
)

type ConcurrentAIVillageClient struct {
    client      *aivillage.APIClient
    auth        context.Context
    rateLimiter chan struct{}
}

func NewConcurrentClient(client *aivillage.APIClient, apiKey string, maxConcurrent int) *ConcurrentAIVillageClient {
    auth := context.WithValue(context.Background(), aivillage.ContextAccessToken, apiKey)

    // Rate limiter to prevent too many concurrent requests
    rateLimiter := make(chan struct{}, maxConcurrent)

    return &ConcurrentAIVillageClient{
        client:      client,
        auth:        auth,
        rateLimiter: rateLimiter,
    }
}

func (c *ConcurrentAIVillageClient) BatchChat(messages []string) ([]*aivillage.ChatResponse, []error) {
    responses := make([]*aivillage.ChatResponse, len(messages))
    errors := make([]error, len(messages))

    var wg sync.WaitGroup

    for i, message := range messages {
        wg.Add(1)

        go func(index int, msg string) {
            defer wg.Done()

            // Acquire rate limiter
            c.rateLimiter <- struct{}{}
            defer func() { <-c.rateLimiter }()

            request := aivillage.ChatRequest{
                Message: msg,
                Mode:    aivillage.PtrString("fast"),
            }

            response, httpResp, err := c.client.ChatApi.Chat(c.auth).ChatRequest(request).Execute()
            if httpResp != nil {
                defer httpResp.Body.Close()
            }

            responses[index] = response
            errors[index] = err

        }(i, message)
    }

    wg.Wait()
    return responses, errors
}

func (c *ConcurrentAIVillageClient) BatchRAGQueries(queries []string) ([]*aivillage.QueryResponse, []error) {
    responses := make([]*aivillage.QueryResponse, len(queries))
    errors := make([]error, len(queries))

    var wg sync.WaitGroup

    for i, query := range queries {
        wg.Add(1)

        go func(index int, q string) {
            defer wg.Done()

            // Acquire rate limiter
            c.rateLimiter <- struct{}{}
            defer func() { <-c.rateLimiter }()

            request := aivillage.QueryRequest{
                Query:      q,
                Mode:       aivillage.PtrString("fast"),
                MaxResults: aivillage.PtrInt32(5),
            }

            response, httpResp, err := c.client.RAGApi.ProcessQuery(c.auth).QueryRequest(request).Execute()
            if httpResp != nil {
                defer httpResp.Body.Close()
            }

            responses[index] = response
            errors[index] = err

        }(i, query)
    }

    wg.Wait()
    return responses, errors
}

// Pipeline processing example
func (c *ConcurrentAIVillageClient) ProcessPipeline(userQuery string) (*PipelineResult, error) {
    type PipelineResult struct {
        RAGResponse  *aivillage.QueryResponse
        ChatResponse *aivillage.ChatResponse
        Duration     time.Duration
    }

    start := time.Now()

    // Step 1: Get background information via RAG
    ragRequest := aivillage.QueryRequest{
        Query:          "Background information: " + userQuery,
        Mode:           aivillage.PtrString("fast"),
        IncludeSources: aivillage.PtrBool(true),
        MaxResults:     aivillage.PtrInt32(5),
    }

    ragResponse, httpResp1, err := c.client.RAGApi.ProcessQuery(c.auth).QueryRequest(ragRequest).Execute()
    if httpResp1 != nil {
        defer httpResp1.Body.Close()
    }
    if err != nil {
        return nil, fmt.Errorf("RAG query failed: %w", err)
    }

    // Step 2: Use RAG results to inform chat
    context := fmt.Sprintf("Based on this background: %s\n\nUser question: %s",
        ragResponse.GetResponse(), userQuery)

    chatRequest := aivillage.ChatRequest{
        Message:         context,
        AgentPreference: aivillage.PtrString("sage"),
        Mode:            aivillage.PtrString("comprehensive"),
    }

    chatResponse, httpResp2, err := c.client.ChatApi.Chat(c.auth).ChatRequest(chatRequest).Execute()
    if httpResp2 != nil {
        defer httpResp2.Body.Close()
    }
    if err != nil {
        return nil, fmt.Errorf("chat request failed: %w", err)
    }

    return &PipelineResult{
        RAGResponse:  ragResponse,
        ChatResponse: chatResponse,
        Duration:     time.Since(start),
    }, nil
}
```

### Idempotency Keys

```go
import (
    "crypto/rand"
    "encoding/hex"
    "fmt"
    "time"
)

func GenerateIdempotencyKey(operation, context string) string {
    timestamp := time.Now().Format("20060102-150405")

    // Generate random bytes
    randomBytes := make([]byte, 4)
    rand.Read(randomBytes)
    randomHex := hex.EncodeToString(randomBytes)

    if context != "" {
        return fmt.Sprintf("%s-%s-%s-%s", operation, context, timestamp, randomHex)
    }
    return fmt.Sprintf("%s-%s-%s", operation, timestamp, randomHex)
}

func IdempotentChat(client *aivillage.APIClient, auth context.Context, message, agentPreference string) (*aivillage.ChatResponse, error) {
    idempotencyKey := GenerateIdempotencyKey("chat", agentPreference)

    request := aivillage.ChatRequest{
        Message:         message,
        AgentPreference: aivillage.PtrString(agentPreference),
        Mode:            aivillage.PtrString("balanced"),
    }

    // Add idempotency key to context or headers
    // Note: This depends on the specific SDK implementation
    ctxWithIdempotency := context.WithValue(auth, "Idempotency-Key", idempotencyKey)

    response, httpResp, err := client.ChatApi.Chat(ctxWithIdempotency).ChatRequest(request).Execute()
    if httpResp != nil {
        defer httpResp.Body.Close()
    }

    return response, err
}

type IdempotentClient struct {
    client *aivillage.APIClient
    auth   context.Context
    cache  map[string]*aivillage.ChatResponse
    mutex  sync.RWMutex
}

func NewIdempotentClient(client *aivillage.APIClient, apiKey string) *IdempotentClient {
    auth := context.WithValue(context.Background(), aivillage.ContextAccessToken, apiKey)

    return &IdempotentClient{
        client: client,
        auth:   auth,
        cache:  make(map[string]*aivillage.ChatResponse),
    }
}

func (ic *IdempotentClient) SafeChat(message, agentPreference string) (*aivillage.ChatResponse, error) {
    // Generate deterministic key based on content
    key := fmt.Sprintf("%s-%s", message, agentPreference)

    // Check cache first
    ic.mutex.RLock()
    if cached, exists := ic.cache[key]; exists {
        ic.mutex.RUnlock()
        return cached, nil
    }
    ic.mutex.RUnlock()

    // Make request with idempotency
    response, err := IdempotentChat(ic.client, ic.auth, message, agentPreference)
    if err != nil {
        return nil, err
    }

    // Cache successful response
    ic.mutex.Lock()
    ic.cache[key] = response
    ic.mutex.Unlock()

    return response, nil
}
```

## Testing

### Unit Testing

```go
package main

import (
    "context"
    "net/http"
    "net/http/httptest"
    "testing"
    "encoding/json"

    aivillage "github.com/DNYoussef/AIVillage/clients/go"
)

func TestChatAPI(t *testing.T) {
    // Create mock server
    server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        // Verify request
        if r.URL.Path != "/chat" {
            t.Errorf("Expected path /chat, got %s", r.URL.Path)
        }

        if r.Method != http.MethodPost {
            t.Errorf("Expected POST method, got %s", r.Method)
        }

        // Mock response
        response := aivillage.ChatResponse{
            Response:         "Test response from mock server",
            AgentUsed:        "test-agent",
            ProcessingTimeMs: 100,
        }

        w.Header().Set("Content-Type", "application/json")
        json.NewEncoder(w).Encode(response)
    }))
    defer server.Close()

    // Configure client to use mock server
    config := aivillage.NewConfiguration()
    config.Servers = aivillage.ServerConfigurations{
        {
            URL: server.URL,
        },
    }

    client := aivillage.NewAPIClient(config)
    auth := context.WithValue(context.Background(), aivillage.ContextAccessToken, "test-key")

    // Make request
    request := aivillage.ChatRequest{
        Message: "Test message",
        Mode:    aivillage.PtrString("fast"),
    }

    response, httpResp, err := client.ChatApi.Chat(auth).ChatRequest(request).Execute()
    if httpResp != nil {
        defer httpResp.Body.Close()
    }

    // Verify response
    if err != nil {
        t.Fatalf("Unexpected error: %v", err)
    }

    if response.GetResponse() != "Test response from mock server" {
        t.Errorf("Expected 'Test response from mock server', got '%s'", response.GetResponse())
    }

    if response.GetAgentUsed() != "test-agent" {
        t.Errorf("Expected 'test-agent', got '%s'", response.GetAgentUsed())
    }

    if response.GetProcessingTimeMs() != 100 {
        t.Errorf("Expected 100ms, got %d", response.GetProcessingTimeMs())
    }
}

func TestErrorHandling(t *testing.T) {
    // Mock server that returns errors
    server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        w.Header().Set("Retry-After", "60")
        w.WriteHeader(http.StatusTooManyRequests)
        json.NewEncoder(w).Encode(map[string]string{
            "error": "Rate limit exceeded",
        })
    }))
    defer server.Close()

    config := aivillage.NewConfiguration()
    config.Servers = aivillage.ServerConfigurations{
        {URL: server.URL},
    }

    client := aivillage.NewAPIClient(config)
    auth := context.WithValue(context.Background(), aivillage.ContextAccessToken, "test-key")

    request := aivillage.ChatRequest{Message: "Test"}

    _, httpResp, err := client.ChatApi.Chat(auth).ChatRequest(request).Execute()
    if httpResp != nil {
        defer httpResp.Body.Close()
    }

    // Should receive rate limit error
    if err == nil {
        t.Fatal("Expected error, got nil")
    }

    if httpResp.StatusCode != 429 {
        t.Errorf("Expected status 429, got %d", httpResp.StatusCode)
    }

    retryAfter := httpResp.Header.Get("Retry-After")
    if retryAfter != "60" {
        t.Errorf("Expected Retry-After: 60, got %s", retryAfter)
    }
}
```

### Integration Testing

```go
package main

import (
    "context"
    "os"
    "testing"
    "time"

    aivillage "github.com/DNYoussef/AIVillage/clients/go"
)

func TestIntegration(t *testing.T) {
    apiKey := os.Getenv("AIVILLAGE_TEST_API_KEY")
    if apiKey == "" {
        t.Skip("AIVILLAGE_TEST_API_KEY not set, skipping integration tests")
    }

    config := aivillage.NewConfiguration()
    config.Host = "staging-api.aivillage.io" // Use staging for tests

    client := aivillage.NewAPIClient(config)
    auth := context.WithValue(context.Background(), aivillage.ContextAccessToken, apiKey)

    t.Run("ChatIntegration", func(t *testing.T) {
        request := aivillage.ChatRequest{
            Message: "This is an integration test message",
            Mode:    aivillage.PtrString("fast"),
        }

        response, httpResp, err := client.ChatApi.Chat(auth).ChatRequest(request).Execute()
        if httpResp != nil {
            defer httpResp.Body.Close()
        }

        if err != nil {
            t.Fatalf("Chat integration test failed: %v", err)
        }

        if response.GetResponse() == "" {
            t.Error("Expected non-empty response")
        }

        if response.GetProcessingTimeMs() <= 0 {
            t.Error("Expected positive processing time")
        }

        t.Logf("Integration test response: %s", response.GetResponse())
    })

    t.Run("RAGIntegration", func(t *testing.T) {
        request := aivillage.QueryRequest{
            Query:      "What is artificial intelligence?",
            Mode:       aivillage.PtrString("fast"),
            MaxResults: aivillage.PtrInt32(3),
        }

        response, httpResp, err := client.RAGApi.ProcessQuery(auth).QueryRequest(request).Execute()
        if httpResp != nil {
            defer httpResp.Body.Close()
        }

        if err != nil {
            t.Fatalf("RAG integration test failed: %v", err)
        }

        if response.GetResponse() == "" {
            t.Error("Expected non-empty response")
        }

        if response.GetQueryId() == "" {
            t.Error("Expected non-empty query ID")
        }

        t.Logf("RAG integration test response: %s", response.GetResponse())
    })
}

func TestPerformance(t *testing.T) {
    apiKey := os.Getenv("AIVILLAGE_TEST_API_KEY")
    if apiKey == "" {
        t.Skip("AIVILLAGE_TEST_API_KEY not set, skipping performance tests")
    }

    config := aivillage.NewConfiguration()
    config.Host = "staging-api.aivillage.io"

    client := aivillage.NewAPIClient(config)
    concurrentClient := NewConcurrentClient(client, apiKey, 10)

    t.Run("ConcurrentPerformance", func(t *testing.T) {
        messages := make([]string, 20)
        for i := range messages {
            messages[i] = fmt.Sprintf("Performance test message %d", i)
        }

        start := time.Now()
        responses, errors := concurrentClient.BatchChat(messages)
        duration := time.Since(start)

        successCount := 0
        for i, err := range errors {
            if err == nil && responses[i] != nil {
                successCount++
            } else if err != nil {
                t.Logf("Request %d failed: %v", i, err)
            }
        }

        t.Logf("Performance test results:")
        t.Logf("  Total requests: %d", len(messages))
        t.Logf("  Successful: %d", successCount)
        t.Logf("  Duration: %v", duration)
        t.Logf("  Requests/second: %.2f", float64(len(messages))/duration.Seconds())

        if successCount == 0 {
            t.Fatal("No successful requests in performance test")
        }
    })
}
```

## Deployment

### Production Configuration

```go
package main

import (
    "crypto/tls"
    "net/http"
    "os"
    "time"
    "log"

    aivillage "github.com/DNYoussef/AIVillage/clients/go"
)

type ProductionClientConfig struct {
    APIKey      string
    APIURL      string
    Timeout     time.Duration
    MaxRetries  int
    UserAgent   string
}

func LoadProductionConfig() *ProductionClientConfig {
    config := &ProductionClientConfig{
        APIKey:     os.Getenv("AIVILLAGE_API_KEY"),
        APIURL:     os.Getenv("AIVILLAGE_API_URL"),
        Timeout:    60 * time.Second,
        MaxRetries: 3,
        UserAgent:  "AIVillage-Go-Production/1.0.0",
    }

    if config.APIURL == "" {
        config.APIURL = "https://api.aivillage.io/v1"
    }

    if timeoutEnv := os.Getenv("AIVILLAGE_TIMEOUT"); timeoutEnv != "" {
        if duration, err := time.ParseDuration(timeoutEnv); err == nil {
            config.Timeout = duration
        }
    }

    return config
}

func CreateProductionClient(config *ProductionClientConfig) *aivillage.APIClient {
    if config.APIKey == "" {
        log.Fatal("AIVILLAGE_API_KEY environment variable is required")
    }

    // Production HTTP transport
    transport := &http.Transport{
        MaxIdleConns:        100,
        MaxIdleConnsPerHost: 10,
        IdleConnTimeout:     90 * time.Second,
        TLSClientConfig: &tls.Config{
            MinVersion: tls.VersionTLS12,
        },
        DisableCompression: false,
    }

    // Production HTTP client
    httpClient := &http.Client{
        Timeout:   config.Timeout,
        Transport: transport,
    }

    // Configure SDK
    sdkConfig := aivillage.NewConfiguration()
    sdkConfig.Host = config.APIURL
    sdkConfig.HTTPClient = httpClient
    sdkConfig.DefaultHeader = map[string]string{
        "User-Agent": config.UserAgent,
    }

    client := aivillage.NewAPIClient(sdkConfig)

    log.Printf("AIVillage client configured for production: %s", config.APIURL)

    return client
}

// Health check function
func HealthCheck(client *aivillage.APIClient, apiKey string) error {
    auth := context.WithValue(context.Background(), aivillage.ContextAccessToken, apiKey)

    // Simple chat to verify connectivity
    request := aivillage.ChatRequest{
        Message: "Health check",
        Mode:    aivillage.PtrString("fast"),
    }

    ctx, cancel := context.WithTimeout(auth, 30*time.Second)
    defer cancel()

    _, httpResp, err := client.ChatApi.Chat(ctx).ChatRequest(request).Execute()
    if httpResp != nil {
        defer httpResp.Body.Close()
    }

    return err
}

func main() {
    config := LoadProductionConfig()
    client := CreateProductionClient(config)

    // Verify connectivity
    if err := HealthCheck(client, config.APIKey); err != nil {
        log.Printf("Health check failed: %v", err)
    } else {
        log.Println("AIVillage client is healthy")
    }

    // Your application logic here
}
```

### Docker Integration

```dockerfile
# Dockerfile
FROM golang:1.21-alpine AS builder

WORKDIR /app

# Copy Go modules files
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build the application
RUN CGO_ENABLED=0 GOOS=linux go build -o aivillage-app ./cmd/app

FROM alpine:latest

# Install CA certificates for HTTPS
RUN apk --no-cache add ca-certificates

WORKDIR /root/

# Copy binary
COPY --from=builder /app/aivillage-app .

# Environment variables
ENV AIVILLAGE_API_URL=https://api.aivillage.io/v1
ENV AIVILLAGE_TIMEOUT=60s

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD ./aivillage-app -health-check || exit 1

# Run the binary
CMD ["./aivillage-app"]
```

### Kubernetes Deployment

```yaml
# aivillage-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aivillage-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: aivillage-app
  template:
    metadata:
      labels:
        app: aivillage-app
    spec:
      containers:
      - name: aivillage-app
        image: your-registry/aivillage-app:latest
        ports:
        - containerPort: 8080
        env:
        - name: AIVILLAGE_API_KEY
          valueFrom:
            secretKeyRef:
              name: aivillage-secret
              key: api-key
        - name: AIVILLAGE_API_URL
          value: "https://api.aivillage.io/v1"
        - name: AIVILLAGE_TIMEOUT
          value: "60s"
        resources:
          requests:
            memory: "64Mi"
            cpu: "50m"
          limits:
            memory: "128Mi"
            cpu: "100m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Secret
metadata:
  name: aivillage-secret
type: Opaque
data:
  api-key: <base64-encoded-api-key>
```

## Troubleshooting

### Common Issues

**Connection Timeout:**
```go
// Increase timeout
config := aivillage.NewConfiguration()
transport := &http.Transport{}
httpClient := &http.Client{
    Timeout:   120 * time.Second, // 2 minutes
    Transport: transport,
}
config.HTTPClient = httpClient
```

**TLS Certificate Issues:**
```go
// Skip TLS verification (development only)
transport := &http.Transport{
    TLSClientConfig: &tls.Config{
        InsecureSkipVerify: true, // NOT for production
    },
}
```

**Memory Issues:**
```go
// Optimize for memory usage
transport := &http.Transport{
    MaxIdleConns:        10,   // Reduce connection pool
    MaxIdleConnsPerHost: 2,    // Limit per-host connections
    IdleConnTimeout:     30 * time.Second,
}
```

### Debug Logging

```go
import (
    "net/http/httputil"
    "log"
)

// HTTP request/response logging
type LoggingRoundTripper struct {
    rt http.RoundTripper
}

func (lrt *LoggingRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
    // Log request
    if dump, err := httputil.DumpRequest(req, true); err == nil {
        log.Printf("Request: %s", string(dump))
    }

    resp, err := lrt.rt.RoundTrip(req)

    // Log response
    if resp != nil && err == nil {
        if dump, dumpErr := httputil.DumpResponse(resp, true); dumpErr == nil {
            log.Printf("Response: %s", string(dump))
        }
    }

    return resp, err
}

// Usage
transport := &http.Transport{}
httpClient := &http.Client{
    Transport: &LoggingRoundTripper{rt: transport},
}

config := aivillage.NewConfiguration()
config.HTTPClient = httpClient
```

## Support

- **Documentation**: [docs.aivillage.io](https://docs.aivillage.io)
- **API Reference**: [docs.aivillage.io/api](https://docs.aivillage.io/api)
- **GitHub Issues**: [github.com/DNYoussef/AIVillage/issues](https://github.com/DNYoussef/AIVillage/issues)
- **Go Package**: [pkg.go.dev/github.com/DNYoussef/AIVillage/clients/go](https://pkg.go.dev/github.com/DNYoussef/AIVillage/clients/go)

## License

MIT License - see [LICENSE](../../LICENSE) for details.
