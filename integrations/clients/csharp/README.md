# AIVillage C#/.NET SDK

A production-ready C#/.NET client library for the AIVillage API with full async/await support, comprehensive error handling, and modern .NET features.

## Features

- **Modern .NET**: Supports .NET 6+ with nullable reference types and native async/await
- **Type Safety**: Comprehensive model classes with JSON serialization
- **HttpClient Integration**: Uses System.Net.Http with proper connection pooling
- **Automatic Retries**: Exponential backoff with Polly for resilience
- **Circuit Breaker**: Fail-fast behavior with automatic recovery
- **Dependency Injection**: Full DI container support with service registration
- **Configuration Options**: Multiple configuration patterns including IOptions
- **Logging Integration**: Microsoft.Extensions.Logging support

## Installation

### Package Manager Console
```powershell
Install-Package AIVillage.Client
```

### .NET CLI
```bash
dotnet add package AIVillage.Client
```

### PackageReference
```xml
<PackageReference Include="AIVillage.Client" Version="1.0.0" />
```

## Quick Start

```csharp
using AIVillage.Client;
using AIVillage.Client.Models;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

// Program.cs (Minimal API / .NET 6+)
var builder = WebApplication.CreateBuilder(args);

// Register AIVillage client
builder.Services.AddAIVillageClient(options =>
{
    options.BaseUrl = "https://api.aivillage.io/v1";
    options.ApiKey = "your-api-key";
    options.Timeout = TimeSpan.FromSeconds(30);
});

var app = builder.Build();

// Dependency injection usage
app.MapGet("/chat", async (IAIVillageClient client) =>
{
    var request = new ChatRequest
    {
        Message = "How can I optimize neural network inference on mobile devices?",
        AgentPreference = "magi", // Research specialist
        Mode = "comprehensive",
        UserContext = new UserContext
        {
            DeviceType = "mobile",
            BatteryLevel = 75,
            NetworkType = "wifi"
        }
    };

    var response = await client.Chat.SendAsync(request);

    return new
    {
        Agent = response.AgentUsed,
        Response = response.Response,
        ProcessingTime = $"{response.ProcessingTimeMs}ms"
    };
});

app.Run();
```

## Configuration

### appsettings.json Configuration

```json
{
  "AIVillage": {
    "BaseUrl": "https://api.aivillage.io/v1",
    "ApiKey": "your-api-key",
    "Timeout": "00:00:30",
    "MaxRetries": 3,
    "CircuitBreaker": {
      "FailureThreshold": 5,
      "TimeoutSeconds": 60,
      "SamplingDuration": "00:01:00"
    }
  }
}
```

### Service Registration with Options

```csharp
using AIVillage.Client;
using AIVillage.Client.Configuration;

// Startup.cs or Program.cs
public void ConfigureServices(IServiceCollection services)
{
    // Method 1: Configuration binding
    services.Configure<AIVillageOptions>(Configuration.GetSection("AIVillage"));
    services.AddAIVillageClient();

    // Method 2: Direct configuration
    services.AddAIVillageClient(options =>
    {
        options.BaseUrl = "https://api.aivillage.io/v1";
        options.ApiKey = Environment.GetEnvironmentVariable("AIVILLAGE_API_KEY");
        options.Timeout = TimeSpan.FromSeconds(60);
        options.MaxRetries = 5;
    });

    // Method 3: Custom HttpClient configuration
    services.AddHttpClient<IAIVillageClient, AIVillageClient>(client =>
    {
        client.BaseAddress = new Uri("https://api.aivillage.io/v1");
        client.DefaultRequestHeaders.Add("User-Agent", "MyApp/1.0");
        client.Timeout = TimeSpan.FromMinutes(2);
    });
}
```

### Manual Client Creation

```csharp
using AIVillage.Client;
using AIVillage.Client.Configuration;

// Create client manually
var options = new AIVillageOptions
{
    BaseUrl = "https://api.aivillage.io/v1",
    ApiKey = "your-api-key",
    Timeout = TimeSpan.FromSeconds(30)
};

using var httpClient = new HttpClient();
using var client = new AIVillageClient(httpClient, options);

// Use client
var response = await client.Health.GetStatusAsync();
```

## API Reference

### Chat API

Interact with AIVillage's specialized AI agents.

```csharp
using AIVillage.Client.Models;

public class ChatService
{
    private readonly IAIVillageClient _client;
    private readonly ILogger<ChatService> _logger;

    public ChatService(IAIVillageClient client, ILogger<ChatService> logger)
    {
        _client = client;
        _logger = logger;
    }

    public async Task<ChatResponse> GetResearchAdviceAsync(
        string question,
        CancellationToken cancellationToken = default)
    {
        var request = new ChatRequest
        {
            Message = question,
            AgentPreference = "magi", // Research specialist
            Mode = "comprehensive",
            UserContext = new UserContext
            {
                DeviceType = "desktop",
                NetworkType = "ethernet"
            }
        };

        try
        {
            var response = await _client.Chat.SendAsync(request, cancellationToken);

            _logger.LogInformation(
                "Chat completed: Agent={Agent}, Time={ProcessingTime}ms",
                response.AgentUsed,
                response.ProcessingTimeMs
            );

            return response;
        }
        catch (AIVillageException ex)
        {
            _logger.LogError(ex, "Chat request failed: {ErrorCode}", ex.ErrorCode);
            throw;
        }
    }

    public async Task<ChatResponse> GetMobileOptimizedAdviceAsync(
        string question,
        int batteryLevel,
        string networkType,
        CancellationToken cancellationToken = default)
    {
        var request = new ChatRequest
        {
            Message = question,
            AgentPreference = "navigator", // Mobile optimization specialist
            Mode = "balanced",
            UserContext = new UserContext
            {
                DeviceType = "mobile",
                BatteryLevel = batteryLevel,
                NetworkType = networkType,
                DataBudgetMb = networkType == "cellular" ? 50 : null
            }
        };

        return await _client.Chat.SendAsync(request, cancellationToken);
    }
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

```csharp
using AIVillage.Client.Models;

public class KnowledgeService
{
    private readonly IAIVillageClient _client;

    public KnowledgeService(IAIVillageClient client)
    {
        _client = client;
    }

    public async Task<QueryResult> SearchKnowledgeAsync(
        string query,
        QueryMode mode = QueryMode.Comprehensive,
        CancellationToken cancellationToken = default)
    {
        var request = new QueryRequest
        {
            Query = query,
            Mode = mode,
            IncludeSources = true,
            MaxResults = 10
        };

        var result = await _client.RAG.ProcessQueryAsync(request, cancellationToken);

        Console.WriteLine($"Query ID: {result.QueryId}");
        Console.WriteLine($"Response: {result.Response}");
        Console.WriteLine($"Bayesian confidence: {result.Metadata.BayesianConfidence:F3}");

        // Process sources with confidence scores
        foreach (var source in result.Sources)
        {
            Console.WriteLine($"Source: {source.Title}");
            Console.WriteLine($"  Confidence: {source.Confidence:F3}");
            Console.WriteLine($"  Type: {source.SourceType}");
            if (!string.IsNullOrEmpty(source.Url))
            {
                Console.WriteLine($"  URL: {source.Url}");
            }
        }

        return result;
    }

    public async Task<IList<QueryResult>> BatchSearchAsync(
        IEnumerable<string> queries,
        CancellationToken cancellationToken = default)
    {
        var tasks = queries.Select(query =>
            _client.RAG.ProcessQueryAsync(new QueryRequest
            {
                Query = query,
                Mode = QueryMode.Fast,
                IncludeSources = false
            }, cancellationToken)
        );

        var results = await Task.WhenAll(tasks);
        return results.ToList();
    }
}
```

### Agents API

Manage and monitor AI agents.

```csharp
using AIVillage.Client.Models;

public class AgentManagementService
{
    private readonly IAIVillageClient _client;
    private readonly ILogger<AgentManagementService> _logger;

    public AgentManagementService(IAIVillageClient client, ILogger<AgentManagementService> logger)
    {
        _client = client;
        _logger = logger;
    }

    public async Task<AgentListResponse> GetAvailableAgentsAsync(
        string category = "knowledge",
        bool availableOnly = true,
        CancellationToken cancellationToken = default)
    {
        var agents = await _client.Agents.ListAgentsAsync(
            category: category,
            availableOnly: availableOnly,
            cancellationToken: cancellationToken
        );

        foreach (var agent in agents.Agents)
        {
            _logger.LogInformation(
                "Agent: {Name}, Status: {Status}, Load: {Load}%, Capabilities: {Capabilities}",
                agent.Name,
                agent.Status,
                agent.CurrentLoad,
                string.Join(", ", agent.Capabilities)
            );
        }

        return agents;
    }

    public async Task<TaskResult> AssignTaskAsync(
        string agentId,
        string taskDescription,
        TaskPriority priority = TaskPriority.High,
        TimeSpan? timeout = null,
        CancellationToken cancellationToken = default)
    {
        var request = new AgentTaskRequest
        {
            TaskDescription = taskDescription,
            Priority = priority,
            TimeoutSeconds = (int)(timeout?.TotalSeconds ?? 600),
            Context = new Dictionary<string, object>
            {
                ["domain"] = "ai_research",
                ["depth"] = "comprehensive",
                ["include_implementation"] = true
            }
        };

        var result = await _client.Agents.AssignTaskAsync(agentId, request, cancellationToken);

        _logger.LogInformation(
            "Task assigned: {TaskId}, Estimated completion: {Completion}",
            result.TaskId,
            result.EstimatedCompletionTime
        );

        return result;
    }
}
```

### P2P API

Monitor peer-to-peer mesh network status.

```csharp
public class NetworkMonitoringService
{
    private readonly IAIVillageClient _client;

    public NetworkMonitoringService(IAIVillageClient client)
    {
        _client = client;
    }

    public async Task<P2PStatus> GetNetworkStatusAsync(CancellationToken cancellationToken = default)
    {
        var status = await _client.P2P.GetStatusAsync(cancellationToken);

        Console.WriteLine($"Network status: {status.Status}");
        Console.WriteLine($"Connected peers: {status.PeerCount}");
        Console.WriteLine($"Network health: {status.HealthScore}");

        return status;
    }

    public async Task MonitorTransportTypesAsync(CancellationToken cancellationToken = default)
    {
        var transportTypes = new[] { "bitchat", "betanet" };

        foreach (var transport in transportTypes)
        {
            var peers = await _client.P2P.ListPeersAsync(transport, cancellationToken);

            Console.WriteLine($"\n{transport.ToUpper()} peers ({peers.Peers.Count}):");
            foreach (var peer in peers.Peers)
            {
                Console.WriteLine($"  {peer.Id[..8]}... - {peer.Status} ({peer.LatencyMs}ms)");
            }
        }
    }
}
```

### Digital Twin API

Privacy-preserving personal AI assistant.

```csharp
using AIVillage.Client.Models;

public class DigitalTwinService
{
    private readonly IAIVillageClient _client;

    public DigitalTwinService(IAIVillageClient client)
    {
        _client = client;
    }

    public async Task<DigitalTwinProfile> GetProfileAsync(CancellationToken cancellationToken = default)
    {
        var profile = await _client.DigitalTwin.GetProfileAsync(cancellationToken);

        Console.WriteLine($"Model size: {profile.ModelSizeMb}MB");
        Console.WriteLine($"Accuracy: {profile.LearningStats.AccuracyScore:F3}");
        Console.WriteLine($"Privacy level: {profile.PrivacySettings.Level}");

        return profile;
    }

    public async Task UpdateInteractionDataAsync(
        double userSatisfaction,
        double predictionAccuracy,
        string context,
        CancellationToken cancellationToken = default)
    {
        var update = new DigitalTwinDataUpdate
        {
            DataType = "interaction",
            DataPoints = new[]
            {
                new DataPoint
                {
                    Timestamp = DateTimeOffset.UtcNow,
                    Content = new Dictionary<string, object>
                    {
                        ["interaction_type"] = "chat",
                        ["user_satisfaction"] = userSatisfaction,
                        ["context"] = context
                    },
                    PredictionAccuracy = predictionAccuracy
                }
            }
        };

        await _client.DigitalTwin.UpdateDataAsync(update, cancellationToken);
        Console.WriteLine("Digital twin updated with new interaction data");
    }
}
```

## Error Handling

### Exception Types

```csharp
using AIVillage.Client.Exceptions;

public class RobustChatService
{
    private readonly IAIVillageClient _client;
    private readonly ILogger<RobustChatService> _logger;

    public RobustChatService(IAIVillageClient client, ILogger<RobustChatService> logger)
    {
        _client = client;
        _logger = logger;
    }

    public async Task<ChatResponse?> SafeChatAsync(
        string message,
        CancellationToken cancellationToken = default)
    {
        try
        {
            var request = new ChatRequest
            {
                Message = message,
                Mode = "fast"
            };

            return await _client.Chat.SendAsync(request, cancellationToken);
        }
        catch (RateLimitException ex)
        {
            _logger.LogWarning("Rate limited. Retry after {RetryAfter}s", ex.RetryAfter);

            // Automatic retry after rate limit period
            await Task.Delay(TimeSpan.FromSeconds(ex.RetryAfter), cancellationToken);
            return await _client.Chat.SendAsync(new ChatRequest { Message = message, Mode = "fast" }, cancellationToken);
        }
        catch (AuthenticationException ex)
        {
            _logger.LogError(ex, "Authentication failed: {Detail}", ex.Detail);
            _logger.LogError("Please check your API key");
            throw;
        }
        catch (ValidationException ex)
        {
            _logger.LogError(ex, "Validation error: {Detail}", ex.Detail);
            _logger.LogError("Field errors: {FieldErrors}", string.Join(", ", ex.FieldErrors));
            throw;
        }
        catch (ServerException ex)
        {
            _logger.LogError(ex, "Server error: {Status} - {Detail}", ex.StatusCode, ex.Detail);
            _logger.LogError("Request ID: {RequestId}", ex.RequestId);
            throw;
        }
        catch (AIVillageException ex)
        {
            _logger.LogError(ex, "General API error: {ErrorCode}", ex.ErrorCode);
            throw;
        }
        catch (HttpRequestException ex)
        {
            _logger.LogError(ex, "Network error occurred");
            throw;
        }
        catch (TaskCanceledException ex) when (ex.InnerException is TimeoutException)
        {
            _logger.LogError(ex, "Request timeout");
            throw;
        }
    }
}
```

### Retry Pattern with Polly

```csharp
using Polly;
using Polly.CircuitBreaker;
using Polly.Extensions.Http;
using AIVillage.Client.Exceptions;

public class ResilientChatService
{
    private readonly IAIVillageClient _client;
    private readonly IAsyncPolicy<ChatResponse> _retryPolicy;

    public ResilientChatService(IAIVillageClient client)
    {
        _client = client;

        // Configure retry policy with exponential backoff
        _retryPolicy = Policy
            .Handle<ServerException>()
            .Or<HttpRequestException>()
            .OrResult<ChatResponse>(r => r == null)
            .WaitAndRetryAsync(
                retryCount: 3,
                sleepDurationProvider: retryAttempt => TimeSpan.FromSeconds(Math.Pow(2, retryAttempt)),
                onRetry: (outcome, timespan, retryCount, context) =>
                {
                    Console.WriteLine($"Retry {retryCount} after {timespan} seconds");
                }
            )
            .WrapAsync(
                // Circuit breaker to fail fast when service is down
                Policy.Handle<ServerException>()
                    .CircuitBreakerAsync(
                        exceptionsAllowedBeforeBreaking: 5,
                        durationOfBreak: TimeSpan.FromMinutes(1),
                        onBreak: (exception, duration) =>
                        {
                            Console.WriteLine($"Circuit breaker opened for {duration}");
                        },
                        onReset: () =>
                        {
                            Console.WriteLine("Circuit breaker closed");
                        }
                    )
            );
    }

    public async Task<ChatResponse> ResilientChatAsync(
        string message,
        string agentPreference = "any",
        CancellationToken cancellationToken = default)
    {
        return await _retryPolicy.ExecuteAsync(async () =>
        {
            var request = new ChatRequest
            {
                Message = message,
                AgentPreference = agentPreference,
                Mode = "balanced"
            };

            return await _client.Chat.SendAsync(request, cancellationToken);
        });
    }
}
```

## Advanced Usage

### Dependency Injection with Multiple Clients

```csharp
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Options;

public class Startup
{
    public void ConfigureServices(IServiceCollection services)
    {
        // Register multiple clients for different environments
        services.AddHttpClient<IAIVillageClient, ProductionAIVillageClient>("production", client =>
        {
            client.BaseAddress = new Uri("https://api.aivillage.io/v1");
            client.Timeout = TimeSpan.FromSeconds(30);
        });

        services.AddHttpClient<IAIVillageClient, StagingAIVillageClient>("staging", client =>
        {
            client.BaseAddress = new Uri("https://staging-api.aivillage.io/v1");
            client.Timeout = TimeSpan.FromMinutes(2);
        });

        // Register factory for client selection
        services.AddSingleton<IAIVillageClientFactory, AIVillageClientFactory>();
    }
}

public interface IAIVillageClientFactory
{
    IAIVillageClient CreateClient(string environment);
}

public class AIVillageClientFactory : IAIVillageClientFactory
{
    private readonly IServiceProvider _serviceProvider;

    public AIVillageClientFactory(IServiceProvider serviceProvider)
    {
        _serviceProvider = serviceProvider;
    }

    public IAIVillageClient CreateClient(string environment)
    {
        return environment switch
        {
            "production" => _serviceProvider.GetRequiredKeyedService<IAIVillageClient>("production"),
            "staging" => _serviceProvider.GetRequiredKeyedService<IAIVillageClient>("staging"),
            _ => throw new ArgumentException($"Unknown environment: {environment}")
        };
    }
}
```

### Custom Middleware and Request Modification

```csharp
using System.Net.Http;

public class AIVillageClientWithCustomHeaders : AIVillageClient
{
    public AIVillageClientWithCustomHeaders(HttpClient httpClient, AIVillageOptions options)
        : base(httpClient, options)
    {
        // Add custom headers
        httpClient.DefaultRequestHeaders.Add("X-Client-Version", "1.0.0");
        httpClient.DefaultRequestHeaders.Add("X-Environment", Environment.GetEnvironmentVariable("ENVIRONMENT") ?? "development");
    }

    protected override async Task<HttpRequestMessage> PrepareRequestAsync(HttpRequestMessage request)
    {
        // Add custom logic before each request
        request.Headers.Add("X-Request-Id", Guid.NewGuid().ToString());
        request.Headers.Add("X-Timestamp", DateTimeOffset.UtcNow.ToUnixTimeSeconds().ToString());

        // Add correlation ID from current context
        if (Activity.Current?.Id is not null)
        {
            request.Headers.Add("X-Correlation-Id", Activity.Current.Id);
        }

        return await base.PrepareRequestAsync(request);
    }
}
```

### Idempotency for Safe Retries

```csharp
using System.Security.Cryptography;
using System.Text;

public class IdempotentChatService
{
    private readonly IAIVillageClient _client;

    public IdempotentChatService(IAIVillageClient client)
    {
        _client = client;
    }

    public async Task<ChatResponse> IdempotentChatAsync(
        string message,
        string agentPreference = "any",
        CancellationToken cancellationToken = default)
    {
        var idempotencyKey = GenerateIdempotencyKey("chat", message, agentPreference);

        var request = new ChatRequest
        {
            Message = message,
            AgentPreference = agentPreference,
            Mode = "balanced"
        };

        // Add idempotency key to request headers
        var headers = new Dictionary<string, string>
        {
            ["Idempotency-Key"] = idempotencyKey
        };

        return await _client.Chat.SendAsync(request, headers, cancellationToken);
    }

    private static string GenerateIdempotencyKey(string operation, params string[] parameters)
    {
        var content = $"{operation}-{string.Join("-", parameters)}-{DateTime.UtcNow:yyyyMMdd}";
        var hash = SHA256.HashData(Encoding.UTF8.GetBytes(content));
        return Convert.ToHexString(hash)[..16].ToLowerInvariant();
    }
}
```

## Testing

### Unit Testing with xUnit and Moq

```csharp
using Xunit;
using Moq;
using Microsoft.Extensions.Logging;
using AIVillage.Client;
using AIVillage.Client.Models;

public class ChatServiceTests
{
    private readonly Mock<IAIVillageClient> _mockClient;
    private readonly Mock<ILogger<ChatService>> _mockLogger;
    private readonly ChatService _chatService;

    public ChatServiceTests()
    {
        _mockClient = new Mock<IAIVillageClient>();
        _mockLogger = new Mock<ILogger<ChatService>>();
        _chatService = new ChatService(_mockClient.Object, _mockLogger.Object);
    }

    [Fact]
    public async Task GetResearchAdviceAsync_ValidRequest_ReturnsResponse()
    {
        // Arrange
        var expectedResponse = new ChatResponse
        {
            Response = "Mocked response for testing",
            AgentUsed = "magi",
            ProcessingTimeMs = 100,
            ConversationId = "test-conv-123",
            Metadata = new ChatMetadata
            {
                Confidence = 0.95m,
                FeaturesUsed = new[] { "test-feature" }
            }
        };

        _mockClient.Setup(x => x.Chat.SendAsync(It.IsAny<ChatRequest>(), It.IsAny<CancellationToken>()))
                   .ReturnsAsync(expectedResponse);

        // Act
        var result = await _chatService.GetResearchAdviceAsync("Test question");

        // Assert
        Assert.Equal(expectedResponse.Response, result.Response);
        Assert.Equal(expectedResponse.AgentUsed, result.AgentUsed);

        _mockClient.Verify(x => x.Chat.SendAsync(
            It.Is<ChatRequest>(r =>
                r.Message == "Test question" &&
                r.AgentPreference == "magi" &&
                r.Mode == "comprehensive"),
            It.IsAny<CancellationToken>()),
            Times.Once);
    }

    [Fact]
    public async Task GetResearchAdviceAsync_ClientThrowsException_LogsErrorAndRethrows()
    {
        // Arrange
        var exception = new AIVillageException("Test error", "TEST_ERROR");
        _mockClient.Setup(x => x.Chat.SendAsync(It.IsAny<ChatRequest>(), It.IsAny<CancellationToken>()))
                   .ThrowsAsync(exception);

        // Act & Assert
        var thrownException = await Assert.ThrowsAsync<AIVillageException>(
            () => _chatService.GetResearchAdviceAsync("Test question")
        );

        Assert.Equal(exception.ErrorCode, thrownException.ErrorCode);

        // Verify logging
        _mockLogger.Verify(
            x => x.Log(
                LogLevel.Error,
                It.IsAny<EventId>(),
                It.Is<It.IsAnyType>((v, t) => v.ToString().Contains("Chat request failed")),
                exception,
                It.IsAny<Func<It.IsAnyType, Exception, string>>()),
            Times.Once);
    }
}
```

### Integration Testing

```csharp
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using Xunit;

public class AIVillageIntegrationTests : IClassFixture<WebApplicationFactory<Program>>
{
    private readonly WebApplicationFactory<Program> _factory;
    private readonly HttpClient _client;

    public AIVillageIntegrationTests(WebApplicationFactory<Program> factory)
    {
        _factory = factory;
        _client = _factory.CreateClient();
    }

    [Fact]
    [Trait("Category", "Integration")]
    public async Task ChatEndpoint_ValidRequest_ReturnsExpectedResponse()
    {
        // Arrange
        var apiKey = Environment.GetEnvironmentVariable("AIVILLAGE_TEST_API_KEY");
        if (string.IsNullOrEmpty(apiKey))
        {
            Skip.If(true, "AIVILLAGE_TEST_API_KEY not set");
        }

        // Use test server with real API key
        var testServer = _factory.WithWebHostBuilder(builder =>
        {
            builder.ConfigureServices(services =>
            {
                services.AddAIVillageClient(options =>
                {
                    options.BaseUrl = "https://staging-api.aivillage.io/v1";
                    options.ApiKey = apiKey;
                });
            });
        });

        var client = testServer.CreateClient();

        // Act
        var response = await client.GetAsync("/chat");

        // Assert
        response.EnsureSuccessStatusCode();
        var content = await response.Content.ReadAsStringAsync();

        Assert.Contains("Agent", content);
        Assert.Contains("Response", content);
        Assert.Contains("ProcessingTime", content);
    }
}
```

## Performance and Monitoring

### Performance Monitoring

```csharp
using System.Diagnostics;
using Microsoft.Extensions.Logging;

public class PerformanceMonitoringService
{
    private readonly IAIVillageClient _client;
    private readonly ILogger<PerformanceMonitoringService> _logger;

    public PerformanceMonitoringService(IAIVillageClient client, ILogger<PerformanceMonitoringService> logger)
    {
        _client = client;
        _logger = logger;
    }

    public async Task<PerformanceMetrics> BenchmarkChatPerformanceAsync(int requestCount = 10)
    {
        var stopwatch = Stopwatch.StartNew();
        var successCount = 0;
        var errorCount = 0;
        var totalResponseTime = 0L;

        // Sequential requests
        var sequentialStopwatch = Stopwatch.StartNew();
        for (int i = 0; i < requestCount; i++)
        {
            try
            {
                var response = await _client.Chat.SendAsync(new ChatRequest
                {
                    Message = $"Test message {i}",
                    Mode = "fast"
                });

                successCount++;
                totalResponseTime += response.ProcessingTimeMs;
            }
            catch (Exception ex)
            {
                errorCount++;
                _logger.LogWarning(ex, "Request {RequestNumber} failed", i);
            }
        }
        var sequentialTime = sequentialStopwatch.Elapsed;

        // Concurrent requests
        var concurrentStopwatch = Stopwatch.StartNew();
        var tasks = Enumerable.Range(0, requestCount).Select(i =>
            _client.Chat.SendAsync(new ChatRequest
            {
                Message = $"Concurrent test {i}",
                Mode = "fast"
            })
        ).ToArray();

        try
        {
            await Task.WhenAll(tasks);
            var concurrentSuccessCount = tasks.Count(t => t.IsCompletedSuccessfully);
            successCount += concurrentSuccessCount;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Some concurrent requests failed");
        }
        var concurrentTime = concurrentStopwatch.Elapsed;

        var metrics = new PerformanceMetrics
        {
            TotalRequests = requestCount * 2, // Sequential + concurrent
            SuccessfulRequests = successCount,
            FailedRequests = errorCount,
            SequentialTime = sequentialTime,
            ConcurrentTime = concurrentTime,
            AverageResponseTime = successCount > 0 ? totalResponseTime / successCount : 0,
            RequestsPerSecond = requestCount / sequentialTime.TotalSeconds,
            ConcurrentRequestsPerSecond = requestCount / concurrentTime.TotalSeconds
        };

        _logger.LogInformation(
            "Performance test completed: {SuccessRate:P2} success rate, " +
            "{RequestsPerSecond:F1} req/s sequential, {ConcurrentRequestsPerSecond:F1} req/s concurrent",
            (double)successCount / (requestCount * 2),
            metrics.RequestsPerSecond,
            metrics.ConcurrentRequestsPerSecond
        );

        return metrics;
    }
}

public record PerformanceMetrics
{
    public int TotalRequests { get; init; }
    public int SuccessfulRequests { get; init; }
    public int FailedRequests { get; init; }
    public TimeSpan SequentialTime { get; init; }
    public TimeSpan ConcurrentTime { get; init; }
    public long AverageResponseTime { get; init; }
    public double RequestsPerSecond { get; init; }
    public double ConcurrentRequestsPerSecond { get; init; }
}
```

## Deployment

### ASP.NET Core Production Configuration

```csharp
// Program.cs
using AIVillage.Client;
using Serilog;

var builder = WebApplication.CreateBuilder(args);

// Configure logging
builder.Host.UseSerilog((context, configuration) =>
{
    configuration
        .ReadFrom.Configuration(context.Configuration)
        .WriteTo.Console()
        .WriteTo.ApplicationInsights(context.Configuration.GetConnectionString("ApplicationInsights"));
});

// Configure AIVillage client with health checks
builder.Services.AddAIVillageClient(builder.Configuration.GetSection("AIVillage"));
builder.Services.AddHealthChecks()
    .AddCheck<AIVillageHealthCheck>("aivillage-api");

// Configure HTTP client with Polly
builder.Services.AddHttpClient<IAIVillageClient>()
    .AddPolicyHandler(GetRetryPolicy())
    .AddPolicyHandler(GetCircuitBreakerPolicy());

var app = builder.Build();

// Configure production middleware
if (app.Environment.IsProduction())
{
    app.UseExceptionHandler("/error");
    app.UseHsts();
}

app.UseHealthChecks("/health");
app.MapControllers();

app.Run();

static IAsyncPolicy<HttpResponseMessage> GetRetryPolicy()
{
    return HttpPolicyExtensions
        .HandleTransientHttpError()
        .WaitAndRetryAsync(
            retryCount: 3,
            sleepDurationProvider: retryAttempt => TimeSpan.FromSeconds(Math.Pow(2, retryAttempt))
        );
}

static IAsyncPolicy<HttpResponseMessage> GetCircuitBreakerPolicy()
{
    return HttpPolicyExtensions
        .HandleTransientHttpError()
        .CircuitBreakerAsync(
            exceptionsAllowedBeforeBreaking: 3,
            durationOfBreak: TimeSpan.FromSeconds(30)
        );
}
```

### Docker Support

```dockerfile
# Dockerfile
FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS base
WORKDIR /app
EXPOSE 80
EXPOSE 443

FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src
COPY ["MyApp.csproj", "."]
RUN dotnet restore "MyApp.csproj"

COPY . .
WORKDIR "/src"
RUN dotnet build "MyApp.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "MyApp.csproj" -c Release -o /app/publish /p:UseAppHost=false

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:80/health || exit 1

ENTRYPOINT ["dotnet", "MyApp.dll"]
```

## Troubleshooting

### Common Issues and Solutions

```csharp
// SSL/TLS Certificate Issues
var options = new AIVillageOptions
{
    BaseUrl = "https://api.aivillage.io/v1",
    ApiKey = "your-api-key"
};

// For development only - bypass SSL validation
var handler = new HttpClientHandler()
{
    ServerCertificateCustomValidationCallback = HttpClientHandler.DangerousAcceptAnyServerCertificateValidator
};

using var httpClient = new HttpClient(handler);
using var client = new AIVillageClient(httpClient, options);
```

### Debug Logging

```csharp
// Enable detailed HTTP logging
builder.Services.AddLogging(builder =>
{
    builder.AddConsole();
    builder.SetMinimumLevel(LogLevel.Debug);
});

// Add HTTP client logging
builder.Services.AddHttpClient<IAIVillageClient>()
    .AddLogger(); // This will log all HTTP requests/responses
```

## Support

- **Documentation**: [docs.aivillage.io](https://docs.aivillage.io)
- **NuGet Package**: [nuget.org/packages/AIVillage.Client](https://nuget.org/packages/AIVillage.Client)
- **GitHub Issues**: [github.com/DNYoussef/AIVillage/issues](https://github.com/DNYoussef/AIVillage/issues)
- **API Reference**: [docs.aivillage.io/api](https://docs.aivillage.io/api)

## License

MIT License - see [LICENSE](../../LICENSE) for details.
