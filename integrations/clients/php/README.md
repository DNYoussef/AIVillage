# AIVillage PHP SDK

A comprehensive, production-ready PHP client library for the AIVillage API with modern PHP 8+ features, PSR compliance, and built-in reliability patterns.

## Features

- **Modern PHP**: Requires PHP 8.1+ with typed properties, union types, and attributes
- **PSR Compliance**: PSR-4 autoloading, PSR-7 HTTP messages, PSR-18 HTTP client
- **Type Safety**: Strict typing with comprehensive data transfer objects
- **Async Support**: ReactPHP integration for non-blocking operations
- **Reliability**: Automatic retries with exponential backoff and circuit breaker
- **Logging**: PSR-3 compliant logging with configurable levels
- **Caching**: PSR-6 and PSR-16 cache integration for performance
- **Validation**: Input validation with detailed error messages

## Installation

### Composer

```bash
# Install via Composer
composer require aivillage/php-client

# With development dependencies
composer require aivillage/php-client --dev

# Specific version
composer require "aivillage/php-client:^1.0"
```

### Requirements

```json
{
    "require": {
        "php": "^8.1",
        "guzzlehttp/guzzle": "^7.0",
        "psr/http-client": "^1.0",
        "psr/log": "^3.0",
        "psr/cache": "^3.0"
    }
}
```

## Quick Start

```php
<?php

require_once 'vendor/autoload.php';

use AIVillage\Client\AIVillageClient;
use AIVillage\Client\Configuration;
use AIVillage\Client\Models\ChatRequest;
use AIVillage\Client\Models\UserContext;
use AIVillage\Client\Exceptions\AIVillageException;

// Configure the client
$config = new Configuration(
    baseUrl: 'https://api.aivillage.io/v1',
    apiKey: 'your-api-key',
    timeout: 30.0,
    retries: 3
);

// Create client instance
$client = new AIVillageClient($config);

// Chat with AI agents
try {
    $request = new ChatRequest(
        message: 'How can I optimize machine learning model inference on mobile devices?',
        agentPreference: 'magi', // Research specialist
        mode: 'comprehensive',
        userContext: new UserContext(
            deviceType: 'mobile',
            batteryLevel: 75,
            networkType: 'wifi'
        )
    );

    $response = $client->chat()->send($request);

    echo "Agent: {$response->agentUsed}\n";
    echo "Response: {$response->response}\n";
    echo "Processing time: {$response->processingTimeMs}ms\n";

} catch (AIVillageException $e) {
    echo "API Error: {$e->getMessage()}\n";
    echo "Error Code: {$e->getErrorCode()}\n";
}
```

## Configuration

### Basic Configuration

```php
<?php

use AIVillage\Client\Configuration;
use AIVillage\Client\AIVillageClient;

// Method 1: Constructor parameters
$config = new Configuration(
    baseUrl: 'https://api.aivillage.io/v1',
    apiKey: 'your-api-key',
    timeout: 30.0,
    retries: 3,
    verifySSL: true
);

// Method 2: Environment variables
$config = Configuration::fromEnvironment();
// Reads from: AIVILLAGE_API_URL, AIVILLAGE_API_KEY, etc.

// Method 3: Array configuration
$config = Configuration::fromArray([
    'base_url' => 'https://api.aivillage.io/v1',
    'api_key' => getenv('AIVILLAGE_API_KEY'),
    'timeout' => 60.0,
    'retries' => 5,
    'circuit_breaker' => [
        'failure_threshold' => 5,
        'timeout' => 60,
        'sample_duration' => 300
    ]
]);

$client = new AIVillageClient($config);
```

### Advanced Configuration with Dependency Injection

```php
<?php

use AIVillage\Client\AIVillageClient;
use AIVillage\Client\Configuration;
use GuzzleHttp\Client as GuzzleClient;
use GuzzleHttp\HandlerStack;
use GuzzleHttp\Middleware;
use Psr\Log\LoggerInterface;
use Psr\Cache\CacheItemPoolInterface;

class AIVillageServiceProvider
{
    public function createClient(
        LoggerInterface $logger,
        CacheItemPoolInterface $cache
    ): AIVillageClient {
        // Create Guzzle stack with middleware
        $stack = HandlerStack::create();

        // Add retry middleware
        $stack->push(Middleware::retry($this->createRetryDecider(), $this->createRetryDelay()));

        // Add logging middleware
        $stack->push(Middleware::log($logger, new \GuzzleHttp\MessageFormatter()));

        // Create HTTP client
        $httpClient = new GuzzleClient([
            'handler' => $stack,
            'timeout' => 30.0,
            'connect_timeout' => 10.0,
            'verify' => true,
            'headers' => [
                'User-Agent' => 'AIVillage-PHP-Client/1.0',
                'Accept' => 'application/json',
                'Content-Type' => 'application/json'
            ]
        ]);

        // Configuration
        $config = new Configuration(
            baseUrl: $_ENV['AIVILLAGE_API_URL'] ?? 'https://api.aivillage.io/v1',
            apiKey: $_ENV['AIVILLAGE_API_KEY'],
            timeout: 30.0,
            retries: 3
        );

        return new AIVillageClient($config, $httpClient, $logger, $cache);
    }

    private function createRetryDecider(): callable
    {
        return function ($retries, $request, $response = null, $exception = null) {
            if ($retries >= 3) {
                return false;
            }

            if ($exception instanceof \GuzzleHttp\Exception\ConnectException) {
                return true;
            }

            if ($response && $response->getStatusCode() >= 500) {
                return true;
            }

            if ($response && $response->getStatusCode() === 429) {
                return true;
            }

            return false;
        };
    }

    private function createRetryDelay(): callable
    {
        return function ($retries) {
            return 1000 * (2 ** $retries); // Exponential backoff in milliseconds
        };
    }
}
```

## API Reference

### Chat API

Interact with AIVillage's specialized AI agents.

```php
<?php

use AIVillage\Client\Models\ChatRequest;
use AIVillage\Client\Models\UserContext;
use AIVillage\Client\Exceptions\RateLimitException;
use AIVillage\Client\Exceptions\ValidationException;

class ChatService
{
    public function __construct(
        private AIVillageClient $client,
        private LoggerInterface $logger
    ) {}

    public function getResearchAdvice(
        string $question,
        string $mode = 'comprehensive'
    ): ChatResponse {
        $request = new ChatRequest(
            message: $question,
            agentPreference: 'magi', // Research specialist
            mode: $mode,
            userContext: new UserContext(
                deviceType: 'desktop',
                networkType: 'ethernet'
            )
        );

        try {
            $response = $this->client->chat()->send($request);

            $this->logger->info('Chat completed', [
                'agent' => $response->agentUsed,
                'processing_time' => $response->processingTimeMs,
                'confidence' => $response->metadata->confidence
            ]);

            return $response;

        } catch (RateLimitException $e) {
            $this->logger->warning('Rate limited', [
                'retry_after' => $e->getRetryAfter()
            ]);
            throw $e;

        } catch (ValidationException $e) {
            $this->logger->error('Validation failed', [
                'field_errors' => $e->getFieldErrors()
            ]);
            throw $e;
        }
    }

    public function getMobileOptimizedAdvice(
        string $question,
        int $batteryLevel,
        string $networkType,
        ?int $dataBudgetMb = null
    ): ChatResponse {
        $request = new ChatRequest(
            message: $question,
            agentPreference: 'navigator', // Mobile optimization specialist
            mode: 'balanced',
            userContext: new UserContext(
                deviceType: 'mobile',
                batteryLevel: $batteryLevel,
                networkType: $networkType,
                dataBudgetMb: $dataBudgetMb
            )
        );

        return $this->client->chat()->send($request);
    }

    public function continueChatConversation(
        string $message,
        string $conversationId,
        string $agentPreference = 'any'
    ): ChatResponse {
        $request = new ChatRequest(
            message: $message,
            conversationId: $conversationId,
            agentPreference: $agentPreference,
            mode: 'balanced'
        );

        return $this->client->chat()->send($request);
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

```php
<?php

use AIVillage\Client\Models\QueryRequest;

class KnowledgeService
{
    public function __construct(private AIVillageClient $client) {}

    public function searchKnowledge(
        string $query,
        string $mode = 'comprehensive',
        bool $includeSources = true,
        int $maxResults = 10
    ): QueryResult {
        $request = new QueryRequest(
            query: $query,
            mode: $mode,
            includeSources: $includeSources,
            maxResults: $maxResults
        );

        $result = $this->client->rag()->processQuery($request);

        echo "Query ID: {$result->queryId}\n";
        echo "Response: {$result->response}\n";
        echo "Bayesian confidence: " . number_format($result->metadata->bayesianConfidence, 3) . "\n";

        // Process sources with confidence scores
        foreach ($result->sources as $source) {
            echo "Source: {$source->title}\n";
            echo "  Confidence: " . number_format($source->confidence, 3) . "\n";
            echo "  Type: {$source->sourceType}\n";
            if ($source->url) {
                echo "  URL: {$source->url}\n";
            }
        }

        return $result;
    }

    /**
     * Process multiple queries concurrently using async operations
     */
    public function batchSearch(array $queries, string $mode = 'fast'): array
    {
        $promises = [];

        foreach ($queries as $query) {
            $request = new QueryRequest(
                query: $query,
                mode: $mode,
                includeSources: false
            );

            // Create async promise (requires ReactPHP integration)
            $promises[] = $this->client->rag()->processQueryAsync($request);
        }

        // Wait for all queries to complete
        $results = \React\Promise\all($promises)->wait();

        foreach ($results as $i => $result) {
            echo "Query " . ($i + 1) . ": " . substr($result->response, 0, 100) . "...\n";
        }

        return $results;
    }
}
```

### Agents API

Manage and monitor AI agents.

```php
<?php

use AIVillage\Client\Models\AgentTaskRequest;

class AgentManagementService
{
    public function __construct(
        private AIVillageClient $client,
        private LoggerInterface $logger
    ) {}

    public function getAvailableAgents(
        string $category = 'knowledge',
        bool $availableOnly = true
    ): AgentListResponse {
        $agents = $this->client->agents()->listAgents(
            category: $category,
            availableOnly: $availableOnly
        );

        foreach ($agents->agents as $agent) {
            $this->logger->info('Agent discovered', [
                'name' => $agent->name,
                'status' => $agent->status,
                'load' => $agent->currentLoad,
                'capabilities' => implode(', ', $agent->capabilities)
            ]);
        }

        return $agents;
    }

    public function assignTask(
        string $agentId,
        string $taskDescription,
        string $priority = 'high',
        int $timeoutSeconds = 600,
        array $context = []
    ): TaskResult {
        $request = new AgentTaskRequest(
            taskDescription: $taskDescription,
            priority: $priority,
            timeoutSeconds: $timeoutSeconds,
            context: array_merge([
                'domain' => 'ai_research',
                'depth' => 'comprehensive',
                'include_implementation' => true
            ], $context)
        );

        $result = $this->client->agents()->assignTask($agentId, $request);

        $this->logger->info('Task assigned', [
            'task_id' => $result->taskId,
            'agent_id' => $agentId,
            'estimated_completion' => $result->estimatedCompletionTime
        ]);

        return $result;
    }

    public function monitorTaskProgress(string $taskId): TaskStatus
    {
        return $this->client->agents()->getTaskStatus($taskId);
    }
}
```

### P2P API

Monitor peer-to-peer mesh network status.

```php
<?php

class NetworkMonitoringService
{
    public function __construct(private AIVillageClient $client) {}

    public function getNetworkStatus(): P2PStatus
    {
        $status = $this->client->p2p()->getStatus();

        echo "Network status: {$status->status}\n";
        echo "Connected peers: {$status->peerCount}\n";
        echo "Network health: {$status->healthScore}\n";

        return $status;
    }

    public function monitorTransportTypes(): void
    {
        $transportTypes = ['bitchat', 'betanet'];

        foreach ($transportTypes as $transport) {
            $peers = $this->client->p2p()->listPeers($transport);

            echo "\n" . strtoupper($transport) . " peers (" . count($peers->peers) . "):\n";

            foreach ($peers->peers as $peer) {
                $shortId = substr($peer->id, 0, 8);
                echo "  {$shortId}... - {$peer->status} ({$peer->latencyMs}ms)\n";
            }
        }
    }

    public function checkNetworkHealth(): array
    {
        $healthMetrics = [];

        try {
            $status = $this->getNetworkStatus();
            $healthMetrics['overall_status'] = $status->status;
            $healthMetrics['peer_count'] = $status->peerCount;
            $healthMetrics['health_score'] = $status->healthScore;
            $healthMetrics['is_healthy'] = $status->healthScore > 0.8;

        } catch (\Exception $e) {
            $healthMetrics['error'] = $e->getMessage();
            $healthMetrics['is_healthy'] = false;
        }

        return $healthMetrics;
    }
}
```

### Digital Twin API

Privacy-preserving personal AI assistant.

```php
<?php

use AIVillage\Client\Models\DigitalTwinDataUpdate;
use AIVillage\Client\Models\DataPoint;

class DigitalTwinService
{
    public function __construct(private AIVillageClient $client) {}

    public function getProfile(): DigitalTwinProfile
    {
        $profile = $this->client->digitalTwin()->getProfile();

        echo "Model size: {$profile->modelSizeMb}MB\n";
        echo "Accuracy: " . number_format($profile->learningStats->accuracyScore, 3) . "\n";
        echo "Privacy level: {$profile->privacySettings->level}\n";

        return $profile;
    }

    public function updateInteractionData(
        float $userSatisfaction,
        float $predictionAccuracy,
        string $context
    ): void {
        $update = new DigitalTwinDataUpdate(
            dataType: 'interaction',
            dataPoints: [
                new DataPoint(
                    timestamp: new DateTimeImmutable(),
                    content: [
                        'interaction_type' => 'chat',
                        'user_satisfaction' => $userSatisfaction,
                        'context' => $context
                    ],
                    predictionAccuracy: $predictionAccuracy
                )
            ]
        );

        $this->client->digitalTwin()->updateData($update);
        echo "Digital twin updated with new interaction data\n";
    }

    public function getLearningInsights(): array
    {
        $profile = $this->getProfile();

        return [
            'model_efficiency' => $profile->modelSizeMb < 10.0 ? 'excellent' : 'good',
            'learning_progress' => $profile->learningStats->accuracyScore,
            'privacy_compliance' => $profile->privacySettings->level,
            'recommendations' => $this->generateRecommendations($profile)
        ];
    }

    private function generateRecommendations(DigitalTwinProfile $profile): array
    {
        $recommendations = [];

        if ($profile->learningStats->accuracyScore < 0.8) {
            $recommendations[] = 'Consider providing more interaction feedback to improve accuracy';
        }

        if ($profile->modelSizeMb > 15.0) {
            $recommendations[] = 'Model size is large - consider enabling compression optimizations';
        }

        return $recommendations;
    }
}
```

## Error Handling

### Exception Types

```php
<?php

use AIVillage\Client\Exceptions\{
    AIVillageException,
    RateLimitException,
    AuthenticationException,
    ValidationException,
    ServerException
};

class RobustChatService
{
    public function __construct(
        private AIVillageClient $client,
        private LoggerInterface $logger
    ) {}

    public function safeChatRequest(string $message): ?ChatResponse
    {
        try {
            $request = new ChatRequest(
                message: $message,
                mode: 'fast'
            );

            return $this->client->chat()->send($request);

        } catch (RateLimitException $e) {
            $this->logger->warning('Rate limited', [
                'retry_after' => $e->getRetryAfter()
            ]);

            // Automatic retry after rate limit period
            sleep($e->getRetryAfter());
            return $this->client->chat()->send($request);

        } catch (AuthenticationException $e) {
            $this->logger->error('Authentication failed', [
                'detail' => $e->getDetail()
            ]);
            throw $e;

        } catch (ValidationException $e) {
            $this->logger->error('Validation error', [
                'detail' => $e->getDetail(),
                'field_errors' => $e->getFieldErrors()
            ]);
            throw $e;

        } catch (ServerException $e) {
            $this->logger->error('Server error', [
                'status' => $e->getStatusCode(),
                'detail' => $e->getDetail(),
                'request_id' => $e->getRequestId()
            ]);
            throw $e;

        } catch (AIVillageException $e) {
            $this->logger->error('General API error', [
                'error_code' => $e->getErrorCode(),
                'message' => $e->getMessage()
            ]);
            throw $e;
        }
    }
}
```

### Circuit Breaker Pattern

```php
<?php

use AIVillage\Client\Resilience\CircuitBreaker;
use AIVillage\Client\Resilience\CircuitState;

class ResilientChatService
{
    private CircuitBreaker $circuitBreaker;

    public function __construct(
        private AIVillageClient $client,
        private LoggerInterface $logger
    ) {
        $this->circuitBreaker = new CircuitBreaker(
            failureThreshold: 5,
            timeout: 60, // seconds
            logger: $logger
        );
    }

    public function resilientChatRequest(
        string $message,
        string $agentPreference = 'any',
        int $maxRetries = 3
    ): ChatResponse {
        return $this->circuitBreaker->call(function () use ($message, $agentPreference, $maxRetries) {
            return $this->retryWithExponentialBackoff(
                fn() => $this->client->chat()->send(new ChatRequest(
                    message: $message,
                    agentPreference: $agentPreference,
                    mode: 'balanced'
                )),
                $maxRetries
            );
        });
    }

    private function retryWithExponentialBackoff(callable $operation, int $maxRetries): mixed
    {
        $attempt = 0;

        while ($attempt <= $maxRetries) {
            try {
                return $operation();

            } catch (ServerException|ConnectException $e) {
                $attempt++;

                if ($attempt > $maxRetries) {
                    throw $e;
                }

                // Exponential backoff with jitter
                $delay = (2 ** $attempt) + random_int(0, 1000) / 1000;
                $this->logger->info("Retrying in {$delay} seconds (attempt {$attempt})");

                usleep((int)($delay * 1_000_000)); // Convert to microseconds

            } catch (\Exception $e) {
                // Don't retry client errors
                throw $e;
            }
        }

        throw new \RuntimeException('Max retries exceeded');
    }
}
```

## Advanced Usage

### Data Transfer Objects with Validation

```php
<?php

use AIVillage\Client\Models\ChatRequest;
use AIVillage\Client\Models\UserContext;
use Respect\Validation\Validator as v;

class MobileChatRequest
{
    public function __construct(
        public readonly string $message,
        public readonly string $agentPreference = 'navigator',
        public readonly ?int $batteryLevel = null,
        public readonly ?string $networkType = null,
        public readonly ?int $dataBudgetMb = null
    ) {
        $this->validate();
    }

    private function validate(): void
    {
        if (empty(trim($this->message))) {
            throw new ValidationException('Message cannot be empty');
        }

        if ($this->batteryLevel !== null && !v::intVal()->between(0, 100)->validate($this->batteryLevel)) {
            throw new ValidationException('Battery level must be between 0 and 100');
        }

        if ($this->networkType !== null && !v::in(['wifi', 'cellular', 'ethernet'])->validate($this->networkType)) {
            throw new ValidationException('Network type must be wifi, cellular, or ethernet');
        }

        if ($this->dataBudgetMb !== null && !v::intVal()->min(1)->validate($this->dataBudgetMb)) {
            throw new ValidationException('Data budget must be a positive integer');
        }
    }

    public function toAIVillageRequest(): ChatRequest
    {
        $userContext = null;

        if ($this->batteryLevel !== null || $this->networkType !== null || $this->dataBudgetMb !== null) {
            $userContext = new UserContext(
                deviceType: 'mobile',
                batteryLevel: $this->batteryLevel,
                networkType: $this->networkType,
                dataBudgetMb: $this->dataBudgetMb
            );
        }

        return new ChatRequest(
            message: $this->message,
            agentPreference: $this->agentPreference,
            mode: 'balanced',
            userContext: $userContext
        );
    }
}

// Usage with validation
try {
    $mobileRequest = new MobileChatRequest(
        message: 'How can I optimize my app\'s battery usage?',
        batteryLevel: 45,
        networkType: 'cellular',
        dataBudgetMb: 50
    );

    $apiRequest = $mobileRequest->toAIVillageRequest();
    $response = $client->chat()->send($apiRequest);

    echo "Mobile-optimized response: {$response->response}\n";

} catch (ValidationException $e) {
    echo "Validation error: {$e->getMessage()}\n";
}
```

### Async Operations with ReactPHP

```php
<?php

use React\EventLoop\Loop;
use React\Promise\PromiseInterface;
use AIVillage\Client\Async\AsyncAIVillageClient;

class AsyncChatService
{
    public function __construct(private AsyncAIVillageClient $client) {}

    public function processMultipleQueriesAsync(array $queries): PromiseInterface
    {
        $promises = [];

        foreach ($queries as $query) {
            $request = new ChatRequest(
                message: $query,
                mode: 'fast'
            );

            $promises[] = $this->client->chat()->sendAsync($request);
        }

        return \React\Promise\all($promises)->then(function (array $responses) {
            $results = [];

            foreach ($responses as $i => $response) {
                $results[] = [
                    'query' => $queries[$i] ?? "Query {$i}",
                    'agent' => $response->agentUsed,
                    'response' => $response->response,
                    'processing_time' => $response->processingTimeMs
                ];
            }

            return $results;
        });
    }
}

// Usage with event loop
$loop = Loop::get();
$asyncClient = new AsyncAIVillageClient($config, $loop);
$asyncService = new AsyncChatService($asyncClient);

$queries = [
    'What is federated learning?',
    'How does differential privacy work?',
    'Explain edge computing benefits'
];

$asyncService->processMultipleQueriesAsync($queries)
    ->then(function (array $results) {
        foreach ($results as $result) {
            echo "Q: {$result['query']}\n";
            echo "A: {$result['response']}\n\n";
        }
    })
    ->catch(function (\Exception $e) {
        echo "Error: {$e->getMessage()}\n";
    });

$loop->run();
```

### Caching Integration

```php
<?php

use Psr\Cache\CacheItemPoolInterface;
use AIVillage\Client\Models\QueryRequest;

class CachedKnowledgeService
{
    public function __construct(
        private AIVillageClient $client,
        private CacheItemPoolInterface $cache
    ) {}

    public function cachedQuery(
        string $query,
        string $mode = 'comprehensive',
        int $cacheTtl = 3600
    ): QueryResult {
        $cacheKey = 'aivillage_query_' . md5($query . $mode);
        $cacheItem = $this->cache->getItem($cacheKey);

        if ($cacheItem->isHit()) {
            return $cacheItem->get();
        }

        // Not in cache, make API call
        $request = new QueryRequest(
            query: $query,
            mode: $mode,
            includeSources: true
        );

        $result = $this->client->rag()->processQuery($request);

        // Cache the result
        $cacheItem->set($result);
        $cacheItem->expiresAfter($cacheTtl);
        $this->cache->save($cacheItem);

        return $result;
    }

    public function warmupCache(array $commonQueries): void
    {
        foreach ($commonQueries as $query) {
            try {
                $this->cachedQuery($query);
                echo "Cached: {$query}\n";
            } catch (\Exception $e) {
                echo "Failed to cache: {$query} - {$e->getMessage()}\n";
            }
        }
    }
}
```

## Testing

### PHPUnit Testing

```php
<?php

use PHPUnit\Framework\TestCase;
use PHPUnit\Framework\MockObject\MockObject;
use AIVillage\Client\AIVillageClient;
use AIVillage\Client\Models\{ChatRequest, ChatResponse};

class ChatServiceTest extends TestCase
{
    private MockObject|AIVillageClient $mockClient;
    private ChatService $chatService;

    protected function setUp(): void
    {
        $this->mockClient = $this->createMock(AIVillageClient::class);
        $this->chatService = new ChatService(
            $this->mockClient,
            $this->createMock(\Psr\Log\LoggerInterface::class)
        );
    }

    public function testGetResearchAdviceReturnsExpectedResponse(): void
    {
        // Arrange
        $expectedResponse = new ChatResponse(
            response: 'Mocked response for testing',
            agentUsed: 'magi',
            processingTimeMs: 100,
            conversationId: 'test-conv-123',
            metadata: new ChatMetadata(
                confidence: 0.95,
                featuresUsed: ['test-feature']
            )
        );

        $this->mockClient
            ->expects($this->once())
            ->method('chat')
            ->willReturn($this->createMock(\AIVillage\Client\Api\ChatApi::class));

        $chatApi = $this->mockClient->chat();
        $chatApi
            ->expects($this->once())
            ->method('send')
            ->with($this->callback(function (ChatRequest $request) {
                return $request->message === 'Test question'
                    && $request->agentPreference === 'magi'
                    && $request->mode === 'comprehensive';
            }))
            ->willReturn($expectedResponse);

        // Act
        $result = $this->chatService->getResearchAdvice('Test question');

        // Assert
        $this->assertEquals($expectedResponse->response, $result->response);
        $this->assertEquals($expectedResponse->agentUsed, $result->agentUsed);
    }

    public function testGetResearchAdviceHandlesRateLimitException(): void
    {
        // Arrange
        $exception = new RateLimitException('Rate limited', 429, null, 60);

        $chatApi = $this->createMock(\AIVillage\Client\Api\ChatApi::class);
        $chatApi
            ->expects($this->once())
            ->method('send')
            ->willThrowException($exception);

        $this->mockClient
            ->expects($this->once())
            ->method('chat')
            ->willReturn($chatApi);

        // Act & Assert
        $this->expectException(RateLimitException::class);
        $this->expectExceptionMessage('Rate limited');

        $this->chatService->getResearchAdvice('Test question');
    }

    /**
     * @dataProvider validationDataProvider
     */
    public function testMobileRequestValidation(
        string $message,
        ?int $batteryLevel,
        ?string $networkType,
        bool $shouldThrow
    ): void {
        if ($shouldThrow) {
            $this->expectException(ValidationException::class);
        }

        $request = new MobileChatRequest(
            message: $message,
            batteryLevel: $batteryLevel,
            networkType: $networkType
        );

        if (!$shouldThrow) {
            $this->assertInstanceOf(MobileChatRequest::class, $request);
        }
    }

    public function validationDataProvider(): array
    {
        return [
            'valid_request' => ['Test message', 75, 'wifi', false],
            'empty_message' => ['', 75, 'wifi', true],
            'invalid_battery' => ['Test', 150, 'wifi', true],
            'invalid_network' => ['Test', 75, 'invalid', true],
        ];
    }
}
```

### Integration Testing

```php
<?php

use PHPUnit\Framework\TestCase;

/**
 * @group integration
 */
class AIVillageIntegrationTest extends TestCase
{
    private AIVillageClient $client;

    protected function setUp(): void
    {
        $apiKey = $_ENV['AIVILLAGE_TEST_API_KEY'] ?? null;

        if (!$apiKey) {
            $this->markTestSkipped('AIVILLAGE_TEST_API_KEY environment variable not set');
        }

        $config = new Configuration(
            baseUrl: 'https://staging-api.aivillage.io/v1',
            apiKey: $apiKey,
            timeout: 60.0
        );

        $this->client = new AIVillageClient($config);
    }

    public function testRealChatRequest(): void
    {
        $request = new ChatRequest(
            message: 'Hello, this is a test message from PHP SDK',
            mode: 'fast'
        );

        $response = $this->client->chat()->send($request);

        $this->assertNotNull($response->response);
        $this->assertNotNull($response->agentUsed);
        $this->assertGreaterThan(0, $response->processingTimeMs);
    }

    public function testHealthEndpoint(): void
    {
        $health = $this->client->health()->getStatus();

        $this->assertEquals('healthy', $health->status);
        $this->assertIsArray($health->services);
    }
}
```

### Performance Testing

```php
<?php

class PerformanceBenchmark
{
    private AIVillageClient $client;
    private array $metrics = [];

    public function __construct(AIVillageClient $client)
    {
        $this->client = $client;
    }

    public function benchmarkChatPerformance(int $requestCount = 10): array
    {
        echo "Running performance benchmark with {$requestCount} requests...\n";

        // Sequential requests
        $startTime = microtime(true);
        $successCount = 0;
        $errorCount = 0;
        $totalResponseTime = 0;

        for ($i = 0; $i < $requestCount; $i++) {
            try {
                $response = $this->client->chat()->send(new ChatRequest(
                    message: "Performance test message {$i}",
                    mode: 'fast'
                ));

                $successCount++;
                $totalResponseTime += $response->processingTimeMs;

            } catch (\Exception $e) {
                $errorCount++;
                echo "Request {$i} failed: {$e->getMessage()}\n";
            }
        }

        $sequentialTime = microtime(true) - $startTime;

        $metrics = [
            'total_requests' => $requestCount,
            'successful_requests' => $successCount,
            'failed_requests' => $errorCount,
            'sequential_time' => $sequentialTime,
            'requests_per_second' => $requestCount / $sequentialTime,
            'average_response_time' => $successCount > 0 ? $totalResponseTime / $successCount : 0,
            'success_rate' => $successCount / $requestCount
        ];

        echo "Performance Results:\n";
        echo "  Success Rate: " . number_format($metrics['success_rate'] * 100, 1) . "%\n";
        echo "  Requests/sec: " . number_format($metrics['requests_per_second'], 1) . "\n";
        echo "  Avg Response: " . number_format($metrics['average_response_time']) . "ms\n";
        echo "  Total Time: " . number_format($sequentialTime, 2) . "s\n";

        return $metrics;
    }
}

// Run benchmark
$config = new Configuration(
    baseUrl: 'https://api.aivillage.io/v1',
    apiKey: 'your-api-key'
);

$client = new AIVillageClient($config);
$benchmark = new PerformanceBenchmark($client);
$results = $benchmark->benchmarkChatPerformance(20);
```

## Deployment

### Production Environment

```php
<?php

use Monolog\Logger;
use Monolog\Handler\StreamHandler;
use Monolog\Handler\RotatingFileHandler;
use Symfony\Component\Cache\Adapter\RedisAdapter;

class ProductionAIVillageService
{
    private AIVillageClient $client;
    private Logger $logger;
    private CacheItemPoolInterface $cache;

    public function __construct()
    {
        $this->setupLogging();
        $this->setupCaching();
        $this->setupClient();
    }

    private function setupLogging(): void
    {
        $this->logger = new Logger('aivillage');

        // Production logging to files
        $this->logger->pushHandler(
            new RotatingFileHandler('/var/log/aivillage/app.log', 0, Logger::INFO)
        );

        // Error logging
        $this->logger->pushHandler(
            new StreamHandler('php://stderr', Logger::ERROR)
        );
    }

    private function setupCaching(): void
    {
        $redisUrl = $_ENV['REDIS_URL'] ?? 'redis://localhost:6379';
        $this->cache = new RedisAdapter(
            RedisAdapter::createConnection($redisUrl),
            'aivillage_cache',
            3600 // Default TTL: 1 hour
        );
    }

    private function setupClient(): void
    {
        $config = new Configuration(
            baseUrl: $_ENV['AIVILLAGE_API_URL'] ?? 'https://api.aivillage.io/v1',
            apiKey: $_ENV['AIVILLAGE_API_KEY'],
            timeout: (float)($_ENV['AIVILLAGE_TIMEOUT'] ?? 30.0),
            retries: (int)($_ENV['AIVILLAGE_RETRIES'] ?? 3),
            verifySSL: ($_ENV['AIVILLAGE_VERIFY_SSL'] ?? 'true') === 'true'
        );

        $this->client = new AIVillageClient($config, null, $this->logger, $this->cache);
    }

    public function healthCheck(): array
    {
        try {
            $start = microtime(true);
            $health = $this->client->health()->getStatus();
            $responseTime = (microtime(true) - $start) * 1000;

            return [
                'status' => 'healthy',
                'api_status' => $health->status,
                'response_time_ms' => $responseTime,
                'timestamp' => date('c')
            ];

        } catch (\Exception $e) {
            $this->logger->error('Health check failed', [
                'error' => $e->getMessage(),
                'trace' => $e->getTraceAsString()
            ]);

            return [
                'status' => 'unhealthy',
                'error' => $e->getMessage(),
                'timestamp' => date('c')
            ];
        }
    }
}
```

### Docker Support

```dockerfile
# Dockerfile
FROM php:8.1-fpm-alpine

# Install dependencies
RUN apk add --no-cache \
    curl \
    git \
    unzip \
    && docker-php-ext-install \
    pdo_mysql \
    bcmath \
    opcache

# Install Composer
COPY --from=composer:latest /usr/bin/composer /usr/bin/composer

# Copy application
WORKDIR /var/www/html
COPY . .

# Install PHP dependencies
RUN composer install --no-dev --optimize-autoloader

# Configure PHP
COPY docker/php.ini /usr/local/etc/php/php.ini
COPY docker/opcache.ini /usr/local/etc/php/conf.d/opcache.ini

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD php healthcheck.php || exit 1

EXPOSE 9000
CMD ["php-fpm"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    environment:
      - AIVILLAGE_API_URL=https://api.aivillage.io/v1
      - AIVILLAGE_API_KEY=${AIVILLAGE_API_KEY}
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
    volumes:
      - ./logs:/var/log/aivillage

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  redis_data:
```

## Troubleshooting

### Common Issues

```php
<?php

// SSL Certificate Issues (development only)
$config = new Configuration(
    baseUrl: 'https://api.aivillage.io/v1',
    apiKey: 'your-api-key',
    verifySSL: false // Don't use in production!
);

// Timeout Issues
$config = new Configuration(
    baseUrl: 'https://api.aivillage.io/v1',
    apiKey: 'your-api-key',
    timeout: 120.0 // 2 minutes
);

// Memory Issues with Large Responses
ini_set('memory_limit', '512M');

// Debug Logging
$logger = new Logger('debug');
$logger->pushHandler(new StreamHandler('php://stdout', Logger::DEBUG));

$client = new AIVillageClient($config, null, $logger);
```

## Support

- **Documentation**: [docs.aivillage.io](https://docs.aivillage.io)
- **Packagist**: [packagist.org/packages/aivillage/php-client](https://packagist.org/packages/aivillage/php-client)
- **GitHub Issues**: [github.com/DNYoussef/AIVillage/issues](https://github.com/DNYoussef/AIVillage/issues)
- **API Reference**: [docs.aivillage.io/api](https://docs.aivillage.io/api)

## License

MIT License - see [LICENSE](../../LICENSE) for details.
