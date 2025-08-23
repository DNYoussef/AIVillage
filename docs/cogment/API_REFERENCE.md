# Cogment API Reference

## Overview

The Cogment API provides access to the unified 23.7M parameter model that replaces the HRRM 3-model approach. This reference covers all endpoints, data structures, and integration patterns for the production Cogment system.

**Key Features:**
- 🎯 **Unified Interface**: Single API replacing 3 separate HRRM APIs
- ⚡ **High Performance**: 2.8x faster response times vs HRRM
- 💾 **Memory Efficient**: 4x lower resource usage
- 🔐 **Secure**: API key authentication and rate limiting
- 📊 **Observable**: Comprehensive metrics and monitoring

## Base URL

```
Production: https://api.cogment.aivillage.dev
Staging: https://staging-api.cogment.aivillage.dev
```

## Authentication

All API requests require authentication using API keys provided in the `X-API-Key` header.

```bash
curl -H "X-API-Key: your-api-key-here" \
  https://api.cogment.aivillage.dev/health
```

### Obtaining API Keys

Contact your system administrator or use the management interface to generate API keys with appropriate permissions.

## Rate Limits

- **Default**: 100 requests per minute per API key
- **Burst**: Up to 20 requests in a short burst
- **Response Headers**: Rate limit status included in all responses

```json
{
  "X-RateLimit-Limit": "100",
  "X-RateLimit-Remaining": "95",
  "X-RateLimit-Reset": "1640995200"
}
```

## Endpoints

### Health Check

Check the health and status of the Cogment service.

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "parameter_count": 23722496,
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "memory_usage_mb": 145.2
}
```

### Text Generation

Generate text using the unified Cogment model with planning, reasoning, and memory capabilities.

```http
POST /generate
```

**Request Body:**
```json
{
  "prompt": "Explain the concept of quantum computing",
  "max_tokens": 150,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "repetition_penalty": 1.1,
  "stop_sequences": ["\n\n", "END"],
  "task_type": "explanation",
  "enable_reasoning": true,
  "enable_memory": true
}
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | **required** | Input text prompt |
| `max_tokens` | integer | 100 | Maximum tokens to generate |
| `temperature` | float | 0.7 | Sampling temperature (0.0-2.0) |
| `top_p` | float | 0.9 | Nucleus sampling threshold |
| `top_k` | integer | 50 | Top-k sampling limit |
| `repetition_penalty` | float | 1.1 | Repetition penalty (1.0-2.0) |
| `stop_sequences` | array | null | Stop generation at these sequences |
| `task_type` | string | "general" | Task type hint for optimization |
| `enable_reasoning` | boolean | true | Enable ACT reasoning |
| `enable_memory` | boolean | true | Enable LTM memory |

**Task Types:**
- `general`: General text generation
- `explanation`: Explanatory text
- `reasoning`: Mathematical/logical reasoning
- `creative`: Creative writing
- `summarization`: Text summarization
- `qa`: Question answering
- `coding`: Code generation

**Response:**
```json
{
  "generated_text": "Quantum computing is a revolutionary approach to computation that leverages the principles of quantum mechanics...",
  "tokens_used": 125,
  "inference_time_ms": 45.2,
  "reasoning_steps": 3,
  "memory_retrievals": 2,
  "model_info": {
    "model_type": "cogment_unified",
    "parameter_count": 23722496,
    "device": "cuda",
    "capabilities": ["planning", "reasoning", "memory"]
  },
  "usage": {
    "prompt_tokens": 8,
    "completion_tokens": 125,
    "total_tokens": 133
  }
}
```

### Visual Reasoning (ARC Tasks)

Process visual reasoning tasks using the integrated ARC capabilities.

```http
POST /visual-reasoning
```

**Request Body:**
```json
{
  "task": {
    "input_grid": [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
    "output_grid": [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
    "task_type": "pattern_transformation"
  },
  "examples": [
    {
      "input": [[1, 1], [0, 0]],
      "output": [[0, 0], [1, 1]]
    }
  ],
  "max_reasoning_steps": 8,
  "return_explanation": true
}
```

**Response:**
```json
{
  "predicted_output": [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
  "confidence": 0.92,
  "reasoning_steps": [
    {
      "step": 1,
      "operation": "pattern_analysis",
      "description": "Analyzing input grid pattern..."
    },
    {
      "step": 2,
      "operation": "transformation_hypothesis",
      "description": "Hypothesis: flip along diagonal"
    }
  ],
  "explanation": "The task involves flipping the grid along the main diagonal...",
  "processing_time_ms": 78.5
}
```

### Mathematical Reasoning

Solve mathematical problems with step-by-step reasoning.

```http
POST /math-reasoning
```

**Request Body:**
```json
{
  "problem": "If a train travels 240 miles in 3 hours, what is its average speed in miles per hour?",
  "problem_type": "arithmetic",
  "show_work": true,
  "verify_answer": true
}
```

**Response:**
```json
{
  "answer": "80 miles per hour",
  "solution_steps": [
    {
      "step": 1,
      "operation": "identify_given",
      "content": "Distance = 240 miles, Time = 3 hours"
    },
    {
      "step": 2,
      "operation": "apply_formula",
      "content": "Speed = Distance ÷ Time"
    },
    {
      "step": 3,
      "operation": "calculate",
      "content": "Speed = 240 ÷ 3 = 80 miles per hour"
    }
  ],
  "verification": {
    "correct": true,
    "check": "80 mph × 3 hours = 240 miles ✓"
  },
  "confidence": 0.98,
  "processing_time_ms": 52.1
}
```

### Long Context Processing

Process long documents with context-aware understanding.

```http
POST /long-context
```

**Request Body:**
```json
{
  "document": "Very long document text...",
  "query": "What are the main findings?",
  "max_context_length": 4096,
  "chunk_overlap": 128,
  "summarize_context": true
}
```

**Response:**
```json
{
  "answer": "The main findings include...",
  "relevant_chunks": [
    {
      "chunk_id": 3,
      "start_position": 1024,
      "end_position": 1536,
      "relevance_score": 0.89
    }
  ],
  "context_summary": "Document discusses research findings on...",
  "processing_time_ms": 156.3,
  "memory_utilization": 0.73
}
```

### Batch Processing

Process multiple requests in a single API call for efficiency.

```http
POST /batch
```

**Request Body:**
```json
{
  "requests": [
    {
      "id": "req_1",
      "type": "generate",
      "data": {
        "prompt": "First prompt",
        "max_tokens": 50
      }
    },
    {
      "id": "req_2",
      "type": "math-reasoning",
      "data": {
        "problem": "2 + 2 = ?",
        "problem_type": "arithmetic"
      }
    }
  ],
  "max_parallel": 4
}
```

**Response:**
```json
{
  "results": [
    {
      "id": "req_1",
      "status": "success",
      "data": {
        "generated_text": "Response to first prompt...",
        "tokens_used": 45
      }
    },
    {
      "id": "req_2",
      "status": "success",
      "data": {
        "answer": "4",
        "confidence": 1.0
      }
    }
  ],
  "total_processing_time_ms": 123.4,
  "successful_requests": 2,
  "failed_requests": 0
}
```

### Model Information

Get detailed information about the Cogment model and its capabilities.

```http
GET /model/info
```

**Response:**
```json
{
  "model_type": "cogment_unified",
  "version": "1.0.0",
  "architecture": "RefinementCore + GatedLTM + Optimized Heads",
  "parameter_count": 23722496,
  "configuration": {
    "d_model": 512,
    "n_layers": 6,
    "n_head": 8,
    "vocab_size": 13000,
    "max_seq_len": 2048,
    "memory_capacity": 1024
  },
  "capabilities": [
    "text_generation",
    "visual_reasoning",
    "mathematical_reasoning",
    "long_context_processing",
    "adaptive_computation",
    "memory_augmented_generation"
  ],
  "efficiency_metrics": {
    "vs_hrrm_parameter_reduction": "6.3x",
    "vs_hrrm_memory_improvement": "4.2x",
    "vs_hrrm_speed_improvement": "3.2x"
  },
  "supported_task_types": [
    "general", "explanation", "reasoning", "creative",
    "summarization", "qa", "coding", "visual", "mathematical"
  ]
}
```

### Performance Metrics

Get real-time performance metrics for monitoring and optimization.

```http
GET /metrics
```

**Response:**
```json
{
  "system_metrics": {
    "cpu_usage_percent": 45.2,
    "memory_usage_mb": 1456.7,
    "gpu_memory_mb": 892.3,
    "disk_usage_percent": 12.4
  },
  "model_metrics": {
    "requests_per_minute": 87,
    "average_latency_ms": 52.1,
    "cache_hit_rate": 0.73,
    "error_rate": 0.001
  },
  "performance_comparison": {
    "hrrm_baseline_latency_ms": 165,
    "cogment_current_latency_ms": 52,
    "speedup_factor": 3.17
  },
  "memory_efficiency": {
    "peak_memory_mb": 145.2,
    "hrrm_baseline_memory_mb": 612,
    "improvement_factor": 4.21
  }
}
```

## WebSocket API

For real-time streaming generation, use the WebSocket endpoint.

```javascript
// WebSocket connection
const ws = new WebSocket('wss://api.cogment.aivillage.dev/stream');

ws.onopen = function() {
  // Send generation request
  ws.send(JSON.stringify({
    type: 'generate',
    data: {
      prompt: 'Tell me a story about',
      max_tokens: 200,
      stream: true
    },
    auth: 'your-api-key'
  }));
};

ws.onmessage = function(event) {
  const response = JSON.parse(event.data);

  if (response.type === 'token') {
    // Streaming token
    console.log(response.data.token);
  } else if (response.type === 'complete') {
    // Generation complete
    console.log('Complete:', response.data);
  }
};
```

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Temperature must be between 0.0 and 2.0",
    "details": {
      "parameter": "temperature",
      "provided_value": 3.5,
      "valid_range": [0.0, 2.0]
    },
    "request_id": "req_abc123"
  }
}
```

### Error Codes

| Code | Status | Description |
|------|--------|-------------|
| `INVALID_REQUEST` | 400 | Request validation failed |
| `UNAUTHORIZED` | 401 | Invalid or missing API key |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Endpoint not found |
| `RATE_LIMITED` | 429 | Rate limit exceeded |
| `INTERNAL_ERROR` | 500 | Server error |
| `MODEL_ERROR` | 503 | Model processing error |
| `TIMEOUT` | 504 | Request timeout |

## SDK Examples

### Python SDK

```python
import requests

class CogmentClient:
    def __init__(self, api_key: str, base_url: str = "https://api.cogment.aivillage.dev"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"X-API-Key": api_key})

    def generate(self, prompt: str, **kwargs) -> dict:
        """Generate text using Cogment model."""
        response = self.session.post(
            f"{self.base_url}/generate",
            json={"prompt": prompt, **kwargs}
        )
        response.raise_for_status()
        return response.json()

    def solve_math(self, problem: str, **kwargs) -> dict:
        """Solve mathematical problems."""
        response = self.session.post(
            f"{self.base_url}/math-reasoning",
            json={"problem": problem, **kwargs}
        )
        response.raise_for_status()
        return response.json()

    def process_visual(self, grid_data: list, **kwargs) -> dict:
        """Process visual reasoning tasks."""
        response = self.session.post(
            f"{self.base_url}/visual-reasoning",
            json={"task": {"input_grid": grid_data}, **kwargs}
        )
        response.raise_for_status()
        return response.json()

# Usage example
client = CogmentClient(api_key="your-api-key")

# Text generation
result = client.generate(
    prompt="Explain machine learning in simple terms",
    max_tokens=150,
    temperature=0.7
)
print(result["generated_text"])

# Math reasoning
math_result = client.solve_math(
    problem="What is 15% of 240?",
    show_work=True
)
print(f"Answer: {math_result['answer']}")
```

### JavaScript SDK

```javascript
class CogmentClient {
  constructor(apiKey, baseUrl = 'https://api.cogment.aivillage.dev') {
    this.apiKey = apiKey;
    this.baseUrl = baseUrl;
  }

  async request(endpoint, data = null) {
    const options = {
      method: data ? 'POST' : 'GET',
      headers: {
        'X-API-Key': this.apiKey,
        'Content-Type': 'application/json'
      }
    };

    if (data) {
      options.body = JSON.stringify(data);
    }

    const response = await fetch(`${this.baseUrl}${endpoint}`, options);

    if (!response.ok) {
      const error = await response.json();
      throw new Error(`API Error: ${error.error.message}`);
    }

    return response.json();
  }

  async generate(prompt, options = {}) {
    return this.request('/generate', { prompt, ...options });
  }

  async solveMath(problem, options = {}) {
    return this.request('/math-reasoning', { problem, ...options });
  }

  async getModelInfo() {
    return this.request('/model/info');
  }
}

// Usage example
const client = new CogmentClient('your-api-key');

client.generate('Write a haiku about AI', { max_tokens: 50 })
  .then(result => console.log(result.generated_text))
  .catch(error => console.error(error));
```

### cURL Examples

```bash
# Basic text generation
curl -X POST https://api.cogment.aivillage.dev/generate \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The future of AI is",
    "max_tokens": 100,
    "temperature": 0.8
  }'

# Mathematical reasoning
curl -X POST https://api.cogment.aivillage.dev/math-reasoning \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "problem": "A circle has radius 5. What is its area?",
    "show_work": true
  }'

# Visual reasoning (ARC task)
curl -X POST https://api.cogment.aivillage.dev/visual-reasoning \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "task": {
      "input_grid": [[1, 0], [0, 1]],
      "task_type": "pattern_completion"
    },
    "return_explanation": true
  }'

# Batch processing
curl -X POST https://api.cogment.aivillage.dev/batch \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [
      {
        "id": "gen1",
        "type": "generate",
        "data": {"prompt": "Hello", "max_tokens": 10}
      },
      {
        "id": "math1",
        "type": "math-reasoning",
        "data": {"problem": "5 * 8 = ?"}
      }
    ]
  }'
```

## Performance Optimization

### Request Optimization

1. **Use Appropriate Task Types**: Specify `task_type` for optimized processing
2. **Batch Similar Requests**: Use `/batch` endpoint for multiple requests
3. **Cache Results**: Implement client-side caching for repeated requests
4. **Optimize Parameters**: Use lower `max_tokens` when possible

### Best Practices

```python
# Optimized request example
optimized_request = {
    "prompt": "Your prompt here",
    "max_tokens": 50,  # Only what you need
    "task_type": "explanation",  # Specific task type
    "enable_reasoning": True,  # Only if needed
    "enable_memory": False,  # Disable if not required
    "temperature": 0.7  # Appropriate sampling
}

# Batch multiple requests
batch_request = {
    "requests": [
        {"id": "1", "type": "generate", "data": req1_data},
        {"id": "2", "type": "generate", "data": req2_data}
    ],
    "max_parallel": 4  # Optimize for your use case
}
```

## Migration from HRRM

### API Mapping

| HRRM Endpoint | Cogment Equivalent | Notes |
|---------------|-------------------|-------|
| `/planner/generate` | `/generate` | Unified generation |
| `/reasoner/solve` | `/math-reasoning` | Enhanced reasoning |
| `/memory/retrieve` | `/generate` (with memory) | Integrated memory |
| `/planner/control` | `/generate` (with task_type) | Unified control |
| `/reasoner/chain` | `/generate` (enable_reasoning) | ACT reasoning |

### Migration Example

```python
# Old HRRM approach (3 separate calls)
planner_result = requests.post("/planner/generate", data=planner_data)
reasoner_result = requests.post("/reasoner/solve", data=reasoner_data)
memory_result = requests.post("/memory/retrieve", data=memory_data)

# New Cogment approach (single unified call)
cogment_result = requests.post("/generate", data={
    "prompt": combined_prompt,
    "task_type": "reasoning",
    "enable_reasoning": True,
    "enable_memory": True,
    "max_tokens": 200
})
```

## Support and Troubleshooting

### Common Issues

1. **Rate Limiting**: Implement exponential backoff
2. **Large Requests**: Use batch processing for multiple requests
3. **Timeouts**: Increase timeout for complex reasoning tasks
4. **Memory Issues**: Disable memory for simple tasks

### Debug Information

Enable debug mode in requests:

```json
{
  "prompt": "Your prompt",
  "debug": true,
  "include_metrics": true
}
```

Response includes debug information:
```json
{
  "generated_text": "Response...",
  "debug_info": {
    "reasoning_steps_taken": 3,
    "memory_retrievals": 1,
    "processing_breakdown": {
      "tokenization_ms": 2.1,
      "model_forward_ms": 45.2,
      "decoding_ms": 3.7
    }
  }
}
```

### Contact Support

- **API Issues**: api-support@cogment.aivillage.dev
- **Documentation**: docs@cogment.aivillage.dev
- **Emergency**: emergency@cogment.aivillage.dev

---

This API reference provides comprehensive documentation for integrating with the Cogment unified model. The single API replaces multiple HRRM endpoints while providing enhanced capabilities and improved performance.

📚 **Cogment API: Unified Intelligence, Simplified Integration**
