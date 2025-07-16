import json
import os
import random
import sys
import time
import tracemalloc

from locust import HttpUser, between, events, task
from prometheus_client import CollectorRegistry, Counter, Gauge, push_to_gateway

# Prometheus metrics
PUSHGATEWAY_URL = os.getenv("PUSHGATEWAY_URL", "localhost:9091")
registry = CollectorRegistry()

memory_usage = Gauge(
    "soak_test_memory_usage_mb", "Memory usage in MB", ["service"], registry=registry
)
error_count = Counter(
    "soak_test_errors_total",
    "Total errors during soak test",
    ["service", "error_type"],
    registry=registry,
)
task_duration = Gauge(
    "soak_test_task_duration_seconds",
    "Task completion time",
    ["task_type"],
    registry=registry,
)

# Enable memory tracking
tracemalloc.start()


class AdvancedVillageUser(HttpUser):
    wait_time = between(0.5, 2.0)

    def on_start(self):
        """Initialize user session with context"""
        self.user_id = f"soak_user_{random.randint(1000, 9999)}"
        self.conversation_history = []
        self.session_id = f"session_{int(time.time())}"

    @task(60)
    def chat_simple(self):
        """Simple single-turn chat - most common"""
        prompts = [
            "What is the capital of France?",
            "Explain quantum computing in simple terms",
            "How do I make pasta carbonara?",
            "Tell me about machine learning",
            "What are the benefits of meditation?",
        ]

        payload = {
            "prompt": random.choice(prompts),
            "user_id": self.user_id,
            "session_id": self.session_id,
            "history": self.conversation_history[-5:],
        }

        start_time = time.time()
        with self.client.post(
            "/v1/chat", json=payload, catch_response=True
        ) as response:
            duration = time.time() - start_time
            task_duration.labels(task_type="simple_chat").set(duration)

            if response.status_code == 200:
                response.success()
                result = response.json()
                self.conversation_history.append(
                    {"user": payload["prompt"], "assistant": result.get("response", "")}
                )
            elif response.status_code == 429:
                response.failure("Rate limited")
                error_count.labels(service="gateway", error_type="rate_limit").inc()
            else:
                response.failure(f"Got status code {response.status_code}")
                error_count.labels(
                    service="twin", error_type=f"http_{response.status_code}"
                ).inc()

    @task(30)
    def chat_complex(self):
        """Multi-turn conversation with context"""
        initial_prompts = [
            "Create a detailed marketing plan for a new AI startup",
            "Debug this Python code: def factorial(n): return n * factorial(n-1)",
            "Write a comprehensive analysis of climate change impacts",
            "Design a REST API for a social media platform",
        ]

        payload = {
            "prompt": random.choice(initial_prompts),
            "user_id": self.user_id,
            "session_id": f"complex_{self.session_id}",
            "history": self.conversation_history[-10:],
            "temperature": 0.8,
            "max_tokens": 2000,
        }

        start_time = time.time()
        with self.client.post(
            "/v1/chat", json=payload, catch_response=True
        ) as response:
            duration = time.time() - start_time
            task_duration.labels(task_type="complex_chat").set(duration)

            if response.status_code == 200:
                response.success()
                if random.random() < 0.5:
                    self._send_follow_up()
            else:
                response.failure(f"Complex chat failed: {response.status_code}")
                error_count.labels(service="twin", error_type="complex_failure").inc()

    @task(10)
    def check_health(self):
        """Periodic health checks"""
        endpoints = ["/health", "/v1/health", "/metrics"]
        endpoint = random.choice(endpoints)

        with self.client.get(endpoint, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {endpoint}")

    def _send_follow_up(self):
        """Simulate conversation continuation"""
        follow_ups = [
            "Can you elaborate on that?",
            "What are the potential risks?",
            "How would you implement this?",
            "Can you provide a concrete example?",
        ]

        payload = {
            "prompt": random.choice(follow_ups),
            "user_id": self.user_id,
            "session_id": self.session_id,
            "history": self.conversation_history[-5:],
        }

        self.client.post("/v1/chat", json=payload)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Generate final report"""
    stats = environment.stats

    current, peak = tracemalloc.get_traced_memory()
    memory_mb = peak / 1024 / 1024

    report = {
        "test_summary": {
            "duration_seconds": stats.last_request_timestamp - stats.start_time,
            "total_requests": stats.num_requests,
            "failure_rate": stats.fail_ratio,
            "current_rps": stats.current_rps,
            "peak_memory_mb": memory_mb,
        },
        "response_times": {
            "min": stats.min_response_time,
            "max": stats.max_response_time,
            "avg": stats.avg_response_time,
            "p50": stats.get_response_time_percentile(0.5),
            "p95": stats.get_response_time_percentile(0.95),
            "p99": stats.get_response_time_percentile(0.99),
        },
        "pass_criteria": {
            "error_rate_ok": stats.fail_ratio < 0.005,
            "memory_ok": memory_mb < 500,
            "p99_ok": stats.get_response_time_percentile(0.99) < 1000,
        },
    }

    with open("soak_test_report.json", "w") as f:
        json.dump(report, f, indent=2)
    push_to_gateway(PUSHGATEWAY_URL, job="soak_test", registry=registry)

    all_passed = all(report["pass_criteria"].values())
    if all_passed:
        print("✅ SOAK TEST PASSED")
    else:
        print("❌ SOAK TEST FAILED")
        print(json.dumps(report["pass_criteria"], indent=2))
        sys.exit(1)
