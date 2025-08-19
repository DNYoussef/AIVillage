# AIVillage Fog Marketplace User Guide

## Overview

The AIVillage Fog Marketplace is a minimal viable marketplace for renting distributed computing resources across the fog network. It enables dynamic pricing, trust-based matching, and cost-effective job execution across edge devices and cloud resources.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Key Concepts](#key-concepts)
3. [Getting Price Quotes](#getting-price-quotes)
4. [Submitting Bids](#submitting-bids)
5. [Job Execution with Budget Constraints](#job-execution-with-budget-constraints)
6. [Understanding Pricing](#understanding-pricing)
7. [Trust and Security](#trust-and-security)
8. [Monitoring and Billing](#monitoring-and-billing)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

## Quick Start

### Basic Example: Running a Job with Budget Constraints

```python
from packages.fog.sdk.python.fog_client import FogClient

async def run_my_job():
    async with FogClient(
        base_url="https://gateway.aivillage.ai",
        api_key="your-api-key",
        namespace="myorg/team"
    ) as client:

        # Run job with $0.50 budget
        result = await client.run_job_with_budget(
            image="myapp:latest",
            max_price=0.50,
            args=["python", "main.py"],
            resources={"cpu_cores": 2.0, "memory_gb": 4.0},
            bid_type="spot"
        )

        print(f"Job completed with exit code: {result.exit_code}")
        print(f"Total cost: ${result.actual_cost:.4f}")
```

### Get Price Quote Before Submitting

```python
# Get price estimate first
quote = await client.get_price_quote(
    cpu_cores=2.0,
    memory_gb=4.0,
    estimated_duration_hours=1.5,
    bid_type="spot",
    pricing_tier="basic"
)

if quote.available:
    print(f"Price range: ${quote.min_price:.4f} - ${quote.max_price:.4f}")
    print(f"Suggested max price: ${quote.suggested_max_price:.4f}")
else:
    print(f"No resources available: {quote.reason}")
```

## Key Concepts

### Bid Types

- **Spot**: Dynamic pricing based on supply/demand. Jobs can be preempted if higher bids arrive. Typically 20-50% cheaper than on-demand.
- **On-Demand**: Fixed pricing with guaranteed execution. No preemption risk.
- **Reserved**: Pre-purchased capacity at discounted rates (future feature).

### Pricing Tiers

- **Basic (B-class)**: Best effort execution, lowest cost
- **Standard (A-class)**: Replicated execution for reliability, 50% premium
- **Premium (S-class)**: Replicated + cryptographically attested, 100% premium

### Trust Scores

- Range from 0.0 (untrusted) to 1.0 (fully trusted)
- Based on historical performance, security features, and reputation
- Higher trust nodes typically charge premium rates
- You can specify minimum trust requirements for your jobs

## Getting Price Quotes

### Basic Price Quote

```python
quote = await client.get_price_quote(
    cpu_cores=1.0,
    memory_gb=2.0,
    estimated_duration_hours=0.5,
    bid_type="spot"
)

print(f"Available: {quote.available}")
print(f"Average price: ${quote.avg_price:.4f}")
print(f"Available providers: {quote.available_providers}")
```

### Advanced Price Quote with Constraints

```python
quote = await client.get_price_quote(
    cpu_cores=4.0,
    memory_gb=8.0,
    disk_gb=20.0,
    estimated_duration_hours=2.0,
    bid_type="on_demand",
    pricing_tier="standard",
    min_trust_score=0.7,
    max_latency_ms=200.0
)

if quote.available:
    print(f"Price: ${quote.market_estimate:.4f}")
    print(f"Volatility: {quote.price_volatility:.2%}")
    print(f"Wait time: {quote.estimated_wait_time_minutes} minutes")
```

### Understanding Quote Response

```python
# Price information
quote.min_price          # Cheapest available option
quote.avg_price          # Average market price
quote.max_price          # Most expensive option
quote.market_estimate    # Current market rate estimate

# Market conditions
quote.current_spot_rate       # Current spot price per CPU-hour
quote.current_on_demand_rate  # Current on-demand price per CPU-hour
quote.price_volatility        # Price volatility (0.0-1.0)
quote.available_providers     # Number of matching providers

# Recommendations
quote.suggested_max_price           # Recommended bid amount
quote.estimated_wait_time_minutes   # Expected matching time
```

## Submitting Bids

### Spot Bid Example

```python
bid_result = await client.submit_bid(
    image="data-processor:v1.2",
    cpu_cores=2.0,
    memory_gb=4.0,
    max_price=0.30,
    args=["python", "process.py", "--input", "data.csv"],
    env={"BATCH_SIZE": "1000"},
    bid_type="spot",
    pricing_tier="basic",
    min_trust_score=0.5
)

print(f"Bid submitted: {bid_result['bid_id']}")
print(f"Status: {bid_result['status']}")
```

### Monitoring Bid Status

```python
bid_id = bid_result["bid_id"]

# Check bid status
status = await client.get_bid_status(bid_id)
print(f"Bid status: {status['status']}")

if status['status'] == 'matched':
    print(f"Matched with job: {status['job_id']}")
    print(f"Final cost: ${status['actual_cost']:.4f}")
elif status['status'] == 'failed':
    print(f"Bid failed: {status['message']}")
```

### Canceling Bids

```python
# Cancel pending bid
result = await client.cancel_bid(bid_id)
print(f"Cancellation: {result['message']}")
```

## Job Execution with Budget Constraints

### Simple Budget-Constrained Execution

```python
# Submit job that won't exceed $1.00
result = await client.run_job_with_budget(
    image="ml-training:latest",
    max_price=1.00,
    args=["python", "train.py", "--epochs", "10"],
    resources={
        "cpu_cores": 4.0,
        "memory_gb": 8.0
    },
    bid_type="spot",
    timeout=1800  # 30 minute timeout
)

print(f"Training completed in {result.duration_ms/1000:.1f} seconds")
print(f"Used {result.cpu_seconds_used/3600:.2f} CPU-hours")
print(f"Total cost: ${result.actual_cost:.4f}")
```

### Cost Estimation Before Execution

```python
# Estimate cost first
estimate = await client.estimate_job_cost(
    image="ml-training:latest",
    cpu_cores=4.0,
    memory_gb=8.0,
    estimated_duration_hours=0.5,
    bid_type="spot"
)

print(f"Estimated cost: ${estimate.estimated_cost:.4f}")
print(f"Confidence: {estimate.confidence_level:.1%}")
print(f"Cost breakdown:")
for resource, cost in estimate.cost_breakdown.items():
    print(f"  {resource}: ${cost:.4f}")

# Show recommendations
for rec in estimate.recommendations:
    print(f"💡 {rec}")
```

### On-Demand for Critical Jobs

```python
# Use on-demand for mission-critical jobs
result = await client.run_job_with_budget(
    image="critical-app:latest",
    max_price=2.00,
    bid_type="on_demand",  # Guaranteed execution
    pricing_tier="standard",  # Replicated for reliability
    args=["./critical_process"],
    resources={"cpu_cores": 2.0, "memory_gb": 4.0}
)
```

## Understanding Pricing

### Dynamic Spot Pricing

Spot prices change based on:
- **Supply and demand**: More demand = higher prices
- **Utilization rates**: High utilization = premium pricing
- **Time of day**: Peak hours typically more expensive
- **Geographic region**: Different regions have different costs

### Pricing Components

```python
# Get current market prices
prices = await client.get_marketplace_prices()

print(f"Current spot rate: ${prices.spot_price_per_cpu_hour:.4f}/cpu-hour")
print(f"Current on-demand: ${prices.on_demand_price_per_cpu_hour:.4f}/cpu-hour")
print(f"Market volatility: {prices.price_volatility:.2%}")
print(f"Utilization rate: {prices.utilization_rate:.1%}")
```

### Trust-Based Pricing

Higher trust nodes charge premium rates:
- Trust score 0.9-1.0: Up to 50% premium
- Trust score 0.7-0.9: 20-30% premium
- Trust score 0.5-0.7: Standard rates
- Trust score 0.0-0.5: Potential discounts

### Device-Specific Pricing

Different device types have different cost profiles:
- **Servers**: Lowest cost, highest reliability
- **Desktops**: Standard pricing
- **Laptops**: 10% premium for portability
- **Tablets**: 30% premium for limited resources
- **Mobile phones**: 50% premium, limited job duration

## Trust and Security

### Choosing Trust Levels

```python
# High-trust for sensitive workloads
quote = await client.get_price_quote(
    cpu_cores=2.0,
    memory_gb=4.0,
    estimated_duration_hours=1.0,
    min_trust_score=0.8,  # High trust requirement
    pricing_tier="premium"  # Attested execution
)

# Basic trust for general workloads
quote = await client.get_price_quote(
    cpu_cores=2.0,
    memory_gb=4.0,
    estimated_duration_hours=1.0,
    min_trust_score=0.3,  # Basic trust
    pricing_tier="basic"
)
```

### Security Considerations

- **Sensitive data**: Use premium tier with high trust scores (0.8+)
- **Public computations**: Basic tier with standard trust (0.5+) is sufficient
- **Crypto/ML training**: Standard tier recommended for balance of cost and reliability

## Monitoring and Billing

### Usage Tracking

```python
# Get usage for your namespace
usage = await client.get_usage(
    namespace="myorg/team",
    period="day"  # "hour", "day", "week", "month"
)

for report in usage:
    print(f"Namespace: {report.namespace}")
    print(f"Total cost: ${report.total_cost:.2f}")
    print(f"CPU hours: {report.cpu_seconds/3600:.1f}")
    print(f"Job executions: {report.job_executions}")
```

### Cost Optimization

```python
# Monitor your quotas
quotas = await client.get_quotas(namespace="myorg/team")

for quota in quotas:
    usage_pct = (quota["used"] / quota["limit"]) * 100
    print(f"{quota['resource']}: {usage_pct:.1f}% used")

    if usage_pct > 80:
        print(f"⚠️ High usage on {quota['resource']}")
```

## Best Practices

### Cost Optimization

1. **Use spot instances** for fault-tolerant workloads
2. **Right-size resources** - don't over-provision
3. **Batch small jobs** to reduce overhead
4. **Monitor price volatility** and time your jobs accordingly
5. **Set appropriate trust levels** - higher trust = higher cost

### Reliability

1. **Use on-demand** for critical workloads
2. **Choose higher pricing tiers** for important jobs
3. **Set realistic timeouts** for job completion
4. **Implement retry logic** for spot instance preemptions

### Security

1. **Never include secrets** in job images or arguments
2. **Use premium tier** for sensitive computations
3. **Validate trust scores** match your security requirements
4. **Monitor job logs** for unexpected behavior

### Performance

1. **Get price quotes** before submitting large batches
2. **Use appropriate regions** to minimize latency
3. **Consider time zones** for cost-sensitive workloads
4. **Monitor utilization** to understand peak pricing periods

## Troubleshooting

### Common Issues

#### "No resources available"

```python
quote = await client.get_price_quote(cpu_cores=1.0, memory_gb=2.0)
if not quote.available:
    print(f"Issue: {quote.reason}")
    # Check suggestions
    if hasattr(quote, 'suggestions'):
        print("Suggestions:")
        for category, items in quote.suggestions.items():
            for item in items:
                print(f"  - {item}")
```

**Solutions:**
- Reduce resource requirements
- Increase maximum price
- Lower trust score requirements
- Try different time periods
- Use different pricing tier

#### "Bid not matched within timeout"

```python
try:
    result = await client.run_job_with_budget(
        image="app:latest",
        max_price=0.10,  # Maybe too low
        timeout=60
    )
except FogClientError as e:
    if "not matched" in str(e):
        print("Try increasing max_price or timeout")
```

**Solutions:**
- Increase `max_price` parameter
- Increase `timeout` value
- Use `on_demand` instead of `spot`
- Reduce resource requirements

#### High costs

**Check market conditions:**
```python
prices = await client.get_marketplace_prices()
if prices.price_volatility > 0.3:
    print("High volatility - consider waiting or using on-demand")
if prices.utilization_rate > 0.8:
    print("High utilization - expect premium pricing")
```

**Solutions:**
- Wait for off-peak hours
- Use smaller resource allocations
- Switch to spot bidding
- Consider lower trust requirements

### Getting Help

1. **Check marketplace status**:
   ```python
   status = await client.get_marketplace_status()
   print(f"Available providers: {status['active_listings']}")
   print(f"Market health: {status['liquidity_score']}")
   ```

2. **Review pricing recommendations**:
   ```python
   quote = await client.get_price_quote(cpu_cores=2.0, memory_gb=4.0)
   for rec in quote.recommendations:
       print(f"💡 {rec}")
   ```

3. **Monitor bid status closely**:
   ```python
   status = await client.get_bid_status(bid_id)
   if status['status'] == 'failed':
       print(f"Failure reason: {status.get('message', 'Unknown')}")
   ```

## API Reference

### FogClient Methods

| Method | Description |
|--------|-------------|
| `get_price_quote()` | Get price estimate for resource requirements |
| `submit_bid()` | Submit marketplace bid for resources |
| `get_bid_status()` | Check status of submitted bid |
| `cancel_bid()` | Cancel pending bid |
| `run_job_with_budget()` | Execute job with budget constraints |
| `estimate_job_cost()` | Estimate job execution cost |
| `get_marketplace_prices()` | Get current market pricing |
| `get_marketplace_status()` | Get marketplace health metrics |

### Response Models

- **PriceQuote**: Price estimation response
- **MarketplacePrices**: Current market pricing information
- **CostEstimate**: Detailed cost breakdown and recommendations
- **JobResult**: Job execution results with costs

For complete API documentation, see the [API Reference](../api/fog_marketplace_api.md).
