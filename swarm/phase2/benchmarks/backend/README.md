# Backend Performance Benchmark Suite

Comprehensive performance benchmarking system for comparing monolithic vs microservices backend architectures in the AI Village project. Validates that refactoring maintains performance characteristics while improving maintainability and scalability.

## 🎯 Objectives

- **No Performance Regression**: Ensure microservices architecture doesn't degrade performance
- **Improved Resource Utilization**: Validate better memory and CPU efficiency  
- **Enhanced Scalability**: Confirm improved concurrent request handling
- **Maintainability Gains**: Verify architectural benefits without performance cost

## 📊 Benchmark Coverage

### 1. Training Throughput
- **Metric**: Models processed per second
- **Focus**: ML training pipeline efficiency
- **Validation**: ±5% throughput tolerance

### 2. WebSocket Latency  
- **Metric**: Message round-trip time (RTT)
- **Focus**: Real-time communication performance
- **Validation**: P99 latency under acceptable thresholds

### 3. API Response Times
- **Metric**: HTTP request/response latency
- **Focus**: REST API performance across endpoints
- **Validation**: P95 latency targets maintained

### 4. Memory Usage Patterns
- **Metric**: Peak and average memory consumption
- **Focus**: Memory efficiency and leak detection
- **Validation**: Memory footprint improvements

### 5. Concurrent Request Handling
- **Metric**: Requests per second under load
- **Focus**: Scalability and throughput under stress
- **Validation**: Improved concurrent capacity

## 🏗️ Architecture

```
swarm/phase2/benchmarks/backend/
├── core/                          # Core benchmarking framework
│   └── performance_benchmarker.py # Main benchmarker class
├── suite/                         # Benchmark execution suite
│   └── benchmark_suite.py         # Orchestrates all benchmarks
├── tools/                         # Analysis and profiling tools
│   ├── memory_profiler.py         # Memory usage analysis
│   ├── visualization.py           # Charts and dashboards
│   ├── regression_detector.py     # Performance regression detection
│   └── config_manager.py          # Configuration management
├── config/                        # Benchmark configurations
├── results/                       # Benchmark results and data
├── reports/                       # Generated visualizations
└── run_benchmarks.py             # Main execution entry point
```

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure both architectures are running
# Monolithic: http://localhost:8080
# Microservices: http://localhost:8081
```

### Basic Usage

```bash
# Run complete benchmark suite
python run_benchmarks.py

# High-throughput optimized benchmarks
python run_benchmarks.py --workload high_throughput

# Low-latency optimized benchmarks  
python run_benchmarks.py --workload low_latency

# Memory-constrained environment
python run_benchmarks.py --workload memory_constrained

# Skip memory profiling for faster execution
python run_benchmarks.py --no-memory

# Skip visualizations
python run_benchmarks.py --no-visualizations
```

### Advanced Usage

```python
from suite.benchmark_suite import BackendBenchmarkSuite
from tools.config_manager import ConfigManager

# Custom benchmark configuration
config_manager = ConfigManager()
mono_config = config_manager.get_optimized_config('monolithic', 'high_throughput')
micro_config = config_manager.get_optimized_config('microservices', 'high_throughput')

# Run benchmarks
suite = BackendBenchmarkSuite()
results = await suite.run_full_comparison(mono_config, micro_config)

# Validate performance requirements
validation = suite.validate_performance_requirements(results)
```

## 📋 Configuration Options

### Workload Profiles

- **`default`**: Balanced testing across all metrics
- **`high_throughput`**: Maximum request/model processing rates
- **`low_latency`**: Minimal response times optimization
- **`memory_constrained`**: Resource-limited environment testing
- **`stress_test`**: Maximum load and duration testing

### Environment Configurations

```json
{
  "monolithic": {
    "base_url": "http://localhost:8080",
    "websocket_url": "ws://localhost:8080/ws",
    "connection_pool_size": 20
  },
  "microservices": {
    "base_url": "http://localhost:8081", 
    "websocket_url": "ws://localhost:8082/ws",
    "connection_pool_size": 10,
    "compression": true,
    "circuit_breaker": true
  }
}
```

## 🔍 Analysis Features

### Performance Validation

- ✅ **No Regression Detection**: Automated threshold checking
- ✅ **Memory Efficiency**: Peak/average usage comparison  
- ✅ **Latency Acceptability**: P95/P99 latency targets
- ✅ **Scalability Improvement**: Concurrent handling capacity
- ✅ **Overall Quality Score**: Comprehensive performance rating

### Regression Detection

- Statistical anomaly detection using historical baselines
- Configurable severity thresholds (Warning/Critical)
- Trend analysis for sustained performance changes
- Actionable recommendations for detected issues

### Memory Profiling

- Real-time memory usage tracking during benchmarks
- Memory leak detection with growth rate analysis
- Garbage collection impact measurement
- Per-benchmark memory footprint comparison

### Visualization Suite

- Interactive performance dashboards (Plotly)
- Latency distribution heatmaps
- Memory usage trend analysis
- Resource utilization comparisons
- Comprehensive HTML reports

## 📊 Output Examples

### Console Output
```
🚀 Starting comprehensive backend performance benchmark comparison
================================================================================

📋 Generating optimized benchmark configurations...
🔧 Executing performance benchmarks...
  Running training_throughput benchmark...
  ✅ Completed - Throughput: 15.2, Avg Latency: 125.3ms
  Running websocket_latency benchmark...
  ✅ Completed - Throughput: 142.8, Avg Latency: 23.1ms

✅ Validating performance requirements...
🔍 Performing regression analysis...
📊 Generating performance reports...
📈 Creating performance visualizations...

🎯 BENCHMARK EXECUTION SUMMARY
================================================================================
⏱️  Execution Time: 245.3 seconds
🔧 Benchmarks Executed: 8
🏗️  Architectures Tested: monolithic, microservices

📊 PERFORMANCE OVERVIEW:
   Throughput Change: +8.2%
   Memory Change: -15.4%
   Latency Change: +3.1%

✅ VALIDATION RESULTS:
   Success Rate: 85.0% (4/5)

📋 KEY RECOMMENDATIONS:
   ✅ Microservices refactoring is successful - proceed with deployment
   💾 Significant memory improvement achieved
   📈 Consider: Improve websocket_latency efficiency
```

### Generated Files

- `comprehensive_results_YYYYMMDD_HHMMSS.json` - Raw benchmark data
- `validation_report_YYYYMMDD_HHMMSS.txt` - Performance validation analysis  
- `regression_report_YYYYMMDD_HHMMSS.txt` - Regression detection results
- `dashboard_YYYYMMDD_HHMMSS.html` - Interactive performance dashboard
- `latency_heatmap_YYYYMMDD_HHMMSS.png` - Latency distribution visualization
- `memory_analysis_YYYYMMDD_HHMMSS.html` - Memory usage analysis
- `execution_summary_YYYYMMDD_HHMMSS.json` - Complete execution summary

## 🎯 Validation Criteria

### Performance Regression Thresholds

| Metric | Warning | Critical |
|--------|---------|----------|
| Throughput | -5% | -15% |
| Latency | +10% | +25% |
| Memory | +20% | +50% |
| Success Rate | -2% | -5% |

### Quality Ratings

- **EXCELLENT**: All metrics improved or maintained within 5%
- **GOOD**: Minor regressions, overall positive impact
- **FAIR**: Mixed results, some optimization needed
- **POOR**: Significant regressions requiring attention

## 🔧 Extending the Framework

### Adding New Benchmarks

```python
class CustomBenchmark:
    def __init__(self, benchmarker):
        self.benchmarker = benchmarker
    
    async def custom_benchmark(self, config: Dict[str, Any]) -> Dict[str, Any]:
        # Implement benchmark logic
        return {
            'throughput': calculated_throughput,
            'latency_stats': latency_statistics,
            'success_rate': success_rate,
            'metadata': additional_data
        }

# Register with benchmarker
benchmarker.register_benchmark('custom_test', custom_bench.custom_benchmark)
```

### Custom Visualizations

```python
from tools.visualization import PerformanceVisualizer

visualizer = PerformanceVisualizer()

# Create custom chart
def create_custom_visualization(results):
    # Custom Plotly/Matplotlib implementation
    pass

# Add to visualization suite
visualizer.create_custom_chart = create_custom_visualization
```

## 📈 Integration with CI/CD

### GitHub Actions Example

```yaml
name: Performance Benchmarks
on:
  pull_request:
    paths: ['backend/**']

jobs:
  performance-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Performance Benchmarks
        run: |
          python swarm/phase2/benchmarks/backend/run_benchmarks.py \
            --workload default \
            --results-dir ./benchmark-results
      - name: Upload Results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: ./benchmark-results/
```

### Performance Gates

```python
# Example validation in CI pipeline
validation_results = suite.validate_performance_requirements(results)

if not all(validation_results.values()):
    print("❌ Performance regression detected - blocking deployment")
    sys.exit(1)
else:
    print("✅ Performance requirements met - proceeding with deployment")
```

## 🤝 Contributing

1. **New Benchmarks**: Add benchmark implementations in `tools/`
2. **Visualizations**: Extend `visualization.py` with new chart types
3. **Analysis**: Enhance regression detection in `regression_detector.py`
4. **Configurations**: Add workload profiles in `config_manager.py`

## 📝 License

Part of the AI Village project. See main repository for license details.

---

**Built for AI Village Phase 2 Backend Refactoring Validation** 🚀