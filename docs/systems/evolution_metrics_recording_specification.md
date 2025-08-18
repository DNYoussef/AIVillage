# Evolution Metrics Recording System - Technical Specification

## Executive Summary

The AIVillage evolution system is 92% complete but lacks metrics persistence. The `evolution_metrics.py` module contains functional collection code but missing persistence methods (`start()` and `stop()` at lines 84-94 are stubs). This specification outlines a complete metrics recording architecture to capture, persist, and integrate evolution metrics with existing Prometheus infrastructure.

## Current State Analysis

### Existing Components

#### 1. Metrics Collection (`evolution_metrics.py`)
- **EvolutionMetrics dataclass**: Complete data structure with 15 metric fields
- **EvolutionMetricsCollector**: Functional collector with active tracking
- **Recording Methods**: `record_evolution_start()`, `record_evolution_completion()` fully implemented
- **Missing**: Persistence layer in `start()` and `stop()` methods

#### 2. KPI System (`kpi_evolution_engine.py`)
- **AgentKPI dataclass**: 18 performance metrics tracked
- **Performance Calculation**: `overall_performance()` weighted scoring
- **Evolution Decisions**: `should_retire()`, `should_evolve()` thresholds
- **Storage Path**: Basic JSON storage at `evolution_data/evolution_state.json`

#### 3. Prometheus Integration (`mobile_metrics.py`)
- **Full Prometheus Client**: Counter, Gauge, Histogram, Summary metrics
- **Push Gateway Support**: `push_to_gateway()` implementation
- **Export Format**: `generate_latest()` for scraping
- **Custom Metrics**: Registration system for dynamic metrics

## Data Flow Architecture

### Evolution Pipeline Flow

```
1. Evolution Trigger
   └─> NightlyEvolutionOrchestrator.evolve_agent()
       ├─> Pre-evolution KPI evaluation
       ├─> Strategy selection
       ├─> EvolutionMetricsCollector.record_evolution_start()
       │   └─> Captures: timestamp, CPU, memory, pre-KPIs
       ├─> Evolution execution
       ├─> Post-evolution KPI evaluation
       └─> EvolutionMetricsCollector.record_evolution_completion()
           └─> Calculates: duration, deltas, success, errors

2. Metrics Generation Points
   - Agent instantiation: Initial baseline metrics
   - Evolution start: Resource snapshot, pre-evolution KPIs
   - During evolution: Progress tracking, resource monitoring
   - Evolution completion: Performance delta, success metrics
   - Post-evolution: Continuous KPI monitoring

3. Current Storage
   - In-memory: metrics_history[], system_events[]
   - Temporary JSON: evolution_data/evolution_state.json
   - No persistence: Metrics lost on restart
```

## Proposed Metrics Recording Architecture

### 1. Persistence Layer

#### TimeSeries Database Integration
```python
class MetricsPersistence:
    """Handles metrics persistence to multiple backends."""

    def __init__(self, config: dict):
        self.backends = []

        # Primary: SQLite for local persistence
        self.sqlite_backend = SQLiteMetricsBackend(
            db_path="evolution_metrics.db",
            retention_days=90
        )

        # Secondary: Prometheus pushgateway
        if config.get("prometheus_gateway"):
            self.prometheus_backend = PrometheusBackend(
                gateway_url=config["prometheus_gateway"],
                job_name="evolution_metrics"
            )

        # Optional: InfluxDB for advanced analytics
        if config.get("influxdb_url"):
            self.influx_backend = InfluxDBBackend(
                url=config["influxdb_url"],
                bucket="aivillage_evolution",
                org="aivillage"
            )
```

#### Schema Design

##### SQLite Schema
```sql
-- Core metrics table
CREATE TABLE evolution_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    agent_id TEXT NOT NULL,
    evolution_type TEXT NOT NULL,
    evolution_id TEXT NOT NULL,

    -- Performance metrics
    performance_score REAL,
    improvement_delta REAL,
    quality_score REAL,

    -- Resource metrics
    memory_used_mb INTEGER,
    cpu_percent_avg REAL,
    duration_minutes REAL,

    -- Success metrics
    success BOOLEAN,
    error_count INTEGER,
    warning_count INTEGER,

    -- Metadata as JSON
    metadata TEXT,

    -- Indexes
    INDEX idx_agent_timestamp (agent_id, timestamp),
    INDEX idx_evolution_id (evolution_id),
    INDEX idx_success (success)
);

-- KPI snapshots table
CREATE TABLE kpi_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    agent_id TEXT NOT NULL,

    -- Performance KPIs
    accuracy REAL,
    response_time_ms REAL,
    throughput_tps REAL,
    success_rate REAL,

    -- Resource KPIs
    memory_usage_mb REAL,
    cpu_utilization REAL,
    energy_efficiency REAL,

    -- Learning KPIs
    adaptation_rate REAL,
    knowledge_retention REAL,
    generalization_ability REAL,

    -- Quality KPIs
    output_quality REAL,
    consistency REAL,
    reliability REAL,

    -- Meta KPIs
    overall_performance REAL,
    evaluation_count INTEGER,
    confidence_interval REAL,

    INDEX idx_agent_time (agent_id, timestamp)
);

-- Evolution strategies table
CREATE TABLE evolution_strategies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    evolution_id TEXT NOT NULL,
    strategy_name TEXT NOT NULL,
    risk_level TEXT,
    target_gain REAL,
    actual_gain REAL,
    success BOOLEAN,
    applied_at REAL,

    FOREIGN KEY (evolution_id) REFERENCES evolution_metrics(evolution_id)
);

-- System events table
CREATE TABLE system_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    event_type TEXT NOT NULL,
    data TEXT,  -- JSON

    INDEX idx_event_time (timestamp)
);
```

### 2. Enhanced EvolutionMetricsCollector Implementation

```python
class EvolutionMetricsCollector:
    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}
        self.metrics_history: list[EvolutionMetrics] = []
        self.active_collections: dict[str, dict[str, Any]] = {}

        # Initialize persistence
        self.persistence = MetricsPersistence(self.config)

        # Metrics buffer for batch writes
        self.metrics_buffer: list[EvolutionMetrics] = []
        self.buffer_size = self.config.get("buffer_size", 100)
        self.flush_interval = self.config.get("flush_interval", 60)  # seconds

        # Background thread for periodic flush
        self.flush_thread: threading.Thread | None = None
        self.running = False

    async def start(self) -> None:
        """Start metrics collection with persistence."""
        logger.info("Starting evolution metrics collector")

        # Initialize database
        await self.persistence.initialize()

        # Start background flush thread
        self.running = True
        self.flush_thread = threading.Thread(
            target=self._flush_loop,
            daemon=True
        )
        self.flush_thread.start()

        # Register Prometheus metrics
        self._register_prometheus_metrics()

        logger.info("Evolution metrics collector started with persistence")

    async def stop(self) -> None:
        """Stop metrics collection and flush remaining data."""
        logger.info("Stopping evolution metrics collector")

        # Stop background thread
        self.running = False
        if self.flush_thread:
            self.flush_thread.join(timeout=5)

        # Flush remaining metrics
        await self._flush_metrics()

        # Close persistence connections
        await self.persistence.close()

        logger.info("Evolution metrics collector stopped")

    def _flush_loop(self) -> None:
        """Background thread for periodic metric flushing."""
        while self.running:
            time.sleep(self.flush_interval)
            asyncio.run(self._flush_metrics())

    async def _flush_metrics(self) -> None:
        """Flush buffered metrics to persistence layer."""
        if not self.metrics_buffer:
            return

        # Copy buffer and clear
        metrics_to_flush = self.metrics_buffer.copy()
        self.metrics_buffer.clear()

        # Persist to all backends
        for metric in metrics_to_flush:
            await self.persistence.store_metric(metric)

        logger.debug(f"Flushed {len(metrics_to_flush)} metrics")
```

### 3. Integration Points

#### A. Nightly Evolution Orchestrator Integration
```python
class NightlyEvolutionOrchestrator:
    async def evolve_agent(self, agent: EvolvableAgent) -> bool:
        # Initialize metrics collector if not exists
        if not hasattr(self, 'metrics_collector'):
            self.metrics_collector = EvolutionMetricsCollector(self.config)
            await self.metrics_collector.start()

        # Record evolution start
        evolution_event = EvolutionEvent(
            agent_id=agent.agent_id,
            timestamp=time.time(),
            pre_evolution_kpis=agent.evaluate_kpi().to_dict()
        )
        await self.metrics_collector.record_evolution_start(evolution_event)

        # ... evolution logic ...

        # Record completion
        completion_event = EvolutionCompletionEvent(
            agent_id=agent.agent_id,
            timestamp=time.time(),
            success=success,
            post_evolution_kpis=post_kpis.to_dict(),
            duration_seconds=elapsed_time
        )
        await self.metrics_collector.record_evolution_completion(completion_event)
```

#### B. Prometheus Metrics Export
```python
# Additional Prometheus metrics for evolution
evolution_total = Counter(
    'evolution_attempts_total',
    'Total evolution attempts',
    ['agent_id', 'evolution_type', 'status']
)

evolution_duration = Histogram(
    'evolution_duration_seconds',
    'Evolution duration in seconds',
    ['agent_id', 'evolution_type'],
    buckets=[60, 300, 600, 1800, 3600, 7200]
)

evolution_improvement = Gauge(
    'evolution_performance_improvement',
    'Performance improvement from evolution',
    ['agent_id']
)

kpi_current = Gauge(
    'agent_kpi_current',
    'Current KPI value',
    ['agent_id', 'kpi_name']
)
```

#### C. Query Interface
```python
class MetricsQuery:
    """Query interface for evolution metrics."""

    async def get_agent_evolution_history(
        self,
        agent_id: str,
        start_time: float | None = None,
        end_time: float | None = None
    ) -> list[EvolutionMetrics]:
        """Get evolution history for an agent."""

    async def get_top_performing_evolutions(
        self,
        limit: int = 10,
        time_window_hours: int = 24
    ) -> list[EvolutionMetrics]:
        """Get top performing evolutions."""

    async def get_evolution_success_rate(
        self,
        agent_id: str | None = None,
        evolution_type: str | None = None
    ) -> float:
        """Calculate evolution success rate."""

    async def get_resource_usage_trends(
        self,
        agent_id: str,
        metric: str = "memory_used_mb"
    ) -> dict[str, Any]:
        """Get resource usage trends."""
```

### 4. Data Structures

#### Enhanced EvolutionMetrics
```python
@dataclass
class EvolutionMetrics:
    # Existing fields...

    # Additional tracking
    strategy_used: str | None = None
    rollback_performed: bool = False
    convergence_iterations: int = 0
    fitness_trajectory: list[float] = field(default_factory=list)

    # Resource tracking
    gpu_memory_mb: int | None = None
    network_bytes_transferred: int = 0

    # Comparison metrics
    baseline_performance: float | None = None
    relative_improvement: float | None = None

    def calculate_efficiency(self) -> float:
        """Calculate evolution efficiency score."""
        if self.duration_minutes > 0 and self.improvement_delta > 0:
            return self.improvement_delta / (self.duration_minutes * self.memory_used_mb / 1000)
        return 0.0
```

### 5. Monitoring & Alerting

#### Alert Rules
```yaml
# Prometheus alert rules
groups:
  - name: evolution_alerts
    rules:
      - alert: EvolutionFailureRate
        expr: rate(evolution_attempts_total{status="failed"}[1h]) > 0.3
        for: 10m
        annotations:
          summary: "High evolution failure rate"

      - alert: EvolutionResourceExhaustion
        expr: avg(evolution_memory_used_mb) > 8000
        for: 5m
        annotations:
          summary: "Evolution using excessive memory"

      - alert: NoEvolutionProgress
        expr: avg(evolution_performance_improvement) < 0.01
        for: 24h
        annotations:
          summary: "No performance improvement from evolutions"
```

### 6. Implementation Risks & Mitigations

#### Risks
1. **Data Volume**: Evolution metrics can grow rapidly
   - Mitigation: Implement retention policies, data aggregation

2. **Performance Impact**: Metrics collection overhead
   - Mitigation: Async writes, buffering, sampling

3. **Schema Evolution**: Metrics structure may change
   - Mitigation: Version tracking, migration scripts

4. **Integration Complexity**: Multiple systems to coordinate
   - Mitigation: Modular design, fallback mechanisms

#### Dependencies
- `prometheus_client>=0.16.0`
- `sqlite3` (standard library)
- Optional: `influxdb-client>=1.36.0`
- Optional: `psutil>=5.9.0` for system metrics

### 7. Testing Strategy

#### Unit Tests
```python
class TestEvolutionMetrics:
    def test_metrics_collection(self):
        """Test metric collection and buffering."""

    def test_persistence_backends(self):
        """Test each persistence backend."""

    def test_metric_calculations(self):
        """Test derived metric calculations."""

    async def test_concurrent_writes(self):
        """Test concurrent metric writes."""
```

#### Integration Tests
```python
class TestMetricsIntegration:
    async def test_end_to_end_flow(self):
        """Test complete evolution with metrics."""

    async def test_prometheus_export(self):
        """Test Prometheus metric export."""

    async def test_query_interface(self):
        """Test metrics query interface."""
```

### 8. Migration Plan

#### Phase 1: Core Implementation (Week 1)
1. Implement SQLite backend
2. Update EvolutionMetricsCollector start/stop
3. Add buffering and flush logic
4. Basic testing

#### Phase 2: Integration (Week 2)
1. Integrate with NightlyEvolutionOrchestrator
2. Add Prometheus metrics export
3. Implement query interface
4. Integration testing

#### Phase 3: Production Hardening (Week 3)
1. Add monitoring and alerting
2. Performance optimization
3. Documentation
4. Deployment scripts

## Conclusion

This specification provides a complete metrics recording architecture that:
- Preserves all existing functionality
- Adds robust persistence with multiple backends
- Integrates seamlessly with Prometheus monitoring
- Provides queryable historical data
- Supports real-time monitoring and alerting

The implementation focuses on minimal disruption to the existing 92% complete system while adding the critical missing persistence layer. The modular design allows for incremental rollout and testing.
