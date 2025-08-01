# Sprint 6 Test Dashboard - Comprehensive Guide

The Sprint 6 Test Dashboard provides real-time monitoring, test execution, and performance tracking for all Sprint 6 infrastructure components including P2P communication, resource management, and evolution systems.

## Overview

Sprint 6 introduced critical infrastructure components that require comprehensive monitoring:
- **P2P Communication Layer**: Peer-to-peer networking, message protocols, encryption
- **Resource Management System**: Device profiling, resource monitoring, constraint management
- **Evolution Infrastructure**: Infrastructure-aware evolution, resource-constrained evolution
- **Integration Systems**: End-to-end workflow coordination

## Dashboard Components

### 1. Test Coverage Dashboard (`tests/test_coverage_dashboard.py`)

The enhanced test coverage dashboard now includes comprehensive Sprint 6 monitoring:

#### Key Features:
- **Sprint 6 Validation**: Automated execution of `validate_sprint6.py`
- **Infrastructure Tests**: P2P, resource management, and evolution system tests
- **Performance Benchmarks**: Real-time performance monitoring and regression detection
- **Coverage Analysis**: Component-specific coverage tracking with Sprint 6 focus
- **Interactive Reports**: HTML dashboard with collapsible detailed results

#### Usage:
```bash
# Run comprehensive analysis
python tests/test_coverage_dashboard.py

# Generate only Sprint 6 test results
python -c "from tests.test_coverage_dashboard import TestCoverageDashboard; d = TestCoverageDashboard(); print(d.run_sprint6_test_suite())"
```

### 2. Dashboard Runner Script (`scripts/run_sprint6_dashboard.py`)

Centralized script for running all Sprint 6 tests and monitoring:

#### Usage Examples:
```bash
# Full test suite and coverage analysis
python scripts/run_sprint6_dashboard.py --full

# Run only Sprint 6 tests
python scripts/run_sprint6_dashboard.py --tests-only

# Run only coverage analysis
python scripts/run_sprint6_dashboard.py --coverage-only

# Quick health check
python scripts/run_sprint6_dashboard.py --quick

# Skip HTML report generation
python scripts/run_sprint6_dashboard.py --full --no-reports
```

### 3. Real-time Infrastructure Monitor (`src/monitoring/sprint6_monitor.py`)

Continuous monitoring of Sprint 6 infrastructure health:

#### Features:
- Real-time health checks for all Sprint 6 components
- Performance metrics collection
- Automated alerting for threshold violations
- Integration with validation system
- Historical data tracking

#### Usage:
```bash
# Start monitoring (runs continuously)
python src/monitoring/sprint6_monitor.py

# Get current status programmatically
python -c "from src.monitoring.sprint6_monitor import Sprint6Monitor; m = Sprint6Monitor(); print(m.get_status_summary())"
```

### 4. CI/CD Integration (`.github/workflows/sprint6-tests.yml`)

Automated testing pipeline that runs on:
- Push to main/develop branches (for Sprint 6 files)
- Pull requests affecting Sprint 6 components
- Scheduled runs every 4 hours
- Manual triggers with test mode selection

#### Workflow Jobs:
1. **sprint6-validation**: Runs `validate_sprint6.py`
2. **sprint6-infrastructure**: Matrix of infrastructure tests
3. **sprint6-performance**: Performance benchmarks
4. **sprint6-dashboard**: Generates comprehensive dashboard
5. **health-check**: Quick validation for scheduled runs

## Test Categories and Coverage

### Sprint 6 Infrastructure Tests

#### P2P Communication Layer
- **Location**: `tests/communications/test_p2p.py`, `tests/test_sprint6_infrastructure.py::TestP2PNodeIntegration`
- **Components Tested**:
  - P2P Node creation and startup
  - Peer discovery mechanisms
  - Message protocol handling
  - Encryption layer functionality
- **Coverage Target**: 85%+

#### Resource Management System
- **Location**: `tests/test_sprint6_infrastructure.py::TestDeviceProfiler`, `TestResourceMonitor`, `TestConstraintManager`, `TestAdaptiveLoader`
- **Components Tested**:
  - Device profiling and capability detection
  - Resource monitoring and trend analysis
  - Constraint management and task registration
  - Adaptive model loading strategies
- **Coverage Target**: 90%+

#### Evolution Systems
- **Location**: `tests/test_sprint6_infrastructure.py::TestInfrastructureAwareEvolution`, `TestResourceConstrainedEvolution`
- **Components Tested**:
  - Infrastructure-aware evolution configuration
  - Resource-constrained evolution policies
  - Evolution coordination protocols
  - Integration with resource management
- **Coverage Target**: 80%+

### Performance Tests

#### Location: `tests/test_sprint6_performance.py`

#### Benchmarks:
- Device profiler snapshot performance (< 100ms)
- Constraint manager task registration (< 10ms)
- P2P node startup time (< 1 second)
- Adaptive loader variant selection (< 50ms)
- High-frequency resource monitoring
- Concurrent task management
- Message throughput testing

### Integration Tests

#### End-to-End Scenarios:
- Full infrastructure stack initialization
- Resource constraint workflows
- Multi-component integration
- Complete Sprint 6 integration test

## Dashboard Features

### Real-time Status Indicators

The dashboard provides visual status indicators:
- 🟢 **Healthy**: All systems operational
- 🟡 **Warning**: Some issues detected, system functional
- 🔴 **Critical**: Significant issues requiring attention

### Interactive Elements

- **Collapsible Details**: Click to expand test output and error details
- **Auto-refresh**: Dashboard updates every 5 minutes automatically
- **Progress Bars**: Visual representation of test pass rates
- **Execution Times**: Performance tracking for all test components

### Alert System

The monitoring system provides alerts for:
- **Critical P2P Latency**: > 500ms
- **High Resource Utilization**: > 95%
- **Memory Pressure**: > 90%
- **Stale Validation**: > 2 hours old

## Configuration and Customization

### Monitoring Thresholds

Customize alert thresholds in `src/monitoring/sprint6_monitor.py`:

```python
self.thresholds = {
    "p2p_latency_warning": 100.0,  # ms
    "p2p_latency_critical": 500.0,  # ms
    "resource_utilization_warning": 80.0,  # %
    "resource_utilization_critical": 95.0,  # %
    "memory_pressure_warning": 75.0,  # %
    "memory_pressure_critical": 90.0,  # %
    "validation_age_warning": 3600,  # seconds (1 hour)
    "validation_age_critical": 7200,  # seconds (2 hours)
}
```

### Dashboard Customization

Modify critical components tracked in `tests/test_coverage_dashboard.py`:

```python
def _identify_critical_components(self) -> Dict[str, List[str]]:
    return {
        "sprint6_infrastructure": [
            "src/core/p2p/",
            "src/core/resources/",
            # Add more paths as needed
        ],
        # Add more categories
    }
```

## Integration with Existing Systems

### Monitoring Dashboard Integration

The Sprint 6 dashboard integrates with the existing monitoring infrastructure:
- Extends `src/monitoring/dashboard.py` capabilities
- Uses same styling and patterns as existing dashboards
- Shares data storage formats and conventions

### Test Infrastructure Integration

- Builds on existing test framework in `tests/`
- Uses same pytest configuration and fixtures
- Integrates with coverage reporting system
- Follows existing test organization patterns

## Troubleshooting

### Common Issues

#### Tests Not Running
```bash
# Check Python environment
python --version  # Should be 3.11+

# Install missing dependencies
pip install -r requirements.txt
pip install pytest pytest-cov pytest-benchmark pytest-asyncio

# Verify test files exist
ls tests/test_sprint6_*.py
ls validate_sprint6.py
```

#### Dashboard Not Loading
```bash
# Check for missing dependencies
pip install psutil

# Verify data directory permissions
ls -la monitoring_data/
ls -la coverage_reports/
```

#### Monitoring Failures
```bash
# Check import paths
python -c "from src.core.p2p.p2p_node import P2PNode; print('P2P OK')"
python -c "from src.core.resources.device_profiler import DeviceProfiler; print('Resources OK')"

# Run quick health check
python scripts/run_sprint6_dashboard.py --quick
```

### Performance Issues

If tests are running slowly:
1. Check system resources (CPU, memory)
2. Reduce monitoring frequency in `sprint6_monitor.py`
3. Use `--tests-only` mode to skip coverage analysis
4. Run specific test categories instead of full suite

### Coverage Issues

If coverage is lower than expected:
1. Check that all Sprint 6 components are properly imported
2. Verify test files are correctly linked to source files
3. Review the critical components configuration
4. Run with `--coverage-only` to focus on coverage analysis

## Best Practices

### For Developers

1. **Run validation before commits**:
   ```bash
   python scripts/run_sprint6_dashboard.py --quick
   ```

2. **Monitor performance impact**:
   ```bash
   python scripts/run_sprint6_dashboard.py --tests-only
   ```

3. **Keep tests focused**: Each test should validate specific Sprint 6 functionality

### For Operations

1. **Set up scheduled monitoring**:
   - Use cron or systemd to run monitoring continuously
   - Configure alerts for critical thresholds
   - Archive historical data regularly

2. **Dashboard deployment**:
   - Deploy HTML dashboards to web server
   - Set up automated refresh of reports
   - Configure access controls as needed

3. **Performance baseline tracking**:
   - Run benchmarks regularly to establish baselines
   - Track performance trends over time
   - Set up regression detection

## Future Enhancements

### Planned Features

1. **WebSocket Integration**: Real-time dashboard updates
2. **Metrics Export**: Prometheus/Grafana integration
3. **Advanced Alerting**: Slack/email notifications
4. **Historical Trends**: Long-term performance tracking
5. **Distributed Testing**: Multi-node test execution

### Extensibility

The dashboard is designed to be extensible:
- Add new test categories by extending critical components
- Implement custom alert conditions in the monitor
- Create additional dashboard views for specific use cases
- Integrate with external monitoring systems

## Conclusion

The Sprint 6 Test Dashboard provides comprehensive monitoring and testing capabilities for the critical infrastructure components introduced in Sprint 6. By consolidating test execution, performance monitoring, and health checking into a unified system, it ensures the reliability and performance of the P2P, resource management, and evolution systems that form the foundation for distributed AI agent operations.

For questions or issues, refer to the troubleshooting section or check the individual component documentation in their respective directories.