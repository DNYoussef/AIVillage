# Fog Computing Integration Testing Guide

## 🚀 Overview

This guide provides comprehensive instructions for running integration tests on the fog computing infrastructure. The integration test suite validates all components working together in realistic scenarios.

## 📋 Test Categories

### 1. Component Startup Tests
- **Purpose**: Verify each component starts correctly and initializes properly
- **Components Tested**:
  - Mobile Resource Manager
  - Fog Harvest Manager
  - Onion Router
  - Mixnet Client
  - Fog Marketplace
  - Token System
  - Hidden Service Host
  - Contribution Ledger
  - SLO Monitor
  - Chaos Testing Framework
  - Fog Coordinator

### 2. Component Interaction Tests
- **Purpose**: Test integration between different components
- **Key Interactions**:
  - Harvest Manager ↔ Mobile Resource Manager
  - Onion Router ↔ Mixnet Client
  - Marketplace ↔ Token System
  - Hidden Service ↔ Onion Router
  - Contribution Ledger ↔ Token System
  - SLO Monitor ↔ All Components

### 3. End-to-End Workflow Tests
- **Purpose**: Validate complete user workflows
- **Workflows Tested**:
  - Complete Fog Compute Workflow (mobile → harvest → rewards)
  - Hidden Service Hosting Workflow
  - Anonymous Communication Workflow
  - Contribution Tracking and Rewards Workflow
  - DAO Governance Workflow

### 4. Performance Tests
- **Purpose**: Measure system performance under various loads
- **Tests Include**:
  - Compute Harvesting Performance
  - Onion Routing Latency
  - Marketplace Scalability
  - Token Transaction Throughput

### 5. Security Tests
- **Purpose**: Validate security and privacy features
- **Security Aspects**:
  - Privacy Preservation
  - Anonymous Communication Security
  - Token System Security
  - Access Control Validation

### 6. Resilience Tests
- **Purpose**: Test system fault tolerance and recovery
- **Resilience Scenarios**:
  - Component Failure Recovery
  - Network Partition Tolerance
  - SLO Breach Recovery

### 7. Scalability Tests
- **Purpose**: Test system scaling characteristics
- **Scaling Scenarios**:
  - Node Scaling (adding compute nodes)
  - Traffic Scaling (increased request volume)

## 🏃‍♂️ Running Integration Tests

### Quick Start

```bash
# Run complete integration test suite
python run_fog_integration_tests.py
```

### Programmatic Usage

```python
from infrastructure.fog.integration.integration_test_suite import run_fog_integration_tests

# Run tests programmatically
test_suite = await run_fog_integration_tests()
print(f"Success rate: {test_suite.passed_tests / test_suite.total_tests * 100:.1f}%")
```

### Advanced Configuration

```python
from infrastructure.fog.integration.integration_test_suite import FogIntegrationTester

# Custom test configuration
tester = FogIntegrationTester(test_data_dir="custom_test_data")
tester.timeout_seconds = 600  # 10 minute timeout
tester.cleanup_after_tests = False  # Keep test data

# Run specific test categories
test_suite = await tester.run_complete_integration_test_suite()
```

## 📊 Understanding Test Results

### Test Status Values
- **PASSED**: Test completed successfully
- **FAILED**: Test failed due to errors
- **SKIPPED**: Test was skipped (missing dependencies)
- **TIMEOUT**: Test exceeded timeout limit

### Success Rate Interpretation
- **90%+**: ✅ Excellent - Ready for production
- **75-89%**: 🟡 Good - Minor issues to address
- **50-74%**: 🟠 Needs Improvement - Significant issues
- **<50%**: 🔴 Critical - Major problems

### Key Metrics
- **Overall Success Rate**: Percentage of tests that passed
- **Category Breakdown**: Success rate by test category
- **Duration Metrics**: Test execution times
- **Component Health**: Individual component status
- **Critical Failures**: Number of critical test failures

## 🐛 Troubleshooting Common Issues

### Missing Dependencies
**Error**: "Required components not available"
**Solution**: Ensure all fog components are properly installed and importable

### Timeout Errors
**Error**: "Test timed out after X seconds"
**Solutions**:
- Increase timeout: `tester.timeout_seconds = 600`
- Check system performance
- Review test requirements

### Import Errors
**Error**: "ModuleNotFoundError: No module named 'infrastructure.fog...'"
**Solutions**:
- Check PYTHONPATH includes project root
- Verify all component files exist
- Install required dependencies

### Component Startup Failures
**Error**: Component fails to start during tests
**Solutions**:
- Check component logs for specific errors
- Verify system resources available
- Review component configuration

## 📈 Performance Expectations

### Baseline Performance Targets
- **Component Startup**: <5 seconds per component
- **Circuit Building**: <2 seconds for 3-hop circuit
- **Service Registration**: <1 second per service
- **Token Transactions**: >10 transactions/second
- **Marketplace Operations**: >5 operations/second

### Memory Usage
- **Total Suite**: ~500MB RAM
- **Individual Components**: ~50-100MB each
- **Peak Usage**: During scalability tests

### Network Requirements
- **Bandwidth**: Minimal (tests use mocked networking)
- **Latency**: Local operations only
- **Connections**: Tests create mock network connections

## 🔧 Customizing Tests

### Adding New Test Categories
```python
async def _run_custom_tests(self):
    """Add custom test category."""
    await self._run_test(
        "custom_test_id",
        "Custom Test Name",
        TestCategory.CUSTOM,  # Add to enum
        self._test_custom_functionality
    )

async def _test_custom_functionality(self, result: IntegrationTestResult):
    """Implement custom test logic."""
    # Test implementation
    result.assertions_passed += 1
    result.logs.append("Custom test completed")
```

### Modifying Test Configuration
```python
# Adjust timeouts
tester.timeout_seconds = 300

# Disable cleanup
tester.cleanup_after_tests = False

# Custom test data directory
tester = FogIntegrationTester(test_data_dir="my_tests")
```

### Adding Custom Metrics
```python
async def _test_with_custom_metrics(self, result: IntegrationTestResult):
    # Collect custom metrics
    result.metrics["custom_latency"] = measure_latency()
    result.metrics["custom_throughput"] = measure_throughput()
    result.logs.append(f"Custom metric: {result.metrics['custom_latency']}")
```

## 🔍 Test Data and Artifacts

### Generated Files
- `integration_test_results_<suite_id>.json` - Detailed test results
- `fog_integration_tests.log` - Execution logs
- Component data directories (if cleanup disabled)

### Test Result Format
```json
{
  "suite_id": "fog_integration_1234567890",
  "start_time": "2025-01-27T10:00:00",
  "total_tests": 45,
  "passed_tests": 42,
  "failed_tests": 2,
  "skipped_tests": 1,
  "overall_metrics": {
    "category_breakdown": {...},
    "total_duration": 120.5,
    "avg_test_duration": 2.68
  },
  "test_results": [...]
}
```

## 🚨 Critical Test Failures

### Must-Pass Tests for Production
1. **Component Startup Tests**: All components must start
2. **Complete Fog Compute Workflow**: End-to-end functionality
3. **Privacy Preservation**: Security requirements
4. **Token System Security**: Financial integrity
5. **SLO Monitor Integration**: Operational monitoring

### Recommended Pass Rate by Category
- **Component Startup**: 100%
- **Component Interaction**: 95%+
- **End-to-End Workflows**: 90%+
- **Performance**: 80%+
- **Security**: 95%+
- **Resilience**: 70%+
- **Scalability**: 70%+

## 📞 Support and Resources

### Getting Help
- Check logs in `fog_integration_tests.log`
- Review failed test error messages
- Examine component-specific logs
- Verify system requirements met

### Additional Documentation
- Component-specific documentation in `docs/fog/components/`
- Architecture overview in `docs/fog/ARCHITECTURE.md`
- Deployment guide in `docs/fog/DEPLOYMENT.md`

### Reporting Issues
When reporting test failures, include:
- Complete error messages
- Test suite ID and results file
- System configuration details
- Steps to reproduce

## 🔄 Continuous Integration

### CI/CD Integration
```yaml
# GitHub Actions example
- name: Run Fog Integration Tests
  run: |
    python run_fog_integration_tests.py
    if [ $? -ne 0 ]; then
      echo "Integration tests failed"
      exit 1
    fi
```

### Pre-deployment Validation
Run integration tests before any production deployment:

```bash
# Validate before deployment
python run_fog_integration_tests.py
if [ $? -eq 0 ]; then
  echo "✅ Ready for deployment"
else
  echo "❌ Fix issues before deploying"
  exit 1
fi
```

## 📊 Monitoring Integration Test Health

### Regular Test Execution
- **Development**: Run on every code change
- **Staging**: Run daily and before releases
- **Production**: Run weekly for regression testing

### Test Result Tracking
- Track success rates over time
- Monitor test duration trends
- Alert on critical test failures
- Review skipped tests regularly

---

**Ready to validate your fog computing infrastructure? Run the integration tests and ensure your system is production-ready!** 🚀
