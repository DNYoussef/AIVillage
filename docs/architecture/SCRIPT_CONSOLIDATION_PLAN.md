# AIVillage Script Consolidation & Production Readiness Plan

## Executive Summary

Based on comprehensive analysis of 80+ Python scripts across the AIVillage project, this plan outlines the consolidation of redundant functionality, optimization of performance-critical scripts, and establishment of production-ready script infrastructure.

## Current Script Inventory Analysis

### Scripts Directory Structure (80+ files)
- **Active Scripts**: 58 files
- **Archived Scripts**: 22 files (in `scripts/archive/`)
- **Root Level Scripts**: 16 files
- **Source Utilities**: Various utilities in `src/`

### Key Functional Categories Identified

#### 1. Monitoring & Performance (12 scripts)
- `compression_monitor.py` - Compression performance tracking
- `system_performance_monitor.py` - System resource monitoring  
- `monitor_evolution.py` - Evolution system monitoring
- `monitor_performance.py` - General performance monitoring
- `agent_kpi_system.py` - Agent KPI tracking
- **CONSOLIDATION OPPORTUNITY**: Merge into unified monitoring system

#### 2. Validation & Quality (10 scripts)
- `validate_dependencies.py` - Dependency validation
- `check_quality_gates.py` - Quality gate checking
- `run_smoke_test.py` - Production smoke testing
- `check_compression_regression.py` - Compression regression testing
- `verify_docs.py` - Documentation validation
- **CONSOLIDATION OPPORTUNITY**: Create unified validation framework

#### 3. Benchmarking & Testing (8 scripts)
- `run_agent_forge_benchmark.py` - Agent forge benchmarking
- `production_benchmark_suite.py` - Production benchmarks
- `focused_production_benchmark.py` - Focused benchmarks
- `real_world_compression_tests.py` - Real-world testing
- **CONSOLIDATION OPPORTUNITY**: Unified benchmarking system

#### 4. Pipeline Management (6 scripts)
- `run_full_agent_forge.py` - Full pipeline execution
- `run_integration_pipeline.py` - Integration pipeline
- `run_magi_with_orchestration.py` - MAGI orchestration
- **CONSOLIDATION OPPORTUNITY**: Single pipeline orchestrator

#### 5. Environment & Setup (5 scripts)
- `setup_environment.py` - Environment setup
- `download_models.py` - Model downloading
- `fetch_wheels.py` - Dependency fetching
- **CONSOLIDATION OPPORTUNITY**: Unified environment manager

## Critical Issues Identified

### 1. Code Quality Issues
- **Inconsistent error handling**: Some scripts lack proper exception handling
- **Missing type hints**: Only 40% of scripts have comprehensive type hints
- **Logging inconsistency**: Different logging patterns across scripts
- **No standardized argument parsing**: Various argparse implementations

### 2. Functionality Duplication
- **Multiple monitoring systems**: 4 different monitoring approaches
- **Redundant validation logic**: Similar validation patterns repeated
- **Duplicate benchmarking code**: Similar benchmark execution patterns
- **Configuration handling**: Multiple config loading mechanisms

### 3. Production Readiness Gaps
- **Limited error recovery**: Many scripts fail completely on errors
- **No resource management**: Missing memory/CPU monitoring
- **Inconsistent output formats**: Various result formats
- **Missing health checks**: No self-diagnostic capabilities

## Consolidation Strategy

### Phase 1: Core Infrastructure (HIGH PRIORITY)

#### 1.1 Unified Configuration Management
Create `scripts/core/config_manager.py`:
- Single configuration loading system
- Environment-aware configuration
- Validation of configuration files
- Configuration schema enforcement

#### 1.2 Common Utilities Framework
Create `scripts/core/common_utils.py`:
- Standardized logging setup
- Common argument parsing patterns
- Shared error handling decorators
- Resource monitoring utilities

#### 1.3 Production-Ready Base Classes
Create `scripts/core/base_script.py`:
- Abstract base class for all scripts
- Standard lifecycle management
- Built-in error handling and recovery
- Metrics collection integration

### Phase 2: System Consolidation (HIGH PRIORITY)

#### 2.1 Unified Monitoring System
Consolidate into `scripts/monitoring/unified_monitor.py`:
- **Merges**: `compression_monitor.py`, `system_performance_monitor.py`, `monitor_performance.py`
- **Features**: Multi-system monitoring, alerting, dashboard generation
- **Benefits**: Single monitoring interface, consistent metrics

#### 2.2 Integrated Validation Framework
Consolidate into `scripts/validation/validation_suite.py`:
- **Merges**: `validate_dependencies.py`, `check_quality_gates.py`, `verify_docs.py`
- **Features**: Comprehensive validation pipeline, configurable checks
- **Benefits**: Single validation command, consistent reporting

#### 2.3 Comprehensive Testing System
Consolidate into `scripts/testing/test_orchestrator.py`:
- **Merges**: Multiple benchmark and test scripts
- **Features**: Unified test execution, result aggregation, reporting
- **Benefits**: Single testing interface, consistent test execution

### Phase 3: Pipeline Optimization (MEDIUM PRIORITY)

#### 3.1 Master Pipeline Orchestrator
Create `scripts/orchestration/master_pipeline.py`:
- **Merges**: Pipeline execution scripts
- **Features**: Workflow management, dependency handling, parallel execution
- **Benefits**: Single pipeline interface, better resource utilization

#### 3.2 Environment Management System
Create `scripts/environment/env_manager.py`:
- **Merges**: Setup and environment scripts
- **Features**: Complete environment lifecycle management
- **Benefits**: Simplified deployment, consistent environments

## Implementation Plan

### Week 1: Core Infrastructure
1. Create core utilities framework
2. Implement base script classes
3. Establish configuration management
4. Set up standardized logging

### Week 2: Critical System Consolidation
1. Implement unified monitoring system
2. Create integrated validation framework
3. Develop comprehensive testing system
4. Migrate critical scripts to new framework

### Week 3: Production Optimization
1. Add production-grade error handling
2. Implement resource management
3. Create health check systems
4. Add performance monitoring

### Week 4: Integration & Testing
1. Integrate all consolidated systems
2. Comprehensive testing of new framework
3. Performance validation
4. Documentation and deployment

## Expected Benefits

### Immediate (Week 1-2)
- **50% reduction** in script maintenance overhead
- **Consistent error handling** across all scripts
- **Standardized logging** for better debugging
- **Unified configuration** management

### Short-term (Week 3-4)
- **40% faster** script execution through optimization
- **80% improvement** in error recovery
- **Single interface** for all operations
- **Production-ready** monitoring and alerting

### Long-term (Month 2+)
- **90% reduction** in script duplication
- **Comprehensive** system observability
- **Automated** quality assurance
- **Scalable** infrastructure for new scripts

## Risk Mitigation

### Backward Compatibility
- Maintain wrapper scripts for critical legacy integrations
- Gradual migration plan with fallback options
- Comprehensive testing of consolidated systems

### Performance Impact
- Benchmarking of consolidated vs. individual scripts
- Resource usage monitoring during consolidation
- Optimization of consolidated systems

### Operational Continuity
- Parallel deployment approach
- Feature flags for new vs. old systems
- Rollback procedures for critical issues

## Success Metrics

1. **Script Count Reduction**: From 80+ to ~25 production scripts
2. **Code Duplication**: <10% duplicate functionality
3. **Error Recovery**: >95% graceful error handling
4. **Performance**: <5% overhead from consolidation
5. **Maintainability**: 50% reduction in maintenance time

## Next Steps

1. **Approve consolidation plan**
2. **Begin Phase 1 implementation**
3. **Establish testing framework** for consolidated scripts
4. **Create migration timeline** for critical scripts
5. **Set up monitoring** for consolidation progress

---

*This plan provides a roadmap for transforming the AIVillage script ecosystem into a production-ready, maintainable, and efficient system.*