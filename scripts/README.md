# Scripts Directory

This directory contains specialized scripts that are still actively used. Most common automation tasks have been consolidated into `tools/scripts/` and the main `Makefile`.

## Active Scripts

### Core Pipeline Scripts
- `compression_monitor.py` - Compression performance monitoring (also integrated into tools/scripts/run_benchmarks.py)
- `run_agent_forge.py` - Agent forge pipeline execution
- `run_full_agent_forge.py` - Full agent forge with all features
- `run_integration_pipeline.py` - Integration testing pipeline

### Benchmarking & Testing
- `check_compression_regression.py` - Compression regression detection
- `collect_baselines.py` - Baseline performance collection
- `real_world_compression_tests.py` - Real-world compression testing
- `create_integration_tests.py` - Integration test generation
- `create_production_tests.py` - Production test suite creation

### Agent & Model Management
- `agent_kpi_system.py` - Agent KPI tracking system
- `create_agent_templates.py` - Agent template generation
- `create_mobile_sdk.py` - Mobile SDK creation
- `download_models.py` - Model downloading utilities
- `enhance_compression_mobile.py` - Mobile compression optimization

### Specialized Tools
- `enforce_style_guide.py` - Style guide enforcement
- `check_quality_gates.py` - Quality gate validation
- `monitor_performance.py` - Performance monitoring
- `validate_dependencies.py` - Dependency validation

## Recommended Usage

For common tasks, use the Makefile instead:

```bash
# Instead of python scripts/setup_environment.py
make setup

# Instead of python scripts/compression_monitor.py --benchmark
make benchmark-compression

# Instead of multiple deployment scripts
make deploy-staging
```

## Archived Scripts

Many scripts have been moved to `scripts/archive/` including:
- One-off migration scripts
- Legacy setup scripts  
- Redundant dependency fix scripts
- Documentation generation scripts

See `scripts/archive/README.md` for details on archived scripts.

## Script Organization

- **Essential automation**: `tools/scripts/` (3 core scripts)
- **Specialized tools**: `scripts/` (current directory)
- **Archived/deprecated**: `scripts/archive/`
- **Main interface**: `Makefile` (simple commands for common tasks)
