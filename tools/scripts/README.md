# Essential Automation Scripts

This directory contains the core automation scripts for AIVillage development and deployment.

## Core Scripts

### setup_dev_env.py
**Purpose**: Complete development environment setup and validation
**Usage**: `python tools/scripts/setup_dev_env.py [--skip-deps] [--skip-validation]`
**Features**:
- Python version validation
- Dependency installation (poetry/pip)
- Directory structure creation
- Environment file creation
- GPU/CUDA detection
- Setup validation

### run_benchmarks.py
**Purpose**: Unified benchmark execution and monitoring system
**Usage**: `python tools/scripts/run_benchmarks.py --categories [compression|agents|integration|performance|all]`
**Features**:
- Compression pipeline benchmarks
- Agent system performance tests
- Integration test suite execution
- Performance profiling
- System metrics collection
- Automated report generation

### deploy_production.py
**Purpose**: Production deployment orchestration
**Usage**: `python tools/scripts/deploy_production.py --environment [staging|production] [--action deploy|status|rollback]`
**Features**:
- Docker image building
- Kubernetes deployment
- Helm chart management
- Health checks and validation
- Automatic rollback on failure
- Deployment status monitoring

## Makefile Integration

These scripts are integrated with the main Makefile for easy access:

```bash
# Environment setup
make setup              # Run setup_dev_env.py

# Benchmarking
make benchmark          # Run all benchmarks
make benchmark-compression  # Compression benchmarks only
make benchmark-quick    # Quick benchmark suite

# Deployment
make deploy-staging     # Deploy to staging
make deploy-prod        # Deploy to production
make status             # Check deployment status
```

### Windows Usage

On Windows systems without make, use the PowerShell equivalent:

```powershell
# Environment setup
.\tools\scripts\windows_make.ps1 setup

# Benchmarking
.\tools\scripts\windows_make.ps1 benchmark
.\tools\scripts\windows_make.ps1 benchmark-compression
.\tools\scripts\windows_make.ps1 benchmark-quick

# Deployment
.\tools\scripts\windows_make.ps1 deploy-staging
.\tools\scripts\windows_make.ps1 deploy-prod
.\tools\scripts\windows_make.ps1 status

# Show all available commands
.\tools\scripts\windows_make.ps1 help
```

## Script Consolidation

The following functionality has been consolidated from the old scripts/ directory:

### Environment Setup
- `setup_environment.py` → `setup_dev_env.py`
- `validate_dependencies.py` → integrated into setup
- Various dependency fix scripts → archived

### Benchmarking
- `compression_monitor.py` → `run_benchmarks.py`
- `run_agent_forge.py` → `run_benchmarks.py`
- `real_world_compression_tests.py` → integrated
- Performance monitoring scripts → integrated

### Deployment
- `deploy/scripts/deploy.py` → `deploy_production.py`
- `deploy/scripts/health_check.py` → integrated
- `deploy/scripts/production_verification.py` → integrated
- Various deployment scripts → consolidated

## Archived Scripts

Scripts that are no longer actively used have been moved to `scripts/archive/`:
- One-off migration scripts
- Legacy setup and fix scripts
- Redundant validation scripts
- Documentation generation scripts

## Development Workflow

1. **Initial Setup**: `make setup`
2. **Development**: `make dev-up` (start services)
3. **Testing**: `make test` or `make test-fast`
4. **Quality Checks**: `make lint` and `make fmt`
5. **Benchmarking**: `make benchmark-quick` during development
6. **Pre-commit**: `make pre-commit`
7. **Deployment**: `make deploy-staging` then `make deploy-prod`

## Adding New Scripts

When adding new automation scripts:

1. Place them in `tools/scripts/`
2. Make them executable: `chmod +x script_name.py`
3. Add proper argparse interface
4. Include help text and logging
5. Add integration to Makefile if needed
6. Update this README

## Error Handling

All scripts include:
- Proper error handling and logging
- Exit codes (0 for success, non-zero for failure)
- Timeout handling for long-running operations
- Graceful cleanup on interruption