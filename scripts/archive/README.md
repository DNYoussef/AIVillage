# Archived Scripts

This directory contains scripts that have been consolidated or are no longer actively used. They are kept for reference but are not part of the main development workflow.

## Consolidated Scripts

The following functionality has been consolidated into the main tools/scripts:

- `setup_environment.py` → `tools/scripts/setup_dev_env.py`
- `compression_monitor.py` → `tools/scripts/run_benchmarks.py`
- `run_agent_forge.py` → `tools/scripts/run_benchmarks.py`
- Deployment scripts → `tools/scripts/deploy_production.py`

## Scripts moved to archive:

- One-off migration scripts
- Legacy setup scripts
- Redundant validation scripts
- Security patch scripts (integrated into CI)

## Usage

These scripts are kept for reference only. For current development workflows, use:

- `make help` - Show available commands
- `make setup` - Set up development environment
- `make benchmark` - Run benchmarks
- `make deploy-staging` - Deploy to staging
- `make deploy-prod` - Deploy to production

See the main Makefile for all available commands.