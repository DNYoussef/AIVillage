# Requirements Organization

This folder contains the historical requirements files that have been consolidated into the main `requirements.txt` file in the root directory.

## Structure

- `requirements_base.txt` - Original base requirements from pyproject.toml
- `requirements-dev.txt` - Development dependencies
- `requirements-test.txt` - Testing dependencies  
- `requirements-production.txt` - Production-specific dependencies
- `requirements-security.txt` - Security-related dependencies
- `requirements-experimental.txt` - Experimental features dependencies
- `requirements_sprint6.txt` - Sprint 6 specific dependencies
- `CONSOLIDATED_REQUIREMENTS.txt` - Previous consolidation attempt
- `CONSOLIDATED_REQUIREMENTS.md` - Previous consolidation documentation

## Current Status

All requirements have been consolidated into the root `requirements.txt` file which now serves as the single source of truth for all dependencies.

The consolidated file is organized into clear sections:
- Core Dependencies
- AI/ML Dependencies  
- Visualization & Monitoring
- Security Dependencies
- P2P & Mesh Networking
- Utilities
- Development Dependencies
- Experimental Dependencies

## Installation

Use the main requirements.txt file in the root directory:

```bash
pip install -r requirements.txt
```

For specific environments, you can still reference the individual files here if needed.