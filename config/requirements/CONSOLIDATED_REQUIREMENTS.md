# AIVillage Consolidated Requirements

This document consolidates all requirement files and dependencies for the AIVillage project across all sprints.

## Table of Contents

- [Sprint 6 Infrastructure Requirements](#sprint-6-infrastructure-requirements)
- [Core Dependencies](#core-dependencies)
- [Development Dependencies](#development-dependencies)
- [Production Dependencies](#production-dependencies)
- [Security Dependencies](#security-dependencies)
- [Experimental Dependencies](#experimental-dependencies)
- [Platform-Specific Notes](#platform-specific-notes)

## Sprint 6 Infrastructure Requirements

### P2P Communication Layer
```
# Core P2P dependencies
pydantic==2.5.3
pydantic-core==2.14.6
annotated-types==0.6.0
bitarray==2.9.2
lz4==4.3.3

# Networking
aiohttp==3.9.1
asyncio-mqtt==0.16.1
websockets==12.0
```

### Resource Management System
```
# System monitoring
psutil==5.9.8

# Mobile platform support (optional)
# pyjnius==1.6.1      # Android only
# pyobjc-core==10.1   # macOS/iOS only
```

### Evolution System
```
# AI/ML components
numpy==1.24.3
torch==2.1.2+cpu
transformers==4.36.2
arxiv==2.1.0
```

### Monitoring and Metrics
```
prometheus-client==0.19.0
grafana-api==1.0.3
```

## Core Dependencies

Based on requirements.txt:
```
asyncio==3.4.3
numpy==1.24.3
torch==2.1.2+cpu
transformers==4.36.2
aiohttp==3.9.1
pydantic==2.5.3
psutil==5.9.8
click==8.1.7
python-dotenv==1.0.0
colorama==0.4.6
```

## Development Dependencies

Based on requirements-dev.txt:
```
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-timeout==2.2.0
pytest-cov==4.1.0
black==23.11.0
flake8==6.1.0
mypy==1.7.1
pre-commit==3.6.0
```

## Production Dependencies

Based on requirements-production.txt:
```
gunicorn==21.2.0
uvicorn==0.24.0
redis==5.0.1
celery==5.3.4
prometheus-client==0.19.0
sentry-sdk==1.38.0
```

## Security Dependencies

Based on requirements-security.txt:
```
cryptography==41.0.8
bcrypt==4.1.2
PyJWT==2.8.0
python-jose==3.3.0
passlib==1.7.4
```

## Experimental Dependencies

Based on requirements-experimental.txt:
```
# Advanced ML/NLP
spacy==3.7.2
sentence-transformers==2.2.2
langchain==0.0.350
openai==1.3.8

# Graph and database
networkx==3.2.1
neo4j==5.15.0
faiss-cpu==1.7.4
```

## Platform-Specific Notes

### Windows
- Some Bluetooth mesh dependencies may not be available
- Use CPU versions of PyTorch for compatibility
- WSL recommended for better Linux compatibility

### Linux
- Full P2P networking support available
- Can use GPU versions of ML libraries if hardware supports
- Better performance for distributed systems

### macOS
- Good P2P networking support
- Metal GPU acceleration available with appropriate PyTorch versions
- iOS development possible with pyobjc

### Mobile Platforms

#### Android
```
# Android-specific dependencies
pyjnius==1.6.1
kivy==2.2.0  # If GUI needed
buildozer==1.5.0  # For packaging
```

#### iOS
```
# iOS-specific dependencies
pyobjc-core==10.1
pyobjc-framework-Cocoa==10.1
```

## Installation Instructions

### Full Installation
```bash
pip install -r CONSOLIDATED_REQUIREMENTS.txt
```

### Minimal Installation (Core only)
```bash
pip install -r requirements.txt
```

### Development Setup
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Production Setup
```bash
pip install -r requirements.txt
pip install -r requirements-production.txt
pip install -r requirements-security.txt
```

### Sprint 6 Infrastructure Setup
```bash
pip install -r requirements_sprint6.txt
```

## Version Compatibility

- Python: 3.10+
- PyTorch: 2.1.2+ (CPU version recommended for compatibility)
- Node.js: 16+ (for any JavaScript components)
- Docker: 20+ (for containerized deployments)

## Dependency Management Best Practices

1. **Pin exact versions** for production deployments
2. **Use virtual environments** for all development
3. **Test on target platforms** before deployment
4. **Keep security dependencies updated** regularly
5. **Document platform-specific requirements** clearly

## Sprint Evolution

- **Sprint 1-3**: Basic AI and evolution components
- **Sprint 4-5**: Advanced evolution and optimization
- **Sprint 6**: Infrastructure, P2P networking, resource management
- **Sprint 7+**: Distributed inference and coordination

## Troubleshooting Common Issues

### Import Errors
- Ensure all dependencies are installed with correct versions
- Check platform-specific requirements
- Verify Python version compatibility

### Performance Issues
- Use GPU versions of ML libraries when available
- Consider resource-constrained configurations for mobile
- Monitor memory usage with psutil

### Network Issues
- Check firewall settings for P2P communication
- Verify network permissions on mobile platforms
- Test connectivity before deploying distributed features
