# Requirements Organization

This folder contains the comprehensive requirements files for AIVillage project dependencies, organized by environment and use case.

## Structure

### **Current Production Requirements (Updated for Production Readiness)**
- `requirements.txt` - **Main runtime dependencies** (Web frameworks, databases, ML, P2P networking)
- `requirements-dev.txt` - **Development dependencies** (Testing, linting, documentation tools)
- `requirements-production.txt` - **Production-specific dependencies** (Monitoring, deployment, performance)
- `requirements-security.txt` - **Security dependencies** (Authentication, encryption, vulnerability scanning)

### **Legacy Requirements Files**
- `requirements_base.txt` - Original base requirements from pyproject.toml
- `requirements-test.txt` - Testing dependencies (legacy)
- `requirements-experimental.txt` - Experimental features dependencies
- `requirements_sprint6.txt` - Sprint 6 specific dependencies
- `CONSOLIDATED_REQUIREMENTS.txt` - Previous consolidation attempt
- `CONSOLIDATED_REQUIREMENTS.md` - Previous consolidation documentation

## Current Status

✅ **Production Ready**: Updated comprehensive requirements files with:
- **FastAPI & Web Framework stack** for production APIs
- **Database support** (PostgreSQL, Neo4j, Redis, Qdrant)
- **Machine Learning stack** (PyTorch, Transformers, FAISS)
- **Security hardening** (Cryptography, JWT, Passlib)
- **Monitoring & Observability** (Prometheus, OpenTelemetry)
- **P2P Communication** (WebSockets, Bluetooth adapters)

## Installation

### **Full Production Environment:**
```bash
pip install -r config/requirements/requirements.txt
pip install -r config/requirements/requirements-production.txt
pip install -r config/requirements/requirements-security.txt
```

### **Development Environment:**
```bash
pip install -r config/requirements/requirements.txt
pip install -r config/requirements/requirements-dev.txt
```

### **With Constraints (Recommended):**
```bash
pip install -r config/requirements/requirements.txt -c config/constraints.txt
```

The `config/constraints.txt` file provides version pinning for reproducible builds.
