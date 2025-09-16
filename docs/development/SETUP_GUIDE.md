# AIVillage Development Setup Guide

## Overview

This guide provides comprehensive setup instructions for AIVillage development based on the actual codebase structure. It covers environment setup, dependency management, and development workflows for the modular Python architecture.

## Prerequisites

### System Requirements

**Operating System**
- Linux (Ubuntu 20.04+ recommended)
- macOS (10.15+ recommended)
- Windows 10+ with WSL2

**Hardware Requirements**
- CPU: 8+ cores recommended (4 cores minimum)
- RAM: 16GB recommended (8GB minimum)
- Storage: 50GB+ available space
- GPU: CUDA-compatible GPU recommended for Agent Forge training

**Software Dependencies**
- Python 3.11+ (required for async improvements)
- Node.js 18+ (for web interface)
- Docker 20.10+ (for containerized deployment)
- Git 2.30+ (for version control)

### Development Tools

**Required**
```bash
# Python development
pip install uv  # Fast Python package manager
pip install pre-commit  # Git hooks for code quality

# Node.js development
npm install -g pnpm  # Fast package manager

# Database tools
docker pull postgres:15
docker pull redis:7
docker pull neo4j:5
```

**Optional but Recommended**
```bash
# Code quality tools
pip install black isort flake8 mypy
pip install pytest pytest-asyncio pytest-cov

# Infrastructure tools
curl -LO https://dl.k8s.io/release/v1.28.0/bin/linux/amd64/kubectl
pip install docker-compose
```

## Repository Setup

### 1. Clone the Repository

```bash
# Clone the main repository
git clone https://github.com/DNYoussef/AIVillage.git
cd AIVillage

# Set up Git configuration
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Install pre-commit hooks
pre-commit install
```

### 2. Environment Configuration

```bash
# Create and activate Python virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows

# Verify Python version
python --version  # Should show Python 3.11+
```

### 3. Dependency Installation

AIVillage uses a sophisticated dependency management system with constraint files:

```bash
# Install core dependencies
pip install -r config/requirements/requirements.txt -c config/constraints.txt

# Install development dependencies
pip install -r config/requirements/requirements-dev.txt -c config/constraints.txt

# Install production dependencies (optional)
pip install -r config/requirements/requirements-production.txt -c config/constraints.txt
```

**Dependency Files Overview:**
- `requirements.txt`: Core runtime dependencies
- `requirements-dev.txt`: Development and testing tools
- `requirements-production.txt`: Production-specific packages
- `constraints.txt`: Version constraints for reproducible builds

### 4. Environment Variables

Create environment configuration:

```bash
# Copy example environment file
cp config/example.env .env

# Edit environment variables
nano .env  # or your preferred editor
```

**Required Environment Variables:**
```bash
# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/aivillage
REDIS_URL=redis://localhost:6379/0
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Security Configuration
SECRET_KEY=your-secret-key-here
JWT_SECRET=your-jwt-secret-here
ENCRYPTION_KEY=your-encryption-key-here

# External Services
OPENAI_API_KEY=your-openai-key  # Optional for AI features
ANTHROPIC_API_KEY=your-anthropic-key  # Optional for Claude integration

# Development Settings
DEBUG=true
LOG_LEVEL=DEBUG
ENVIRONMENT=development
```

## Infrastructure Setup

### Local Development Infrastructure

Use Docker Compose for local development infrastructure:

```bash
# Start infrastructure services
docker-compose -f docker/docker-compose.dev.yml up -d

# Verify services are running
docker-compose -f docker/docker-compose.dev.yml ps

# View service logs
docker-compose -f docker/docker-compose.dev.yml logs -f
```

**Services Started:**
- PostgreSQL (port 5432)
- Redis (port 6379)
- Neo4j (port 7687, 7474)
- RabbitMQ (port 5672, 15672)
- Kafka (port 9092)

### Database Setup

```bash
# Create database schema
python scripts/setup_database.py

# Run database migrations
python scripts/migrate_database.py

# Seed development data (optional)
python scripts/seed_database.py --environment development
```

### Verify Infrastructure

```bash
# Test database connections
python scripts/test_infrastructure.py

# Expected output:
# ✅ PostgreSQL connection successful
# ✅ Redis connection successful  
# ✅ Neo4j connection successful
# ✅ RabbitMQ connection successful
# ✅ All infrastructure services ready
```

## Core System Setup

### 1. Initialize Core Components

```bash
# Initialize core business logic
python -c "from core import domain, agents, rag; print('Core modules imported successfully')"

# Initialize Agent Forge pipeline
python -c "from core.agent_forge import PhaseController; print('Agent Forge initialized')"

# Initialize HyperRAG system
python -c "from core.hyperrag import HyperRAGSystem; print('HyperRAG initialized')"
```

### 2. Start Enhanced API Gateway

```bash
# Start the enhanced unified API gateway
cd infrastructure/gateway
python enhanced_unified_api_gateway.py

# Gateway will start on http://localhost:8000
# API documentation: http://localhost:8000/docs
# Admin interface: http://localhost:8000/admin_interface.html
```

### 3. Test System Integration

```bash
# Run integration tests
python scripts/test_enhanced_fog_integration.py

# Expected output:
# ✅ TEE Runtime operational
# ✅ Cryptographic proofs functional
# ✅ Zero-knowledge predicates working
# ✅ Market pricing engine ready
# ✅ All 8 fog components operational
```

## Development Workflow

### Project Structure Navigation

Understanding the modular architecture:

```
AIVillage/
├── core/                    # Business Logic Layer
│   ├── agents/              # 54 Specialized AI Agents
│   ├── agent_forge/         # 7-Phase ML Pipeline
│   ├── domain/              # Core Entities & Services
│   ├── rag/                 # Knowledge Retrieval
│   ├── hyperrag/           # Neural Memory System
│   └── security/           # Security Domain Logic
├── infrastructure/         # Technical Infrastructure
│   ├── gateway/            # API Gateway (FastAPI)
│   ├── fog/               # Fog Computing Platform
│   ├── p2p/               # P2P Communication
│   ├── data/              # Data Persistence
│   └── messaging/         # Event-Driven Architecture
├── tools/                 # Development Tools
│   ├── development/       # Build System
│   ├── ci-cd/            # Deployment Automation
│   └── monitoring/       # Observability
└── apps/                 # Application Layer
    ├── web/              # React Admin Dashboard
    └── mobile/           # Mobile Applications
```

### Code Quality Standards

The project enforces strict code quality standards:

**Pre-commit Hooks**
```bash
# Hooks run automatically on commit:
# - black (code formatting)
# - isort (import sorting)  
# - flake8 (linting)
# - mypy (type checking)
# - pytest (unit tests)

# Run hooks manually
pre-commit run --all-files
```

**Code Formatting**
```bash
# Format code with black
black .

# Sort imports with isort
isort .

# Check with flake8
flake8 .

# Type checking with mypy
mypy core/ infrastructure/
```

### Testing Framework

Comprehensive testing strategy with >90% coverage:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=core --cov=infrastructure --cov-report=html

# Run specific test categories
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests
pytest tests/e2e/          # End-to-end tests

# Run Agent Forge tests
pytest tests/agent_forge/

# Run security tests
pytest tests/security/
```

**Test Organization:**
- `tests/unit/`: Fast unit tests for individual components
- `tests/integration/`: Integration tests for system components
- `tests/e2e/`: End-to-end tests for complete workflows
- `tests/performance/`: Performance and load tests
- `tests/security/`: Security vulnerability tests

### Development Commands

**Core Development**
```bash
# Start development server with hot reload
uvicorn infrastructure.gateway.enhanced_unified_api_gateway:app --reload --host 0.0.0.0 --port 8000

# Run agent development mode
python tools/development/agent_dev_server.py

# Start HyperRAG development environment
python tools/development/hyperrag_dev.py
```

**Agent Forge Development**
```bash
# Initialize Agent Forge development environment
python tools/development/agent_forge_dev.py

# Run Phase 1 (Cognate Pretraining) in development mode
python core/agent_forge/phases/cognate_pretrain/dev_runner.py

# Test model compression
python core/agent_forge/compression/test_compression.py

# Validate training pipeline
python core/agent_forge/core/validate_pipeline.py
```

**P2P Network Development**
```bash
# Start LibP2P development node
python infrastructure/p2p/libp2p/dev_node.py

# Test BitChat mobile bridge
python infrastructure/p2p/bitchat/test_mobile_bridge.py

# Validate BetaNet integration
python infrastructure/p2p/betanet/validate_circuits.py
```

## IDE Configuration

### VS Code Setup

Recommended VS Code extensions and settings:

**.vscode/extensions.json:**
```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.black-formatter",
    "ms-python.isort",
    "ms-python.flake8",
    "ms-python.mypy-type-checker",
    "ms-vscode.test-adapter-converter",
    "ms-vscode.live-server"
  ]
}
```

**.vscode/settings.json:**
```json
{
  "python.defaultInterpreterPath": "./.venv/bin/python",
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.linting.mypyEnabled": true,
  "python.testing.pytestEnabled": true,
  "python.testing.unittestEnabled": false,
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  }
}
```

### PyCharm Setup

**Python Interpreter Configuration:**
1. File → Settings → Project → Python Interpreter
2. Add Interpreter → Existing environment
3. Point to `.venv/bin/python`

**Code Style Configuration:**
1. File → Settings → Editor → Code Style → Python
2. Import Black code style
3. Enable "Optimize imports on the fly"

## Debugging and Troubleshooting

### Common Issues and Solutions

**1. Missing grokfast Package**
```bash
# Current known issue - package not available in PyPI
# Temporary workaround:
pip install torch>=2.0.0
# Note: Agent Forge Phase 1 may have reduced functionality
```

**2. Import Path Conflicts**
```bash
# Clear Python cache
find . -type d -name "__pycache__" -exec rm -r {} +
find . -name "*.pyc" -delete

# Reinstall in development mode
pip install -e .
```

**3. Database Connection Issues**
```bash
# Verify database services
docker-compose -f docker/docker-compose.dev.yml ps

# Reset database
docker-compose -f docker/docker-compose.dev.yml down -v
docker-compose -f docker/docker-compose.dev.yml up -d
python scripts/setup_database.py
```

**4. P2P Network Issues**
```bash
# Check firewall settings
sudo ufw allow 9090  # P2P mesh port
sudo ufw allow 8080  # API server port

# Verify LibP2P dependencies
python -c "import libp2p; print('LibP2P available')"
```

### Debug Configuration

**Python Debugging**
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use development configuration
import os
os.environ["DEBUG"] = "true"
os.environ["LOG_LEVEL"] = "DEBUG"
```

**API Gateway Debugging**
```bash
# Start with debug mode
uvicorn infrastructure.gateway.enhanced_unified_api_gateway:app --reload --log-level debug

# Enable request tracing
export TRACE_REQUESTS=true
export ENABLE_REQUEST_LOGGING=true
```

### Performance Profiling

```bash
# Profile API performance
python tools/development/profile_api.py

# Profile Agent Forge training
python tools/development/profile_training.py

# Memory profiling
pip install memory-profiler
python -m memory_profiler scripts/memory_test.py
```

## Production Deployment Preparation

### Build and Package

```bash
# Build production package
python setup.py sdist bdist_wheel

# Build Docker images
docker build -t aivillage/api-gateway:latest -f docker/Dockerfile.api-gateway .
docker build -t aivillage/agent-forge:latest -f docker/Dockerfile.agent-forge .

# Test production build
docker run -p 8000:8000 aivillage/api-gateway:latest
```

### Configuration Validation

```bash
# Validate production configuration
python scripts/validate_production_config.py

# Check security configuration
python scripts/security_audit.py

# Verify all dependencies
pip check
```

### Deployment Testing

```bash
# Run deployment tests
pytest tests/deployment/

# Load testing
python tools/testing/load_test.py

# Security testing
python tools/testing/security_test.py
```

## Contributing Guidelines

### Code Contribution Process

1. **Fork and Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Development**
   - Follow code quality standards
   - Write comprehensive tests
   - Update documentation

3. **Testing**
   ```bash
   pytest  # All tests must pass
   pre-commit run --all-files  # All checks must pass
   ```

4. **Commit and Push**
   ```bash
   git commit -m "feat: add new feature description"
   git push origin feature/your-feature-name
   ```

5. **Pull Request**
   - Use descriptive PR title
   - Include comprehensive description
   - Link related issues

### Code Review Standards

**Required Checks:**
- [ ] All tests pass
- [ ] Code coverage >90%
- [ ] No security vulnerabilities
- [ ] Documentation updated
- [ ] Performance impact assessed

### Architecture Guidelines

**Core Principles:**
- **Separation of Concerns**: Core business logic separate from infrastructure
- **Dependency Injection**: Infrastructure concerns injected into core
- **Type Safety**: Comprehensive type hints throughout
- **Error Handling**: Explicit error handling with custom exceptions
- **Performance**: Sub-100ms response times for APIs

## Support and Resources

### Documentation
- **Architecture Documentation**: `docs/architecture/`
- **API Documentation**: `http://localhost:8000/docs` (when running)
- **Agent Documentation**: `docs/agents/`
- **Security Documentation**: `docs/security/`

### Community
- **GitHub Issues**: Report bugs and request features
- **GitHub Discussions**: Community Q&A and ideas
- **Contributing Guide**: `CONTRIBUTING.md`

### Development Tools
- **Local Admin Interface**: `http://localhost:8000/admin_interface.html`
- **Monitoring Dashboard**: `http://localhost:3000` (Grafana)
- **Database Admin**: `http://localhost:8080` (pgAdmin)

## Next Steps

After completing setup:

1. **Explore the Codebase**: Start with `core/__init__.py` and `infrastructure/__init__.py`
2. **Run Example Workflows**: Execute example scripts in `examples/`
3. **Contribute**: Pick an issue from the GitHub issue tracker
4. **Join Community**: Participate in GitHub discussions

## Advanced Development Topics

### Agent Development

```bash
# Create new agent
python tools/development/create_agent.py --name YourAgent --category specialized

# Test agent integration
python tools/development/test_agent.py --agent YourAgent

# Deploy agent to swarm
python tools/development/deploy_agent.py --agent YourAgent --swarm development
```

### HyperRAG Development

```bash
# Initialize HyperRAG development
python tools/development/hyperrag_setup.py

# Create custom cognitive service
python tools/development/create_cognitive_service.py --name YourService

# Test neural memory system
python tools/development/test_neural_memory.py
```

### Constitutional Computing Development

```bash
# Test constitutional compliance
python tools/development/test_constitutional.py

# Develop moderation rules
python tools/development/moderation_dev.py

# Test privacy tiers
python tools/development/test_privacy_tiers.py
```

This setup guide provides a comprehensive foundation for AIVillage development. The modular architecture and professional tooling support efficient development workflows while maintaining high code quality standards.