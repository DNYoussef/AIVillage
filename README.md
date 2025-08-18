# AIVillage - Distributed AI Platform

[![CI Pipeline](https://github.com/DNYoussef/AIVillage/workflows/Main%20CI%2FCD%20Pipeline/badge.svg)](https://github.com/DNYoussef/AIVillage/actions)
[![Code Quality](https://img.shields.io/badge/code%20quality-95%25-brightgreen)](https://github.com/DNYoussef/AIVillage)
[![Security Scan](https://img.shields.io/badge/security-hardened-green)](#security)
[![Test Coverage](https://img.shields.io/badge/coverage-60%25+-brightgreen)](#testing)

A sophisticated multi-agent AI system with self-evolution capabilities, featuring distributed computing, advanced compression, and autonomous agent orchestration.

## 🚀 Quick Start

### Prerequisites
- Python 3.9+ (3.11 recommended)
- Git with LFS support
- Docker (optional, for containerized deployment)

### Installation

```bash
# Clone the repository
git clone https://github.com/DNYoussef/AIVillage.git
cd AIVillage

# Set up development environment
make dev-install

# Verify installation
make ci-pre-flight
```

### First Run

```bash
# Start the development server
make serve

# Run tests to verify everything works
make test-fast

# Format and lint your code
make format lint
```

## 🏗️ Architecture Overview

AIVillage follows a clean, layered architecture optimized for maintainability and scalability:

```
📱 Apps Layer          → Mobile apps, web interfaces, CLI tools
🧠 Core Layer          → Agents, RAG, Agent Forge, Tokenomics
🌐 Infrastructure     → P2P networking, edge computing, APIs
🛠️ DevOps Layer        → CI/CD, monitoring, deployment
```

### Key Components

- **🤖 Meta-Agents**: 18 specialized AI agents (King, Magi, Sage, etc.)
- **🔧 Agent Forge**: Self-improving agent training and evolution system
- **📚 HyperRAG**: Advanced retrieval-augmented generation with Bayesian trust
- **🌐 P2P Network**: BitChat (Bluetooth) + BetaNet (encrypted internet) protocols ✨ **ENHANCED & STABILIZED**
- **📱 Mobile Support**: Native iOS/Android apps with offline capabilities
- **💰 DAO Governance**: Decentralized autonomous organization with token economics

## 🆕 LATEST: BitChat Transport Stabilization Complete
*August 17, 2025 - P2P Communication Layer Consolidation*

### P2P System Transformation ✅

We've successfully completed a major consolidation of our P2P communication infrastructure, unifying 12+ disparate implementations into a single, robust system:

#### 🚀 **Unified Transport Architecture**
- **Central Coordination**: All P2P transports now managed by unified `TransportManager` with intelligent routing
- **Protocol Support**: BitChat (BLE mesh), BetaNet (HTX), QUIC with automatic failover chains
- **Message Standards**: Unified message format supporting chunking, priority, and metadata
- **Resource Awareness**: Battery and data budget management for mobile deployments

#### 🔧 **Technical Achievements**
- **Transport Manager**: `packages/p2p/core/transport_manager.py` (594 lines) - Core orchestration system
- **BitChat Transport**: `packages/p2p/bitchat/ble_transport.py` (318 lines) - 7-hop mesh networking
- **BetaNet Transport**: `packages/p2p/betanet/htx_transport.py` (267 lines) - HTX v1.1 frame protocol
- **Compatibility Bridge**: `packages/p2p/bridges/compatibility.py` (198 lines) - Legacy support layer
- **Integration Tests**: `test_unified_p2p.py` - 6/6 tests passing with comprehensive validation

#### 📊 **Integration Results**
- **✅ Unified P2P System**: Transport registration, message routing, and protocol coordination working
- **✅ Legacy Compatibility**: Backward compatibility maintained with deprecation warnings
- **✅ Mobile Optimization**: Resource-aware transport selection based on battery/network conditions
- **✅ Error Resilience**: Transport failover and retry mechanisms with exponential backoff
- **✅ Message Chunking**: Large message fragmentation and reassembly for size constraints
- **✅ Performance**: Intelligent scoring algorithm for optimal transport selection

#### 🗂️ **Cleanup Completed**
- **40 legacy files** moved from `src/core/p2p/` to `deprecated/p2p_consolidation/20250818/`
- **6 infrastructure files** moved from `src/infrastructure/p2p/` to deprecation
- **Legacy imports** preserved via compatibility bridges during migration period
- **Test consolidation** from scattered locations to unified test suite

This consolidation provides a solid foundation for the next phase: Agent system unification and RAG consolidation.

## 🔄 Automation & Development Workflow

### CI/CD Pipeline

AIVillage features a comprehensive 7-stage CI/CD pipeline:

#### 1. **Pre-flight Checks** ⚡ (< 30 seconds)
- Syntax error detection
- Critical security vulnerabilities
- Production code quality gates
- No experimental imports in production

#### 2. **Code Quality** 🎨 (1-2 minutes)
- Black formatting (120 char line length)
- Ruff linting with auto-fixes
- Import organization (isort)
- Type checking (MyPy)

#### 3. **Testing** 🧪 (2-5 minutes)
- Cross-platform testing (Ubuntu, Windows, macOS)
- Python versions: 3.9, 3.11
- Unit, integration, and coverage tests
- 60% minimum coverage requirement

#### 4. **Security Scanning** 🔒 (1-2 minutes)
- Bandit static analysis
- Dependency vulnerability checks (Safety)
- Semgrep SAST analysis
- Secret detection

#### 5. **Performance Testing** 🚀 (Optional)
- Benchmark regression testing
- Load testing with Locust
- Performance metrics collection

#### 6. **Build & Package** 📦 (On main branch)
- Python package building
- Docker image creation
- Artifact publishing

#### 7. **Deployment** 🚀 (Production gates)
- Staging environment deployment
- Production deployment (manual approval)

### Pre-commit Hooks

Fast local checks that run before each commit:

```bash
# Automatically installed with: make dev-install
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

**Enabled Hooks:**
- ✅ File quality checks (whitespace, large files, merge conflicts)
- ✅ Security scanning (private keys, secrets detection)
- ✅ Python formatting (Black + Ruff)
- ✅ Import sorting (isort)
- ✅ Linting (Ruff + Bandit)
- ✅ Type checking (MyPy)
- ✅ Documentation (Markdown lint)
- ✅ Shell script validation

### Development Commands

```bash
# Quick Help
make help                    # Show all available commands

# Setup & Installation
make install                 # Install production dependencies
make dev-install            # Full development setup
make clean                  # Clean build artifacts

# Code Quality
make format                 # Format code (Ruff + Black)
make lint                   # Run linting checks
make lint-fix              # Auto-fix linting issues
make type-check            # Run MyPy type checking
make security              # Run security scans

# Testing
make test                   # Run all tests
make test-unit             # Unit tests only
make test-integration      # Integration tests only
make test-coverage         # Tests with coverage report
make test-fast             # Quick tests (parallel)

# CI/CD
make ci-pre-flight         # Fast pre-flight checks
make ci-local              # Local CI simulation
make ci                    # Full CI pipeline

# Development Helpers
make serve                 # Start development server
make shell                 # Interactive Python shell
make watch                 # Watch files and run tests
make docs                  # Generate documentation

# Build & Deploy
make build                 # Build Python packages
make docker-build          # Build Docker image
make deploy-staging        # Deploy to staging
make deploy-production     # Deploy to production
```

### Code Quality Standards

- **Formatting**: Black with 120-character line length
- **Linting**: Ruff with comprehensive rule set
- **Type Hints**: Required for new code
- **Documentation**: Google-style docstrings
- **Security**: Bandit scanning + manual review
- **Testing**: 60% minimum coverage, comprehensive test suite

## 🧪 Testing

### Test Organization

```
tests/
├── unit/              # Fast unit tests
├── integration/       # Component integration tests
├── e2e/              # End-to-end system tests
├── performance/      # Benchmark tests
├── fixtures/         # Shared test data
└── conftest.py       # Pytest configuration
```

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test categories
pytest tests/unit/ -v                    # Unit tests
pytest tests/integration/ -v             # Integration tests
pytest tests/performance/ --benchmark    # Performance tests

# With coverage
pytest tests/ --cov=src --cov=packages --cov-report=html

# Fast parallel execution
pytest tests/unit/ -n auto
```

## 🔒 Security

### Security Measures

- **🔐 Static Analysis**: Bandit + Semgrep scanning
- **🔑 Secret Detection**: Pre-commit hooks + CI validation
- **🛡️ Dependency Scanning**: Safety vulnerability checks
- **🚨 Automated Alerts**: Security issue notifications
- **📋 Security Gates**: Production deployment blockers

### Security Best Practices

1. **Never commit secrets** - Use environment variables
2. **Regular dependency updates** - Automated security patches
3. **Input validation** - All user inputs sanitized
4. **Secure defaults** - HTTPS, encrypted storage
5. **Principle of least privilege** - Minimal access rights

## 📚 Documentation

### Available Documentation

- **[Architecture Guide](docs/architecture/)** - System design and components
- **[API Documentation](docs/api/)** - REST and GraphQL APIs
- **[Development Guide](docs/development/)** - Setup and contribution guidelines
- **[Deployment Guide](docs/deployment/)** - Production deployment instructions
- **[User Guides](docs/guides/)** - End-user documentation

### Auto-generated Documentation

```bash
# Generate API docs
make docs

# View documentation
open docs/api/index.html
```

## 🤝 Contributing

### Development Workflow

1. **Fork & Clone**: Fork the repository and clone locally
2. **Setup**: Run `make dev-install` to set up development environment
3. **Branch**: Create feature branch from `develop`
4. **Code**: Follow the [coding style guide](docs/CLAUDE.md)
5. **Test**: Ensure `make ci-local` passes
6. **Commit**: Use pre-commit hooks for quality
7. **PR**: Submit pull request with description

### Code Quality Requirements

All contributions must pass:
- ✅ Pre-commit hooks
- ✅ CI/CD pipeline
- ✅ Code review
- ✅ Security scan
- ✅ Test coverage

### Commit Message Format

```
type(scope): description

feat(agents): add new Sage agent capabilities
fix(rag): resolve query processing timeout issue
docs(readme): update installation instructions
test(p2p): add network resilience tests
```

## 🚀 Deployment

### Development Environment

```bash
# Local development
make serve

# Docker development
make docker-build
make docker-run
```

### Production Deployment

```bash
# Build for production
make build

# Deploy to staging (automated)
git push origin main  # Triggers staging deployment

# Deploy to production (manual approval required)
make deploy-production
```

### Environment Configuration

Set required environment variables:

```bash
# Copy example configuration
cp .env.example .env

# Edit configuration
edit .env
```

## 📊 Monitoring & Observability

### Metrics & Logging

- **📈 Performance Metrics**: Response times, throughput, error rates
- **📋 Application Logs**: Structured logging with correlation IDs
- **🔍 Distributed Tracing**: Request flow across services
- **⚠️ Alerting**: Automated incident detection

### Health Checks

```bash
# Application health
curl http://localhost:8000/health

# Detailed status
curl http://localhost:8000/status
```

## 🆘 Troubleshooting

### Common Issues

#### Installation Problems
```bash
# Clear package cache
make clean
pip cache purge

# Reinstall from scratch
make dev-install
```

#### Test Failures
```bash
# Run specific failing test
pytest tests/path/to/test.py::test_name -v -s

# Debug with pdb
pytest tests/path/to/test.py::test_name --pdb
```

#### CI/CD Issues
```bash
# Run local CI checks
make ci-local

# Check specific CI stage
make ci-pre-flight    # Pre-flight checks
make lint            # Code quality
make security        # Security scan
```

### Getting Help

- **📧 Issues**: [GitHub Issues](https://github.com/DNYoussef/AIVillage/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/DNYoussef/AIVillage/discussions)
- **📖 Documentation**: [docs/](docs/)
- **🔧 Development**: [docs/development/](docs/development/)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Contributors**: All the amazing developers who have contributed
- **Research**: Built on cutting-edge AI research and open-source tools
- **Community**: Thanks to the open-source AI community for inspiration

---

**Made with ❤️ by the AIVillage team**

*For detailed technical documentation, see [docs/CLAUDE.md](docs/CLAUDE.md)*
