# Contributing to AIVillage

Thank you for your interest in contributing to AIVillage! This document provides guidelines and information for contributors.

**Archaeological Integration Status**: ACTIVE (v2.1.0)  
**Latest Enhancement**: Archaeological innovations from 81 branches integrated  
**Current Focus**: Production-ready AI infrastructure with enhanced security, monitoring, and optimization

## 🚀 Quick Start

1. **Fork & Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/AIVillage.git
   cd AIVillage
   ```

2. **Setup Development Environment**
   ```bash
   # Install development dependencies
   pip install -r config/requirements/requirements-dev.txt -c config/constraints.txt
   
   # Start local services (if using Docker)
   # docker-compose up -d
   
   # Install development dependencies
   pip install -r config/requirements/requirements-dev.txt -c config/constraints.txt
   ```

3. **Configure Archaeological Features (Optional)**
   ```bash
   cp .env.example .env.local
   # Edit .env.local to configure local development settings
   export AIVILLAGE_ENV=development
   ```

4. **Verify Setup**
   ```bash
   # Run basic validation
   python scripts/test_enhanced_fog_integration.py
   
   # Run tests
   pytest tests/ -v --tb=short
   
   # Check code quality
   ruff check . --fix
   black . --check
   ```

## 📊 Current Status (August 29, 2025)

**Before Contributing, Please Note:**
- **Archaeological Integration**: v2.1.0 ACTIVE - 3 major enhancements integrated
- **Tests**: 196/295 unit tests currently passing (66.4%), plus comprehensive archaeological tests
- **Test Files**: 491 test files, 93 have import/dependency issues
- **Infrastructure**: Core systems consolidated and production-ready with archaeological enhancements
- **Architecture**: Well-documented in [docs/architecture/ARCHITECTURE.md](docs/architecture/ARCHITECTURE.md)

### Archaeological Integration Summary
- ✅ **Enhanced Security Layer**: ECH + Noise Protocol (Innovation Score: 8.3/10)
- ✅ **Emergency Triage System**: ML-based anomaly detection (Innovation Score: 8.0/10)  
- ✅ **Tensor Memory Optimization**: Memory leak prevention (Innovation Score: 6.9/10)
- 📍 **Phase 2 Planned**: Distributed Inference, Evolution Scheduler, LibP2P Advanced Features

## 🛠️ Development Workflow

### Branch Strategy
- `main` - Production-ready code
- `develop` - Integration branch for features
- `feature/*` - Individual feature branches
- `fix/*` - Bug fix branches

### Making Changes

1. **Create Feature Branch**
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes**
   - Follow our [coding standards](#coding-standards)
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Locally**
   ```bash
   # Auto-format code
   black . --check --diff
   ruff check . --fix
   
   # Check code quality
   mypy . --ignore-missing-imports
   
   # Run quick tests
   pytest tests/ -v --tb=short -x
   
   # Full local CI equivalent
   ruff check . && black . --check && mypy . --ignore-missing-imports && pytest
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat(component): description of changes"
   ```

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a pull request via GitHub.

## 🎯 Areas Needing Contributions

### 🏺 Archaeological Integration (High Priority)
- **Phase 2 Implementation**: Distributed Inference Enhancement (20h estimated)
- **Evolution Scheduler Integration**: Automated model evolution with regression detection (28h)
- **LibP2P Advanced Features**: Enhanced mesh reliability and performance (40h)
- **Archaeological Pattern Mining**: Analyze remaining branches for additional innovations
- **Performance Benchmarking**: Quantify benefits of archaeological integrations

### High Priority (Core Systems)
- **Test Infrastructure**: Fix 93 failing test files with import issues
- **Integration Tests**: Add database setup automation
- **Documentation**: API documentation and examples, archaeological integration guides
- **Performance**: Optimize slow test execution, validate archaeological performance gains

### Medium Priority
- **Security**: Enhance threat modeling, extend ECH + Noise Protocol capabilities
- **Mobile**: iOS/Android client improvements
- **P2P**: Network resilience, integration with archaeological security enhancements
- **RAG**: Knowledge graph expansion and optimization
- **Monitoring**: Extend Emergency Triage System patterns and ML models

### Low Priority
- **UI/UX**: Dashboard improvements, archaeological feature dashboards
- **Observability**: Enhanced metrics beyond archaeological monitoring
- **Deployment**: Additional cloud provider support, archaeological deployment automation

## 📝 Coding Standards

### Python Code Style
- **Formatter**: Black with 120-character line length
- **Linter**: Ruff with comprehensive rule set
- **Type Hints**: Required for new code
- **Docstrings**: Google-style documentation

```python
def process_message(message: str, user_id: int) -> dict[str, Any]:
    """Process incoming message from user.

    Args:
        message: The message content to process
        user_id: Unique identifier for the user

    Returns:
        Dictionary containing processed message data

    Raises:
        ValueError: If message is empty or user_id is invalid
    """
    # Implementation here
    pass
```

### Commit Message Format
```
type(scope): description

feat(agents): add new Sage agent capabilities
fix(p2p): resolve BitChat connection timeout
docs(readme): update installation instructions
test(rag): add integration tests for knowledge graph
refactor(core): simplify message routing logic
```

**Types**: `feat`, `fix`, `docs`, `test`, `refactor`, `style`, `chore`
**Scopes**: `agents`, `p2p`, `rag`, `core`, `mobile`, `security`, `deploy`

## 🧪 Testing Guidelines

### Test Structure
```
tests/
├── unit/              # Fast unit tests
├── integration/       # Cross-component tests
├── e2e/              # End-to-end system tests
└── fixtures/         # Shared test data
```

### Writing Tests
- **Unit Tests**: Test individual functions/classes
- **Integration Tests**: Test component interactions
- **E2E Tests**: Test complete user workflows

```python
import pytest
from aivillage.core.agents import KingAgent

class TestKingAgent:
    """Test suite for King Agent functionality."""

    def test_task_assignment(self):
        """Test that King Agent can assign tasks to other agents."""
        king = KingAgent()
        task = {"type": "analyze", "data": "test_data"}

        result = king.assign_task(task, agent_type="sage")

        assert result["status"] == "assigned"
        assert result["agent_type"] == "sage"
```

### Test Requirements
- **Coverage**: Aim for 60%+ test coverage
- **Performance**: Unit tests should complete in <1s each
- **Isolation**: Tests should not depend on external services
- **Clarity**: Test names should describe the behavior being tested

## 🔒 Security Guidelines

### Security Review Required For
- Authentication/authorization changes
- Data persistence modifications
- P2P protocol changes
- API endpoint additions/changes
- Cryptographic implementations

### Security Checklist
- [ ] No hardcoded secrets or credentials
- [ ] Input validation implemented
- [ ] SQL injection prevention
- [ ] XSS protection in place
- [ ] Rate limiting on public endpoints
- [ ] Audit logging for sensitive operations

## 📋 Pull Request Process

### PR Template
Your PR description should include:

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix/feature causing existing functionality to break)
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] All tests passing locally
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No merge conflicts
```

### Review Process
1. **Automated Checks**: CI pipeline must pass
2. **Code Review**: At least one maintainer approval required
3. **Security Review**: Required for security-sensitive changes
4. **Documentation**: Ensure documentation is updated
5. **Testing**: Verify adequate test coverage

### Definition of Done
- [ ] Feature/fix implemented and working
- [ ] Unit tests written and passing
- [ ] Integration tests passing
- [ ] Documentation updated
- [ ] Code reviewed and approved
- [ ] Security review completed (if applicable)
- [ ] No breaking changes without migration plan

## 🤝 Community Guidelines

### Code of Conduct
- Be respectful and inclusive
- Provide constructive feedback
- Help others learn and grow
- Focus on what's best for the project

### Getting Help
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and general discussion
- **Documentation**: [docs/](docs/) directory
- **Architecture**: [TABLE_OF_CONTENTS.md](TABLE_OF_CONTENTS.md)

### Maintainer Response Times
- **Critical Issues**: 24 hours
- **Bug Reports**: 3-5 business days
- **Feature Requests**: 1-2 weeks
- **PRs**: 3-7 business days

## 🏗️ Development Setup Details

### Prerequisites
- Python 3.9+ (3.11 recommended)
- Docker & Docker Compose
- Git with LFS support
- Node.js 18+ (for some tools)

### Local Development Stack
```bash
# Start all services
# Start services manually or using Docker
# For development, configure database connections in .env

# Gateway service:
python infrastructure/gateway/enhanced_unified_api_gateway.py

# Available endpoints:
# - API Gateway: localhost:8000
# - Admin Dashboard: localhost:8000/admin_interface.html
# - API Docs: localhost:8000/docs
```

### Environment Variables
Copy `.env.example` to `.env` and configure:
```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/aivillage
NEO4J_URL=bolt://localhost:7687
REDIS_URL=redis://localhost:6379

# API Keys (optional for local dev)
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

## 📚 Additional Resources

- **Architecture Overview**: [docs/architecture/ARCHITECTURE.md](docs/architecture/ARCHITECTURE.md)
- **API Documentation**: [docs/api/](docs/api/)
- **Deployment Guide**: [docs/deployment/](docs/deployment/)
- **Development Troubleshooting**: [docs/development/](docs/development/)

## 🙏 Recognition

Contributors will be recognized in:
- Project README
- Release notes for significant contributions
- GitHub contributors section

Thank you for helping make AIVillage better! 🚀
