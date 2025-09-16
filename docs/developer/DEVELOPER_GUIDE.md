# AIVillage Developer Onboarding Guide

## Welcome to AIVillage

This comprehensive guide will help you get started with AIVillage development, from setting up your environment to contributing to the codebase.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Project Structure](#project-structure)
4. [Development Workflow](#development-workflow)
5. [Core Concepts](#core-concepts)
6. [API Development](#api-development)
7. [Agent Development](#agent-development)
8. [Testing Guidelines](#testing-guidelines)
9. [Contributing Guidelines](#contributing-guidelines)
10. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

**Minimum Requirements**:
- Python 3.9+
- Node.js 16+
- Git
- 8 GB RAM
- 50 GB storage

**Recommended**:
- Python 3.11+
- Node.js 18+
- Docker Desktop
- 16 GB RAM
- 100 GB SSD
- CUDA-compatible GPU (optional)

### Required Knowledge

- **Python**: Intermediate to advanced
- **FastAPI**: Basic understanding
- **AsyncIO**: Asynchronous programming
- **Machine Learning**: Basic concepts (PyTorch, Transformers)
- **APIs**: REST API design and development
- **Git**: Version control workflows

### Recommended Knowledge

- **Docker**: Containerization
- **Kubernetes**: Container orchestration
- **P2P Networks**: Distributed systems
- **Cryptography**: Security concepts
- **Graph Databases**: Neo4j, knowledge graphs

## Environment Setup

### 1. Clone the Repository

```bash
# Clone the main repository
git clone https://github.com/aivillage/aivillage.git
cd aivillage

# Clone submodules (if any)
git submodule update --init --recursive
```

### 2. Python Environment Setup

**Using pyenv (Recommended)**:

```bash
# Install Python 3.11
pyenv install 3.11.7
pyenv local 3.11.7

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

**Using conda**:

```bash
# Create environment
conda create -n aivillage python=3.11
conda activate aivillage
```

### 3. Install Dependencies

```bash
# Install core dependencies
pip install -e .

# Install development dependencies
pip install -e ".[dev,test]"

# Install P2P dependencies (optional)
pip install -e ".[p2p]"
```

### 4. Environment Configuration

Create a `.env` file in the project root:

```bash
# Copy example environment file
cp .env.example .env

# Edit with your settings
nano .env
```

**Required Environment Variables**:

```env
# Core Configuration
AIVILLAGE_ENV=development
DEBUG=true
API_KEY=your-development-api-key

# Database Configuration
DATABASE_URL=postgresql://user:pass@localhost:5432/aivillage
REDIS_URL=redis://localhost:6379/0
NEO4J_URL=bolt://localhost:7687
QDRANT_URL=http://localhost:6333

# AI/ML Configuration
HUGGINGFACE_TOKEN=your-hf-token
OPENAI_API_KEY=your-openai-key  # Optional
ANTHROPIC_API_KEY=your-anthropic-key  # Optional

# P2P Configuration
P2P_ENABLED=true
BITCHAT_ENABLED=true
BETANET_PORT=8888
QUIC_PORT=8889

# Security
JWT_SECRET_KEY=your-jwt-secret-key
ENCRYPTION_KEY=your-32-byte-encryption-key

# Development
LOG_LEVEL=DEBUG
ENABLE_PROFILING=true
WANDB_PROJECT=aivillage-dev
```

### 5. Database Setup

**PostgreSQL** (Primary database):

```bash
# Using Docker
docker run --name aivillage-postgres \
  -e POSTGRES_DB=aivillage \
  -e POSTGRES_USER=aivillage \
  -e POSTGRES_PASSWORD=password \
  -p 5432:5432 -d postgres:15

# Run migrations
alembic upgrade head
```

**Redis** (Caching):

```bash
# Using Docker
docker run --name aivillage-redis -p 6379:6379 -d redis:7
```

**Neo4j** (Knowledge Graph):

```bash
# Using Docker
docker run --name aivillage-neo4j \
  -e NEO4J_AUTH=neo4j/password \
  -p 7474:7474 -p 7687:7687 \
  -d neo4j:5.0
```

**Qdrant** (Vector Database):

```bash
# Using Docker
docker run --name aivillage-qdrant -p 6333:6333 -d qdrant/qdrant
```

### 6. Verify Installation

```bash
# Run health check
python -m core.health_check

# Start development server
python -m infrastructure.gateway.server

# Test API
curl http://localhost:8080/healthz
```

## Project Structure

```
aivillage/
├── core/                           # Core system components
│   ├── agent_forge/               # 7-phase ML pipeline
│   │   ├── phases/                # Training phases
│   │   ├── compression/           # Model compression
│   │   └── unified_pipeline.py    # Main pipeline orchestrator
│   ├── agents/                    # 54 specialized agents
│   │   ├── governance/           # Governance agents
│   │   ├── infrastructure/       # Infrastructure agents
│   │   └── specialized/          # Domain-specific agents
│   └── hyperrag/                 # Neural memory system
│       ├── retrieval/            # Hybrid retrieval
│       ├── cognition/            # Cognitive nexus
│       └── bayesian/             # Trust networks
├── infrastructure/                 # Infrastructure services
│   ├── gateway/                   # API gateway
│   ├── p2p/                      # P2P networking
│   │   ├── bitchat/              # BLE mesh transport
│   │   ├── betanet/              # HTX transport
│   │   └── quic/                 # QUIC transport
│   ├── fog/                      # Fog computing
│   │   ├── marketplace/          # Resource marketplace
│   │   ├── scheduler/            # Task scheduler
│   │   └── security/             # Constitutional policies
│   └── security/                 # Security framework
│       ├── auth/                 # Authentication
│       ├── encryption/           # Cryptography
│       └── compliance/           # Regulatory compliance
├── tests/                         # Test suites
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   ├── e2e/                      # End-to-end tests
│   └── fixtures/                 # Test fixtures
├── docs/                          # Documentation
│   ├── architecture/             # Architecture docs
│   ├── api/                      # API documentation
│   ├── developer/                # Developer guides
│   └── deployment/               # Deployment guides
├── scripts/                       # Utility scripts
│   ├── setup/                    # Setup scripts
│   ├── migration/                # Database migrations
│   └── deployment/               # Deployment scripts
├── config/                        # Configuration files
│   ├── development.yaml          # Dev configuration
│   ├── production.yaml           # Prod configuration
│   └── testing.yaml              # Test configuration
└── requirements/                  # Dependency files
    ├── base.txt                  # Core dependencies
    ├── development.txt           # Dev dependencies
    └── production.txt            # Prod dependencies
```

## Development Workflow

### 1. Git Workflow

We use a modified GitFlow workflow:

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and commit
git add .
git commit -m "feat: add new feature"

# Push branch
git push origin feature/your-feature-name

# Create pull request on GitHub
```

**Commit Message Convention**:

```
<type>(<scope>): <description>

<body>

<footer>
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`
**Scopes**: `core`, `agents`, `p2p`, `fog`, `security`, `docs`

**Examples**:
```
feat(agents): add new governance agent for policy enforcement
fix(p2p): resolve BitChat connection timeout issue
docs(api): update REST API documentation
test(fog): add integration tests for scheduler
```

### 2. Development Server

**Start Development Environment**:

```bash
# Start all services with Docker Compose
docker-compose -f docker-compose.dev.yml up -d

# Or start manually
# Terminal 1: Start databases
docker-compose up -d postgres redis neo4j qdrant

# Terminal 2: Start development server
AIVILLAGE_DEV_MODE=true python -m infrastructure.gateway.server

# Terminal 3: Start P2P network (optional)
python -m infrastructure.p2p.node_runner

# Terminal 4: Start agent system
python -m core.agents.agent_runner
```

**Hot Reload**:
The development server supports hot reload for most Python files. For structural changes, restart the server.

### 3. Code Quality Tools

**Linting and Formatting**:

```bash
# Format code with Black
black .

# Lint with Ruff
ruff check .

# Fix auto-fixable issues
ruff check --fix .

# Type checking with MyPy
mypy .

# Security scanning with Bandit
bandit -r .
```

**Pre-commit Hooks**:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

## Core Concepts

### 1. Agent System

AIVillage implements 54 specialized agents organized into domains:

```python
from core.agents import Agent, AgentCapabilities
from core.agents.base import BaseAgent

class CustomAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_id="custom_agent",
            capabilities=AgentCapabilities(
                rag_access=True,
                memory_access=True,
                p2p_communication=True,
                mcp_tools=True
            )
        )
    
    async def process_message(self, message: dict) -> dict:
        """Process incoming message."""
        # Agent-specific logic here
        return {"response": "Processed message"}
```

### 2. Agent Forge Pipeline

The 7-phase ML training pipeline:

```python
from core.agent_forge import UnifiedPipeline, UnifiedConfig

# Configure pipeline
config = UnifiedConfig(
    base_models=[
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "Qwen/Qwen2-1.5B-Instruct"
    ],
    enable_cognate=True,
    enable_evomerge=True,
    enable_quietstar=True,
    # ... other configuration
)

# Create and run pipeline
pipeline = UnifiedPipeline(config)
result = await pipeline.run_pipeline()
```

### 3. P2P Networking

Multi-transport P2P communication:

```python
from infrastructure.p2p import TransportManager, Message

# Initialize transport manager
transport = TransportManager()

# Send message
message = Message(
    type="agent_communication",
    sender="agent_1",
    recipient="agent_2",
    payload={"text": "Hello from agent 1"}
)

await transport.send_message("agent_2", message.serialize())
```

### 4. RAG System Integration

```python
from core.hyperrag import EnhancedRAGPipeline

# Initialize RAG system
rag = EnhancedRAGPipeline()
await rag.initialize()

# Query knowledge base
result = await rag.process("What is the P2P network architecture?")
print(result["answer"])
```

## API Development

### 1. Creating New Endpoints

```python
from fastapi import APIRouter, Depends, HTTPException
from infrastructure.gateway.auth import JWTBearer, TokenPayload
from pydantic import BaseModel

router = APIRouter(prefix="/v1/custom", tags=["custom"])
jwt_auth = JWTBearer()

class CustomRequest(BaseModel):
    data: str
    options: dict = {}

class CustomResponse(BaseModel):
    result: str
    status: str = "success"

@router.post("/process", response_model=CustomResponse)
async def process_custom_request(
    request: CustomRequest,
    token: TokenPayload = Depends(jwt_auth)
):
    """Process custom request."""
    try:
        # Process request
        result = await process_data(request.data, request.options)
        
        return CustomResponse(
            result=result,
            status="success"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )

# Register router in main application
from infrastructure.gateway.unified_api_gateway import app
app.include_router(router)
```

### 2. Authentication and Authorization

```python
from infrastructure.gateway.auth import JWTBearer, require_permissions

# Require authentication
@router.get("/protected")
async def protected_endpoint(token: TokenPayload = Depends(jwt_auth)):
    return {"user_id": token.sub, "message": "Access granted"}

# Require specific permissions
@router.post("/admin-only")
async def admin_endpoint(
    token: TokenPayload = Depends(require_permissions(["admin"]))
):
    return {"message": "Admin access granted"}
```

### 3. Database Integration

```python
from sqlalchemy.orm import Session
from infrastructure.database import get_db, models

@router.get("/users/{user_id}")
async def get_user(
    user_id: int,
    db: Session = Depends(get_db),
    token: TokenPayload = Depends(jwt_auth)
):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
```

## Agent Development

### 1. Creating Custom Agents

```python
from core.agents.base import BaseAgent
from core.agents.capabilities import AgentCapabilities
from core.agents.memory import AgentMemory

class DataAnalystAgent(BaseAgent):
    """Custom data analysis agent."""
    
    def __init__(self):
        super().__init__(
            agent_id="data_analyst",
            name="Data Analyst",
            description="Specialized in data analysis and visualization",
            capabilities=AgentCapabilities(
                rag_access=True,
                memory_access=True,
                p2p_communication=True,
                mcp_tools=True,
                specialized_tools=["pandas", "matplotlib", "seaborn"]
            )
        )
        self.memory = AgentMemory(agent_id=self.agent_id)
    
    async def analyze_data(self, data: dict) -> dict:
        """Analyze provided data."""
        # Store analysis request in memory
        await self.memory.store(
            key=f"analysis_{int(time.time())}",
            value=data,
            metadata={"type": "analysis_request"}
        )
        
        # Perform analysis
        results = self._perform_analysis(data)
        
        # Store results
        await self.memory.store(
            key=f"results_{int(time.time())}",
            value=results,
            metadata={"type": "analysis_results"}
        )
        
        return results
    
    def _perform_analysis(self, data: dict) -> dict:
        """Internal analysis logic."""
        # Implementation here
        return {"summary": "Analysis complete"}
```

### 2. Agent Communication

```python
class CommunicatingAgent(BaseAgent):
    async def send_message_to_agent(self, recipient: str, message: dict):
        """Send message to another agent."""
        await self.p2p_bridge.send_agent_message(
            sender_id=self.agent_id,
            recipient_id=recipient,
            message=message,
            channel="default"
        )
    
    async def handle_incoming_message(self, sender: str, message: dict):
        """Handle message from another agent."""
        # Process incoming message
        response = await self.process_message(message)
        
        # Send response back
        if response:
            await self.send_message_to_agent(sender, response)
```

### 3. RAG Integration

```python
class KnowledgeAgent(BaseAgent):
    def __init__(self):
        super().__init__(agent_id="knowledge_agent")
        self.rag_system = None
    
    async def initialize(self):
        """Initialize RAG system connection."""
        from core.hyperrag import EnhancedRAGPipeline
        self.rag_system = EnhancedRAGPipeline()
        await self.rag_system.initialize()
    
    async def query_knowledge(self, query: str) -> dict:
        """Query the knowledge base."""
        if not self.rag_system:
            raise RuntimeError("RAG system not initialized")
        
        result = await self.rag_system.process(query)
        
        # Store query and result in memory
        await self.memory.store(
            key=f"query_{int(time.time())}",
            value={"query": query, "result": result},
            metadata={"type": "knowledge_query"}
        )
        
        return result
```

## Testing Guidelines

### 1. Unit Tests

```python
import pytest
from unittest.mock import AsyncMock, Mock
from core.agents.custom_agent import CustomAgent

@pytest.fixture
async def custom_agent():
    agent = CustomAgent()
    await agent.initialize()
    return agent

@pytest.mark.asyncio
async def test_agent_message_processing(custom_agent):
    """Test agent message processing."""
    message = {"type": "test", "data": "hello"}
    result = await custom_agent.process_message(message)
    
    assert result["status"] == "processed"
    assert "response" in result

@pytest.mark.asyncio
async def test_agent_communication(custom_agent):
    """Test agent-to-agent communication."""
    # Mock P2P bridge
    custom_agent.p2p_bridge = AsyncMock()
    
    await custom_agent.send_message_to_agent(
        "other_agent",
        {"message": "test"}
    )
    
    # Verify message was sent
    custom_agent.p2p_bridge.send_agent_message.assert_called_once()
```

### 2. Integration Tests

```python
@pytest.mark.integration
async def test_full_agent_workflow():
    """Test complete agent workflow."""
    # Start test environment
    async with TestEnvironment() as env:
        # Create agents
        agent1 = env.create_agent("test_agent_1")
        agent2 = env.create_agent("test_agent_2")
        
        # Send message from agent1 to agent2
        message = {"type": "greeting", "text": "Hello agent2"}
        await agent1.send_message_to_agent("test_agent_2", message)
        
        # Wait for message processing
        await asyncio.sleep(0.1)
        
        # Verify agent2 received message
        received_messages = await agent2.get_received_messages()
        assert len(received_messages) == 1
        assert received_messages[0]["text"] == "Hello agent2"
```

### 3. API Tests

```python
from fastapi.testclient import TestClient
from infrastructure.gateway.unified_api_gateway import app

client = TestClient(app)

def test_health_endpoint():
    """Test health check endpoint."""
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_protected_endpoint():
    """Test protected endpoint requires authentication."""
    # Test without token
    response = client.get("/v1/protected")
    assert response.status_code == 401
    
    # Test with valid token
    token = generate_test_token()
    headers = {"Authorization": f"Bearer {token}"}
    response = client.get("/v1/protected", headers=headers)
    assert response.status_code == 200
```

### 4. Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest -m "not slow"  # Skip slow tests

# Run with coverage
pytest --cov=core --cov=infrastructure --cov-report=html

# Run performance tests
pytest tests/performance/ -v

# Run security tests
pytest tests/security/ -v
```

## Contributing Guidelines

### 1. Code Style

- **Python**: Follow PEP 8, use Black formatter
- **Naming**: Use descriptive names, follow Python conventions
- **Documentation**: All public functions must have docstrings
- **Type Hints**: Use type hints for function parameters and returns
- **Comments**: Explain why, not what

### 2. Pull Request Process

1. **Fork and Branch**: Create feature branch from `develop`
2. **Implement**: Write code following style guidelines
3. **Test**: Add tests for new functionality
4. **Document**: Update documentation as needed
5. **Review**: Submit pull request for code review
6. **CI/CD**: Ensure all checks pass
7. **Merge**: Maintainer merges after approval

### 3. Code Review Checklist

**Functionality**:
- [ ] Code works as intended
- [ ] Handles edge cases and errors
- [ ] Performance is acceptable
- [ ] No security vulnerabilities

**Quality**:
- [ ] Follows coding standards
- [ ] Adequate test coverage
- [ ] Documentation is updated
- [ ] No code duplication

**Architecture**:
- [ ] Fits well with existing architecture
- [ ] Doesn't break existing functionality
- [ ] Follows SOLID principles
- [ ] Proper separation of concerns

## Troubleshooting

### Common Issues

#### 1. Installation Problems

**Issue**: Dependency conflicts
```bash
# Solution: Use fresh virtual environment
rm -rf venv/
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install --upgrade pip
pip install -e ".[dev]"
```

**Issue**: CUDA/PyTorch problems
```bash
# Solution: Install correct PyTorch version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 2. Database Connection Issues

**Issue**: PostgreSQL connection failed
```bash
# Check if PostgreSQL is running
docker ps | grep postgres

# Restart PostgreSQL container
docker restart aivillage-postgres

# Check connection
psql -h localhost -U aivillage -d aivillage
```

**Issue**: Redis connection failed
```bash
# Check Redis status
docker ps | grep redis

# Test Redis connection
redis-cli -h localhost ping
```

#### 3. Agent Communication Issues

**Issue**: Agents not receiving messages
```bash
# Check P2P network status
curl http://localhost:8080/v1/p2p/status

# Check agent status
curl http://localhost:8080/v1/agents/status

# Enable debug logging
export LOG_LEVEL=DEBUG
```

#### 4. Performance Issues

**Issue**: High memory usage
```bash
# Profile memory usage
python -m memory_profiler your_script.py

# Monitor resource usage
htop
# or
docker stats
```

**Issue**: Slow API responses
```bash
# Enable profiling
export ENABLE_PROFILING=true

# Check slow query log
tail -f logs/slow_queries.log

# Profile API endpoints
curl -w "@curl-format.txt" http://localhost:8080/api/endpoint
```

### Getting Help

1. **Documentation**: Check existing documentation first
2. **Issues**: Search GitHub issues for similar problems
3. **Discussions**: Use GitHub Discussions for questions
4. **Discord**: Join the developer Discord server
5. **Stack Overflow**: Tag questions with `aivillage`

### Debugging Tips

1. **Enable Debug Logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Use debugger**:
   ```python
   import pdb; pdb.set_trace()  # Python debugger
   # or
   breakpoint()  # Python 3.7+
   ```

3. **Profile Performance**:
   ```python
   import cProfile
   cProfile.run('your_function()')
   ```

4. **Monitor Resources**:
   ```bash
   # System resources
   htop
   
   # Docker resources
   docker stats
   
   # Application metrics
   curl http://localhost:8080/metrics
   ```

## Next Steps

After completing this guide:

1. **Explore Codebase**: Browse the code to understand the architecture
2. **Run Examples**: Try the example scripts in `examples/`
3. **Read Architecture Docs**: Review detailed architecture documentation
4. **Join Community**: Participate in discussions and code reviews
5. **Start Contributing**: Pick up a "good first issue" and submit a PR

## Additional Resources

- [API Documentation](../api/REST_API.md)
- [Architecture Overview](../architecture/SYSTEM_ARCHITECTURE.md)
- [P2P Networking Guide](../architecture/P2P_NETWORKING.md)
- [Security Guidelines](../security/SECURITY_GUIDE.md)
- [Deployment Guide](../deployment/DEPLOYMENT_GUIDE.md)

---

*Happy coding! Welcome to the AIVillage community!* 🚀

---

*Last Updated: January 2025*
*Version: 3.0.0*
*Status: Production Ready*