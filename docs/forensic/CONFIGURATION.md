# AIVillage Configuration Analysis

## Overview

This document provides a comprehensive analysis of the AIVillage project configuration, build systems, dependencies, CI/CD pipeline, and environment setup based on forensic analysis of configuration files.

## Project Structure

The AIVillage project is a distributed AI platform with multiple components:

- **Core**: Main application infrastructure
- **UI/Web**: React-based user interface with TypeScript
- **Infrastructure**: P2P networking, monitoring, and deployment
- **Services**: Microservices architecture with Docker containers
- **Tools**: Development, CI/CD, and build automation

## Package Configuration

### JavaScript/TypeScript Projects

#### Web UI Application (`apps/web` & `ui/web`)
- **Framework**: React 18.2.0 with TypeScript 4.9.5
- **Build System**: Vite 6.3.5 with TypeScript compilation
- **Version**: 2.0.0

**Key Dependencies**:
- React ecosystem: `react`, `react-dom`, `react-router-dom`
- Charting: `chart.js`, `react-chartjs-2`, `recharts`
- Crypto: `@noble/ciphers`, `crypto-js`
- P2P: `simple-peer`, `ws`

**Development Tools**:
- Testing: Jest 29.4.3 with Testing Library
- Linting: ESLint with TypeScript support
- Formatting: Prettier 2.8.4
- Bundling: Vite with React plugin

**Build Scripts**:
```json
{
  "dev": "vite",
  "build": "tsc && vite build",
  "test": "jest",
  "lint": "eslint . --ext .ts,.tsx --fix",
  "typecheck": "tsc --noEmit",
  "preview": "vite preview"
}
```

### Python Projects

#### Main Application (`config/build/pyproject.toml`)
- **Name**: aivillage
- **Version**: 0.5.1
- **Python**: >=3.9 (supports 3.9-3.12)
- **Build System**: setuptools with wheel

**Core Dependencies**:
- **Web Framework**: FastAPI 0.112.0+ with uvicorn
- **AI/ML Stack**: 
  - PyTorch 2.4.0+ ecosystem
  - Transformers 4.44.0+
  - Sentence-transformers 3.0.1+
  - Accelerate, PEFT, Datasets
- **Databases**:
  - PostgreSQL: `psycopg2-binary`, `sqlalchemy`, `alembic`
  - Redis: `redis` 5.0.8+
  - Neo4j: `neo4j` 5.3.0+
  - Vector: `qdrant-client` 1.11.1+
- **Data Science**: NumPy, Pandas, SciKit-learn, Matplotlib
- **Security**: `cryptography`, `PyJWT`, `bcrypt`

**CLI Scripts**:
```toml
[project.scripts]
aivillage = "core.cli:main"
aivillage-server = "bin.server:main"
forge = "agent_forge.cli:main"
village-dashboard = "core.dashboard:main"
```

#### P2P Infrastructure (`infrastructure/p2p/pyproject.toml`)
- **Name**: aivillage-p2p
- **Focus**: Peer-to-peer networking infrastructure
- **Python**: >=3.8 (broader compatibility)

**P2P Dependencies**:
- **Protocols**: libp2p, multiaddr, multihash
- **Mesh**: aioble, zeroconf
- **Privacy**: pynacl, noise-protocol, stem (Tor)
- **SCION**: scionproto

## Build Systems & Tools

### Make-based Build System (`config/build/Makefile`)

**Core Commands**:
- `setup`: Complete development environment setup
- `clean`: Remove build artifacts and caches
- `format`: Code formatting with Ruff and Black
- `lint`: Linting checks with configurable rules
- `test`: Comprehensive test suite
- `security`: Security scans (Bandit, Safety, Ruff)
- `ci`: Full CI pipeline with artifacts
- `artifacts`: Collect operational artifacts

**HRRM Bootstrap System** (Hierarchical Recursive Reasoning Model):
- `hrrm-train-all`: Train all HRRM models (Planner, Reasoner, Memory)
- `hrrm-eval-all`: Evaluate all models
- `hrrm-export`: Export to HuggingFace format
- `hrrm-acceptance`: Run acceptance criteria

**Docker Integration**:
- `compose-up/compose-down`: Development stack management
- `docker-build/docker-run`: Container operations

### Code Quality Configuration

#### Ruff (Primary Linter)
```toml
[tool.ruff]
target-version = "py311"
line-length = 120
select = ["E", "F", "I", "UP"]  # Essential rules
ignore = ["E501", "PLR0912", "PLR2004"]  # Compatibility ignores
```

#### Black (Code Formatter)
```toml
[tool.black]
line-length = 120
target-version = ['py312']
```

#### MyPy (Type Checking)
```toml
[tool.mypy]
python_version = "3.12"
ignore_missing_imports = true
strict = false
```

## CI/CD Pipeline

### GitHub Actions Configuration (`.github/workflows/main-ci.yml`)

**Multi-Stage Pipeline**:

1. **Pre-flight Checks** (Fast Fail)
   - Syntax validation with Ruff
   - Security scan for critical issues
   - Production code placeholder checks
   - Experimental import validation

2. **Code Quality**
   - Format checking (Black, line-length=120)
   - Linting (Ruff with multiple rule sets)
   - Type checking (MyPy with lenient config)

3. **Testing** (Matrix Strategy)
   - **OS Support**: Ubuntu, Windows, macOS
   - **Python Versions**: 3.9, 3.11
   - **Test Types**: Unit, Integration, P2P Network
   - **Coverage**: 60% minimum threshold

4. **Security Scanning** (Comprehensive)
   - Secret detection with detect-secrets
   - High/Critical CVE blocking (pip-audit)
   - Dependency vulnerability checking (Safety)
   - Static analysis (Bandit, Semgrep)
   - Anti-pattern detection

5. **Performance Testing** (Optional)
   - Benchmark execution
   - Load testing with Locust

6. **Build & Package**
   - SBOM generation
   - Docker image building
   - Artifact collection

7. **Security Gate Validation**
   - Deployment blocking on security failures
   - Comprehensive security report evaluation

**Environment Variables**:
```yaml
env:
  PYTHON_VERSION: '3.11'
  NODE_VERSION: '18'
  CARGO_TERM_COLOR: always
  PIP_DISABLE_PIP_VERSION_CHECK: 1
```

## Docker Configuration

### Main Compose Stack (`tools/ci-cd/deployment/docker/docker-compose.yml`)

**Core Services**:
- **twin**: AI model service (port 8001)
- **gateway**: API gateway (port 8000)
- **credits-api**: Blockchain credits API (port 8002)
- **credits-worker**: Background processing

**Databases**:
- **PostgreSQL 15**: Credits and transactional data
- **Neo4j 5.11**: Hypergraph knowledge base
- **Redis 7**: Caching and sessions
- **Qdrant**: Vector embeddings

**Additional Services**:
- **HyperRAG MCP**: WebSocket server (port 8765)
- **Monitoring**: Prometheus, Grafana, Pushgateway

**Network Configuration**:
```yaml
networks:
  ai-village-net:
    external: false
```

**Health Checks**: All services include comprehensive health monitoring

## Environment Configuration

### Development Environment (`.env` and variants)

**Core Settings**:
```env
# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=Atlantis

# Model Configuration  
DEFAULT_MODEL=nvidia/llama-3.1-nemotron-70b-instruct
LOCAL_MODEL=Qwen/Qwen2.5-3B-Instruct

# System Configuration
LOG_LEVEL=INFO
ENVIRONMENT=development
UI_PORT=8080
UI_HOST=0.0.0.0

# RAG System
VECTOR_STORE_PATH=./data/vector_store
GRAPH_STORE_PATH=./data/graph_store
```

**Environment Variants**:
- `.env.development`: Development settings
- `.env.production`: Production configuration
- `.env.staging`: Staging environment
- `.env.security`: Security-specific settings
- `.env.test`: Testing configuration

**Application Configuration** (`config/aivillage_config.yaml`):
```yaml
integration:
  evolution_metrics:
    enabled: true
    backend: sqlite
    db_path: ./data/evolution_metrics.db
  
  rag_pipeline:
    enabled: true
    embedding_model: paraphrase-MiniLM-L3-v2
    cache_enabled: true
    chunk_size: 512
  
  p2p_networking:
    enabled: true
    transport: libp2p
    discovery_method: mdns
    max_peers: 50
```

## Security Configuration

### Security Scanning Tools
- **detect-secrets**: Baseline secret detection
- **Bandit**: Python security static analysis
- **Safety**: Dependency vulnerability scanning
- **pip-audit**: CVE detection with CycloneDx SBOM
- **Semgrep**: SAST security analysis

### Security Gates
- **Blocking**: High/Critical CVEs and security violations
- **Non-blocking**: Medium/Low issues with reporting
- **Comprehensive reporting**: JSON artifacts for all tools

### Cryptographic Validation
- Algorithm validation scripts
- Secret sanitization checks
- Configuration externalization validation

## Development Workflow

### Local Development Setup
1. `make setup`: Install dependencies and configure environment
2. `make compose-up`: Start development stack
3. `make run-dev`: Start development server
4. `make test`: Run test suite

### Quality Assurance
1. `make format`: Code formatting
2. `make lint`: Linting checks
3. `make type-check`: Type validation
4. `make security`: Security scans
5. `make ci-local`: Full local CI

### Artifact Collection
- **Security**: Bandit, Safety reports
- **SBOM**: Software Bill of Materials
- **Performance**: Benchmark results
- **Quality**: Code quality metrics
- **Coverage**: Test coverage reports

## Key Dependencies Summary

### Python Core Stack
- **Web**: FastAPI + uvicorn + pydantic
- **AI/ML**: PyTorch + transformers + accelerate
- **Data**: NumPy + pandas + scikit-learn
- **Databases**: SQLAlchemy + psycopg2 + redis + neo4j + qdrant-client
- **Security**: cryptography + PyJWT + bcrypt
- **Async**: aiohttp + aiofiles + websockets

### JavaScript/TypeScript Stack
- **Frontend**: React 18 + TypeScript 4.9 + Vite 6
- **Testing**: Jest 29 + Testing Library
- **Tooling**: ESLint + Prettier + Babel
- **Crypto**: Noble ciphers + crypto-js
- **P2P**: simple-peer + ws

### Development Tools
- **Python**: ruff + black + mypy + pytest
- **Security**: bandit + safety + semgrep + detect-secrets
- **Containers**: Docker + docker-compose
- **CI/CD**: GitHub Actions with comprehensive pipeline

## Performance & Optimization

### Build Optimization
- **Parallel execution**: Matrix builds, concurrent jobs
- **Caching**: pip, node_modules, Docker layers
- **Artifact retention**: 30-90 days based on type
- **Fast failure**: Pre-flight checks for early exit

### Runtime Optimization
- **Health checks**: All services monitored
- **Resource limits**: Configured in compose files
- **Monitoring**: Prometheus + Grafana stack
- **Caching**: Redis for session and application cache

## Deployment Architecture

### Service Mesh
- **Gateway**: API gateway with rate limiting
- **Twin**: AI model inference service
- **Credits**: Blockchain-based credit system
- **HyperRAG**: Knowledge graph enhancement

### Infrastructure
- **Databases**: Multi-modal (relational, graph, vector, cache)
- **Networking**: Internal docker network with health checks
- **Monitoring**: Complete observability stack
- **Security**: Multiple layers of validation and scanning

This configuration analysis reveals a sophisticated, production-ready distributed AI platform with comprehensive tooling, security measures, and development workflows.