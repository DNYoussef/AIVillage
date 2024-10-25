# AI Village

A sophisticated multi-agent system featuring King, Sage, and Magi agents working together through a RAG (Retrieval-Augmented Generation) system.

## Overview

AI Village is a comprehensive system that combines:
- Advanced agent-based architecture
- Graph-based RAG system
- Real-time visualization
- Interactive web interface

### Core Components

1. **Agent System**
   - King Agent: Task planning and management
   - Sage Agent: Research and knowledge management
   - Magi Agent: Code generation and tool creation

2. **RAG System**
   - Vector-based retrieval
   - Knowledge graph integration
   - Dynamic knowledge updates
   - Query optimization

3. **Communication System**
   - Asynchronous messaging
   - Priority-based routing
   - Real-time updates
   - Secure channels

4. **User Interface**
   - Interactive dashboard
   - Knowledge graph visualization
   - Decision tree visualization
   - Chat interface

## Quick Start

1. **Prerequisites**
   ```bash
   # Install Python 3.8+
   python -m pip install --upgrade pip
   
   # Install Redis
   # Windows: Download from https://redis.io/download
   # Linux:
   sudo apt-get install redis-server
   
   # Install PostgreSQL
   # Windows: Download from https://postgresql.org/download
   # Linux:
   sudo apt-get install postgresql
   
   # Install Node.js
   # Download from https://nodejs.org/
   ```

2. **Installation**
   ```bash
   # Clone repository
   git clone https://github.com/yourusername/ai_village.git
   cd ai_village
   
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   .\venv\Scripts\activate   # Windows
   
   # Install dependencies
   pip install -r requirements.txt
   cd ui && npm install && cd ..
   ```

3. **Configuration**
   ```bash
   # Copy default config
   cp config/default.yaml config/local.yaml
   
   # Edit configuration
   # Set database credentials
   # Configure Redis connection
   # Set security keys
   ```

4. **Run the System**
   ```bash
   # Start Redis
   redis-server
   
   # Start API server
   cd api
   uvicorn main:app --reload
   
   # Start UI development server
   cd ui
   npm run dev
   ```

5. **Access**
   - Dashboard: http://localhost:3000
   - API docs: http://localhost:8000/docs

## Documentation

### System Architecture
- [System Overview](docs/system_architecture.md)
- [Component Interactions](docs/system_architecture.md#system-integration)
- [Security Model](docs/system_architecture.md#security)

### Getting Started
- [Installation Guide](docs/getting_started.md)
- [Configuration](docs/getting_started.md#configuration)
- [Development Setup](docs/getting_started.md#development-workflow)
- [Troubleshooting](docs/getting_started.md#troubleshooting)

### RAG System
- [Architecture](docs/rag_system.md#architecture)
- [Data Flow](docs/rag_system.md#data-flow)
- [Integration](docs/rag_system.md#integration-with-agents)
- [Performance](docs/rag_system.md#performance-optimization)

### Agent Communication
- [Protocol](docs/agent_communication.md#communication-protocol)
- [Message Types](docs/agent_communication.md#message-types)
- [Interaction Patterns](docs/agent_communication.md#communication-patterns)
- [Error Handling](docs/agent_communication.md#error-handling)

### UI Interaction
- [Components](docs/ui_interaction.md#components)
- [Features](docs/ui_interaction.md#features)
- [Best Practices](docs/ui_interaction.md#best-practices)
- [Configuration](docs/ui_interaction.md#configuration)

## System Requirements

### Minimum Requirements
- CPU: 4 cores
- RAM: 8GB
- Storage: 20GB
- Python 3.8+
- Node.js 14+
- Redis 6+
- PostgreSQL 12+

### Recommended
- CPU: 8+ cores
- RAM: 16GB+
- Storage: 50GB+
- SSD storage
- High-speed internet connection

## Development

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/test_rag_system.py
pytest tests/test_agents.py
```

### Code Quality
```bash
# Run linter
flake8 .

# Run type checking
mypy .

# Run security checks
bandit -r .
```

### Documentation
```bash
# Generate API documentation
cd docs
sphinx-build -b html source build
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Support

- [Issue Tracker](https://github.com/yourusername/ai_village/issues)
- [Documentation](docs/)
- [Community Forum](https://forum.aivillage.dev)

## Acknowledgments

- OpenAI for language models
- HuggingFace for transformers
- FAISS for vector search
- NetworkX for graph operations

## Project Status

Current version: 1.0.0
Status: Active Development

See [CHANGELOG.md](CHANGELOG.md) for version history.
