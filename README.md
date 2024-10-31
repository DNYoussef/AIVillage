# AI Village

AI Village is a sophisticated multi-agent system that combines strategic planning, research synthesis, and code generation capabilities through specialized AI agents working in harmony.

## Core Agents

### King Agent
- Strategic planning and decision making
- Resource allocation and optimization
- Complex problem decomposition
- Uses nvidia/llama-3.1-nemotron-70b-instruct (frontier) and Qwen/Qwen2.5-3B-Instruct (local)

### Sage Agent
- Research and knowledge synthesis
- Pattern recognition
- Evidence-based analysis
- Uses anthropic/claude-3.5-sonnet (frontier) and deepseek-ai/Janus-1.3B (local)

### Magi Agent
- Code generation and experimentation
- Technical implementation
- Performance optimization
- Uses openai/o1-mini-2024-09-12 (frontier) and ibm-granite/granite-3b-code-instruct-128k (local)

## System Architecture

The AI Village system is built on several key components:

1. **RAG System**
   - Enhanced knowledge retrieval
   - Cognitive nexus for information synthesis
   - Vector and graph-based storage

2. **Communication System**
   - Inter-agent messaging
   - Task coordination
   - Community hub for agent interaction

3. **Core Systems**
   - Unified configuration management
   - Data collection and processing
   - Performance monitoring
   - Task management

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Git
- 16GB RAM minimum (32GB recommended)
- CUDA-compatible GPU recommended

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai_village.git
cd ai_village
```

2. Run the setup script:
```bash
python download_ai_village.py
```

3. Set up your OpenRouter API key:
   - Create a `.env` file in the root directory
   - Add your API key: `OPENROUTER_API_KEY=your_api_key_here`

4. Initialize the system:
```bash
python initialize_village.py
```

### Configuration

The system uses several configuration files in the `config/` directory:

- `default.yaml`: System-wide settings
- `rag_config.yaml`: RAG system configuration
- `openrouter_agents.yaml`: Agent-specific settings

## Directory Structure

```
ai_village/
├── agent_forge/          # Core agent functionality
├── agents/              # Agent implementations
├── communications/      # Communication system
├── config/             # Configuration files
├── data/               # Data storage
├── docs/               # Documentation
├── logs/               # System logs
├── rag_system/         # RAG system components
├── tests/              # Test suite
└── utils/              # Utility functions
```

## Usage

1. Start the AI Village system:
```bash
python initialize_village.py
```

2. The system will initialize all agents and components, setting up:
   - Communication channels
   - RAG system
   - Data storage
   - Monitoring systems

3. Monitor the system through logs in the `logs/` directory

## Development

### Running Tests
```bash
pytest tests/
```

### Adding New Agents

1. Create a new agent class inheriting from UnifiedBaseAgent
2. Implement required methods:
   - `_process_task`
   - `generate_response`
3. Add configuration in `config/openrouter_agents.yaml`

### Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Code style
- Pull request process
- Development workflow

## Performance Monitoring

The system includes comprehensive monitoring:

- Agent performance metrics
- System resource usage
- Task completion rates
- Model performance comparison

Access monitoring data through:
- Log files in `logs/`
- Performance metrics API
- System status endpoints

## Troubleshooting

Common issues and solutions:

1. **Initialization Failures**
   - Check OpenRouter API key
   - Verify system requirements
   - Check log files

2. **Memory Issues**
   - Adjust batch sizes in configuration
   - Monitor resource usage
   - Consider upgrading RAM

3. **Model Loading Issues**
   - Verify model downloads
   - Check CUDA installation
   - Ensure sufficient disk space

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenRouter for API access
- HuggingFace for model implementations
- Open source community for various components

## Support

For support:
1. Check the documentation in `docs/`
2. Review troubleshooting guides
3. Open an issue on GitHub
4. Contact the development team

## Roadmap

Future development plans:

1. Enhanced agent capabilities
2. Additional specialized agents
3. Improved performance optimization
4. Extended RAG system features
5. Advanced monitoring tools

## Security

The system implements several security measures:

- API key management
- Input validation
- Rate limiting
- Error handling

Report security issues through GitHub's security advisory system.
