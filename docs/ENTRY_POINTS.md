# AIVillage Entry Points Guide

## Overview

This document describes the new unified entry point structure for the AIVillage platform, providing a consistent CLI interface across all services.

## Unified Entry Point

The main entry point is `main.py` in the project root, which provides a unified CLI interface for all services.

### Usage

```bash
python main.py --mode MODE --action ACTION [OPTIONS]
```

### Available Modes

| Mode | Description | Example |
|------|-------------|---------|
| `agent-forge` | Agent creation and management | `python main.py --mode agent-forge --action train --config config.yaml` |
| `king` | KING agent system operations | `python main.py --mode king --action run --task "analyze data"` |
| `rag` | Retrieval-augmented generation | `python main.py --mode rag --action query --question "What is AI?"` |
| `core` | Core utilities and configuration | `python main.py --mode core --action status` |

### Common Options

- `--config, -c`: Configuration file path
- `--verbose, -v`: Enable verbose output
- `--debug`: Enable debug mode
- `--input`: Input file or directory
- `--output`: Output file or directory

## Service-Specific Entry Points

Each service also has its own dedicated entry point for direct usage:

### Agent Forge Service
```bash
python agent_forge/main.py ACTION [OPTIONS]
```

**Actions:**
- `train`: Train an agent
- `create`: Create a new agent
- `list`: List all agents
- `delete`: Delete an agent
- `status`: Get service status

**Options:**
- `--agent-type`: Type of agent (king, sage, magi, base)
- `--name`: Agent name
- `--config`: Configuration file

### KING Agent Service
```bash
python agents/king/main.py ACTION [OPTIONS]
```

**Actions:**
- `run`: Run a task
- `plan`: Plan a task
- `analyze`: Analyze data
- `status`: Get service status
- `config`: Configure service

**Options:**
- `--task`: Task description
- `--config`: Configuration file

### RAG System Service
```bash
python rag_system/main.py ACTION [OPTIONS]
```

**Actions:**
- `query`: Query the system
- `index`: Index a document
- `search`: Search documents
- `status`: Get service status
- `config`: Configure service

**Options:**
- `--question`: Question to query
- `--document`: Document to index
- `--config`: Configuration file

## Examples

### Using Unified Entry Point

```bash
# Train an agent
python main.py --mode agent-forge --action train --config configs/agent.yaml

# Run a KING agent task
python main.py --mode king --action run --task "Analyze customer feedback"

# Query the RAG system
python main.py --mode rag --action query --question "What are the latest trends in AI?"

# Get system status
python main.py --mode core --action status
```

### Using Service-Specific Entry Points

```bash
# Direct Agent Forge usage
python agent_forge/main.py train --agent-type king --name my_agent

# Direct KING agent usage
python agents/king/main.py run --task "Process dataset"

# Direct RAG system usage
python rag_system/main.py query --question "Explain machine learning"
```

## Migration from Old Structure

### Old Usage (Legacy)
```bash
# These files are now in legacy_mains/
python legacy_mains/main.py.agent_forge
python legacy_mains/main.py.king
python legacy_mains/main.py.rag
```

### New Usage (Recommended)
```bash
# Unified interface
python main.py --mode agent-forge --action train
python main.py --mode king --action run
python main.py --mode rag --action query

# Or service-specific
python agent_forge/main.py train
python agents/king/main.py run
python rag_system/main.py query
```

## Testing

Run the entry point tests to verify everything works:

```bash
python tests/test_entry_points.py
```

## Configuration

Each service can be configured using YAML configuration files. See the respective service directories for configuration examples.

## Troubleshooting

### Common Issues

1. **Module not found errors**: Ensure you're running from the project root
2. **Import errors**: Check that all dependencies are installed
3. **Configuration errors**: Verify configuration file paths

### Debug Mode

Enable debug mode for detailed output:
```bash
python main.py --mode MODE --action ACTION --debug
```

## Development

### Adding New Services

To add a new service mode:

1. Create a new service directory
2. Add a `main.py` file with the service entry point
3. Update the unified main.py to include the new mode
4. Add tests for the new service

### Extending CLI

To add new actions or options:

1. Update the service-specific main.py
2. Update the unified main.py if needed
3. Update this documentation
4. Add tests for new functionality
