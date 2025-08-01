# Entry Point Mapping Document

## Current main.py Files Analysis

### 1. `main.py` (Root)
- **Size**: 5,514 bytes
- **Purpose**: Primary application entry point
- **Functionality**: CLI interface, service orchestration
- **Dependencies**: Various core modules

### 2. `agent_forge/main.py`
- **Size**: 5,098 bytes
- **Purpose**: Agent Forge framework entry point
- **Functionality**: Agent creation, training, deployment
- **Dependencies**: Agent Forge core modules

### 3. `agent_forge/core/main.py`
- **Size**: 664 bytes
- **Purpose**: Core Agent Forge utilities
- **Functionality**: Basic initialization and imports
- **Dependencies**: Minimal core dependencies

### 4. `agents/king/main.py`
- **Size**: 7,481 bytes
- **Purpose**: KING agent system entry point
- **Functionality**: Advanced agent orchestration, task management
- **Dependencies**: KING agent modules, analytics

### 5. `rag_system/main.py`
- **Size**: 6,448 bytes
- **Purpose**: RAG system entry point
- **Functionality**: Retrieval-augmented generation pipeline
- **Dependencies**: RAG system components

## Proposed Unified Entry Point Structure

### Root Entry Point: `main.py`
Unified CLI interface with mode selection:
- `--mode agent-forge` → Agent Forge operations
- `--mode king` → KING agent operations
- `--mode rag` → RAG system operations
- `--mode core` → Core utilities

### Service-Specific Entry Points
- `src/agent_forge/main.py` → Agent Forge specific
- `src/agents/king/main.py` → KING agent specific
- `src/rag_system/main.py` → RAG system specific
- `src/core/main.py` → Core utilities specific

### CLI Structure
```bash
# Unified usage
python main.py --mode agent-forge --action train --config config.yaml
python main.py --mode king --action run --task "analyze data"
python main.py --mode rag --action query --question "What is..."

# Direct service usage
python -m agent_forge.main --action train
python -m agents.king.main --action run
```

## Migration Plan

### Phase 1: Backup Current Files
- [ ] Create `legacy_mains/` directory
- [ ] Backup all current main.py files
- [ ] Document current functionality

### Phase 2: Create Unified Structure
- [ ] Design unified CLI interface
- [ ] Create service-specific entry points
- [ ] Implement mode selection

### Phase 3: Update References
- [ ] Update import statements
- [ ] Update documentation
- [ ] Update deployment scripts

### Phase 4: Testing & Validation
- [ ] Test all entry points
- [ ] Validate backward compatibility
- [ ] Update usage examples
