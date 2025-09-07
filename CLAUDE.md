# Claude Code Configuration - SPARC Development Environment

## CRITICAL: CONCURRENT EXECUTION & FILE MANAGEMENT

**ABSOLUTE RULES**:
1. ALL operations MUST be concurrent/parallel in a single message
2. **NEVER save working files, text/mds and tests to the root folder**
3. ALWAYS organize files in appropriate subdirectories
4. **USE CLAUDE CODE'S TASK TOOL** for spawning agents concurrently, not just MCP
5. **NO UNICODE** - Never use Unicode characters, emojis, or special symbols in any output or files

### GOLDEN RULE: "1 MESSAGE = ALL RELATED OPERATIONS"

**MANDATORY PATTERNS:**
- **TodoWrite**: ALWAYS batch ALL todos in ONE call (5-10+ todos minimum)
- **Task tool (Claude Code)**: ALWAYS spawn ALL agents in ONE message with full instructions
- **File operations**: ALWAYS batch ALL reads/writes/edits in ONE message
- **Bash commands**: ALWAYS batch ALL terminal operations in ONE message
- **Memory operations**: ALWAYS batch ALL memory store/retrieve in ONE message

### CRITICAL: Claude Code Task Tool for Agent Execution

**Claude Code's Task tool is the PRIMARY way to spawn agents:**
```javascript
// CORRECT: Use Claude Code's Task tool for parallel agent execution
[Single Message]:
  Task("Research agent", "Analyze requirements and patterns...", "researcher")
  Task("Coder agent", "Implement core features...", "coder")
  Task("Tester agent", "Create comprehensive tests...", "tester")
  Task("Reviewer agent", "Review code quality...", "reviewer")
  Task("Architect agent", "Design system architecture...", "system-architect")
```

**MCP tools are ONLY for coordination setup:**
- `mcp__claude-flow__swarm_init` - Initialize coordination topology
- `mcp__claude-flow__agent_spawn` - Define agent types for coordination
- `mcp__claude-flow__task_orchestrate` - Orchestrate high-level workflows

### File Organization Rules

**NEVER save to root folder. Use these directories:**
- `/src` - Source code files
- `/tests` - Test files
- `/docs` - Documentation and markdown files
- `/config` - Configuration files
- `/scripts` - Utility scripts
- `/examples` - Example code

## Project Overview

This project uses SPARC (Specification, Pseudocode, Architecture, Refinement, Completion) methodology with Claude-Flow orchestration for systematic Test-Driven Development.

## SPARC Commands

### Core Commands
- `npx claude-flow sparc modes` - List available modes
- `npx claude-flow sparc run <mode> "<task>"` - Execute specific mode
- `npx claude-flow sparc tdd "<feature>"` - Run complete TDD workflow
- `npx claude-flow sparc info <mode>` - Get mode details

### Batchtools Commands
- `npx claude-flow sparc batch <modes> "<task>"` - Parallel execution
- `npx claude-flow sparc pipeline "<task>"` - Full pipeline processing
- `npx claude-flow sparc concurrent <mode> "<tasks-file>"` - Multi-task processing

### Build Commands
- `npm run build` - Build project
- `npm run test` - Run tests
- `npm run lint` - Linting
- `npm run typecheck` - Type checking

## SPARC Workflow Phases

1. **Specification** - Requirements analysis (`sparc run spec-pseudocode`)
2. **Pseudocode** - Algorithm design (`sparc run spec-pseudocode`)
3. **Architecture** - System design (`sparc run architect`)
4. **Refinement** - TDD implementation (`sparc tdd`)
5. **Completion** - Integration (`sparc run integration`)

## Code Style & Best Practices

- **Modular Design**: Files under 500 lines
- **Environment Safety**: Never hardcode secrets
- **Test-First**: Write tests before implementation
- **Clean Architecture**: Separate concerns
- **Documentation**: Keep updated

## Available Agents (54 Total)

### Core Development
`coder`, `reviewer`, `tester`, `planner`, `researcher`

### Swarm Coordination
`hierarchical-coordinator`, `mesh-coordinator`, `adaptive-coordinator`, `collective-intelligence-coordinator`, `swarm-memory-manager`

### Consensus & Distributed
`byzantine-coordinator`, `raft-manager`, `gossip-coordinator`, `consensus-builder`, `crdt-synchronizer`, `quorum-manager`, `security-manager`

### Performance & Optimization
`perf-analyzer`, `performance-benchmarker`, `task-orchestrator`, `memory-coordinator`, `smart-agent`

### GitHub & Repository
`github-modes`, `pr-manager`, `code-review-swarm`, `issue-tracker`, `release-manager`, `workflow-automation`, `project-board-sync`, `repo-architect`, `multi-repo-swarm`

### SPARC Methodology
`sparc-coord`, `sparc-coder`, `specification`, `pseudocode`, `architecture`, `refinement`

### Specialized Development
`backend-dev`, `mobile-dev`, `ml-developer`, `cicd-engineer`, `api-docs`, `system-architect`, `code-analyzer`, `base-template-generator`

### Testing & Validation
`tdd-london-swarm`, `production-validator`

### Migration & Planning
`migration-planner`, `swarm-init`

## 🧠 Gemini CLI Integration - Large Context Analysis

### Dual-AI Decision Matrix

**Use Gemini CLI for**:
- Large codebase analysis (>100KB files or >50 files)
- Implementation verification across entire projects
- Architectural understanding and pattern detection
- Multi-directory comparative analysis
- Research with Google Search grounding
- Security vulnerability scanning across codebase
- Feature implementation validation

**Use Claude Code for**:
- File editing and implementation
- MCP agent coordination and swarm orchestration
- SPARC methodology execution
- Concurrent task execution
- Git operations and project management
- TodoWrite and task tracking

### Gemini CLI Command Patterns and File Inclusion Syntax

**Use the `@` syntax to include files and directories in your Gemini prompts. The paths should be relative to WHERE you run the gemini command:**

#### Basic File and Directory Analysis:
```bash
# Single file analysis
gemini -p "@src/main.py Explain this file's purpose and structure"

# Multiple files
gemini -p "@package.json @src/index.js Analyze the dependencies used in the code"

# Entire directory
gemini -p "@src/ Summarize the architecture of this codebase"

# Multiple directories
gemini -p "@src/ @tests/ Analyze test coverage for the source code"

# Current directory and subdirectories
gemini -p "@./ Give me an overview of this entire project"

# All files flag (alternative)
gemini --all_files -p "Analyze the project structure and dependencies"
```

#### Implementation Verification Examples:
```bash
# Check if a feature is implemented
gemini -p "@src/ @lib/ Has dark mode been implemented in this codebase? Show me the relevant files and functions"

# Verify authentication implementation
gemini -p "@src/ @middleware/ Is JWT authentication implemented? List all auth-related endpoints and middleware"

# Check for specific patterns
gemini -p "@src/ Are there any React hooks that handle WebSocket connections? List them with file paths"

# Verify error handling
gemini -p "@src/ @api/ Is proper error handling implemented for all API endpoints? Show examples of try-catch blocks"

# Check for rate limiting
gemini -p "@backend/ @middleware/ Is rate limiting implemented for the API? Show the implementation details"

# Verify caching strategy
gemini -p "@src/ @lib/ @services/ Is Redis caching implemented? List all cache-related functions and their usage"

# Check for specific security measures
gemini -p "@src/ @api/ Are SQL injection protections implemented? Show how user inputs are sanitized"

# Verify test coverage for features
gemini -p "@src/payment/ @tests/ Is the payment processing module fully tested? List all test cases"
```

#### Architecture and Pattern Analysis:
```bash
# Architecture Analysis
gemini -p "@./ Analyze entire project architecture and dependencies"

# Implementation Verification
gemini -p "@src/ @tests/ Verify [feature] implementation with test coverage"

# Security Analysis
gemini --all_files -p "Scan for security vulnerabilities and implementation gaps"

# Multi-directory Analysis
gemini -p "@backend/ @frontend/ @shared/ Check API consistency across layers"

# Pattern Detection
gemini -p "@./ Find all instances of [pattern] and suggest improvements"
```

### When to Use Gemini CLI

**Use `gemini -p` when:**
- Analyzing entire codebases or large directories
- Comparing multiple large files
- Need to understand project-wide patterns or architecture
- Current context window is insufficient for the task
- Working with files totaling more than 100KB
- Verifying if specific features, patterns, or security measures are implemented
- Checking for the presence of certain coding patterns across the entire codebase

### Important Notes for Gemini CLI Usage

- **Paths in `@` syntax are relative to your current working directory** when invoking gemini
- **The CLI will include file contents directly in the context**
- **No need for --yolo flag for read-only analysis**
- **Gemini's context window can handle entire codebases** that would overflow Claude's context
- **When checking implementations, be specific** about what you're looking for to get accurate results
- **Authentication required**: Set `GEMINI_API_KEY` environment variable
- **Rate limits apply**: Free tier has quota restrictions, use `gemini-2.5-flash` for higher limits

### 🔄 Dual-AI Workflow Patterns

#### Three-Phase Integration Pattern

**Phase 1: Analysis (Gemini CLI)**
```bash
gemini -p "@./ Analyze codebase for [specific requirement or issue]"
```

**Phase 2: Implementation (Claude Code + MCP)**
```javascript
[Single Message - ALL operations]:
  Task("Implement based on Gemini analysis", analysis_results, "coder")
  Task("Create comprehensive tests", test_requirements, "tester")
  Task("Review security implications", security_analysis, "reviewer")
  mcp__claude-flow__task_orchestrate { task: "coordinate_implementation" }
  TodoWrite({ todos: [...8-10 todos based on Gemini analysis...] })
  Write("multiple files based on analysis...")
```

**Phase 3: Verification (Gemini CLI)**
```bash
gemini -p "@./ @tests/ Verify implementation meets requirements and standards"
```

## Claude Code vs MCP Tools vs Gemini CLI

### Claude Code Handles ALL EXECUTION:
- **Task tool**: Spawn and run agents concurrently for actual work
- File operations (Read, Write, Edit, MultiEdit, Glob, Grep)
- Code generation and programming
- Bash commands and system operations
- Implementation work
- Project navigation and analysis
- TodoWrite and task management
- Git operations
- Package management
- Testing and debugging

### MCP Tools ONLY COORDINATE:
- Swarm initialization (topology setup)
- Agent type definitions (coordination patterns)
- Task orchestration (high-level planning)
- Memory management
- Neural features
- Performance tracking
- GitHub integration

### Gemini CLI Handles LARGE-SCALE ANALYSIS:
- Massive context window analysis (1M tokens)
- Google Search grounding for real-time information
- Implementation verification across entire codebases
- Architectural pattern detection
- Security vulnerability scanning
- Cross-directory comparative analysis

**KEY**: MCP coordinates strategy, Claude Code executes with real agents, Gemini CLI provides large-context analysis and verification.

## Quick Setup

```bash
# Add Claude Flow MCP server
claude mcp add claude-flow npx claude-flow@alpha mcp start
```

## MCP Servers (Enhanced Capabilities) - ALL FUNCTIONAL

### CRITICAL: Working MCP Servers (Verified):
1. **Memory** - Persistent memory and knowledge graph (.mcp/memory.db) ✅ WORKING
2. **Sequential Thinking** - Multi-step reasoning and planning ✅ WORKING
3. **HypeRAG** - Local knowledge graph and memory server ✅ WORKING

### Configuration Status:
- Memory MCP: `@modelcontextprotocol/server-memory@2025.4.25` - FUNCTIONAL
- Sequential Thinking MCP: `@modelcontextprotocol/server-sequential-thinking@2025.7.1` - FUNCTIONAL
- HypeRAG MCP: Local Python server - FUNCTIONAL
- Markdown MCP: DISABLED (package unavailable)

### Additional Available MCP Servers (Comprehensive):
4. **GitHub MCP** - Repository management with 29+ tools for issues, PRs, CI/CD, security analysis
   - Tools: Repository ops, issue management, PR operations, GitHub Actions, security scanning
   - Auth: OAuth/PAT with remote server at https://api.githubcopilot.com/mcp/
   - Use for: Repository intelligence, automated workflows, cross-repo operations
   - Rate: 5K requests/hour, supports dynamic toolsets

5. **HuggingFace MCP** - ML operations with unified multi-provider interface
   - Tools: Model inference, embeddings, multi-modal processing, OpenAI compatibility
   - Performance: 25K tokens/response, 164+ concurrent clients
   - Providers: fal, Replicate, Sambanova, Together AI integration
   - Use for: RAG systems, code assistants, multi-modal content, research tools

6. **MarkItDown MCP** - Document format conversion to LLM-friendly Markdown
   - Formats: PDF, DOCX, PPTX, XLSX, HTML, images (OCR), audio, CSV/JSON/XML
   - Features: Structure preservation, semantic meaning focus, multi-format pipelines
   - Use for: Knowledge base construction, document ingestion, automated workflows
   - Limitations: PDF processing requires OCR, formatting loss in complex docs

7. **DeepWiki MCP** - GitHub repository documentation access (NOT Wikipedia)
   - Tools: read_wiki_structure, read_wiki_contents, ask_question
   - Scope: GitHub repositories only, no general knowledge or Wikipedia access
   - Use for: Rapid codebase understanding, documentation discovery, technical Q&A
   - Endpoint: https://mcp.deepwiki.com/sse

8. **Context7 MCP** - Real-time documentation retrieval (NOT distributed caching)
   - Tools: resolve-library-id, get-library-docs
   - Features: Version-specific docs, multi-platform integration, real-time injection
   - Use for: Multi-agent development teams, documentation-driven architecture
   - Limitations: No traditional caching, requires complementary memory systems

9. **Apify MCP** - Web scraping and browser automation platform
   - Actors: 5,000+ available through dynamic tool discovery
   - Features: JavaScript execution, form automation, structured data extraction
   - Rate: 30 requests/second, ethical scraping built-in, proxy rotation
   - Use for: Competitive intelligence, content monitoring, QA automation

10. **Firecrawl MCP** - Deep web crawling optimized for AI/LLM workflows
    - Performance: 50x faster than traditional scrapers, sub-second responses
    - Features: JavaScript rendering, smart page settlement, AI-powered research
    - Coverage: 96% of web content, intelligent content filtering
    - Tools: FIRECRAWL_SCRAPE_EXTRACT_DATA_LLM, FIRECRAWL_CRAWL_URLS, FIRECRAWL_BATCH_SCRAPE

### MCP Tool Categories

#### Coordination & Orchestration
`swarm_init`, `agent_spawn`, `task_orchestrate`

#### Monitoring & Metrics
`swarm_status`, `agent_list`, `agent_metrics`, `task_status`, `task_results`

#### Memory & Neural
`memory_usage`, `neural_status`, `neural_train`, `neural_patterns`

#### GitHub Integration
`github_swarm`, `repo_analyze`, `pr_enhance`, `issue_triage`, `code_review`

#### System & Performance
`benchmark_run`, `features_detect`, `swarm_monitor`

#### External MCP Servers (Enhanced Integration)
- **GitHub MCP**: 29+ tools for repo management, issues, PRs, CI/CD, security
- **HuggingFace MCP**: ML inference, embeddings, multi-modal, 164+ concurrent clients
- **Memory MCP**: Cross-session persistence with SQLite storage and knowledge graphs
- **Sequential Thinking MCP**: Multi-step reasoning with branching and revision capabilities
- **Firecrawl MCP**: 50x faster web crawling, LLM-optimized, JavaScript rendering
- **MarkItDown MCP**: Document conversion, PDF/DOCX/HTML to Markdown, multi-format
- **Context7 MCP**: Real-time documentation retrieval, version-specific, multi-platform
- **DeepWiki MCP**: GitHub repository documentation access, technical Q&A
- **Apify MCP**: 5,000+ web scraping actors, browser automation, ethical scraping

## Agent Execution Flow with Claude Code

### The Correct Pattern:

1. **Optional**: Use MCP tools to set up coordination topology
2. **REQUIRED**: Use Claude Code's Task tool to spawn agents that do actual work
3. **REQUIRED**: Each agent runs hooks for coordination
4. **REQUIRED**: Batch all operations in single messages
5. **NEW**: Leverage MCP servers for enhanced capabilities

### Example Full-Stack Development:

```javascript
// Single message with all agent spawning via Claude Code's Task tool
[Parallel Agent Execution]:
  Task("Backend Developer", "Build REST API with Express. Use hooks for coordination.", "backend-dev")
  Task("Frontend Developer", "Create React UI. Coordinate with backend via memory.", "coder")
  Task("Database Architect", "Design PostgreSQL schema. Store schema in memory.", "code-analyzer")
  Task("Test Engineer", "Write Jest tests. Check memory for API contracts.", "tester")
  Task("DevOps Engineer", "Setup Docker and CI/CD. Document in memory.", "cicd-engineer")
  Task("Security Auditor", "Review authentication. Report findings via hooks.", "reviewer")
  
  // All todos batched together
  TodoWrite { todos: [...8-10 todos...] }
  
  // All file operations together
  Write "backend/server.js"
  Write "frontend/App.jsx"
  Write "database/schema.sql"
```

## Enhanced Agent Coordination Protocol

### CRITICAL: Enhanced Agent Spawning with Memory, Sequential Thinking, and DSPy

**ALWAYS use this enhanced pattern when spawning agents:**

```javascript
// 1. Initialize Enhanced Coordination
import { EnhancedAgentCoordinator } from './src/coordination/enhanced_agent_coordinator.py'
coordinator = new EnhancedAgentCoordinator()

// 2. Spawn ALL agents with enhanced capabilities in ONE message
[Single Message with ALL agents]:
  Task("Research agent with memory and sequential thinking", 
       coordinator.generate_enhanced_agent_prompt("researcher", "task description", 
       ["context_key1", "context_key2"], "Analyze -> Research -> Synthesize -> Validate"), 
       "researcher")
  Task("Coder agent with DSPy optimization", 
       coordinator.generate_enhanced_agent_prompt("coder", "implementation task",
       ["research_results", "architecture"], "Plan -> Implement -> Test -> Review"), 
       "coder")
  Task("System architect with cross-agent memory", 
       coordinator.generate_enhanced_agent_prompt("system-architect", "design task",
       ["requirements", "constraints"], "Analyze -> Design -> Validate -> Document"), 
       "system-architect")
```

### Enhanced Agent Requirements - MANDATORY FOR ALL AGENTS:

**1️⃣ MEMORY INTEGRATION (REQUIRED):**
```bash
# Check shared memory BEFORE starting
memory_key = "[agent_type]_context_[session_id]"
# Store ALL significant findings 
# Update coordination state regularly
```

**2️⃣ SEQUENTIAL THINKING (REQUIRED):**
```bash
# Follow step-by-step reasoning chain
# Document decision rationale
# Validate each step before proceeding
# Store reasoning process for DSPy learning
```

**3️⃣ DSPY OPTIMIZATION (REQUIRED):**
```bash
# Provide structured, measurable outputs
# Include confidence scores (0-1) for decisions
# Store performance metrics for learning
# Use optimized prompts when available
```

**4️⃣ COORDINATION HOOKS (REQUIRED):**
```bash
npx claude-flow@alpha hooks pre-task --description "[task]" --session-id "[session]"
npx claude-flow@alpha hooks post-edit --file "[file]" --memory-key "[agent]_progress_[session]"
npx claude-flow@alpha hooks notify --message "[progress]"
npx claude-flow@alpha hooks post-task --task-id "[agent]_[session]"
npx claude-flow@alpha hooks session-end --export-metrics true
```

## Concurrent Execution Examples

### CORRECT WORKFLOW: MCP Coordinates, Claude Code Executes

```javascript
// Step 1: MCP tools set up coordination (optional, for complex tasks)
[Single Message - Coordination Setup]:
  mcp__claude-flow__swarm_init { topology: "mesh", maxAgents: 6 }
  mcp__claude-flow__agent_spawn { type: "researcher" }
  mcp__claude-flow__agent_spawn { type: "coder" }
  mcp__claude-flow__agent_spawn { type: "tester" }

// Step 2: Claude Code Task tool spawns ACTUAL agents that do the work
[Single Message - Parallel Agent Execution]:
  // Claude Code's Task tool spawns real agents concurrently
  Task("Research agent", "Analyze API requirements and best practices. Check memory for prior decisions.", "researcher")
  Task("Coder agent", "Implement REST endpoints with authentication. Coordinate via hooks.", "coder")
  Task("Database agent", "Design and implement database schema. Store decisions in memory.", "code-analyzer")
  Task("Tester agent", "Create comprehensive test suite with 90% coverage.", "tester")
  Task("Reviewer agent", "Review code quality and security. Document findings.", "reviewer")
  
  // Batch ALL todos in ONE call
  TodoWrite { todos: [
    {id: "1", content: "Research API patterns", status: "in_progress", priority: "high"},
    {id: "2", content: "Design database schema", status: "in_progress", priority: "high"},
    {id: "3", content: "Implement authentication", status: "pending", priority: "high"},
    {id: "4", content: "Build REST endpoints", status: "pending", priority: "high"},
    {id: "5", content: "Write unit tests", status: "pending", priority: "medium"},
    {id: "6", content: "Integration tests", status: "pending", priority: "medium"},
    {id: "7", content: "API documentation", status: "pending", priority: "low"},
    {id: "8", content: "Performance optimization", status: "pending", priority: "low"}
  ]}
  
  // Parallel file operations
  Bash "mkdir -p app/{src,tests,docs,config}"
  Write "app/package.json"
  Write "app/src/server.js"
  Write "app/tests/server.test.js"
  Write "app/docs/API.md"
```

### WRONG (Multiple Messages):
```javascript
Message 1: mcp__claude-flow__swarm_init
Message 2: Task("agent 1")
Message 3: TodoWrite { todos: [single todo] }
Message 4: Write "file.js"
// This breaks parallel coordination!
```

## Performance Benefits

- **84.8% SWE-Bench solve rate**
- **32.3% token reduction**
- **2.8-4.4x speed improvement**
- **27+ neural models**

## Hooks Integration

### Pre-Operation
- Auto-assign agents by file type
- Validate commands for safety
- Prepare resources automatically
- Optimize topology by complexity
- Cache searches

### Post-Operation
- Auto-format code
- Train neural patterns
- Update memory
- Analyze performance
- Track token usage

### Session Management
- Generate summaries
- Persist state
- Track metrics
- Restore context
- Export workflows

## Advanced Features (v2.1.0)

- Automatic Topology Selection
- Parallel Execution (2.8-4.4x speed)
- Neural Training with DSPy Optimization
- Bottleneck Analysis
- Smart Auto-Spawning with MCP Server Selection
- Self-Healing Workflows
- Enhanced Cross-Session Memory (Memory MCP)
- GitHub Integration (GitHub MCP)
- ML Model Access (HuggingFace MCP)
- Deep Web Research (Firecrawl + DeepWiki MCP)
- DSPy Prompt Optimization
- Sequential Thinking Chains

## DSPy Prompting System

### How to Use DSPy Correctly:

```python
# Initialize DSPy with MCP servers
from .claude.dspy_integration.enhanced_dspy_mcp import DSPyMCPSystem

system = DSPyMCPSystem()
await system.initialize(['memory', 'sequentialthinking', 'github'])

# Chain of Thought Reasoning
reasoning, answer = await system.reason("Complex architectural question")

# Code Generation with Research
code_result = await system.code("Implement OAuth2 authentication")

# Knowledge Synthesis
knowledge = await system.research("Best practices for microservices")
```

### DSPy Automatic Enhancements:
- **Sequential Thinking**: Breaks down complex problems step-by-step
- **Memory Integration**: Retrieves historical context and patterns
- **Research Phase**: Gathers information from GitHub, DeepWiki, HuggingFace
- **Prompt Optimization**: Uses examples to improve prompt quality
- **Parallel Processing**: Executes MCP server calls concurrently

### DSPy Best Practices:
1. Always initialize required MCP servers first
2. Let DSPy handle context management automatically
3. Use memory server for persistent learning
4. Add examples to improve optimization
5. Leverage server-specific capabilities

## Integration Tips

1. Start with basic swarm init
2. Scale agents gradually
3. Use memory for context
4. Monitor progress regularly
5. Train patterns from success
6. Enable hooks automation
7. Use GitHub tools first
8. Initialize MCP servers based on task requirements
9. Use DSPy for complex reasoning tasks

## MCP Server Selection Guide

### Task-Based Server Recommendations:

**For Repository Work**:
- Primary: GitHub MCP + Context7 MCP + Memory MCP
- GitHub MCP: Repository management, issues, PRs, CI/CD operations
- Context7 MCP: Real-time documentation access for current library versions
- Memory MCP: Persistent learning from repository patterns and decisions

**For ML/AI Development**:
- Primary: HuggingFace MCP + Sequential Thinking MCP + Memory MCP
- HuggingFace MCP: Model inference, embeddings, multi-modal processing
- Sequential Thinking MCP: Complex reasoning chains for AI architecture decisions
- Memory MCP: Cross-session learning from ML experiments and results

**For Research & Content**:
- Primary: Firecrawl MCP + MarkItDown MCP + DeepWiki MCP + Memory MCP
- Firecrawl MCP: Deep web crawling optimized for AI workflows (50x faster)
- MarkItDown MCP: Convert documents to LLM-friendly Markdown
- DeepWiki MCP: GitHub repository documentation analysis
- Memory MCP: Persistent research findings and patterns

**For Web Automation**:
- Primary: Apify MCP + Firecrawl MCP + Sequential Thinking MCP
- Apify MCP: 5,000+ web scraping actors with ethical automation
- Firecrawl MCP: Modern JavaScript-heavy sites with smart page settlement
- Sequential Thinking MCP: Multi-step reasoning for complex automation workflows

**For Comprehensive Analysis**:
- All servers coordinated through Memory MCP as central hub
- Sequential Thinking MCP orchestrates complex multi-system reasoning
- Task-specific servers provide specialized capabilities

### Performance Characteristics:

**High-Speed Servers**: Firecrawl, Memory, Context7 (sub-second to millisecond responses)
**High-Concurrency**: HuggingFace (164+ clients), Sequential Thinking, Memory
**Rate-Limited**: GitHub (5K req/hour), Apify (30 req/sec) - plan accordingly
**Processing-Intensive**: MarkItDown, HuggingFace - consider resource allocation

## Support

- Documentation: https://github.com/ruvnet/claude-flow
- Issues: https://github.com/ruvnet/claude-flow/issues

---

Remember: **Claude Flow coordinates, Claude Code creates!**

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
Never save working files, text/mds and tests to the root folder.
