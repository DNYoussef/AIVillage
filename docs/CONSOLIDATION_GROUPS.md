# AIVillage Consolidation Groups & Unification Strategy

## Executive Summary
This document groups all redundant implementations and provides specific Claude prompts for consolidation. The codebase has **70-80% redundancy** with the same features implemented 3-10 times across different locations.

---

## GROUP 1: RAG SYSTEMS (10+ Implementations)
**CRITICAL PRIORITY - Most fragmented component**

### All RAG Locations to Consolidate:
```
1. src/production/rag/rag_system/           [Most complete - 50+ files]
2. src/software/hyper_rag/                  [Hyper RAG variant]
3. py/aivillage/rag/                        [Python package]
4. packages/rag/                            [Modular package]
5. python/aivillage/hyperrag/               [Alt implementation]
6. experimental/hyperrag/                   [Experimental features]
7. src/mcp_servers/hyperag/                 [MCP server variant]
8. deprecated/backup_20250813/experimental_rag/
9. deprecated/backup_20250813/rag_system/
10. src/core/knowledge/rag_offline_config.py
```

### Claude Consolidation Prompt:
```
"Analyze and unify all RAG implementations from these directories:
- src/production/rag/rag_system/ (use as base)
- src/software/hyper_rag/
- py/aivillage/rag/
- experimental/hyperrag/
- src/mcp_servers/hyperag/

Create a single unified RAG system in packages/rag/ that:
1. Takes the best features from each implementation
2. Preserves Bayesian trust graphs from production
3. Includes Hyper RAG's cognitive nexus
4. Maintains offline capabilities
5. Integrates MCP server features
Delete all redundant implementations after unification."
```

---

## GROUP 2: AGENT FORGE & TRAINING (8+ Implementations)

### All Agent Forge Locations:
```
1. src/agent_forge/                         [Main implementation - 100+ files]
2. src/software/agent_forge/                [Duplicate]
3. py/aivillage/agent_forge/                [Python package]
4. experimental/agent_forge_experimental/   [Experimental]
5. src/production/agent_forge/              [Production variant]
6. evomerge/                                [Evolution merge]
7. deprecated/legacy/legacy_mains/main.py.agent_forge*
```

### Training Components:
```
1. src/agent_forge/training/                [Main training]
2. src/agent_forge/curriculum/              [Curriculum learning]
3. src/agent_forge/evolution/               [Evolution system]
4. src/agent_forge/adas/                    [ADAS implementation]
5. src/agent_forge/compression/             [Compression]
6. src/agent_forge/quiet_star/              [Quiet-STaR]
7. experimental/training/                   [Experimental training]
```

### Claude Consolidation Prompt:
```
"Unify all Agent Forge implementations:
PRIMARY: src/agent_forge/ (use as base)
MERGE IN:
- src/production/agent_forge/evolution/
- experimental/agent_forge_experimental/
- py/aivillage/agent_forge/
- evomerge/

Create unified system in packages/agent_forge/ with:
1. Single training pipeline incorporating all phases
2. Unified evolution system (combine evomerge + evolution/)
3. Consolidated ADAS implementation
4. Single curriculum system
5. Integrated compression pipeline
Delete all redundant implementations."
```

---

## GROUP 3: SPECIALIZED AGENTS (15+ Locations)

### All Agent Implementations:
```
1. agents/atlantis_meta_agents/             [18 specialized agents]
2. src/agents/                              [Core agents]
3. src/production/agents/                   [Production agents]
4. experimental/agents/agents/              [Experimental agents]
   - king/
   - magi/
   - sage/
   - navigator/
5. src/software/meta_agents/                [Meta agents]
6. py/aivillage/agents/                     [Python agents]
7. packages/agents/                         [Package agents]
```

### Claude Consolidation Prompt:
```
"Consolidate all agent implementations:
BASE: agents/atlantis_meta_agents/ (most complete)
MERGE:
- experimental/agents/agents/ (has enhanced versions)
- src/production/agents/
- src/software/meta_agents/

Create unified in packages/agents/ with:
1. Single implementation per agent type (King, Magi, Sage, etc.)
2. Preserve best features from each version
3. Unified agent interface
4. Consolidated task management
5. Single coordination system
Remove all duplicates."
```

---

## GROUP 4: P2P/COMMUNICATION SYSTEMS (12+ Implementations)

### BitChat Implementations:
```
1. src/core/p2p/bitchat_*                   [Core BitChat]
2. py/aivillage/p2p/bitchat_bridge.py       [Python bridge]
3. platforms/mobile/android/.../BitChatService.kt
4. platforms/mobile/ios/.../BitChatManager.swift
5. crates/bitchat-cla/                      [Rust implementation]
6. src/federation/protocols/bitchat_enhanced.py
```

### BetaNet Implementations:
```
1. src/core/p2p/betanet_*                   [7 files]
2. py/aivillage/p2p/betanet/                [Python package]
3. crates/betanet-*/                        [8 Rust crates]
4. platforms/rust/betanet/                  [Rust platform]
5. archive/consolidated_communications/old_python_p2p/betanet_*
```

### General P2P:
```
1. src/communications/                      [Main comms]
2. src/core/p2p/                           [Core P2P]
3. packages/p2p/                           [Package P2P]
4. experimental/mesh/                      [Mesh networking]
```

### Claude Consolidation Prompt:
```
"Unify all P2P/communication systems:
CREATE: packages/p2p/ with subdirectories:
- packages/p2p/bitchat/ (all BitChat variants)
- packages/p2p/betanet/ (all BetaNet variants)
- packages/p2p/libp2p/ (LibP2P implementations)
- packages/p2p/core/ (shared components)

CONSOLIDATE:
1. Single BitChat protocol implementation
2. Single BetaNet transport layer
3. Unified mesh networking
4. Single transport manager
5. Consolidated discovery mechanisms
Delete all redundant implementations."
```

---

## GROUP 5: COMPRESSION SYSTEMS (6+ Implementations)

### All Compression Locations:
```
1. src/compression/                         [Core compression]
2. src/core/compression/                    [Core module]
3. src/production/compression/              [Production]
4. src/agent_forge/compression/             [Agent Forge variant]
5. tests/compression/                       [Test implementations]
6. src/production/compression/compression/  [Nested duplicate]
```

### Claude Consolidation Prompt:
```
"Unify compression implementations:
BASE: src/production/compression/ (most complete)
MERGE: src/agent_forge/compression/ (has BitNet, SeedLM)
TARGET: packages/compression/

Include:
1. BitNet implementation
2. SeedLM compression
3. VPTQ quantization
4. Unified pipeline
5. Mobile optimizations
Remove all duplicates."
```

---

## GROUP 6: TESTING (Scattered Everywhere)

### Test Locations:
```
1. tests/                                   [Main tests - 200+ files]
2. src/*/tests/                            [Embedded tests]
3. experimental/*/tests/                   [Experimental tests]
4. stress_tests/                           [Stress tests]
5. benchmarks/                             [Benchmarks]
6. tests/manual/                           [Manual tests]
7. tests/compression/retired/              [Old tests]
8. deprecated/*/tests/                    [Deprecated tests]
```

### Claude Consolidation Prompt:
```
"Consolidate all tests into tests/ directory:
STRUCTURE:
- tests/unit/ (all unit tests)
- tests/integration/ (all integration tests)
- tests/e2e/ (end-to-end tests)
- tests/benchmarks/ (performance tests)
- tests/fixtures/ (shared fixtures)

ACTION:
1. Move all src/*/tests/ to tests/unit/
2. Consolidate duplicate test cases
3. Update all imports
4. Remove test files from source directories
5. Delete deprecated test directories"
```

---

## GROUP 7: MOBILE/PLATFORM CODE

### Mobile Implementations:
```
1. platforms/mobile/android/               [Android]
2. platforms/mobile/ios/                   [iOS]
3. deprecated/mobile_archive/              [Old mobile]
4. AIVillageEducation/src/                [Education app]
5. clients/mobile/                        [Mobile clients]
6. src/android/                           [Android source]
```

### Claude Consolidation Prompt:
```
"Consolidate mobile implementations:
TARGET: platforms/mobile/
- platforms/mobile/android/ (unified Android)
- platforms/mobile/ios/ (unified iOS)
- platforms/mobile/shared/ (shared code)

MERGE:
1. AIVillageEducation features
2. BitChat mobile implementations
3. P2P mobile bridges
Delete deprecated mobile code."
```

---

## GROUP 8: CONFIGURATION & DEPLOYMENT

### Configuration Files:
```
1. config/                                 [Main configs]
2. src/*/config.py                        [Scattered configs]
3. deploy/                                [Deployment]
4. ops/                                   [Operations]
5. docker/                                [Docker configs]
6. infra/                                 [Infrastructure]
7. k8s/ files                            [Kubernetes]
```

### Claude Consolidation Prompt:
```
"Consolidate configuration and deployment:
TARGET:
- config/ (all configuration files)
- deploy/ (all deployment scripts)

ACTION:
1. Move all config.py files to config/
2. Consolidate Docker files to deploy/docker/
3. Move k8s configs to deploy/k8s/
4. Unify environment variables
5. Single deployment pipeline"
```

---

## GROUP 9: EVOLUTION & TRAINING SYSTEMS

### Evolution Implementations:
```
1. src/production/agent_forge/evolution/   [Production evolution]
2. src/agent_forge/evolution/              [Agent Forge evolution]
3. evomerge/                               [EvoMerge system]
4. scripts/run_evolution_merge.py         [Evolution scripts]
5. scripts/run_50gen_evolution.py         [50-gen evolution]
```

### Claude Consolidation Prompt:
```
"Unify evolution systems:
BASE: src/production/agent_forge/evolution/
MERGE: evomerge/ features
TARGET: packages/evolution/

Create single evolution pipeline with:
1. Resource-constrained evolution
2. 50-generation support
3. EvoMerge integration
4. Unified metrics
5. Single orchestrator"
```

---

## GROUP 10: DOCUMENTATION & REPORTS

### Documentation Chaos:
```
1. docs/                                  [Main docs]
2. deprecated/old_reports/                [50+ old reports]
3. *.md files in root                    [30+ markdown files]
4. deprecated/backup_*/                  [Old documentation]
```

### Claude Consolidation Prompt:
```
"Organize documentation:
TARGET: docs/
- docs/api/ (API documentation)
- docs/guides/ (user guides)
- docs/architecture/ (architecture docs)
- docs/archive/ (historical docs)

ACTION:
1. Move all root *.md files to appropriate docs/ subdirectory
2. Archive old reports to docs/archive/
3. Delete duplicate documentation
4. Create single README.md"
```

---

## EXECUTION STRATEGY

### Phase 1: Critical Systems (Week 1)
1. **RAG System** - Highest redundancy (10+ implementations)
2. **Agent Forge** - Core training infrastructure
3. **Testing** - Centralize all tests

### Phase 2: Core Components (Week 2)
4. **P2P/Communications** - Unify transport layers
5. **Specialized Agents** - Single implementation per agent
6. **Compression** - Unified pipeline

### Phase 3: Infrastructure (Week 3)
7. **Configuration** - Single config system
8. **Deployment** - Unified deployment
9. **Evolution** - Single evolution pipeline

### Phase 4: Cleanup (Week 4)
10. **Documentation** - Organize all docs
11. **Delete deprecated/** - Remove 30% of codebase
12. **Delete archives/** - Remove old code

---

## SPECIFIC CLAUDE PROMPTS FOR EACH GROUP

### Prompt Template:
```
"I need you to consolidate [COMPONENT] implementations:

ANALYZE these directories:
[List all source directories]

IDENTIFY:
1. Core functionality in each implementation
2. Unique features worth preserving
3. Duplicate code to eliminate
4. Best implementation to use as base

CREATE unified version in: [TARGET_DIRECTORY]

PRESERVE:
- All unique features
- Best practices from each
- Working tests
- Production-ready code

DELETE:
- All redundant implementations
- Deprecated versions
- Duplicate tests

UPDATE:
- All imports to new location
- Configuration references
- Documentation

Generate a migration script that handles all import updates automatically."
```

---

## METRICS TO TRACK

### Before Consolidation:
- **Total Files**: 5,000+
- **Duplicate Implementations**: 31+ components
- **Redundant Code**: ~50% of codebase
- **Test Files**: 200+ scattered
- **Config Files**: 50+ locations

### After Consolidation Target:
- **Total Files**: <2,000
- **Duplicate Implementations**: 0
- **Redundant Code**: 0%
- **Test Files**: All in tests/
- **Config Files**: All in config/

---

## VALIDATION CHECKLIST

After each consolidation:
- [ ] All tests pass
- [ ] No broken imports
- [ ] Features preserved
- [ ] Performance maintained
- [ ] Documentation updated
- [ ] Old code deleted
- [ ] Git history preserved

---

*This consolidation will reduce codebase by 60-70% while preserving all functionality*
