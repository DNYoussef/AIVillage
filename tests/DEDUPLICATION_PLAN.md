# Test Deduplication Plan

## Identified Redundancies

### 1. Database Validation (4 files → 2 files)
- **KEEP**: `tests/validation/databases/validate_database_integrity.py` (868 lines, comprehensive)
- **KEEP**: `tests/validation/databases/validate_databases_simple.py` (452 lines, simplified version)
- **REMOVE**: `tests/validation/system/validate_database_integrity.py` (exact duplicate)
- **REMOVE**: `tests/validation/system/validate_databases_simple.py` (exact duplicate)

### 2. Compression Tests (20 files → 4 files)
**KEEP THE BEST**:
- `tests/unit/test_compression_comprehensive.py` - Most complete test suite
- `tests/unit/test_seedlm_core.py` - Core SeedLM implementation tests
- `tests/unit/test_stage1_compression.py` - Stage 1 specific tests
- `tests/unit/test_cpu_only_compression.py` - Mobile/CPU optimization tests

**REMOVE REDUNDANT**:
- `test_compression.py` - Basic duplicate of comprehensive
- `test_compression_integration.py` - Covered by comprehensive
- `test_compression_only.py` - Subset of comprehensive
- `test_compression_pipeline.py` - Duplicate pipeline tests
- `test_compression_real.py` - Covered by comprehensive
- `test_advanced_compression.py` - Features in comprehensive
- `test_seedlm.py` - Duplicate of seedlm_core
- `test_seedlm_fast.py` - Quick subset of core
- `test_seedlm_simple.py` - Simplified subset
- `test_stage1.py` - Duplicate of stage1_compression
- `test_stage1_minimal.py` - Minimal subset
- `test_stage1_simple.py` - Simple subset
- `test_stage2_compression.py` - Keep one copy (has duplicate)
- `test_unified_compression.py` - Covered by comprehensive

### 3. Agent Tests (Multiple variants → Consolidated)
**KEEP**:
- `tests/unit/test_agent_specialization.py` - Comprehensive agent tests
- `tests/integration/test_agent_system_integration.py` - System integration

**REMOVE**:
- `test_specialized_agents.py` - Duplicate of agent_specialization
- `test_validate_all_agents.py` - Subset of specialization tests

### 4. King Agent Tests (4 files → 1 file)
**KEEP**:
- `tests/unit/test_king_agent.py` - Most comprehensive

**REMOVE**:
- `test_king_agent_simple.py` - Simplified subset
- `test_king_agent.py.backup` - Backup file
- `test_king_agent_simple.py.backup` - Backup file

### 5. Coordination Tests (3 files → 1 file)
**KEEP**:
- `tests/unit/test_coordination_system.py` - Most complete

**REMOVE**:
- `test_coordination_system_working.py` - Working draft
- `test_coord_simple.py` - Simplified subset

### 6. Dashboard Tests (3 files → 1 file)
**KEEP**:
- `tests/unit/test_dashboard_generator.py` - Primary implementation

**REMOVE**:
- `test_coverage_dashboard.py` - Specific subset
- In root: `test_dashboard.py` - Old version

### 7. RAG Tests (10+ files → 3 files)
**KEEP**:
- `tests/unit/test_rag_comprehensive_integration.py` - Complete suite
- `tests/unit/test_rag_offline_config.py` - Offline-specific
- `tests/integration/test_rag_pipeline.py` - Pipeline integration

**REMOVE**:
- `test_rag_basic.py` - Basic subset
- `test_rag_simple.py` - Simple subset
- `test_rag_comprehensive_simple.py` - Simplified comprehensive
- `test_rag_system_integration.py` - Duplicate integration
- `test_rag_factory_integration.py` - Factory pattern subset
- `test_rag_edge_integration.py` - Edge case subset
- `test_rag_p2p_integration.py` - P2P subset

### 8. Mesh Network Tests (4 files → 1 file)
**KEEP**:
- `tests/experimental/mesh/test_mesh_network_comprehensive.py` - Most complete

**REMOVE**:
- `test_mesh_integration.py` - Integration subset
- `test_mesh_simple.py` - Simple subset
- `tests/unit/test_mesh_network_comprehensive.py` - Duplicate

### 9. Evolution Tests (Multiple → 2 files)
**KEEP**:
- `tests/evolution/test_evolution_comprehensive.py` - Complete suite
- `tests/integration/test_evolution_metrics_integration.py` - Metrics integration

**REMOVE**:
- `test_evolution_system.py` - System subset
- `test_evolution_constraints.py` - Constraints subset
- `test_evolution_metrics_persistence.py` - Persistence subset
- `test_corrected_evolution.py` - Corrected version duplicate

### 10. Pipeline Tests (8+ files → 2 files)  
**KEEP**:
- `tests/unit/test_unified_pipeline.py` - Unified implementation
- `tests/unit/test_pipeline_integration.py` - Integration tests

**REMOVE**:
- `test_pipeline_simple.py` - Simple subset
- `test_pipeline_improvements.py` - Improvements subset
- `test_advanced_pipeline.py` - Advanced subset
- `test_training_pipeline.py` - Training subset
- `test_unified_pipeline_resume.py` - Resume feature subset

## Summary Statistics
- **Before**: ~350+ test files with significant redundancy
- **After**: ~120 test files with unique functionality
- **Reduction**: ~65% fewer test files
- **Benefit**: Faster test execution, easier maintenance, clearer purpose