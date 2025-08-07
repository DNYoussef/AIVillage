# Changelog

## [Unreleased]
### Added
- Added `digital_twin` module, increasing core module count to 17.

## [1.0.0] - 2025-07-27
### Added
- **Sprint 2 Complete**: Production/experimental separation with quality gates
- Production components: compression, evolution, RAG, memory, benchmarking, geometry
- Comprehensive test suites with 70%+ coverage requirement
- Quality gate enforcement via CI/CD and pre-commit hooks
- Sprint 2 migration guide and completion documentation
- Graceful dependency handling with fallback imports

### Changed
- **Major reorganization**: Code moved to production/, experimental/, deprecated/ structure
- Updated all import paths for new organization
- README.md now reflects realistic project status vs promotional claims
- All documentation aligned with actual implementation status

### Removed
- Archived misleading "historic success" promotional documents
- Removed temporary sync and cleanup reports
- Consolidated redundant analysis documents into Sprint 2 final report
- Eliminated 13,000+ stub functions through proper organization

### Fixed
- Python syntax errors in agent_forge/results_analyzer.py
- Pytest collection errors with proper import handling
- Production module exports with graceful fallbacks
- YAML syntax issues in GitHub workflow files
- Import path corrections throughout codebase

### Security
- Import separation enforcement: no experimental code in production
- Automated quality gate checking in CI/CD
- Security scanning with bandit for production code

## [0.5.1] - 2025-07-11
### Changed
- Simplified branching strategy to use a single `main` branch for feature and hotfix work.

## [0.5.0] - 2025-07-15
### Added
- Confidence scoring with tiers
- Conformal calibration utilities
- Explanation endpoints
- Qdrant development Terraform and migration script
