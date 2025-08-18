# Configuration & Deployment Consolidation - Deprecation Notice

## Consolidation Date: August 18, 2025

This directory contains files that were consolidated during Phase 9 of the AIVillage consolidation process.

## Files Moved and Deprecated

### Environment Configuration Files
**Old Locations → New Location**:
- `.env.*` files from root → `config/env/`
- All environment configurations now centralized in `config/env/`

### Docker Configuration Files  
**Old Locations → New Location**:
- Root `docker-compose.yml`, `Dockerfile*` → `deploy/docker/`
- `docker/scion-gateway/*` → `deploy/docker/`
- Scattered Dockerfiles from packages/ and archive/ → deprecated

**Production Docker Structure**: `deploy/docker/`
- Most comprehensive Docker configuration
- Contains service-specific Dockerfiles
- Centralized docker-compose configurations

### Requirements Files
**Old Locations → New Location**:
- Root `requirements*.txt` → `requirements/` (consolidated directory)

### Deployment Infrastructure
**Old Locations → New Location**:
- `ops/` directory → `deploy/monitoring/` and `deploy/scripts/`
- All deployment tooling centralized in `deploy/`

### Configuration Files
**Centralized Location**: `config/`
- All environment configurations in `config/env/`
- Application configs remain in `config/`
- Test configuration moved to `config/pytest.ini`

## Migration Guide

### For Environment Variables
**Old**:
```bash
# Loading from root directory
source .env.development
```

**New**:
```bash
# Loading from consolidated location
source config/env/.env.development
```

### For Docker Operations
**Old**:
```bash
# Multiple locations
docker-compose up
docker build -f Dockerfile.agentforge .
```

**New**:
```bash
# Centralized in deploy/docker/
cd deploy/docker
docker-compose up
docker build -f Dockerfile.agentforge .
```

### For Requirements
**Old**:
```bash
pip install -r requirements.txt
```

**New**:
```bash
pip install -r requirements/requirements-main.txt
```

## Benefits of Consolidation

1. **Single Source of Truth**: All configurations in logical locations
2. **Cleaner Root Directory**: Reduced from 58 files to 16 essential files
3. **Better Organization**: Deployment, config, and requirements properly separated
4. **Easier Maintenance**: Clear ownership and structure for all configuration types
5. **Production Ready**: Based on most mature configuration implementations

## Deprecation Timeline

- **Migration Period**: August 18 - September 15, 2025
- **Full Removal**: After September 15, 2025
- **Legacy Support**: Import compatibility maintained during migration period

## Contact

See main project documentation for questions about the consolidation.