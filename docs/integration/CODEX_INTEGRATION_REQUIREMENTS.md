# CODEX Integration Requirements

## Overview
This document specifies all requirements for integrating CODEX-built components with the existing AIVillage codebase.

## Environment Variables

### Evolution Metrics System
```bash
# Database Configuration
AIVILLAGE_DB_PATH=/path/to/evolution_metrics.db
AIVILLAGE_STORAGE_BACKEND=sqlite  # Options: sqlite, redis, file
AIVILLAGE_REDIS_URL=redis://localhost:6379/0

# Metrics Collection
AIVILLAGE_METRICS_FLUSH_THRESHOLD=50
AIVILLAGE_METRICS_FILE=evolution_metrics.json
AIVILLAGE_LOG_DIR=./evolution_logs

# Optional: Redis for distributed metrics
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
```

### RAG Pipeline System
```bash
# Cache Configuration
RAG_CACHE_ENABLED=true
RAG_L1_CACHE_SIZE=128
RAG_REDIS_URL=redis://localhost:6379/1
RAG_DISK_CACHE_DIR=/tmp/rag_disk_cache

# Embedding Model
RAG_EMBEDDING_MODEL=paraphrase-MiniLM-L3-v2
RAG_CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-2-v2

# Vector Store
RAG_VECTOR_DIM=384
RAG_FAISS_INDEX_PATH=/path/to/faiss_index
RAG_BM25_CORPUS_PATH=/path/to/bm25_corpus

# Query Processing
RAG_DEFAULT_K=10
RAG_CHUNK_SIZE=512
RAG_CHUNK_OVERLAP=50
```

### P2P Networking
```bash
# LibP2P Configuration
LIBP2P_HOST=0.0.0.0
LIBP2P_PORT=4001
LIBP2P_PEER_ID_FILE=/path/to/peer_id
LIBP2P_PRIVATE_KEY_FILE=/path/to/private_key

# mDNS Discovery
MDNS_SERVICE_NAME=_aivillage._tcp
MDNS_DISCOVERY_INTERVAL=30
MDNS_TTL=120

# Mesh Network
MESH_MAX_PEERS=50
MESH_HEARTBEAT_INTERVAL=10
MESH_CONNECTION_TIMEOUT=30

# Fallback Transports
MESH_ENABLE_BLUETOOTH=true
MESH_ENABLE_WIFI_DIRECT=true
MESH_ENABLE_FILE_TRANSPORT=true
MESH_FILE_TRANSPORT_DIR=/tmp/aivillage_mesh
```

### Digital Twin System
```bash
# Encryption
DIGITAL_TWIN_ENCRYPTION_KEY=base64-encoded-32-byte-key
DIGITAL_TWIN_VAULT_PATH=/secure/path/to/vault

# Database
DIGITAL_TWIN_DB_PATH=/path/to/digital_twin.db
DIGITAL_TWIN_SQLITE_WAL=true

# Privacy Settings
DIGITAL_TWIN_COPPA_COMPLIANT=true
DIGITAL_TWIN_FERPA_COMPLIANT=true
DIGITAL_TWIN_GDPR_COMPLIANT=true

# Personalization
DIGITAL_TWIN_MAX_PROFILES=10000
DIGITAL_TWIN_PROFILE_TTL_DAYS=365
```

## Port Configurations

### Required Ports
| Service | Port | Protocol | Purpose |
|---------|------|----------|---------|
| LibP2P Main | 4001 | TCP/UDP | Primary P2P communication |
| LibP2P WebSocket | 4002 | TCP | WebSocket transport |
| mDNS Discovery | 5353 | UDP | Multicast DNS discovery |
| Digital Twin API | 8080 | HTTP | REST API endpoints |
| Evolution Metrics | 8081 | HTTP | Metrics collection API |
| RAG Pipeline | 8082 | HTTP | Query processing API |
| Redis (Optional) | 6379 | TCP | Caching and pub/sub |

### Port Ranges for Mobile
```
Android P2P: 4000-4010
iOS P2P: 4010-4020
WiFi Direct: 4020-4030
Bluetooth: Dynamic allocation
```

## Database Connections

### SQLite Databases
1. **Evolution Metrics DB** (`evolution_metrics.db`)
   - Tables: `evolution_rounds`, `fitness_metrics`, `resource_metrics`, `selection_outcomes`
   - WAL mode enabled for concurrent access
   - Automatic schema migration

2. **Digital Twin DB** (`digital_twin.db`)
   - Tables: `learning_profiles`, `learning_sessions`, `knowledge_states`
   - Encrypted sensitive data fields
   - GDPR compliance features

3. **RAG Index DB** (`rag_index.db`)
   - Tables: `documents`, `chunks`, `embeddings_metadata`
   - FAISS index storage metadata

### Redis Connections (Optional)
- **Database 0**: Evolution metrics real-time data
- **Database 1**: RAG pipeline caching
- **Database 2**: P2P peer discovery cache

## External Service Dependencies

### Required Services
1. **FAISS Vector Index**
   - Version: ≥1.7.0
   - GPU support optional but recommended
   - Memory requirements: ~2GB for 100K documents

2. **SentenceTransformers**
   - Models: `paraphrase-MiniLM-L3-v2` (required)
   - Models: `cross-encoder/ms-marco-MiniLM-L-2-v2` (optional)
   - GPU acceleration supported

3. **LibP2P Python Bindings**
   - Package: `py-libp2p`
   - Version: ≥0.2.0
   - System dependencies: `libp2p-dev`

### Optional Services
1. **Redis Server**
   - Version: ≥6.0
   - Persistence: RDB + AOF recommended
   - Memory: 512MB minimum

2. **Weights & Biases**
   - For evolution metrics tracking
   - API key required: `WANDB_API_KEY`

## Configuration Files

### Main Configuration (`aivillage_config.yaml`)
```yaml
# CODEX Integration Configuration
integration:
  evolution_metrics:
    enabled: true
    backend: sqlite
    db_path: ./data/evolution_metrics.db
    flush_interval_seconds: 30

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

  digital_twin:
    enabled: true
    encryption_enabled: true
    privacy_mode: strict
    max_profiles: 10000
```

### P2P Network Config (`p2p_config.json`)
```json
{
  "host": "0.0.0.0",
  "port": 4001,
  "peer_discovery": {
    "mdns_enabled": true,
    "bootstrap_peers": [],
    "discovery_interval": 30
  },
  "transports": {
    "tcp_enabled": true,
    "websocket_enabled": true,
    "bluetooth_enabled": false,
    "wifi_direct_enabled": false
  },
  "security": {
    "tls_enabled": true,
    "peer_verification": true
  }
}
```

### RAG Pipeline Config (`rag_config.json`)
```json
{
  "embedder": {
    "model_name": "paraphrase-MiniLM-L3-v2",
    "device": "cpu",
    "batch_size": 32
  },
  "retrieval": {
    "vector_top_k": 20,
    "keyword_top_k": 20,
    "final_top_k": 10,
    "rerank_enabled": false
  },
  "cache": {
    "l1_size": 128,
    "l2_enabled": false,
    "l3_directory": "/tmp/rag_cache"
  }
}
```

## Integration Checklist

### Pre-Integration Requirements
- [ ] Python ≥3.8 installed
- [ ] SQLite ≥3.35 available
- [ ] Required Python packages installed (see requirements.txt)
- [ ] Environment variables configured
- [ ] Port availability verified
- [ ] Directory permissions set

### Post-Integration Verification
- [ ] All databases created successfully
- [ ] Evolution metrics collection working
- [ ] RAG pipeline processing documents
- [ ] P2P peer discovery functional
- [ ] Digital Twin API responding
- [ ] Integration tests passing

### Performance Tuning
- [ ] Database indexes created
- [ ] Cache hit rates optimized
- [ ] Memory usage within limits
- [ ] Network latency acceptable
- [ ] Error rates below thresholds

## Troubleshooting

### Common Issues
1. **Port Conflicts**: Use `netstat -an | grep :4001` to check port availability
2. **Database Locks**: Ensure WAL mode is enabled for SQLite
3. **Memory Issues**: Monitor RAM usage, especially for FAISS indexes
4. **Network Discovery**: Check firewall settings for UDP multicast

### Debug Mode
```bash
export AIVILLAGE_LOG_LEVEL=DEBUG
export AIVILLAGE_DEBUG_MODE=true
export AIVILLAGE_PROFILE_PERFORMANCE=true
```

### Health Check Endpoints
- Evolution Metrics: `GET /health/evolution`
- RAG Pipeline: `GET /health/rag`
- P2P Network: `GET /health/p2p`
- Digital Twin: `GET /health/twin`

## Security Considerations

### Data Encryption
- Digital Twin profiles encrypted at rest
- P2P messages encrypted in transit
- API endpoints require authentication tokens

### Privacy Compliance
- COPPA compliance for users under 13
- FERPA compliance for educational records
- GDPR compliance for EU users
- Data retention policies enforced

### Network Security
- TLS 1.3 for all HTTP traffic
- mTLS for P2P communications
- Rate limiting on API endpoints
- Input validation on all interfaces

## Migration Notes

### From Existing Systems
1. **Evolution Metrics**: Migrate from JSON files to database
2. **RAG System**: Replace SHA256 embeddings with real vectors
3. **P2P Network**: Migrate from mock Bluetooth to LibP2P
4. **Agents**: Update to use new RAG and evolution interfaces

### Backward Compatibility
- Legacy APIs maintained during transition
- Data format converters available
- Gradual migration path documented
- Rollback procedures defined

---
*Last Updated: August 8, 2025*
*Document Version: 1.0*
