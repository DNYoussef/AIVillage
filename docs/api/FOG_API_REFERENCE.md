# Enhanced Fog Computing API Reference

## Overview

AIVillage's Enhanced Fog Computing Platform provides 32+ REST API endpoints across 8 advanced security components. This comprehensive API reference covers all endpoints with detailed schemas, authentication requirements, and usage examples.

## Base URL and Authentication

**Base URL**: `http://localhost:8000` (default development)
**Production URL**: `https://your-domain.com`

**Authentication**: JWT Bearer Token
```bash
# Authenticate and get token
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "secure_password"}'

# Use token in requests
curl -X GET "http://localhost:8000/v1/fog/system/health" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

## System Overview Endpoints

### GET `/`
**Description**: System overview with all available fog endpoints  
**Authentication**: None required  
**Response**:
```json
{
  "name": "Enhanced Fog Computing Platform",
  "version": "3.0.0", 
  "description": "Privacy-first fog cloud with 8 advanced security layers",
  "data": {
    "fog_endpoints": {
      "tee_runtime": "/v1/fog/tee/",
      "cryptographic_proofs": "/v1/fog/proofs/",
      "zero_knowledge": "/v1/fog/zk/",
      "market_pricing": "/v1/fog/pricing/",
      "job_scheduler": "/v1/fog/scheduler/",
      "quorum_manager": "/v1/fog/quorum/",
      "onion_routing": "/v1/fog/onion/",
      "reputation_system": "/v1/fog/reputation/",
      "vrf_topology": "/v1/fog/vrf/"
    }
  }
}
```

### GET `/health`
**Description**: Comprehensive system health check  
**Authentication**: None required  
**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-08-28T10:30:00Z",
  "services": {
    "tee_runtime": {"status": "running", "uptime": 3600},
    "proof_system": {"status": "running", "uptime": 3600},
    "zk_predicates": {"status": "running", "uptime": 3600},
    "market_engine": {"status": "running", "uptime": 3600},
    "scheduler": {"status": "running", "uptime": 3600},
    "quorum_manager": {"status": "running", "uptime": 3600},
    "onion_router": {"status": "running", "uptime": 3600},
    "reputation": {"status": "running", "uptime": 3600},
    "vrf_topology": {"status": "running", "uptime": 3600}
  }
}
```

## 1. TEE Runtime System (`/v1/fog/tee/`)

### GET `/v1/fog/tee/status`
**Description**: Get TEE system status and capabilities  
**Authentication**: Required  
**Response**:
```json
{
  "success": true,
  "data": {
    "available_tees": [
      {
        "type": "amd_sev_snp",
        "available": true,
        "version": "1.51",
        "max_memory_mb": 16384,
        "max_enclaves": 16
      },
      {
        "type": "intel_tdx", 
        "available": false,
        "reason": "Hardware not detected"
      }
    ],
    "active_enclaves": 3,
    "total_capacity": 16384
  }
}
```

### POST `/v1/fog/tee/create-enclave`
**Description**: Create new secure enclave  
**Authentication**: Required  
**Request Body**:
```json
{
  "name": "ml_training_enclave",
  "description": "Secure ML model training environment",
  "memory_mb": 2048,
  "cpu_cores": 4,
  "code_hash": "sha256:abc123...",
  "config": {
    "measurement_policy": "strict",
    "network_isolation": true,
    "allow_debug": false
  }
}
```
**Response**:
```json
{
  "success": true,
  "data": {
    "enclave_id": "enclave_a1b2c3d4e5f6",
    "state": "created",
    "tee_type": "amd_sev_snp",
    "created_at": "2025-08-28T10:30:00Z",
    "estimated_attestation_time": "500ms"
  }
}
```

### POST `/v1/fog/tee/attest`
**Description**: Generate attestation report for enclave  
**Authentication**: Required  
**Request Body**:
```json
{
  "enclave_id": "enclave_a1b2c3d4e5f6",
  "nonce": "optional_nonce_bytes",
  "include_certificate_chain": true
}
```
**Response**:
```json
{
  "success": true,
  "data": {
    "report_id": "att_1234567890abcdef",
    "enclave_id": "enclave_a1b2c3d4e5f6",
    "tee_type": "amd_sev_snp",
    "timestamp": "2025-08-28T10:30:00Z",
    "measurements": [
      {
        "type": "mrenclave",
        "index": 0,
        "value": "sha256:def456...",
        "algorithm": "sha256",
        "description": "Code measurement"
      }
    ],
    "quote": "hex_encoded_hardware_quote",
    "certificate_chain": ["cert1_hex", "cert2_hex"],
    "status": "verified"
  }
}
```

### GET `/v1/fog/tee/metrics`
**Description**: Get enclave performance metrics  
**Authentication**: Required  
**Query Parameters**:
- `enclave_id` (optional): Specific enclave ID
- `timerange` (optional): 1h, 24h, 7d (default: 1h)

**Response**:
```json
{
  "success": true,
  "data": {
    "enclaves": [
      {
        "enclave_id": "enclave_a1b2c3d4e5f6",
        "cpu_usage_percent": 45.2,
        "memory_usage_mb": 1024,
        "memory_usage_percent": 50.0,
        "io_read_bytes": 1048576,
        "io_write_bytes": 524288,
        "tasks_completed": 142,
        "task_success_rate": 98.6,
        "uptime_seconds": 3600
      }
    ],
    "system_totals": {
      "total_cpu_usage": 45.2,
      "total_memory_usage": 3072,
      "total_tasks_completed": 456
    }
  }
}
```

## 2. Cryptographic Proof System (`/v1/fog/proofs/`)

### POST `/v1/fog/proofs/generate`
**Description**: Generate cryptographic proof for computation  
**Authentication**: Required  
**Request Body**:
```json
{
  "proof_type": "proof_of_execution",
  "enclave_id": "enclave_a1b2c3d4e5f6",
  "task_id": "task_98765",
  "input_hash": "sha256:input123...",
  "computation_log": "execution_trace_data",
  "include_blockchain_anchor": true
}
```
**Response**:
```json
{
  "success": true,
  "data": {
    "proof_id": "prf_1a2b3c4d5e6f",
    "proof_type": "proof_of_execution",
    "enclave_id": "enclave_a1b2c3d4e5f6",
    "timestamp": "2025-08-28T10:30:00Z",
    "proof_data": {
      "merkle_root": "abc123...",
      "signature": "def456...",
      "witness": "ghi789..."
    },
    "blockchain_anchor": {
      "tx_hash": "0x123abc...",
      "block_number": 18234567,
      "confirmation_time": "2025-08-28T10:32:00Z"
    },
    "verification_time_ms": 250
  }
}
```

### POST `/v1/fog/proofs/verify`
**Description**: Verify proof validity  
**Authentication**: Required  
**Request Body**:
```json
{
  "proof_id": "prf_1a2b3c4d5e6f",
  "proof_data": {
    "merkle_root": "abc123...",
    "signature": "def456...", 
    "witness": "ghi789..."
  },
  "expected_input_hash": "sha256:input123..."
}
```
**Response**:
```json
{
  "success": true,
  "data": {
    "valid": true,
    "proof_id": "prf_1a2b3c4d5e6f",
    "verification_time_ms": 180,
    "details": {
      "signature_valid": true,
      "merkle_path_valid": true,
      "input_hash_matches": true,
      "blockchain_confirmed": true
    }
  }
}
```

### GET `/v1/fog/proofs/batch`
**Description**: Batch proof operations  
**Authentication**: Required  
**Query Parameters**:
- `operation`: generate|verify|anchor
- `proof_ids`: Comma-separated proof IDs
- `batch_size`: Maximum batch size (default: 10)

**Response**:
```json
{
  "success": true,
  "data": {
    "batch_id": "batch_789xyz",
    "operation": "verify",
    "results": [
      {"proof_id": "prf_1", "valid": true, "time_ms": 150},
      {"proof_id": "prf_2", "valid": true, "time_ms": 142}
    ],
    "total_time_ms": 292,
    "success_rate": 100.0
  }
}
```

### POST `/v1/fog/proofs/anchor`
**Description**: Anchor proofs to blockchain  
**Authentication**: Required  
**Request Body**:
```json
{
  "proof_ids": ["prf_1a2b3c4d5e6f", "prf_2b3c4d5e6f7"],
  "blockchain": "ethereum",
  "priority": "high"
}
```
**Response**:
```json
{
  "success": true,
  "data": {
    "anchor_id": "anc_xyz789",
    "proof_ids": ["prf_1a2b3c4d5e6f", "prf_2b3c4d5e6f7"],
    "blockchain": "ethereum",
    "tx_hash": "0x456def...",
    "estimated_confirmation": "2025-08-28T10:35:00Z",
    "cost_wei": 50000000000000000
  }
}
```

## 3. Zero-Knowledge Predicates (`/v1/fog/zk/`)

### POST `/v1/fog/zk/verify`
**Description**: Verify zero-knowledge predicate  
**Authentication**: Required  
**Request Body**:
```json
{
  "predicate_type": "network_policy_compliance",
  "proof": "zk_proof_data_hex",
  "public_inputs": ["policy_hash", "network_id"],
  "verification_key": "vk_hex_data"
}
```
**Response**:
```json
{
  "success": true,
  "data": {
    "verification_id": "zkv_abc123def456",
    "predicate_type": "network_policy_compliance",
    "result": true,
    "confidence": 1.0,
    "verification_time_ms": 850,
    "proof_size_bytes": 2048
  }
}
```

### GET `/v1/fog/zk/predicates`
**Description**: List available ZK predicates  
**Authentication**: Required  
**Response**:
```json
{
  "success": true,
  "data": {
    "predicates": [
      {
        "type": "network_policy_compliance",
        "description": "Verify network access without revealing topology",
        "proof_size": "2KB",
        "verification_time": "~800ms"
      },
      {
        "type": "content_classification",
        "description": "Validate content types without accessing data",
        "proof_size": "1.5KB", 
        "verification_time": "~600ms"
      },
      {
        "type": "model_integrity",
        "description": "Verify AI model properties without exposing weights",
        "proof_size": "3KB",
        "verification_time": "~1200ms"
      }
    ]
  }
}
```

### POST `/v1/fog/zk/audit`
**Description**: Privacy-preserving compliance audit  
**Authentication**: Required  
**Request Body**:
```json
{
  "audit_type": "gdpr_compliance",
  "data_hash": "sha256:data123...",
  "compliance_proof": "zk_compliance_proof",
  "audit_parameters": {
    "data_retention_days": 365,
    "anonymization_level": "k_anonymity_5"
  }
}
```
**Response**:
```json
{
  "success": true,
  "data": {
    "audit_id": "aud_789xyz123",
    "audit_type": "gdpr_compliance",
    "result": "compliant",
    "confidence": 0.95,
    "details": {
      "data_retention_compliant": true,
      "anonymization_sufficient": true,
      "consent_tracking_valid": true
    },
    "audit_time": "2025-08-28T10:30:00Z"
  }
}
```

## 4. Market-Based Pricing (`/v1/fog/pricing/`)

### POST `/v1/fog/pricing/quote`
**Description**: Get dynamic price quote for resources  
**Authentication**: Required  
**Request Body**:
```json
{
  "resource_requirements": {
    "cpu_cores": 4,
    "memory_gb": 8,
    "storage_gb": 100,
    "duration_hours": 2,
    "privacy_level": "private",
    "sla_tier": "gold"
  },
  "max_price_per_hour": 5.00
}
```
**Response**:
```json
{
  "success": true,
  "data": {
    "quote_id": "qte_456def789",
    "base_price": 2.50,
    "privacy_multiplier": 1.5,
    "sla_multiplier": 3.0,
    "final_price_per_hour": 4.25,
    "total_cost": 8.50,
    "quote_valid_until": "2025-08-28T11:00:00Z",
    "providers_available": 8,
    "estimated_start_time": "2025-08-28T10:35:00Z"
  }
}
```

### POST `/v1/fog/pricing/auction`
**Description**: Create reverse auction for resources  
**Authentication**: Required  
**Request Body**:
```json
{
  "resource_spec": {
    "cpu_cores": 8,
    "memory_gb": 16,
    "duration_hours": 4,
    "privacy_level": "confidential"
  },
  "max_bid": 10.00,
  "auction_duration_minutes": 15,
  "required_providers": 3,
  "anti_griefing_deposit": 1.00
}
```
**Response**:
```json
{
  "success": true,
  "data": {
    "auction_id": "auc_abc123xyz789",
    "status": "active",
    "current_lowest_bid": 7.50,
    "bidding_providers": 5,
    "time_remaining_seconds": 847,
    "deposit_required": true,
    "deposit_amount": 1.00,
    "ends_at": "2025-08-28T10:45:00Z"
  }
}
```

### GET `/v1/fog/pricing/market`
**Description**: Get current market conditions  
**Authentication**: Required  
**Response**:
```json
{
  "success": true,
  "data": {
    "market_conditions": {
      "demand_level": "moderate",
      "average_price_per_core_hour": 1.25,
      "privacy_premium": 1.8,
      "gold_sla_premium": 2.5
    },
    "price_trends": {
      "1h_change": "+2.3%",
      "24h_change": "-1.1%",
      "7d_change": "+5.7%"
    },
    "provider_availability": {
      "total_providers": 234,
      "available_capacity": 1280,
      "average_response_time_ms": 450
    }
  }
}
```

### POST `/v1/fog/pricing/allocate`
**Description**: Allocate resources after winning auction  
**Authentication**: Required  
**Request Body**:
```json
{
  "quote_id": "qte_456def789",
  "auction_id": "auc_abc123xyz789",
  "provider_selection": "automatic",
  "payment_method": "token_escrow"
}
```
**Response**:
```json
{
  "success": true,
  "data": {
    "allocation_id": "alloc_def456ghi789",
    "provider_ids": ["prov_123", "prov_456"],
    "total_cost": 8.50,
    "escrow_tx": "0x789abc...",
    "resource_endpoints": [
      "https://provider1.fog.com:8443",
      "https://provider2.fog.com:8443"
    ],
    "access_credentials": "encrypted_creds_data",
    "allocated_until": "2025-08-28T12:35:00Z"
  }
}
```

## 5. Job Scheduler (`/v1/fog/scheduler/`)

### POST `/v1/fog/scheduler/submit`
**Description**: Submit job for fog scheduling  
**Authentication**: Required  
**Request Body**:
```json
{
  "job_spec": {
    "name": "ml_training_job",
    "description": "Train 25M parameter model",
    "docker_image": "aivillage/cognate:latest",
    "resource_requirements": {
      "cpu_cores": 8,
      "memory_gb": 32,
      "gpu_count": 1,
      "storage_gb": 500
    },
    "environment_variables": {
      "BATCH_SIZE": "64",
      "EPOCHS": "100"
    },
    "privacy_level": "private",
    "sla_tier": "gold"
  },
  "scheduling_preferences": {
    "preferred_regions": ["us-west", "us-east"],
    "avoid_providers": [],
    "deadline": "2025-08-28T18:00:00Z"
  }
}
```
**Response**:
```json
{
  "success": true,
  "data": {
    "job_id": "job_abc123def456",
    "status": "queued",
    "estimated_start_time": "2025-08-28T10:40:00Z",
    "estimated_completion": "2025-08-28T14:40:00Z",
    "allocated_providers": 2,
    "total_cost_estimate": 45.00,
    "queue_position": 3
  }
}
```

### GET `/v1/fog/scheduler/status/{job_id}`
**Description**: Get job execution status  
**Authentication**: Required  
**Response**:
```json
{
  "success": true,
  "data": {
    "job_id": "job_abc123def456",
    "status": "running",
    "progress": 0.35,
    "current_phase": "training_phase_2",
    "started_at": "2025-08-28T10:40:00Z",
    "estimated_completion": "2025-08-28T14:20:00Z",
    "resource_usage": {
      "cpu_utilization": 78.5,
      "memory_utilization": 85.2,
      "gpu_utilization": 92.1
    },
    "logs_url": "https://logs.fog.com/job_abc123def456",
    "cost_accrued": 28.75
  }
}
```

### GET `/v1/fog/scheduler/queue`
**Description**: Get scheduler queue status  
**Authentication**: Required  
**Response**:
```json
{
  "success": true,
  "data": {
    "queue_stats": {
      "total_jobs": 24,
      "queued_jobs": 8,
      "running_jobs": 12,
      "completed_jobs": 4,
      "average_wait_time_minutes": 15
    },
    "resource_utilization": {
      "cpu_utilization": 72.3,
      "memory_utilization": 68.9,
      "gpu_utilization": 88.4,
      "available_capacity": 156
    }
  }
}
```

## 6. Heterogeneous Quorum Manager (`/v1/fog/quorum/`)

### GET `/v1/fog/quorum/status`
**Description**: Get quorum health status  
**Authentication**: Required  
**Response**:
```json
{
  "success": true,
  "data": {
    "quorum_health": "healthy",
    "diversity_compliance": {
      "asn_diversity": {
        "required": 3,
        "current": 5,
        "compliant": true
      },
      "tee_vendor_diversity": {
        "required": 2,
        "current": 3,
        "vendors": ["AMD", "Intel", "ARM"],
        "compliant": true
      },
      "geographic_diversity": {
        "required": 2,
        "current": 4,
        "regions": ["us-east", "us-west", "eu-central", "ap-southeast"],
        "compliant": true
      }
    },
    "active_nodes": 12,
    "consensus_latency_ms": 250
  }
}
```

### POST `/v1/fog/quorum/validate`
**Description**: Validate infrastructure diversity for SLA tier  
**Authentication**: Required  
**Request Body**:
```json
{
  "target_sla_tier": "gold",
  "proposed_providers": ["prov_123", "prov_456", "prov_789"],
  "resource_requirements": {
    "cpu_cores": 16,
    "memory_gb": 64
  }
}
```
**Response**:
```json
{
  "success": true,
  "data": {
    "validation_result": "compliant",
    "sla_tier_supported": "gold",
    "diversity_analysis": {
      "asn_diversity": 3,
      "tee_vendor_diversity": 2,
      "geographic_diversity": 3,
      "power_region_diversity": 2
    },
    "fault_tolerance": {
      "single_provider_failure": "tolerant",
      "region_failure": "tolerant",
      "network_partition": "tolerant"
    },
    "estimated_sla": {
      "uptime_percentage": 99.95,
      "p95_latency_ms": 350
    }
  }
}
```

### GET `/v1/fog/quorum/tiers`
**Description**: Get SLA tier information  
**Authentication**: Required  
**Response**:
```json
{
  "success": true,
  "data": {
    "sla_tiers": {
      "bronze": {
        "uptime_guarantee": 97.0,
        "p95_latency_ms": 2500,
        "replication_factor": 1,
        "price_multiplier": 1.0,
        "requirements": "Single instance"
      },
      "silver": {
        "uptime_guarantee": 99.0,
        "p95_latency_ms": 1200,
        "replication_factor": 2,
        "price_multiplier": 2.5,
        "requirements": "Primary + canary"
      },
      "gold": {
        "uptime_guarantee": 99.9,
        "p95_latency_ms": 400,
        "replication_factor": 3,
        "price_multiplier": 5.0,
        "requirements": {
          "min_asn_diversity": 3,
          "min_tee_vendors": 2,
          "min_power_regions": 2
        }
      }
    }
  }
}
```

## 7. Onion Routing Integration (`/v1/fog/onion/`)

### POST `/v1/fog/onion/circuit`
**Description**: Create privacy circuit  
**Authentication**: Required  
**Request Body**:
```json
{
  "privacy_level": "confidential",
  "hops": 5,
  "destination_hint": "ml_training",
  "bandwidth_requirements": "high",
  "cover_traffic": true
}
```
**Response**:
```json
{
  "success": true,
  "data": {
    "circuit_id": "circ_xyz789abc123",
    "privacy_level": "confidential",
    "hops": 5,
    "circuit_established": true,
    "estimated_latency_ms": 850,
    "bandwidth_mbps": 100,
    "expires_at": "2025-08-28T12:30:00Z",
    "entry_node": "encrypted_address"
  }
}
```

### GET `/v1/fog/onion/status`
**Description**: Get circuit status  
**Authentication**: Required  
**Query Parameters**:
- `circuit_id`: Circuit ID to check

**Response**:
```json
{
  "success": true,
  "data": {
    "circuit_id": "circ_xyz789abc123",
    "status": "active",
    "privacy_level": "confidential",
    "hops_active": 5,
    "bandwidth_utilization": 0.45,
    "latency_p95_ms": 820,
    "data_transferred_mb": 125.6,
    "cover_traffic_active": true,
    "time_remaining_seconds": 5400
  }
}
```

### POST `/v1/fog/onion/route`
**Description**: Route data through circuit  
**Authentication**: Required  
**Request Body**:
```json
{
  "circuit_id": "circ_xyz789abc123",
  "data": "encrypted_payload_base64",
  "destination": "target_service_endpoint",
  "priority": "high"
}
```
**Response**:
```json
{
  "success": true,
  "data": {
    "routing_id": "rt_456def789ghi",
    "circuit_id": "circ_xyz789abc123",
    "status": "routed",
    "latency_ms": 832,
    "data_size_bytes": 4096,
    "hops_traversed": 5
  }
}
```

### GET `/v1/fog/onion/hidden`
**Description**: Get hidden service endpoints  
**Authentication**: Required  
**Response**:
```json
{
  "success": true,
  "data": {
    "hidden_services": [
      {
        "service_id": "hs_abc123def456",
        "onion_address": "a1b2c3d4e5f6g7h8.onion",
        "service_type": "ml_training",
        "status": "active",
        "clients_connected": 3
      }
    ],
    "directory_services": 8,
    "total_bandwidth_mbps": 500
  }
}
```

## 8. Bayesian Reputation System (`/v1/fog/reputation/`)

### GET `/v1/fog/reputation/score`
**Description**: Get reputation score for entity  
**Authentication**: Required  
**Query Parameters**:
- `entity_id`: Entity to score (required)
- `entity_type`: provider|user|service

**Response**:
```json
{
  "success": true,
  "data": {
    "entity_id": "prov_123",
    "entity_type": "provider",
    "reputation_score": 0.87,
    "confidence_interval": [0.82, 0.92],
    "tier": "platinum",
    "statistics": {
      "total_interactions": 1247,
      "successful_interactions": 1198,
      "success_rate": 0.961,
      "alpha_parameter": 1199,
      "beta_parameter": 50
    },
    "recent_trend": "stable",
    "last_updated": "2025-08-28T10:25:00Z"
  }
}
```

### POST `/v1/fog/reputation/update`
**Description**: Update reputation based on interaction  
**Authentication**: Required  
**Request Body**:
```json
{
  "entity_id": "prov_123",
  "entity_type": "provider",
  "interaction_result": "success",
  "interaction_data": {
    "task_id": "task_abc123",
    "completion_time_ms": 15000,
    "quality_score": 0.95,
    "cost_accuracy": 1.0
  },
  "weight": 1.0
}
```
**Response**:
```json
{
  "success": true,
  "data": {
    "entity_id": "prov_123",
    "old_score": 0.87,
    "new_score": 0.875,
    "confidence_change": 0.002,
    "update_applied": true,
    "new_tier": "platinum",
    "tier_changed": false
  }
}
```

### GET `/v1/fog/reputation/tiers`
**Description**: Get reputation tier information  
**Authentication**: Required  
**Response**:
```json
{
  "success": true,
  "data": {
    "tiers": {
      "diamond": {"min_score": 0.95, "min_interactions": 1000, "benefits": ["Priority scheduling", "Premium pricing"]},
      "platinum": {"min_score": 0.85, "min_interactions": 500, "benefits": ["Fast track approval", "Reduced deposits"]},
      "gold": {"min_score": 0.75, "min_interactions": 200, "benefits": ["Standard access", "Normal rates"]},
      "silver": {"min_score": 0.60, "min_interactions": 50, "benefits": ["Limited access", "Higher deposits"]},
      "bronze": {"min_score": 0.0, "min_interactions": 0, "benefits": ["Basic access", "Maximum deposits"]}
    },
    "tier_distribution": {
      "diamond": 23,
      "platinum": 89,
      "gold": 245,
      "silver": 178,
      "bronze": 67
    }
  }
}
```

### GET `/v1/fog/reputation/analytics`
**Description**: Get trust analytics and insights  
**Authentication**: Required  
**Query Parameters**:
- `timerange`: 1h|24h|7d|30d (default: 24h)
- `entity_type`: provider|user|service (optional)

**Response**:
```json
{
  "success": true,
  "data": {
    "timerange": "24h",
    "overall_metrics": {
      "average_reputation": 0.78,
      "median_reputation": 0.82,
      "reputation_volatility": 0.05,
      "total_interactions": 15678
    },
    "trust_trends": {
      "improving_entities": 145,
      "declining_entities": 67,
      "stable_entities": 390
    },
    "risk_analysis": {
      "high_risk_entities": 12,
      "moderate_risk_entities": 45,
      "low_risk_entities": 545
    }
  }
}
```

## 9. VRF Neighbor Selection (`/v1/fog/vrf/`)

### GET `/v1/fog/vrf/neighbors`
**Description**: Get current VRF-selected neighbors  
**Authentication**: Required  
**Query Parameters**:
- `node_id` (optional): Specific node ID

**Response**:
```json
{
  "success": true,
  "data": {
    "node_id": "node_abc123def456",
    "neighbors": [
      {
        "neighbor_id": "node_def456ghi789",
        "distance": 0.23,
        "connection_quality": 0.95,
        "last_seen": "2025-08-28T10:29:00Z",
        "reputation_score": 0.87
      }
    ],
    "neighbor_count": 8,
    "network_degree": 12,
    "selection_entropy": 4.2,
    "last_selection": "2025-08-28T10:15:00Z"
  }
}
```

### POST `/v1/fog/vrf/select`
**Description**: Trigger new VRF neighbor selection  
**Authentication**: Required  
**Request Body**:
```json
{
  "node_id": "node_abc123def456",
  "target_neighbors": 8,
  "selection_criteria": {
    "min_reputation": 0.7,
    "max_latency_ms": 100,
    "diversity_preference": "geographic"
  },
  "force_reselection": false
}
```
**Response**:
```json
{
  "success": true,
  "data": {
    "selection_id": "sel_789ghi123jkl",
    "node_id": "node_abc123def456",
    "new_neighbors": [
      {
        "neighbor_id": "node_jkl456mno789",
        "vrf_proof": "vrf_proof_hex_data",
        "selection_probability": 0.125,
        "verification_passed": true
      }
    ],
    "selection_time_ms": 50,
    "topology_improved": true,
    "eclipse_resistance_score": 0.92
  }
}
```

### GET `/v1/fog/vrf/topology`
**Description**: Get network topology information  
**Authentication**: Required  
**Response**:
```json
{
  "success": true,
  "data": {
    "network_stats": {
      "total_nodes": 1247,
      "average_degree": 8.5,
      "clustering_coefficient": 0.35,
      "diameter": 6,
      "connectivity": 0.95
    },
    "security_metrics": {
      "eclipse_resistance": 0.89,
      "sybil_resistance": 0.92,
      "partition_resistance": 0.87
    },
    "topology_properties": {
      "is_expander_graph": true,
      "expansion_ratio": 0.76,
      "spectral_gap": 0.42
    }
  }
}
```

### POST `/v1/fog/vrf/verify`
**Description**: Verify VRF selection proof  
**Authentication**: Required  
**Request Body**:
```json
{
  "node_id": "node_abc123def456",
  "neighbor_id": "node_jkl456mno789",
  "vrf_proof": "vrf_proof_hex_data",
  "public_key": "public_key_hex",
  "selection_seed": "seed_hex_data"
}
```
**Response**:
```json
{
  "success": true,
  "data": {
    "verification_id": "ver_456jkl789mno",
    "proof_valid": true,
    "selection_fair": true,
    "verification_time_ms": 25,
    "proof_details": {
      "vrf_output": "output_hex",
      "proof_pi": "pi_hex",
      "randomness_entropy": 4.8
    }
  }
}
```

## System Status and Monitoring

### GET `/v1/fog/system/status`
**Description**: Comprehensive system status  
**Authentication**: Required  
**Response**:
```json
{
  "success": true,
  "data": {
    "system_health": "healthy",
    "uptime_seconds": 86400,
    "version": "3.0.0",
    "components": {
      "tee_runtime": {"status": "healthy", "load": 0.45, "errors_24h": 0},
      "proof_system": {"status": "healthy", "load": 0.23, "errors_24h": 1},
      "zk_predicates": {"status": "healthy", "load": 0.67, "errors_24h": 0},
      "market_engine": {"status": "healthy", "load": 0.34, "errors_24h": 0},
      "job_scheduler": {"status": "healthy", "load": 0.78, "errors_24h": 2},
      "quorum_manager": {"status": "healthy", "load": 0.12, "errors_24h": 0},
      "onion_router": {"status": "healthy", "load": 0.56, "errors_24h": 0},
      "reputation": {"status": "healthy", "load": 0.23, "errors_24h": 0},
      "vrf_topology": {"status": "healthy", "load": 0.34, "errors_24h": 0}
    },
    "performance_metrics": {
      "requests_per_second": 245.7,
      "average_response_time_ms": 125,
      "error_rate_percent": 0.02,
      "memory_usage_percent": 67.5,
      "cpu_usage_percent": 45.2
    }
  }
}
```

## Error Handling

All API endpoints follow consistent error response format:

```json
{
  "success": false,
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Missing required parameter: enclave_id",
    "details": {
      "parameter": "enclave_id",
      "expected_type": "string",
      "provided_value": null
    },
    "trace_id": "tr_abc123def456"
  }
}
```

### Common Error Codes
- `INVALID_REQUEST` - Malformed request or missing parameters
- `AUTHENTICATION_FAILED` - Invalid or expired JWT token
- `AUTHORIZATION_FAILED` - Insufficient permissions
- `RESOURCE_NOT_FOUND` - Requested resource doesn't exist
- `RESOURCE_UNAVAILABLE` - Resource temporarily unavailable
- `RATE_LIMIT_EXCEEDED` - Too many requests
- `INTERNAL_ERROR` - Server-side error
- `TEE_UNAVAILABLE` - TEE hardware not available
- `PROOF_VERIFICATION_FAILED` - Cryptographic proof verification failed
- `AUCTION_EXPIRED` - Auction has ended
- `INSUFFICIENT_BALANCE` - Not enough tokens for operation
- `CIRCUIT_ESTABLISHMENT_FAILED` - Cannot create privacy circuit
- `QUORUM_NOT_ACHIEVED` - Cannot establish required diversity

## Rate Limiting

All endpoints are rate-limited based on authentication:
- **Unauthenticated**: 10 requests/minute
- **Authenticated**: 100 requests/minute  
- **Premium**: 1000 requests/minute

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
```

## WebSocket API

Real-time updates available via WebSocket at `/ws/fog`:

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8000/ws/fog');

// Subscribe to events
ws.send(JSON.stringify({
  action: 'subscribe',
  topics: ['job_status', 'auction_updates', 'circuit_events']
}));

// Receive updates
ws.onmessage = (event) => {
  const update = JSON.parse(event.data);
  console.log('Fog update:', update);
};
```

---

This comprehensive API reference covers all 32+ endpoints across AIVillage's Enhanced Fog Computing Platform. For interactive testing, visit the API documentation at `http://localhost:8000/docs` when the system is running.