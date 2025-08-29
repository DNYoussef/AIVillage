# Archaeological Integration API Reference

**Version**: v2.1.0  
**Integration Status**: ACTIVE  
**Date**: 2025-08-29  

This document describes the API endpoints and integrations added through the Archaeological Integration project, which systematically preserved and integrated innovations from 81 analyzed branches.

## Base URL

All archaeological endpoints are integrated into the unified API gateway:

```
Base URL: http://localhost:8000
API Prefix: /v1
```

## Authentication

All archaeological endpoints require JWT authentication:

```http
Authorization: Bearer <jwt_token>
```

## Emergency Triage System API

### Report Triage Incident

**Endpoint**: `POST /v1/monitoring/triage/incident`

**Description**: Report a new emergency triage incident for ML-based anomaly detection and automated response.

**Request Body**:
```json
{
  "source_component": "string",     // Required: Component where incident occurred
  "incident_type": "string",        // Required: Type of incident
  "description": "string",          // Required: Incident description
  "threat_level": "low|medium|high|critical", // Optional: Override threat level
  "raw_data": {                     // Optional: Additional incident data
    "cpu_usage": 95.0,
    "memory_usage": 85.0,
    "custom_metrics": {}
  }
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "incident_id": "uuid-string",
    "threat_level": "high",
    "status": "detected",
    "confidence_score": 0.92,
    "timestamp": "2025-08-29T12:00:00Z"
  },
  "message": "Triage incident 123e4567-e89b-12d3-a456-426614174000 created and classified as high"
}
```

**Archaeological Enhancement**: Based on findings from `codex/audit-critical-stub-implementations` branch.

---

### Get Triage Incidents

**Endpoint**: `GET /v1/monitoring/triage/incidents`

**Description**: Retrieve triage incidents with optional filtering.

**Query Parameters**:
- `status` (optional): Filter by incident status (`detected`, `investigating`, `resolved`, `false_positive`)
- `threat_level` (optional): Filter by threat level (`low`, `medium`, `high`, `critical`)
- `limit` (optional): Maximum number of incidents to return (default: 50)

**Example Request**:
```http
GET /v1/monitoring/triage/incidents?status=detected&threat_level=high&limit=10
```

**Response**:
```json
{
  "success": true,
  "data": {
    "incidents": [
      {
        "incident_id": "uuid-string",
        "source_component": "agent_forge",
        "incident_type": "memory_leak",
        "description": "High memory usage detected",
        "threat_level": "high",
        "status": "detected",
        "confidence_score": 0.92,
        "timestamp": "2025-08-29T12:00:00Z",
        "age_seconds": 300,
        "is_active": true,
        "raw_data": {
          "memory_usage": 85.0
        },
        "response_log": [
          {
            "timestamp": "2025-08-29T12:00:30Z",
            "action": "alert_sent",
            "details": "High priority alert dispatched"
          }
        ]
      }
    ],
    "total_count": 1,
    "filters_applied": {
      "status": "detected",
      "threat_level": "high",
      "limit": 10
    }
  },
  "message": "Retrieved 1 triage incidents"
}
```

---

### Get Specific Triage Incident

**Endpoint**: `GET /v1/monitoring/triage/incident/{incident_id}`

**Description**: Get detailed information about a specific triage incident.

**Path Parameters**:
- `incident_id`: UUID of the incident

**Response**:
```json
{
  "success": true,
  "data": {
    "incident_id": "uuid-string",
    "source_component": "agent_forge",
    "incident_type": "performance_degradation",
    "description": "Response time increased significantly",
    "threat_level": "medium",
    "status": "investigating",
    "confidence_score": 0.78,
    "timestamp": "2025-08-29T11:30:00Z",
    "resolution_time": null,
    "age_seconds": 1800,
    "is_active": true,
    "raw_data": {
      "response_time_ms": 2500,
      "baseline_ms": 150,
      "degradation_factor": 16.7
    },
    "response_log": [
      {
        "timestamp": "2025-08-29T11:30:15Z",
        "action": "notification_sent",
        "details": "Performance team notified"
      },
      {
        "timestamp": "2025-08-29T11:45:00Z",
        "action": "investigation_started",
        "details": "Assigned to performance analysis team"
      }
    ]
  },
  "message": "Retrieved triage incident uuid-string"
}
```

---

### Update Triage Incident Status

**Endpoint**: `POST /v1/monitoring/triage/incident/{incident_id}/status`

**Description**: Update the status of a triage incident.

**Path Parameters**:
- `incident_id`: UUID of the incident

**Request Body**:
```json
{
  "new_status": "investigating|resolved|false_positive", // Required
  "notes": "Additional notes about status change"        // Optional
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "incident_id": "uuid-string",
    "old_status": "detected",
    "new_status": "resolved",
    "timestamp": "2025-08-29T12:30:00Z"
  },
  "message": "Incident uuid-string status updated to resolved"
}
```

---

### Get Triage Statistics

**Endpoint**: `GET /v1/monitoring/triage/statistics`

**Description**: Get comprehensive Emergency Triage System statistics.

**Response**:
```json
{
  "success": true,
  "data": {
    "total_incidents": 1247,
    "incidents_by_status": {
      "detected": 23,
      "investigating": 8,
      "resolved": 1198,
      "false_positive": 18
    },
    "incidents_by_threat_level": {
      "low": 890,
      "medium": 201,
      "high": 134,
      "critical": 22
    },
    "active_incidents_by_threat_level": {
      "low": 12,
      "medium": 8,
      "high": 10,
      "critical": 1
    },
    "performance_metrics": {
      "avg_detection_time_ms": 45.2,
      "avg_response_time_ms": 87.6,
      "false_positive_rate": 0.014,
      "confidence_score_avg": 0.847
    },
    "anomaly_detector_stats": {
      "model_accuracy": 0.952,
      "last_training": "2025-08-28T10:00:00Z",
      "patterns_learned": 1247,
      "feature_dimensions": 15
    },
    "timestamp": "2025-08-29T12:30:00Z"
  },
  "message": "Emergency triage statistics retrieved"
}
```

---

### Create Test Incident

**Endpoint**: `POST /v1/monitoring/triage/test-incident`

**Description**: Create a test incident for system validation and archaeological integration verification.

**Response**:
```json
{
  "success": true,
  "data": {
    "test_incident_id": "test-uuid-string",
    "status": "detected",
    "message": "Archaeological Emergency Triage System integration successful"
  },
  "message": "Test incident created successfully - Emergency Triage System operational"
}
```

**Archaeological Enhancement**: Validates integration success from archaeological findings.

---

## Enhanced Security API

### ECH Configuration Status

**Endpoint**: `GET /v1/security/ech/status`

**Description**: Get Encrypted Client Hello (ECH) configuration status.

**Response**:
```json
{
  "success": true,
  "data": {
    "ech_enabled": true,
    "supported_cipher_suites": [
      "chacha20_poly1305_sha256",
      "aes_256_gcm_sha384",
      "aes_128_gcm_sha256"
    ],
    "active_configurations": 3,
    "sni_protection": "enabled",
    "quantum_resistance": "prepared",
    "archaeological_integration": {
      "status": "active",
      "innovation_score": 8.3,
      "source_branch": "codex/add-ech-config-parsing-and-validation"
    }
  },
  "message": "ECH configuration status retrieved"
}
```

**Archaeological Enhancement**: Based on findings from ECH configuration parsing branch.

---

### Noise Protocol Status

**Endpoint**: `GET /v1/security/noise/status`

**Description**: Get enhanced Noise XK protocol status with archaeological integrations.

**Response**:
```json
{
  "success": true,
  "data": {
    "noise_protocol_enabled": true,
    "handshake_type": "XK_Enhanced",
    "perfect_forward_secrecy": true,
    "ech_integration": true,
    "active_sessions": 42,
    "cryptographic_strength": "256-bit",
    "archaeological_enhancement": {
      "status": "integrated",
      "innovation_score": 8.3,
      "security_improvement": "85%",
      "source_branch": "codex/implement-noise-protocol-with-perfect-forward-secrecy"
    }
  },
  "message": "Enhanced Noise protocol status retrieved"
}
```

---

## Tensor Memory Optimization API

### Get Memory Report

**Endpoint**: `GET /v1/memory/tensor/report`

**Description**: Get comprehensive tensor memory usage report with archaeological optimizations.

**Response**:
```json
{
  "success": true,
  "data": {
    "timestamp": 1693305600.0,
    "optimizer_enabled": true,
    "auto_cleanup_active": true,
    "registry_stats": {
      "total_tensors": 1534,
      "active_tensor_ids": 892,
      "memory_usage_mb": 2048.5,
      "peak_memory_mb": 3072.8,
      "cleanup_count": 642,
      "leak_prevention_count": 23,
      "gc_trigger_count": 8
    },
    "active_tensors": 892,
    "cuda_memory": {
      "allocated_mb": 1024.2,
      "reserved_mb": 1536.0,
      "max_allocated_mb": 2048.5,
      "max_reserved_mb": 3072.0
    },
    "archaeological_enhancement": {
      "status": "active",
      "innovation_score": 6.9,
      "memory_reduction": "30%",
      "source_branch": "codex/cleanup-tensor-id-in-receive_tensor"
    }
  },
  "message": "Tensor memory report retrieved"
}
```

**Archaeological Enhancement**: Memory leak prevention from archaeological findings.

---

### Force Memory Cleanup

**Endpoint**: `POST /v1/memory/tensor/cleanup`

**Description**: Force tensor memory cleanup with archaeological optimizations.

**Response**:
```json
{
  "success": true,
  "data": {
    "cleanup_performed": true,
    "tensors_before": 1203,
    "tensors_after": 856,
    "tensors_cleaned": 347,
    "memory_freed_mb": 512.3,
    "gc_triggered": true,
    "cuda_cache_cleared": true
  },
  "message": "Forced tensor memory cleanup completed"
}
```

---

## Error Responses

All archaeological endpoints follow standard error response format:

```json
{
  "success": false,
  "error": {
    "code": "ARCHAEOLOGICAL_ERROR_CODE",
    "message": "Human readable error message",
    "details": {
      "component": "emergency_triage",
      "archaeological_integration": "active",
      "additional_info": "..."
    }
  },
  "message": "Operation failed"
}
```

### Common Error Codes

- `TRIAGE_SERVICE_UNAVAILABLE`: Emergency triage service not running
- `INVALID_THREAT_LEVEL`: Invalid threat level specified
- `INCIDENT_NOT_FOUND`: Triage incident not found
- `INVALID_STATUS`: Invalid status for incident update
- `ECH_CONFIGURATION_ERROR`: ECH configuration parsing failed
- `NOISE_PROTOCOL_ERROR`: Noise protocol handshake failed
- `TENSOR_OPTIMIZATION_ERROR`: Tensor memory optimization failed
- `AUTHENTICATION_REQUIRED`: JWT token missing or invalid
- `INSUFFICIENT_PERMISSIONS`: User lacks required permissions

## Rate Limiting

All archaeological endpoints are subject to rate limiting:

- **Default Limit**: 1000 requests per hour per authenticated user
- **Triage Endpoints**: 100 incident reports per hour per user
- **Memory Cleanup**: 10 forced cleanups per hour per user

**Rate Limit Headers**:
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1693309200
```

## SDK Integration

Python SDK example for archaeological endpoints:

```python
import requests
from typing import Dict, Any

class ArchaeologicalClient:
    """Client for Archaeological Integration API endpoints."""
    
    def __init__(self, base_url: str, jwt_token: str):
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {jwt_token}",
            "Content-Type": "application/json"
        }
    
    def report_triage_incident(self, 
                             source_component: str,
                             incident_type: str, 
                             description: str,
                             threat_level: str = None,
                             raw_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Report emergency triage incident."""
        payload = {
            "source_component": source_component,
            "incident_type": incident_type,
            "description": description
        }
        if threat_level:
            payload["threat_level"] = threat_level
        if raw_data:
            payload["raw_data"] = raw_data
            
        response = requests.post(
            f"{self.base_url}/v1/monitoring/triage/incident",
            json=payload,
            headers=self.headers
        )
        return response.json()
    
    def get_triage_incidents(self, status: str = None, 
                           threat_level: str = None, 
                           limit: int = 50) -> Dict[str, Any]:
        """Get triage incidents with filtering."""
        params = {"limit": limit}
        if status:
            params["status"] = status
        if threat_level:
            params["threat_level"] = threat_level
            
        response = requests.get(
            f"{self.base_url}/v1/monitoring/triage/incidents",
            params=params,
            headers=self.headers
        )
        return response.json()
    
    def get_tensor_memory_report(self) -> Dict[str, Any]:
        """Get tensor memory optimization report."""
        response = requests.get(
            f"{self.base_url}/v1/memory/tensor/report",
            headers=self.headers
        )
        return response.json()
    
    def force_memory_cleanup(self) -> Dict[str, Any]:
        """Force tensor memory cleanup."""
        response = requests.post(
            f"{self.base_url}/v1/memory/tensor/cleanup",
            headers=self.headers
        )
        return response.json()

# Usage example
client = ArchaeologicalClient("http://localhost:8000", "your-jwt-token")

# Report incident
incident = client.report_triage_incident(
    source_component="agent_forge",
    incident_type="memory_leak",
    description="High memory usage detected in training pipeline",
    threat_level="high",
    raw_data={"memory_usage": 95.0, "threshold": 80.0}
)

# Get memory report
memory_report = client.get_tensor_memory_report()
print(f"Memory usage: {memory_report['data']['registry_stats']['memory_usage_mb']}MB")
```

## WebSocket Integration

Real-time updates for archaeological features:

```javascript
// WebSocket connection for real-time triage updates
const ws = new WebSocket('ws://localhost:8000/ws/triage');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    if (data.type === 'triage_incident') {
        console.log('New triage incident:', data.incident);
        updateTriageDashboard(data.incident);
    }
    
    if (data.type === 'memory_alert') {
        console.log('Memory usage alert:', data.alert);
        updateMemoryMonitor(data.alert);
    }
};
```

## Conclusion

The Archaeological Integration API provides comprehensive access to:

1. **Emergency Triage System**: ML-based incident detection and management
2. **Enhanced Security**: ECH + Noise protocol status and configuration
3. **Memory Optimization**: Tensor memory management and reporting
4. **Real-time Updates**: WebSocket integration for live monitoring
5. **Production Ready**: Full authentication, rate limiting, and error handling

All endpoints maintain backward compatibility while providing access to advanced archaeological enhancements that significantly improve system security, reliability, and performance.

---

**Maintained by**: Archaeological Integration Team  
**API Version**: v2.1.0  
**Last Updated**: 2025-08-29  
**Status**: Production Ready
