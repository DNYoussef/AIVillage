# P2P/Fog Computing API Documentation

## Overview

The AIVillage P2P/Fog Computing API provides comprehensive endpoints for monitoring and managing distributed peer-to-peer networks and fog computing resources. All endpoints are integrated into the Unified Agent Forge Backend running on port 8083.

## Base URL

```
http://localhost:8083
```

## Authentication

Currently no authentication required for local development. Production deployments should implement proper authentication mechanisms.

## WebSocket Endpoint

**Endpoint:** `ws://localhost:8083/ws`

Real-time updates for P2P network status, fog resource changes, token economics, and privacy metrics.

### WebSocket Message Types:
- `connection_established` - Initial connection confirmation
- `p2p_update` - P2P network status changes
- `fog_update` - Fog computing resource updates
- `token_update` - FOG token balance and transaction updates
- `privacy_update` - Privacy and security metrics updates
- `system_alert` - System-wide alerts and notifications

## P2P Network Endpoints

### GET /api/p2p/status

Get current P2P network status including BitChat and BetaNet connectivity.

**Response:**
```json
{
  "status": "operational",
  "services_available": true,
  "bitchat": {
    "connected": true,
    "platform": "unified_backend",
    "status": "online"
  },
  "betanet": {
    "connected": true,
    "mixnodes": ["mix1.betanet.ai:9443", "mix2.betanet.ai:9443", "mix3.betanet.ai:9443"],
    "active_circuits": 3,
    "privacy_level": 85
  },
  "timestamp": "2025-08-27T14:35:06.839871"
}
```

### GET /api/p2p/peers

Get information about connected peers in the mesh network.

**Response:**
```json
{
  "status": "operational",
  "total_peers": 5,
  "connected_peers": 0,
  "topology": "adaptive_mesh",
  "peers": [],
  "mesh_stats": {
    "connectivity_ratio": 0.92,
    "avg_latency_ms": 45,
    "total_bandwidth_mbps": 0,
    "redundancy_level": 4,
    "active_circuits": 0
  },
  "timestamp": "2025-08-27T14:35:06.839871"
}
```

### GET /api/p2p/messages

Get recent P2P message statistics and routing information.

**Response:**
```json
{
  "status": "operational",
  "message_stats": {
    "total_sent": 0,
    "total_received": 0,
    "avg_latency_ms": 45,
    "success_rate": 0.98
  },
  "routing_table": [],
  "recent_messages": [],
  "timestamp": "2025-08-27T14:35:12.056273"
}
```

## Fog Computing Endpoints

### GET /api/fog/nodes

Get information about active fog computing nodes.

**Response:**
```json
{
  "status": "operational",
  "total_nodes": 1,
  "active_nodes": 1,
  "node_types": {
    "coordinator": 1,
    "worker": 0,
    "hybrid": 0
  },
  "nodes": [
    {
      "node_id": "unified_backend_node",
      "type": "coordinator",
      "status": "active",
      "resources": {
        "cpu_cores": 8,
        "memory_gb": 16,
        "storage_gb": 500,
        "bandwidth_mbps": 100
      },
      "last_seen": "2025-08-27T14:35:22.232945"
    }
  ],
  "timestamp": "2025-08-27T14:35:22.232945"
}
```

### GET /api/fog/resources

Get current fog computing resource utilization and availability.

**Response:**
```json
{
  "status": "operational",
  "harvesting": {
    "active_devices": 0,
    "total_registered": 0,
    "harvest_rate_per_hour": 75.2,
    "idle_capacity_percent": 28.5
  },
  "resources": {
    "cpu_hours_available": 0.0,
    "memory_gb_hours": 0.0,
    "storage_gb_available": 0,
    "bandwidth_mbps_available": 0
  },
  "utilization": {
    "cpu_utilization": 52.1,
    "memory_utilization": 38.9,
    "storage_utilization": 22.3,
    "bandwidth_utilization": 64.7
  },
  "energy": {
    "devices_charging": 1,
    "battery_threshold_met": 0,
    "thermal_throttling": 0,
    "green_energy_ratio": 0.78
  },
  "rewards": {
    "tokens_distributed_today": 0.0,
    "avg_reward_per_hour": 8.5,
    "quality_bonus_rate": 0.12
  },
  "timestamp": "2025-08-27T14:35:12.056273"
}
```

### GET /api/fog/marketplace

Get fog computing marketplace information including service offerings and pricing.

**Response:**
```json
{
  "status": "operational",
  "marketplace_stats": {
    "total_offerings": 0,
    "active_contracts": 0,
    "avg_price_per_hour": 0.0,
    "total_providers": 0
  },
  "services": [],
  "demand_metrics": {},
  "hidden_services": {
    "total_hidden_services": 0,
    "avg_uptime": 0.96,
    "censorship_resistance": "high"
  },
  "timestamp": "2025-08-27T14:35:22.232945"
}
```

## Token Economics Endpoints

### GET /api/fog/tokens

Get FOG token economics information including balances, transactions, and network stats.

**Response:**
```json
{
  "status": "operational",
  "token_info": {
    "symbol": "FOG",
    "name": "Fog Computing Token",
    "decimals": 18,
    "total_supply": 10000000000.0,
    "current_supply": 1000000000.0
  },
  "network_stats": {
    "total_staked": 0.0,
    "staking_apy": 0.05,
    "total_validators": 0,
    "active_proposals": 0,
    "total_accounts": 0
  },
  "user_balance": {
    "account_id": "unified_backend_account",
    "balance": 2000.0,
    "staked_balance": 0.0,
    "locked_balance": 0.0,
    "total_balance": 2000.0,
    "voting_power": 0.0,
    "validator_node": false,
    "total_contributed": 1000.0,
    "total_consumed": 0.0,
    "created_at": "2025-08-27T18:35:17.463481+00:00",
    "last_activity": "2025-08-27T18:35:17.463481+00:00"
  },
  "recent_transactions": [
    {
      "tx_id": "363075b8-db69-4307-91e2-1dc83da74615",
      "type": "mint",
      "amount": 1000.0,
      "timestamp": "2025-08-27T18:35:17.463481+00:00",
      "status": "confirmed",
      "from_account": "system",
      "to_account": "unified_backend_account"
    }
  ],
  "timestamp": "2025-08-27T14:35:17.463481"
}
```

## Health Monitoring

### GET /health

System health check including all P2P/Fog services.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-08-27T12:32:23.308947",
  "services": {
    "agent_forge": {
      "available": true,
      "active_phases": 0,
      "total_models": 0,
      "training_instances": 0
    },
    "p2p_fog": {
      "available": true,
      "mobile_bridge_connected": true,
      "mixnode_client_connected": true,
      "fog_coordinator_running": true
    }
  },
  "websocket_connections": 0
}
```

## Error Responses

All endpoints return appropriate HTTP status codes:

- `200 OK` - Request successful
- `400 Bad Request` - Invalid request parameters
- `500 Internal Server Error` - Server-side error
- `503 Service Unavailable` - Service temporarily unavailable

Error response format:
```json
{
  "error": "Error message",
  "details": "Detailed error information",
  "timestamp": "2025-08-27T14:35:17.463481"
}
```

## Rate Limiting

Currently no rate limiting implemented for local development. Production deployments should implement appropriate rate limiting.

## Support

For API support and questions:
- GitHub Issues: https://github.com/DNYoussef/AIVillage/issues
- Documentation: See `docs/` directory for additional guides
