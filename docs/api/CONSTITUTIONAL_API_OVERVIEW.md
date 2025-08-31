# Constitutional Fog Compute API Overview

**Version**: v3.0.0-constitutional  
**Status**: Production Ready ✅  
**Authentication**: JWT Bearer Token + Constitutional Tier Validation  
**Base URL**: `https://api.aivillage.com/constitutional/v3`

## 🏛️ Constitutional API Architecture

The Constitutional Fog Compute API provides access to machine-only moderation, democratic governance, and privacy-preserving fog computing capabilities across Bronze, Silver, Gold, and Platinum constitutional tiers.

### Core API Endpoints

#### 🔐 Constitutional Authentication
```http
POST /auth/constitutional-login
POST /auth/tier-verification
GET  /auth/constitutional-profile
```

#### ⚖️ Constitutional Moderation
```http
POST /moderation/classify-content
POST /moderation/submit-appeal
GET  /moderation/appeal-status/{appeal_id}
GET  /moderation/decision-receipt/{decision_id}
```

#### 🏛️ Democratic Governance  
```http
GET  /governance/proposals
POST /governance/vote
GET  /governance/voting-history
POST /governance/create-proposal
```

#### 💰 H200-Hour Pricing
```http
GET  /pricing/h200-quote
POST /pricing/calculate-equivalent
GET  /pricing/constitutional-tiers
POST /pricing/governance-fee-estimate
```

#### 🌫️ Constitutional Fog Compute
```http
POST /fog/submit-workload
GET  /fog/workload-status/{workload_id}
POST /fog/constitutional-routing
GET  /fog/tier-availability
```

#### 🔍 Transparency & Audit
```http
GET  /transparency/public-metrics
GET  /transparency/audit-trail/{decision_id}
GET  /transparency/merkle-proof/{receipt_id}
POST /transparency/verify-receipt
```

## 📊 Constitutional Tier Access Matrix

| Endpoint Category | Bronze | Silver | Gold | Platinum |
|------------------|--------|--------|------|----------|
| **Basic Moderation** | ✅ | ✅ | ✅ | ✅ |
| **Appeal Process** | Basic | Enhanced | Premium | Expert |
| **H200-Hour Pricing** | ✅ | ✅ | ✅ | ✅ |
| **Fog Computing** | Best-effort | Regional | TEE-only | Max-privacy |
| **Transparency Access** | Full | Selective | Privacy-preserving | Minimal |
| **Governance Voting** | ✅ | ✅ | ✅ | Premium |

## 🔒 Constitutional API Security

### Authentication Levels
- **Bronze**: JWT + Basic tier verification
- **Silver**: JWT + Enhanced identity verification
- **Gold**: JWT + TEE attestation required
- **Platinum**: JWT + Hardware-backed identity + Expert verification

### Rate Limits by Tier
- **Bronze**: 100 requests/hour
- **Silver**: 500 requests/hour  
- **Gold**: 2,000 requests/hour
- **Platinum**: 10,000 requests/hour

## 📝 Example API Usage

### Constitutional Content Classification
```javascript
POST /moderation/classify-content
Content-Type: application/json
Authorization: Bearer <constitutional-jwt>

{
  "content": {
    "text": "Content to classify",
    "media": ["base64-encoded-image"],
    "context": "commerce_storefront"
  },
  "tier": "gold",
  "privacy_mode": "zero_knowledge"
}

Response:
{
  "decision_id": "dec_123abc",
  "harm_level": "H1",
  "confidence": 0.94,
  "constitutional_protected": true,
  "zk_proof": "proof_data_for_gold_tier",
  "transparency_receipt": "receipt_hash",
  "appeal_deadline": "2025-02-06T10:00:00Z"
}
```

### H200-Hour Price Calculation
```javascript
POST /pricing/calculate-equivalent
Content-Type: application/json
Authorization: Bearer <constitutional-jwt>

{
  "device_tops": 1500,
  "utilization_rate": 0.8,
  "time_hours": 2.5,
  "constitutional_tier": "silver",
  "tee_required": false
}

Response:
{
  "h200_hours": 0.759,
  "base_price": 0.569,
  "constitutional_multiplier": 1.15,
  "governance_fee": 0.028,
  "total_price": 0.597,
  "formula_used": "(1500 × 0.8 × 2.5) / 3958"
}
```

### Democratic Governance Vote
```javascript
POST /governance/vote
Content-Type: application/json
Authorization: Bearer <constitutional-jwt>

{
  "proposal_id": "prop_456def",
  "vote": "approve",
  "rationale": "Constitutional protection enhancement",
  "stake_amount": 100
}

Response:
{
  "vote_id": "vote_789ghi",
  "recorded_at": "2025-01-30T15:30:00Z",
  "voting_power": 150,
  "proposal_status": "active",
  "current_tally": {
    "approve": 1247,
    "reject": 386,
    "abstain": 92
  }
}
```

## 🔍 Constitutional Compliance Features

### Harm Taxonomy (H0-H3)
- **H0**: Constitutional content (fully protected)
- **H1**: Minor concerns (warnings, monitoring)  
- **H2**: Moderate harm (conditional approval)
- **H3**: Severe harm (quarantine, escalation)

### Viewpoint Firewall
All API responses exclude political/ideological classifications:
```json
{
  "harm_classification": {
    "violence": 0.02,
    "hate_speech": 0.15,
    "misinformation": 0.31
  },
  "viewpoint_signals": "FILTERED",
  "political_leaning": "NOT_ACTIONABLE",
  "constitutional_protection": "FIRST_AMENDMENT_PROTECTED"
}
```

### Privacy-Preserving Verification
Gold/Platinum tiers receive zero-knowledge proofs instead of raw classifications:
```json
{
  "constitutional_compliance": "VERIFIED",
  "zk_proof": {
    "proof_type": "SNARK",
    "verification_key": "vk_constitutional_v3",
    "proof_data": "base64_encoded_proof"
  },
  "public_inputs": {
    "harm_threshold_met": true,
    "constitutional_protected": true
  }
}
```

## 📚 API Documentation Structure

### Complete Documentation Available
- **[Constitutional API Reference](constitutional-fog-compute-api.md)** - Complete endpoint documentation
- **[OpenAPI Specification](openapi-constitutional-spec.yaml)** - Machine-readable API spec
- **[SDK Integration Guide](sdk-integration-guide.md)** - Language-specific SDKs
- **[Constitutional Compliance Guide](constitutional-compliance-guide.md)** - Legal and ethical guidelines
- **[Tier-Based API Reference](tier-based-api-reference.md)** - Tier-specific capabilities
- **[Democratic Governance API](democratic-governance-api.md)** - Governance and voting endpoints

### Integration Examples
- **JavaScript/TypeScript**: React constitutional dashboard integration
- **Python**: ML workload submission with constitutional compliance
- **Go**: High-performance fog computing with TEE attestation
- **Rust**: BetaNet transport integration with privacy preservation

## 🚀 Getting Started

### 1. Authentication Setup
```bash
curl -X POST https://api.aivillage.com/constitutional/v3/auth/constitutional-login \
  -H "Content-Type: application/json" \
  -d '{"tier": "silver", "credentials": "..."}'
```

### 2. Verify Constitutional Tier
```bash
curl -X GET https://api.aivillage.com/constitutional/v3/auth/constitutional-profile \
  -H "Authorization: Bearer <token>"
```

### 3. Submit Constitutional Workload
```bash
curl -X POST https://api.aivillage.com/constitutional/v3/fog/submit-workload \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"workload": "...", "tier": "silver", "constitutional_required": true}'
```

## 🔗 Related Resources

- [Constitutional System Architecture](../architecture/CONSTITUTIONAL_FOG_ARCHITECTURE.md)
- [H200-Hour Pricing Implementation](../implementation/H200_HOUR_CONSTITUTIONAL_PRICING.md)
- [BetaNet Constitutional Research](../PHASE_3_BETANET_CONSTITUTIONAL_RESEARCH_ANALYSIS.md)
- [TEE Security Integration](../security/TEE_INTEGRATION_COMPLETE.md)

---

The Constitutional Fog Compute API enables developers to build constitutionally compliant applications with machine-only moderation, democratic governance, and privacy-preserving fog computing capabilities.