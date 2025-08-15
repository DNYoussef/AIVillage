# AIVillage Bounty Implementation - Execution Order

## Phase 1: HTX Covert Transport (Week 1-2)
**Goal**: Replace JSON on port 4001 with proper TLS/QUIC encrypted HTX protocol

### Step 1.1: TLS/QUIC Foundation (3 days)
```python
# File: src/core/p2p/betanet_transport_v2.py
# Tasks:
1. Add ssl.SSLContext for TLS 1.3
2. Integrate aioquic for QUIC support
3. Replace JSON serialization with binary HTX frames
4. Implement frame chunking (16KB max)
5. Add encryption key rotation (epochs)
```

### Step 1.2: Chrome Fingerprinting Activation (2 days)
```python
# File: src/core/p2p/chrome_fingerprint.py (NEW)
# Tasks:
1. Port Rust ChromeFingerprinter to Python
2. Implement JA3/JA4 hash generation
3. Add H2 settings mimicry (±15% variation)
4. Create origin-specific calibration loop
5. Wire to BetanetTransportV2
```

### Step 1.3: Python-Rust Bridge (2 days)
```bash
# Use PyO3 or cffi for Rust integration
# Files:
- betanet-client/src/python_bindings.rs (NEW)
- src/core/p2p/htx_native.py (NEW)

# Tasks:
1. Export HtxTransport to Python
2. Create shared memory channel
3. Implement async message passing
4. Add error handling across FFI
```

### Step 1.4: Covert Channels Implementation (2 days)
```python
# File: src/core/p2p/covert_channels.py (NEW)
# Tasks:
1. HeaderCovertChannel with custom headers
2. CookieCovertChannel with realistic patterns
3. TimingCovertChannel using delays
4. Implement message embedding/extraction
5. Add channel capacity management
```

### Step 1.5: Integration Testing (1 day)
```python
# File: tests/test_htx_transport.py (NEW)
# Verify:
- TLS 1.3 handshake with Chrome ciphers
- QUIC connection establishment
- Binary frame serialization
- Covert channel operation
- 95%+ delivery rate
```

---

## Phase 2: Indistinguishability Enhancements (Week 2-3)
**Goal**: Implement real onion routing and cover traffic generation

### Step 2.1: Onion Routing Implementation (3 days)
```python
# File: src/core/p2p/onion_routing.py (NEW)
# Tasks:
1. Multi-layer encryption with X25519
2. Circuit establishment protocol
3. Cell-based forwarding (512 bytes)
4. Directory service for relay nodes
5. Path selection algorithm (3 hops default)
```

### Step 2.2: Cover Traffic Scheduler (2 days)
```python
# File: src/core/p2p/cover_traffic.py (NEW)
# Tasks:
1. Poisson process for dummy messages
2. Traffic pattern templates (web, streaming, etc)
3. Bandwidth-aware scheduling
4. Battery-optimized mode for mobile
5. Statistical indistinguishability tests
```

### Step 2.3: Mixnode Implementation (2 days)
```python
# File: src/core/p2p/mixnode.py (NEW)
# Tasks:
1. Store-and-forward message mixing
2. Batch processing with delays
3. Cover traffic injection at mixnodes
4. Reputation system for node selection
5. Byzantine fault tolerance
```

### Step 2.4: Integration (1 day)
```python
# Update: src/core/p2p/betanet_transport_v2.py
# Wire together:
- Onion routing for all messages
- Cover traffic scheduler activation
- Mixnode path selection
- Verify indistinguishability
```

---

## Phase 3: Latency/Jitter Measurement & Adaptation (Week 3)
**Goal**: Active measurement and dynamic adaptation

### Step 3.1: Measurement Infrastructure (2 days)
```python
# File: src/core/p2p/latency_monitor.py (NEW)
# Tasks:
1. Continuous RTT measurement
2. Jitter calculation (variance tracking)
3. Bandwidth estimation
4. Packet loss detection
5. Network condition classification
```

### Step 3.2: QoS Adaptation Engine (2 days)
```python
# File: src/core/p2p/qos_engine.py (NEW)
# Tasks:
1. Dynamic parameter adjustment
2. Congestion control algorithm
3. Priority queue management
4. Deadline-based scheduling
5. Fairness enforcement
```

### Step 3.3: Integration with Transport (1 day)
```python
# Update: src/core/p2p/unified_transport.py
# Tasks:
1. Hook measurement to all transports
2. Feed metrics to QoS engine
3. Apply QoS decisions to routing
4. Add telemetry export
5. Create dashboard endpoints
```

---

## Phase 4: SCION Gateway Implementation (Week 4-5)
**Goal**: Complete SCION protocol support with fallback

### Step 4.1: SCION Protocol Core (4 days)
```python
# File: src/core/p2p/scion_protocol.py (NEW)
# Tasks:
1. SCION packet format implementation
2. Path construction and validation
3. DRKey cryptographic framework
4. AS-level routing tables
5. ISD/AS addressing scheme
```

### Step 4.2: SCION Daemon Integration (2 days)
```python
# File: src/core/p2p/scion_daemon.py (NEW)
# Tasks:
1. Connect to local SCION daemon
2. Path lookup and caching
3. Beacon service integration
4. Certificate validation
5. Control plane messaging
```

### Step 4.3: Python Bindings (2 days)
```python
# Options:
1. Use PySCION library if available
2. Create ctypes wrapper for C implementation
3. Build pure Python implementation
4. Focus on essential features only
```

### Step 4.4: Fallback Integration (1 day)
```python
# Update: src/core/p2p/fallback_transports.py
# Tasks:
1. Add SCION as primary transport
2. Implement path-aware fallback
3. Handle SCION failures gracefully
4. Test with network partitions
```

---

## Phase 5: Multi-Agent Coordination (Week 5-6)
**Goal**: Distributed agent system with identities and consensus

### Step 5.1: Agent Identity System (3 days)
```python
# File: src/agents/identity.py (NEW)
# Tasks:
1. Ed25519 keypair generation
2. Agent ID derivation from pubkey
3. Certificate chain for trust
4. Signature on all messages
5. Key rotation protocol
```

### Step 5.2: Discovery Protocol (2 days)
```python
# File: src/agents/discovery.py (NEW)
# Tasks:
1. mDNS for local discovery
2. DHT for global discovery
3. Capability advertisement
4. Service registration
5. Heartbeat mechanism
```

### Step 5.3: Consensus Mechanism (3 days)
```python
# File: src/agents/consensus.py (NEW)
# Tasks:
1. Raft or PBFT implementation
2. Leader election protocol
3. State machine replication
4. Conflict resolution
5. Byzantine fault tolerance
```

### Step 5.4: Distributed Testing (2 days)
```python
# File: tests/test_multi_agent.py (NEW)
# Tasks:
1. Spawn 10+ agent nodes
2. Test discovery and registration
3. Verify consensus operations
4. Simulate network partitions
5. Measure coordination overhead
```

---

## Execution Timeline

| Week | Phase | Deliverable | Verification |
|------|-------|-------------|--------------|
| 1 | HTX Transport | TLS/QUIC + Chrome mimicry | Wireshark indistinguishable from Chrome |
| 2 | HTX + Indistinguishability | Onion routing + cover traffic | Statistical analysis shows uniform distribution |
| 3 | Latency/Jitter | Active measurement + QoS | <500ms adaptation to link changes |
| 4 | SCION Gateway | Protocol implementation | Path verification passes |
| 5 | SCION + Multi-Agent | Full integration | 10-node consensus achieved |
| 6 | Final Integration | All axes complete | Bounty requirements met |

## Dependencies & Prerequisites

### Required Libraries
```bash
pip install aioquic cryptography py-libp2p zeroconf
pip install pynacl scionproto  # If available
```

### Rust Components
```bash
cd betanet-client
cargo build --release
```

### Testing Infrastructure
```bash
# Multi-node test environment
docker-compose up -d  # 10 node test cluster
pytest tests/integration/ -v
```

## Success Criteria

1. **HTX**: Wireshark cannot distinguish from real Chrome traffic
2. **SCION**: Successfully routes through 3+ ASes with path validation
3. **Indistinguishability**: Cover traffic maintains 40% overhead maximum
4. **Latency**: 500ms link change detection and adaptation achieved
5. **Multi-Agent**: 10 agents coordinate with Byzantine fault tolerance

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| SCION complexity | Start with simplified path selection |
| Chrome detection | Use real Chrome TLS library if needed |
| Battery drain | Adaptive cover traffic based on power |
| Latency overhead | Parallel processing and caching |
| Agent consensus | Start with trusted environment, add Byzantine later |
