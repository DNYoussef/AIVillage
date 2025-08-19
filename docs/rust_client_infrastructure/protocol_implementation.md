# Rust Client Infrastructure - Protocol Implementation

## Protocol Stack Overview

The AIVillage Rust Client Infrastructure implements a comprehensive protocol stack designed for secure, anonymous, and resilient P2P communication. This document details the technical implementation of each protocol layer.

## Core Protocol Specifications

### 1. HTX (Hybrid Transport eXchange) Protocol v1.1

**Implementation**: `clients/rust/betanet-htx/src/`

HTX provides frame-based multiplexed communication with integrated security and flow control.

#### Frame Format

```text
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|    Frame Type |                Frame Length (24-bit)          |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                         Stream ID (varint)                    |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                         Frame Payload                         |
|                              ...                              |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

#### Frame Types

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameType {
    Data = 0x00,
    WindowUpdate = 0x08,
    Ping = 0x06,
    Pong = 0x07,
    KeyUpdate = 0x0C,
    Reset = 0x03,
}
```

#### Stream Management

- **Stream ID Assignment**: Odd IDs for initiator, even for responder
- **Flow Control**: Per-stream and connection-level windows
- **Multiplexing**: Up to 2^32 concurrent streams
- **Maximum Frame Size**: 16,777,215 bytes (2^24 - 1)

```rust
impl HtxSession {
    pub fn create_stream(&mut self) -> Result<u32> {
        let stream_id = self.next_stream_id;
        self.next_stream_id += 2; // Maintain odd/even separation
        self.streams.insert(stream_id, HtxStream::new(stream_id));
        Ok(stream_id)
    }
}
```

#### Noise XK Handshake Integration

HTX integrates Noise XK for authenticated key exchange:

1. **Message 1**: Initiator sends ephemeral public key
2. **Message 2**: Responder sends ephemeral + static public keys
3. **Message 3**: Initiator authenticates with static key
4. **Transport**: ChaCha20-Poly1305 encrypted frames

```rust
pub async fn begin_handshake(&mut self) -> Result<Option<Bytes>> {
    if let Some(ref mut noise) = self.noise {
        if noise.is_initiator {
            let fragments = noise.create_message_1()?;
            return Ok(fragments.first().map(|f| f.data.clone()));
        }
    }
    Ok(None)
}
```

### 2. Mixnode Protocol (Nym-Style Anonymous Communication)

**Implementation**: `clients/rust/betanet-mixnode/src/`

Implements Sphinx packet processing for anonymous communication with traffic analysis resistance.

#### Packet Structure

```text
┌─────────────────────────────────────────────────────────────┐
│                    Sphinx Header (1024 bytes)              │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐  │
│  │   Layer 1   │   Layer 2   │   Layer 3   │  Padding    │  │
│  │ (256 bytes) │ (256 bytes) │ (256 bytes) │ (256 bytes) │  │
│  └─────────────┴─────────────┴─────────────┴─────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    Payload (1024 bytes)                    │
│             ChaCha20-Poly1305 Encrypted Data               │
└─────────────────────────────────────────────────────────────┘
```

#### Sphinx Processing

```rust
impl Mixnode for SphinxMixnode {
    async fn process_packet(&self, packet: &[u8]) -> Result<Option<Vec<u8>>> {
        // 1. Decrypt outer layer
        let decrypted = self.decrypt_layer(packet)?;

        // 2. Extract routing information
        let routing_info = self.parse_routing(decrypted)?;

        // 3. Apply VRF delay
        let delay = self.vrf_delay.calculate_delay(&routing_info)?;
        tokio::time::sleep(delay).await;

        // 4. Forward or deliver
        match routing_info.next_hop {
            Some(next) => Ok(Some(self.reencrypt_for_forward(decrypted, next)?)),
            None => Ok(self.deliver_to_destination(decrypted)?),
        }
    }
}
```

#### VRF-Based Delays

Verifiable Random Function (VRF) provides cryptographically secure random delays:

```rust
pub struct VrfDelay {
    secret_key: VrfSecretKey,
    delay_params: DelayParameters,
}

impl VrfDelay {
    pub fn calculate_delay(&self, packet_info: &PacketInfo) -> Result<Duration> {
        let vrf_output = self.secret_key.prove(&packet_info.hash)?;
        let delay_factor = u64::from_be_bytes(vrf_output[..8].try_into()?);
        let delay_ms = (delay_factor % self.delay_params.max_delay_ms) +
                      self.delay_params.min_delay_ms;
        Ok(Duration::from_millis(delay_ms))
    }
}
```

#### Cover Traffic Generation

```rust
pub struct CoverTrafficGenerator {
    rate_limiter: TokenBucket,
    traffic_pattern: TrafficPattern,
}

impl CoverTrafficGenerator {
    pub async fn generate_cover_traffic(&self) -> Result<Vec<u8>> {
        // Generate realistic traffic patterns
        let packet_size = self.traffic_pattern.sample_packet_size();
        let dummy_packet = self.create_dummy_packet(packet_size)?;

        // Schedule according to Poisson process
        let interval = self.traffic_pattern.next_interval();
        tokio::time::sleep(interval).await;

        Ok(dummy_packet)
    }
}
```

### 3. uTLS Protocol (TLS Fingerprinting Evasion)

**Implementation**: `clients/rust/betanet-utls/src/`

Provides Chrome browser TLS fingerprinting mimicry for traffic analysis evasion.

#### Chrome Template Generation

```rust
pub struct TlsTemplate {
    pub version: ChromeVersion,
    pub cipher_suites: Vec<u16>,
    pub extensions: Vec<TlsExtension>,
    pub signature_algorithms: Vec<u16>,
    pub supported_groups: Vec<u16>,
}

impl TlsTemplate {
    pub fn chrome_stable_n2() -> Self {
        Self {
            version: ChromeVersion::new(119, 0, 6045, 123),
            cipher_suites: vec![
                grease::grease_cipher_suite(),
                cipher_suites::TLS_AES_128_GCM_SHA256,
                cipher_suites::TLS_AES_256_GCM_SHA384,
                cipher_suites::TLS_CHACHA20_POLY1305_SHA256,
                cipher_suites::TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
                cipher_suites::TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
            ],
            extensions: Self::chrome_extensions(),
            signature_algorithms: Self::chrome_signature_algorithms(),
            supported_groups: Self::chrome_supported_groups(),
        }
    }
}
```

#### JA3/JA4 Fingerprint Calculation

```rust
pub struct TlsFingerprint {
    pub ja3: String,
    pub ja4: String,
    pub ja3_hash: String,
    pub ja4_hash: String,
}

impl TlsFingerprint {
    pub fn from_client_hello(client_hello: &ClientHello) -> Result<Self> {
        // JA3: version,ciphers,extensions,elliptic_curves,elliptic_curve_point_formats
        let ja3_parts = vec![
            client_hello.version.to_string(),
            client_hello.cipher_suites.iter().map(|c| c.to_string()).collect::<Vec<_>>().join("-"),
            client_hello.extensions.iter().map(|e| e.extension_type.to_string()).collect::<Vec<_>>().join("-"),
            Self::extract_supported_groups(&client_hello.extensions)?,
            Self::extract_ec_point_formats(&client_hello.extensions)?,
        ];

        let ja3 = ja3_parts.join(",");
        let ja3_hash = md5::compute(ja3.as_bytes());

        // JA4: version_sni_alpn_ciphers_extensions_groups
        let ja4 = Self::calculate_ja4(client_hello)?;
        let ja4_hash = sha256::hash(ja4.as_bytes());

        Ok(Self {
            ja3,
            ja4,
            ja3_hash: format!("{:x}", ja3_hash),
            ja4_hash: format!("{:x}", ja4_hash),
        })
    }
}
```

#### GREASE Value Injection

Generate Random Extensions And Sustain Extensibility (GREASE) values:

```rust
pub mod grease {
    pub const GREASE_VALUES: [u16; 16] = [
        0x0A0A, 0x1A1A, 0x2A2A, 0x3A3A, 0x4A4A, 0x5A5A, 0x6A6A, 0x7A7A,
        0x8A8A, 0x9A9A, 0xAAAA, 0xBABA, 0xCACA, 0xDADA, 0xEAEA, 0xFAFA,
    ];

    pub fn get_grease_value(index: usize) -> u16 {
        GREASE_VALUES[index % GREASE_VALUES.len()]
    }
}
```

### 4. DTN Bundle Protocol v7

**Implementation**: `build/core-build/crates/betanet-dtn/src/`

Implements RFC 9171 Bundle Protocol for delay-tolerant networking.

#### Bundle Structure

```text
┌─────────────────────────────────────────────────────────────┐
│                    Primary Block                            │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐  │
│  │   Version   │    Flags    │  CRC Type   │ Destination │  │
│  │ (1 byte)    │ (varint)    │ (varint)    │ (EID)       │  │
│  └─────────────┴─────────────┴─────────────┴─────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                     Payload Block                          │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐  │
│  │ Block Type  │   Flags     │   Length    │   Data      │  │
│  │ (varint)    │ (varint)    │ (varint)    │ (variable)  │  │
│  └─────────────┴─────────────┴─────────────┴─────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

#### Endpoint Identification

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EndpointId {
    pub scheme: String,
    pub specific_part: String,
}

impl EndpointId {
    pub fn node(node_id: impl Into<String>) -> Self {
        Self {
            scheme: "dtn".to_string(),
            specific_part: node_id.into(),
        }
    }

    pub fn service(node_id: impl Into<String>, service: impl Into<String>) -> Self {
        Self {
            scheme: "dtn".to_string(),
            specific_part: format!("{}/{}", node_id.into(), service.into()),
        }
    }
}
```

#### Store-and-Forward Implementation

```rust
pub struct DtnNode {
    node_id: EndpointId,
    storage: Arc<DtnStorage>,
    routing_table: Arc<RwLock<RoutingTable>>,
    bundle_queue: Arc<Mutex<VecDeque<Bundle>>>,
}

impl DtnNode {
    pub async fn forward_bundle(&self, bundle: Bundle) -> Result<()> {
        // Check bundle lifetime
        if bundle.is_expired() {
            return Ok(()); // Drop expired bundles
        }

        // Determine next hop
        let next_hop = self.routing_table.read().await
            .next_hop(&bundle.destination)?;

        match next_hop {
            Some(hop) => {
                // Forward to next hop
                self.send_to_neighbor(hop, bundle).await?;
            }
            None => {
                // Store for later delivery
                self.storage.store_bundle(bundle).await?;
            }
        }

        Ok(())
    }
}
```

### 5. Agent Fabric Messaging Protocol

**Implementation**: `build/core-build/crates/agent-fabric/src/`

Unified messaging API abstracting transport selection and security.

#### Message Types

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMessage {
    pub message_id: String,
    pub message_type: String,
    pub timestamp: u64,
    pub payload: Bytes,
    pub metadata: HashMap<String, String>,
}

impl AgentMessage {
    pub fn new(message_type: impl Into<String>, payload: Bytes) -> Self {
        Self {
            message_id: uuid::Uuid::new_v4().to_string(),
            message_type: message_type.into(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            payload,
            metadata: HashMap::new(),
        }
    }
}
```

#### Transport Selection Algorithm

```rust
impl AgentFabric {
    pub async fn send_message(
        &self,
        to: AgentId,
        message: AgentMessage,
        options: DeliveryOptions,
    ) -> Result<Option<AgentResponse>> {
        match options.transport {
            Transport::Rpc => self.send_via_rpc(to, message, options).await,
            Transport::Bundle => self.send_via_bundle(to, message, options).await,
            Transport::Auto => {
                // Try RPC first, fallback to bundle
                match self.send_via_rpc(to.clone(), message.clone(), options.clone()).await {
                    Ok(response) => Ok(response),
                    Err(AgentFabricError::TransportUnavailable) => {
                        self.send_via_bundle(to, message, options).await
                    }
                    Err(e) => Err(e),
                }
            }
        }
    }
}
```

### 6. Twin Vault CRDT Protocol

**Implementation**: `build/core-build/crates/twin-vault/src/`

Conflict-free replicated data types with cryptographic receipts.

#### CRDT Operations

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LwwMap<K, V> {
    entries: HashMap<K, (V, VectorClock)>,
    actor_id: String,
}

impl<K: Hash + Eq + Clone, V: Clone> LwwMap<K, V> {
    pub fn set(&mut self, key: K, value: V, timestamp: u64) {
        let mut clock = self.get_clock(&key);
        clock.increment(&self.actor_id, timestamp);
        self.entries.insert(key, (value, clock));
    }

    pub fn merge(&mut self, other: &LwwMap<K, V>) -> Vec<CrdtConflict> {
        let mut conflicts = Vec::new();

        for (key, (other_value, other_clock)) in &other.entries {
            match self.entries.get(key) {
                Some((self_value, self_clock)) => {
                    match self_clock.partial_cmp(other_clock) {
                        Some(Ordering::Less) => {
                            // Other is newer
                            self.entries.insert(key.clone(), (other_value.clone(), other_clock.clone()));
                        }
                        Some(Ordering::Greater) => {
                            // Self is newer, keep current
                        }
                        _ => {
                            // Concurrent update - resolve by actor ID
                            if other_clock.latest_actor() > self_clock.latest_actor() {
                                self.entries.insert(key.clone(), (other_value.clone(), other_clock.clone()));
                            }
                            conflicts.push(CrdtConflict::ConcurrentUpdate {
                                key: format!("{:?}", key),
                                self_clock: self_clock.clone(),
                                other_clock: other_clock.clone(),
                            });
                        }
                    }
                }
                None => {
                    // New entry
                    self.entries.insert(key.clone(), (other_value.clone(), other_clock.clone()));
                }
            }
        }

        conflicts
    }
}
```

#### Cryptographic Receipts

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Receipt {
    pub operation_id: String,
    pub twin_id: TwinId,
    pub operation: TwinOperation,
    pub timestamp: u64,
    pub actor_id: AgentId,
    pub signature: Bytes,
    pub success: bool,
}

impl ReceiptSigner {
    pub async fn sign_operation(
        &self,
        twin_id: &TwinId,
        operation: &TwinOperation,
        actor: &AgentId,
        success: bool,
    ) -> Result<Receipt> {
        let receipt = Receipt {
            operation_id: uuid::Uuid::new_v4().to_string(),
            twin_id: twin_id.clone(),
            operation: operation.clone(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            actor_id: actor.clone(),
            signature: Bytes::new(), // Placeholder
            success,
        };

        // Create signature payload
        let payload = self.create_signature_payload(&receipt)?;

        // Sign with Ed25519
        let signature = self.signing_key.sign(&payload);

        Ok(Receipt {
            signature: Bytes::copy_from_slice(signature.as_bytes()),
            ..receipt
        })
    }
}
```

### 7. Federated Learning Protocol

**Implementation**: `build/core-build/crates/federated/src/`

Privacy-preserving federated learning with SecureAgg and differential privacy.

#### Round Coordination

```rust
pub struct RoundOrchestrator {
    session_id: String,
    cohort_manager: CohortManager,
    aggregator: FedAvgAggregator,
    agent_fabric: Arc<AgentFabric>,
}

impl RoundOrchestrator {
    pub async fn execute_round(&mut self, round_id: RoundId) -> Result<AggregationResult> {
        // 1. Select participants
        let participants = self.cohort_manager.select_cohort(&round_id).await?;

        // 2. Distribute global model
        self.distribute_model(&round_id, &participants).await?;

        // 3. Coordinate local training
        let training_results = self.coordinate_training(&round_id, &participants).await?;

        // 4. Aggregate updates
        let global_update = self.aggregator.aggregate(&training_results).await?;

        // 5. Generate receipts
        let receipts = self.generate_round_receipts(&round_id, &training_results).await?;

        Ok(AggregationResult {
            round_id,
            global_model: global_update,
            participants: participants.into_iter().map(|p| p.participant_id).collect(),
            stats: self.calculate_stats(&training_results)?,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
        })
    }
}
```

#### Secure Aggregation

```rust
pub struct SecureAggregation {
    threshold: usize,
    participants: Vec<ParticipantId>,
    shares: HashMap<ParticipantId, SecretShares>,
}

impl SecureAggregation {
    pub async fn aggregate_with_privacy(
        &self,
        model_updates: &[ModelParameters],
    ) -> Result<ModelParameters> {
        // 1. Generate additive secret shares
        let shares = self.generate_additive_shares(model_updates).await?;

        // 2. Distribute shares among participants
        self.distribute_shares(&shares).await?;

        // 3. Collect shares from honest participants
        let collected_shares = self.collect_shares().await?;

        // 4. Reconstruct aggregate if threshold met
        if collected_shares.len() >= self.threshold {
            let aggregate = self.reconstruct_aggregate(&collected_shares)?;
            Ok(aggregate)
        } else {
            Err(FederatedError::InsufficientParticipants {
                got: collected_shares.len() as u32,
                need: self.threshold as u32,
            })
        }
    }
}
```

## Protocol Performance Characteristics

### HTX Protocol
- **Frame Overhead**: 8 bytes header + varint stream ID
- **Encryption Overhead**: 16 bytes ChaCha20-Poly1305 authentication tag
- **Throughput**: 100+ MB/s sustained on modern hardware
- **Latency**: <1ms frame processing time

### Mixnode Protocol
- **Packet Overhead**: 1024-byte Sphinx header + 16-byte auth tag
- **Processing Time**: <1ms average per packet
- **Throughput**: 28,750+ packets/second (2KB packets)
- **Anonymity Set**: Configurable mix strategy (threshold, timed, etc.)

### DTN Protocol
- **Bundle Overhead**: Variable based on EID length and block count
- **Storage**: Efficient CBOR encoding for compact representation
- **Delivery**: Best-effort with custody transfer for reliability
- **Lifetime**: Configurable TTL with automatic expiration

This protocol stack provides a comprehensive foundation for secure, anonymous, and resilient P2P communication across diverse network conditions and threat models.
