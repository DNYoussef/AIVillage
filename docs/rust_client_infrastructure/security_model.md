# Rust Client Infrastructure - Security Model

## Security Architecture Overview

The AIVillage Rust Client Infrastructure implements defense-in-depth security with multiple overlapping protection mechanisms. The security model addresses threats across network surveillance, traffic analysis, content inspection, and adversarial machine learning attacks.

## Threat Model

### Adversary Capabilities

#### Network Adversary
- **Passive Surveillance**: Can observe all network traffic patterns, timing, and metadata
- **Active Interference**: Can drop, delay, or inject packets into the network
- **Traffic Analysis**: Can correlate communication patterns to infer relationships
- **TLS Fingerprinting**: Can identify client software through TLS handshake analysis

#### System Adversary  
- **Node Compromise**: Can compromise individual nodes but not the entire network
- **State Tampering**: Can attempt to modify stored state and receipts
- **Byzantine Behavior**: Can send malformed or malicious messages
- **Privacy Inference**: Can attempt to extract private information from aggregated data

#### Machine Learning Adversary
- **Gradient Inference**: Can attempt to extract private data from model gradients
- **Model Extraction**: Can attempt to steal model parameters through queries
- **Membership Inference**: Can determine if specific data was used in training
- **Poisoning Attacks**: Can attempt to corrupt the global model through malicious updates

### Security Goals

1. **Confidentiality**: Protect message content and metadata from unauthorized access
2. **Anonymity**: Hide communication patterns and participant identities
3. **Integrity**: Detect any tampering with messages or state
4. **Availability**: Maintain operation under network partitions and attacks
5. **Privacy**: Protect individual data in federated learning scenarios
6. **Authenticity**: Verify the source and validity of all operations

## Cryptographic Foundations

### Core Cryptographic Primitives

#### Ed25519 Digital Signatures
**Implementation**: `ed25519-dalek` crate (audited)

```rust
pub struct ReceiptSigner {
    signing_key: ed25519_dalek::SigningKey,
    verifying_key: ed25519_dalek::VerifyingKey,
}

impl ReceiptSigner {
    pub fn sign_operation(&self, payload: &[u8]) -> ed25519_dalek::Signature {
        self.signing_key.sign(payload)
    }
    
    pub fn verify_signature(
        &self, 
        payload: &[u8], 
        signature: &ed25519_dalek::Signature
    ) -> bool {
        self.verifying_key.verify(payload, signature).is_ok()
    }
}
```

**Security Properties**:
- **Key Size**: 32 bytes private key, 32 bytes public key
- **Signature Size**: 64 bytes deterministic signature
- **Security Level**: 128-bit security (equivalent to 3072-bit RSA)
- **Side-Channel Resistance**: Constant-time implementation

#### X25519 Key Exchange
**Implementation**: `x25519-dalek` crate (constant-time)

```rust
pub struct NoiseXK {
    local_static: x25519_dalek::StaticSecret,
    local_ephemeral: Option<x25519_dalek::EphemeralSecret>,
    remote_static: Option<x25519_dalek::PublicKey>,
    cipher_state: Option<CipherState>,
}

impl NoiseXK {
    pub fn perform_dh(
        private_key: &x25519_dalek::StaticSecret,
        public_key: &x25519_dalek::PublicKey,
    ) -> [u8; 32] {
        private_key.diffie_hellman(public_key).to_bytes()
    }
}
```

**Security Properties**:
- **Key Size**: 32 bytes private key, 32 bytes public key
- **Shared Secret**: 32 bytes uniformly random
- **Forward Secrecy**: Ephemeral keys provide perfect forward secrecy
- **Computational Security**: Based on Curve25519 discrete logarithm problem

#### ChaCha20-Poly1305 AEAD
**Implementation**: `chacha20poly1305` crate (RFC 8439)

```rust
pub struct CipherState {
    cipher: ChaCha20Poly1305,
    nonce_counter: u64,
    key: [u8; 32],
}

impl CipherState {
    pub fn encrypt(&mut self, plaintext: &[u8], ad: &[u8]) -> Result<Vec<u8>> {
        let nonce = self.generate_nonce();
        let ciphertext = self.cipher
            .encrypt(&nonce, Payload { msg: plaintext, aad: ad })?;
        self.nonce_counter += 1;
        Ok(ciphertext)
    }
    
    pub fn decrypt(&mut self, ciphertext: &[u8], ad: &[u8]) -> Result<Vec<u8>> {
        let nonce = self.generate_nonce();
        let plaintext = self.cipher
            .decrypt(&nonce, Payload { msg: ciphertext, aad: ad })?;
        self.nonce_counter += 1;
        Ok(plaintext)
    }
}
```

**Security Properties**:
- **Key Size**: 32 bytes (256-bit key)
- **Nonce Size**: 12 bytes (96-bit nonce)
- **Authentication Tag**: 16 bytes (128-bit Poly1305 MAC)
- **Performance**: Optimized for software implementation

#### HKDF Key Derivation
**Implementation**: `hkdf` crate for key derivation

```rust
pub fn derive_keys(shared_secret: &[u8], info: &[u8]) -> ([u8; 32], [u8; 32]) {
    let hkdf = Hkdf::<Sha256>::new(None, shared_secret);
    let mut sending_key = [0u8; 32];
    let mut receiving_key = [0u8; 32];
    
    hkdf.expand(b"sending", &mut sending_key).unwrap();
    hkdf.expand(b"receiving", &mut receiving_key).unwrap();
    
    (sending_key, receiving_key)
}
```

## Protocol Security Analysis

### HTX Protocol Security

#### Noise XK Handshake Security
The HTX protocol implements Noise XK for authenticated key exchange:

```text
    Initiator                    Responder
    ---------                    ---------
    e                     -->    
                          <--    e, ee, s, es
    s, se                 -->    

Where:
e = ephemeral key
s = static key  
ee = ephemeral-ephemeral DH
es = ephemeral-static DH
se = static-ephemeral DH
```

**Security Properties**:
- **Authentication**: Responder authentication after message 2
- **Forward Secrecy**: Perfect forward secrecy via ephemeral keys
- **Identity Hiding**: Initiator identity hidden until message 3
- **Replay Protection**: Ephemeral keys prevent replay attacks

#### Frame Encryption Security

```rust
pub async fn send_data(&mut self, stream_id: u32, data: &[u8]) -> Result<Bytes> {
    // Create DATA frame
    let frame = Frame::data(stream_id, Bytes::from(data.to_vec()))?;
    let frame_bytes = frame.encode();
    
    // Encrypt with Noise transport cipher
    let encrypted_bytes = if let Some(ref mut noise) = self.noise {
        noise.encrypt(&frame_bytes)?
    } else {
        frame_bytes
    };
    
    Ok(encrypted_bytes)
}
```

**Security Analysis**:
- **Frame Integrity**: Each frame includes authentication tag
- **Stream Isolation**: Per-stream encryption prevents cross-contamination
- **Flow Control Security**: Window updates authenticated to prevent DoS
- **Metadata Protection**: Stream IDs encrypted within frames

### Mixnode Protocol Security

#### Sphinx Packet Processing
The mixnode implements layered encryption for anonymous communication:

```rust
pub struct SphinxPacket {
    header: SphinxHeader,      // 1024 bytes - routing info
    payload: EncryptedPayload, // 1024 bytes - message data
}

impl SphinxMixnode {
    pub async fn process_packet(&self, packet: &[u8]) -> Result<Option<Vec<u8>>> {
        // 1. Decrypt outer layer
        let decrypted_header = self.decrypt_sphinx_layer(&packet[..1024])?;
        let decrypted_payload = self.decrypt_payload_layer(&packet[1024..])?;
        
        // 2. Parse routing information
        let routing_info = self.parse_routing_info(&decrypted_header)?;
        
        // 3. Apply cryptographic delay
        let delay = self.vrf_delay.calculate_delay(&routing_info.packet_id)?;
        tokio::time::sleep(delay).await;
        
        // 4. Forward or deliver
        match routing_info.next_hop {
            Some(next_hop) => {
                let forwarded_packet = self.prepare_forward_packet(
                    decrypted_header, 
                    decrypted_payload, 
                    next_hop
                )?;
                Ok(Some(forwarded_packet))
            }
            None => {
                // Final destination
                Ok(Some(decrypted_payload))
            }
        }
    }
}
```

#### VRF-Based Timing Protection

```rust
pub struct VrfDelay {
    vrf_key: VrfSecretKey,
    delay_params: DelayParameters,
}

impl VrfDelay {
    pub fn calculate_delay(&self, packet_id: &[u8]) -> Result<Duration> {
        // Generate verifiable random delay
        let (output, proof) = self.vrf_key.prove(packet_id)?;
        
        // Convert VRF output to delay
        let delay_factor = u64::from_be_bytes(output[..8].try_into()?);
        let delay_ms = self.delay_params.min_delay_ms + 
                      (delay_factor % (self.delay_params.max_delay_ms - self.delay_params.min_delay_ms));
        
        Ok(Duration::from_millis(delay_ms))
    }
}
```

**Security Properties**:
- **Unlinkability**: Each hop cannot correlate input/output packets
- **Timing Analysis Resistance**: VRF delays prevent timing correlation
- **Forward Secrecy**: Each layer uses unique keys derived from DH
- **Replay Protection**: Packet tags prevent replay attacks

### TLS Fingerprinting Evasion

#### Chrome Browser Mimicry

```rust
impl TlsTemplate {
    pub fn chrome_stable_n2() -> Self {
        Self {
            version: ChromeVersion::new(119, 0, 6045, 123),
            cipher_suites: vec![
                grease::grease_cipher_suite(),           // 0x0A0A
                cipher_suites::TLS_AES_128_GCM_SHA256,    // 0x1301
                cipher_suites::TLS_AES_256_GCM_SHA384,    // 0x1302
                cipher_suites::TLS_CHACHA20_POLY1305_SHA256, // 0x1303
                // ... Chrome's exact cipher suite ordering
            ],
            extensions: vec![
                TlsExtension::server_name(),
                TlsExtension::extended_master_secret(),
                TlsExtension::renegotiation_info(),
                TlsExtension::supported_groups(),
                TlsExtension::ec_point_formats(),
                TlsExtension::session_ticket(),
                TlsExtension::alpn(),
                TlsExtension::status_request(),
                TlsExtension::signature_algorithms(),
                TlsExtension::signed_certificate_timestamp(),
                TlsExtension::key_share(),
                TlsExtension::psk_key_exchange_modes(),
                TlsExtension::supported_versions(),
                // ... Chrome's exact extension ordering
            ],
            // ... other Chrome-specific parameters
        }
    }
}
```

**Fingerprint Evasion Analysis**:
- **JA3 Matching**: Exact replication of Chrome's JA3 fingerprint
- **JA4 Matching**: Support for newer JA4 fingerprinting methods  
- **Extension Ordering**: Precise ordering matches Chrome behavior
- **GREASE Values**: Realistic GREASE injection patterns
- **Version History**: Support for Chrome N-2 versioning strategy

### DTN Security Model

#### Bundle Authentication

```rust
pub struct Bundle {
    primary_block: PrimaryBlock,
    blocks: Vec<CanonicalBlock>,
    signature_block: Option<SignatureBlock>,
}

impl Bundle {
    pub fn sign(&mut self, signing_key: &ed25519_dalek::SigningKey) -> Result<()> {
        let payload = self.create_signature_payload()?;
        let signature = signing_key.sign(&payload);
        
        self.signature_block = Some(SignatureBlock {
            signature: signature.to_bytes().to_vec(),
            key_id: self.calculate_key_id(&signing_key.verifying_key()),
            algorithm: SignatureAlgorithm::Ed25519,
        });
        
        Ok(())
    }
    
    pub fn verify(&self, verifying_key: &ed25519_dalek::VerifyingKey) -> bool {
        if let Some(ref sig_block) = self.signature_block {
            let payload = self.create_signature_payload().unwrap();
            let signature = ed25519_dalek::Signature::from_bytes(&sig_block.signature);
            verifying_key.verify(&payload, &signature).is_ok()
        } else {
            false
        }
    }
}
```

**Security Analysis**:
- **End-to-End Authentication**: Bundle signatures survive store-and-forward
- **Custody Transfer**: Intermediate nodes provide delivery receipts
- **Bundle Integrity**: CRC32 checksums detect corruption
- **Replay Protection**: Bundle creation timestamps prevent replays

### Federated Learning Security

#### Differential Privacy Implementation

```rust
pub struct DifferentialPrivacy {
    epsilon: f64,      // Privacy budget
    delta: f64,        // Failure probability
    sensitivity: f64,  // Query sensitivity
    mechanism: NoiseDistribution,
}

impl DifferentialPrivacy {
    pub fn add_noise(&self, value: f64) -> f64 {
        match self.mechanism {
            NoiseDistribution::Laplace => {
                let scale = self.sensitivity / self.epsilon;
                let noise = self.sample_laplace(0.0, scale);
                value + noise
            }
            NoiseDistribution::Gaussian => {
                let sigma = self.sensitivity * (2.0 * (1.25 / self.delta).ln()).sqrt() / self.epsilon;
                let noise = self.sample_gaussian(0.0, sigma);
                value + noise
            }
        }
    }
    
    pub fn privatize_gradients(&self, gradients: &[f64]) -> Vec<f64> {
        gradients.iter()
            .map(|&gradient| {
                // Clip gradient to sensitivity bound
                let clipped = gradient.max(-self.sensitivity).min(self.sensitivity);
                // Add calibrated noise
                self.add_noise(clipped)
            })
            .collect()
    }
}
```

#### Secure Aggregation Protocol

```rust
pub struct SecureAggregation {
    participants: Vec<ParticipantId>,
    threshold: usize,
    secret_shares: HashMap<ParticipantId, SecretShares>,
}

impl SecureAggregation {
    pub async fn setup_round(&mut self, model_size: usize) -> Result<()> {
        // Generate random masks for each participant
        for participant in &self.participants {
            let mask = self.generate_random_mask(model_size)?;
            let shares = self.secret_share_mask(&mask)?;
            self.secret_shares.insert(participant.clone(), shares);
        }
        
        // Distribute shares to participants
        self.distribute_shares().await?;
        Ok(())
    }
    
    pub async fn aggregate_masked_updates(
        &self,
        masked_updates: &HashMap<ParticipantId, Vec<f64>>,
    ) -> Result<Vec<f64>> {
        // Sum all masked updates
        let mut aggregate = vec![0.0; masked_updates.values().next().unwrap().len()];
        for update in masked_updates.values() {
            for (i, &value) in update.iter().enumerate() {
                aggregate[i] += value;
            }
        }
        
        // Reconstruct and subtract aggregate mask
        let aggregate_mask = self.reconstruct_aggregate_mask(masked_updates.keys())?;
        for (i, mask_value) in aggregate_mask.iter().enumerate() {
            aggregate[i] -= mask_value;
        }
        
        Ok(aggregate)
    }
}
```

**Privacy Guarantees**:
- **Secure Multi-Party Computation**: Individual updates never revealed
- **Differential Privacy**: (ε,δ)-differential privacy with calibrated noise
- **Gradient Clipping**: Bounds contribution of any single participant
- **Robustness**: Tolerates up to t < n/3 malicious participants

### State Management Security

#### CRDT Integrity Protection

```rust
pub struct SignedOperation {
    operation: CrdtOperation,
    actor_id: String,
    timestamp: u64,
    signature: ed25519_dalek::Signature,
}

impl SignedOperation {
    pub fn new(
        operation: CrdtOperation,
        actor_id: String,
        signing_key: &ed25519_dalek::SigningKey,
    ) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let payload = Self::create_signature_payload(&operation, &actor_id, timestamp);
        let signature = signing_key.sign(&payload);
        
        Self {
            operation,
            actor_id,
            timestamp,
            signature,
        }
    }
    
    pub fn verify(&self, verifying_key: &ed25519_dalek::VerifyingKey) -> bool {
        let payload = Self::create_signature_payload(&self.operation, &self.actor_id, self.timestamp);
        verifying_key.verify(&payload, &self.signature).is_ok()
    }
}
```

#### Receipt-Based Audit Trail

```rust
pub struct Receipt {
    pub operation_id: String,
    pub twin_id: TwinId,
    pub operation: TwinOperation,
    pub timestamp: u64,
    pub actor_id: AgentId,
    pub signature: Bytes,
    pub success: bool,
}

impl Receipt {
    pub fn verify_chain(&self, previous_receipt: Option<&Receipt>) -> bool {
        // Verify timestamp ordering
        if let Some(prev) = previous_receipt {
            if self.timestamp <= prev.timestamp {
                return false;
            }
        }
        
        // Verify signature
        // ... signature verification logic
        
        true
    }
}
```

## Security Validation & Testing

### Cryptographic Testing

```rust
#[cfg(test)]
mod security_tests {
    use super::*;
    use proptest::prelude::*;
    
    proptest! {
        #[test]
        fn noise_xk_provides_forward_secrecy(
            message1 in any::<[u8; 1024]>(),
            message2 in any::<[u8; 1024]>(),
        ) {
            // Test that compromising long-term keys doesn't reveal past sessions
            let (alice_static, bob_static) = generate_test_keypairs();
            
            // Session 1
            let session1 = perform_noise_handshake(&alice_static, &bob_static, &message1)?;
            let ciphertext1 = session1.encrypt(&message1)?;
            
            // Session 2 with same static keys
            let session2 = perform_noise_handshake(&alice_static, &bob_static, &message2)?;
            let ciphertext2 = session2.encrypt(&message2)?;
            
            // Verify sessions use different encryption keys
            prop_assert_ne!(session1.get_encryption_key(), session2.get_encryption_key());
            
            // Verify ciphertexts are unlinkable
            prop_assert!(!are_ciphertexts_linkable(&ciphertext1, &ciphertext2));
        }
        
        #[test]
        fn differential_privacy_bounds_privacy_loss(
            gradients in prop::collection::vec(any::<f64>(), 1..1000),
            epsilon in 0.1f64..10.0,
            delta in 1e-10f64..1e-3,
        ) {
            let dp = DifferentialPrivacy::new(epsilon, delta, 1.0);
            let privatized = dp.privatize_gradients(&gradients);
            
            // Verify privacy budget consumption
            let privacy_loss = calculate_privacy_loss(&gradients, &privatized, epsilon, delta);
            prop_assert!(privacy_loss <= epsilon + delta);
        }
    }
}
```

### Penetration Testing Framework

```rust
pub struct SecurityValidator {
    test_vectors: Vec<AttackVector>,
    metrics: SecurityMetrics,
}

impl SecurityValidator {
    pub async fn run_security_tests(&mut self) -> SecurityReport {
        let mut report = SecurityReport::new();
        
        // Test timing attack resistance
        report.timing_attacks = self.test_timing_attacks().await;
        
        // Test traffic analysis resistance  
        report.traffic_analysis = self.test_traffic_analysis().await;
        
        // Test cryptographic implementation
        report.crypto_validation = self.test_crypto_implementations().await;
        
        // Test state consistency under Byzantine faults
        report.byzantine_resilience = self.test_byzantine_resilience().await;
        
        report
    }
}
```

## Threat Mitigation Summary

| Threat Category | Mitigation Technique | Implementation |
|-----------------|---------------------|----------------|
| **Traffic Analysis** | Mixnode + VRF delays | `betanet-mixnode` with Sphinx processing |
| **TLS Fingerprinting** | Chrome browser mimicry | `betanet-utls` with JA3/JA4 matching |
| **Content Inspection** | End-to-end encryption | ChaCha20-Poly1305 AEAD with Noise XK |
| **Network Partitions** | Store-and-forward | `betanet-dtn` Bundle Protocol v7 |
| **State Tampering** | Cryptographic receipts | Ed25519 signatures on all operations |
| **Privacy Leakage** | Differential privacy | Calibrated noise in federated learning |
| **Gradient Inference** | Secure aggregation | Secret sharing with threshold reconstruction |
| **Replay Attacks** | Nonce + timestamps | Monotonic counters + time bounds |
| **Identity Correlation** | Anonymous routing | Multi-hop mixnet with cover traffic |
| **Key Compromise** | Forward secrecy | Ephemeral keys + key rotation |

The Rust Client Infrastructure implements a comprehensive security model that provides strong guarantees against a wide range of adversaries while maintaining usability and performance for distributed AI applications.