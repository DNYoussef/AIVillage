//! Access Ticket System for HTX Authentication
//!
//! Implements access tickets with:
//! - HKDF-based ticket derivation
//! - Token bucket rate limiting
//! - Replay protection with nonce tracking
//! - Ed25519 signature validation
//! - Ticket expiration and renewal

use bytes::{Buf, BufMut, Bytes, BytesMut};
use rand::rngs::OsRng;
use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

/// Maximum ticket size (64 bytes per spec)
const MAX_TICKET_SIZE: usize = 64;
/// Minimum ticket size (24 bytes per spec)
const MIN_TICKET_SIZE: usize = 24;
/// Duplicate detection window (2 hours)
const DUPLICATE_WINDOW_SECONDS: u64 = 2 * 3600;
/// Ed25519 key length
const ED25519_KEY_LEN: usize = 32;
/// Ed25519 signature length
const ED25519_SIG_LEN: usize = 64;

// Stub types for Ed25519 (would normally use ed25519-dalek)
type SigningKey = [u8; 32];
type VerifyingKey = [u8; 32];
type Signature = [u8; 64];

/// Access ticket errors
#[derive(Debug, Error)]
pub enum TicketError {
    #[error("Invalid ticket size: {0} (must be between {1} and {2})")]
    InvalidSize(usize, usize, usize),

    #[error("Ticket expired at {0}")]
    Expired(u64),

    #[error("Invalid signature")]
    InvalidSignature,

    #[error("Replay detected for nonce: {0:?}")]
    ReplayDetected(Bytes),

    #[error("Rate limited: {0}")]
    RateLimited(String),

    #[error("Unknown issuer: {0}")]
    UnknownIssuer(String),

    #[error("Malformed ticket: {0}")]
    Malformed(String),

    #[error("Invalid key length: expected {expected}, got {actual}")]
    InvalidKeyLength { expected: usize, actual: usize },

    #[error("Cryptographic error: {0}")]
    Crypto(String),

    #[error("Serialization error: {0}")]
    Serialization(String),
}

/// Access ticket types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TicketType {
    Standard = 0,
    Premium = 1,
    Burst = 2,
    Maintenance = 3,
}

impl TryFrom<u8> for TicketType {
    type Error = TicketError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Standard),
            1 => Ok(Self::Premium),
            2 => Ok(Self::Burst),
            3 => Ok(Self::Maintenance),
            _ => Err(TicketError::Malformed(format!(
                "Invalid ticket type: {}",
                value
            ))),
        }
    }
}

/// Ticket validation status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TicketStatus {
    Valid,
    Expired,
    InvalidSignature,
    ReplayDetected,
    RateLimited,
    Malformed,
    UnknownIssuer,
}

/// Token bucket rate limiting configuration
#[derive(Debug, Clone)]
pub struct TokenBucketConfig {
    pub capacity: u32,
    pub refill_rate: f64, // tokens per second
    pub burst_capacity: u32,
    pub window_seconds: u32,
}

impl TokenBucketConfig {
    pub fn new(capacity: u32, refill_rate: f64, burst_capacity: u32) -> Self {
        Self {
            capacity,
            refill_rate,
            burst_capacity,
            window_seconds: 60,
        }
    }

    /// Get configuration for ticket type
    pub fn for_ticket_type(ticket_type: TicketType) -> Self {
        match ticket_type {
            TicketType::Standard => Self::new(100, 10.0, 20),
            TicketType::Premium => Self::new(500, 50.0, 100),
            TicketType::Burst => Self::new(1000, 20.0, 500),
            TicketType::Maintenance => Self::new(50, 5.0, 10),
        }
    }
}

/// Token bucket for rate limiting
#[derive(Debug, Clone)]
pub struct TokenBucket {
    config: TokenBucketConfig,
    tokens: f64,
    last_refill: u64,
}

impl TokenBucket {
    pub fn new(config: TokenBucketConfig) -> Self {
        Self {
            tokens: config.capacity as f64,
            config,
            last_refill: current_timestamp(),
        }
    }

    /// Attempt to consume tokens from bucket
    pub fn consume(&mut self, tokens_requested: u32) -> bool {
        let now = current_timestamp();

        // Refill tokens based on elapsed time
        let elapsed = (now - self.last_refill) as f64;
        self.tokens = (self.config.capacity as f64)
            .min(self.tokens + (elapsed * self.config.refill_rate));
        self.last_refill = now;

        // Check if we have enough tokens
        if self.tokens >= tokens_requested as f64 {
            self.tokens -= tokens_requested as f64;
            true
        } else {
            false
        }
    }

    /// Get current bucket status
    pub fn status(&self) -> TokenBucketStatus {
        TokenBucketStatus {
            available_tokens: self.tokens as u32,
            capacity: self.config.capacity,
            refill_rate: self.config.refill_rate,
            utilization: (self.config.capacity as f64 - self.tokens) / self.config.capacity as f64,
        }
    }
}

/// Token bucket status
#[derive(Debug, Clone)]
pub struct TokenBucketStatus {
    pub available_tokens: u32,
    pub capacity: u32,
    pub refill_rate: f64,
    pub utilization: f64,
}

/// Access ticket structure
#[derive(Debug, Clone)]
pub struct AccessTicket {
    pub ticket_id: Bytes,
    pub issuer_id: String,
    pub subject_id: String,
    pub ticket_type: TicketType,
    pub max_bandwidth_bps: u32,
    pub max_connections: u32,
    pub allowed_protocols: Vec<String>,
    pub issued_at: u64,       // Unix timestamp in milliseconds
    pub expires_at: u64,      // Unix timestamp in milliseconds
    pub nonce: Bytes,         // 12 bytes for replay protection
    pub sequence_number: u64,
    pub signature: Bytes,     // Ed25519 signature
    pub issuer_public_key: Bytes, // Ed25519 public key
}

impl AccessTicket {
    /// Create new access ticket
    pub fn new(
        issuer_id: String,
        subject_id: String,
        ticket_type: TicketType,
        validity_seconds: u64,
    ) -> Self {
        let now = current_timestamp() * 1000; // Convert to milliseconds
        let mut nonce = vec![0u8; 12];
        let mut nonce_data = vec![0u8; 8];
        rand::Rng::fill(&mut OsRng, &mut nonce_data[..]);
        nonce[..8].copy_from_slice(&nonce_data);

        Self {
            ticket_id: Bytes::from((0..16).map(|_| rand::random::<u8>()).collect::<Vec<u8>>()),
            issuer_id,
            subject_id,
            ticket_type,
            max_bandwidth_bps: Self::default_bandwidth(ticket_type),
            max_connections: Self::default_connections(ticket_type),
            allowed_protocols: Self::default_protocols(ticket_type),
            issued_at: now,
            expires_at: now + (validity_seconds * 1000),
            nonce: Bytes::from(nonce),
            sequence_number: now, // Use timestamp as sequence number
            signature: Bytes::new(),
            issuer_public_key: Bytes::new(),
        }
    }

    /// Check if ticket is expired
    pub fn is_expired(&self) -> bool {
        let now = current_timestamp() * 1000;
        now > self.expires_at
    }

    /// Get remaining validity time in seconds
    pub fn time_remaining(&self) -> u64 {
        let now = current_timestamp() * 1000;
        if now >= self.expires_at {
            0
        } else {
            (self.expires_at - now) / 1000
        }
    }

    /// Serialize ticket to bytes for signing/transmission
    pub fn serialize(&self) -> Result<Bytes, TicketError> {
        let mut buf = BytesMut::new();

        // Compact format to fit within 64 bytes
        // Ticket ID (8 bytes - shortened)
        buf.put_slice(&self.ticket_id[..8]);

        // Issuer ID (8 bytes, truncated)
        let issuer_bytes = self.issuer_id.as_bytes();
        buf.put_slice(&issuer_bytes[..issuer_bytes.len().min(8)]);
        buf.put_bytes(0, 8 - issuer_bytes.len().min(8));

        // Subject ID (8 bytes, truncated)
        let subject_bytes = self.subject_id.as_bytes();
        buf.put_slice(&subject_bytes[..subject_bytes.len().min(8)]);
        buf.put_bytes(0, 8 - subject_bytes.len().min(8));

        // Ticket type (1 byte)
        buf.put_u8(self.ticket_type as u8);

        // Timestamps (4 bytes each, using seconds since epoch)
        buf.put_u32((self.issued_at / 1000) as u32);
        buf.put_u32((self.expires_at / 1000) as u32);

        // Nonce (8 bytes - shortened)
        buf.put_slice(&self.nonce[..8]);

        // Bandwidth and connections (4 bytes each)
        buf.put_u32(self.max_bandwidth_bps);
        buf.put_u32(self.max_connections);

        // Current size: 8+8+8+1+4+4+8+4+4 = 49 bytes
        // Add minimal padding to reach min size
        let current_size = buf.len();
        if current_size < MIN_TICKET_SIZE {
            let padding_size = MIN_TICKET_SIZE - current_size;
            let padding: Vec<u8> = (0..padding_size).map(|_| 0u8).collect();
            buf.put_slice(&padding);
        }

        let final_size = buf.len();
        if final_size > MAX_TICKET_SIZE {
            return Err(TicketError::InvalidSize(final_size, MIN_TICKET_SIZE, MAX_TICKET_SIZE));
        }

        Ok(buf.freeze())
    }

    /// Deserialize ticket from bytes
    pub fn deserialize(data: &[u8]) -> Result<Self, TicketError> {
        if data.len() < MIN_TICKET_SIZE {
            return Err(TicketError::InvalidSize(data.len(), MIN_TICKET_SIZE, MAX_TICKET_SIZE));
        }

        let mut buf = Bytes::from(data.to_vec());

        // Extract compact fields
        let mut ticket_id_full = vec![0u8; 16];
        let ticket_id_partial = buf.split_to(8);
        ticket_id_full[..8].copy_from_slice(&ticket_id_partial);
        let ticket_id = Bytes::from(ticket_id_full);

        let issuer_bytes = buf.split_to(8);
        let issuer_id = String::from_utf8_lossy(&issuer_bytes)
            .trim_end_matches('\0')
            .to_string();

        let subject_bytes = buf.split_to(8);
        let subject_id = String::from_utf8_lossy(&subject_bytes)
            .trim_end_matches('\0')
            .to_string();

        let ticket_type = TicketType::try_from(buf.get_u8())?;

        // Convert 32-bit timestamps back to 64-bit milliseconds
        let issued_at = (buf.get_u32() as u64) * 1000;
        let expires_at = (buf.get_u32() as u64) * 1000;

        let mut nonce_full = vec![0u8; 12];
        let nonce_partial = buf.split_to(8);
        nonce_full[..8].copy_from_slice(&nonce_partial);
        let nonce = Bytes::from(nonce_full);

        let max_bandwidth_bps = buf.get_u32();
        let max_connections = buf.get_u32();

        Ok(Self {
            ticket_id,
            issuer_id,
            subject_id,
            ticket_type,
            max_bandwidth_bps,
            max_connections,
            allowed_protocols: Self::default_protocols(ticket_type),
            issued_at,
            expires_at,
            nonce,
            sequence_number: issued_at, // Use timestamp as sequence number
            signature: Bytes::new(),
            issuer_public_key: Bytes::new(),
        })
    }

    /// Sign ticket with Ed25519 private key
    pub fn sign(&mut self, private_key: &[u8]) -> Result<(), TicketError> {
        use ed25519_dalek::{SigningKey, Signature, Signer, VerifyingKey};

        if private_key.len() != ED25519_KEY_LEN {
            return Err(TicketError::InvalidKeyLength {
                expected: ED25519_KEY_LEN,
                actual: private_key.len(),
            });
        }

        // SECURITY FIX: Use proper Ed25519 signing
        let signing_key = SigningKey::from_bytes(
            private_key.try_into()
                .map_err(|_| TicketError::InvalidKey("Invalid private key format".to_string()))?
        );

        // Derive the correct public key from the private key
        let verifying_key: VerifyingKey = (&signing_key).into();
        self.issuer_public_key = Bytes::from(verifying_key.to_bytes().to_vec());

        // Create ticket message to sign
        let mut message = Vec::new();
        message.extend_from_slice(&self.subject);
        message.extend_from_slice(&self.issued_at.to_le_bytes());
        message.extend_from_slice(&self.expires_at.to_le_bytes());
        message.push(self.ticket_type as u8);
        message.extend_from_slice(&self.nonce);

        // Sign the message
        let signature: Signature = signing_key.sign(&message);
        self.signature = Bytes::from(signature.to_bytes().to_vec());

        Ok(())
    }

    /// Verify ticket signature using Ed25519
    pub fn verify(&self, public_key: &[u8]) -> Result<bool, TicketError> {
        use ed25519_dalek::{VerifyingKey, Signature, Verifier};

        if public_key.len() != ED25519_KEY_LEN {
            return Err(TicketError::InvalidKeyLength {
                expected: ED25519_KEY_LEN,
                actual: public_key.len(),
            });
        }

        if self.signature.len() != ED25519_SIG_LEN {
            return Ok(false);
        }

        // SECURITY FIX: Use proper Ed25519 verification
        let verifying_key = VerifyingKey::from_bytes(
            public_key.try_into()
                .map_err(|_| TicketError::InvalidKey("Invalid public key format".to_string()))?
        ).map_err(|_| TicketError::InvalidKey("Invalid public key".to_string()))?;

        let signature = Signature::from_bytes(
            self.signature.as_ref().try_into()
                .map_err(|_| TicketError::InvalidKey("Invalid signature format".to_string()))?
        );

        // Recreate the message that was signed
        let mut message = Vec::new();
        message.extend_from_slice(&self.subject);
        message.extend_from_slice(&self.issued_at.to_le_bytes());
        message.extend_from_slice(&self.expires_at.to_le_bytes());
        message.push(self.ticket_type as u8);
        message.extend_from_slice(&self.nonce);

        // Verify the signature
        Ok(verifying_key.verify(&message, &signature).is_ok())
    }

    fn default_bandwidth(ticket_type: TicketType) -> u32 {
        match ticket_type {
            TicketType::Standard => 1_000_000,     // 1 Mbps
            TicketType::Premium => 10_000_000,     // 10 Mbps
            TicketType::Burst => 100_000_000,      // 100 Mbps
            TicketType::Maintenance => 100_000,    // 100 Kbps
        }
    }

    fn default_connections(ticket_type: TicketType) -> u32 {
        match ticket_type {
            TicketType::Standard => 10,
            TicketType::Premium => 50,
            TicketType::Burst => 5,
            TicketType::Maintenance => 2,
        }
    }

    fn default_protocols(ticket_type: TicketType) -> Vec<String> {
        match ticket_type {
            TicketType::Standard => vec!["htx".to_string()],
            TicketType::Premium => vec!["htx".to_string(), "quic".to_string()],
            TicketType::Burst => vec!["htx".to_string(), "quic".to_string(), "direct".to_string()],
            TicketType::Maintenance => vec!["htx".to_string()],
        }
    }
}

/// Access ticket manager for validation and rate limiting
pub struct AccessTicketManager {
    trusted_issuers: HashMap<String, VerifyingKey>,
    used_nonces: HashSet<Bytes>,
    nonce_timestamps: HashMap<Bytes, u64>,
    rate_limiters: HashMap<String, TokenBucket>,
    active_tickets: HashMap<String, AccessTicket>,
    hour_keys: HashMap<u64, Bytes>,
    master_secret: Bytes,
    max_nonce_history: usize,
    stats: TicketStats,
}

/// Ticket system statistics
#[derive(Debug, Clone, Default)]
pub struct TicketStats {
    pub tickets_validated: u64,
    pub tickets_rejected: u64,
    pub replay_attempts: u64,
    pub rate_limit_hits: u64,
    pub expired_tickets: u64,
}

impl AccessTicketManager {
    /// Create new ticket manager
    pub fn new(max_nonce_history: usize) -> Self {
        let mut master_secret = vec![0u8; 32];
        rand::Rng::fill(&mut OsRng, &mut master_secret[..]);

        Self {
            trusted_issuers: HashMap::new(),
            used_nonces: HashSet::new(),
            nonce_timestamps: HashMap::new(),
            rate_limiters: HashMap::new(),
            active_tickets: HashMap::new(),
            hour_keys: HashMap::new(),
            master_secret: Bytes::from(master_secret),
            max_nonce_history,
            stats: TicketStats::default(),
        }
    }

    /// Add trusted issuer public key
    pub fn add_trusted_issuer(&mut self, issuer_id: String, public_key: &[u8]) -> Result<(), TicketError> {
        if public_key.len() != ED25519_KEY_LEN {
            return Err(TicketError::InvalidKeyLength {
                expected: ED25519_KEY_LEN,
                actual: public_key.len(),
            });
        }

        let mut verifying_key = [0u8; 32];
        verifying_key.copy_from_slice(public_key);

        self.trusted_issuers.insert(issuer_id, verifying_key);
        Ok(())
    }

    /// Validate access ticket comprehensively
    pub fn validate_ticket(&mut self, ticket: &AccessTicket) -> TicketStatus {
        // 1. Check expiration
        if ticket.is_expired() {
            self.stats.expired_tickets += 1;
            return TicketStatus::Expired;
        }

        // 2. Check issuer trust
        let issuer_key = match self.trusted_issuers.get(&ticket.issuer_id) {
            Some(key) => key,
            None => return TicketStatus::UnknownIssuer,
        };

        // 3. Verify signature
        match ticket.verify(issuer_key) {
            Ok(true) => {},
            Ok(false) => {
                self.stats.tickets_rejected += 1;
                return TicketStatus::InvalidSignature;
            },
            Err(_) => {
                self.stats.tickets_rejected += 1;
                return TicketStatus::Malformed;
            }
        }

        // 4. Check replay protection
        if self.is_duplicate_within_window(&ticket.nonce) {
            self.stats.replay_attempts += 1;
            return TicketStatus::ReplayDetected;
        }

        // 5. Check rate limiting
        if !self.check_rate_limit(ticket) {
            self.stats.rate_limit_hits += 1;
            return TicketStatus::RateLimited;
        }

        // Record successful validation
        self.record_ticket_usage(ticket);
        self.stats.tickets_validated += 1;
        TicketStatus::Valid
    }

    /// Issue new access ticket
    pub fn issue_ticket(
        &mut self,
        issuer_id: String,
        subject_id: String,
        ticket_type: TicketType,
        validity_seconds: u64,
        private_key: &[u8],
    ) -> Result<AccessTicket, TicketError> {
        if !self.trusted_issuers.contains_key(&issuer_id) {
            return Err(TicketError::UnknownIssuer(issuer_id));
        }

        let mut ticket = AccessTicket::new(issuer_id, subject_id, ticket_type, validity_seconds);
        ticket.sign(private_key)?;

        Ok(ticket)
    }

    /// Get subject status
    pub fn get_subject_status(&self, subject_id: &str) -> Option<SubjectStatus> {
        let ticket = self.active_tickets.get(subject_id)?;
        let rate_limiter = self.rate_limiters.get(subject_id);

        Some(SubjectStatus {
            subject_id: subject_id.to_string(),
            ticket_type: ticket.ticket_type,
            time_remaining: ticket.time_remaining(),
            max_bandwidth_bps: ticket.max_bandwidth_bps,
            max_connections: ticket.max_connections,
            rate_limiter_status: rate_limiter.map(|rl| rl.status()),
        })
    }

    /// Get system statistics
    pub fn statistics(&self) -> ManagerStats {
        ManagerStats {
            tickets: self.stats.clone(),
            trusted_issuers: self.trusted_issuers.len(),
            active_tickets: self.active_tickets.len(),
            rate_limiters: self.rate_limiters.len(),
            nonce_history_size: self.used_nonces.len(),
        }
    }

    /// Clean up expired data
    pub fn cleanup_expired(&mut self) -> usize {
        let mut cleaned = 0;
        let now = current_timestamp();

        // Clean expired tickets
        let expired_subjects: Vec<_> = self
            .active_tickets
            .iter()
            .filter(|(_, ticket)| ticket.is_expired())
            .map(|(subject, _)| subject.clone())
            .collect();

        for subject in expired_subjects {
            self.active_tickets.remove(&subject);
            cleaned += 1;
        }

        // Clean old nonces
        let expired_nonces: Vec<_> = self
            .nonce_timestamps
            .iter()
            .filter(|(_, &timestamp)| (now - timestamp) > DUPLICATE_WINDOW_SECONDS)
            .map(|(nonce, _)| nonce.clone())
            .collect();

        for nonce in expired_nonces {
            self.used_nonces.remove(&nonce);
            self.nonce_timestamps.remove(&nonce);
            cleaned += 1;
        }

        // Clean inactive rate limiters
        let inactive_subjects: Vec<_> = self
            .rate_limiters
            .iter()
            .filter(|(_, limiter)| (now - limiter.last_refill) > 3600) // 1 hour
            .map(|(subject, _)| subject.clone())
            .collect();

        for subject in inactive_subjects {
            self.rate_limiters.remove(&subject);
            cleaned += 1;
        }

        cleaned
    }

    fn is_duplicate_within_window(&self, nonce: &Bytes) -> bool {
        if let Some(&timestamp) = self.nonce_timestamps.get(nonce) {
            let now = current_timestamp();
            (now - timestamp) < DUPLICATE_WINDOW_SECONDS
        } else {
            false
        }
    }

    fn check_rate_limit(&mut self, ticket: &AccessTicket) -> bool {
        let subject_id = &ticket.subject_id;

        // Get or create rate limiter
        if !self.rate_limiters.contains_key(subject_id) {
            let config = TokenBucketConfig::for_ticket_type(ticket.ticket_type);
            self.rate_limiters
                .insert(subject_id.clone(), TokenBucket::new(config));
        }

        let rate_limiter = self.rate_limiters.get_mut(subject_id).unwrap();

        // Calculate tokens needed based on bandwidth
        let tokens_needed = (ticket.max_bandwidth_bps / 100_000).max(1); // 1 token per 100KB/s

        rate_limiter.consume(tokens_needed)
    }

    fn record_ticket_usage(&mut self, ticket: &AccessTicket) {
        let now = current_timestamp();

        // Record nonce usage
        self.used_nonces.insert(ticket.nonce.clone());
        self.nonce_timestamps.insert(ticket.nonce.clone(), now);

        // Limit nonce history size
        if self.used_nonces.len() > self.max_nonce_history {
            let oldest_nonces: Vec<_> = self
                .nonce_timestamps
                .iter()
                .take(self.used_nonces.len() - self.max_nonce_history)
                .map(|(nonce, _)| nonce.clone())
                .collect();

            for nonce in oldest_nonces {
                self.used_nonces.remove(&nonce);
                self.nonce_timestamps.remove(&nonce);
            }
        }

        // Cache active ticket
        self.active_tickets
            .insert(ticket.subject_id.clone(), ticket.clone());
    }

    fn _get_hour_key(&mut self, hour_timestamp: u64) -> Bytes {
        if let Some(key) = self.hour_keys.get(&hour_timestamp) {
            return key.clone();
        }

        // Stub HKDF implementation - in real code would use hkdf crate
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.master_secret.hash(&mut hasher);
        hour_timestamp.hash(&mut hasher);
        let hash_result = hasher.finish();

        let mut derived_key = [0u8; 32];
        derived_key[..8].copy_from_slice(&hash_result.to_le_bytes());

        let key = Bytes::from(derived_key.to_vec());
        self.hour_keys.insert(hour_timestamp, key.clone());
        key
    }
}

/// Subject status information
#[derive(Debug, Clone)]
pub struct SubjectStatus {
    pub subject_id: String,
    pub ticket_type: TicketType,
    pub time_remaining: u64,
    pub max_bandwidth_bps: u32,
    pub max_connections: u32,
    pub rate_limiter_status: Option<TokenBucketStatus>,
}

/// Manager statistics
#[derive(Debug, Clone)]
pub struct ManagerStats {
    pub tickets: TicketStats,
    pub trusted_issuers: usize,
    pub active_tickets: usize,
    pub rate_limiters: usize,
    pub nonce_history_size: usize,
}

/// Generate Ed25519 keypair for ticket issuing
pub fn generate_issuer_keypair() -> (Bytes, Bytes) {
    // SECURITY FIX: Use proper Ed25519 key generation
    use ed25519_dalek::{SigningKey, VerifyingKey};

    // Generate cryptographically secure signing key
    let signing_key = SigningKey::generate(&mut OsRng);
    let verifying_key: VerifyingKey = (&signing_key).into();

    (
        Bytes::from(signing_key.to_bytes().to_vec()),
        Bytes::from(verifying_key.to_bytes().to_vec()),
    )
}

/// Get current Unix timestamp in seconds
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Cookie/Query/Body carrier rotation via Markov chains
pub mod rotation {
    use super::*;
    use rand::Rng;
    use std::collections::BTreeMap;

    /// Markov chain states for carrier rotation
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum CarrierState {
        Cookie,
        QueryParam,
        BodyField,
        Header,
    }

    /// Transition probabilities between carrier states
    #[derive(Debug, Clone)]
    pub struct MarkovTransition {
        pub from: CarrierState,
        pub transitions: BTreeMap<CarrierState, f64>,
    }

    impl MarkovTransition {
        pub fn new(from: CarrierState) -> Self {
            Self {
                from,
                transitions: BTreeMap::new(),
            }
        }

        pub fn add_transition(mut self, to: CarrierState, probability: f64) -> Self {
            self.transitions.insert(to, probability);
            self
        }

        pub fn normalize_probabilities(&mut self) {
            let total: f64 = self.transitions.values().sum();
            if total > 0.0 {
                for prob in self.transitions.values_mut() {
                    *prob /= total;
                }
            }
        }

        pub fn next_state(&self, rng: &mut impl Rng) -> CarrierState {
            let r: f64 = rng.gen();
            let mut cumulative = 0.0;

            for (&state, &probability) in &self.transitions {
                cumulative += probability;
                if r <= cumulative {
                    return state;
                }
            }

            // Fallback to first state if no transition found
            self.transitions.keys().next().copied().unwrap_or(self.from)
        }
    }

    /// Markov chain for carrier rotation
    #[derive(Debug, Clone)]
    pub struct CarrierMarkovChain {
        transitions: HashMap<CarrierState, MarkovTransition>,
        current_state: CarrierState,
        state_history: VecDeque<CarrierState>,
        max_history: usize,
    }

    impl Default for CarrierMarkovChain {
        fn default() -> Self {
            let mut chain = Self {
                transitions: HashMap::new(),
                current_state: CarrierState::Cookie,
                state_history: VecDeque::new(),
                max_history: 50,
            };

            // Define realistic transition probabilities based on web traffic patterns

            // From Cookie state
            chain.add_transition(MarkovTransition::new(CarrierState::Cookie)
                .add_transition(CarrierState::Cookie, 0.4) // Stay in cookie (session persistence)
                .add_transition(CarrierState::QueryParam, 0.3) // Move to query params
                .add_transition(CarrierState::BodyField, 0.2) // POST requests
                .add_transition(CarrierState::Header, 0.1) // Custom headers
            );

            // From QueryParam state
            chain.add_transition(MarkovTransition::new(CarrierState::QueryParam)
                .add_transition(CarrierState::Cookie, 0.35) // Back to cookies
                .add_transition(CarrierState::QueryParam, 0.25) // Stay in query params
                .add_transition(CarrierState::BodyField, 0.3) // Form submissions
                .add_transition(CarrierState::Header, 0.1) // Custom headers
            );

            // From BodyField state
            chain.add_transition(MarkovTransition::new(CarrierState::BodyField)
                .add_transition(CarrierState::Cookie, 0.45) // Response sets cookies
                .add_transition(CarrierState::QueryParam, 0.2) // Redirect with params
                .add_transition(CarrierState::BodyField, 0.25) // Multiple form fields
                .add_transition(CarrierState::Header, 0.1) // API responses
            );

            // From Header state
            chain.add_transition(MarkovTransition::new(CarrierState::Header)
                .add_transition(CarrierState::Cookie, 0.4) // Headers often set cookies
                .add_transition(CarrierState::QueryParam, 0.25) // API parameters
                .add_transition(CarrierState::BodyField, 0.25) // POST data
                .add_transition(CarrierState::Header, 0.1) // Stay in headers
            );

            chain
        }
    }

    impl CarrierMarkovChain {
        pub fn new(initial_state: CarrierState) -> Self {
            let mut chain = Self::default();
            chain.current_state = initial_state;
            chain
        }

        pub fn add_transition(&mut self, mut transition: MarkovTransition) {
            transition.normalize_probabilities();
            self.transitions.insert(transition.from, transition);
        }

        pub fn next_carrier(&mut self, rng: &mut impl Rng) -> CarrierState {
            let next_state = if let Some(transition) = self.transitions.get(&self.current_state) {
                transition.next_state(rng)
            } else {
                // If no transition defined, stay in current state
                self.current_state
            };

            // Record transition
            self.state_history.push_back(self.current_state);
            if self.state_history.len() > self.max_history {
                self.state_history.pop_front();
            }

            self.current_state = next_state;
            next_state
        }

        pub fn current_state(&self) -> CarrierState {
            self.current_state
        }

        pub fn state_distribution(&self) -> HashMap<CarrierState, f64> {
            let total = self.state_history.len() as f64;
            if total == 0.0 {
                return HashMap::new();
            }

            let mut distribution = HashMap::new();
            for &state in &self.state_history {
                *distribution.entry(state).or_insert(0.0) += 1.0;
            }

            for count in distribution.values_mut() {
                *count /= total;
            }

            distribution
        }
    }

    /// Padding size generator with log-normal distribution
    #[derive(Debug, Clone)]
    pub struct PaddingGenerator {
        min_size: usize,
        max_size: usize,
        histogram: Vec<(usize, f64)>, // (size, probability)
    }

    impl PaddingGenerator {
        pub fn log_normal_50_4096() -> Self {
            // Create log-normal distribution from 50B to 4096B
            let mut histogram = Vec::new();

            // Define log-normal parameters
            // μ ≈ 6.5 (ln(650) for median around 650 bytes)
            // σ ≈ 1.2 for reasonable spread

            let sizes = vec![
                (50, 0.02),    // Very small padding
                (100, 0.05),   // Small padding
                (200, 0.10),   // Common small
                (400, 0.15),   // Common medium-small
                (600, 0.20),   // Peak of distribution
                (800, 0.18),   // Common medium-large
                (1200, 0.12),  // Large padding
                (1800, 0.08),  // Larger padding
                (2500, 0.05),  // Very large
                (3500, 0.03),  // Rare very large
                (4096, 0.02),  // Maximum size
            ];

            histogram.extend(sizes);

            Self {
                min_size: 50,
                max_size: 4096,
                histogram,
            }
        }

        pub fn sample_padding_size(&self, rng: &mut impl Rng) -> usize {
            let r: f64 = rng.gen();
            let mut cumulative = 0.0;

            for &(size, probability) in &self.histogram {
                cumulative += probability;
                if r <= cumulative {
                    // Add some jitter around the selected size
                    let jitter_range = (size as f64 * 0.1) as usize; // ±10%
                    let jitter = rng.gen_range(0..=jitter_range * 2);
                    let jittered_size = size.saturating_sub(jitter_range).saturating_add(jitter);
                    return jittered_size.clamp(self.min_size, self.max_size);
                }
            }

            // Fallback to median size
            600
        }

        pub fn generate_padding(&self, size: usize, rng: &mut impl Rng) -> Vec<u8> {
            let mut padding = vec![0u8; size];

            // Fill with realistic-looking data patterns
            for (i, byte) in padding.iter_mut().enumerate() {
                match i % 4 {
                    0 => *byte = rng.gen_range(0x20..=0x7E), // ASCII printable
                    1 => *byte = rng.gen_range(0x30..=0x39), // ASCII digits
                    2 => *byte = rng.gen_range(0x41..=0x5A), // ASCII uppercase
                    3 => *byte = rng.gen_range(0x61..=0x7A), // ASCII lowercase
                    _ => unreachable!(),
                }
            }

            padding
        }
    }

    /// Ticket carrier with rotation and padding
    #[derive(Debug, Clone)]
    pub struct TicketCarrier {
        carrier_type: CarrierState,
        field_name: String,
        field_value: String,
        padding: Vec<u8>,
        blinded_client_public: Bytes, // Blinded client public key per request
    }

    impl TicketCarrier {
        pub fn new(
            carrier_type: CarrierState,
            ticket_data: &[u8],
            padding_size: usize,
            rng: &mut impl Rng,
        ) -> Result<Self, TicketError> {
            let field_name = Self::generate_field_name(carrier_type, rng);

            // Generate blinded client public key (stub implementation)
            let mut blinded_pubkey = vec![0u8; 32];
            rng.fill(&mut blinded_pubkey[..]);

            // Encode ticket data with padding
            let mut field_value = base64::encode_config(ticket_data, base64::URL_SAFE_NO_PAD);

            // Generate padding
            let padding_gen = PaddingGenerator::log_normal_50_4096();
            let padding = padding_gen.generate_padding(padding_size, rng);
            let padding_b64 = base64::encode_config(&padding, base64::URL_SAFE_NO_PAD);

            // Combine ticket and padding
            field_value.push('.');
            field_value.push_str(&padding_b64);

            Ok(Self {
                carrier_type,
                field_name,
                field_value,
                padding,
                blinded_client_public: Bytes::from(blinded_pubkey),
            })
        }

        fn generate_field_name(carrier_type: CarrierState, rng: &mut impl Rng) -> String {
            match carrier_type {
                CarrierState::Cookie => {
                    let cookie_names = vec![
                        "session_id", "auth_token", "user_prefs", "csrf_token",
                        "tracking_id", "analytics", "cart_id", "locale",
                    ];
                    cookie_names[rng.gen_range(0..cookie_names.len())].to_string()
                }
                CarrierState::QueryParam => {
                    let param_names = vec![
                        "q", "search", "filter", "sort", "page", "limit",
                        "category", "type", "format", "callback", "v", "ts",
                    ];
                    param_names[rng.gen_range(0..param_names.len())].to_string()
                }
                CarrierState::BodyField => {
                    let field_names = vec![
                        "data", "payload", "content", "message", "text",
                        "description", "details", "metadata", "config",
                    ];
                    field_names[rng.gen_range(0..field_names.len())].to_string()
                }
                CarrierState::Header => {
                    let header_names = vec![
                        "X-Request-ID", "X-Session-Token", "X-API-Key",
                        "X-Client-Version", "X-Forwarded-For", "X-Real-IP",
                        "X-Correlation-ID", "X-Trace-ID",
                    ];
                    header_names[rng.gen_range(0..header_names.len())].to_string()
                }
            }
        }

        pub fn carrier_type(&self) -> CarrierState {
            self.carrier_type
        }

        pub fn field_name(&self) -> &str {
            &self.field_name
        }

        pub fn field_value(&self) -> &str {
            &self.field_value
        }

        pub fn extract_ticket_data(&self) -> Result<Vec<u8>, TicketError> {
            // Split field value at the first '.' to separate ticket from padding
            let parts: Vec<&str> = self.field_value.splitn(2, '.').collect();
            if parts.is_empty() {
                return Err(TicketError::Malformed("Invalid field value format".to_string()));
            }

            let ticket_b64 = parts[0];
            base64::decode_config(ticket_b64, base64::URL_SAFE_NO_PAD)
                .map_err(|e| TicketError::Malformed(format!("Base64 decode error: {}", e)))
        }

        pub fn padding_size(&self) -> usize {
            self.padding.len()
        }

        pub fn blinded_client_public(&self) -> &Bytes {
            &self.blinded_client_public
        }

        pub fn format_for_http(&self) -> String {
            match self.carrier_type {
                CarrierState::Cookie => {
                    format!("{}={}", self.field_name, self.field_value)
                }
                CarrierState::QueryParam => {
                    format!("{}={}", self.field_name, urlencoding::encode(&self.field_value))
                }
                CarrierState::BodyField => {
                    format!("{}={}", self.field_name, urlencoding::encode(&self.field_value))
                }
                CarrierState::Header => {
                    format!("{}: {}", self.field_name, self.field_value)
                }
            }
        }
    }

    /// Ticket rotation manager
    pub struct TicketRotationManager {
        markov_chain: CarrierMarkovChain,
        padding_generator: PaddingGenerator,
        rotation_stats: RotationStats,
    }

    #[derive(Debug, Clone, Default)]
    pub struct RotationStats {
        pub total_rotations: u64,
        pub carrier_usage: HashMap<CarrierState, u64>,
        pub avg_padding_size: f64,
        pub total_padding_bytes: u64,
    }

    impl TicketRotationManager {
        pub fn new() -> Self {
            Self {
                markov_chain: CarrierMarkovChain::default(),
                padding_generator: PaddingGenerator::log_normal_50_4096(),
                rotation_stats: RotationStats::default(),
            }
        }

        pub fn with_initial_state(initial_state: CarrierState) -> Self {
            Self {
                markov_chain: CarrierMarkovChain::new(initial_state),
                padding_generator: PaddingGenerator::log_normal_50_4096(),
                rotation_stats: RotationStats::default(),
            }
        }

        pub fn create_ticket_carrier(
            &mut self,
            ticket_data: &[u8],
            rng: &mut impl Rng,
        ) -> Result<TicketCarrier, TicketError> {
            // Get next carrier type from Markov chain
            let carrier_type = self.markov_chain.next_carrier(rng);

            // Generate padding size from log-normal distribution
            let padding_size = self.padding_generator.sample_padding_size(rng);

            // Update statistics
            self.rotation_stats.total_rotations += 1;
            *self.rotation_stats.carrier_usage.entry(carrier_type).or_insert(0) += 1;
            self.rotation_stats.total_padding_bytes += padding_size as u64;
            self.rotation_stats.avg_padding_size =
                self.rotation_stats.total_padding_bytes as f64 / self.rotation_stats.total_rotations as f64;

            // Create carrier
            TicketCarrier::new(carrier_type, ticket_data, padding_size, rng)
        }

        pub fn current_state(&self) -> CarrierState {
            self.markov_chain.current_state()
        }

        pub fn state_distribution(&self) -> HashMap<CarrierState, f64> {
            self.markov_chain.state_distribution()
        }

        pub fn statistics(&self) -> &RotationStats {
            &self.rotation_stats
        }
    }
}

// Additional imports for ticket rotation features
use base64;
use urlencoding;
use std::collections::VecDeque;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keypair_generation() {
        let (private_key, public_key) = generate_issuer_keypair();
        assert_eq!(private_key.len(), 32);
        assert_eq!(public_key.len(), 32);
    }

    #[test]
    fn test_ticket_creation_and_signing() {
        let (private_key, public_key) = generate_issuer_keypair();
        let mut ticket = AccessTicket::new(
            "test-issuer".to_string(),
            "test-subject".to_string(),
            TicketType::Standard,
            3600,
        );

        assert!(ticket.sign(&private_key).is_ok());
        assert!(ticket.verify(&public_key).unwrap());
    }

    #[test]
    fn test_ticket_serialization() {
        let ticket = AccessTicket::new(
            "test-issuer".to_string(),
            "test-subject".to_string(),
            TicketType::Premium,
            7200,
        );

        let serialized = ticket.serialize().unwrap();
        assert!(serialized.len() >= MIN_TICKET_SIZE);
        assert!(serialized.len() <= MAX_TICKET_SIZE);

        let deserialized = AccessTicket::deserialize(&serialized).unwrap();
        // Note: issuer_id and subject_id are truncated to 8 bytes in compact format
        assert_eq!(&ticket.issuer_id[..ticket.issuer_id.len().min(8)], &deserialized.issuer_id[..deserialized.issuer_id.len().min(8)]);
        assert_eq!(&ticket.subject_id[..ticket.subject_id.len().min(8)], &deserialized.subject_id[..deserialized.subject_id.len().min(8)]);
        assert_eq!(ticket.ticket_type, deserialized.ticket_type);
    }

    #[test]
    fn test_ticket_validation() {
        let (private_key, public_key) = generate_issuer_keypair();
        let mut manager = AccessTicketManager::new(1000);
        manager.add_trusted_issuer("test-issuer".to_string(), &public_key).unwrap();

        let ticket = manager
            .issue_ticket(
                "test-issuer".to_string(),
                "test-subject".to_string(),
                TicketType::Standard,
                3600,
                &private_key,
            )
            .unwrap();

        let status = manager.validate_ticket(&ticket);
        assert_eq!(status, TicketStatus::Valid);
    }

    #[test]
    fn test_expired_ticket() {
        // Create ticket that expired 1 second ago
        let past_time = (current_timestamp() - 1) * 1000;
        let mut ticket = AccessTicket::new(
            "test-issuer".to_string(),
            "test-subject".to_string(),
            TicketType::Standard,
            1, // 1 second validity
        );

        // Manually set expiration to the past
        ticket.expires_at = past_time;

        assert!(ticket.is_expired());
    }

    #[test]
    fn test_token_bucket() {
        let config = TokenBucketConfig::new(10, 1.0, 5);
        let mut bucket = TokenBucket::new(config);

        // Should be able to consume initial capacity
        assert!(bucket.consume(5));
        assert!(bucket.consume(5));
        assert!(!bucket.consume(1)); // Should fail - no tokens left
    }

    #[test]
    fn test_replay_detection() {
        let (private_key, public_key) = generate_issuer_keypair();
        let mut manager = AccessTicketManager::new(1000);
        manager.add_trusted_issuer("test-issuer".to_string(), &public_key).unwrap();

        let ticket = manager
            .issue_ticket(
                "test-issuer".to_string(),
                "test-subject".to_string(),
                TicketType::Standard,
                3600,
                &private_key,
            )
            .unwrap();

        // First validation should succeed
        assert_eq!(manager.validate_ticket(&ticket), TicketStatus::Valid);

        // Second validation with same nonce should fail
        assert_eq!(manager.validate_ticket(&ticket), TicketStatus::ReplayDetected);
    }

    proptest::proptest! {
        #[test]
        fn prop_ticket_serialization_roundtrip(
            issuer_id in "[a-zA-Z0-9-]{1,30}",
            subject_id in "[a-zA-Z0-9-]{1,30}",
            ticket_type in 0u8..4,
            validity in 1u64..86400,
        ) {
            let ticket_type = TicketType::try_from(ticket_type).unwrap();
            let ticket = AccessTicket::new(issuer_id, subject_id, ticket_type, validity);

            let serialized = ticket.serialize().unwrap();
            let deserialized = AccessTicket::deserialize(&serialized).unwrap();

            // Note: issuer_id and subject_id are truncated to 8 bytes in compact format
            proptest::prop_assert_eq!(&ticket.issuer_id[..ticket.issuer_id.len().min(8)], &deserialized.issuer_id[..deserialized.issuer_id.len().min(8)]);
            proptest::prop_assert_eq!(&ticket.subject_id[..ticket.subject_id.len().min(8)], &deserialized.subject_id[..deserialized.subject_id.len().min(8)]);
            proptest::prop_assert_eq!(ticket.ticket_type, deserialized.ticket_type);
        }
    }
}
