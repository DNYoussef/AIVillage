// SCION packet encapsulation and decapsulation for HTX transport
// Handles CBOR framing, ChaCha20-Poly1305 encryption, and packet validation

use anyhow::{Context, Result, bail};
use bytes::{Bytes, BytesMut, BufMut};
use chacha20poly1305::{
    aead::{Aead, AeadCore, KeyInit, OsRng},
    ChaCha20Poly1305, Nonce
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

/// SCION packet metadata for encapsulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScionPacketMeta {
    /// Unique packet ID for tracking
    pub packet_id: String,

    /// Source ISD-AS address
    pub src_ia: String,

    /// Destination ISD-AS address
    pub dst_ia: String,

    /// Source endpoint address
    pub src_addr: String,

    /// Destination endpoint address
    pub dst_addr: String,

    /// Path fingerprint for multipath selection
    pub path_fingerprint: String,

    /// Packet priority (0-255)
    pub priority: u8,

    /// Traffic class for QoS
    pub traffic_class: u8,

    /// Flow label for connection tracking
    pub flow_label: u32,

    /// Timestamp when packet was encapsulated (nanoseconds since epoch)
    pub timestamp_ns: u64,

    /// Hop limit/TTL
    pub hop_limit: u8,

    /// Payload length in bytes
    pub payload_length: usize,

    /// Additional metadata fields
    pub metadata: HashMap<String, String>,
}

/// Encapsulated packet structure for HTX transport
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncapsulatedPacket {
    /// Protocol version
    pub version: u8,

    /// Encapsulation flags
    pub flags: u16,

    /// Sequence number for anti-replay
    pub sequence: u64,

    /// SCION packet metadata
    pub metadata: ScionPacketMeta,

    /// Encrypted payload
    pub encrypted_payload: Vec<u8>,

    /// Authentication tag
    pub auth_tag: Vec<u8>,

    /// Nonce used for encryption
    pub nonce: Vec<u8>,
}

/// Encapsulation configuration
pub struct EncapConfig {
    /// Enable packet compression
    pub enable_compression: bool,

    /// Maximum packet size in bytes
    pub max_packet_size: usize,

    /// Encryption key derivation salt
    pub key_salt: Vec<u8>,

    /// Enable metadata signing
    pub enable_metadata_signing: bool,

    /// Packet timeout in seconds
    pub packet_timeout_secs: u64,
}

impl Default for EncapConfig {
    fn default() -> Self {
        Self {
            enable_compression: true,
            max_packet_size: 64 * 1024, // 64KB
            key_salt: b"betanet-gateway-encap-v1".to_vec(),
            enable_metadata_signing: true,
            packet_timeout_secs: 300, // 5 minutes
        }
    }
}

/// SCION packet encapsulator with AEAD encryption
pub struct ScionEncapsulator {
    config: EncapConfig,
    cipher_cache: parking_lot::RwLock<HashMap<String, ChaCha20Poly1305>>,
}

impl ScionEncapsulator {
    pub fn new(config: EncapConfig) -> Self {
        Self {
            config,
            cipher_cache: parking_lot::RwLock::new(HashMap::new()),
        }
    }

    /// Encapsulate SCION packet for HTX transport
    pub fn encapsulate_packet(
        &self,
        raw_packet: &[u8],
        metadata: ScionPacketMeta,
        peer_key: &[u8],
        sequence: u64,
    ) -> Result<EncapsulatedPacket> {
        // Validate inputs
        if raw_packet.is_empty() {
            bail!("Cannot encapsulate empty packet");
        }

        if raw_packet.len() > self.config.max_packet_size {
            bail!("Packet exceeds maximum size: {} > {}",
                  raw_packet.len(), self.config.max_packet_size);
        }

        if peer_key.len() != 32 {
            bail!("Invalid peer key length: expected 32 bytes, got {}", peer_key.len());
        }

        // Get or create cipher for this peer
        let cipher = self.get_or_create_cipher(peer_key)?;

        // Generate nonce
        let nonce = ChaCha20Poly1305::generate_nonce(&mut OsRng);

        // Prepare payload for encryption
        let mut payload = BytesMut::new();

        // Add compression if enabled
        let compressed_packet = if self.config.enable_compression {
            self.compress_packet(raw_packet)?
        } else {
            raw_packet.to_vec()
        };

        payload.put_slice(&compressed_packet);

        // Encrypt payload
        let encrypted_payload = cipher
            .encrypt(&nonce, payload.as_ref())
            .map_err(|e| anyhow::anyhow!("Encryption failed: {}", e))?;

        // Create encapsulated packet
        let encap_packet = EncapsulatedPacket {
            version: 1,
            flags: if self.config.enable_compression { 0x01 } else { 0x00 },
            sequence,
            metadata,
            encrypted_payload,
            auth_tag: Vec::new(), // ChaCha20Poly1305 includes auth tag in encrypted data
            nonce: nonce.to_vec(),
        };

        Ok(encap_packet)
    }

    /// Decapsulate HTX transport packet to SCION packet
    pub fn decapsulate_packet(
        &self,
        encap_packet: &EncapsulatedPacket,
        peer_key: &[u8],
    ) -> Result<Vec<u8>> {
        // Validate encapsulated packet
        if encap_packet.version != 1 {
            bail!("Unsupported encapsulation version: {}", encap_packet.version);
        }

        if encap_packet.encrypted_payload.is_empty() {
            bail!("Empty encrypted payload");
        }

        if encap_packet.nonce.len() != 12 {
            bail!("Invalid nonce length: expected 12 bytes, got {}", encap_packet.nonce.len());
        }

        // Check packet age
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let packet_age_ns = current_time - encap_packet.metadata.timestamp_ns;
        let max_age_ns = self.config.packet_timeout_secs * 1_000_000_000;

        if packet_age_ns > max_age_ns {
            bail!("Packet expired: age {} > max {}", packet_age_ns, max_age_ns);
        }

        // Get cipher for this peer
        let cipher = self.get_or_create_cipher(peer_key)?;

        // Reconstruct nonce
        let nonce = Nonce::from_slice(&encap_packet.nonce);

        // Decrypt payload
        let decrypted_payload = cipher
            .decrypt(nonce, encap_packet.encrypted_payload.as_ref())
            .map_err(|e| anyhow::anyhow!("Decryption failed: {}", e))?;

        // Decompress if needed
        let raw_packet = if encap_packet.flags & 0x01 != 0 {
            self.decompress_packet(&decrypted_payload)?
        } else {
            decrypted_payload
        };

        // Validate reconstructed packet
        if raw_packet.len() != encap_packet.metadata.payload_length {
            bail!("Payload length mismatch: expected {}, got {}",
                  encap_packet.metadata.payload_length, raw_packet.len());
        }

        Ok(raw_packet)
    }

    /// Serialize encapsulated packet to CBOR for transport
    pub fn serialize_to_cbor(&self, packet: &EncapsulatedPacket) -> Result<Vec<u8>> {
        serde_cbor::to_vec(packet)
            .context("Failed to serialize packet to CBOR")
    }

    /// Deserialize CBOR data to encapsulated packet
    pub fn deserialize_from_cbor(&self, cbor_data: &[u8]) -> Result<EncapsulatedPacket> {
        if cbor_data.is_empty() {
            bail!("Cannot deserialize empty CBOR data");
        }

        let packet: EncapsulatedPacket = serde_cbor::from_slice(cbor_data)
            .context("Failed to deserialize CBOR to packet")?;

        // Basic validation
        if packet.version == 0 {
            bail!("Invalid packet version");
        }

        if packet.encrypted_payload.is_empty() {
            bail!("Invalid encrypted payload");
        }

        if packet.nonce.is_empty() {
            bail!("Invalid nonce");
        }

        Ok(packet)
    }

    /// Create packet metadata from SCION header information
    pub fn create_packet_metadata(
        src_ia: &str,
        dst_ia: &str,
        src_addr: &str,
        dst_addr: &str,
        path_fingerprint: &str,
        payload_length: usize,
    ) -> ScionPacketMeta {
        let timestamp_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        ScionPacketMeta {
            packet_id: Uuid::new_v4().to_string(),
            src_ia: src_ia.to_string(),
            dst_ia: dst_ia.to_string(),
            src_addr: src_addr.to_string(),
            dst_addr: dst_addr.to_string(),
            path_fingerprint: path_fingerprint.to_string(),
            priority: 0,
            traffic_class: 0,
            flow_label: 0,
            timestamp_ns,
            hop_limit: 64,
            payload_length,
            metadata: HashMap::new(),
        }
    }

    /// Get or create cipher for peer key
    fn get_or_create_cipher(&self, peer_key: &[u8]) -> Result<ChaCha20Poly1305> {
        let key_hash = blake3::hash(peer_key);
        let key_str = hex::encode(key_hash.as_bytes());

        // Check cache first
        {
            let cache = self.cipher_cache.read();
            if let Some(cipher) = cache.get(&key_str) {
                return Ok(cipher.clone());
            }
        }

        // Create new cipher
        let derived_key = self.derive_key(peer_key)?;
        let cipher = ChaCha20Poly1305::new_from_slice(&derived_key)
            .map_err(|e| anyhow::anyhow!("Failed to create cipher: {}", e))?;

        // Cache cipher
        {
            let mut cache = self.cipher_cache.write();
            cache.insert(key_str, cipher.clone());
        }

        Ok(cipher)
    }

    /// Derive encryption key using BLAKE3
    fn derive_key(&self, peer_key: &[u8]) -> Result<[u8; 32]> {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.config.key_salt);
        hasher.update(peer_key);
        hasher.update(b"betanet-gateway-aead");

        let hash = hasher.finalize();
        Ok(*hash.as_bytes())
    }

    /// Compress packet data using simple run-length encoding
    fn compress_packet(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Simple compression - could be replaced with zstd/lz4 for production
        if data.len() < 64 {
            // Don't compress small packets
            return Ok(data.to_vec());
        }

        let mut compressed = Vec::new();
        let mut i = 0;

        while i < data.len() {
            let byte = data[i];
            let mut count = 1u8;

            // Count consecutive identical bytes (up to 255)
            while i + count as usize < data.len() &&
                  data[i + count as usize] == byte &&
                  count < 255 {
                count += 1;
            }

            if count >= 3 {
                // Use RLE for runs of 3 or more
                compressed.push(0xFF); // Escape byte
                compressed.push(count);
                compressed.push(byte);
            } else {
                // Copy literal bytes
                for _ in 0..count {
                    if byte == 0xFF {
                        compressed.push(0xFF);
                        compressed.push(0x00); // Escaped literal 0xFF
                    } else {
                        compressed.push(byte);
                    }
                }
            }

            i += count as usize;
        }

        // Only return compressed if it's actually smaller
        if compressed.len() < data.len() {
            Ok(compressed)
        } else {
            Ok(data.to_vec())
        }
    }

    /// Decompress packet data
    fn decompress_packet(&self, compressed: &[u8]) -> Result<Vec<u8>> {
        let mut decompressed = Vec::new();
        let mut i = 0;

        while i < compressed.len() {
            if compressed[i] == 0xFF && i + 1 < compressed.len() {
                if compressed[i + 1] == 0x00 {
                    // Escaped literal 0xFF
                    decompressed.push(0xFF);
                    i += 2;
                } else {
                    // RLE sequence
                    if i + 2 >= compressed.len() {
                        bail!("Invalid RLE sequence at position {}", i);
                    }

                    let count = compressed[i + 1];
                    let byte = compressed[i + 2];

                    for _ in 0..count {
                        decompressed.push(byte);
                    }

                    i += 3;
                }
            } else {
                // Literal byte
                decompressed.push(compressed[i]);
                i += 1;
            }
        }

        Ok(decompressed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encapsulation_roundtrip() {
        let config = EncapConfig::default();
        let encapsulator = ScionEncapsulator::new(config);

        let raw_packet = b"Hello, SCION world!";
        let peer_key = b"test_key_32_bytes_long_for_test!";
        let sequence = 12345;

        let metadata = ScionEncapsulator::create_packet_metadata(
            "1-ff00:0:110",
            "1-ff00:0:120",
            "192.168.1.1:8080",
            "10.0.0.1:9090",
            "path_fingerprint_123",
            raw_packet.len(),
        );

        // Encapsulate
        let encap_packet = encapsulator.encapsulate_packet(
            raw_packet, metadata, peer_key, sequence
        ).unwrap();

        // Verify structure
        assert_eq!(encap_packet.version, 1);
        assert_eq!(encap_packet.sequence, sequence);
        assert!(!encap_packet.encrypted_payload.is_empty());
        assert!(!encap_packet.nonce.is_empty());

        // Decapsulate
        let decrypted = encapsulator.decapsulate_packet(&encap_packet, peer_key).unwrap();
        assert_eq!(decrypted, raw_packet);
    }

    #[test]
    fn test_cbor_serialization() {
        let config = EncapConfig::default();
        let encapsulator = ScionEncapsulator::new(config);

        let raw_packet = b"Test packet data";
        let peer_key = b"test_key_32_bytes_long_for_test!";
        let sequence = 999;

        let metadata = ScionEncapsulator::create_packet_metadata(
            "1-ff00:0:110",
            "1-ff00:0:120",
            "127.0.0.1:8080",
            "127.0.0.1:9090",
            "test_path",
            raw_packet.len(),
        );

        let encap_packet = encapsulator.encapsulate_packet(
            raw_packet, metadata, peer_key, sequence
        ).unwrap();

        // Serialize to CBOR
        let cbor_data = encapsulator.serialize_to_cbor(&encap_packet).unwrap();
        assert!(!cbor_data.is_empty());

        // Deserialize from CBOR
        let deserialized = encapsulator.deserialize_from_cbor(&cbor_data).unwrap();

        // Verify fields match
        assert_eq!(deserialized.version, encap_packet.version);
        assert_eq!(deserialized.sequence, encap_packet.sequence);
        assert_eq!(deserialized.encrypted_payload, encap_packet.encrypted_payload);
        assert_eq!(deserialized.nonce, encap_packet.nonce);
    }

    #[test]
    fn test_compression() {
        let config = EncapConfig {
            enable_compression: true,
            ..Default::default()
        };
        let encapsulator = ScionEncapsulator::new(config);

        // Create data with repetitive pattern
        let mut raw_packet = Vec::new();
        for _ in 0..100 {
            raw_packet.extend_from_slice(b"AAAA");
        }

        let peer_key = b"test_key_32_bytes_long_for_test!";
        let sequence = 1;

        let metadata = ScionEncapsulator::create_packet_metadata(
            "1-ff00:0:110",
            "1-ff00:0:120",
            "127.0.0.1:8080",
            "127.0.0.1:9090",
            "test_path",
            raw_packet.len(),
        );

        let encap_packet = encapsulator.encapsulate_packet(
            &raw_packet, metadata, peer_key, sequence
        ).unwrap();

        // Verify compression flag is set
        assert_ne!(encap_packet.flags & 0x01, 0);

        // Roundtrip should work
        let decrypted = encapsulator.decapsulate_packet(&encap_packet, peer_key).unwrap();
        assert_eq!(decrypted, raw_packet);
    }

    #[test]
    fn test_packet_expiration() {
        let config = EncapConfig {
            packet_timeout_secs: 1, // 1 second timeout
            ..Default::default()
        };
        let encapsulator = ScionEncapsulator::new(config);

        let raw_packet = b"Expiring packet";
        let peer_key = b"test_key_32_bytes_long_for_test!";
        let sequence = 1;

        let mut metadata = ScionEncapsulator::create_packet_metadata(
            "1-ff00:0:110",
            "1-ff00:0:120",
            "127.0.0.1:8080",
            "127.0.0.1:9090",
            "test_path",
            raw_packet.len(),
        );

        // Set timestamp to past
        metadata.timestamp_ns = 0;

        let encap_packet = encapsulator.encapsulate_packet(
            raw_packet, metadata, peer_key, sequence
        ).unwrap();

        // Decapsulation should fail due to expiration
        let result = encapsulator.decapsulate_packet(&encap_packet, peer_key);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("expired"));
    }
}
