//! Bundle Protocol v7 bundle format implementation
//!
//! Implements bundles with Primary, Canonical, and Payload blocks according to RFC 9171.

use std::fmt;

use bytes::{Bytes, BytesMut, Buf, BufMut};
use serde::{Deserialize, Serialize, Deserializer, Serializer};
use thiserror::Error;

use crate::{BundleControlFlags, CreationTimestamp, DTN_VERSION};

/// Serializable wrapper for bytes::Bytes
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SerializableBytes(pub Bytes);

impl Serialize for SerializableBytes {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_bytes(&self.0)
    }
}

impl<'de> Deserialize<'de> for SerializableBytes {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let bytes = Vec::<u8>::deserialize(deserializer)?;
        Ok(SerializableBytes(Bytes::from(bytes)))
    }
}

impl From<Bytes> for SerializableBytes {
    fn from(bytes: Bytes) -> Self {
        SerializableBytes(bytes)
    }
}

impl From<SerializableBytes> for Bytes {
    fn from(wrapper: SerializableBytes) -> Self {
        wrapper.0
    }
}

impl SerializableBytes {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// Unique bundle identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BundleId {
    pub source: EndpointId,
    pub timestamp: CreationTimestamp,
}

impl fmt::Display for BundleId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}@{}.{}", self.source, self.timestamp.dtn_time, self.timestamp.sequence_number)
    }
}

/// Endpoint identifier per BPv7 specification
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EndpointId {
    pub scheme: String,
    pub specific_part: String,
}

impl EndpointId {
    pub fn new(scheme: impl Into<String>, specific_part: impl Into<String>) -> Self {
        Self {
            scheme: scheme.into(),
            specific_part: specific_part.into(),
        }
    }

    /// Create a DTN null endpoint
    pub fn null() -> Self {
        Self::new("dtn", "none")
    }

    /// Create a node endpoint
    pub fn node(node_id: impl Into<String>) -> Self {
        Self::new("dtn", node_id)
    }

    /// Create an IPN endpoint
    pub fn ipn(node: u64, service: u64) -> Self {
        Self::new("ipn", format!("{}.{}", node, service))
    }

    pub fn is_null(&self) -> bool {
        self.scheme == "dtn" && self.specific_part == "none"
    }
}

impl fmt::Display for EndpointId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.scheme, self.specific_part)
    }
}

impl std::str::FromStr for EndpointId {
    type Err = BundleError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if let Some((scheme, specific)) = s.split_once(':') {
            Ok(Self::new(scheme, specific))
        } else {
            Err(BundleError::InvalidEndpointId(s.to_string()))
        }
    }
}

/// Bundle block types per BPv7 specification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum BlockType {
    Payload = 1,
    PreviousNode = 6,
    BundleAge = 7,
    MetadataExtension = 8,
    ExtensionBlock = 9,
    Hop = 10,
}

impl TryFrom<u8> for BlockType {
    type Error = BundleError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Payload),
            6 => Ok(Self::PreviousNode),
            7 => Ok(Self::BundleAge),
            8 => Ok(Self::MetadataExtension),
            9 => Ok(Self::ExtensionBlock),
            10 => Ok(Self::Hop),
            _ => Err(BundleError::InvalidBlockType(value)),
        }
    }
}

/// Primary block per BPv7 specification
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrimaryBlock {
    pub version: u8,
    pub bundle_control_flags: BundleControlFlags,
    pub crc_type: CrcType,
    pub destination: EndpointId,
    pub source: EndpointId,
    pub report_to: EndpointId,
    pub creation_timestamp: CreationTimestamp,
    pub lifetime: u64, // milliseconds
    pub fragment_offset: Option<u64>,
    pub total_application_data_length: Option<u64>,
}

impl PrimaryBlock {
    pub fn new(
        destination: EndpointId,
        source: EndpointId,
        lifetime_ms: u64,
    ) -> Self {
        Self {
            version: DTN_VERSION,
            bundle_control_flags: BundleControlFlags::NONE,
            crc_type: CrcType::Crc32,
            destination,
            source: source.clone(),
            report_to: source,
            creation_timestamp: CreationTimestamp::now(),
            lifetime: lifetime_ms,
            fragment_offset: None,
            total_application_data_length: None,
        }
    }

    pub fn bundle_id(&self) -> BundleId {
        BundleId {
            source: self.source.clone(),
            timestamp: self.creation_timestamp.clone(),
        }
    }

    pub fn is_expired(&self) -> bool {
        self.creation_timestamp.is_expired(self.lifetime)
    }
}

/// CRC types supported
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum CrcType {
    None = 0,
    Crc16 = 1,
    Crc32 = 2,
}

impl TryFrom<u8> for CrcType {
    type Error = BundleError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::None),
            1 => Ok(Self::Crc16),
            2 => Ok(Self::Crc32),
            _ => Err(BundleError::InvalidCrcType(value)),
        }
    }
}

/// Canonical block (extension blocks)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CanonicalBlock {
    pub block_type: BlockType,
    pub block_number: u64,
    pub block_processing_control_flags: u32,
    pub crc_type: CrcType,
    pub data: SerializableBytes,
}

impl CanonicalBlock {
    pub fn new(block_type: BlockType, block_number: u64, data: Bytes) -> Self {
        Self {
            block_type,
            block_number,
            block_processing_control_flags: 0,
            crc_type: CrcType::Crc32,
            data: SerializableBytes::from(data),
        }
    }

    /// Create a bundle age block
    pub fn bundle_age(age_ms: u64) -> Self {
        let mut data = BytesMut::new();
        data.put_u64(age_ms);
        Self::new(BlockType::BundleAge, 2, data.freeze())
    }

    /// Create a previous node block
    pub fn previous_node(node_id: EndpointId) -> Self {
        let data = bincode::serialize(&node_id).unwrap_or_default().into();
        Self::new(BlockType::PreviousNode, 3, data)
    }

    /// Create a hop count block
    pub fn hop_count(limit: u32, count: u32) -> Self {
        let mut data = BytesMut::new();
        data.put_u32(limit);
        data.put_u32(count);
        Self::new(BlockType::Hop, 4, data.freeze())
    }
}

/// Payload block containing application data
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PayloadBlock {
    pub block_number: u64,
    pub block_processing_control_flags: u32,
    pub crc_type: CrcType,
    pub data: SerializableBytes,
}

impl PayloadBlock {
    pub fn new(data: Bytes) -> Self {
        Self {
            block_number: 1, // Payload is always block 1
            block_processing_control_flags: 0,
            crc_type: CrcType::Crc32,
            data: SerializableBytes::from(data),
        }
    }
}

/// Complete bundle structure
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Bundle {
    pub primary: PrimaryBlock,
    pub canonical_blocks: Vec<CanonicalBlock>,
    pub payload: PayloadBlock,
}

impl Bundle {
    pub fn new(
        destination: EndpointId,
        source: EndpointId,
        payload: Bytes,
        lifetime_ms: u64,
    ) -> Self {
        let primary = PrimaryBlock::new(destination, source, lifetime_ms);
        let payload_block = PayloadBlock::new(payload);

        Self {
            primary,
            canonical_blocks: Vec::new(),
            payload: payload_block,
        }
    }

    pub fn id(&self) -> BundleId {
        self.primary.bundle_id()
    }

    pub fn is_expired(&self) -> bool {
        self.primary.is_expired()
    }

    pub fn add_canonical_block(&mut self, block: CanonicalBlock) {
        self.canonical_blocks.push(block);
    }

    /// Add bundle age tracking
    pub fn add_age_block(&mut self, age_ms: u64) {
        self.add_canonical_block(CanonicalBlock::bundle_age(age_ms));
    }

    /// Add previous node tracking
    pub fn add_previous_node(&mut self, node_id: EndpointId) {
        self.add_canonical_block(CanonicalBlock::previous_node(node_id));
    }

    /// Add hop count tracking
    pub fn add_hop_count(&mut self, limit: u32, count: u32) {
        self.add_canonical_block(CanonicalBlock::hop_count(limit, count));
    }

    /// Calculate total bundle size in bytes
    pub fn size(&self) -> usize {
        let encoded = self.encode().unwrap_or_default();
        encoded.len()
    }

    /// Encode bundle to bytes (simplified CBOR-like encoding)
    pub fn encode(&self) -> Result<Bytes, BundleError> {
        let mut buf = BytesMut::new();

        // Encode primary block
        let primary_data = bincode::serialize(&self.primary)
            .map_err(|e| BundleError::EncodingError(e.to_string()))?;
        buf.put_u32(primary_data.len() as u32);
        buf.put_slice(&primary_data);

        // Encode canonical blocks
        buf.put_u32(self.canonical_blocks.len() as u32);
        for block in &self.canonical_blocks {
            let block_data = bincode::serialize(block)
                .map_err(|e| BundleError::EncodingError(e.to_string()))?;
            buf.put_u32(block_data.len() as u32);
            buf.put_slice(&block_data);
        }

        // Encode payload block
        let payload_data = bincode::serialize(&self.payload)
            .map_err(|e| BundleError::EncodingError(e.to_string()))?;
        buf.put_u32(payload_data.len() as u32);
        buf.put_slice(&payload_data);

        // Add CRC32 checksum
        let crc = crc::Crc::<u32>::new(&crc::CRC_32_ISO_HDLC);
        let checksum = crc.checksum(&buf);
        buf.put_u32(checksum);

        Ok(buf.freeze())
    }

    /// Decode bundle from bytes
    pub fn decode(mut data: Bytes) -> Result<Self, BundleError> {
        if data.len() < 4 {
            return Err(BundleError::TruncatedBundle);
        }

        // Verify CRC32 checksum
        let bundle_data = data.split_to(data.len() - 4);
        let checksum = data.get_u32();

        let crc = crc::Crc::<u32>::new(&crc::CRC_32_ISO_HDLC);
        let calculated_checksum = crc.checksum(&bundle_data);

        if checksum != calculated_checksum {
            return Err(BundleError::InvalidChecksum {
                expected: checksum,
                calculated: calculated_checksum,
            });
        }

        let mut data = bundle_data;

        // Decode primary block
        if data.len() < 4 {
            return Err(BundleError::TruncatedBundle);
        }
        let primary_len = data.get_u32() as usize;
        if data.len() < primary_len {
            return Err(BundleError::TruncatedBundle);
        }
        let primary_data = data.split_to(primary_len);
        let primary: PrimaryBlock = bincode::deserialize(&primary_data)
            .map_err(|e| BundleError::DecodingError(e.to_string()))?;

        // Decode canonical blocks
        if data.len() < 4 {
            return Err(BundleError::TruncatedBundle);
        }
        let canonical_count = data.get_u32();
        let mut canonical_blocks = Vec::new();

        for _ in 0..canonical_count {
            if data.len() < 4 {
                return Err(BundleError::TruncatedBundle);
            }
            let block_len = data.get_u32() as usize;
            if data.len() < block_len {
                return Err(BundleError::TruncatedBundle);
            }
            let block_data = data.split_to(block_len);
            let block: CanonicalBlock = bincode::deserialize(&block_data)
                .map_err(|e| BundleError::DecodingError(e.to_string()))?;
            canonical_blocks.push(block);
        }

        // Decode payload block
        if data.len() < 4 {
            return Err(BundleError::TruncatedBundle);
        }
        let payload_len = data.get_u32() as usize;
        if data.len() < payload_len {
            return Err(BundleError::TruncatedBundle);
        }
        let payload_data = data.split_to(payload_len);
        let payload: PayloadBlock = bincode::deserialize(&payload_data)
            .map_err(|e| BundleError::DecodingError(e.to_string()))?;

        Ok(Self {
            primary,
            canonical_blocks,
            payload,
        })
    }
}

/// Bundle-related errors
#[derive(Debug, Error)]
pub enum BundleError {
    #[error("Invalid endpoint ID: {0}")]
    InvalidEndpointId(String),

    #[error("Invalid block type: {0}")]
    InvalidBlockType(u8),

    #[error("Invalid CRC type: {0}")]
    InvalidCrcType(u8),

    #[error("Bundle encoding error: {0}")]
    EncodingError(String),

    #[error("Bundle decoding error: {0}")]
    DecodingError(String),

    #[error("Truncated bundle data")]
    TruncatedBundle,

    #[error("Invalid checksum: expected {expected:08x}, calculated {calculated:08x}")]
    InvalidChecksum { expected: u32, calculated: u32 },

    #[error("Bundle too large: {size} > {max_size}")]
    BundleTooLarge { size: usize, max_size: usize },

    #[error("Bundle expired")]
    BundleExpired,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_endpoint_id_parsing() {
        let eid = "dtn://example.com/test".parse::<EndpointId>().unwrap();
        assert_eq!(eid.scheme, "dtn");
        assert_eq!(eid.specific_part, "//example.com/test");

        let ipn_eid = EndpointId::ipn(1, 2);
        assert_eq!(ipn_eid.to_string(), "ipn:1.2");

        let null_eid = EndpointId::null();
        assert!(null_eid.is_null());
    }

    #[test]
    fn test_bundle_creation() {
        let src = EndpointId::node("node1");
        let dst = EndpointId::node("node2");
        let payload = Bytes::from("Hello, DTN!");

        let bundle = Bundle::new(dst.clone(), src.clone(), payload.clone(), 60000);

        assert_eq!(bundle.primary.destination, dst);
        assert_eq!(bundle.primary.source, src);
        assert_eq!(bundle.payload.data, SerializableBytes::from(payload));
        assert_eq!(bundle.primary.lifetime, 60000);
    }

    #[test]
    fn test_bundle_encoding_decoding() {
        let src = EndpointId::node("node1");
        let dst = EndpointId::node("node2");
        let payload = Bytes::from("Hello, DTN!");

        let mut bundle = Bundle::new(dst, src, payload, 60000);
        bundle.add_age_block(1000);

        let encoded = bundle.encode().unwrap();
        let decoded = Bundle::decode(encoded).unwrap();

        assert_eq!(bundle.primary.destination, decoded.primary.destination);
        assert_eq!(bundle.primary.source, decoded.primary.source);
        assert_eq!(bundle.payload.data, decoded.payload.data);
        assert_eq!(bundle.canonical_blocks.len(), decoded.canonical_blocks.len());
    }

    #[test]
    fn test_bundle_age_block() {
        let age_block = CanonicalBlock::bundle_age(5000);
        assert_eq!(age_block.block_type, BlockType::BundleAge);
        assert_eq!(age_block.block_number, 2);
    }

    #[test]
    fn test_bundle_id() {
        let src = EndpointId::node("node1");
        let dst = EndpointId::node("node2");
        let payload = Bytes::from("test");

        let bundle = Bundle::new(dst, src.clone(), payload, 60000);
        let id = bundle.id();

        assert_eq!(id.source, src);
        assert_eq!(id.timestamp, bundle.primary.creation_timestamp);
    }

    #[test]
    fn test_invalid_checksum() {
        let mut data = BytesMut::new();
        data.put_u32(4); // Primary block length
        data.put_slice(b"test"); // Invalid primary data
        data.put_u32(0); // Canonical blocks count
        data.put_u32(4); // Payload length
        data.put_slice(b"test"); // Invalid payload data
        data.put_u32(0x12345678); // Wrong checksum

        let result = Bundle::decode(data.freeze());
        assert!(matches!(result, Err(BundleError::InvalidChecksum { .. })));
    }
}
