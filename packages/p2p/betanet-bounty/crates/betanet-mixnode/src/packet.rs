//! Packet format and processing

use bytes::{Buf, BufMut, Bytes, BytesMut};
use serde::{Deserialize, Serialize};

use crate::{MixnodeError, Result, MAX_PACKET_SIZE, MIXNODE_VERSION};

/// Packet type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum PacketType {
    /// Data packet
    Data = 0x01,
    /// Control packet
    Control = 0x02,
    /// Cover traffic packet
    Cover = 0x03,
}

impl TryFrom<u8> for PacketType {
    type Error = MixnodeError;

    fn try_from(value: u8) -> Result<Self> {
        match value {
            0x01 => Ok(Self::Data),
            0x02 => Ok(Self::Control),
            0x03 => Ok(Self::Cover),
            _ => Err(MixnodeError::Packet(format!(
                "Invalid packet type: {}",
                value
            ))),
        }
    }
}

/// Packet header
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PacketHeader {
    /// Protocol version
    pub version: u8,
    /// Packet type
    pub packet_type: PacketType,
    /// Packet flags
    pub flags: u8,
    /// Payload length
    pub length: u16,
    /// Layer number (for Sphinx)
    pub layer: u8,
    /// Checksum
    pub checksum: u32,
}

impl PacketHeader {
    /// Create a new packet header
    pub fn new(packet_type: PacketType, payload_len: usize, layer: u8) -> Self {
        Self {
            version: MIXNODE_VERSION,
            packet_type,
            flags: 0,
            length: payload_len as u16,
            layer,
            checksum: 0, // Will be calculated later
        }
    }

    /// Encode header to bytes
    pub fn encode(&self) -> Bytes {
        let mut buf = BytesMut::with_capacity(8);
        buf.put_u8(self.version);
        buf.put_u8(self.packet_type as u8);
        buf.put_u8(self.flags);
        buf.put_u8(self.layer);
        buf.put_u16(self.length);
        buf.put_u16(0); // Reserved
        buf.freeze()
    }

    /// Decode header from bytes
    pub fn decode(mut buf: Bytes) -> Result<Self> {
        if buf.len() < 8 {
            return Err(MixnodeError::Packet("Header too short".to_string()));
        }

        let version = buf.get_u8();
        if version != MIXNODE_VERSION {
            return Err(MixnodeError::Packet(format!(
                "Unsupported version: {}",
                version
            )));
        }

        let packet_type = PacketType::try_from(buf.get_u8())?;
        let flags = buf.get_u8();
        let layer = buf.get_u8();
        let length = buf.get_u16();
        let _reserved = buf.get_u16();

        Ok(Self {
            version,
            packet_type,
            flags,
            length,
            layer,
            checksum: 0,
        })
    }
}

/// Mixnode packet
#[derive(Debug, Clone)]
pub struct Packet {
    /// Packet header
    pub header: PacketHeader,
    /// Packet payload
    pub payload: Bytes,
}

impl Packet {
    /// Create a new packet
    pub fn new(packet_type: PacketType, payload: Bytes, layer: u8) -> Self {
        let header = PacketHeader::new(packet_type, payload.len(), layer);
        Self { header, payload }
    }

    /// Create a data packet
    pub fn data(payload: Bytes, layer: u8) -> Self {
        Self::new(PacketType::Data, payload, layer)
    }

    /// Create a control packet
    pub fn control(payload: Bytes) -> Self {
        Self::new(PacketType::Control, payload, 0)
    }

    /// Create a cover traffic packet
    pub fn cover_traffic(size: usize, layer: u8) -> Self {
        let payload = Bytes::from(vec![0u8; size]);
        Self::new(PacketType::Cover, payload, layer)
    }

    /// Parse packet from raw bytes
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < 8 {
            return Err(MixnodeError::Packet("Packet too short".to_string()));
        }

        if data.len() > MAX_PACKET_SIZE {
            return Err(MixnodeError::Packet(format!(
                "Packet too large: {} > {}",
                data.len(),
                MAX_PACKET_SIZE
            )));
        }

        let header_bytes = Bytes::copy_from_slice(&data[..8]);
        let header = PacketHeader::decode(header_bytes)?;

        if data.len() < 8 + header.length as usize {
            return Err(MixnodeError::Packet("Payload too short".to_string()));
        }

        let payload = Bytes::copy_from_slice(&data[8..8 + header.length as usize]);

        Ok(Self { header, payload })
    }

    /// Encode packet to bytes
    pub fn encode(&self) -> Result<Bytes> {
        if self.payload.len() > MAX_PACKET_SIZE - 8 {
            return Err(MixnodeError::Packet(format!(
                "Payload too large: {} > {}",
                self.payload.len(),
                MAX_PACKET_SIZE - 8
            )));
        }

        let mut buf = BytesMut::with_capacity(8 + self.payload.len());
        buf.put(self.header.encode());
        buf.put(self.payload.as_ref());

        Ok(buf.freeze())
    }

    /// Get packet size
    pub fn size(&self) -> usize {
        8 + self.payload.len()
    }

    /// Check if packet is cover traffic
    pub fn is_cover_traffic(&self) -> bool {
        self.header.packet_type == PacketType::Cover
    }

    /// Get layer number
    pub fn layer(&self) -> u8 {
        self.header.layer
    }

    /// Calculate checksum
    pub fn calculate_checksum(&self) -> u32 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        self.payload.hash(&mut hasher);
        hasher.finish() as u32
    }

    /// Verify checksum
    pub fn verify_checksum(&self) -> bool {
        self.calculate_checksum() == self.header.checksum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packet_creation() {
        let payload = Bytes::from("Hello, world!");
        let packet = Packet::data(payload.clone(), 1);

        assert_eq!(packet.header.packet_type, PacketType::Data);
        assert_eq!(packet.header.layer, 1);
        assert_eq!(packet.payload, payload);
    }

    #[test]
    fn test_packet_encode_decode() {
        let original = Packet::data(Bytes::from("test payload"), 2);
        let encoded = original.encode().unwrap();
        let decoded = Packet::parse(&encoded).unwrap();

        assert_eq!(original.header.packet_type, decoded.header.packet_type);
        assert_eq!(original.header.layer, decoded.header.layer);
        assert_eq!(original.payload, decoded.payload);
    }

    #[test]
    fn test_cover_traffic() {
        let packet = Packet::cover_traffic(100, 3);
        assert!(packet.is_cover_traffic());
        assert_eq!(packet.layer(), 3);
        assert_eq!(packet.payload.len(), 100);
    }

    #[test]
    fn test_packet_size_limit() {
        let large_payload = Bytes::from(vec![0u8; MAX_PACKET_SIZE]);
        let packet = Packet::data(large_payload, 1);
        assert!(packet.encode().is_err());
    }
}
