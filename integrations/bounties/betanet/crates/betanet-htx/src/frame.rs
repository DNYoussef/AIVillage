//! HTX frame format implementation
//!
//! Implements the HTX v1.1 frame format with zero-copy parsing using BytesMut:
//! - uint24 length field (big-endian)
//! - varint stream_id encoding (LEB128)
//! - uint8 frame type
//! - variable payload

use bytes::{Buf, BufMut, Bytes, BytesMut};
#[cfg(test)]
use proptest::prelude::*;
use thiserror::Error;

/// Maximum frame size (2^24 - 1)
pub const MAX_FRAME_SIZE: usize = 16_777_215;
/// Maximum stream ID (2^28 - 1)
pub const MAX_STREAM_ID: u32 = 268_435_455;

/// HTX frame types per Betanet v1.1 specification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum FrameType {
    Data = 0x00,
    WindowUpdate = 0x01,
    KeyUpdate = 0x02,
    Ping = 0x03,
    Priority = 0x04,
    Padding = 0x05,
    AccessTicket = 0x06,
    Control = 0x07,
}

impl TryFrom<u8> for FrameType {
    type Error = FrameError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0x00 => Ok(Self::Data),
            0x01 => Ok(Self::WindowUpdate),
            0x02 => Ok(Self::KeyUpdate),
            0x03 => Ok(Self::Ping),
            0x04 => Ok(Self::Priority),
            0x05 => Ok(Self::Padding),
            0x06 => Ok(Self::AccessTicket),
            0x07 => Ok(Self::Control),
            _ => Err(FrameError::InvalidFrameType(value)),
        }
    }
}

/// HTX frame errors
#[derive(Debug, Error)]
pub enum FrameError {
    #[error("Invalid frame type: 0x{0:02x}")]
    InvalidFrameType(u8),

    #[error("Frame too large: {0} > {}", MAX_FRAME_SIZE)]
    FrameTooLarge(usize),

    #[error("Stream ID too large: {0} > {}", MAX_STREAM_ID)]
    StreamIdTooLarge(u32),

    #[error("Frame too short: need at least {expected}, got {actual}")]
    FrameTooShort { expected: usize, actual: usize },

    #[error("Invalid varint: {0}")]
    InvalidVarint(String),

    #[error("Incomplete frame: need {needed} more bytes")]
    IncompleteFrame { needed: usize },

    #[error("Window delta must be positive, got {0}")]
    InvalidWindowDelta(u32),

    #[error("PING payload too large: {0} > 8")]
    PingPayloadTooLarge(usize),
}

/// Parsed HTX frame
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    pub frame_type: FrameType,
    pub stream_id: u32,
    pub payload: Bytes,
}

impl Frame {
    /// Create a new frame
    pub fn new(frame_type: FrameType, stream_id: u32, payload: Bytes) -> Result<Self, FrameError> {
        if stream_id > MAX_STREAM_ID {
            return Err(FrameError::StreamIdTooLarge(stream_id));
        }

        let varint_len = varint_length(stream_id);
        let total_content_len = varint_len + 1 + payload.len(); // stream_id + type + payload

        if total_content_len > MAX_FRAME_SIZE {
            return Err(FrameError::FrameTooLarge(total_content_len));
        }

        Ok(Self {
            frame_type,
            stream_id,
            payload,
        })
    }

    /// Create a DATA frame
    pub fn data(stream_id: u32, payload: Bytes) -> Result<Self, FrameError> {
        if stream_id == 0 {
            return Err(FrameError::StreamIdTooLarge(0)); // DATA frames need stream_id > 0
        }
        Self::new(FrameType::Data, stream_id, payload)
    }

    /// Create a WINDOW_UPDATE frame
    pub fn window_update(stream_id: u32, window_delta: u32) -> Result<Self, FrameError> {
        if window_delta == 0 || window_delta > 0x7FFF_FFFF {
            return Err(FrameError::InvalidWindowDelta(window_delta));
        }

        let mut payload = BytesMut::with_capacity(4);
        payload.put_u32(window_delta);
        Self::new(FrameType::WindowUpdate, stream_id, payload.freeze())
    }

    /// Create a PING frame
    pub fn ping(ping_data: Option<Bytes>) -> Result<Self, FrameError> {
        let payload = match ping_data {
            Some(data) => {
                if data.len() > 8 {
                    return Err(FrameError::PingPayloadTooLarge(data.len()));
                }
                data
            }
            None => {
                // Default to current timestamp in milliseconds
                let mut buf = BytesMut::with_capacity(8);
                let timestamp = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64;
                buf.put_u64(timestamp);
                buf.freeze()
            }
        };

        Self::new(FrameType::Ping, 0, payload)
    }

    /// Create a KEY_UPDATE frame
    pub fn key_update(ephemeral_public_key: Bytes) -> Result<Self, FrameError> {
        Self::new(FrameType::KeyUpdate, 0, ephemeral_public_key)
    }

    /// Encode frame to bytes with HTX v1.1 format
    pub fn encode(&self) -> Bytes {
        let stream_id_bytes = encode_varint(self.stream_id);
        let content_length = stream_id_bytes.len() + 1 + self.payload.len();

        let mut buf = BytesMut::with_capacity(3 + content_length);

        // uint24 length (big-endian)
        buf.put_u8((content_length >> 16) as u8);
        buf.put_u8((content_length >> 8) as u8);
        buf.put_u8(content_length as u8);

        // varint stream_id
        buf.put(stream_id_bytes);

        // uint8 frame type
        buf.put_u8(self.frame_type as u8);

        // payload
        buf.put(self.payload.as_ref());

        buf.freeze()
    }

    /// Get total frame size including headers
    pub fn size(&self) -> usize {
        3 + varint_length(self.stream_id) + 1 + self.payload.len()
    }
}

/// HTX frame buffer for accumulating and parsing frames from stream data
#[derive(Debug)]
pub struct FrameBuffer {
    buffer: BytesMut,
    max_buffer_size: usize,
}

impl FrameBuffer {
    /// Create new frame buffer
    pub fn new(max_buffer_size: usize) -> Self {
        Self {
            buffer: BytesMut::new(),
            max_buffer_size,
        }
    }

    /// Append data to buffer
    pub fn append_data(&mut self, data: &[u8]) -> Result<(), FrameError> {
        if self.buffer.len() + data.len() > self.max_buffer_size {
            return Err(FrameError::FrameTooLarge(self.buffer.len() + data.len()));
        }

        self.buffer.extend_from_slice(data);
        Ok(())
    }

    /// Parse all complete frames from buffer
    pub fn parse_frames(&mut self) -> Result<Vec<Frame>, FrameError> {
        let mut frames = Vec::new();
        let mut offset = 0;

        while offset < self.buffer.len() {
            match parse_frame(&self.buffer[offset..]) {
                Ok((frame, consumed)) => {
                    frames.push(frame);
                    offset += consumed;
                }
                Err(FrameError::IncompleteFrame { .. }) => {
                    // Need more data
                    break;
                }
                Err(e) => {
                    // Skip malformed data and continue
                    tracing::warn!("Skipping malformed frame data at offset {}: {}", offset, e);
                    offset += 1;
                }
            }
        }

        // Remove consumed bytes
        if offset > 0 {
            self.buffer.advance(offset);
        }

        Ok(frames)
    }

    /// Get buffer utilization
    pub fn utilization(&self) -> f64 {
        self.buffer.len() as f64 / self.max_buffer_size as f64
    }

    /// Get buffer statistics
    pub fn stats(&self) -> FrameBufferStats {
        FrameBufferStats {
            buffer_size: self.buffer.len(),
            max_buffer_size: self.max_buffer_size,
            utilization: self.utilization(),
        }
    }
}

impl Default for FrameBuffer {
    fn default() -> Self {
        Self::new(1048576) // 1MB default
    }
}

/// Frame buffer statistics
#[derive(Debug, Clone)]
pub struct FrameBufferStats {
    pub buffer_size: usize,
    pub max_buffer_size: usize,
    pub utilization: f64,
}

/// Parse a single frame from bytes
pub fn parse_frame(data: &[u8]) -> Result<(Frame, usize), FrameError> {
    if data.len() < 4 {
        return Err(FrameError::IncompleteFrame {
            needed: 4 - data.len(),
        });
    }

    // Parse uint24 length (big-endian)
    let length = ((data[0] as usize) << 16) | ((data[1] as usize) << 8) | (data[2] as usize);

    if length > MAX_FRAME_SIZE {
        return Err(FrameError::FrameTooLarge(length));
    }

    let total_frame_size = 3 + length;
    if data.len() < total_frame_size {
        return Err(FrameError::IncompleteFrame {
            needed: total_frame_size - data.len(),
        });
    }

    // Parse frame content
    let content = &data[3..3 + length];

    // Decode varint stream_id
    let (stream_id, varint_consumed) = decode_varint(content)?;

    if content.len() < varint_consumed + 1 {
        return Err(FrameError::FrameTooShort {
            expected: varint_consumed + 1,
            actual: content.len(),
        });
    }

    // Decode frame type
    let frame_type = FrameType::try_from(content[varint_consumed])?;

    // Extract payload
    let payload_start = varint_consumed + 1;
    let payload = Bytes::copy_from_slice(&content[payload_start..]);

    let frame = Frame {
        frame_type,
        stream_id,
        payload,
    };

    Ok((frame, total_frame_size))
}

/// Encode integer as varint (LEB128)
fn encode_varint(mut value: u32) -> Bytes {
    let mut result = Vec::new();

    while value >= 0x80 {
        result.push((value & 0x7F) as u8 | 0x80);
        value >>= 7;
    }
    result.push(value as u8);

    Bytes::from(result)
}

/// Decode varint from bytes
fn decode_varint(data: &[u8]) -> Result<(u32, usize), FrameError> {
    if data.is_empty() {
        return Err(FrameError::InvalidVarint("empty data".to_string()));
    }

    let mut value = 0u32;
    let mut shift = 0;
    let mut bytes_consumed = 0;

    for &byte in data {
        bytes_consumed += 1;

        value |= ((byte & 0x7F) as u32) << shift;

        if (byte & 0x80) == 0 {
            // End of varint
            if value > MAX_STREAM_ID {
                return Err(FrameError::StreamIdTooLarge(value));
            }
            return Ok((value, bytes_consumed));
        }

        shift += 7;
        if shift >= 28 {
            return Err(FrameError::InvalidVarint("varint too long".to_string()));
        }
    }

    Err(FrameError::InvalidVarint("incomplete varint".to_string()))
}

/// Get length of varint encoding for a value
fn varint_length(value: u32) -> usize {
    if value < 0x80 {
        1
    } else if value < 0x4000 {
        2
    } else if value < 0x200000 {
        3
    } else {
        4 // Max for 28-bit values
    }
}

// Property tests for encode/decode roundtrip
#[cfg(test)]
prop_compose! {
    fn arb_frame_type()(idx in 0u8..8) -> FrameType {
        match idx {
            0 => FrameType::Data,
            1 => FrameType::WindowUpdate,
            2 => FrameType::KeyUpdate,
            3 => FrameType::Ping,
            4 => FrameType::Priority,
            5 => FrameType::Padding,
            6 => FrameType::AccessTicket,
            _ => FrameType::Control,
        }
    }
}

#[cfg(test)]
prop_compose! {
    fn arb_stream_id()(id in 0u32..=MAX_STREAM_ID) -> u32 {
        id
    }
}

#[cfg(test)]
prop_compose! {
    fn arb_payload()(size in 0usize..1024, data in prop::collection::vec(any::<u8>(), 0..1024)) -> Bytes {
        Bytes::from(data[..size.min(data.len())].to_vec())
    }
}

#[cfg(test)]
prop_compose! {
    fn arb_frame()(frame_type in arb_frame_type(), stream_id in arb_stream_id(), payload in arb_payload()) -> Frame {
        Frame { frame_type, stream_id, payload }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_type_conversion() {
        assert_eq!(FrameType::try_from(0x00).unwrap(), FrameType::Data);
        assert_eq!(FrameType::try_from(0x03).unwrap(), FrameType::Ping);
        assert!(FrameType::try_from(0xFF).is_err());
    }

    #[test]
    fn test_varint_encode_decode() {
        let test_values = [0, 127, 128, 16383, 16384, 2097151, 2097152, MAX_STREAM_ID];

        for &value in &test_values {
            let encoded = encode_varint(value);
            let (decoded, consumed) = decode_varint(&encoded).unwrap();
            assert_eq!(decoded, value);
            assert_eq!(consumed, encoded.len());
        }
    }

    #[test]
    fn test_frame_creation() {
        let frame = Frame::data(1, Bytes::from("hello")).unwrap();
        assert_eq!(frame.frame_type, FrameType::Data);
        assert_eq!(frame.stream_id, 1);
        assert_eq!(frame.payload, Bytes::from("hello"));

        // DATA frames require stream_id > 0
        assert!(Frame::data(0, Bytes::from("hello")).is_err());
    }

    #[test]
    fn test_window_update_frame() {
        let frame = Frame::window_update(1, 1024).unwrap();
        assert_eq!(frame.frame_type, FrameType::WindowUpdate);
        assert_eq!(frame.stream_id, 1);

        // Invalid window delta
        assert!(Frame::window_update(1, 0).is_err());
        assert!(Frame::window_update(1, 0x8000_0000).is_err());
    }

    #[test]
    fn test_ping_frame() {
        let frame = Frame::ping(None).unwrap();
        assert_eq!(frame.frame_type, FrameType::Ping);
        assert_eq!(frame.stream_id, 0);
        assert_eq!(frame.payload.len(), 8); // timestamp

        let custom_ping = Frame::ping(Some(Bytes::from("test"))).unwrap();
        assert_eq!(custom_ping.payload, Bytes::from("test"));

        // Payload too large
        let large_payload = Bytes::from(vec![0u8; 9]);
        assert!(Frame::ping(Some(large_payload)).is_err());
    }

    #[test]
    fn test_frame_buffer() {
        let mut buffer = FrameBuffer::new(1024);

        // Create test frame
        let frame = Frame::ping(Some(Bytes::from("test"))).unwrap();
        let encoded = frame.encode();

        // Test normal parsing
        buffer.append_data(&encoded).unwrap();
        let frames = buffer.parse_frames().unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0], frame);

        // Test buffer overflow
        let large_data = vec![0u8; 2000];
        assert!(buffer.append_data(&large_data).is_err());
    }

    proptest! {
        #[test]
        fn prop_frame_encode_decode_roundtrip(frame in arb_frame()) {
            let encoded = frame.encode();
            let (decoded, consumed) = parse_frame(&encoded).unwrap();

            assert_eq!(decoded.frame_type, frame.frame_type);
            assert_eq!(decoded.stream_id, frame.stream_id);
            assert_eq!(decoded.payload, frame.payload);
            assert_eq!(consumed, encoded.len());
        }

        #[test]
        fn prop_varint_roundtrip(value in 0u32..=MAX_STREAM_ID) {
            let encoded = encode_varint(value);
            let (decoded, consumed) = decode_varint(&encoded).unwrap();
            assert_eq!(decoded, value);
            assert_eq!(consumed, encoded.len());
        }

        #[test]
        fn prop_frame_size_consistency(frame in arb_frame()) {
            let encoded = frame.encode();
            assert_eq!(frame.size(), encoded.len());
        }
    }
}
