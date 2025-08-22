#![no_main]

use libfuzzer_sys::fuzz_target;
use betanet_htx::{Frame, FrameType, parse_frame, FrameBuffer};
use bytes::Bytes;

fuzz_target!(|data: &[u8]| {
    // Fuzz frame parsing with arbitrary input
    if data.len() < 4 {
        return;
    }

    // Test direct frame parsing from raw bytes
    let _ = parse_frame(data);

    // Try to parse specific frame types with valid stream IDs
    let frame_types = [
        FrameType::Data,
        FrameType::WindowUpdate,
        FrameType::KeyUpdate,
        FrameType::Ping,
        FrameType::Priority,
        FrameType::Padding,
        FrameType::AccessTicket,
        FrameType::Control,
    ];

    for &frame_type in &frame_types {
        // Extract stream_id and payload from input data
        if data.len() >= 8 {
            let stream_id = u32::from_be_bytes([data[0], data[1], data[2], data[3]]) % 268_435_455; // MAX_STREAM_ID
            let payload = Bytes::from(data[4..].to_vec());

            // Try creating frames with different constructors
            match frame_type {
                FrameType::Data if stream_id > 0 => {
                    let _ = Frame::data(stream_id, payload.clone());
                }
                FrameType::WindowUpdate => {
                    if payload.len() >= 4 {
                        let window_delta = u32::from_be_bytes([
                            payload[0], payload[1], payload[2], payload[3]
                        ]) % 0x7FFF_FFFF + 1; // Valid window delta range
                        let _ = Frame::window_update(stream_id, window_delta);
                    }
                }
                FrameType::Ping => {
                    let ping_data = if payload.len() <= 8 { Some(payload.clone()) } else { None };
                    let _ = Frame::ping(ping_data);
                }
                FrameType::KeyUpdate => {
                    let _ = Frame::key_update(payload.clone());
                }
                _ => {
                    let _ = Frame::new(frame_type, stream_id, payload.clone());
                }
            }
        }
    }

    // Test frame buffer operations
    let mut frame_buffer = FrameBuffer::new(65536);
    let _ = frame_buffer.append_data(data);
    let _ = frame_buffer.parse_frames();
    let _ = frame_buffer.utilization();
    let _ = frame_buffer.stats();

    // Test encode/decode roundtrips on valid frames
    if data.len() >= 8 {
        let stream_id = u32::from_be_bytes([data[0], data[1], data[2], data[3]]) % 268_435_455;
        let payload = Bytes::from(data[4..].to_vec());

        if let Ok(frame) = Frame::new(FrameType::Data, stream_id.max(1), payload) {
            let encoded = frame.encode();
            let _ = parse_frame(&encoded);

            // Test frame size consistency
            let calculated_size = frame.size();
            assert_eq!(calculated_size, encoded.len());
        }
    }
});
