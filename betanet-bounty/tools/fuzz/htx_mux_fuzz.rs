#![no_main]

use libfuzzer_sys::fuzz_target;
use betanet_htx::{HtxSession, HtxConfig, HtxStream, Frame, FrameType};
use bytes::Bytes;
use std::net::SocketAddr;

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }

    // Create HTX session for multiplexing tests
    let mut config = HtxConfig::default();
    config.enable_noise_xk = false; // Disable for fuzzing simplicity

    let mut session = match HtxSession::new(config, true) {
        Ok(s) => s,
        Err(_) => return,
    };

    // Split input data for different operations
    let (stream_id_bytes, frame_data) = data.split_at(4);
    let stream_id = u32::from_be_bytes([
        stream_id_bytes[0],
        stream_id_bytes[1],
        stream_id_bytes[2],
        stream_id_bytes[3]
    ]) % 268_435_455; // Clamp to MAX_STREAM_ID

    // Test stream creation and management
    let _ = session.create_stream();
    session.close_stream(stream_id);

    // Test individual stream operations
    let mut stream = HtxStream::new(stream_id);

    // Test stream state operations
    let _ = stream.can_send();
    let _ = stream.has_data();

    // Test window operations
    if frame_data.len() >= 4 {
        let window_delta = u32::from_be_bytes([
            frame_data[0], frame_data[1], frame_data[2], frame_data[3]
        ]);

        stream.update_send_window(window_delta);
        let _ = stream.consume_send_window(window_delta % 65536);
    }

    // Test stream data operations
    stream.write_to_buffer(&frame_data[4..]);

    let mut read_buf = vec![0u8; frame_data.len()];
    let _ = stream.read(&mut read_buf);

    // Test frame processing through session
    if frame_data.len() >= 4 {
        let frame_type_byte = frame_data[0] % 8; // HTX has 8 frame types
        let frame_type = match frame_type_byte {
            0 => FrameType::Data,
            1 => FrameType::WindowUpdate,
            2 => FrameType::KeyUpdate,
            3 => FrameType::Ping,
            4 => FrameType::Priority,
            5 => FrameType::Padding,
            6 => FrameType::AccessTicket,
            _ => FrameType::Control,
        };

        // Try to create and process frames
        let payload = Bytes::from(frame_data[1..].to_vec());
        match frame_type {
            FrameType::Data if stream_id > 0 => {
                if let Ok(frame) = Frame::data(stream_id, payload) {
                    let encoded = frame.encode();
                    // Test async data processing (can't await in fuzz target, but we can test sync parts)
                    // Runtime would be: let _ = session.process_data(&encoded).await;
                }
            }
            FrameType::WindowUpdate => {
                if payload.len() >= 4 {
                    let window_delta = u32::from_be_bytes([
                        payload[0], payload[1], payload[2], payload[3]
                    ]) % 0x7FFF_FFFF + 1;
                    if let Ok(frame) = Frame::window_update(stream_id, window_delta) {
                        let _ = frame.encode();
                    }
                }
            }
            FrameType::Ping => {
                let ping_data = if payload.len() <= 8 { Some(payload) } else { None };
                if let Ok(frame) = Frame::ping(ping_data) {
                    let _ = frame.encode();
                }
            }
            _ => {
                if let Ok(frame) = Frame::new(frame_type, stream_id, payload) {
                    let _ = frame.encode();
                }
            }
        }
    }

    // Test session status and statistics
    let _ = session.status();
    let _ = session.stats.clone();

    // Test configuration edge cases
    let mut test_config = HtxConfig::default();
    test_config.max_connections = (stream_id % 10000).max(1);
    test_config.connection_timeout_secs = (frame_data[0] as u64 % 3600).max(1);
    test_config.frame_buffer_size = ((stream_id % 1048576).max(1024)) as usize;

    // Try creating session with fuzzed config
    let _ = HtxSession::new(test_config, stream_id % 2 == 0);
});
