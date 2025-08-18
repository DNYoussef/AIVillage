#![no_main]

use libfuzzer_sys::fuzz_target;
use betanet_htx::{NoiseXK, generate_keypair, HandshakePhase};
use bytes::Bytes;

fuzz_target!(|data: &[u8]| {
    if data.len() < 16 {
        return;
    }

    // Split data for different test scenarios
    let (keypair_seed, handshake_data) = data.split_at(8);
    let (message_data, remaining) = handshake_data.split_at(handshake_data.len() / 2);

    // Test keypair generation with seeded randomness
    let _ = generate_keypair();

    // Create initiator and responder
    let initiator_result = NoiseXK::new(true, None, None);
    let responder_result = NoiseXK::new(false, None, None);

    if let (Ok(mut initiator), Ok(mut responder)) = (initiator_result, responder_result) {
        // Test handshake phase tracking without panicking
        if initiator.phase() != HandshakePhase::Uninitialized
            || responder.phase() != HandshakePhase::Uninitialized
        {
            return;
        }

        // Test message 1 creation and processing
        if let Ok(fragments) = initiator.create_message_1() {
            if let Some(first_fragment) = fragments.first() {
                // Fuzz message 1 processing
                let fuzzed_msg1 = [&first_fragment.data[..message_data.len().min(first_fragment.data.len())], message_data].concat();
                let _ = responder.process_message_1(&fuzzed_msg1);

                // Test with original message 1
                if responder.process_message_1(&first_fragment.data).is_ok() {
                    // Test message 2 creation and processing
                    if let Ok(msg2_fragments) = responder.create_message_2() {
                        if let Some(msg2_fragment) = msg2_fragments.first() {
                            // Fuzz message 2
                            let fuzzed_msg2 = [&msg2_fragment.data[..message_data.len().min(msg2_fragment.data.len())], remaining].concat();
                            let _ = initiator.process_message_2(&fuzzed_msg2);

                            // Test with original message 2
                            if initiator.process_message_2(&msg2_fragment.data).is_ok() {
                                // Test message 3 creation and processing
                                if let Ok(msg3_fragments) = initiator.create_message_3() {
                                    if let Some(msg3_fragment) = msg3_fragments.first() {
                                        // Fuzz message 3
                                        let fuzzed_msg3 = [&msg3_fragment.data[..remaining.len().min(msg3_fragment.data.len())], message_data].concat();
                                        let _ = responder.process_message_3(&fuzzed_msg3);

                                        // Complete handshake with original message 3
                                        if responder.process_message_3(&msg3_fragment.data).is_ok() {
                                            // Test transport mode operations
                                            if initiator.is_transport_ready() && responder.is_transport_ready() {
                                                // Test encryption/decryption with fuzzed data
                                                if let Ok(ciphertext) = initiator.encrypt(message_data) {
                                                    let _ = responder.decrypt(&ciphertext);

                                                    // Fuzz the ciphertext
                                                    let mut fuzzed_ciphertext = ciphertext.to_vec();
                                                    for (i, &byte) in remaining.iter().enumerate() {
                                                        if i < fuzzed_ciphertext.len() {
                                                            fuzzed_ciphertext[i] ^= byte;
                                                        }
                                                    }
                                                    let _ = responder.decrypt(&fuzzed_ciphertext);
                                                }

                                                // Test bidirectional encryption
                                                if let Ok(response_ct) = responder.encrypt(remaining) {
                                                    let _ = initiator.decrypt(&response_ct);
                                                }

                                                // Test key update operations
                                                if remaining.len() >= 32 {
                                                    let key_update_data = &remaining[..32];
                                                    let _ = initiator.process_key_update(key_update_data);
                                                    let _ = responder.process_key_update(key_update_data);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Test status reporting
        let _ = initiator.status();
        let _ = responder.status();

        // Test error conditions
        let _ = initiator.encrypt(&[]);
        let _ = responder.decrypt(&[]);
        let _ = initiator.process_message_1(message_data);
        let _ = responder.process_message_2(remaining);
    }

    // Test with custom static keys
    if data.len() >= 64 {
        let static_key = &data[..32];
        let remote_key = &data[32..64];

        let _ = NoiseXK::new(true, Some(static_key), Some(remote_key));
        let _ = NoiseXK::new(false, Some(static_key), Some(remote_key));
    }

    // Test invalid key lengths
    if data.len() >= 16 {
        let short_key = &data[..16];
        let _ = NoiseXK::new(true, Some(short_key), None);
    }

    // Test empty and minimal inputs
    let _ = NoiseXK::new(true, Some(&[]), None);
    let _ = NoiseXK::new(false, None, Some(&[]));
});
