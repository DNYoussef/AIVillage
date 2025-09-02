//! Packet fragmentation for BLE low MTU support
//!
//! Handles packet fragmentation and reassembly for BLE constraints (20-100 bytes).

use std::collections::HashMap;
use std::time::Instant;

/// Maximum BLE packet size (conservative estimate)
const MAX_BLE_PACKET_SIZE: usize = 100;

/// Fragment header size (UUID + sequence + flags)
const FRAGMENT_HEADER_SIZE: usize = 20;

/// Maximum fragment payload size
pub const MAX_FRAGMENT_PAYLOAD: usize = MAX_BLE_PACKET_SIZE - FRAGMENT_HEADER_SIZE;

/// Fragment flags
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FragmentFlags {
    pub is_first: bool,
    pub is_last: bool,
    pub has_fec: bool,
    pub reserved: u8,
}

impl FragmentFlags {
    pub fn first() -> Self {
        Self { is_first: true, is_last: false, has_fec: false, reserved: 0 }
    }
    
    pub fn single() -> Self {
        Self { is_first: true, is_last: true, has_fec: false, reserved: 0 }
    }
}

/// Fragment header
#[derive(Debug, Clone)]
pub struct FragmentHeader {
    pub message_id: String,
    pub sequence: u16,
    pub total_fragments: u16,
    pub flags: FragmentFlags,
}

/// Individual fragment
#[derive(Debug, Clone)]
pub struct Fragment {
    pub header: FragmentHeader,
    pub payload: Vec<u8>,
}

/// Fragmentation engine with complete implementation
pub struct FragmentationEngine {
    reassembly_buffers: HashMap<String, ReassemblyBuffer>,
    max_message_size: usize,
}

/// Reassembly buffer for tracking incomplete messages
#[derive(Debug)]
struct ReassemblyBuffer {
    message_id: String,
    total_fragments: u16,
    fragments: HashMap<u16, Vec<u8>>,
    created_at: Instant,
}

impl FragmentationEngine {
    pub fn new() -> Self {
        Self {
            reassembly_buffers: HashMap::new(),
            max_message_size: 64 * 1024, // 64KB max message size
        }
    }

    /// Fragment a message into BLE-compatible chunks
    pub fn fragment_message(&self, payload: &[u8]) -> Result<Vec<Fragment>, FragmentationError> {
        if payload.len() > self.max_message_size {
            return Err(FragmentationError::MessageTooLarge(payload.len()));
        }

        let message_id = format!("msg_{}", std::process::id());
        let total_size = payload.len();
        
        if total_size <= MAX_FRAGMENT_PAYLOAD {
            // Single fragment case
            let fragment = Fragment {
                header: FragmentHeader {
                    message_id,
                    sequence: 0,
                    total_fragments: 1,
                    flags: FragmentFlags::single(),
                },
                payload: payload.to_vec(),
            };
            return Ok(vec![fragment]);
        }

        // Multi-fragment case
        let total_fragments = (total_size + MAX_FRAGMENT_PAYLOAD - 1) / MAX_FRAGMENT_PAYLOAD;
        let mut fragments = Vec::with_capacity(total_fragments);

        for (sequence, chunk) in payload.chunks(MAX_FRAGMENT_PAYLOAD).enumerate() {
            let flags = if sequence == 0 {
                FragmentFlags::first()
            } else if sequence == total_fragments - 1 {
                FragmentFlags { is_first: false, is_last: true, has_fec: false, reserved: 0 }
            } else {
                FragmentFlags { is_first: false, is_last: false, has_fec: false, reserved: 0 }
            };

            let fragment = Fragment {
                header: FragmentHeader {
                    message_id: message_id.clone(),
                    sequence: sequence as u16,
                    total_fragments: total_fragments as u16,
                    flags,
                },
                payload: chunk.to_vec(),
            };

            fragments.push(fragment);
        }

        Ok(fragments)
    }

    /// Process received fragment and attempt reassembly
    pub fn process_fragment(&mut self, fragment: Fragment) -> Result<Option<Vec<u8>>, FragmentationError> {
        let message_id = fragment.header.message_id.clone();
        let sequence = fragment.header.sequence;
        let total_fragments = fragment.header.total_fragments;

        // Handle single fragment case
        if fragment.header.flags.is_first && fragment.header.flags.is_last {
            return Ok(Some(fragment.payload));
        }

        // Get or create reassembly buffer
        let buffer = self.reassembly_buffers.entry(message_id.clone()).or_insert_with(|| {
            ReassemblyBuffer {
                message_id: message_id.clone(),
                total_fragments,
                fragments: HashMap::new(),
                created_at: Instant::now(),
            }
        });

        // Store fragment payload
        buffer.fragments.insert(sequence, fragment.payload);

        // Check if reassembly is complete
        if buffer.fragments.len() == total_fragments as usize {
            let mut reassembled = Vec::new();
            
            // Reassemble in sequence order
            for seq in 0..total_fragments {
                if let Some(payload) = buffer.fragments.get(&seq) {
                    reassembled.extend_from_slice(payload);
                } else {
                    return Err(FragmentationError::MissingFragment(seq));
                }
            }

            // Remove completed buffer
            self.reassembly_buffers.remove(&message_id);
            return Ok(Some(reassembled));
        }

        Ok(None) // Not yet complete
    }

    /// Get fragmentation statistics
    pub fn get_stats(&self) -> FragmentationStats {
        FragmentationStats {
            active_buffers: self.reassembly_buffers.len(),
            total_memory_usage: self.estimate_memory_usage(),
        }
    }

    fn estimate_memory_usage(&self) -> usize {
        self.reassembly_buffers
            .values()
            .map(|buffer| {
                buffer.fragments.values().map(|f| f.len()).sum::<usize>()
                    + std::mem::size_of::<ReassemblyBuffer>()
            })
            .sum()
    }
}

impl Default for FragmentationEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Fragmentation statistics
#[derive(Debug)]
pub struct FragmentationStats {
    pub active_buffers: usize,
    pub total_memory_usage: usize,
}

/// Fragmentation errors
#[derive(Debug, thiserror::Error)]
pub enum FragmentationError {
    #[error("Message too large: {0} bytes")]
    MessageTooLarge(usize),
    
    #[error("Missing fragment at sequence {0}")]
    MissingFragment(u16),
    
    #[error("Invalid fragment header")]
    InvalidHeader,
}
