//! Forward Error Correction (FEC) with XOR fountain codes for BitChat BLE
//!
//! Implements simple XOR-based erasure coding to recover lost BLE packets.

use std::collections::HashMap;
use thiserror::Error;

/// FEC configuration parameters
#[derive(Debug, Clone)]
pub struct FecConfig {
    /// Redundancy ratio (0.0 to 1.0) - fraction of additional parity packets
    pub redundancy_ratio: f32,
    /// Maximum number of data packets per FEC block
    pub max_block_size: usize,
    /// Enable systematic encoding (original data + parity)
    pub systematic: bool,
}

impl Default for FecConfig {
    fn default() -> Self {
        Self {
            redundancy_ratio: 0.3, // 30% redundancy
            max_block_size: 16,
            systematic: true,
        }
    }
}

/// FEC encoder for creating parity packets
pub struct FecEncoder {
    config: FecConfig,
}

/// FEC decoder for recovering lost packets
pub struct FecDecoder {
    config: FecConfig,
    partial_blocks: HashMap<u32, PartialBlock>,
}

/// FEC-encoded packet
#[derive(Debug, Clone)]
pub struct FecPacket {
    pub block_id: u32,
    pub packet_index: u16,
    pub is_parity: bool,
    pub total_data_packets: u16,
    pub total_parity_packets: u16,
    pub payload: Vec<u8>,
}

/// Partial block during decoding
#[derive(Debug)]
struct PartialBlock {
    block_id: u32,
    data_packets: HashMap<u16, Vec<u8>>,
    parity_packets: HashMap<u16, Vec<u8>>,
    total_data_packets: u16,
    total_parity_packets: u16,
    expected_packet_size: usize,
}

impl FecEncoder {
    pub fn new(config: FecConfig) -> Self {
        Self { config }
    }

    /// Encode data into FEC packets with parity information
    pub fn encode(&self, data: &[u8], block_id: u32) -> Result<Vec<FecPacket>, FecError> {
        if data.is_empty() {
            return Err(FecError::EmptyData);
        }

        // Calculate packet size based on data length and block size
        let packet_size = (data.len() + self.config.max_block_size - 1) / self.config.max_block_size;
        let num_data_packets = (data.len() + packet_size - 1) / packet_size;
        let num_parity_packets = ((num_data_packets as f32) * self.config.redundancy_ratio).ceil() as usize;

        let mut result = Vec::new();

        // Create data packets
        for (i, chunk) in data.chunks(packet_size).enumerate() {
            let mut padded_chunk = chunk.to_vec();
            // Pad to uniform packet size
            padded_chunk.resize(packet_size, 0);

            let packet = FecPacket {
                block_id,
                packet_index: i as u16,
                is_parity: false,
                total_data_packets: num_data_packets as u16,
                total_parity_packets: num_parity_packets as u16,
                payload: padded_chunk,
            };
            result.push(packet);
        }

        // Create parity packets using simple XOR
        for parity_index in 0..num_parity_packets {
            let mut parity_data = vec![0u8; packet_size];
            
            // XOR relevant data packets to create parity
            for data_index in 0..num_data_packets {
                if self.should_include_in_parity(data_index, parity_index, num_data_packets) {
                    let data_packet = &result[data_index].payload;
                    for (p, d) in parity_data.iter_mut().zip(data_packet.iter()) {
                        *p ^= *d;
                    }
                }
            }

            let parity_packet = FecPacket {
                block_id,
                packet_index: parity_index as u16,
                is_parity: true,
                total_data_packets: num_data_packets as u16,
                total_parity_packets: num_parity_packets as u16,
                payload: parity_data,
            };
            result.push(parity_packet);
        }

        Ok(result)
    }

    /// Simple XOR pattern for determining which data packets contribute to each parity
    fn should_include_in_parity(&self, data_index: usize, parity_index: usize, total_data: usize) -> bool {
        // Simple pattern: each parity packet XORs a different subset of data packets
        let pattern = (parity_index + 1) * 17; // Prime number for better distribution
        (data_index * pattern) % (total_data + 1) <= total_data / 2
    }
}

impl FecDecoder {
    pub fn new(config: FecConfig) -> Self {
        Self {
            config,
            partial_blocks: HashMap::new(),
        }
    }

    /// Process received FEC packet and attempt to decode block
    pub fn process_packet(&mut self, packet: FecPacket) -> Result<Option<Vec<u8>>, FecError> {
        let block_id = packet.block_id;
        
        // Get or create partial block
        let partial = self.partial_blocks.entry(block_id).or_insert_with(|| {
            PartialBlock {
                block_id,
                data_packets: HashMap::new(),
                parity_packets: HashMap::new(),
                total_data_packets: packet.total_data_packets,
                total_parity_packets: packet.total_parity_packets,
                expected_packet_size: packet.payload.len(),
            }
        });

        // Validate packet consistency
        if partial.total_data_packets != packet.total_data_packets ||
           partial.total_parity_packets != packet.total_parity_packets ||
           partial.expected_packet_size != packet.payload.len() {
            return Err(FecError::InconsistentBlock);
        }

        // Store packet
        if packet.is_parity {
            partial.parity_packets.insert(packet.packet_index, packet.payload);
        } else {
            partial.data_packets.insert(packet.packet_index, packet.payload);
        }

        // Check if we can decode
        self.attempt_decode(block_id)
    }

    /// Attempt to decode a complete block
    fn attempt_decode(&mut self, block_id: u32) -> Result<Option<Vec<u8>>, FecError> {
        let partial = self.partial_blocks.get(&block_id)
            .ok_or(FecError::BlockNotFound)?;

        // Check if we have all data packets
        if partial.data_packets.len() == partial.total_data_packets as usize {
            // We have all data, can reconstruct immediately
            return self.reconstruct_data(block_id);
        }

        // Check if we have enough packets total to potentially recover
        let total_received = partial.data_packets.len() + partial.parity_packets.len();
        if total_received < partial.total_data_packets as usize {
            return Ok(None); // Not enough packets yet
        }

        // Attempt recovery using available parity packets
        self.recover_with_parity(block_id)
    }

    /// Reconstruct original data from complete data packets
    fn reconstruct_data(&mut self, block_id: u32) -> Result<Option<Vec<u8>>, FecError> {
        let partial = self.partial_blocks.remove(&block_id)
            .ok_or(FecError::BlockNotFound)?;

        let mut result = Vec::new();
        
        // Concatenate data packets in order
        for i in 0..partial.total_data_packets {
            if let Some(packet_data) = partial.data_packets.get(&i) {
                result.extend_from_slice(packet_data);
            } else {
                return Err(FecError::MissingDataPacket(i));
            }
        }

        // Remove padding (simplified - assumes zeros are padding)
        while result.last() == Some(&0) {
            result.pop();
        }

        Ok(Some(result))
    }

    /// Attempt to recover missing data using parity packets
    fn recover_with_parity(&mut self, block_id: u32) -> Result<Option<Vec<u8>>, FecError> {
        let partial = self.partial_blocks.get_mut(&block_id)
            .ok_or(FecError::BlockNotFound)?;

        // Simple XOR-based recovery (simplified implementation)
        let missing_indices: Vec<u16> = (0..partial.total_data_packets)
            .filter(|&i| !partial.data_packets.contains_key(&i))
            .collect();

        if missing_indices.len() > partial.parity_packets.len() {
            return Ok(None); // Cannot recover - too many missing
        }

        // For each missing packet, find a parity that can recover it
        let mut recovered_packets = Vec::new();
        let config = self.config.clone(); // Clone config to avoid borrow issues
        
        for &missing_idx in &missing_indices {
            if let Some(recovered) = Self::recover_single_packet_static(&config, partial, missing_idx) {
                recovered_packets.push((missing_idx, recovered));
            } else {
                return Ok(None); // Cannot recover this packet
            }
        }
        
        // Insert recovered packets
        for (idx, data) in recovered_packets {
            partial.data_packets.insert(idx, data);
        }

        // If we recovered all missing packets, reconstruct
        if partial.data_packets.len() == partial.total_data_packets as usize {
            self.reconstruct_data(block_id)
        } else {
            Ok(None)
        }
    }

    /// Recover a single missing packet using available parity (static method)
    fn recover_single_packet_static(config: &FecConfig, partial: &PartialBlock, missing_idx: u16) -> Option<Vec<u8>> {
        // Find a parity packet that includes this missing data packet
        for (&parity_idx, parity_data) in &partial.parity_packets {
            let encoder = FecEncoder::new(config.clone());
            if encoder.should_include_in_parity(missing_idx as usize, parity_idx as usize, partial.total_data_packets as usize) {
                // XOR this parity with all known data packets that contribute to it
                let mut recovered = parity_data.clone();
                
                for (&data_idx, data_packet) in &partial.data_packets {
                    if encoder.should_include_in_parity(data_idx as usize, parity_idx as usize, partial.total_data_packets as usize) {
                        for (r, d) in recovered.iter_mut().zip(data_packet.iter()) {
                            *r ^= *d;
                        }
                    }
                }
                
                return Some(recovered);
            }
        }
        None
    }

    /// Get decoding statistics
    pub fn get_stats(&self) -> FecStats {
        let total_blocks = self.partial_blocks.len();
        let total_packets = self.partial_blocks.values()
            .map(|b| b.data_packets.len() + b.parity_packets.len())
            .sum();

        FecStats {
            active_blocks: total_blocks,
            total_packets_cached: total_packets,
        }
    }

    /// Clean up old partial blocks
    pub fn cleanup_old_blocks(&mut self, max_blocks: usize) {
        if self.partial_blocks.len() > max_blocks {
            // Remove oldest blocks (simplified - just remove some)
            let to_remove: Vec<u32> = self.partial_blocks.keys()
                .take(self.partial_blocks.len() - max_blocks)
                .cloned()
                .collect();
            
            for block_id in to_remove {
                self.partial_blocks.remove(&block_id);
            }
        }
    }
}

/// FEC statistics
#[derive(Debug)]
pub struct FecStats {
    pub active_blocks: usize,
    pub total_packets_cached: usize,
}

/// FEC errors
#[derive(Debug, Error)]
pub enum FecError {
    #[error("Empty data provided")]
    EmptyData,
    
    #[error("Inconsistent block parameters")]
    InconsistentBlock,
    
    #[error("Block not found")]
    BlockNotFound,
    
    #[error("Missing data packet: {0}")]
    MissingDataPacket(u16),
    
    #[error("Cannot recover - insufficient parity")]
    InsufficientParity,
}
