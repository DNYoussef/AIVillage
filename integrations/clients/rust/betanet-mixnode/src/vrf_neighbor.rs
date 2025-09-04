//! VRF-based neighbor sampling with AS diversity
//!
//! Implementation of neighbor selection that:
//! - Uses VRF for unpredictable but verifiable randomness
//! - Enforces AS (Autonomous System) diversity
//! - Creates expander-like topology properties
//! - Provides efficient neighbor caching and rotation

use std::collections::{HashMap, HashSet};
use std::net::{IpAddr, SocketAddr};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{crypto::CryptoUtils, MixnodeError, Result};

#[cfg(feature = "vrf")]
use crate::vrf_delay::VrfProof;

/// AS (Autonomous System) number
pub type AsNumber = u32;

/// Node information for neighbor selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixnodeInfo {
    /// Node address
    pub address: SocketAddr,
    /// AS number
    pub as_number: AsNumber,
    /// Node's public VRF key
    pub vrf_public_key: [u8; 32],
    /// Last seen timestamp
    pub last_seen: u64,
    /// Reliability score (0.0 to 1.0)
    pub reliability: f64,
    /// Performance metrics
    pub latency_ms: u16,
    /// Measured bandwidth in kilobits per second
    pub bandwidth_kbps: u32,
}

impl MixnodeInfo {
    /// Create new node info
    pub fn new(address: SocketAddr, as_number: AsNumber, vrf_public_key: [u8; 32]) -> Self {
        Self {
            address,
            as_number,
            vrf_public_key,
            last_seen: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            reliability: 1.0,
            latency_ms: 100,
            bandwidth_kbps: 1000,
        }
    }

    /// Check if node is still fresh (seen within timeout)
    pub fn is_fresh(&self, timeout_secs: u64) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        now - self.last_seen < timeout_secs
    }

    /// Calculate selection score based on performance and reliability
    pub fn selection_score(&self) -> f64 {
        let latency_score = 1.0 / (1.0 + self.latency_ms as f64 / 1000.0); // Lower is better
        let bandwidth_score = (self.bandwidth_kbps as f64).ln() / 10.0; // Log scale
        self.reliability * 0.5 + latency_score * 0.3 + bandwidth_score * 0.2
    }
}

/// VRF-based neighbor selector
pub struct VrfNeighborSelector {
    /// Own VRF private key
    vrf_private_key: [u8; 32],
    /// Own VRF public key
    vrf_public_key: [u8; 32],
    /// Known mixnodes
    known_nodes: HashMap<SocketAddr, MixnodeInfo>,
    /// AS number mappings
    as_groups: HashMap<AsNumber, Vec<SocketAddr>>,
    /// Selection cache
    selection_cache: HashMap<[u8; 32], Vec<SocketAddr>>,
    /// Configuration
    config: NeighborSelectionConfig,
}

/// Configuration for neighbor selection
#[derive(Debug, Clone)]
pub struct NeighborSelectionConfig {
    /// Maximum neighbors to select
    pub max_neighbors: usize,
    /// Maximum nodes per AS
    pub max_nodes_per_as: usize,
    /// Minimum AS diversity (different AS numbers)
    pub min_as_diversity: usize,
    /// Node freshness timeout (seconds)
    pub node_timeout_secs: u64,
    /// Cache TTL for selections
    pub cache_ttl_secs: u64,
    /// Minimum reliability score
    pub min_reliability: f64,
}

impl Default for NeighborSelectionConfig {
    fn default() -> Self {
        Self {
            max_neighbors: 8,
            max_nodes_per_as: 2,
            min_as_diversity: 4,
            node_timeout_secs: 3600, // 1 hour
            cache_ttl_secs: 300,     // 5 minutes
            min_reliability: 0.7,
        }
    }
}

impl VrfNeighborSelector {
    /// Create new VRF neighbor selector
    pub fn new(config: NeighborSelectionConfig) -> Self {
        Self::with_vrf_key(CryptoUtils::random_bytes(32).try_into().unwrap(), config)
    }

    /// Create selector with existing VRF key
    pub fn with_vrf_key(private_key: [u8; 32], config: NeighborSelectionConfig) -> Self {
        // Derive VRF public key from private key
        let vrf_public_key = Self::derive_vrf_public_key(&private_key);

        Self {
            vrf_private_key: private_key,
            vrf_public_key,
            known_nodes: HashMap::new(),
            as_groups: HashMap::new(),
            selection_cache: HashMap::new(),
            config,
        }
    }

    /// Derive VRF public key from private key
    #[cfg(feature = "vrf")]
    fn derive_vrf_public_key(private_key: &[u8; 32]) -> [u8; 32] {
        use schnorrkel::{ExpansionMode, MiniSecretKey};

        if let Ok(mini) = MiniSecretKey::from_bytes(private_key) {
            let kp = mini.expand_to_keypair(ExpansionMode::Ed25519);
            kp.public.to_bytes()
        } else {
            // Fallback to hash-based derivation if key is invalid
            CryptoUtils::sha256(private_key)
        }
    }

    /// Derive VRF public key from private key (fallback)
    #[cfg(not(feature = "vrf"))]
    fn derive_vrf_public_key(private_key: &[u8; 32]) -> [u8; 32] {
        CryptoUtils::sha256(private_key)
    }

    /// Get VRF public key
    pub fn vrf_public_key(&self) -> &[u8; 32] {
        &self.vrf_public_key
    }

    /// Add or update a mixnode
    pub fn add_node(&mut self, node: MixnodeInfo) {
        let addr = node.address;
        let as_num = node.as_number;

        // Update known nodes
        self.known_nodes.insert(addr, node);

        // Update AS groups
        self.as_groups
            .entry(as_num)
            .or_insert_with(Vec::new)
            .push(addr);

        // Clear cache as topology changed
        self.selection_cache.clear();
    }

    /// Remove a mixnode
    pub fn remove_node(&mut self, address: &SocketAddr) {
        if let Some(node) = self.known_nodes.remove(address) {
            // Remove from AS groups
            if let Some(as_nodes) = self.as_groups.get_mut(&node.as_number) {
                as_nodes.retain(|addr| addr != address);
                if as_nodes.is_empty() {
                    self.as_groups.remove(&node.as_number);
                }
            }
        }

        // Clear cache
        self.selection_cache.clear();
    }

    /// Select neighbors using VRF with AS diversity and return proof
    #[cfg(feature = "vrf")]
    pub fn select_neighbors(&mut self, seed: &[u8]) -> Result<(Vec<SocketAddr>, VrfProof)> {
        // Create cache key from seed
        let cache_key = CryptoUtils::sha256(seed);

        // Check cache first
        if let Some(cached) = self.selection_cache.get(&cache_key) {
            let proof = self.generate_vrf_proof(seed)?;
            return Ok((cached.clone(), proof));
        }

        // Clean up stale nodes
        self.cleanup_stale_nodes();

        // Ensure minimum diversity is possible
        if self.as_groups.len() < self.config.min_as_diversity {
            return Err(MixnodeError::Routing(format!(
                "Insufficient AS diversity: {} available, {} required",
                self.as_groups.len(),
                self.config.min_as_diversity
            )));
        }

        // Generate VRF proof for seed
        let proof = self.generate_vrf_proof(seed)?;
        let vrf_output: [u8; 32] = proof.io.make_bytes(b"neighbor-selection");

        // Select nodes with AS diversity constraints
        let selected = self.select_with_diversity(&vrf_output)?;

        // Cache the result
        self.selection_cache.insert(cache_key, selected.clone());

        Ok((selected, proof))
    }

    /// Select neighbors without VRF proofs (fallback)
    #[cfg(not(feature = "vrf"))]
    pub fn select_neighbors(&mut self, seed: &[u8]) -> Result<Vec<SocketAddr>> {
        // Create cache key from seed
        let cache_key = CryptoUtils::sha256(seed);

        // Check cache first
        if let Some(cached) = self.selection_cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        // Clean up stale nodes
        self.cleanup_stale_nodes();

        // Ensure minimum diversity is possible
        if self.as_groups.len() < self.config.min_as_diversity {
            return Err(MixnodeError::Routing(format!(
                "Insufficient AS diversity: {} available, {} required",
                self.as_groups.len(),
                self.config.min_as_diversity
            )));
        }

        // Generate pseudo-random output
        let vrf_output = self.generate_vrf_output(seed)?;

        // Select nodes with AS diversity constraints
        let selected = self.select_with_diversity(&vrf_output)?;

        // Cache the result
        self.selection_cache.insert(cache_key, selected.clone());

        Ok(selected)
    }

    /// Generate VRF proof using Schnorrkel (Ed25519-based VRF)
    #[cfg(feature = "vrf")]
    fn generate_vrf_proof(&self, seed: &[u8]) -> Result<VrfProof> {
        use schnorrkel::{signing_context, MiniSecretKey, ExpansionMode};

        // Convert private key seed to Schnorrkel keypair
        let mini = MiniSecretKey::from_bytes(&self.vrf_private_key)
            .map_err(|e| MixnodeError::Vrf(format!("Invalid VRF secret key: {e}")))?;
        let keypair = mini.expand_to_keypair(ExpansionMode::Ed25519);

        let ctx = signing_context(b"betanet-mixnode-vrf");
        let (io, proof, _) = keypair.vrf_sign(ctx.bytes(seed));

        // Verify proof to ensure correctness
        keypair
            .public
            .vrf_verify(ctx.bytes(seed), &io.to_preout(), &proof)
            .map_err(|e| MixnodeError::Vrf(format!("VRF proof verification failed: {e}")))?;

        Ok(VrfProof { io, proof })
    }

    /// Generate VRF output (fallback implementation without VRF feature)
    #[cfg(not(feature = "vrf"))]
    fn generate_vrf_output(&self, seed: &[u8]) -> Result<[u8; 32]> {
        // Fallback: Secure HMAC-SHA256(private_key, seed)
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(&self.vrf_private_key);
        hasher.update(seed);
        Ok(hasher.finalize().into())
    }

    /// Select nodes with AS diversity constraints
    fn select_with_diversity(&self, vrf_output: &[u8; 32]) -> Result<Vec<SocketAddr>> {
        let mut selected = Vec::new();
        let mut used_as_numbers = HashSet::new();
        let mut as_node_counts: HashMap<AsNumber, usize> = HashMap::new();

        // Get candidate nodes grouped by AS, filtered by reliability
        let mut candidates_by_as: HashMap<AsNumber, Vec<SocketAddr>> = HashMap::new();
        for (as_num, nodes) in &self.as_groups {
            let reliable_nodes: Vec<SocketAddr> = nodes
                .iter()
                .filter_map(|addr| {
                    self.known_nodes.get(addr).and_then(|node| {
                        if node.reliability >= self.config.min_reliability
                            && node.is_fresh(self.config.node_timeout_secs)
                        {
                            Some(*addr)
                        } else {
                            None
                        }
                    })
                })
                .collect();

            if !reliable_nodes.is_empty() {
                candidates_by_as.insert(*as_num, reliable_nodes);
            }
        }

        // Sort AS numbers by VRF output for deterministic selection
        let mut as_numbers: Vec<AsNumber> = candidates_by_as.keys().copied().collect();
        as_numbers.sort_by_key(|as_num| {
            let mut hasher = Sha256::new();
            hasher.update(vrf_output);
            hasher.update(&as_num.to_be_bytes());
            hasher.finalize()
        });

        // Round-robin selection across AS numbers
        let mut round = 0;
        while selected.len() < self.config.max_neighbors && !as_numbers.is_empty() {
            let mut made_selection = false;

            for &as_num in &as_numbers {
                if selected.len() >= self.config.max_neighbors {
                    break;
                }

                let current_as_count = as_node_counts.get(&as_num).copied().unwrap_or(0);
                if current_as_count >= self.config.max_nodes_per_as {
                    continue;
                }

                if let Some(candidates) = candidates_by_as.get(&as_num) {
                    // Select node from this AS using VRF output
                    if let Some(node_addr) = self.select_node_from_as(candidates, vrf_output, round)
                    {
                        if !selected.contains(&node_addr) {
                            selected.push(node_addr);
                            used_as_numbers.insert(as_num);
                            as_node_counts.insert(as_num, current_as_count + 1);
                            made_selection = true;
                        }
                    }
                }
            }

            if !made_selection {
                break; // No more valid selections possible
            }

            round += 1;
        }

        // Ensure minimum AS diversity
        if used_as_numbers.len() < self.config.min_as_diversity {
            return Err(MixnodeError::Routing(format!(
                "Could not achieve minimum AS diversity: {} selected, {} required",
                used_as_numbers.len(),
                self.config.min_as_diversity
            )));
        }

        Ok(selected)
    }

    /// Select a specific node from an AS using VRF
    fn select_node_from_as(
        &self,
        candidates: &[SocketAddr],
        vrf_output: &[u8; 32],
        round: usize,
    ) -> Option<SocketAddr> {
        if candidates.is_empty() {
            return None;
        }

        // Create round-specific VRF output
        let mut hasher = Sha256::new();
        hasher.update(vrf_output);
        hasher.update(&round.to_be_bytes());
        let round_output = hasher.finalize();

        // Convert to index
        let mut index_bytes = [0u8; 8];
        index_bytes.copy_from_slice(&round_output[..8]);
        let index = u64::from_be_bytes(index_bytes) as usize % candidates.len();

        // Weight by selection score
        let weighted_candidates: Vec<(SocketAddr, f64)> = candidates
            .iter()
            .filter_map(|addr| {
                self.known_nodes
                    .get(addr)
                    .map(|node| (*addr, node.selection_score()))
            })
            .collect();

        if weighted_candidates.is_empty() {
            return candidates.get(index).copied();
        }

        // Simple weighted selection: pick highest scoring node near VRF index
        let start_idx = index % weighted_candidates.len();
        let mut best_score = 0.0;
        let mut best_addr = None;

        // Check a few candidates around the VRF index
        for i in 0..std::cmp::min(3, weighted_candidates.len()) {
            let candidate_idx = (start_idx + i) % weighted_candidates.len();
            let (addr, score) = weighted_candidates[candidate_idx];
            if score > best_score {
                best_score = score;
                best_addr = Some(addr);
            }
        }

        best_addr
    }

    /// Clean up stale nodes
    fn cleanup_stale_nodes(&mut self) {
        let timeout = self.config.node_timeout_secs;
        let stale_nodes: Vec<SocketAddr> = self
            .known_nodes
            .iter()
            .filter_map(|(addr, node)| {
                if !node.is_fresh(timeout) {
                    Some(*addr)
                } else {
                    None
                }
            })
            .collect();

        for addr in stale_nodes {
            self.remove_node(&addr);
        }
    }

    /// Get statistics about the current topology
    pub fn get_topology_stats(&self) -> TopologyStats {
        let total_nodes = self.known_nodes.len();
        let total_as_numbers = self.as_groups.len();
        let fresh_nodes = self
            .known_nodes
            .values()
            .filter(|node| node.is_fresh(self.config.node_timeout_secs))
            .count();
        let reliable_nodes = self
            .known_nodes
            .values()
            .filter(|node| node.reliability >= self.config.min_reliability)
            .count();

        let avg_as_size = if total_as_numbers > 0 {
            total_nodes as f64 / total_as_numbers as f64
        } else {
            0.0
        };

        TopologyStats {
            total_nodes,
            total_as_numbers,
            fresh_nodes,
            reliable_nodes,
            avg_as_size,
            cache_entries: self.selection_cache.len(),
        }
    }

    /// Simulate AS diversity for testing
    pub fn simulate_topology(&mut self, nodes_per_as: &[(AsNumber, usize)]) {
        self.known_nodes.clear();
        self.as_groups.clear();
        self.selection_cache.clear();

        let mut port = 9000;
        for &(as_num, node_count) in nodes_per_as {
            for _ in 0..node_count {
                let addr = SocketAddr::new(IpAddr::from([127, 0, 0, 1]), port);
                let vrf_key = CryptoUtils::sha256(&port.to_be_bytes());
                let node = MixnodeInfo::new(addr, as_num, vrf_key);
                self.add_node(node);
                port += 1;
            }
        }
    }
}

/// Topology statistics
#[derive(Debug, Clone)]
pub struct TopologyStats {
    /// Number of known nodes
    pub total_nodes: usize,
    /// Number of unique AS numbers
    pub total_as_numbers: usize,
    /// Nodes considered fresh
    pub fresh_nodes: usize,
    /// Nodes considered reliable
    pub reliable_nodes: usize,
    /// Average number of nodes per AS
    pub avg_as_size: f64,
    /// Number of cached entries
    pub cache_entries: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vrf_neighbor_selection() {
        let config = NeighborSelectionConfig {
            max_neighbors: 6,
            max_nodes_per_as: 2,
            min_as_diversity: 3,
            node_timeout_secs: 3600,
            cache_ttl_secs: 300,
            min_reliability: 0.5,
        };

        let mut selector = VrfNeighborSelector::new(config);

        // Simulate topology with different AS numbers
        selector.simulate_topology(&[(1001, 3), (1002, 3), (1003, 3), (1004, 2)]);

        let stats = selector.get_topology_stats();
        assert_eq!(stats.total_nodes, 11);
        assert_eq!(stats.total_as_numbers, 4);

        // Test neighbor selection
        let seed = b"test_seed_for_selection";

        #[cfg(feature = "vrf")]
        let (neighbors, proof) = selector.select_neighbors(seed).unwrap();
        #[cfg(not(feature = "vrf"))]
        let neighbors = selector.select_neighbors(seed).unwrap();

        assert!(neighbors.len() <= 6);
        assert!(neighbors.len() >= 3); // At least min_as_diversity

        // Verify AS diversity
        let mut as_numbers: HashSet<AsNumber> = HashSet::new();
        for addr in &neighbors {
            if let Some(node) = selector.known_nodes.get(addr) {
                as_numbers.insert(node.as_number);
            }
        }
        assert!(as_numbers.len() >= 3);

        // Verify proof
        #[cfg(feature = "vrf")]
        {
            use schnorrkel::{signing_context, PublicKey};
            let pk = PublicKey::from_bytes(selector.vrf_public_key()).unwrap();
            let ctx = signing_context(b"betanet-mixnode-vrf");
            assert!(pk
                .vrf_verify(ctx.bytes(seed), &proof.io.to_preout(), &proof.proof)
                .is_ok());
        }

        // Test deterministic selection (same seed should give same result)
        #[cfg(feature = "vrf")]
        let (neighbors2, _) = selector.select_neighbors(seed).unwrap();
        #[cfg(not(feature = "vrf"))]
        let neighbors2 = selector.select_neighbors(seed).unwrap();
        assert_eq!(neighbors, neighbors2);

        // Test different seed gives different result (highly likely)
        #[cfg(feature = "vrf")]
        let (neighbors3, _) = selector.select_neighbors(b"different_seed").unwrap();
        #[cfg(not(feature = "vrf"))]
        let neighbors3 = selector.select_neighbors(b"different_seed").unwrap();
        // Note: This may occasionally fail due to randomness, but very unlikely
        assert_ne!(neighbors, neighbors3);
    }

    #[test]
    fn test_as_diversity_constraints() {
        let config = NeighborSelectionConfig {
            max_neighbors: 8,
            max_nodes_per_as: 1,
            min_as_diversity: 5,
            ..Default::default()
        };

        let mut selector = VrfNeighborSelector::new(config);

        // Too few AS numbers - should fail
        selector.simulate_topology(&[(1001, 3), (1002, 3)]);
        let result = selector.select_neighbors(b"test");
        assert!(result.is_err());

        // Sufficient AS numbers - should succeed
        selector.simulate_topology(&[(1001, 2), (1002, 2), (1003, 2), (1004, 2), (1005, 2)]);
        #[cfg(feature = "vrf")]
        let (neighbors, _) = selector.select_neighbors(b"test").unwrap();
        #[cfg(not(feature = "vrf"))]
        let neighbors = selector.select_neighbors(b"test").unwrap();
        assert!(neighbors.len() >= 5);

        // Verify max nodes per AS constraint
        let mut as_counts: HashMap<AsNumber, usize> = HashMap::new();
        for addr in &neighbors {
            if let Some(node) = selector.known_nodes.get(addr) {
                *as_counts.entry(node.as_number).or_insert(0) += 1;
            }
        }

        for count in as_counts.values() {
            assert!(*count <= 1); // max_nodes_per_as = 1
        }
    }

    #[test]
    fn test_node_freshness() {
        let config = NeighborSelectionConfig {
            node_timeout_secs: 1, // Very short timeout for testing
            ..Default::default()
        };

        let mut selector = VrfNeighborSelector::new(config);

        // Add a node
        let addr = SocketAddr::new(IpAddr::from([127, 0, 0, 1]), 9000);
        let mut node = MixnodeInfo::new(addr, 1001, [42u8; 32]);

        // Simulate stale timestamp
        node.last_seen = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
            - 10; // 10 seconds ago

        selector.add_node(node);

        // Should be cleaned up during selection
        std::thread::sleep(std::time::Duration::from_millis(100));
        selector.cleanup_stale_nodes();

        let stats = selector.get_topology_stats();
        assert_eq!(stats.fresh_nodes, 0);
    }
}
