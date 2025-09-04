//! Gossip Protocol for Federated Learning
//!
//! Enables peer discovery and communication over BitChat BLE mesh with
//! robust aggregation using trimmed mean and Krum algorithms.

use crate::{FederatedError, ParticipantId, Result};
use std::collections::HashMap;

/// Gossip protocol maintains a local view of known peers.
#[derive(Default)]
pub struct GossipProtocol {
    /// Map of peer identifier string to participant information
    peers: HashMap<String, ParticipantId>,
}

impl GossipProtocol {
    /// Create a new gossip protocol instance with no peers
    pub fn new() -> Self {
        Self {
            peers: HashMap::new(),
        }
    }

    /// Add a peer to the local view
    pub fn add_peer(&mut self, peer: ParticipantId) {
        self.peers.insert(peer.agent_id.to_string(), peer);
    }

    /// Remove a peer from the local view
    pub fn remove_peer(&mut self, agent_id: &str) -> Option<ParticipantId> {
        self.peers.remove(agent_id)
    }

    /// Return a list of currently known peers
    pub fn peers(&self) -> Vec<ParticipantId> {
        self.peers.values().cloned().collect()
    }
}

/// Peer exchange mechanism that merges remote peer lists into the local view.
pub struct PeerExchange;

impl PeerExchange {
    /// Merge remote peers into the local protocol state.  Existing entries are
    /// updated while new peers are inserted.
    pub fn exchange(local: &mut GossipProtocol, remote: &[ParticipantId]) {
        for peer in remote {
            local.add_peer(peer.clone());
        }
    }
}

/// Robust aggregation algorithms (trimmed mean, Krum)
pub struct RobustAggregation;

impl RobustAggregation {
    /// Create a new aggregator helper
    pub fn new() -> Self {
        Self {}
    }

    /// Compute element-wise trimmed mean removing a proportion of extreme
    /// values from both ends before averaging.
    pub fn trimmed_mean(updates: &[Vec<f32>], trim_ratio: f32) -> Result<Vec<f32>> {
        if updates.is_empty() {
            return Err(FederatedError::AggregationError("no updates".into()));
        }
        let n = updates.len();
        let dim = updates[0].len();
        let k = ((n as f32) * trim_ratio).floor() as usize;
        if 2 * k >= n {
            return Err(FederatedError::AggregationError(
                "trim ratio removes all elements".into(),
            ));
        }

        let mut result = vec![0.0; dim];
        for j in 0..dim {
            let mut vals: Vec<f32> = updates.iter().map(|u| u[j]).collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let slice = &vals[k..(n - k)];
            let mean = slice.iter().sum::<f32>() / slice.len() as f32;
            result[j] = mean;
        }
        Ok(result)
    }

    /// Krum aggregation selects the update that is closest to its neighbours in
    /// Euclidean distance, providing Byzantine robustness.
    pub fn krum(updates: &[Vec<f32>]) -> Result<Vec<f32>> {
        if updates.is_empty() {
            return Err(FederatedError::AggregationError("no updates".into()));
        }
        let n = updates.len();
        let dim = updates[0].len();
        // Number of Byzantine faults we can tolerate
        let f = ((n as isize - 2) / 2).max(0) as usize;
        let m = n - f - 2; // number of closest neighbours considered

        let mut scores: Vec<(f32, usize)> = Vec::new();
        for (i, ui) in updates.iter().enumerate() {
            let mut dists = Vec::new();
            for (j, uj) in updates.iter().enumerate() {
                if i == j {
                    continue;
                }
                let dist: f32 = (0..dim)
                    .map(|k| ui[k] - uj[k])
                    .map(|x| x * x)
                    .sum();
                dists.push(dist);
            }
            dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let score: f32 = dists.iter().take(m).sum();
            scores.push((score, i));
        }

        scores.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        Ok(updates[scores[0].1].clone())
    }
}
