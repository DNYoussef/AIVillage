use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use sha2::{Digest, Sha256};
use std::net::SocketAddr;

/// Select next hop using a VRF-like hash of the seed.
pub fn select_next_hop(seed: &[u8], hops: &[SocketAddr]) -> Option<SocketAddr> {
    if hops.is_empty() {
        return None;
    }
    let hash = Sha256::digest(seed);
    let mut rng = StdRng::from_seed(hash.into());
    let idx = rng.gen_range(0..hops.len());
    Some(hops[idx])
}
