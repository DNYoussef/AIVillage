//! Betanet HTX Convergence Layer Adapter for DTN
//!
//! Implements Betanet HTX transport as a DTN convergence layer with:
//! - Stream mode for large bundles over TCP
//! - QUIC DATAGRAM mode for small bundles
//! - Optional Sphinx wrapping for privacy="strict" mode

#![deny(clippy::all)]
#![allow(missing_docs)]
#![allow(dead_code)]

// Module structure (to be implemented)
pub mod stream;
pub mod datagram;
pub mod privacy;

// Re-exports
pub use stream::BetanetStreamCla;

/// Placeholder implementation - will be expanded based on requirements
pub struct BetanetCla;

impl BetanetCla {
    pub fn new() -> Self {
        Self
    }
}

impl Default for BetanetCla {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_placeholder() {
        let _cla = BetanetCla::new();
    }
}
