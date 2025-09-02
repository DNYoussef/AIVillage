//! Stream mode convergence layer for large bundles over TCP

use std::net::SocketAddr;

/// Betanet Stream Convergence Layer Adapter
pub struct BetanetStreamCla {
    local_addr: SocketAddr,
    peer_addr: Option<SocketAddr>,
}

impl BetanetStreamCla {
    /// Create new stream CLA instance
    pub fn new(local_addr: SocketAddr) -> Self {
        Self {
            local_addr,
            peer_addr: None,
        }
    }
    
    /// Connect to peer
    pub fn connect(&mut self, peer_addr: SocketAddr) -> Result<(), Box<dyn std::error::Error>> {
        self.peer_addr = Some(peer_addr);
        Ok(())
    }
    
    /// Get local address
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }
    
    /// Get peer address if connected
    pub fn peer_addr(&self) -> Option<SocketAddr> {
        self.peer_addr
    }
}

impl Default for BetanetStreamCla {
    fn default() -> Self {
        Self::new("127.0.0.1:0".parse().unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_cla_creation() {
        let addr = "127.0.0.1:8080".parse().unwrap();
        let cla = BetanetStreamCla::new(addr);
        assert_eq!(cla.local_addr(), addr);
    }
    
    #[test]
    fn test_stream_cla_connection() {
        let local_addr = "127.0.0.1:8080".parse().unwrap();
        let peer_addr = "127.0.0.1:8081".parse().unwrap();
        
        let mut cla = BetanetStreamCla::new(local_addr);
        assert!(cla.connect(peer_addr).is_ok());
        assert_eq!(cla.peer_addr(), Some(peer_addr));
    }
}