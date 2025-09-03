//! Betanet HTX Convergence Layer Adapter for DTN
//!
//! Implements Betanet HTX transport as a DTN convergence layer with:
//! - Stream mode for large bundles over TCP
//! - QUIC DATAGRAM mode for small bundles
//! - Optional Sphinx wrapping for privacy="strict" mode

#![deny(clippy::all)]
#![allow(missing_docs)]
#![allow(dead_code)]

pub mod datagram;
pub mod privacy;
pub mod stream;

pub use datagram::{BetanetDatagramCla, DatagramError};
pub use privacy::{PrivacyMode, PrivacyError};
pub use stream::{BetanetStreamCla, StreamError};

/// High-level Betanet CLA combining stream and datagram modes
pub struct BetanetCla {
    stream: BetanetStreamCla,
    datagram: BetanetDatagramCla,
    privacy: PrivacyMode,
    /// Bundles under this size (bytes) use datagrams
    datagram_limit: usize,
}

impl BetanetCla {
    pub fn new(stream: BetanetStreamCla, datagram: BetanetDatagramCla, privacy: PrivacyMode) -> Self {
        Self { stream, datagram, privacy, datagram_limit: if cfg!(feature="quic") { 1024 } else { 0 } }
    }

    /// Send bundle selecting transport path based on size
    pub async fn send(&mut self, bundle: &betanet_dtn::bundle::Bundle) -> Result<(), BetanetClaError> {
        if bundle.size() <= self.datagram_limit {
            self.datagram.send_bundle(bundle, self.privacy).await?
        } else {
            self.stream.send_bundle(bundle, self.privacy).await?
        };
        Ok(())
    }

    /// Receive bundle from either path
    pub async fn recv(&mut self) -> Result<Option<betanet_dtn::bundle::Bundle>, BetanetClaError> {
        if let Some(b) = self.datagram.recv_bundle(self.privacy).await? {
            return Ok(Some(b));
        }
        self.stream.recv_bundle(self.privacy).await.map_err(BetanetClaError::from)
    }
}

impl Default for BetanetCla {
    fn default() -> Self {
        Self::new(BetanetStreamCla::new(), BetanetDatagramCla::new(), PrivacyMode::Standard)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum BetanetClaError {
    #[error(transparent)]
    Datagram(#[from] datagram::DatagramError),
    #[error(transparent)]
    Stream(#[from] stream::StreamError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use betanet_dtn::bundle::{Bundle, EndpointId, PrimaryBlock, PayloadBlock};
    use bytes::Bytes;

    #[tokio::test]
    async fn test_send_recv_paths() {
        let primary = PrimaryBlock::new(EndpointId::null(), EndpointId::null(), 1000);
        let payload = PayloadBlock::new(Bytes::from_static(b"hi"));
        let bundle = Bundle::new(primary, vec![], payload);

        let mut cla = BetanetCla::default();
        cla.send(&bundle).await.unwrap();
        let _ = cla.recv().await.unwrap();
    }
}
