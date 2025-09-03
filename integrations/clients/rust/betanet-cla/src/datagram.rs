use bytes::Bytes;
use thiserror::Error;

use betanet_dtn::bundle::{Bundle, BundleError};
use betanet_htx::{Frame, FrameError, FrameType};

use crate::privacy::{apply_privacy, remove_privacy, PrivacyMode, PrivacyError};

#[derive(Debug, Error)]
pub enum DatagramError {
    #[error("datagram transport unsupported in this build")]
    Unsupported,
    #[error("transport error: {0}")]
    Transport(#[from] betanet_htx::HtxError),
    #[error("bundle error: {0}")]
    Bundle(#[from] BundleError),
    #[error("frame error: {0}")]
    Frame(#[from] FrameError),
    #[error("privacy error: {0}")]
    Privacy(#[from] PrivacyError),
}

/// QUIC DATAGRAM based CLA
pub struct BetanetDatagramCla {
    #[cfg(feature = "quic")]
    transport: betanet_htx::quic::QuicTransport,
}

impl BetanetDatagramCla {
    /// Create new DATAGRAM CLA when QUIC feature enabled
    #[cfg(feature = "quic")]
    pub fn new(transport: betanet_htx::quic::QuicTransport) -> Self {
        Self { transport }
    }

    /// Stub constructor when QUIC not compiled
    #[cfg(not(feature = "quic"))]
    pub fn new() -> Self {
        Self {}
    }

    /// Send bundle using QUIC DATAGRAMs
    #[cfg(feature = "quic")]
    pub async fn send_bundle(
        &mut self,
        bundle: &Bundle,
        mode: PrivacyMode,
    ) -> Result<(), DatagramError> {
        let bytes = bundle.encode()?;
        let wrapped = apply_privacy(bytes, mode)?;
        let frame = Frame::data(0, wrapped)?;
        self.transport.send_datagram(frame).await?;
        Ok(())
    }

    /// Receive bundle via DATAGRAM path
    #[cfg(feature = "quic")]
    pub async fn recv_bundle(
        &mut self,
        mode: PrivacyMode,
    ) -> Result<Option<Bundle>, DatagramError> {
        let Some(frame) = self.transport.recv_datagram().await? else {
            return Ok(None);
        };
        if frame.frame_type != FrameType::Data {
            return Ok(None);
        }
        let payload = remove_privacy(frame.payload, mode)?;
        let bundle = Bundle::decode(payload)?;
        Ok(Some(bundle))
    }

    /// Stubs when QUIC not compiled
    #[cfg(not(feature = "quic"))]
    pub async fn send_bundle(
        &mut self,
        _bundle: &Bundle,
        _mode: PrivacyMode,
    ) -> Result<(), DatagramError> {
        Ok(())
    }

    #[cfg(not(feature = "quic"))]
    pub async fn recv_bundle(
        &mut self,
        _mode: PrivacyMode,
    ) -> Result<Option<Bundle>, DatagramError> {
        Ok(None)
    }
}
