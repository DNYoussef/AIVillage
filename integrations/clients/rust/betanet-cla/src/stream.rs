use betanet_dtn::bundle::Bundle;
use thiserror::Error;

use crate::privacy::{apply_privacy, remove_privacy, PrivacyMode, PrivacyError};

#[derive(Debug, Error)]
pub enum StreamError {
    #[error("stream not implemented")] 
    NotImplemented,
    #[error("privacy error: {0}")]
    Privacy(#[from] PrivacyError),
    #[error("bundle error: {0}")]
    Bundle(#[from] betanet_dtn::bundle::BundleError),
}

/// Minimal placeholder stream CLA
pub struct BetanetStreamCla;

impl BetanetStreamCla {
    pub fn new() -> Self {
        Self
    }

    /// Send bundle over stream path (currently no-op)
    pub async fn send_bundle(
        &mut self,
        _bundle: &Bundle,
        _mode: PrivacyMode,
    ) -> Result<(), StreamError> {
        Ok(())
    }

    /// Receive bundle from stream path (currently none)
    pub async fn recv_bundle(
        &mut self,
        _mode: PrivacyMode,
    ) -> Result<Option<Bundle>, StreamError> {
        Ok(None)
    }
}
