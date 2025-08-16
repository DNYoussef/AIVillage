//! HTX server implementation

use std::net::SocketAddr;
use std::sync::Arc;

use tokio::sync::{mpsc, RwLock};
use tracing::{debug, error, info};

use crate::{HtxConfig, HtxConnection, HtxError, Result};

/// HTX server
pub struct HtxServer {
    config: HtxConfig,
    shutdown_tx: Option<mpsc::Sender<()>>,
}

impl HtxServer {
    /// Create a new HTX server
    pub fn new(config: HtxConfig) -> Self {
        Self {
            config,
            shutdown_tx: None,
        }
    }

    /// Start the server
    pub async fn start<F>(&mut self, handler: F) -> Result<()>
    where
        F: Fn(Box<dyn HtxConnection>) + Send + Sync + 'static + Clone,
    {
        info!("Starting HTX server on {}", self.config.listen_addr);

        let (shutdown_tx, mut shutdown_rx) = mpsc::channel(1);
        self.shutdown_tx = Some(shutdown_tx);

        #[cfg(feature = "tcp")]
        if self.config.enable_tcp {
            let config = self.config.clone();
            let handler = handler.clone();
            tokio::spawn(async move {
                if let Err(e) = crate::tcp::TcpTransport::listen(config, handler).await {
                    error!("TCP listener error: {}", e);
                }
            });
        }

        #[cfg(feature = "quic")]
        if self.config.enable_quic {
            let config = self.config.clone();
            tokio::spawn(async move {
                if let Err(e) = crate::quic::QuicTransport::listen(config, handler).await {
                    error!("QUIC listener error: {}", e);
                }
            });
        }

        // Wait for shutdown signal
        let _ = shutdown_rx.recv().await;
        info!("HTX server shutting down");

        Ok(())
    }

    /// Stop the server
    pub async fn stop(&mut self) -> Result<()> {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(()).await;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_creation() {
        let config = HtxConfig::default();
        let server = HtxServer::new(config);
        assert!(server.shutdown_tx.is_none());
    }
}
