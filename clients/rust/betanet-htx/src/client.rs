//! HTX client implementation

use std::net::SocketAddr;
use std::sync::Arc;

use tokio::sync::RwLock;
use tracing::{debug, error, info};

use crate::{HtxConfig, HtxConnection, HtxError, Result};

/// HTX client
pub struct HtxClient {
    config: HtxConfig,
    connection: Option<Arc<RwLock<Box<dyn HtxConnection>>>>,
}

impl HtxClient {
    /// Create a new HTX client
    pub fn new(config: HtxConfig) -> Self {
        Self {
            config,
            connection: None,
        }
    }

    /// Connect to an HTX server
    pub async fn connect(&mut self, addr: SocketAddr) -> Result<()> {
        info!("Connecting to HTX server at {}", addr);

        #[cfg(feature = "tcp")]
        if self.config.enable_tcp {
            let conn = crate::tcp::TcpTransport::connect(addr, &self.config).await?;
            self.connection = Some(Arc::new(RwLock::new(Box::new(conn))));
            return Ok(());
        }

        #[cfg(feature = "quic")]
        if self.config.enable_quic {
            let conn = crate::quic::QuicTransport::connect(addr, &self.config, None).await?;
            self.connection = Some(Arc::new(RwLock::new(Box::new(conn))));
            return Ok(());
        }

        Err(HtxError::Config("No transport enabled".to_string()))
    }

    /// Send data to the server
    pub async fn send(&self, data: &[u8]) -> Result<()> {
        let conn = self
            .connection
            .as_ref()
            .ok_or_else(|| HtxError::Protocol("Not connected".to_string()))?;

        let mut conn = conn.write().await;
        conn.send(data).await
    }

    /// Receive data from the server
    pub async fn recv(&self, buf: &mut [u8]) -> Result<usize> {
        let conn = self
            .connection
            .as_ref()
            .ok_or_else(|| HtxError::Protocol("Not connected".to_string()))?;

        let mut conn = conn.write().await;
        conn.recv(buf).await
    }

    /// Disconnect from the server
    pub async fn disconnect(mut self) -> Result<()> {
        if let Some(conn) = self.connection.take() {
            let conn = Arc::try_unwrap(conn)
                .map_err(|_| HtxError::Protocol("Connection still in use".to_string()))?;
            let conn = conn.into_inner();
            conn.close().await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_client_creation() {
        let config = HtxConfig::default();
        let client = HtxClient::new(config);
        assert!(client.connection.is_none());
    }
}
