//! QUIC transport implementation

#[cfg(feature = "quic")]
use std::net::SocketAddr;
#[cfg(feature = "quic")]
use std::sync::Arc;

#[cfg(feature = "quic")]
use quinn::{ClientConfig, Endpoint, ServerConfig};
#[cfg(feature = "quic")]
use rustls::pki_types::{CertificateDer, PrivatePkcs8KeyDer};
#[cfg(feature = "quic")]
use tracing::{debug, error, info};

#[cfg(feature = "quic")]
use crate::{transport::TransportStats, HtxConfig, HtxConnection, HtxError, Result};

/// QUIC transport
#[cfg(feature = "quic")]
pub struct QuicTransport {
    connection: quinn::Connection,
    stats: TransportStats,
}

#[cfg(feature = "quic")]
impl QuicTransport {
    /// Connect to a remote QUIC endpoint
    pub async fn connect(addr: SocketAddr, _config: &HtxConfig) -> Result<Self> {
        info!("Connecting to QUIC endpoint at {}", addr);

        // Generate self-signed certificate for testing
        let cert = rcgen::generate_simple_self_signed(vec!["localhost".to_string()])
            .map_err(|e| HtxError::Crypto(format!("Failed to generate certificate: {}", e)))?;

        let cert_der = CertificateDer::from(cert.cert);
        let key_der = PrivatePkcs8KeyDer::from(cert.key_pair.serialize_der());

        let mut roots = rustls::RootCertStore::empty();
        roots.add(cert_der.clone()).map_err(|e| {
            HtxError::Crypto(format!("Failed to add root certificate: {}", e))
        })?;

        let client_config = ClientConfig::with_root_certificates(Arc::new(roots))
            .map_err(|e| HtxError::Transport(format!("Failed to create client config: {}", e)))?;

        let mut endpoint = Endpoint::client("0.0.0.0:0".parse().unwrap())
            .map_err(|e| HtxError::Transport(format!("Failed to create endpoint: {}", e)))?;

        endpoint.set_default_client_config(client_config);

        let connection = endpoint
            .connect(addr, "localhost")
            .map_err(|e| HtxError::Transport(format!("Failed to connect: {}", e)))?
            .await
            .map_err(|e| HtxError::Transport(format!("Connection failed: {}", e)))?;

        Ok(Self {
            connection,
            stats: TransportStats::new(),
        })
    }

    /// Listen for incoming QUIC connections
    pub async fn listen<F>(config: HtxConfig, handler: F) -> Result<()>
    where
        F: Fn(Box<dyn HtxConnection>) + Send + Sync + 'static,
    {
        info!("Starting QUIC listener on {}", config.listen_addr);

        // Generate self-signed certificate for testing
        let cert = rcgen::generate_simple_self_signed(vec!["localhost".to_string()])
            .map_err(|e| HtxError::Crypto(format!("Failed to generate certificate: {}", e)))?;

        let cert_der = CertificateDer::from(cert.cert);
        let key_der = PrivatePkcs8KeyDer::from(cert.key_pair.serialize_der());

        let server_config = ServerConfig::with_single_cert(vec![cert_der], key_der.into())
            .map_err(|e| HtxError::Transport(format!("Failed to create server config: {}", e)))?;

        let endpoint = Endpoint::server(server_config, config.listen_addr)
            .map_err(|e| HtxError::Transport(format!("Failed to create endpoint: {}", e)))?;

        while let Some(connecting) = endpoint.accept().await {
            let handler = handler.clone();

            tokio::spawn(async move {
                match connecting.await {
                    Ok(connection) => {
                        debug!("Accepted QUIC connection from {}", connection.remote_address());

                        let transport = QuicTransport {
                            connection,
                            stats: TransportStats::new(),
                        };

                        handler(Box::new(transport));
                    }
                    Err(e) => {
                        error!("Failed to accept QUIC connection: {}", e);
                    }
                }
            });
        }

        Ok(())
    }
}

#[cfg(feature = "quic")]
#[async_trait::async_trait]
impl HtxConnection for QuicTransport {
    async fn send(&mut self, data: &[u8]) -> Result<()> {
        let mut stream = self
            .connection
            .open_uni()
            .await
            .map_err(|e| HtxError::Transport(format!("Failed to open stream: {}", e)))?;

        stream
            .write_all(data)
            .await
            .map_err(|e| HtxError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;

        stream
            .finish()
            .map_err(|e| HtxError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;

        self.stats.record_sent(data.len());
        Ok(())
    }

    async fn recv(&mut self, buf: &mut [u8]) -> Result<usize> {
        let mut stream = self
            .connection
            .accept_uni()
            .await
            .map_err(|e| HtxError::Transport(format!("Failed to accept stream: {}", e)))?;

        let n = stream
            .read(buf)
            .await
            .map_err(|e| HtxError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?
            .unwrap_or(0);

        self.stats.record_received(n);
        Ok(n)
    }

    fn remote_addr(&self) -> Result<SocketAddr> {
        Ok(self.connection.remote_address())
    }

    async fn close(self) -> Result<()> {
        debug!("Closing QUIC connection");
        self.connection.close(0u32.into(), b"closing");
        Ok(())
    }
}

// Stub implementation when QUIC feature is disabled
#[cfg(not(feature = "quic"))]
pub struct QuicTransport;

#[cfg(not(feature = "quic"))]
impl QuicTransport {
    pub async fn connect(_addr: std::net::SocketAddr, _config: &crate::HtxConfig) -> crate::Result<Self> {
        Err(crate::HtxError::Config("QUIC feature not enabled".to_string()))
    }

    pub async fn listen<F>(_config: crate::HtxConfig, _handler: F) -> crate::Result<()>
    where
        F: Fn(Box<dyn crate::HtxConnection>) + Send + Sync + 'static,
    {
        Err(crate::HtxError::Config("QUIC feature not enabled".to_string()))
    }
}
