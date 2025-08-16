//! QUIC transport implementation with H3 ALPN and DATAGRAM support

#[cfg(feature = "quic")]
use std::net::SocketAddr;
#[cfg(feature = "quic")]
use std::path::Path;
#[cfg(feature = "quic")]
use std::sync::Arc;

#[cfg(feature = "quic")]
use quinn::{ClientConfig, Endpoint, ServerConfig, Connection, SendDatagramError};
#[cfg(feature = "quic")]
use rustls::pki_types::{CertificateDer, PrivatePkcs8KeyDer};
#[cfg(feature = "quic")]
use tracing::{debug, error, info, warn};
#[cfg(feature = "quic")]
use bytes::Bytes;

#[cfg(feature = "quic")]
use crate::{transport::TransportStats, HtxConfig, HtxError, Result, Frame};

#[cfg(feature = "quic")]
use base64;
#[cfg(feature = "quic")]
use trust_dns_resolver::{TokioAsyncResolver, config::{ResolverConfig, ResolverOpts}};

/// QUIC transport with DATAGRAM support
#[cfg(feature = "quic")]
pub struct QuicTransport {
    connection: quinn::Connection,
    stats: TransportStats,
    enable_datagrams: bool,
    ech_config: Option<EchConfig>,
}

/// ECH (Encrypted Client Hello) configuration
#[cfg(feature = "quic")]
#[derive(Debug, Clone)]
pub struct EchConfig {
    pub public_name: String,
    pub config_id: u8,
    pub kem_id: u16,
    pub public_key: Vec<u8>,
    pub cipher_suites: Vec<u16>,
    pub maximum_name_length: u8,
    pub extensions: Vec<u8>,
}

impl Default for EchConfig {
    fn default() -> Self {
        Self {
            public_name: "cloudflare.com".to_string(), // Common fronting domain
            config_id: 0,
            kem_id: 0x0020, // X25519
            public_key: vec![0u8; 32], // Placeholder key
            cipher_suites: vec![0x1301, 0x1302, 0x1303], // TLS 1.3 cipher suites
            maximum_name_length: 64,
            extensions: vec![],
        }
    }
}

#[cfg(feature = "quic")]
impl EchConfig {
    fn from_bytes(data: &[u8], public_name: String) -> Result<Self> {
        let mut idx = 0;
        if data.len() < 7 {
            return Err(HtxError::Config("ECH config too short".into()));
        }
        let config_id = data[idx];
        idx += 1;
        let kem_id = u16::from_be_bytes([data[idx], data[idx + 1]]);
        idx += 2;
        let pk_len = u16::from_be_bytes([data[idx], data[idx + 1]]) as usize;
        idx += 2;
        if data.len() < idx + pk_len {
            return Err(HtxError::Config("Invalid ECH public key length".into()));
        }
        let public_key = data[idx..idx + pk_len].to_vec();
        idx += pk_len;
        if data.len() < idx + 2 {
            return Err(HtxError::Config("Missing cipher suites length".into()));
        }
        let suites_len = u16::from_be_bytes([data[idx], data[idx + 1]]) as usize;
        idx += 2;
        if data.len() < idx + suites_len || suites_len % 2 != 0 {
            return Err(HtxError::Config("Invalid cipher suites".into()));
        }
        let mut cipher_suites = Vec::new();
        for chunk in data[idx..idx + suites_len].chunks(2) {
            cipher_suites.push(u16::from_be_bytes([chunk[0], chunk[1]]));
        }
        idx += suites_len;
        if data.len() < idx + 2 {
            return Err(HtxError::Config("Missing maximum name length".into()));
        }
        let maximum_name_length = u16::from_be_bytes([data[idx], data[idx + 1]]) as u8;
        idx += 2;
        if data.len() < idx + 2 {
            return Err(HtxError::Config("Missing extensions length".into()));
        }
        let ext_len = u16::from_be_bytes([data[idx], data[idx + 1]]) as usize;
        idx += 2;
        if data.len() < idx + ext_len {
            return Err(HtxError::Config("Invalid extensions".into()));
        }
        let extensions = data[idx..idx + ext_len].to_vec();

        let cfg = Self {
            public_name,
            config_id,
            kem_id,
            public_key,
            cipher_suites,
            maximum_name_length,
            extensions,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn from_file<P: AsRef<Path>>(path: P, public_name: String) -> Result<Self> {
        let contents = std::fs::read_to_string(path)
            .map_err(|e| HtxError::Config(format!("failed to read ECH config file: {}", e)))?;
        let data = base64::decode(contents.trim())
            .map_err(|e| HtxError::Config(format!("invalid ECH base64: {}", e)))?;
        Self::from_bytes(&data, public_name)
    }

    pub async fn from_dns(domain: &str) -> Result<Self> {
        let resolver = TokioAsyncResolver::tokio(ResolverConfig::default(), ResolverOpts::default())
            .map_err(|e| HtxError::Config(format!("DNS resolver init failed: {}", e)))?;
        let name = format!("_echconfig.{}", domain);
        let response = resolver
            .txt_lookup(name)
            .await
            .map_err(|e| HtxError::Config(format!("ECH DNS lookup failed: {}", e)))?;
        let record = response
            .iter()
            .flat_map(|txt| txt.txt_data().iter())
            .map(|d| String::from_utf8_lossy(d).into_owned())
            .next()
            .ok_or_else(|| HtxError::Config("ECH DNS record not found".to_string()))?;
        let data = base64::decode(record.trim())
            .map_err(|e| HtxError::Config(format!("invalid ECH base64: {}", e)))?;
        Self::from_bytes(&data, domain.to_string())
    }

    pub fn validate(&self) -> Result<()> {
        if self.public_name.is_empty() {
            return Err(HtxError::Config("ECH public name missing".into()));
        }
        if self.public_key.is_empty() {
            return Err(HtxError::Config("ECH public key missing".into()));
        }
        if self.cipher_suites.is_empty() {
            return Err(HtxError::Config("ECH cipher suites missing".into()));
        }
        Ok(())
    }
}

/// QUIC DATAGRAM frame for small message transport
#[cfg(feature = "quic")]
#[derive(Debug, Clone)]
pub struct QuicDatagramFrame {
    pub stream_id: u32,
    pub payload: Bytes,
}

#[cfg(feature = "quic")]
impl QuicTransport {
    /// Connect to a remote QUIC endpoint with H3 ALPN and DATAGRAM support
    pub async fn connect(addr: SocketAddr, config: &HtxConfig) -> Result<Self> {
        info!("Connecting to QUIC endpoint at {} with H3 ALPN", addr);

        // Generate self-signed certificate for testing
        let cert = rcgen::generate_simple_self_signed(vec!["localhost".to_string()])
            .map_err(|e| HtxError::Crypto(format!("Failed to generate certificate: {}", e)))?;

        let cert_der = CertificateDer::from(cert.cert);
        let key_der = PrivatePkcs8KeyDer::from(cert.key_pair.serialize_der());

        let mut roots = rustls::RootCertStore::empty();
        roots.add(cert_der.clone()).map_err(|e| {
            HtxError::Crypto(format!("Failed to add root certificate: {}", e))
        })?;

        let mut client_config = ClientConfig::with_root_certificates(Arc::new(roots))
            .map_err(|e| HtxError::Transport(format!("Failed to create client config: {}", e)))?;

        // Configure H3 ALPN protocols
        client_config.alpn_protocols = vec![
            b"h3".to_vec(),      // HTTP/3
            b"h3-32".to_vec(),   // HTTP/3 draft-32
            b"h3-29".to_vec(),   // HTTP/3 draft-29
            b"htx/1.1".to_vec(), // HTX protocol
        ];

        let mut endpoint = Endpoint::client("0.0.0.0:0".parse().unwrap())
            .map_err(|e| HtxError::Transport(format!("Failed to create endpoint: {}", e)))?;

        endpoint.set_default_client_config(client_config);

        let connection = endpoint
            .connect(addr, "localhost")
            .map_err(|e| HtxError::Transport(format!("Failed to connect: {}", e)))?
            .await
            .map_err(|e| HtxError::Transport(format!("Connection failed: {}", e)))?;

        // Check DATAGRAM support
        let enable_datagrams = connection.max_datagram_size().is_some();
        if enable_datagrams {
            info!("QUIC DATAGRAM support enabled, max size: {:?}", connection.max_datagram_size());
        } else {
            warn!("QUIC DATAGRAM not supported by peer");
        }

        // Configure ECH if requested
        let ech_config = if config.enable_tls_camouflage {
            let ech = if let Some(ref path) = config.ech_config_path {
                EchConfig::from_file(path, config.camouflage_domain.clone().unwrap_or_default())
                    .map_err(|e| HtxError::Config(format!("Failed to load ECH config: {}", e)))?
            } else if let Some(ref domain) = config.camouflage_domain {
                EchConfig::from_dns(domain)
                    .await
                    .map_err(|e| HtxError::Config(format!("Failed to load ECH config: {}", e)))?
            } else {
                return Err(HtxError::Config(
                    "ECH requested but no configuration source provided".to_string(),
                ));
            };
            Some(ech)
        } else {
            None
        };

        if let Some(ref ech) = ech_config {
            info!("ECH configured with public name: {}", ech.public_name);
        }

        Ok(Self {
            connection,
            stats: TransportStats::new(),
            enable_datagrams,
            ech_config,
        })
    }

    /// Send HTX frame as QUIC DATAGRAM (for small frames)
    pub async fn send_datagram(&mut self, frame: Frame) -> Result<()> {
        if !self.enable_datagrams {
            return Err(HtxError::Transport("DATAGRAM not supported".to_string()));
        }

        let encoded_frame = frame.encode();

        // Check if frame fits in DATAGRAM
        if let Some(max_size) = self.connection.max_datagram_size() {
            if encoded_frame.len() > max_size {
                return Err(HtxError::Transport(format!(
                    "Frame too large for DATAGRAM: {} > {}",
                    encoded_frame.len(),
                    max_size
                )));
            }
        }

        match self.connection.send_datagram(encoded_frame.clone()) {
            Ok(()) => {
                self.stats.record_sent(encoded_frame.len());
                debug!("Sent {} bytes via QUIC DATAGRAM", encoded_frame.len());
                Ok(())
            }
            Err(SendDatagramError::UnsupportedByPeer) => {
                Err(HtxError::Transport("DATAGRAM unsupported by peer".to_string()))
            }
            Err(SendDatagramError::Disabled) => {
                Err(HtxError::Transport("DATAGRAM disabled".to_string()))
            }
            Err(SendDatagramError::TooLarge) => {
                Err(HtxError::Transport("DATAGRAM too large".to_string()))
            }
            Err(SendDatagramError::ConnectionLost(e)) => {
                Err(HtxError::Transport(format!("Connection lost: {}", e)))
            }
        }
    }

    /// Receive HTX frames from QUIC DATAGRAMs
    pub async fn recv_datagram(&mut self) -> Result<Option<Frame>> {
        if !self.enable_datagrams {
            return Ok(None);
        }

        match self.connection.read_datagram().await {
            Ok(data) => {
                self.stats.record_received(data.len());

                match crate::parse_frame(&data) {
                    Ok((frame, _consumed)) => {
                        debug!("Received {} bytes via QUIC DATAGRAM", data.len());
                        Ok(Some(frame))
                    }
                    Err(e) => {
                        warn!("Failed to parse DATAGRAM as HTX frame: {}", e);
                        Ok(None)
                    }
                }
            }
            Err(e) => {
                debug!("DATAGRAM read error: {}", e);
                Ok(None)
            }
        }
    }

    /// Get ECH configuration for TLS camouflage
    pub fn ech_config(&self) -> Option<&EchConfig> {
        self.ech_config.as_ref()
    }

    /// Check if DATAGRAM support is enabled
    pub fn has_datagram_support(&self) -> bool {
        self.enable_datagrams
    }

    /// Get maximum DATAGRAM size
    pub fn max_datagram_size(&self) -> Option<usize> {
        self.connection.max_datagram_size()
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
