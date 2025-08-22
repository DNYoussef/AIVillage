//! QUIC transport implementation with H3 ALPN and DATAGRAM support

#[cfg(feature = "quic")]
use std::net::SocketAddr;
#[cfg(feature = "quic")]
use std::path::Path;
#[cfg(feature = "quic")]
use std::sync::Arc;

#[cfg(feature = "quic")]
use bytes::Bytes;
#[cfg(feature = "quic")]
use quinn::{ClientConfig, Connection, Endpoint, SendDatagramError, ServerConfig, TransportConfig};
#[cfg(feature = "quic")]
use rcgen::generate_simple_self_signed;
#[cfg(feature = "quic")]
use rustls::client::danger::ServerCertVerifier;
#[cfg(feature = "quic")]
use rustls::pki_types::{CertificateDer, PrivatePkcs8KeyDer, ServerName, UnixTime};
#[cfg(feature = "quic")]
use rustls::{OwnedTrustAnchor, RootCertStore};
#[cfg(feature = "quic")]
use tracing::{debug, error, info, warn};
#[cfg(feature = "quic")]
use webpki_roots::TLS_SERVER_ROOTS;

#[cfg(feature = "quic")]
use crate::{transport::TransportStats, Frame, HtxConfig, HtxError, Result};

#[cfg(feature = "quic")]
use base64::Engine;
#[cfg(feature = "quic")]
use dashmap::DashMap;
#[cfg(feature = "quic")]
use once_cell::sync::Lazy;
#[cfg(all(feature = "quic", test))]
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(feature = "quic")]
use std::time::{Duration, Instant};
#[cfg(feature = "quic")]
use trust_dns_resolver::TokioAsyncResolver;

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
    // The ECH config encodes `maximum_name_length` as a 16-bit value.
    // We currently restrict it to 255 and store as `u8` until wider
    // support is needed.
    pub maximum_name_length: u8,
    pub extensions: Vec<u8>,
}

impl Default for EchConfig {
    fn default() -> Self {
        Self {
            public_name: "cloudflare.com".to_string(), // Common fronting domain
            config_id: 0,
            kem_id: 0x0020,                              // X25519
            public_key: vec![0u8; 32],                   // Placeholder key
            cipher_suites: vec![0x1301, 0x1302, 0x1303], // TLS 1.3 cipher suites
            maximum_name_length: 64,
            extensions: vec![],
        }
    }
}

#[cfg(feature = "quic")]
static DNS_ECH_CACHE: Lazy<DashMap<String, (Instant, EchConfig)>> = Lazy::new(|| DashMap::new());
#[cfg(feature = "quic")]
const DNS_ECH_TTL: Duration = Duration::from_secs(3600);

#[cfg(all(feature = "quic", test))]
static DNS_RESOLVER_INIT_COUNT: AtomicUsize = AtomicUsize::new(0);

#[cfg(feature = "quic")]
static DNS_RESOLVER: Lazy<TokioAsyncResolver> = Lazy::new(|| {
    #[cfg(test)]
    DNS_RESOLVER_INIT_COUNT.fetch_add(1, Ordering::Relaxed);
    TokioAsyncResolver::tokio_from_system_conf().expect("failed to create DNS resolver")
});

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
        let maximum_name_length = {
            let len = u16::from_be_bytes([data[idx], data[idx + 1]]);
            if len > u8::MAX as u16 {
                return Err(HtxError::Config(
                    "ECH maximum_name_length exceeds 255".into(),
                ));
            }
            idx += 2;
            len as u8
        };
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
        let data = base64::engine::general_purpose::STANDARD
            .decode(contents.trim())
            .map_err(|e| HtxError::Config(format!("invalid ECH base64: {}", e)))?;
        Self::from_bytes(&data, public_name)
    }

    pub async fn from_dns(domain: &str) -> Result<Self> {
        if let Some(entry) = DNS_ECH_CACHE.get(domain) {
            if entry.value().0.elapsed() < DNS_ECH_TTL {
                return Ok(entry.value().1.clone());
            } else {
                DNS_ECH_CACHE.remove(domain);
            }
        }

        let lookup_name = format!("_echconfig.{}", domain);
        let response = DNS_RESOLVER
            .txt_lookup(lookup_name.clone())
            .await
            .map_err(|e| {
                HtxError::Config(format!("ECH TXT lookup failed for {}: {}", lookup_name, e))
            })?;

        let mut txt = String::new();
        for record in response.iter() {
            for data in record.txt_data().iter() {
                if let Ok(part) = std::str::from_utf8(data) {
                    txt.push_str(part);
                }
            }
        }

        if txt.trim().is_empty() {
            return Err(HtxError::Config("ECH TXT record empty".into()));
        }

        let decoded = base64::engine::general_purpose::STANDARD
            .decode(txt.trim())
            .map_err(|e| HtxError::Config(format!("invalid ECH base64: {}", e)))?;

        let cfg = Self::from_bytes(&decoded, domain.to_string())?;
        cfg.validate()?;
        DNS_ECH_CACHE.insert(domain.to_string(), (Instant::now(), cfg.clone()));
        Ok(cfg)
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
    pub async fn connect(
        addr: SocketAddr,
        config: &HtxConfig,
        verifier: Option<Arc<dyn ServerCertVerifier>>,
    ) -> Result<Self> {
        // Build rustls client config with system/root certificates or a custom verifier
        let mut root_store = RootCertStore::empty();
        root_store.add_trust_anchors(TLS_SERVER_ROOTS.iter().map(|ta| {
            OwnedTrustAnchor::from_subject_spki_name_constraints(
                ta.subject,
                ta.spki,
                ta.name_constraints,
            )
        }));

        let mut builder = rustls::ClientConfig::builder().with_safe_defaults();
        let mut crypto = if let Some(v) = verifier {
            builder
                .with_custom_certificate_verifier(v)
                .with_no_client_auth()
        } else {
            builder
                .with_root_certificates(root_store)
                .with_no_client_auth()
        };

        crypto.alpn_protocols = config
            .alpn_protocols
            .iter()
            .map(|p| p.as_bytes().to_vec())
            .collect();

        let mut transport = TransportConfig::default();
        transport.max_datagram_frame_size(Some(65535));

        let mut client_cfg = ClientConfig::new(Arc::new(crypto));
        client_cfg.transport_config(Arc::new(transport));

        let mut endpoint = Endpoint::client("0.0.0.0:0".parse().unwrap())
            .map_err(|e| HtxError::Transport(format!("Failed to create client endpoint: {}", e)))?;
        endpoint.set_default_client_config(client_cfg);

        // Optional ECH configuration loading
        let ech_config = if config.enable_tls_camouflage {
            if let Some(path) = &config.ech_config_path {
                match EchConfig::from_file(
                    path,
                    config.camouflage_domain.clone().unwrap_or_default(),
                ) {
                    Ok(cfg) => Some(cfg),
                    Err(e) => {
                        warn!("Failed to load ECH config: {}", e);
                        None
                    }
                }
            } else if let Some(domain) = &config.camouflage_domain {
                match EchConfig::from_dns(domain).await {
                    Ok(cfg) => Some(cfg),
                    Err(e) => {
                        warn!("Failed to resolve ECH config: {}", e);
                        None
                    }
                }
            } else {
                None
            }
        } else {
            None
        };

        let connecting = endpoint
            .connect(addr, "localhost")
            .map_err(|e| HtxError::Transport(format!("Connection error: {}", e)))?;

        let connection = connecting
            .await
            .map_err(|e| HtxError::Transport(format!("Handshake failed: {}", e)))?;

        let enable_datagrams = connection.max_datagram_size().is_some();

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
            Err(SendDatagramError::UnsupportedByPeer) => Err(HtxError::Transport(
                "DATAGRAM unsupported by peer".to_string(),
            )),
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
        // Generate self-signed certificate for the server
        let cert = generate_simple_self_signed(vec!["localhost".to_string()])
            .map_err(|e| HtxError::Config(format!("Certificate generation failed: {}", e)))?;
        let cert_der = CertificateDer::from(cert.cert);
        let key_der = PrivatePkcs8KeyDer::from(cert.key_pair.serialize_der());

        let mut server_config = ServerConfig::with_single_cert(vec![cert_der], key_der.into())
            .map_err(|e| HtxError::Transport(format!("Server config error: {}", e)))?;

        // Configure ALPN and transport parameters
        if let Some(crypto) = Arc::get_mut(&mut server_config.crypto) {
            crypto.alpn_protocols = config
                .alpn_protocols
                .iter()
                .map(|p| p.as_bytes().to_vec())
                .collect();
        }

        let mut transport = TransportConfig::default();
        transport.max_datagram_frame_size(Some(65535));
        server_config.transport = Arc::new(transport);

        let mut endpoint = Endpoint::server(server_config, config.listen_addr)
            .map_err(|e| HtxError::Transport(format!("Failed to create server endpoint: {}", e)))?;

        // Optional ECH configuration loading
        let ech_config = if config.enable_tls_camouflage {
            if let Some(path) = &config.ech_config_path {
                match EchConfig::from_file(
                    path,
                    config.camouflage_domain.clone().unwrap_or_default(),
                ) {
                    Ok(cfg) => Some(cfg),
                    Err(e) => {
                        warn!("Failed to load ECH config: {}", e);
                        None
                    }
                }
            } else if let Some(domain) = &config.camouflage_domain {
                match EchConfig::from_dns(domain).await {
                    Ok(cfg) => Some(cfg),
                    Err(e) => {
                        warn!("Failed to resolve ECH config: {}", e);
                        None
                    }
                }
            } else {
                None
            }
        } else {
            None
        };

        let handler = Arc::new(handler);
        while let Some(connecting) = endpoint.accept().await {
            let handler = handler.clone();
            let ech_cfg = ech_config.clone();
            tokio::spawn(async move {
                match connecting.await {
                    Ok(conn) => {
                        let transport = QuicTransport {
                            enable_datagrams: conn.max_datagram_size().is_some(),
                            connection: conn,
                            stats: TransportStats::new(),
                            ech_config: ech_cfg,
                        };
                        handler(Box::new(transport));
                    }
                    Err(e) => warn!("Failed to accept QUIC connection: {}", e),
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

#[cfg(all(test, feature = "quic"))]
mod tests {
    use super::*;

    fn build_config_bytes(max_len: u16) -> Vec<u8> {
        let mut bytes = Vec::new();
        // config_id
        bytes.push(0);
        // kem_id
        bytes.extend_from_slice(&0x0020u16.to_be_bytes());
        // public_key length and value (1 byte)
        bytes.extend_from_slice(&1u16.to_be_bytes());
        bytes.push(0);
        // cipher_suites length and one suite
        bytes.extend_from_slice(&2u16.to_be_bytes());
        bytes.extend_from_slice(&0x1301u16.to_be_bytes());
        // maximum_name_length
        bytes.extend_from_slice(&max_len.to_be_bytes());
        // extensions length (0)
        bytes.extend_from_slice(&0u16.to_be_bytes());
        bytes
    }

    #[test]
    fn parses_boundary_maximum_name_length() {
        let data = build_config_bytes(255);
        let cfg = EchConfig::from_bytes(&data, "example.com".into()).unwrap();
        assert_eq!(cfg.maximum_name_length, 255);
    }

    #[test]
    fn rejects_overflowing_maximum_name_length() {
        let data = build_config_bytes(256);
        assert!(EchConfig::from_bytes(&data, "example.com".into()).is_err());
    }

    use std::sync::atomic::Ordering;

    #[tokio::test]
    async fn repeated_lookups_reuse_resolver() {
        assert_eq!(DNS_RESOLVER_INIT_COUNT.load(Ordering::Relaxed), 0);

        let _ = EchConfig::from_dns("localhost").await;
        assert_eq!(DNS_RESOLVER_INIT_COUNT.load(Ordering::Relaxed), 1);

        let _ = EchConfig::from_dns("localhost").await;
        assert_eq!(DNS_RESOLVER_INIT_COUNT.load(Ordering::Relaxed), 1);
    }
}

// Stub implementation when QUIC feature is disabled
#[cfg(not(feature = "quic"))]
pub struct QuicTransport;

#[cfg(not(feature = "quic"))]
impl QuicTransport {
    pub async fn connect(
        _addr: std::net::SocketAddr,
        _config: &crate::HtxConfig,
    ) -> crate::Result<Self> {
        Err(crate::HtxError::Config(
            "QUIC feature not enabled".to_string(),
        ))
    }

    pub async fn listen<F>(_config: crate::HtxConfig, _handler: F) -> crate::Result<()>
    where
        F: Fn(Box<dyn crate::HtxConnection>) + Send + Sync + 'static,
    {
        Err(crate::HtxError::Config(
            "QUIC feature not enabled".to_string(),
        ))
    }
}
