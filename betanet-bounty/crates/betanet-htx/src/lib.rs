//! Betanet HTX transport protocol implementation
//!
//! HTX provides a hybrid transport layer with:
//! - Frame-based protocol with varint stream IDs
//! - Noise XK handshake with key rotation
//! - Access ticket authentication system
//! - TCP and QUIC transport support

#![deny(clippy::all)]
#![allow(missing_docs)]
#![allow(dead_code)]

use std::io;
use std::net::SocketAddr;
use std::path::PathBuf;

use bytes::{Bytes, BytesMut};
use thiserror::Error;
use tokio::net::{TcpListener, TcpStream};

// Core HTX modules
pub mod bootstrap;
pub mod frame;
pub mod noise;
pub mod privacy;
pub mod scion_mac;
// pub mod ticket;  // Temporarily disabled due to base64 API changes
// pub mod tls;  // Temporarily disabled due to dependency issues
pub mod transport;

// Transport implementations
#[cfg(feature = "tcp")]
pub mod tcp;

#[cfg(feature = "quic")]
pub mod quic;

pub mod masque;

// Re-export core types
pub use bootstrap::{
    AbuseTracker, Argon2Params, Argon2PoW, BootstrapError, BootstrapManager, BootstrapMessage,
    CpuPoW, CpuPoWParams, DeviceClass, PoWChallenge, PoWSolution,
};
pub use frame::{parse_frame, Frame, FrameBuffer, FrameError, FrameType};
pub use noise::{
    generate_keypair, HandshakeFragment, HandshakePhase, NoiseError, NoiseStatus, NoiseXK,
};
pub use privacy::{
    BudgetStatus, CompositionMethod, EdgeId, EdgePrivacy, EdgeType, PrivacyBudgetManager,
    PrivacyError, PrivacyPolicy, RoutePrivacy,
};
pub use scion_mac::{
    MacAlgorithm, MacKey, ScionMacError, ScionMacHandler, ScionPacketMac,
};
// pub use ticket::{
//     AccessTicket, AccessTicketManager, TicketError, TicketStatus, TicketType,
//     generate_issuer_keypair,
//     rotation::{
//         CarrierState, TicketCarrier, TicketRotationManager, MarkovTransition,
//         CarrierMarkovChain, PaddingGenerator, RotationStats,
//     },
// };
// pub use tls::{
//     TemplateCache, TemplateKey, TemplateCacheConfig, TlsCamouflageError,
//     BackgroundCalibrator, MixtureModel, SiteClass, FallbackReducer,
// };
pub use transport::{
    ConnectionId, StreamId, Transport, TransportConfig, TransportConnection, TransportError,
    TransportListener, TransportStats, TransportStream,
};

// Re-export transport implementations
#[cfg(feature = "tcp")]
pub use tcp::{TcpConnection, TcpListener443, TcpMultiplexedStream, TcpTransport};

/// HTX protocol version
pub const HTX_VERSION: u8 = 1;

/// Maximum frame size per HTX v1.1 spec (2^24 - 1)
pub const MAX_FRAME_SIZE: usize = 16_777_215;

/// HTX protocol errors
#[derive(Debug, Error)]
pub enum HtxError {
    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    /// Frame error
    #[error("Frame error: {0}")]
    Frame(#[from] FrameError),

    /// Noise protocol error
    #[error("Noise error: {0}")]
    Noise(#[from] NoiseError),

    /// Ticket error (temporarily disabled)
    // #[error("Ticket error: {0}")]
    // Ticket(#[from] TicketError),

    /// Protocol error
    #[error("Protocol error: {0}")]
    Protocol(String),

    /// Handshake error
    #[error("Handshake error: {0}")]
    Handshake(String),

    /// Transport error
    #[error("Transport error: {0}")]
    Transport(String),

    /// Cryptographic error
    #[error("Crypto error: {0}")]
    Crypto(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),

    /// Authentication error
    #[error("Authentication error: {0}")]
    Auth(String),

    /// Stream error
    #[error("Stream error: stream_id={stream_id}, reason={reason}")]
    Stream { stream_id: u32, reason: String },
}

/// Result type for HTX operations
pub type Result<T> = std::result::Result<T, HtxError>;

/// HTX configuration
#[derive(Debug, Clone)]
pub struct HtxConfig {
    /// Listen address
    pub listen_addr: SocketAddr,

    /// Enable TCP transport
    pub enable_tcp: bool,

    /// Enable QUIC transport
    pub enable_quic: bool,

    /// Enable Noise-XK handshake
    pub enable_noise_xk: bool,

    /// Enable access ticket authentication
    pub enable_tickets: bool,

    /// Enable TLS camouflage using Encrypted Client Hello
    pub enable_tls_camouflage: bool,

    /// Domain name used when fetching ECH configuration via DNS
    pub camouflage_domain: Option<String>,

    /// Optional path to a base64-encoded ECH config file
    pub ech_config_path: Option<PathBuf>,

    /// ALPN protocols to advertise
    pub alpn_protocols: Vec<String>,

    /// Static private key for Noise protocol (32 bytes)
    pub static_private_key: Option<Bytes>,

    /// Remote static public key for client connections (32 bytes)
    pub remote_static_key: Option<Bytes>,

    /// Maximum concurrent connections
    pub max_connections: usize,

    /// Connection timeout in seconds
    pub connection_timeout_secs: u64,

    /// Keep-alive interval in seconds
    pub keepalive_interval_secs: u64,

    /// Frame buffer size
    pub frame_buffer_size: usize,
}

impl Default for HtxConfig {
    fn default() -> Self {
        Self {
            listen_addr: "127.0.0.1:9000".parse().unwrap(),
            enable_tcp: true,
            enable_quic: false,
            enable_noise_xk: true,
            enable_tickets: true,
            enable_tls_camouflage: false,
            camouflage_domain: None,
            ech_config_path: None,
            alpn_protocols: vec!["htx/1.1".to_string()],
            static_private_key: None,
            remote_static_key: None,
            max_connections: 1000,
            connection_timeout_secs: 30,
            keepalive_interval_secs: 10,
            frame_buffer_size: 1048576, // 1MB
        }
    }
}

/// HTX session state
#[derive(Debug)]
pub enum SessionState {
    /// Initial state, ready to begin handshake
    Initialize,
    /// Handshake in progress
    Handshaking,
    /// Transport mode active, ready for data
    Transport,
    /// Session closed
    Closed,
}

/// HTX stream
#[derive(Debug)]
pub struct HtxStream {
    /// Stream ID
    pub stream_id: u32,
    /// Stream state
    pub state: StreamState,
    /// Receive buffer
    pub receive_buffer: BytesMut,
    /// Send window size
    pub send_window: u32,
    /// Receive window size
    pub receive_window: u32,
}

/// HTX stream state
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamState {
    /// Stream is open and ready for data
    Open,
    /// Stream is closed
    Closed,
    /// Stream has error
    Error(String),
}

impl HtxStream {
    /// Create new HTX stream
    pub fn new(stream_id: u32) -> Self {
        Self {
            stream_id,
            state: StreamState::Open,
            receive_buffer: BytesMut::new(),
            send_window: 65536,
            receive_window: 65536,
        }
    }

    /// Check if stream can send data
    pub fn can_send(&self) -> bool {
        matches!(self.state, StreamState::Open) && self.send_window > 0
    }

    /// Check if stream has received data
    pub fn has_data(&self) -> bool {
        !self.receive_buffer.is_empty()
    }

    /// Read data from stream
    pub fn read(&mut self, buf: &mut [u8]) -> usize {
        let available = self.receive_buffer.len().min(buf.len());
        if available > 0 {
            let data = self.receive_buffer.split_to(available);
            buf[..available].copy_from_slice(&data);
            available
        } else {
            0
        }
    }

    /// Write data to stream buffer
    pub fn write_to_buffer(&mut self, data: &[u8]) {
        self.receive_buffer.extend_from_slice(data);
    }

    /// Update send window
    pub fn update_send_window(&mut self, delta: u32) {
        self.send_window = self.send_window.saturating_add(delta);
    }

    /// Consume send window
    pub fn consume_send_window(&mut self, amount: u32) -> bool {
        if self.send_window >= amount {
            self.send_window -= amount;
            true
        } else {
            false
        }
    }
}

/// HTX session for managing connection state
pub struct HtxSession {
    /// Session configuration
    pub config: HtxConfig,
    /// Current session state
    pub state: SessionState,
    /// Noise protocol handler
    pub noise: Option<NoiseXK>,
    /// Frame buffer for incoming data
    pub frame_buffer: FrameBuffer,
    /// Active streams
    pub streams: std::collections::HashMap<u32, HtxStream>,
    /// Next stream ID to assign
    pub next_stream_id: u32,
    /// Access ticket manager (temporarily disabled)
    // pub ticket_manager: Option<AccessTicketManager>,
    /// Session statistics
    pub stats: SessionStats,
}

/// Session statistics
#[derive(Debug, Clone, Default)]
pub struct SessionStats {
    /// Total frames sent
    pub frames_sent: u64,
    /// Total frames received
    pub frames_received: u64,
    /// Total bytes sent
    pub bytes_sent: u64,
    /// Total bytes received
    pub bytes_received: u64,
    /// Active streams count
    pub active_streams: u32,
    /// Handshake completed successfully
    pub handshake_complete: bool,
}

impl HtxSession {
    /// Create new HTX session
    pub fn new(config: HtxConfig, is_initiator: bool) -> Result<Self> {
        let noise = if config.enable_noise_xk {
            Some(NoiseXK::new(
                is_initiator,
                config.static_private_key.as_ref().map(|k| k.as_ref()),
                config.remote_static_key.as_ref().map(|k| k.as_ref()),
            )?)
        } else {
            None
        };

        let frame_buffer = FrameBuffer::new(config.frame_buffer_size);

        // let ticket_manager = if config.enable_tickets {
        //     Some(AccessTicketManager::new(10000))
        // } else {
        //     None
        // };

        Ok(Self {
            config,
            state: SessionState::Initialize,
            noise,
            frame_buffer,
            streams: std::collections::HashMap::new(),
            next_stream_id: if is_initiator { 1 } else { 2 }, // Odd for initiator, even for responder
            // ticket_manager,
            stats: SessionStats::default(),
        })
    }

    /// Begin handshake process
    /// NOTE: This is a simplified version that only returns the first fragment
    /// A production implementation would handle all fragments properly
    pub async fn begin_handshake(&mut self) -> Result<Option<Bytes>> {
        if !matches!(self.state, SessionState::Initialize) {
            return Err(HtxError::Protocol(
                "Invalid state for handshake".to_string(),
            ));
        }

        self.state = SessionState::Handshaking;

        if let Some(ref mut noise) = self.noise {
            if noise.is_initiator {
                let fragments = noise.create_message_1()?;
                // For simplicity, return the first fragment's data
                // TODO: Implement proper fragment handling
                return Ok(fragments.first().map(|f| f.data.clone()));
            }
        }

        Ok(None)
    }

    /// Process handshake message
    /// NOTE: This is a simplified version that only handles single fragments
    /// A production implementation would handle reassembly properly
    pub async fn process_handshake(&mut self, message: &[u8]) -> Result<Option<Bytes>> {
        if !matches!(self.state, SessionState::Handshaking) {
            return Err(HtxError::Protocol("Not in handshake state".to_string()));
        }

        if let Some(ref mut noise) = self.noise {
            match noise.phase() {
                HandshakePhase::Uninitialized if !noise.is_initiator => {
                    noise.process_message_1(message)?;
                    let fragments = noise.create_message_2()?;
                    Ok(fragments.first().map(|f| f.data.clone()))
                }
                HandshakePhase::Message1 if noise.is_initiator => {
                    noise.process_message_2(message)?;
                    let fragments = noise.create_message_3()?;
                    self.state = SessionState::Transport;
                    self.stats.handshake_complete = true;
                    Ok(fragments.first().map(|f| f.data.clone()))
                }
                HandshakePhase::Message2 if !noise.is_initiator => {
                    noise.process_message_3(message)?;
                    self.state = SessionState::Transport;
                    self.stats.handshake_complete = true;
                    Ok(None)
                }
                _ => Err(HtxError::Handshake("Invalid handshake phase".to_string())),
            }
        } else {
            // No Noise protocol, complete handshake immediately
            self.state = SessionState::Transport;
            self.stats.handshake_complete = true;
            Ok(None)
        }
    }

    /// Send data on a stream
    pub async fn send_data(&mut self, stream_id: u32, data: &[u8]) -> Result<Bytes> {
        if !matches!(self.state, SessionState::Transport) {
            return Err(HtxError::Protocol(
                "Session not in transport state".to_string(),
            ));
        }

        // Get or create stream
        self.streams
            .entry(stream_id)
            .or_insert_with(|| HtxStream::new(stream_id));

        let stream = self.streams.get_mut(&stream_id).unwrap();
        if !stream.can_send() {
            return Err(HtxError::Stream {
                stream_id,
                reason: "Stream cannot send data".to_string(),
            });
        }

        // Check flow control
        if !stream.consume_send_window(data.len() as u32) {
            return Err(HtxError::Stream {
                stream_id,
                reason: "Send window exhausted".to_string(),
            });
        }

        // Create DATA frame
        let frame = Frame::data(stream_id, Bytes::from(data.to_vec()))?;
        let frame_bytes = frame.encode();

        // Encrypt if Noise is enabled
        let final_bytes = if let Some(ref mut noise) = self.noise {
            noise.encrypt(&frame_bytes)?
        } else {
            frame_bytes
        };

        self.stats.frames_sent += 1;
        self.stats.bytes_sent += final_bytes.len() as u64;

        Ok(final_bytes)
    }

    /// Process incoming data
    pub async fn process_data(&mut self, data: &[u8]) -> Result<Vec<(u32, Bytes)>> {
        let decrypted_data = if let Some(ref mut noise) = self.noise {
            if noise.is_transport_ready() {
                noise.decrypt(data)?
            } else {
                return Err(HtxError::Protocol("Noise transport not ready".to_string()));
            }
        } else {
            Bytes::from(data.to_vec())
        };

        // Add to frame buffer
        self.frame_buffer.append_data(&decrypted_data)?;

        // Parse frames
        let frames = self.frame_buffer.parse_frames()?;
        let mut stream_data = Vec::new();

        for frame in frames {
            self.stats.frames_received += 1;
            self.stats.bytes_received += frame.payload.len() as u64;

            match frame.frame_type {
                FrameType::Data => {
                    if frame.stream_id == 0 {
                        return Err(HtxError::Frame(FrameError::StreamIdTooLarge(0)));
                    }

                    // Get or create stream
                    if let std::collections::hash_map::Entry::Vacant(e) =
                        self.streams.entry(frame.stream_id)
                    {
                        e.insert(HtxStream::new(frame.stream_id));
                    }

                    let stream = self.streams.get_mut(&frame.stream_id).unwrap();
                    stream.write_to_buffer(&frame.payload);
                    stream_data.push((frame.stream_id, frame.payload));
                }
                FrameType::WindowUpdate => {
                    if frame.payload.len() < 4 {
                        return Err(HtxError::Protocol(
                            "Invalid WINDOW_UPDATE payload".to_string(),
                        ));
                    }

                    let window_delta = u32::from_be_bytes([
                        frame.payload[0],
                        frame.payload[1],
                        frame.payload[2],
                        frame.payload[3],
                    ]);

                    if let Some(stream) = self.streams.get_mut(&frame.stream_id) {
                        stream.update_send_window(window_delta);
                    }
                }
                FrameType::Ping => {
                    // Handle ping - could respond with pong
                }
                FrameType::KeyUpdate => {
                    if let Some(ref mut noise) = self.noise {
                        noise.process_key_update(&frame.payload)?;
                    }
                }
                _ => {
                    // Handle other frame types as needed
                }
            }
        }

        self.stats.active_streams = self.streams.len() as u32;
        Ok(stream_data)
    }

    /// Create new stream
    pub fn create_stream(&mut self) -> Result<u32> {
        let stream_id = self.next_stream_id;
        self.next_stream_id += 2; // Increment by 2 to maintain odd/even separation

        self.streams.insert(stream_id, HtxStream::new(stream_id));
        Ok(stream_id)
    }

    /// Close stream
    pub fn close_stream(&mut self, stream_id: u32) {
        if let Some(stream) = self.streams.get_mut(&stream_id) {
            stream.state = StreamState::Closed;
        }
    }

    /// Get session status
    pub fn status(&self) -> HtxSessionStatus {
        HtxSessionStatus {
            state: match self.state {
                SessionState::Initialize => "initialize".to_string(),
                SessionState::Handshaking => "handshaking".to_string(),
                SessionState::Transport => "transport".to_string(),
                SessionState::Closed => "closed".to_string(),
            },
            noise_status: self.noise.as_ref().map(|n| n.status()),
            stats: self.stats.clone(),
            active_streams: self.streams.len(),
        }
    }

    /// Close session
    pub async fn close(&mut self) -> Result<()> {
        self.state = SessionState::Closed;
        self.streams.clear();
        Ok(())
    }
}

/// HTX session status
#[derive(Debug, Clone)]
pub struct HtxSessionStatus {
    /// Current session state
    pub state: String,
    /// Noise protocol status
    pub noise_status: Option<NoiseStatus>,
    /// Session statistics
    pub stats: SessionStats,
    /// Number of active streams
    pub active_streams: usize,
}

/// Dial TCP connection to remote HTX server
pub async fn dial_tcp(addr: SocketAddr, config: HtxConfig) -> Result<HtxTcpConnection> {
    let stream = TcpStream::connect(addr).await?;
    let session = HtxSession::new(config, true)?; // Client is initiator
    Ok(HtxTcpConnection::new(stream, session))
}

/// Accept TCP connection for HTX server
pub async fn accept_tcp(listener: &TcpListener, config: HtxConfig) -> Result<HtxTcpConnection> {
    let (stream, _addr) = listener.accept().await?;
    let session = HtxSession::new(config, false)?; // Server is not initiator
    Ok(HtxTcpConnection::new(stream, session))
}

/// HTX over TCP connection
pub struct HtxTcpConnection {
    stream: TcpStream,
    session: HtxSession,
}

impl HtxTcpConnection {
    /// Create new HTX TCP connection
    pub fn new(stream: TcpStream, session: HtxSession) -> Self {
        Self { stream, session }
    }

    /// Get session reference
    pub fn session(&self) -> &HtxSession {
        &self.session
    }

    /// Get mutable session reference
    pub fn session_mut(&mut self) -> &mut HtxSession {
        &mut self.session
    }

    /// Perform HTX handshake
    pub async fn handshake(&mut self) -> Result<()> {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        // Begin handshake
        if let Some(initial_message) = self.session.begin_handshake().await? {
            self.stream.write_all(&initial_message).await?;
        }

        // Exchange handshake messages
        while !matches!(self.session.state, SessionState::Transport) {
            let mut buf = vec![0u8; 4096];
            let n = self.stream.read(&mut buf).await?;
            buf.truncate(n);

            if let Some(response) = self.session.process_handshake(&buf).await? {
                self.stream.write_all(&response).await?;
            }
        }

        Ok(())
    }

    /// Send data on stream
    pub async fn send(&mut self, stream_id: u32, data: &[u8]) -> Result<()> {
        use tokio::io::AsyncWriteExt;

        let encrypted = self.session.send_data(stream_id, data).await?;
        self.stream.write_all(&encrypted).await?;
        Ok(())
    }

    /// Receive data from any stream
    pub async fn recv(&mut self) -> Result<Vec<(u32, Bytes)>> {
        use tokio::io::AsyncReadExt;

        let mut buf = vec![0u8; 4096];
        let n = self.stream.read(&mut buf).await?;
        buf.truncate(n);

        self.session.process_data(&buf).await
    }

    /// Create new stream
    pub fn create_stream(&mut self) -> Result<u32> {
        self.session.create_stream()
    }

    /// Close the connection
    pub async fn close(mut self) -> Result<()> {
        self.session.close().await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = HtxConfig::default();
        assert!(config.enable_tcp);
        assert!(!config.enable_quic);
        assert!(config.enable_noise_xk);
        assert!(config.enable_tickets);
        assert_eq!(config.max_connections, 1000);
        assert_eq!(config.frame_buffer_size, 1048576);
    }

    #[test]
    fn test_session_creation() {
        let config = HtxConfig {
            enable_noise_xk: false,
            ..Default::default()
        }; // Disable Noise XK for test simplicity

        // Test initiator session
        let initiator_session = HtxSession::new(config.clone(), true);
        assert!(initiator_session.is_ok());
        let session = initiator_session.unwrap();
        assert_eq!(session.next_stream_id, 1); // Odd for initiator

        // Test responder session
        let responder_session = HtxSession::new(config, false);
        assert!(responder_session.is_ok());
        let session = responder_session.unwrap();
        assert_eq!(session.next_stream_id, 2); // Even for responder
    }

    #[test]
    fn test_stream_creation() {
        let config = HtxConfig {
            enable_noise_xk: false,
            ..Default::default()
        };
        let mut session = HtxSession::new(config, true).unwrap();

        let stream_id = session.create_stream().unwrap();
        assert_eq!(stream_id, 1);
        assert!(session.streams.contains_key(&stream_id));

        let next_stream_id = session.create_stream().unwrap();
        assert_eq!(next_stream_id, 3); // Increments by 2
    }

    #[test]
    fn test_stream_operations() {
        let mut stream = HtxStream::new(1);

        assert!(stream.can_send());
        assert!(!stream.has_data());

        // Test window consumption
        assert!(stream.consume_send_window(1000));
        assert_eq!(stream.send_window, 65536 - 1000);

        // Test data writing and reading
        stream.write_to_buffer(b"hello world");
        assert!(stream.has_data());

        let mut buf = [0u8; 20];
        let n = stream.read(&mut buf);
        assert_eq!(n, 11);
        assert_eq!(&buf[..n], b"hello world");
    }

    #[tokio::test]
    async fn test_session_state_transitions() {
        let config = HtxConfig {
            enable_noise_xk: false,
            ..Default::default()
        }; // Disable for simpler test

        let mut session = HtxSession::new(config, true).unwrap();
        assert!(matches!(session.state, SessionState::Initialize));

        // Begin handshake (no Noise)
        let initial_msg = session.begin_handshake().await.unwrap();
        assert!(initial_msg.is_none()); // No Noise = no initial message
        assert!(matches!(session.state, SessionState::Handshaking));

        // Process handshake (should complete immediately without Noise)
        let response = session.process_handshake(&[]).await.unwrap();
        assert!(response.is_none());
        assert!(matches!(session.state, SessionState::Transport));
        assert!(session.stats.handshake_complete);
    }

    #[test]
    fn test_session_status() {
        let config = HtxConfig {
            enable_noise_xk: false,
            ..Default::default()
        };
        let session = HtxSession::new(config, true).unwrap();

        let status = session.status();
        assert_eq!(status.state, "initialize");
        assert_eq!(status.active_streams, 0);
        assert!(!status.stats.handshake_complete);
    }

    proptest::proptest! {
        #[test]
        fn prop_htx_config_valid(
            max_connections in 1usize..10000,
            timeout in 1u64..3600,
            keepalive in 1u64..300,
        ) {
            let config = HtxConfig {
                enable_noise_xk: false, // Disable Noise XK for simpler testing
                max_connections,
                connection_timeout_secs: timeout,
                keepalive_interval_secs: keepalive,
                ..Default::default()
            };

            // Should be able to create sessions with any valid config
            let initiator = HtxSession::new(config.clone(), true);
            let responder = HtxSession::new(config, false);

            proptest::prop_assert!(initiator.is_ok());
            proptest::prop_assert!(responder.is_ok());
        }
    }
}
