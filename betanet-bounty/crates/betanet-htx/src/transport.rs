//! Transport abstraction layer for HTX protocol
//!
//! Provides a unified interface for different transport mechanisms (TCP, QUIC)
//! with TLS camouflage and connection management.

use crate::{Frame, HtxError, Result};
use async_trait::async_trait;
use bytes::Bytes;
use std::net::SocketAddr;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};

/// Transport connection identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConnectionId(pub u64);

/// Transport stream identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StreamId(pub u32);

/// Transport listener for accepting incoming connections
#[async_trait]
pub trait TransportListener: Send + Sync {
    type Connection: TransportConnection;

    /// Accept a new connection from the listener
    async fn accept(&mut self) -> Result<(Self::Connection, SocketAddr)>;

    /// Get the local address this listener is bound to
    fn local_addr(&self) -> Result<SocketAddr>;
}

/// Transport connection interface
#[async_trait]
pub trait TransportConnection: Send + Sync + 'static {
    type Stream: TransportStream;

    /// Open a new outbound stream on this connection
    async fn open_stream(&mut self) -> Result<Self::Stream>;

    /// Accept an incoming stream (for connection receivers)
    async fn accept_stream(&mut self) -> Result<Option<Self::Stream>>;

    /// Read a frame from the connection control channel
    async fn read_frame(&mut self) -> Result<Option<Frame>>;

    /// Write a frame to the connection control channel
    async fn write_frame(&mut self, frame: Frame) -> Result<()>;

    /// Get the remote peer address
    fn peer_addr(&self) -> Result<SocketAddr>;

    /// Get the local address
    fn local_addr(&self) -> Result<SocketAddr>;

    /// Check if the connection is still alive
    fn is_alive(&self) -> bool;

    /// Close the connection gracefully
    async fn close(&mut self) -> Result<()>;
}

/// Transport stream interface for multiplexed streams
#[async_trait]
pub trait TransportStream: AsyncRead + AsyncWrite + Send + Sync + Unpin + 'static {
    /// Get the stream ID
    fn stream_id(&self) -> StreamId;

    /// Check if the stream is still open
    fn is_open(&self) -> bool;

    /// Close the stream gracefully
    async fn close(&mut self) -> Result<()>;

    /// Reset the stream with an error code
    async fn reset(&mut self, error_code: u32) -> Result<()>;
}

/// Transport factory for creating connections and listeners
#[async_trait]
pub trait Transport: Send + Sync + Clone + 'static {
    type Connection: TransportConnection;
    type Listener: TransportListener<Connection = Self::Connection>;

    /// Connect to a remote address
    async fn connect(&self, addr: SocketAddr) -> Result<Self::Connection>;

    /// Bind a listener to a local address
    async fn bind(&self, addr: SocketAddr) -> Result<Self::Listener>;

    /// Get the transport name (for debugging/logging)
    fn name(&self) -> &'static str;
}

/// Configuration for transport connections
#[derive(Debug, Clone)]
pub struct TransportConfig {
    /// Maximum number of concurrent streams per connection
    pub max_streams: u32,
    /// Connection timeout in milliseconds
    pub connect_timeout_ms: u64,
    /// Keep-alive interval in milliseconds
    pub keepalive_interval_ms: u64,
    /// Maximum frame size
    pub max_frame_size: usize,
    /// Enable TLS fingerprint camouflage
    pub enable_tls_camouflage: bool,
    /// Target domain for TLS camouflage (if enabled)
    pub camouflage_domain: Option<String>,
    /// ALPN protocols to advertise
    pub alpn_protocols: Vec<String>,
}

impl Default for TransportConfig {
    fn default() -> Self {
        Self {
            max_streams: 100,
            connect_timeout_ms: 30_000,
            keepalive_interval_ms: 10_000,
            max_frame_size: 65536,
            enable_tls_camouflage: false,
            camouflage_domain: None,
            alpn_protocols: vec!["htx/1.1".to_string()],
        }
    }
}

/// Transport stream wrapper that provides the TransportStream interface
pub struct GenericTransportStream<T> {
    inner: T,
    stream_id: StreamId,
    is_open: bool,
}

impl<T> GenericTransportStream<T> {
    pub fn new(inner: T, stream_id: StreamId) -> Self {
        Self {
            inner,
            stream_id,
            is_open: true,
        }
    }
}

#[async_trait]
impl<T> TransportStream for GenericTransportStream<T>
where
    T: AsyncRead + AsyncWrite + Send + Sync + Unpin + 'static,
{
    fn stream_id(&self) -> StreamId {
        self.stream_id
    }

    fn is_open(&self) -> bool {
        self.is_open
    }

    async fn close(&mut self) -> Result<()> {
        self.is_open = false;
        Ok(())
    }

    async fn reset(&mut self, _error_code: u32) -> Result<()> {
        self.is_open = false;
        Ok(())
    }
}

impl<T> AsyncRead for GenericTransportStream<T>
where
    T: AsyncRead + Unpin,
{
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.inner).poll_read(cx, buf)
    }
}

impl<T> AsyncWrite for GenericTransportStream<T>
where
    T: AsyncWrite + Unpin,
{
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<std::result::Result<usize, std::io::Error>> {
        Pin::new(&mut self.inner).poll_write(cx, buf)
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::result::Result<(), std::io::Error>> {
        Pin::new(&mut self.inner).poll_flush(cx)
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::result::Result<(), std::io::Error>> {
        Pin::new(&mut self.inner).poll_shutdown(cx)
    }
}

/// Transport statistics
#[derive(Debug, Default, Clone)]
pub struct TransportStats {
    /// Bytes sent
    pub bytes_sent: u64,
    /// Bytes received
    pub bytes_received: u64,
    /// Connections established
    pub connections_established: u64,
    /// Streams opened
    pub streams_opened: u64,
    /// Errors encountered
    pub errors: u64,
}

impl TransportStats {
    /// Create new transport statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Record sent data
    pub fn record_sent(&mut self, bytes: usize) {
        self.bytes_sent += bytes as u64;
    }

    /// Record received data
    pub fn record_received(&mut self, bytes: usize) {
        self.bytes_received += bytes as u64;
    }

    /// Record new connection
    pub fn record_connection(&mut self) {
        self.connections_established += 1;
    }

    /// Record new stream
    pub fn record_stream(&mut self) {
        self.streams_opened += 1;
    }

    /// Record error
    pub fn record_error(&mut self) {
        self.errors += 1;
    }
}

/// Error types specific to transport operations
#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),

    #[error("Connection timeout")]
    Timeout,

    #[error("Stream limit exceeded")]
    StreamLimitExceeded,

    #[error("Transport not available: {0}")]
    NotAvailable(String),

    #[error("TLS handshake failed: {0}")]
    TlsHandshakeFailed(String),

    #[error("ALPN negotiation failed")]
    AlpnFailed,

    #[error("Connection closed")]
    ConnectionClosed,

    #[error("Stream reset: {error_code}")]
    StreamReset { error_code: u32 },
}

impl From<TransportError> for HtxError {
    fn from(err: TransportError) -> Self {
        HtxError::Transport(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transport_stats() {
        let mut stats = TransportStats::new();
        assert_eq!(stats.bytes_sent, 0);

        stats.record_sent(100);
        assert_eq!(stats.bytes_sent, 100);

        stats.record_received(200);
        assert_eq!(stats.bytes_received, 200);

        stats.record_connection();
        assert_eq!(stats.connections_established, 1);

        stats.record_stream();
        assert_eq!(stats.streams_opened, 1);

        stats.record_error();
        assert_eq!(stats.errors, 1);
    }

    #[test]
    fn test_transport_config() {
        let config = TransportConfig::default();
        assert_eq!(config.max_streams, 100);
        assert_eq!(config.connect_timeout_ms, 30_000);
        assert!(!config.enable_tls_camouflage);
        assert_eq!(config.alpn_protocols, vec!["htx/1.1"]);
    }
}
