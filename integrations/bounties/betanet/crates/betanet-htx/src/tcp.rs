//! TCP transport implementation with TLS camouflage
//!
//! Implements the Transport trait for TCP connections with TLS encryption
//! and optional fingerprint camouflage.

use crate::transport::{
    StreamId, Transport, TransportConfig, TransportConnection, TransportListener, TransportStats,
    TransportStream,
};
use crate::{Frame, FrameBuffer, HtxError, NoiseXK, Result};
use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use rustls::pki_types::ServerName;
use rustls::{ClientConfig, RootCertStore, ServerConfig};
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Mutex;
use tokio_rustls::{TlsAcceptor, TlsConnector, TlsStream};
use tracing::{debug, info, warn};

/// TCP transport factory
#[derive(Clone)]
pub struct TcpTransport {
    config: TransportConfig,
    tls_connector: Option<TlsConnector>,
    tls_acceptor: Option<TlsAcceptor>,
}

/// TCP connection over TLS
pub struct TcpConnection {
    stream: TlsStream<TcpStream>,
    noise: Option<NoiseXK>,
    frame_buffer: FrameBuffer,
    streams: dashmap::DashMap<u32, Arc<Mutex<TcpMultiplexedStream>>>,
    next_stream_id: Arc<Mutex<u32>>,
    stats: Arc<Mutex<TransportStats>>,
    is_alive: Arc<Mutex<bool>>,
}

/// TCP listener for accepting TLS connections
pub struct TcpListener443 {
    listener: TcpListener,
    acceptor: TlsAcceptor,
    config: TransportConfig,
}

/// Multiplexed stream over TCP connection
pub struct TcpMultiplexedStream {
    stream_id: u32,
    read_buffer: BytesMut,
    write_buffer: BytesMut,
    is_open: bool,
}

impl TcpTransport {
    /// Create a new TCP transport with configuration
    pub fn new(config: TransportConfig) -> Self {
        Self {
            config,
            tls_connector: None,
            tls_acceptor: None,
        }
    }

    /// Create TCP transport with TLS client configuration
    pub fn with_client_tls(mut self, connector: TlsConnector) -> Self {
        self.tls_connector = Some(connector);
        self
    }

    /// Create TCP transport with TLS server configuration
    pub fn with_server_tls(mut self, acceptor: TlsAcceptor) -> Self {
        self.tls_acceptor = Some(acceptor);
        self
    }

    /// Create default client TLS configuration
    pub fn default_client_tls_config() -> Result<ClientConfig> {
        let mut root_store = RootCertStore::empty();
        root_store.extend(webpki_roots::TLS_SERVER_ROOTS.iter().map(|ta| {
            rustls::pki_types::TrustAnchor {
                subject: ta.subject.into(),
                subject_public_key_info: ta.spki.into(),
                name_constraints: ta.name_constraints.map(|nc| nc.into()),
            }
        }));

        let config = ClientConfig::builder()
            .with_root_certificates(root_store)
            .with_no_client_auth();

        Ok(config)
    }

    /// Create default server TLS configuration (for testing)
    /// Note: This is a placeholder - users must provide their own certificates
    pub fn default_server_tls_config() -> Result<ServerConfig> {
        Err(HtxError::Config(
            "Server TLS configuration requires external certificates. \
             Use with_server_tls() with your own TlsAcceptor."
                .to_string(),
        ))
    }
}

// Implement the Transport trait for TcpTransport
#[async_trait]
impl Transport for TcpTransport {
    type Connection = TcpConnection;
    type Listener = TcpListener443;

    async fn connect(&self, addr: SocketAddr) -> Result<Self::Connection> {
        info!("Connecting to {} via TCP-443", addr);

        // Create TCP connection
        let tcp_stream = TcpStream::connect(addr).await.map_err(HtxError::Io)?;

        // Establish TLS connection if configured
        let tls_stream = if let Some(connector) = &self.tls_connector {
            let server_name = ServerName::try_from("localhost")
                .map_err(|e| HtxError::Config(format!("Invalid server name: {}", e)))?;

            connector
                .connect(server_name, tcp_stream)
                .await
                .map_err(|e| HtxError::Transport(format!("TLS handshake failed: {}", e)))?
        } else {
            return Err(HtxError::Config(
                "TLS connector required for TCP transport".to_string(),
            ));
        };

        // Create connection with multiplexing support
        let connection = TcpConnection {
            stream: tokio_rustls::TlsStream::Client(tls_stream),
            noise: None, // Will be set up during Noise handshake
            frame_buffer: FrameBuffer::new(self.config.max_frame_size),
            streams: dashmap::DashMap::new(),
            next_stream_id: Arc::new(Mutex::new(1)), // Odd IDs for client-initiated streams
            stats: Arc::new(Mutex::new(TransportStats::new())),
            is_alive: Arc::new(Mutex::new(true)),
        };

        Ok(connection)
    }

    async fn bind(&self, addr: SocketAddr) -> Result<Self::Listener> {
        info!("Binding TCP-443 listener on {}", addr);

        let listener = TcpListener::bind(addr).await.map_err(HtxError::Io)?;

        let acceptor = self.tls_acceptor.as_ref().ok_or_else(|| {
            HtxError::Config("TLS acceptor required for TCP listener".to_string())
        })?;

        Ok(TcpListener443 {
            listener,
            acceptor: acceptor.clone(),
            config: self.config.clone(),
        })
    }

    fn name(&self) -> &'static str {
        "tcp-443"
    }
}

// Implement the TransportListener trait for TcpListener443
#[async_trait]
impl TransportListener for TcpListener443 {
    type Connection = TcpConnection;

    async fn accept(&mut self) -> Result<(Self::Connection, SocketAddr)> {
        // Accept TCP connection
        let (tcp_stream, peer_addr) = self.listener.accept().await.map_err(HtxError::Io)?;

        // Perform TLS handshake
        let tls_stream = self
            .acceptor
            .accept(tcp_stream)
            .await
            .map_err(|e| HtxError::Transport(format!("TLS handshake failed: {}", e)))?;

        // Create connection with multiplexing support
        let connection = TcpConnection {
            stream: tokio_rustls::TlsStream::Server(tls_stream),
            noise: None, // Will be set up during Noise handshake
            frame_buffer: FrameBuffer::new(self.config.max_frame_size),
            streams: dashmap::DashMap::new(),
            next_stream_id: Arc::new(Mutex::new(2)), // Even IDs for server-initiated streams
            stats: Arc::new(Mutex::new(TransportStats::new())),
            is_alive: Arc::new(Mutex::new(true)),
        };

        Ok((connection, peer_addr))
    }

    fn local_addr(&self) -> Result<SocketAddr> {
        self.listener.local_addr().map_err(HtxError::Io)
    }
}

// Implement the TransportConnection trait for TcpConnection
#[async_trait]
impl TransportConnection for TcpConnection {
    type Stream = TcpMultiplexedStream;

    async fn open_stream(&mut self) -> Result<Self::Stream> {
        // Generate new stream ID
        let stream_id = {
            let mut next_id = self.next_stream_id.lock().await;
            let id = *next_id;
            *next_id += 2; // Increment by 2 to maintain odd/even separation
            id
        };

        debug!("Opening new TCP stream with ID {}", stream_id);

        // Create multiplexed stream
        let stream = TcpMultiplexedStream {
            stream_id,
            read_buffer: BytesMut::new(),
            write_buffer: BytesMut::new(),
            is_open: true,
        };

        // Register stream
        self.streams
            .insert(stream_id, Arc::new(Mutex::new(stream.clone())));

        // Update stats
        {
            let mut stats = self.stats.lock().await;
            stats.record_stream();
        }

        Ok(stream)
    }

    async fn accept_stream(&mut self) -> Result<Option<Self::Stream>> {
        // For now, return None - this would need to wait for incoming stream frames
        // In a full implementation, this would listen for STREAM_OPEN frames
        Ok(None)
    }

    async fn read_frame(&mut self) -> Result<Option<Frame>> {
        use tokio::io::AsyncReadExt;

        // Try to parse a frame from the buffer
        if let Some(frame) = self.frame_buffer.parse_frames()?.pop() {
            return Ok(Some(frame));
        }

        // Read more data from the TLS stream
        let mut buf = [0u8; 4096];
        let n = self.stream.read(&mut buf).await.map_err(HtxError::Io)?;

        if n == 0 {
            // Connection closed
            *self.is_alive.lock().await = false;
            return Ok(None);
        }

        // Add to frame buffer
        self.frame_buffer
            .append_data(&Bytes::copy_from_slice(&buf[..n]))?;

        // Try to parse frame again
        Ok(self.frame_buffer.parse_frames()?.pop())
    }

    async fn write_frame(&mut self, frame: Frame) -> Result<()> {
        use tokio::io::AsyncWriteExt;

        let encoded = frame.encode();
        self.stream
            .write_all(&encoded)
            .await
            .map_err(HtxError::Io)?;

        // Update stats
        {
            let mut stats = self.stats.lock().await;
            stats.record_sent(encoded.len());
        }

        Ok(())
    }

    fn peer_addr(&self) -> Result<SocketAddr> {
        self.stream.get_ref().0.peer_addr().map_err(HtxError::Io)
    }

    fn local_addr(&self) -> Result<SocketAddr> {
        self.stream.get_ref().0.local_addr().map_err(HtxError::Io)
    }

    fn is_alive(&self) -> bool {
        // Check if connection is still alive (non-blocking)
        match self.is_alive.try_lock() {
            Ok(alive) => *alive,
            Err(_) => true, // If we can't get lock, assume alive
        }
    }

    async fn close(&mut self) -> Result<()> {
        use tokio::io::AsyncWriteExt;

        debug!("Closing TCP connection");

        // Close all streams
        for stream_ref in self.streams.iter() {
            let mut stream = stream_ref.value().lock().await;
            stream.is_open = false;
        }
        self.streams.clear();

        // Send close frame (using Control frame type)
        let close_frame = Frame::new(crate::frame::FrameType::Control, 0, "CLOSE".into())?;
        let _ = self.write_frame(close_frame).await;

        // Close TLS stream
        let _ = self.stream.shutdown().await;

        *self.is_alive.lock().await = false;

        Ok(())
    }
}

// Implement the TransportStream trait for TcpMultiplexedStream
#[async_trait]
impl TransportStream for TcpMultiplexedStream {
    fn stream_id(&self) -> StreamId {
        StreamId(self.stream_id)
    }

    fn is_open(&self) -> bool {
        self.is_open
    }

    async fn close(&mut self) -> Result<()> {
        debug!("Closing TCP multiplexed stream {}", self.stream_id);
        self.is_open = false;
        Ok(())
    }

    async fn reset(&mut self, error_code: u32) -> Result<()> {
        warn!(
            "Resetting TCP multiplexed stream {} with error code {}",
            self.stream_id, error_code
        );
        self.is_open = false;
        Ok(())
    }
}

// Implement AsyncRead for TcpMultiplexedStream
impl AsyncRead for TcpMultiplexedStream {
    fn poll_read(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        if !self.is_open {
            return Poll::Ready(Ok(()));
        }

        let available = self.read_buffer.len().min(buf.remaining());
        if available > 0 {
            let data = self.read_buffer.split_to(available);
            buf.put_slice(&data);
            Poll::Ready(Ok(()))
        } else {
            // No data available - would need to coordinate with connection
            // For now, return pending
            Poll::Pending
        }
    }
}

// Implement AsyncWrite for TcpMultiplexedStream
impl AsyncWrite for TcpMultiplexedStream {
    fn poll_write(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<std::result::Result<usize, std::io::Error>> {
        if !self.is_open {
            return Poll::Ready(Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "Stream is closed",
            )));
        }

        // Buffer the data - in a real implementation this would send frames
        self.write_buffer.extend_from_slice(buf);
        Poll::Ready(Ok(buf.len()))
    }

    fn poll_flush(
        self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
    ) -> Poll<std::result::Result<(), std::io::Error>> {
        // In a real implementation, this would flush buffered frames
        Poll::Ready(Ok(()))
    }

    fn poll_shutdown(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
    ) -> Poll<std::result::Result<(), std::io::Error>> {
        self.is_open = false;
        Poll::Ready(Ok(()))
    }
}

// Add Clone trait for TcpMultiplexedStream
impl Clone for TcpMultiplexedStream {
    fn clone(&self) -> Self {
        Self {
            stream_id: self.stream_id,
            read_buffer: self.read_buffer.clone(),
            write_buffer: self.write_buffer.clone(),
            is_open: self.is_open,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_tcp_transport_stats() {
        let stats = TransportStats::new();
        assert_eq!(stats.bytes_sent, 0);
        assert_eq!(stats.bytes_received, 0);
    }
}
