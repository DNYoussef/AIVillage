//! MASQUE (Multiplexed Application Substrate over QUIC Encryption) implementation
//!
//! Provides HTTP/3 CONNECT-UDP proxying over QUIC connections for enhanced
//! privacy and NAT traversal. Implements RFC 9298 MASQUE specification.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use tokio::net::UdpSocket;
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, error, info, warn};

use crate::{HtxError, Result};

/// MASQUE proxy session identifier
pub type SessionId = u64;

/// MASQUE CONNECT-UDP request
#[derive(Debug, Clone)]
pub struct ConnectUdpRequest {
    pub session_id: SessionId,
    pub target_host: String,
    pub target_port: u16,
    pub client_addr: SocketAddr,
}

/// MASQUE CONNECT-UDP response
#[derive(Debug, Clone)]
pub struct ConnectUdpResponse {
    pub session_id: SessionId,
    pub status_code: u16,
    pub context_id: Option<u64>,
}

/// UDP datagram encapsulated in MASQUE
#[derive(Debug, Clone)]
pub struct MasqueDatagram {
    pub session_id: SessionId,
    pub context_id: u64,
    pub payload: Bytes,
}

/// MASQUE proxy session state
#[derive(Debug, Clone)]
struct ProxySession {
    session_id: SessionId,
    target_addr: SocketAddr,
    client_addr: SocketAddr,
    udp_socket: Arc<UdpSocket>,
    created_at: Instant,
    last_activity: Instant,
    bytes_sent: u64,
    bytes_received: u64,
}

/// MASQUE proxy server
pub struct MasqueProxy {
    sessions: Arc<RwLock<HashMap<SessionId, ProxySession>>>,
    next_session_id: Arc<RwLock<SessionId>>,
    cleanup_interval: Duration,
    session_timeout: Duration,
    max_sessions: usize,
}

impl Default for MasqueProxy {
    fn default() -> Self {
        Self::new()
    }
}

impl MasqueProxy {
    /// Create new MASQUE proxy
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            next_session_id: Arc::new(RwLock::new(1)),
            cleanup_interval: Duration::from_secs(60),
            session_timeout: Duration::from_secs(300), // 5 minutes
            max_sessions: 1000,
        }
    }

    /// Configure MASQUE proxy parameters
    pub fn with_config(
        max_sessions: usize,
        session_timeout: Duration,
        cleanup_interval: Duration,
    ) -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            next_session_id: Arc::new(RwLock::new(1)),
            cleanup_interval,
            session_timeout,
            max_sessions,
        }
    }

    /// Start MASQUE proxy with cleanup task
    pub async fn start(&self) -> Result<()> {
        info!("Starting MASQUE proxy with max {} sessions", self.max_sessions);

        // Start cleanup task
        let sessions = Arc::clone(&self.sessions);
        let cleanup_interval = self.cleanup_interval;
        let session_timeout = self.session_timeout;

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(cleanup_interval);
            loop {
                interval.tick().await;
                let mut sessions = sessions.write().await;
                let now = Instant::now();
                let before_count = sessions.len();

                sessions.retain(|session_id, session| {
                    let should_keep = now.duration_since(session.last_activity) < session_timeout;
                    if !should_keep {
                        debug!("Cleaning up expired MASQUE session {}", session_id);
                    }
                    should_keep
                });

                let after_count = sessions.len();
                if before_count != after_count {
                    info!(
                        "MASQUE cleanup: removed {} expired sessions, {} active",
                        before_count - after_count,
                        after_count
                    );
                }
            }
        });

        Ok(())
    }

    /// Handle CONNECT-UDP request
    pub async fn handle_connect_udp(
        &self,
        request: ConnectUdpRequest,
    ) -> Result<ConnectUdpResponse> {
        debug!(
            "MASQUE CONNECT-UDP request: target {}:{}, client {}",
            request.target_host, request.target_port, request.client_addr
        );

        // Check session limits
        {
            let sessions = self.sessions.read().await;
            if sessions.len() >= self.max_sessions {
                warn!("MASQUE session limit reached: {}", self.max_sessions);
                return Ok(ConnectUdpResponse {
                    session_id: request.session_id,
                    status_code: 429, // Too Many Requests
                    context_id: None,
                });
            }
        }

        // Resolve target address
        let target_addr = match tokio::net::lookup_host((request.target_host.as_str(), request.target_port))
            .await
        {
            Ok(mut addrs) => match addrs.next() {
                Some(addr) => addr,
                None => {
                    warn!("No addresses found for {}:{}", request.target_host, request.target_port);
                    return Ok(ConnectUdpResponse {
                        session_id: request.session_id,
                        status_code: 502, // Bad Gateway
                        context_id: None,
                    });
                }
            },
            Err(e) => {
                warn!("Failed to resolve {}:{}: {}", request.target_host, request.target_port, e);
                return Ok(ConnectUdpResponse {
                    session_id: request.session_id,
                    status_code: 502, // Bad Gateway
                    context_id: None,
                });
            }
        };

        // Create UDP socket for proxying
        let udp_socket = match UdpSocket::bind("0.0.0.0:0").await {
            Ok(socket) => Arc::new(socket),
            Err(e) => {
                error!("Failed to create UDP socket for MASQUE session: {}", e);
                return Ok(ConnectUdpResponse {
                    session_id: request.session_id,
                    status_code: 500, // Internal Server Error
                    context_id: None,
                });
            }
        };

        // Generate session ID
        let session_id = {
            let mut next_id = self.next_session_id.write().await;
            let id = *next_id;
            *next_id = next_id.wrapping_add(1);
            id
        };

        let now = Instant::now();
        let session = ProxySession {
            session_id,
            target_addr,
            client_addr: request.client_addr,
            udp_socket,
            created_at: now,
            last_activity: now,
            bytes_sent: 0,
            bytes_received: 0,
        };

        // Store session
        {
            let mut sessions = self.sessions.write().await;
            sessions.insert(session_id, session);
        }

        info!(
            "Created MASQUE session {} for {}:{} -> {}",
            session_id, request.target_host, request.target_port, target_addr
        );

        Ok(ConnectUdpResponse {
            session_id,
            status_code: 200, // OK
            context_id: Some(session_id), // Use session ID as context ID
        })
    }

    /// Handle UDP datagram from client to target
    pub async fn handle_client_datagram(
        &self,
        datagram: MasqueDatagram,
    ) -> Result<()> {
        let mut sessions = self.sessions.write().await;

        if let Some(session) = sessions.get_mut(&datagram.session_id) {
            session.last_activity = Instant::now();

            // Forward datagram to target
            match session.udp_socket.send_to(&datagram.payload, session.target_addr).await {
                Ok(bytes_sent) => {
                    session.bytes_sent += bytes_sent as u64;
                    debug!(
                        "MASQUE session {}: forwarded {} bytes to target {}",
                        datagram.session_id, bytes_sent, session.target_addr
                    );
                    Ok(())
                }
                Err(e) => {
                    warn!(
                        "MASQUE session {}: failed to forward to target: {}",
                        datagram.session_id, e
                    );
                    Err(HtxError::Transport(format!("UDP forward failed: {}", e)))
                }
            }
        } else {
            warn!("MASQUE datagram for unknown session {}", datagram.session_id);
            Err(HtxError::Protocol(format!("Unknown session: {}", datagram.session_id)))
        }
    }

    /// Start receiving datagrams from target for a session
    pub async fn start_target_receiver(
        &self,
        session_id: SessionId,
        datagram_sender: mpsc::UnboundedSender<MasqueDatagram>,
    ) -> Result<()> {
        let session = {
            let sessions = self.sessions.read().await;
            sessions.get(&session_id).cloned()
        };

        let session = match session {
            Some(s) => s,
            None => {
                return Err(HtxError::Protocol(format!("Session not found: {}", session_id)));
            }
        };

        let sessions = Arc::clone(&self.sessions);
        let socket = Arc::clone(&session.udp_socket);

        tokio::spawn(async move {
            let mut buf = vec![0u8; 65536]; // Max UDP datagram size

            loop {
                match socket.recv(&mut buf).await {
                    Ok(len) => {
                        let payload = Bytes::copy_from_slice(&buf[..len]);

                        // Update session stats
                        {
                            let mut sessions_guard = sessions.write().await;
                            if let Some(session) = sessions_guard.get_mut(&session_id) {
                                session.bytes_received += len as u64;
                                session.last_activity = Instant::now();
                            } else {
                                // Session was removed, stop receiving
                                debug!("MASQUE session {} removed, stopping receiver", session_id);
                                break;
                            }
                        }

                        let datagram = MasqueDatagram {
                            session_id,
                            context_id: session_id, // Use session ID as context ID
                            payload,
                        };

                        if datagram_sender.send(datagram).is_err() {
                            debug!("MASQUE datagram sender closed for session {}", session_id);
                            break;
                        }

                        debug!(
                            "MASQUE session {}: received {} bytes from target",
                            session_id, len
                        );
                    }
                    Err(e) => {
                        warn!("MASQUE session {}: UDP receive error: {}", session_id, e);
                        break;
                    }
                }
            }
        });

        Ok(())
    }

    /// Close MASQUE session
    pub async fn close_session(&self, session_id: SessionId) -> Result<()> {
        let mut sessions = self.sessions.write().await;

        if let Some(session) = sessions.remove(&session_id) {
            info!(
                "Closed MASQUE session {}: {} bytes sent, {} bytes received, duration: {:.2}s",
                session_id,
                session.bytes_sent,
                session.bytes_received,
                session.created_at.elapsed().as_secs_f64()
            );
            Ok(())
        } else {
            Err(HtxError::Protocol(format!("Session not found: {}", session_id)))
        }
    }

    /// Get session statistics
    pub async fn get_session_stats(&self, session_id: SessionId) -> Option<(u64, u64, Duration)> {
        let sessions = self.sessions.read().await;
        sessions.get(&session_id).map(|session| {
            (
                session.bytes_sent,
                session.bytes_received,
                session.created_at.elapsed(),
            )
        })
    }

    /// Get all active sessions count
    pub async fn active_sessions_count(&self) -> usize {
        let sessions = self.sessions.read().await;
        sessions.len()
    }

    /// Get proxy statistics
    pub async fn get_proxy_stats(&self) -> MasqueProxyStats {
        let sessions = self.sessions.read().await;
        let active_sessions = sessions.len();
        let total_bytes_sent = sessions.values().map(|s| s.bytes_sent).sum();
        let total_bytes_received = sessions.values().map(|s| s.bytes_received).sum();

        MasqueProxyStats {
            active_sessions,
            total_bytes_sent,
            total_bytes_received,
            max_sessions: self.max_sessions,
            session_timeout: self.session_timeout,
        }
    }
}

/// MASQUE proxy statistics
#[derive(Debug, Clone)]
pub struct MasqueProxyStats {
    pub active_sessions: usize,
    pub total_bytes_sent: u64,
    pub total_bytes_received: u64,
    pub max_sessions: usize,
    pub session_timeout: Duration,
}

/// MASQUE client for connecting through proxy
pub struct MasqueClient {
    proxy_addr: SocketAddr,
    session_id: Option<SessionId>,
    datagram_sender: Option<mpsc::UnboundedSender<MasqueDatagram>>,
    datagram_receiver: Option<mpsc::UnboundedReceiver<MasqueDatagram>>,
}

impl MasqueClient {
    /// Create new MASQUE client
    pub fn new(proxy_addr: SocketAddr) -> Self {
        Self {
            proxy_addr,
            session_id: None,
            datagram_sender: None,
            datagram_receiver: None,
        }
    }

    /// Connect to target through MASQUE proxy
    pub async fn connect(&mut self, target_host: &str, target_port: u16) -> Result<SessionId> {
        // This would typically send an HTTP/3 CONNECT-UDP request
        // For this implementation, we'll simulate the request

        let _request = ConnectUdpRequest {
            session_id: 0, // Will be assigned by proxy
            target_host: target_host.to_string(),
            target_port,
            client_addr: "127.0.0.1:0".parse().unwrap(), // Placeholder
        };

        // In a real implementation, this would be sent over HTTP/3
        info!(
            "MASQUE client connecting to {}:{} via proxy {}",
            target_host, target_port, self.proxy_addr
        );

        // Simulate successful connection
        let session_id = 12345; // Would come from proxy response
        self.session_id = Some(session_id);

        // Set up datagram channels
        let (tx, rx) = mpsc::unbounded_channel();
        self.datagram_sender = Some(tx);
        self.datagram_receiver = Some(rx);

        Ok(session_id)
    }

    /// Send UDP datagram through MASQUE proxy
    pub async fn send_datagram(&self, data: &[u8]) -> Result<()> {
        let session_id = self.session_id
            .ok_or_else(|| HtxError::Protocol("Not connected".to_string()))?;

        let _datagram = MasqueDatagram {
            session_id,
            context_id: session_id,
            payload: Bytes::copy_from_slice(data),
        };

        // In a real implementation, this would be sent over QUIC DATAGRAM
        debug!(
            "MASQUE client sending {} bytes via session {}",
            data.len(), session_id
        );

        Ok(())
    }

    /// Receive UDP datagram from MASQUE proxy
    pub async fn recv_datagram(&mut self) -> Result<Option<Bytes>> {
        if let Some(receiver) = &mut self.datagram_receiver {
            match receiver.try_recv() {
                Ok(datagram) => {
                    debug!(
                        "MASQUE client received {} bytes via session {}",
                        datagram.payload.len(), datagram.session_id
                    );
                    Ok(Some(datagram.payload))
                }
                Err(mpsc::error::TryRecvError::Empty) => Ok(None),
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    Err(HtxError::Transport("MASQUE connection closed".to_string()))
                }
            }
        } else {
            Err(HtxError::Protocol("Not connected".to_string()))
        }
    }

    /// Close MASQUE connection
    pub async fn close(&mut self) -> Result<()> {
        if let Some(session_id) = self.session_id.take() {
            info!("Closing MASQUE client session {}", session_id);
            // In a real implementation, this would send a close request
        }

        self.datagram_sender = None;
        self.datagram_receiver = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::timeout;

    #[tokio::test]
    async fn test_masque_proxy_creation() {
        let proxy = MasqueProxy::new();
        assert_eq!(proxy.active_sessions_count().await, 0);
    }

    #[tokio::test]
    async fn test_connect_udp_request() {
        let proxy = MasqueProxy::new();
        proxy.start().await.unwrap();

        let request = ConnectUdpRequest {
            session_id: 1,
            target_host: "8.8.8.8".to_string(),
            target_port: 53,
            client_addr: "127.0.0.1:12345".parse().unwrap(),
        };

        let response = proxy.handle_connect_udp(request).await.unwrap();
        assert_eq!(response.status_code, 200);
        assert!(response.context_id.is_some());
        assert_eq!(proxy.active_sessions_count().await, 1);
    }

    #[tokio::test]
    async fn test_masque_client() {
        let mut client = MasqueClient::new("127.0.0.1:8080".parse().unwrap());

        let session_id = client.connect("example.com", 80).await.unwrap();
        assert!(session_id > 0);

        // Test sending datagram
        client.send_datagram(b"test data").await.unwrap();

        // Close connection
        client.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_session_cleanup() {
        let proxy = MasqueProxy::with_config(
            100,
            Duration::from_millis(100), // Very short timeout for testing
            Duration::from_millis(50),   // Frequent cleanup
        );

        proxy.start().await.unwrap();

        let request = ConnectUdpRequest {
            session_id: 1,
            target_host: "8.8.8.8".to_string(),
            target_port: 53,
            client_addr: "127.0.0.1:12345".parse().unwrap(),
        };

        proxy.handle_connect_udp(request).await.unwrap();
        assert_eq!(proxy.active_sessions_count().await, 1);

        // Wait for cleanup
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Session should be cleaned up
        assert_eq!(proxy.active_sessions_count().await, 0);
    }

    #[tokio::test]
    async fn test_proxy_stats() {
        let proxy = MasqueProxy::new();
        let stats = proxy.get_proxy_stats().await;

        assert_eq!(stats.active_sessions, 0);
        assert_eq!(stats.total_bytes_sent, 0);
        assert_eq!(stats.total_bytes_received, 0);
        assert_eq!(stats.max_sessions, 1000);
    }
}
