//! Error types for Betanet client

use thiserror::Error;

/// Main error type for Betanet operations
#[derive(Error, Debug)]
pub enum BetanetError {
    /// Configuration errors
    #[error("Configuration error: {0}")]
    Config(String),

    /// Transport layer errors
    #[error("Transport error: {0}")]
    Transport(#[from] TransportError),

    /// Security/cryptography errors
    #[error("Security error: {0}")]
    Security(#[from] SecurityError),

    /// Protocol errors
    #[error("Protocol error: {0}")]
    Protocol(#[from] ProtocolError),

    /// Network errors
    #[error("Network error: {0}")]
    Network(#[from] NetworkError),

    /// Integration errors
    #[error("Integration error: {0}")]
    Integration(String),

    /// Serialization errors
    #[error("Serialization error: {0}")]
    Serialization(#[from] SerializationError),

    /// I/O errors
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Generic internal errors
    #[error("Internal error: {0}")]
    Internal(String),
}

/// Transport layer specific errors
#[derive(Error, Debug)]
pub enum TransportError {
    /// QUIC transport errors
    #[error("QUIC error: {0}")]
    Quic(String),

    /// TCP transport errors
    #[error("TCP error: {0}")]
    Tcp(String),

    /// HTTP transport errors
    #[error("HTTP error: {0}")]
    Http(#[from] HttpError),

    /// Connection errors
    #[error("Connection error: {0}")]
    Connection(String),

    /// Timeout errors
    #[error("Transport timeout: {0}")]
    Timeout(String),

    /// Transport unavailable
    #[error("Transport unavailable: {0}")]
    Unavailable(String),

    /// Fallback errors
    #[error("Fallback error: {0}")]
    Fallback(String),
}

/// HTTP transport specific errors
#[derive(Error, Debug)]
pub enum HttpError {
    /// Invalid HTTP request
    #[error("Invalid HTTP request: {0}")]
    InvalidRequest(String),

    /// Invalid HTTP response
    #[error("Invalid HTTP response: {0}")]
    InvalidResponse(String),

    /// HTTP status error
    #[error("HTTP status error: {status} - {message}")]
    Status { status: u16, message: String },

    /// Header parsing error
    #[error("Header parsing error: {0}")]
    HeaderParsing(String),

    /// Body parsing error
    #[error("Body parsing error: {0}")]
    BodyParsing(String),

    /// Chrome fingerprint error
    #[error("Chrome fingerprint error: {0}")]
    ChromeFingerprint(String),
}

/// Security and cryptography errors
#[derive(Error, Debug)]
pub enum SecurityError {
    /// Noise protocol errors
    #[error("Noise protocol error: {0}")]
    Noise(String),

    /// TLS errors
    #[error("TLS error: {0}")]
    Tls(String),

    /// Peer verification errors
    #[error("Peer verification failed: {0}")]
    PeerVerification(String),

    /// Encryption/decryption errors
    #[error("Encryption error: {0}")]
    Encryption(String),

    /// Key management errors
    #[error("Key management error: {0}")]
    KeyManagement(String),

    /// Certificate errors
    #[error("Certificate error: {0}")]
    Certificate(String),

    /// Signature verification errors
    #[error("Signature verification failed: {0}")]
    SignatureVerification(String),

    /// Rate limiting errors
    #[error("Rate limit exceeded: {0}")]
    RateLimitExceeded(String),

    /// Trust score errors
    #[error("Trust score error: {0}")]
    TrustScore(String),
}

/// Protocol specific errors
#[derive(Error, Debug)]
pub enum ProtocolError {
    /// HTX protocol errors
    #[error("HTX protocol error: {0}")]
    Htx(String),

    /// Message format errors
    #[error("Invalid message format: {0}")]
    InvalidMessageFormat(String),

    /// Protocol version mismatch
    #[error("Protocol version mismatch: expected {expected}, got {actual}")]
    VersionMismatch { expected: String, actual: String },

    /// Unsupported protocol feature
    #[error("Unsupported protocol feature: {0}")]
    UnsupportedFeature(String),

    /// Message validation errors
    #[error("Message validation failed: {0}")]
    MessageValidation(String),

    /// Chunking errors
    #[error("Message chunking error: {0}")]
    Chunking(String),

    /// TTL expired
    #[error("Message TTL expired")]
    TtlExpired,

    /// Invalid peer ID
    #[error("Invalid peer ID: {0}")]
    InvalidPeerId(String),
}

/// Network layer errors
#[derive(Error, Debug)]
pub enum NetworkError {
    /// Connection refused
    #[error("Connection refused: {0}")]
    ConnectionRefused(String),

    /// Connection timeout
    #[error("Connection timeout: {0}")]
    ConnectionTimeout(String),

    /// Network unreachable
    #[error("Network unreachable: {0}")]
    NetworkUnreachable(String),

    /// DNS resolution errors
    #[error("DNS resolution failed: {0}")]
    DnsResolution(String),

    /// Peer discovery errors
    #[error("Peer discovery error: {0}")]
    PeerDiscovery(String),

    /// Routing errors
    #[error("Routing error: {0}")]
    Routing(String),

    /// NAT traversal errors
    #[error("NAT traversal failed: {0}")]
    NatTraversal(String),

    /// Firewall blocked
    #[error("Firewall blocked: {0}")]
    FirewallBlocked(String),
}

/// Serialization errors
#[derive(Error, Debug)]
pub enum SerializationError {
    /// JSON serialization/deserialization errors
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// CBOR serialization/deserialization errors
    #[error("CBOR error: {0}")]
    Cbor(String),

    /// TOML configuration errors
    #[error("TOML error: {0}")]
    Toml(#[from] toml::de::Error),

    /// Message encoding errors
    #[error("Message encoding error: {0}")]
    MessageEncoding(String),

    /// Invalid data format
    #[error("Invalid data format: {0}")]
    InvalidFormat(String),
}

/// Gateway specific errors
#[derive(Error, Debug)]
pub enum GatewayError {
    /// CBOR control message errors
    #[error("CBOR control error: {0}")]
    CborControl(String),

    /// Path validation errors
    #[error("Path validation failed: {0}")]
    PathValidation(String),

    /// Gateway not available
    #[error("Gateway not available: {0}")]
    Unavailable(String),

    /// Routing decision errors
    #[error("Routing decision error: {0}")]
    RoutingDecision(String),

    /// Bridge configuration errors
    #[error("Bridge configuration error: {0}")]
    BridgeConfiguration(String),
}

/// Result type alias for convenience
pub type Result<T> = std::result::Result<T, BetanetError>;

impl BetanetError {
    /// Check if error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            BetanetError::Transport(transport_err) => transport_err.is_recoverable(),
            BetanetError::Network(network_err) => network_err.is_recoverable(),
            BetanetError::Protocol(ProtocolError::TtlExpired) => false,
            BetanetError::Security(SecurityError::RateLimitExceeded(_)) => true,
            BetanetError::Io(_) => true,
            _ => false,
        }
    }

    /// Get error category for metrics
    pub fn category(&self) -> &'static str {
        match self {
            BetanetError::Config(_) => "config",
            BetanetError::Transport(_) => "transport",
            BetanetError::Security(_) => "security",
            BetanetError::Protocol(_) => "protocol",
            BetanetError::Network(_) => "network",
            BetanetError::Integration(_) => "integration",
            BetanetError::Serialization(_) => "serialization",
            BetanetError::Io(_) => "io",
            BetanetError::Internal(_) => "internal",
        }
    }

    /// Check if error should trigger fallback
    pub fn should_fallback(&self) -> bool {
        match self {
            BetanetError::Transport(_) => true,
            BetanetError::Network(NetworkError::ConnectionRefused(_)) => true,
            BetanetError::Network(NetworkError::ConnectionTimeout(_)) => true,
            BetanetError::Network(NetworkError::NetworkUnreachable(_)) => true,
            BetanetError::Security(SecurityError::RateLimitExceeded(_)) => true,
            _ => false,
        }
    }
}

impl TransportError {
    /// Check if transport error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            TransportError::Timeout(_) => true,
            TransportError::Connection(_) => true,
            TransportError::Unavailable(_) => true,
            TransportError::Fallback(_) => false, // Already in fallback
            _ => false,
        }
    }

    /// Get suggested retry delay
    pub fn retry_delay(&self) -> std::time::Duration {
        match self {
            TransportError::Timeout(_) => std::time::Duration::from_secs(1),
            TransportError::Connection(_) => std::time::Duration::from_secs(5),
            TransportError::Unavailable(_) => std::time::Duration::from_secs(10),
            _ => std::time::Duration::from_secs(30),
        }
    }
}

impl NetworkError {
    /// Check if network error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            NetworkError::ConnectionTimeout(_) => true,
            NetworkError::DnsResolution(_) => true,
            NetworkError::PeerDiscovery(_) => true,
            NetworkError::NatTraversal(_) => true,
            NetworkError::ConnectionRefused(_) => false,
            NetworkError::NetworkUnreachable(_) => false,
            NetworkError::Routing(_) => false,
            NetworkError::FirewallBlocked(_) => false,
        }
    }
}

/// Convert from external error types
impl From<quinn::ConnectionError> for BetanetError {
    fn from(err: quinn::ConnectionError) -> Self {
        BetanetError::Transport(TransportError::Quic(err.to_string()))
    }
}

impl From<hyper::Error> for BetanetError {
    fn from(err: hyper::Error) -> Self {
        BetanetError::Transport(TransportError::Http(HttpError::InvalidRequest(err.to_string())))
    }
}

impl From<rustls::Error> for BetanetError {
    fn from(err: rustls::Error) -> Self {
        BetanetError::Security(SecurityError::Tls(err.to_string()))
    }
}

impl From<snow::Error> for BetanetError {
    fn from(err: snow::Error) -> Self {
        BetanetError::Security(SecurityError::Noise(err.to_string()))
    }
}

impl From<ciborium::de::Error<std::io::Error>> for BetanetError {
    fn from(err: ciborium::de::Error<std::io::Error>) -> Self {
        BetanetError::Serialization(SerializationError::Cbor(err.to_string()))
    }
}

impl From<ciborium::ser::Error<std::io::Error>> for BetanetError {
    fn from(err: ciborium::ser::Error<std::io::Error>) -> Self {
        BetanetError::Serialization(SerializationError::Cbor(err.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_categories() {
        assert_eq!(BetanetError::Config("test".to_string()).category(), "config");
        assert_eq!(BetanetError::Transport(TransportError::Quic("test".to_string())).category(), "transport");
        assert_eq!(BetanetError::Security(SecurityError::Noise("test".to_string())).category(), "security");
    }

    #[test]
    fn test_error_recoverability() {
        assert!(BetanetError::Transport(TransportError::Timeout("test".to_string())).is_recoverable());
        assert!(!BetanetError::Protocol(ProtocolError::TtlExpired).is_recoverable());
        assert!(BetanetError::Security(SecurityError::RateLimitExceeded("test".to_string())).is_recoverable());
    }

    #[test]
    fn test_fallback_triggers() {
        assert!(BetanetError::Transport(TransportError::Unavailable("test".to_string())).should_fallback());
        assert!(BetanetError::Network(NetworkError::ConnectionTimeout("test".to_string())).should_fallback());
        assert!(!BetanetError::Config("test".to_string()).should_fallback());
    }

    #[test]
    fn test_retry_delays() {
        let timeout_err = TransportError::Timeout("test".to_string());
        assert_eq!(timeout_err.retry_delay(), std::time::Duration::from_secs(1));

        let connection_err = TransportError::Connection("test".to_string());
        assert_eq!(connection_err.retry_delay(), std::time::Duration::from_secs(5));
    }
}
