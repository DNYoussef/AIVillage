//! Betanet uTLS - Chrome-Stable (N-2) uTLS template generator
//!
//! This crate provides utilities for generating TLS ClientHello messages
//! that mimic Chrome browsers, with support for JA3/JA4 fingerprinting.

#![deny(warnings)]
#![deny(clippy::all)]
#![deny(missing_docs)]

use thiserror::Error;

pub mod chrome;
pub mod clienthello;
pub mod fingerprint;
pub mod refresh;
pub mod template;

pub use clienthello::{ClientHello, TlsExtension};
pub use fingerprint::TlsFingerprint;
pub use template::{TlsTemplate, TlsTemplateBuilder, UtlsTemplate};

#[cfg(feature = "ja3")]
pub mod ja3;

#[cfg(feature = "ja4")]
pub mod ja4;

/// uTLS errors
#[derive(Debug, Error)]
pub enum UtlsError {
    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Template error
    #[error("Template error: {0}")]
    Template(String),

    /// Fingerprint error
    #[error("Fingerprint error: {0}")]
    Fingerprint(String),

    /// TLS error
    #[error("TLS error: {0}")]
    Tls(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),

    /// Network error
    #[error("Network error: {0}")]
    Network(String),
}

/// Result type for uTLS operations
pub type Result<T> = std::result::Result<T, UtlsError>;

/// Chrome version information
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ChromeVersion {
    /// Major version
    pub major: u16,
    /// Minor version
    pub minor: u16,
    /// Build version
    pub build: u16,
    /// Patch version
    pub patch: u16,
}

impl ChromeVersion {
    /// Create new Chrome version
    pub fn new(major: u16, minor: u16, build: u16, patch: u16) -> Self {
        Self {
            major,
            minor,
            build,
            patch,
        }
    }

    /// Get current stable version (N-2 for compatibility)
    pub fn current_stable_n2() -> Self {
        // Chrome 119 (N-2 from current as of writing)
        Self::new(119, 0, 6045, 123)
    }
}

impl std::fmt::Display for ChromeVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}.{}.{}.{}",
            self.major, self.minor, self.build, self.patch
        )
    }
}

impl ChromeVersion {
    /// Parse from string
    pub fn from_string(s: &str) -> Result<Self> {
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() != 4 {
            return Err(UtlsError::Config(format!("Invalid version format: {}", s)));
        }

        let major = parts[0]
            .parse()
            .map_err(|e| UtlsError::Config(format!("Invalid major version: {}", e)))?;
        let minor = parts[1]
            .parse()
            .map_err(|e| UtlsError::Config(format!("Invalid minor version: {}", e)))?;
        let build = parts[2]
            .parse()
            .map_err(|e| UtlsError::Config(format!("Invalid build version: {}", e)))?;
        let patch = parts[3]
            .parse()
            .map_err(|e| UtlsError::Config(format!("Invalid patch version: {}", e)))?;

        Ok(Self::new(major, minor, build, patch))
    }
}

/// TLS version constants
pub mod tls_version {
    /// TLS 1.0
    pub const TLS_1_0: u16 = 0x0301;
    /// TLS 1.1
    pub const TLS_1_1: u16 = 0x0302;
    /// TLS 1.2
    pub const TLS_1_2: u16 = 0x0303;
    /// TLS 1.3
    pub const TLS_1_3: u16 = 0x0304;
}

/// Common cipher suites used by Chrome
pub mod cipher_suites {
    /// TLS_AES_128_GCM_SHA256
    pub const TLS_AES_128_GCM_SHA256: u16 = 0x1301;
    /// TLS_AES_256_GCM_SHA384
    pub const TLS_AES_256_GCM_SHA384: u16 = 0x1302;
    /// TLS_CHACHA20_POLY1305_SHA256
    pub const TLS_CHACHA20_POLY1305_SHA256: u16 = 0x1303;
    /// TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256
    pub const TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256: u16 = 0xc02b;
    /// TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256
    pub const TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256: u16 = 0xc02f;
    /// TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384
    pub const TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384: u16 = 0xc02c;
    /// TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384
    pub const TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384: u16 = 0xc030;
}

/// TLS extension types
pub mod extensions {
    /// Server name indication
    pub const SERVER_NAME: u16 = 0x0000;
    /// Supported groups
    pub const SUPPORTED_GROUPS: u16 = 0x000a;
    /// EC point formats
    pub const EC_POINT_FORMATS: u16 = 0x000b;
    /// Signature algorithms
    pub const SIGNATURE_ALGORITHMS: u16 = 0x000d;
    /// Application layer protocol negotiation
    pub const ALPN: u16 = 0x0010;
    /// Supported versions
    pub const SUPPORTED_VERSIONS: u16 = 0x002b;
    /// Key share
    pub const KEY_SHARE: u16 = 0x0033;
    /// PSK key exchange modes
    pub const PSK_KEY_EXCHANGE_MODES: u16 = 0x002d;
}

/// GREASE (Generate Random Extensions And Sustain Extensibility) values
pub mod grease {
    /// GREASE values used by Chrome for cipher suites, extensions, groups, etc.
    /// These follow the pattern 0x?A?A where ? can be 0-F
    pub const GREASE_VALUES: [u16; 16] = [
        0x0A0A, 0x1A1A, 0x2A2A, 0x3A3A, 0x4A4A, 0x5A5A, 0x6A6A, 0x7A7A, 0x8A8A, 0x9A9A, 0xAAAA,
        0xBABA, 0xCACA, 0xDADA, 0xEAEA, 0xFAFA,
    ];

    /// Get a deterministic GREASE value for a given index
    pub fn get_grease_value(index: usize) -> u16 {
        GREASE_VALUES[index % GREASE_VALUES.len()]
    }

    /// Get Chrome's typical GREASE cipher suite value
    pub fn grease_cipher_suite() -> u16 {
        0x0A0A // Chrome typically uses this for first GREASE cipher
    }

    /// Get Chrome's typical GREASE extension value
    pub fn grease_extension() -> u16 {
        0x1A1A // Chrome typically uses this for GREASE extensions
    }

    /// Get Chrome's typical GREASE named group value
    pub fn grease_named_group() -> u16 {
        0x2A2A // Chrome typically uses this for GREASE named groups
    }

    /// Get Chrome's typical GREASE signature algorithm value
    pub fn grease_signature_algorithm() -> u16 {
        0x0A0A // Chrome typically uses this for GREASE signature algorithms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chrome_version() {
        let version = ChromeVersion::new(119, 0, 6045, 123);
        assert_eq!(version.to_string(), "119.0.6045.123");

        let parsed = ChromeVersion::from_string("119.0.6045.123").unwrap();
        assert_eq!(parsed, version);
    }

    #[test]
    fn test_current_stable() {
        let version = ChromeVersion::current_stable_n2();
        assert_eq!(version.major, 119);
    }

    #[test]
    fn test_invalid_version() {
        let result = ChromeVersion::from_string("invalid");
        assert!(result.is_err());
    }
}
