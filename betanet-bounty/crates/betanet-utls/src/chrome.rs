//! Chrome browser profiles

use crate::{cipher_suites, tls_version, ChromeVersion};
use serde::{Deserialize, Serialize};

/// Chrome TLS profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChromeProfile {
    /// Chrome version
    pub version: ChromeVersion,
    /// TLS version
    pub tls_version: u16,
    /// Cipher suites in Chrome order
    pub cipher_suites: Vec<u16>,
    /// Supported groups
    pub supported_groups: Vec<u16>,
    /// Signature algorithms
    pub signature_algorithms: Vec<u16>,
}

impl ChromeProfile {
    /// Chrome 119 profile (N-2 stable)
    pub fn chrome_119() -> Self {
        Self {
            version: ChromeVersion::new(119, 0, 6045, 123),
            tls_version: tls_version::TLS_1_2,
            cipher_suites: vec![
                cipher_suites::TLS_AES_128_GCM_SHA256,
                cipher_suites::TLS_AES_256_GCM_SHA384,
                cipher_suites::TLS_CHACHA20_POLY1305_SHA256,
                cipher_suites::TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
                cipher_suites::TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
                cipher_suites::TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
                cipher_suites::TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
            ],
            supported_groups: vec![0x001d, 0x0017, 0x0018, 0x0019],
            signature_algorithms: vec![0x0403, 0x0503, 0x0603, 0x0807, 0x0808, 0x0809],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chrome_119_profile() {
        let profile = ChromeProfile::chrome_119();
        assert_eq!(profile.version.major, 119);
        assert!(!profile.cipher_suites.is_empty());
    }
}
