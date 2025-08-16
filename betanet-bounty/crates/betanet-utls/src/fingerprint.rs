//! TLS fingerprinting utilities

use crate::{ClientHello, Result};

/// TLS fingerprint
#[derive(Debug, Clone)]
pub struct TlsFingerprint {
    /// Raw fingerprint string
    pub fingerprint: String,
    /// Fingerprint type
    pub fingerprint_type: String,
}

impl TlsFingerprint {
    /// Create new fingerprint
    pub fn new(fingerprint: String, fingerprint_type: String) -> Self {
        Self {
            fingerprint,
            fingerprint_type,
        }
    }
}

/// Generate fingerprint from ClientHello
pub fn generate_fingerprint(
    _hello: &ClientHello,
    fingerprint_type: &str,
) -> Result<TlsFingerprint> {
    match fingerprint_type {
        #[cfg(feature = "ja3")]
        "ja3" => crate::ja3::generate_ja3(_hello),
        #[cfg(feature = "ja4")]
        "ja4" => crate::ja4::generate_ja4(_hello),
        _ => Ok(TlsFingerprint::new(
            "unsupported".to_string(),
            fingerprint_type.to_string(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fingerprint_creation() {
        let fp = TlsFingerprint::new("test".to_string(), "ja3".to_string());
        assert_eq!(fp.fingerprint, "test");
        assert_eq!(fp.fingerprint_type, "ja3");
    }
}
