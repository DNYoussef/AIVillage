//! uTLS template generation

use crate::{ChromeVersion, Result};
use serde::{Deserialize, Serialize};

/// uTLS template (legacy)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UtlsTemplate {
    /// Template name
    pub name: String,
    /// Chrome version
    pub version: ChromeVersion,
    /// Template data
    pub template_data: Vec<u8>,
}

impl UtlsTemplate {
    /// Generate Chrome N-2 template
    pub fn chrome_n2() -> Self {
        Self {
            name: "Chrome_119".to_string(),
            version: ChromeVersion::current_stable_n2(),
            template_data: vec![], // Placeholder
        }
    }
}

/// TLS ClientHello template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlsTemplate {
    /// Chrome version this template mimics
    pub version: ChromeVersion,
    /// Cipher suites list
    pub cipher_suites: Vec<u16>,
    /// TLS extensions list
    pub extensions: Vec<u16>,
    /// Supported elliptic curves
    pub curves: Vec<u16>,
    /// Signature algorithms
    pub signature_algorithms: Vec<u16>,
    /// ALPN protocols
    pub alpn_protocols: Vec<String>,
    /// Server name
    pub server_name: String,
}

impl TlsTemplate {
    /// Create new template for Chrome version
    pub fn for_chrome(version: ChromeVersion, server_name: impl Into<String>) -> Self {
        use crate::cipher_suites::*;
        use crate::extensions::*;

        Self {
            version,
            cipher_suites: vec![
                TLS_AES_128_GCM_SHA256,
                TLS_AES_256_GCM_SHA384,
                TLS_CHACHA20_POLY1305_SHA256,
                TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
                TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
                TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
                TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
            ],
            extensions: vec![
                SERVER_NAME,
                SUPPORTED_GROUPS,
                EC_POINT_FORMATS,
                SIGNATURE_ALGORITHMS,
                ALPN,
                SUPPORTED_VERSIONS,
                KEY_SHARE,
            ],
            curves: vec![0x001d, 0x0017, 0x0018], // X25519, secp256r1, secp384r1
            signature_algorithms: vec![
                0x0403, // ECDSA-SECP256R1-SHA256
                0x0804, // RSA-PSS-RSAE-SHA256
                0x0401, // RSA-PKCS1-SHA256
                0x0503, // ECDSA-SECP384R1-SHA384
                0x0805, // RSA-PSS-RSAE-SHA384
                0x0501, // RSA-PKCS1-SHA384
                0x0806, // RSA-PSS-RSAE-SHA512
                0x0601, // RSA-PKCS1-SHA512
            ],
            alpn_protocols: vec!["h2".to_string(), "http/1.1".to_string()],
            server_name: server_name.into(),
        }
    }

    /// Serialize template to wire format
    pub fn to_wire_format(&self) -> Result<Vec<u8>> {
        let mut buf = Vec::new();

        // This is a stub implementation - in reality would generate
        // a proper TLS ClientHello message

        // TLS Record Header (5 bytes)
        buf.push(0x16); // Content Type: Handshake
        buf.extend_from_slice(&[0x03, 0x03]); // Version: TLS 1.2
        let record_length = 0u16; // Placeholder
        buf.extend_from_slice(&record_length.to_be_bytes());

        // Handshake Header (4 bytes)
        buf.push(0x01); // Handshake Type: ClientHello
        let handshake_length = 0u32; // Placeholder (24-bit)
        let length_bytes = handshake_length.to_be_bytes();
        buf.extend_from_slice(&length_bytes[1..]); // Skip first byte for 24-bit

        // ClientHello
        buf.extend_from_slice(&[0x03, 0x03]); // Version: TLS 1.2

        // Random (32 bytes)
        let mut random = [0u8; 32];
        for (i, byte) in random.iter_mut().enumerate() {
            *byte = (i as u8).wrapping_mul(7); // Deterministic for testing
        }
        buf.extend_from_slice(&random);

        // Session ID Length + Session ID
        buf.push(0x20); // Session ID Length: 32 bytes
        buf.extend_from_slice(&random); // Reuse random as session ID

        // Cipher Suites
        let cipher_length = (self.cipher_suites.len() * 2) as u16;
        buf.extend_from_slice(&cipher_length.to_be_bytes());
        for &cipher in &self.cipher_suites {
            buf.extend_from_slice(&cipher.to_be_bytes());
        }

        // Compression Methods
        buf.push(0x01); // Length: 1
        buf.push(0x00); // null compression

        // Extensions (simplified)
        let extensions_length = 100u16; // Placeholder
        buf.extend_from_slice(&extensions_length.to_be_bytes());

        // Add server name extension
        buf.extend_from_slice(&[0x00, 0x00]); // Extension Type: server_name
        let name_length = (self.server_name.len() + 5) as u16;
        buf.extend_from_slice(&name_length.to_be_bytes());
        let list_length = (self.server_name.len() + 3) as u16;
        buf.extend_from_slice(&list_length.to_be_bytes());
        buf.push(0x00); // Name Type: host_name
        let hostname_length = self.server_name.len() as u16;
        buf.extend_from_slice(&hostname_length.to_be_bytes());
        buf.extend_from_slice(self.server_name.as_bytes());

        Ok(buf)
    }

    /// Get JA3 fingerprint
    pub fn ja3_fingerprint(&self) -> String {
        // JA3 = MD5(SSLVersion,Cipher,SSLExtension,EllipticCurve,EllipticCurvePointFormat)
        let version = "771"; // TLS 1.2
        let ciphers = self
            .cipher_suites
            .iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join("-");
        let extensions = self
            .extensions
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join("-");
        let curves = self
            .curves
            .iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join("-");
        let point_formats = "0"; // Uncompressed

        // In real implementation would compute MD5 hash
        format!(
            "{}|{}|{}|{}|{}",
            version, ciphers, extensions, curves, point_formats
        )
    }
}

/// Template builder for customization
pub struct TlsTemplateBuilder {
    template: TlsTemplate,
}

impl TlsTemplateBuilder {
    /// Create builder from Chrome version
    pub fn for_chrome(version: ChromeVersion) -> Self {
        Self {
            template: TlsTemplate::for_chrome(version, "example.com"),
        }
    }

    /// Set server name
    pub fn server_name(mut self, name: impl Into<String>) -> Self {
        self.template.server_name = name.into();
        self
    }

    /// Set cipher suites
    pub fn cipher_suites(mut self, ciphers: Vec<u16>) -> Self {
        self.template.cipher_suites = ciphers;
        self
    }

    /// Set extensions
    pub fn extensions(mut self, extensions: Vec<u16>) -> Self {
        self.template.extensions = extensions;
        self
    }

    /// Set ALPN protocols
    pub fn alpn_protocols(mut self, alpn: Vec<String>) -> Self {
        self.template.alpn_protocols = alpn;
        self
    }

    /// Build the template
    pub fn build(self) -> TlsTemplate {
        self.template
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_template_generation() {
        let template = UtlsTemplate::chrome_n2();
        assert_eq!(template.name, "Chrome_119");
    }

    #[test]
    fn test_tls_template_creation() {
        let template = TlsTemplate::for_chrome(ChromeVersion::current_stable_n2(), "example.com");

        assert_eq!(template.server_name, "example.com");
        assert!(!template.cipher_suites.is_empty());
        assert!(!template.extensions.is_empty());
        assert_eq!(template.alpn_protocols, vec!["h2", "http/1.1"]);
    }

    #[test]
    fn test_tls_template_builder() {
        let template = TlsTemplateBuilder::for_chrome(ChromeVersion::current_stable_n2())
            .server_name("test.com")
            .alpn_protocols(vec!["h2".to_string()])
            .build();

        assert_eq!(template.server_name, "test.com");
        assert_eq!(template.alpn_protocols, vec!["h2"]);
    }

    #[test]
    fn test_wire_format_generation() {
        let template = TlsTemplate::for_chrome(ChromeVersion::current_stable_n2(), "example.com");

        let wire = template.to_wire_format().unwrap();
        assert!(!wire.is_empty());

        // Check TLS record header
        assert_eq!(wire[0], 0x16); // Handshake
        assert_eq!(wire[1], 0x03); // TLS 1.2 major
        assert_eq!(wire[2], 0x03); // TLS 1.2 minor
    }

    #[test]
    fn test_ja3_fingerprint() {
        let template = TlsTemplate::for_chrome(ChromeVersion::current_stable_n2(), "example.com");

        let fingerprint = template.ja3_fingerprint();
        assert!(!fingerprint.is_empty());
        assert!(fingerprint.contains('|')); // Should have JA3 separator
    }
}
