//! uTLS template generation

use crate::{ChromeVersion, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

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
        let mut body = Vec::new();

        // Legacy version
        body.extend_from_slice(&[0x03, 0x03]);

        // Random (deterministic for tests)
        let mut random = [0u8; 32];
        for (i, byte) in random.iter_mut().enumerate() {
            *byte = (i as u8).wrapping_mul(7);
        }
        body.extend_from_slice(&random);

        // Session ID
        body.push(32u8);
        body.extend_from_slice(&random);

        // Cipher suites
        let mut cipher_bytes = Vec::new();
        for &c in &self.cipher_suites {
            cipher_bytes.extend_from_slice(&c.to_be_bytes());
        }
        body.extend_from_slice(&(cipher_bytes.len() as u16).to_be_bytes());
        body.extend_from_slice(&cipher_bytes);

        // Compression methods
        body.push(1); // length
        body.push(0); // null

        // Extensions
        let mut ext = Vec::new();

        // Server Name
        let mut sni = Vec::new();
        sni.extend_from_slice(&((self.server_name.len() + 3) as u16).to_be_bytes());
        sni.push(0); // host_name
        sni.extend_from_slice(&(self.server_name.len() as u16).to_be_bytes());
        sni.extend_from_slice(self.server_name.as_bytes());
        ext.extend_from_slice(&0x0000u16.to_be_bytes());
        ext.extend_from_slice(&(sni.len() as u16).to_be_bytes());
        ext.extend_from_slice(&sni);

        // Supported Groups
        let mut groups = Vec::new();
        groups.extend_from_slice(&((self.curves.len() * 2) as u16).to_be_bytes());
        for &g in &self.curves {
            groups.extend_from_slice(&g.to_be_bytes());
        }
        ext.extend_from_slice(&0x000Au16.to_be_bytes());
        ext.extend_from_slice(&(groups.len() as u16).to_be_bytes());
        ext.extend_from_slice(&groups);

        // EC Point Formats
        let ec_pf = [1u8, 0u8];
        ext.extend_from_slice(&0x000Bu16.to_be_bytes());
        ext.extend_from_slice(&(ec_pf.len() as u16).to_be_bytes());
        ext.extend_from_slice(&ec_pf);

        // Signature Algorithms
        let mut sig = Vec::new();
        sig.extend_from_slice(&((self.signature_algorithms.len() * 2) as u16).to_be_bytes());
        for &alg in &self.signature_algorithms {
            sig.extend_from_slice(&alg.to_be_bytes());
        }
        ext.extend_from_slice(&0x000Du16.to_be_bytes());
        ext.extend_from_slice(&(sig.len() as u16).to_be_bytes());
        ext.extend_from_slice(&sig);

        // ALPN
        let mut alpn_list = Vec::new();
        for proto in &self.alpn_protocols {
            alpn_list.push(proto.len() as u8);
            alpn_list.extend_from_slice(proto.as_bytes());
        }
        let mut alpn = Vec::new();
        alpn.extend_from_slice(&(alpn_list.len() as u16).to_be_bytes());
        alpn.extend_from_slice(&alpn_list);
        ext.extend_from_slice(&0x0010u16.to_be_bytes());
        ext.extend_from_slice(&(alpn.len() as u16).to_be_bytes());
        ext.extend_from_slice(&alpn);

        // Supported Versions (TLS1.3 & 1.2)
        let versions = [0x03u8, 0x04, 0x03, 0x03];
        let mut sv = Vec::new();
        sv.push(versions.len() as u8);
        sv.extend_from_slice(&versions);
        ext.extend_from_slice(&0x002Bu16.to_be_bytes());
        ext.extend_from_slice(&(sv.len() as u16).to_be_bytes());
        ext.extend_from_slice(&sv);

        // Key Share (X25519 with zero key)
        let mut ks = Vec::new();
        let mut entry = Vec::new();
        entry.extend_from_slice(&0x001Du16.to_be_bytes());
        let key = [0u8; 32];
        entry.extend_from_slice(&(key.len() as u16).to_be_bytes());
        entry.extend_from_slice(&key);
        ks.extend_from_slice(&(entry.len() as u16).to_be_bytes());
        ks.extend_from_slice(&entry);
        ext.extend_from_slice(&0x0033u16.to_be_bytes());
        ext.extend_from_slice(&(ks.len() as u16).to_be_bytes());
        ext.extend_from_slice(&ks);

        body.extend_from_slice(&(ext.len() as u16).to_be_bytes());
        body.extend_from_slice(&ext);

        // Handshake header
        let mut handshake = Vec::new();
        handshake.push(0x01);
        handshake.extend_from_slice(&(body.len() as u32).to_be_bytes()[1..]);
        handshake.extend_from_slice(&body);

        // Record header
        let mut record = Vec::new();
        record.push(0x16);
        record.extend_from_slice(&[0x03, 0x03]);
        record.extend_from_slice(&(handshake.len() as u16).to_be_bytes());
        record.extend_from_slice(&handshake);

        Ok(record)
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

    /// Get JA4 fingerprint
    pub fn ja4_fingerprint(&self) -> String {
        let protocol = "t";
        let version = "12"; // TLS1.2
        let alpn = self
            .alpn_protocols
            .get(0)
            .cloned()
            .unwrap_or_else(|| "00".to_string());
        let cipher_count = format!("{:02x}", self.cipher_suites.len().min(255));
        let ext_count = format!("{:02x}", self.extensions.len().min(255));
        let alpn_ext = if self.alpn_protocols.is_empty() {
            "00".to_string()
        } else {
            alpn.clone()
        };

        let mut ciphers = self.cipher_suites.clone();
        ciphers.sort_unstable();
        let cipher_string = ciphers
            .iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let cipher_hash = hex::encode(Sha256::digest(cipher_string.as_bytes()));
        let cipher_hash = &cipher_hash[..12];

        let mut extensions = self.extensions.clone();
        extensions.sort_unstable();
        let ext_string = extensions
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let ext_hash = hex::encode(Sha256::digest(ext_string.as_bytes()));
        let ext_hash = &ext_hash[..12];

        format!(
            "{}{}{}{}{}{}_{}{}",
            protocol, alpn, version, cipher_count, ext_count, alpn_ext, cipher_hash, ext_hash
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
        let len = u16::from_be_bytes([wire[3], wire[4]]) as usize;
        assert_eq!(len + 5, wire.len());
    }

    #[test]
    fn test_ja3_fingerprint() {
        let template = TlsTemplate::for_chrome(ChromeVersion::current_stable_n2(), "example.com");

        let fingerprint = template.ja3_fingerprint();
        assert!(!fingerprint.is_empty());
        assert!(fingerprint.contains('|')); // Should have JA3 separator
    }

    #[test]
    fn test_ja4_fingerprint() {
        let template = TlsTemplate::for_chrome(ChromeVersion::current_stable_n2(), "example.com");
        let fingerprint = template.ja4_fingerprint();
        assert!(!fingerprint.is_empty());
        assert!(fingerprint.contains('_'));
    }

    #[test]
    fn test_multiple_version_fingerprints() {
        let versions = vec![
            ChromeVersion::new(119, 0, 6045, 199),
            ChromeVersion::new(120, 0, 6099, 225),
        ];

        for v in versions {
            let template = TlsTemplate::for_chrome(v, "example.com");
            let ja3 = template.ja3_fingerprint();
            let ja4 = template.ja4_fingerprint();
            assert!(ja3.contains('|'));
            assert!(ja4.contains('_'));
        }
    }
}
