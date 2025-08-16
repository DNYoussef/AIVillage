//! TLS ClientHello message generation

use bytes::{BufMut, Bytes, BytesMut};
use serde::{Deserialize, Serialize};

use crate::{
    chrome::ChromeProfile, extensions, grease, tls_version, Result,
};

/// TLS extension
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlsExtension {
    /// Extension type
    pub extension_type: u16,
    /// Extension data
    pub data: Vec<u8>,
}

impl TlsExtension {
    /// Create new extension
    pub fn new(extension_type: u16, data: Vec<u8>) -> Self {
        Self {
            extension_type,
            data,
        }
    }

    /// Create server name indication extension
    pub fn server_name(hostname: &str) -> Self {
        let mut data = BytesMut::new();

        // Server name list length
        data.put_u16((hostname.len() + 5) as u16);

        // Name type (host_name = 0)
        data.put_u8(0);

        // Hostname length
        data.put_u16(hostname.len() as u16);

        // Hostname
        data.put(hostname.as_bytes());

        Self::new(extensions::SERVER_NAME, data.to_vec())
    }

    /// Create supported groups extension with GREASE
    pub fn supported_groups() -> Self {
        Self::supported_groups_with_grease(true)
    }

    /// Create supported groups extension with optional GREASE
    pub fn supported_groups_with_grease(include_grease: bool) -> Self {
        let mut data = BytesMut::new();

        let mut groups = Vec::new();

        // Add GREASE value first (Chrome behavior)
        if include_grease {
            groups.push(grease::grease_named_group());
        }

        // Named groups (Chrome order)
        groups.extend_from_slice(&[
            0x001d, // x25519
            0x0017, // secp256r1
            0x0018, // secp384r1
            0x0019, // secp521r1
            0x0100, // ffdhe2048
        ]);

        // Supported groups list length
        data.put_u16((groups.len() * 2) as u16);

        // Add all groups
        for group in groups {
            data.put_u16(group);
        }

        Self::new(extensions::SUPPORTED_GROUPS, data.to_vec())
    }

    /// Create GREASE extension (empty extension with GREASE type)
    pub fn grease() -> Self {
        Self::new(grease::grease_extension(), vec![])
    }

    /// Create ALPN extension
    pub fn alpn(protocols: &[&str]) -> Self {
        let mut data = BytesMut::new();

        // Calculate total length
        let total_len: usize = protocols.iter().map(|p| p.len() + 1).sum();
        data.put_u16(total_len as u16);

        // Protocol list
        for protocol in protocols {
            data.put_u8(protocol.len() as u8);
            data.put(protocol.as_bytes());
        }

        Self::new(extensions::ALPN, data.to_vec())
    }

    /// Encode extension to bytes
    pub fn encode(&self) -> Bytes {
        let mut buf = BytesMut::with_capacity(4 + self.data.len());
        buf.put_u16(self.extension_type);
        buf.put_u16(self.data.len() as u16);
        buf.put(self.data.as_slice());
        buf.freeze()
    }
}

/// TLS ClientHello message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientHello {
    /// TLS version
    pub version: u16,
    /// Random bytes
    pub random: [u8; 32],
    /// Session ID
    pub session_id: Vec<u8>,
    /// Cipher suites
    pub cipher_suites: Vec<u16>,
    /// Compression methods
    pub compression_methods: Vec<u8>,
    /// Extensions
    pub extensions: Vec<TlsExtension>,
}

impl ClientHello {
    /// Create new ClientHello
    pub fn new() -> Self {
        let mut random = [0u8; 32];
        rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut random);

        Self {
            version: tls_version::TLS_1_2,
            random,
            session_id: vec![],
            cipher_suites: vec![],
            compression_methods: vec![0], // null compression
            extensions: vec![],
        }
    }

    /// Create ClientHello from Chrome profile
    pub fn from_chrome_profile(profile: &ChromeProfile, hostname: &str) -> Self {
        Self::from_chrome_profile_with_grease(profile, hostname, true)
    }

    /// Create ClientHello from Chrome profile with optional GREASE
    pub fn from_chrome_profile_with_grease(profile: &ChromeProfile, hostname: &str, include_grease: bool) -> Self {
        let mut hello = Self::new();

        hello.version = profile.tls_version;

        // Add cipher suites with GREASE at the beginning (Chrome behavior)
        if include_grease {
            let mut cipher_suites = vec![grease::grease_cipher_suite()];
            cipher_suites.extend_from_slice(&profile.cipher_suites);
            hello.cipher_suites = cipher_suites;
        } else {
            hello.cipher_suites = profile.cipher_suites.clone();
        }

        // Add Chrome-like extensions in typical Chrome order
        hello.add_extension(TlsExtension::server_name(hostname));

        // Add GREASE extension early (Chrome behavior)
        if include_grease {
            hello.add_extension(TlsExtension::grease());
        }

        hello.add_extension(TlsExtension::supported_groups_with_grease(include_grease));
        hello.add_extension(TlsExtension::alpn(&["h2", "http/1.1"]));

        hello
    }

    /// Create Chrome N-2 stable ClientHello with GREASE (most accurate)
    pub fn chrome_n2_stable(hostname: &str) -> Self {
        use crate::cipher_suites::*;

        let mut hello = Self::new();
        hello.version = tls_version::TLS_1_2;

        // Chrome N-2 cipher suites with GREASE first
        hello.cipher_suites = vec![
            grease::grease_cipher_suite(), // GREASE cipher suite
            TLS_AES_128_GCM_SHA256,
            TLS_AES_256_GCM_SHA384,
            TLS_CHACHA20_POLY1305_SHA256,
            TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
            TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
            TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
            TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
        ];

        // Chrome N-2 extensions in proper order
        hello.add_extension(TlsExtension::server_name(hostname));
        hello.add_extension(TlsExtension::grease()); // GREASE extension
        hello.add_extension(TlsExtension::supported_groups_with_grease(true));
        hello.add_extension(TlsExtension::alpn(&["h2", "http/1.1"]));

        // Add a session ID that looks like Chrome (32 bytes)
        let mut session_id = vec![0u8; 32];
        rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut session_id);
        hello.set_session_id(session_id);

        hello
    }

    /// Add extension
    pub fn add_extension(&mut self, extension: TlsExtension) {
        self.extensions.push(extension);
    }

    /// Set cipher suites
    pub fn set_cipher_suites(&mut self, cipher_suites: Vec<u16>) {
        self.cipher_suites = cipher_suites;
    }

    /// Set session ID
    pub fn set_session_id(&mut self, session_id: Vec<u8>) {
        self.session_id = session_id;
    }

    /// Encode ClientHello to bytes
    pub fn encode(&self) -> Result<Bytes> {
        let mut buf = BytesMut::new();

        // Handshake type (ClientHello = 1)
        buf.put_u8(1);

        // We'll come back to fill in the length (24-bit)
        let length_pos = buf.len();
        buf.put_u8(0); // High byte
        buf.put_u16(0); // Low 16 bits

        // Protocol version
        buf.put_u16(self.version);

        // Random
        buf.put(&self.random[..]);

        // Session ID
        buf.put_u8(self.session_id.len() as u8);
        buf.put(self.session_id.as_slice());

        // Cipher suites
        buf.put_u16((self.cipher_suites.len() * 2) as u16);
        for cipher in &self.cipher_suites {
            buf.put_u16(*cipher);
        }

        // Compression methods
        buf.put_u8(self.compression_methods.len() as u8);
        buf.put(self.compression_methods.as_slice());

        // Extensions
        let extensions_start = buf.len();
        buf.put_u16(0); // Extensions length placeholder

        for extension in &self.extensions {
            buf.put(extension.encode());
        }

        // Fill in extensions length
        let extensions_len = buf.len() - extensions_start - 2;
        buf[extensions_start..extensions_start + 2]
            .copy_from_slice(&(extensions_len as u16).to_be_bytes());

        // Fill in handshake message length (24-bit)
        let message_len = buf.len() - 4; // Exclude type and length fields
        let length_bytes = (message_len as u32).to_be_bytes();
        buf[length_pos] = length_bytes[1]; // High byte
        buf[length_pos + 1] = length_bytes[2]; // Mid byte
        buf[length_pos + 2] = length_bytes[3]; // Low byte

        Ok(buf.freeze())
    }

    /// Calculate message size
    pub fn size(&self) -> usize {
        let extensions_size: usize = self.extensions.iter()
            .map(|ext| 4 + ext.data.len())
            .sum();

        4 + // Handshake header
        2 + // Version
        32 + // Random
        1 + self.session_id.len() + // Session ID
        2 + (self.cipher_suites.len() * 2) + // Cipher suites
        1 + self.compression_methods.len() + // Compression methods
        2 + extensions_size // Extensions
    }
}

impl Default for ClientHello {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cipher_suites;

    #[test]
    fn test_client_hello_creation() {
        let hello = ClientHello::new();
        assert_eq!(hello.version, tls_version::TLS_1_2);
        assert_eq!(hello.compression_methods, vec![0]);
        assert!(hello.extensions.is_empty());
    }

    #[test]
    fn test_extensions() {
        let sni = TlsExtension::server_name("example.com");
        assert_eq!(sni.extension_type, extensions::SERVER_NAME);
        assert!(!sni.data.is_empty());

        let groups = TlsExtension::supported_groups();
        assert_eq!(groups.extension_type, extensions::SUPPORTED_GROUPS);

        let alpn = TlsExtension::alpn(&["h2", "http/1.1"]);
        assert_eq!(alpn.extension_type, extensions::ALPN);
    }

    #[test]
    fn test_client_hello_encoding() {
        let mut hello = ClientHello::new();
        hello.set_cipher_suites(vec![
            cipher_suites::TLS_AES_128_GCM_SHA256,
            cipher_suites::TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
        ]);
        hello.add_extension(TlsExtension::server_name("test.com"));

        let encoded = hello.encode().unwrap();
        assert!(!encoded.is_empty());
        assert!(encoded.len() >= hello.size() - 10); // Allow some variance
    }

    #[test]
    fn test_chrome_profile_hello() {
        let profile = ChromeProfile::chrome_119();
        let hello = ClientHello::from_chrome_profile(&profile, "example.com");

        // Should have GREASE cipher as first cipher
        assert_eq!(hello.cipher_suites[0], crate::grease::grease_cipher_suite());
        assert!(!hello.extensions.is_empty());
    }

    #[test]
    fn test_chrome_n2_stable_grease() {
        let hello = ClientHello::chrome_n2_stable("test.com");

        // Should have GREASE cipher suite as first cipher
        assert_eq!(hello.cipher_suites[0], crate::grease::grease_cipher_suite());

        // Should have session ID (Chrome-like)
        assert_eq!(hello.session_id.len(), 32);

        // Should have GREASE extension
        let has_grease_ext = hello.extensions.iter()
            .any(|ext| ext.extension_type == crate::grease::grease_extension());
        assert!(has_grease_ext);

        // Should have server name extension
        let has_sni = hello.extensions.iter()
            .any(|ext| ext.extension_type == extensions::SERVER_NAME);
        assert!(has_sni);
    }

    #[test]
    fn test_grease_values() {
        use crate::grease::*;

        // Test GREASE values follow pattern 0x?A?A
        for &value in &grease::GREASE_VALUES {
            assert_eq!(value & 0x0F0F, 0x0A0A);
        }

        // Test deterministic GREASE selection
        assert_eq!(get_grease_value(0), grease::GREASE_VALUES[0]);
        assert_eq!(get_grease_value(16), grease::GREASE_VALUES[0]); // Wraps around
    }

    #[test]
    fn test_supported_groups_with_grease() {
        let groups_ext = TlsExtension::supported_groups_with_grease(true);
        assert_eq!(groups_ext.extension_type, extensions::SUPPORTED_GROUPS);

        // Should have at least 6 groups (1 GREASE + 5 real groups)
        let data = &groups_ext.data;
        assert!(data.len() >= 12); // 2 bytes length + 6*2 bytes groups

        let groups_length = u16::from_be_bytes([data[0], data[1]]) as usize;
        assert_eq!(groups_length, data.len() - 2);
    }
}
