//! JA4 fingerprinting

use crate::{extensions, grease, ClientHello, Result, TlsFingerprint};
use sha2::{Digest, Sha256};

/// Generate JA4 fingerprint from ClientHello
/// JA4 format: q+ALPN+Version+CipherCount+ExtensionCount+ALPN_Extension+_+CiphersHash+ExtensionsHash
pub fn generate_ja4(hello: &ClientHello) -> Result<TlsFingerprint> {
    // 1. Protocol (always "t" for TLS, "q" for QUIC)
    let protocol = "t";

    // 2. TLS Version
    let version = match hello.version {
        0x0301 => "10", // TLS 1.0
        0x0302 => "11", // TLS 1.1
        0x0303 => "12", // TLS 1.2
        0x0304 => "13", // TLS 1.3
        _ => "12",      // Default to TLS 1.2
    };

    // 3. First ALPN value or "00" if none
    let alpn = extract_first_alpn(hello).unwrap_or_else(|| "00".to_string());

    // 4. Cipher suites count (excluding GREASE)
    let filtered_ciphers: Vec<u16> = hello
        .cipher_suites
        .iter()
        .filter(|&&cipher| !is_grease_value(cipher))
        .copied()
        .collect();
    let cipher_count = format!("{:02x}", filtered_ciphers.len().min(255));

    // 5. Extensions count (excluding GREASE)
    let filtered_extensions: Vec<u16> = hello
        .extensions
        .iter()
        .map(|ext| ext.extension_type)
        .filter(|&ext_type| !is_grease_value(ext_type))
        .collect();
    let ext_count = format!("{:02x}", filtered_extensions.len().min(255));

    // 6. ALPN Extension presence ("00" if not present, or first ALPN value)
    let alpn_ext = if has_alpn_extension(hello) {
        alpn.clone()
    } else {
        "00".to_string()
    };

    // 7. Cipher suites hash (first 12 chars of SHA256 of sorted cipher list)
    let mut sorted_ciphers = filtered_ciphers;
    sorted_ciphers.sort_unstable();
    let cipher_string = sorted_ciphers
        .iter()
        .map(|c| c.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let cipher_hash = sha256_truncated(&cipher_string, 12);

    // 8. Extensions hash (first 12 chars of SHA256 of sorted extension list)
    let mut sorted_extensions = filtered_extensions;
    sorted_extensions.sort_unstable();
    let ext_string = sorted_extensions
        .iter()
        .map(|e| e.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let ext_hash = sha256_truncated(&ext_string, 12);

    // Build JA4 string: q+ALPN+Version+CipherCount+ExtensionCount+ALPN_Extension+_+CiphersHash+ExtensionsHash
    let ja4_string = format!(
        "{}{}{}{}{}{}_{}{}",
        protocol, alpn, version, cipher_count, ext_count, alpn_ext, cipher_hash, ext_hash
    );

    Ok(TlsFingerprint::new(ja4_string, "ja4".to_string()))
}

/// Check if a value is a GREASE value
fn is_grease_value(value: u16) -> bool {
    grease::GREASE_VALUES.contains(&value)
}

/// Extract the first ALPN protocol from ClientHello
fn extract_first_alpn(hello: &ClientHello) -> Option<String> {
    for ext in &hello.extensions {
        if ext.extension_type == extensions::ALPN && ext.data.len() >= 2 {
            let protocols_len = u16::from_be_bytes([ext.data[0], ext.data[1]]) as usize;

            if ext.data.len() >= 2 + protocols_len && protocols_len > 0 {
                // First protocol starts at index 2
                let mut offset = 2;
                if offset < ext.data.len() {
                    let proto_len = ext.data[offset] as usize;
                    offset += 1;

                    if offset + proto_len <= ext.data.len() {
                        let protocol =
                            String::from_utf8_lossy(&ext.data[offset..offset + proto_len]);
                        return Some(protocol.to_string());
                    }
                }
            }
        }
    }
    None
}

/// Check if ClientHello has ALPN extension
fn has_alpn_extension(hello: &ClientHello) -> bool {
    hello
        .extensions
        .iter()
        .any(|ext| ext.extension_type == extensions::ALPN)
}

/// Calculate SHA256 hash and truncate to specified length
fn sha256_truncated(input: &str, length: usize) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    let result = hasher.finalize();
    let hex_string = format!("{:x}", result);
    hex_string.chars().take(length).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ja4_generation() {
        let hello = ClientHello::new();
        let ja4 = generate_ja4(&hello).unwrap();
        assert_eq!(ja4.fingerprint_type, "ja4");
        assert!(ja4.fingerprint.starts_with("t"));
    }

    #[test]
    fn test_ja4_chrome_n2_stable() {
        let hello = ClientHello::chrome_n2_stable("test.com");
        let ja4 = generate_ja4(&hello).unwrap();

        // Should generate a JA4 fingerprint
        assert_eq!(ja4.fingerprint_type, "ja4");

        // Should start with "t" for TLS
        assert!(ja4.fingerprint.starts_with("t"));

        // Should contain "h2" ALPN (Chrome uses HTTP/2)
        assert!(ja4.fingerprint.contains("h2"));

        // Should contain "_" separator
        assert!(ja4.fingerprint.contains("_"));

        // Generate again to ensure consistency
        let ja4_2 = generate_ja4(&hello).unwrap();
        assert_eq!(ja4.fingerprint, ja4_2.fingerprint);
    }

    #[test]
    fn test_ja4_alpn_extraction() {
        let hello = ClientHello::chrome_n2_stable("test.com");
        let alpn = extract_first_alpn(&hello);

        // Chrome N-2 should have ALPN with h2
        assert_eq!(alpn, Some("h2".to_string()));

        // Test ALPN extension presence
        assert!(has_alpn_extension(&hello));
    }

    #[test]
    fn test_ja4_grease_filtering() {
        use crate::cipher_suites;

        let mut hello = ClientHello::new();
        hello.cipher_suites = vec![
            0x0A0A, // GREASE cipher - should be filtered
            cipher_suites::TLS_AES_128_GCM_SHA256,
            cipher_suites::TLS_AES_256_GCM_SHA384,
        ];

        let ja4 = generate_ja4(&hello).unwrap();

        // Should filter out GREASE values and count only real ciphers (2)
        assert_eq!(ja4.fingerprint_type, "ja4");
        assert!(ja4.fingerprint.contains("02")); // 2 ciphers in hex
    }

    #[test]
    fn test_ja4_version_encoding() {
        let mut hello = ClientHello::new();

        // Test TLS 1.2 (0x0303 -> "12")
        hello.version = 0x0303;
        let ja4 = generate_ja4(&hello).unwrap();
        assert!(ja4.fingerprint.contains("12"));

        // Test TLS 1.3 (0x0304 -> "13")
        hello.version = 0x0304;
        let ja4_13 = generate_ja4(&hello).unwrap();
        assert!(ja4_13.fingerprint.contains("13"));

        // Different versions should produce different fingerprints
        assert_ne!(ja4.fingerprint, ja4_13.fingerprint);
    }

    #[test]
    fn test_sha256_truncated() {
        let input = "test_string";
        let hash = sha256_truncated(input, 12);

        // Should be exactly 12 characters
        assert_eq!(hash.len(), 12);

        // Should be consistent
        let hash2 = sha256_truncated(input, 12);
        assert_eq!(hash, hash2);

        // Different inputs should produce different hashes
        let hash3 = sha256_truncated("different_string", 12);
        assert_ne!(hash, hash3);
    }

    #[test]
    fn test_ja4_structure() {
        let hello = ClientHello::chrome_n2_stable("example.com");
        let ja4 = generate_ja4(&hello).unwrap();

        // JA4 should have the structure: q+ALPN+Version+CipherCount+ExtensionCount+ALPN_Extension+_+CiphersHash+ExtensionsHash
        let parts: Vec<&str> = ja4.fingerprint.split('_').collect();
        assert_eq!(parts.len(), 2); // Should be split by exactly one underscore

        // First part should start with 't'
        assert!(parts[0].starts_with('t'));

        // Second part should be the hashes (24 chars total: 12 + 12)
        assert_eq!(parts[1].len(), 24);
    }
}
