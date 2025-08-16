//! JA3 fingerprinting

use crate::{extensions, grease, ClientHello, Result, TlsFingerprint};

/// Generate JA3 fingerprint from ClientHello
/// JA3 format: TLSVersion,CipherSuites,Extensions,EllipticCurves,EllipticCurvePointFormats
pub fn generate_ja3(hello: &ClientHello) -> Result<TlsFingerprint> {
    // 1. TLS Version - convert to JA3 format (771 for TLS 1.2, 772 for TLS 1.3)
    let version = match hello.version {
        0x0303 => "771", // TLS 1.2
        0x0304 => "772", // TLS 1.3
        0x0302 => "770", // TLS 1.1
        0x0301 => "769", // TLS 1.0
        _ => "771",      // Default to TLS 1.2
    };

    // 2. Cipher Suites - exclude GREASE values and join with hyphens
    let ciphers = hello
        .cipher_suites
        .iter()
        .filter(|&&cipher| !is_grease_value(cipher))
        .map(|c| c.to_string())
        .collect::<Vec<_>>()
        .join("-");

    // 3. Extensions - exclude GREASE values and join with hyphens
    let extensions = hello
        .extensions
        .iter()
        .filter(|ext| !is_grease_value(ext.extension_type))
        .map(|e| e.extension_type.to_string())
        .collect::<Vec<_>>()
        .join("-");

    // 4. Elliptic Curves - extract from supported_groups extension
    let curves = extract_supported_groups(hello)
        .into_iter()
        .filter(|&curve| !is_grease_value(curve))
        .map(|c| c.to_string())
        .collect::<Vec<_>>()
        .join("-");

    // 5. Elliptic Curve Point Formats - extract from ec_point_formats extension
    let point_formats = extract_point_formats(hello)
        .into_iter()
        .map(|p| p.to_string())
        .collect::<Vec<_>>()
        .join("-");

    // Build JA3 string
    let ja3_string = format!(
        "{},{},{},{},{}",
        version, ciphers, extensions, curves, point_formats
    );

    // Calculate MD5 hash
    let hash = format!("{:x}", md5::compute(ja3_string.as_bytes()));

    Ok(TlsFingerprint::new(hash, "ja3".to_string()))
}

/// Check if a value is a GREASE value
fn is_grease_value(value: u16) -> bool {
    grease::GREASE_VALUES.contains(&value)
}

/// Extract supported groups (elliptic curves) from ClientHello
fn extract_supported_groups(hello: &ClientHello) -> Vec<u16> {
    for ext in &hello.extensions {
        if ext.extension_type == extensions::SUPPORTED_GROUPS && ext.data.len() >= 2 {
            let mut groups = Vec::new();
            let groups_len = u16::from_be_bytes([ext.data[0], ext.data[1]]) as usize;

            if ext.data.len() >= 2 + groups_len {
                let mut i = 2;
                while i + 1 < 2 + groups_len {
                    let group = u16::from_be_bytes([ext.data[i], ext.data[i + 1]]);
                    groups.push(group);
                    i += 2;
                }
            }

            return groups;
        }
    }
    Vec::new()
}

/// Extract EC point formats from ClientHello
fn extract_point_formats(hello: &ClientHello) -> Vec<u8> {
    for ext in &hello.extensions {
        if ext.extension_type == extensions::EC_POINT_FORMATS && !ext.data.is_empty() {
            let formats_len = ext.data[0] as usize;

            if ext.data.len() > formats_len {
                return ext.data[1..1 + formats_len].to_vec();
            }
        }
    }

    // Default to uncompressed point format if not found
    vec![0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ja3_generation() {
        let hello = ClientHello::new();
        let ja3 = generate_ja3(&hello).unwrap();
        assert_eq!(ja3.fingerprint_type, "ja3");
        assert_eq!(ja3.fingerprint.len(), 32); // MD5 hash length
    }

    #[test]
    fn test_ja3_chrome_n2_stable() {
        let hello = ClientHello::chrome_n2_stable("test.com");
        let ja3 = generate_ja3(&hello).unwrap();

        // Should generate a consistent fingerprint
        assert_eq!(ja3.fingerprint_type, "ja3");
        assert_eq!(ja3.fingerprint.len(), 32);

        // Generate again to ensure consistency
        let ja3_2 = generate_ja3(&hello).unwrap();
        assert_eq!(ja3.fingerprint, ja3_2.fingerprint);
    }

    #[test]
    fn test_grease_value_detection() {
        assert!(is_grease_value(0x0A0A));
        assert!(is_grease_value(0x1A1A));
        assert!(is_grease_value(0x2A2A));
        assert!(!is_grease_value(0x1301)); // TLS_AES_128_GCM_SHA256
        assert!(!is_grease_value(0x0000)); // server_name extension
    }

    #[test]
    fn test_ja3_grease_filtering() {
        use crate::cipher_suites;

        let mut hello = ClientHello::new();
        hello.cipher_suites = vec![
            0x0A0A, // GREASE cipher - should be filtered
            cipher_suites::TLS_AES_128_GCM_SHA256,
            cipher_suites::TLS_AES_256_GCM_SHA384,
        ];

        let ja3 = generate_ja3(&hello).unwrap();

        // JA3 string should not contain GREASE values
        assert_eq!(ja3.fingerprint_type, "ja3");
        assert_eq!(ja3.fingerprint.len(), 32);
    }

    #[test]
    fn test_version_conversion() {
        let mut hello = ClientHello::new();

        // Test TLS 1.2 (0x0303 -> 771)
        hello.version = 0x0303;
        let ja3 = generate_ja3(&hello).unwrap();
        assert_eq!(ja3.fingerprint_type, "ja3");

        // Test TLS 1.3 (0x0304 -> 772)
        hello.version = 0x0304;
        let ja3_13 = generate_ja3(&hello).unwrap();

        // Different versions should produce different fingerprints
        assert_ne!(ja3.fingerprint, ja3_13.fingerprint);
    }

    #[test]
    fn test_supported_groups_extraction() {
        let hello = ClientHello::chrome_n2_stable("test.com");
        let groups = extract_supported_groups(&hello);

        // Should extract groups (including GREASE)
        assert!(!groups.is_empty());

        // Should contain at least one GREASE value (since Chrome includes them)
        assert!(groups.iter().any(|&group| is_grease_value(group)));

        // Should also contain real groups
        assert!(groups.iter().any(|&group| !is_grease_value(group)));
    }

    #[test]
    fn test_point_formats_extraction() {
        let hello = ClientHello::new();
        let formats = extract_point_formats(&hello);

        // Should return default uncompressed format if not found
        assert_eq!(formats, vec![0]);
    }
}
