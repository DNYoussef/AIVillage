//! JA3/JA4 PCAP Generator for Betanet uTLS
//!
//! This tool generates TLS ClientHello packets with various fingerprints
//! and captures them in PCAP format for analysis and K-S testing.

use betanet_utls::{chrome::ChromeProfile, ja3, ja4, ClientHello, TlsFingerprint};
use bytes::{BufMut, BytesMut};
use std::fs::File;
use std::io::Write;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::net::TcpStream;
use tokio::time::{sleep, Duration};

/// PCAP global header
#[repr(C)]
struct PcapGlobalHeader {
    magic_number: u32,   // 0xa1b2c3d4 for microsecond resolution
    version_major: u16,  // 2
    version_minor: u16,  // 4
    thiszone: i32,       // GMT offset (0)
    sigfigs: u32,        // Timestamp accuracy (0)
    snaplen: u32,        // Max packet length (65535)
    network: u32,        // Data link type (1 = Ethernet)
}

/// PCAP packet header
#[repr(C)]
struct PcapPacketHeader {
    ts_sec: u32,   // Timestamp seconds
    ts_usec: u32,  // Timestamp microseconds
    incl_len: u32, // Number of bytes saved
    orig_len: u32, // Original packet length
}

/// Test configuration for different browser profiles
struct BrowserProfile {
    name: String,
    hostname: String,
    generate_hello: Box<dyn Fn() -> ClientHello>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Betanet uTLS - JA3/JA4 PCAP Generator");
    println!("=========================================");

    // Create output directory
    std::fs::create_dir_all("artifacts/pcaps")?;

    // Define test profiles
    let profiles = vec![
        BrowserProfile {
            name: "Chrome_N2_Stable".to_string(),
            hostname: "www.google.com".to_string(),
            generate_hello: Box::new(|| ClientHello::chrome_n2_stable("www.google.com")),
        },
        BrowserProfile {
            name: "Chrome_N2_Beta".to_string(),
            hostname: "www.facebook.com".to_string(),
            generate_hello: Box::new(|| ClientHello::chrome_n2_stable("www.facebook.com")),
        },
        BrowserProfile {
            name: "Chrome_Custom_1".to_string(),
            hostname: "www.amazon.com".to_string(),
            generate_hello: Box::new(|| {
                let mut hello = ClientHello::chrome_n2_stable("www.amazon.com");
                // Modify to create variation
                hello.session_id = vec![0x42; 32];
                hello
            }),
        },
        BrowserProfile {
            name: "Chrome_Custom_2".to_string(),
            hostname: "www.twitter.com".to_string(),
            generate_hello: Box::new(|| {
                let mut hello = ClientHello::chrome_n2_stable("www.twitter.com");
                // Add additional extension for variation
                hello.extensions.push(betanet_utls::extensions::Extension {
                    extension_type: 0x0017, // status_request_v2
                    extension_data: vec![],
                });
                hello
            }),
        },
        BrowserProfile {
            name: "Chrome_NoGrease".to_string(),
            hostname: "www.github.com".to_string(),
            generate_hello: Box::new(|| {
                let mut hello = ClientHello::chrome_n2_stable("www.github.com");
                // Remove GREASE values
                hello.cipher_suites.retain(|&c| !betanet_utls::grease::GREASE_VALUES.contains(&c));
                hello.extensions.retain(|e| !betanet_utls::grease::GREASE_VALUES.contains(&e.extension_type));
                hello
            }),
        },
    ];

    // Generate individual PCAPs
    let mut all_fingerprints = Vec::new();

    for (i, profile) in profiles.iter().enumerate() {
        println!("\n📦 Generating PCAP for: {}", profile.name);

        let hello = (profile.generate_hello)();

        // Calculate fingerprints
        let ja3_fp = ja3::generate_ja3(&hello)?;
        let ja4_fp = ja4::generate_ja4(&hello)?;

        println!("  JA3 Hash: {}", ja3_fp.hash);
        println!("  JA4 Hash: {}", ja4_fp.hash);

        all_fingerprints.push((profile.name.clone(), ja3_fp.clone(), ja4_fp.clone()));

        // Generate PCAP
        let pcap_file = format!("artifacts/pcaps/{}_{}.pcap", profile.name, i);
        generate_pcap(&hello, &profile.hostname, &pcap_file)?;
        println!("  ✅ PCAP saved: {}", pcap_file);

        // Also generate raw ClientHello for analysis
        let raw_file = format!("artifacts/pcaps/{}_{}_clienthello.bin", profile.name, i);
        let raw_data = encode_client_hello(&hello)?;
        std::fs::write(&raw_file, &raw_data)?;
        println!("  ✅ Raw ClientHello saved: {}", raw_file);
    }

    // Generate combined PCAP with all profiles
    println!("\n📦 Generating combined PCAP with all profiles...");
    generate_combined_pcap(&profiles, "artifacts/pcaps/combined_ja3_ja4.pcap")?;

    // Generate fingerprint analysis report
    generate_fingerprint_report(&all_fingerprints)?;

    // Generate K-S test data
    generate_ks_test_data(&profiles)?;

    println!("\n✅ PCAP generation complete!");
    println!("\n📊 Generated Artifacts:");
    println!("  • Individual PCAPs: artifacts/pcaps/*.pcap");
    println!("  • Combined PCAP: artifacts/pcaps/combined_ja3_ja4.pcap");
    println!("  • Fingerprint Report: artifacts/pcaps/fingerprint_analysis.json");
    println!("  • K-S Test Data: artifacts/pcaps/ks_test_data.json");

    Ok(())
}

/// Generate a PCAP file with TLS ClientHello
fn generate_pcap(hello: &ClientHello, hostname: &str, output_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut pcap_file = File::create(output_path)?;

    // Write PCAP global header
    let global_header = PcapGlobalHeader {
        magic_number: 0xa1b2c3d4,
        version_major: 2,
        version_minor: 4,
        thiszone: 0,
        sigfigs: 0,
        snaplen: 65535,
        network: 1, // Ethernet
    };

    unsafe {
        let header_bytes = std::slice::from_raw_parts(
            &global_header as *const _ as *const u8,
            std::mem::size_of::<PcapGlobalHeader>(),
        );
        pcap_file.write_all(header_bytes)?;
    }

    // Create TLS ClientHello packet
    let tls_packet = create_tls_packet(hello, hostname)?;

    // Get timestamp
    let now = SystemTime::now().duration_since(UNIX_EPOCH)?;

    // Write packet header
    let packet_header = PcapPacketHeader {
        ts_sec: now.as_secs() as u32,
        ts_usec: now.subsec_micros(),
        incl_len: tls_packet.len() as u32,
        orig_len: tls_packet.len() as u32,
    };

    unsafe {
        let header_bytes = std::slice::from_raw_parts(
            &packet_header as *const _ as *const u8,
            std::mem::size_of::<PcapPacketHeader>(),
        );
        pcap_file.write_all(header_bytes)?;
    }

    // Write packet data
    pcap_file.write_all(&tls_packet)?;

    Ok(())
}

/// Create a complete TLS packet with Ethernet, IP, TCP headers
fn create_tls_packet(hello: &ClientHello, hostname: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut packet = BytesMut::with_capacity(2048);

    // Ethernet header (14 bytes)
    packet.put(&[0x00, 0x11, 0x22, 0x33, 0x44, 0x55][..]); // Destination MAC
    packet.put(&[0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb][..]); // Source MAC
    packet.put_u16(0x0800); // EtherType: IPv4

    // IP header (20 bytes)
    packet.put_u8(0x45); // Version (4) + IHL (5)
    packet.put_u8(0x00); // TOS
    let ip_total_length_offset = packet.len();
    packet.put_u16(0); // Total length (placeholder)
    packet.put_u16(0x1234); // ID
    packet.put_u16(0x4000); // Flags + Fragment offset
    packet.put_u8(64); // TTL
    packet.put_u8(6); // Protocol: TCP
    packet.put_u16(0); // Checksum (placeholder)
    packet.put(&[192, 168, 1, 100][..]); // Source IP
    packet.put(&[93, 184, 216, 34][..]); // Destination IP (example.com)

    // TCP header (20 bytes)
    packet.put_u16(54321); // Source port
    packet.put_u16(443); // Destination port (HTTPS)
    packet.put_u32(0x12345678); // Sequence number
    packet.put_u32(0); // Acknowledgment number
    packet.put_u8(0x50); // Data offset (5) + Reserved
    packet.put_u8(0x18); // Flags: PSH + ACK
    packet.put_u16(8192); // Window
    packet.put_u16(0); // Checksum (placeholder)
    packet.put_u16(0); // Urgent pointer

    // TLS Record Layer
    packet.put_u8(0x16); // Content Type: Handshake
    packet.put_u16(0x0303); // TLS Version 1.2
    let tls_length_offset = packet.len();
    packet.put_u16(0); // Length (placeholder)

    // Handshake Protocol
    packet.put_u8(0x01); // Handshake Type: ClientHello
    let handshake_length_offset = packet.len();
    packet.put(&[0, 0, 0][..]); // Length (24-bit, placeholder)

    // Encode ClientHello
    let hello_start = packet.len();
    encode_client_hello_to_buf(hello, &mut packet)?;
    let hello_end = packet.len();

    // Update lengths
    let handshake_length = (hello_end - hello_start) as u32;
    packet[handshake_length_offset] = ((handshake_length >> 16) & 0xFF) as u8;
    packet[handshake_length_offset + 1] = ((handshake_length >> 8) & 0xFF) as u8;
    packet[handshake_length_offset + 2] = (handshake_length & 0xFF) as u8;

    let tls_length = (hello_end - hello_start + 4) as u16; // +4 for handshake header
    packet[tls_length_offset] = ((tls_length >> 8) & 0xFF) as u8;
    packet[tls_length_offset + 1] = (tls_length & 0xFF) as u8;

    let ip_total_length = (packet.len() - 14) as u16; // Subtract Ethernet header
    packet[ip_total_length_offset] = ((ip_total_length >> 8) & 0xFF) as u8;
    packet[ip_total_length_offset + 1] = (ip_total_length & 0xFF) as u8;

    Ok(packet.to_vec())
}

/// Encode ClientHello structure
fn encode_client_hello(hello: &ClientHello) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut buf = BytesMut::with_capacity(1024);
    encode_client_hello_to_buf(hello, &mut buf)?;
    Ok(buf.to_vec())
}

/// Encode ClientHello to buffer
fn encode_client_hello_to_buf(hello: &ClientHello, buf: &mut BytesMut) -> Result<(), Box<dyn std::error::Error>> {
    // TLS Version
    buf.put_u16(hello.version);

    // Random (32 bytes)
    buf.put(&hello.random[..]);

    // Session ID
    buf.put_u8(hello.session_id.len() as u8);
    buf.put(&hello.session_id[..]);

    // Cipher Suites
    buf.put_u16((hello.cipher_suites.len() * 2) as u16);
    for cipher in &hello.cipher_suites {
        buf.put_u16(*cipher);
    }

    // Compression Methods
    buf.put_u8(hello.compression_methods.len() as u8);
    buf.put(&hello.compression_methods[..]);

    // Extensions
    let ext_length_offset = buf.len();
    buf.put_u16(0); // Placeholder

    let ext_start = buf.len();
    for ext in &hello.extensions {
        buf.put_u16(ext.extension_type);
        buf.put_u16(ext.extension_data.len() as u16);
        buf.put(&ext.extension_data[..]);
    }
    let ext_end = buf.len();

    let ext_length = (ext_end - ext_start) as u16;
    buf[ext_length_offset] = ((ext_length >> 8) & 0xFF) as u8;
    buf[ext_length_offset + 1] = (ext_length & 0xFF) as u8;

    Ok(())
}

/// Generate combined PCAP with all profiles
fn generate_combined_pcap(profiles: &[BrowserProfile], output_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut pcap_file = File::create(output_path)?;

    // Write PCAP global header
    let global_header = PcapGlobalHeader {
        magic_number: 0xa1b2c3d4,
        version_major: 2,
        version_minor: 4,
        thiszone: 0,
        sigfigs: 0,
        snaplen: 65535,
        network: 1,
    };

    unsafe {
        let header_bytes = std::slice::from_raw_parts(
            &global_header as *const _ as *const u8,
            std::mem::size_of::<PcapGlobalHeader>(),
        );
        pcap_file.write_all(header_bytes)?;
    }

    // Generate packets for each profile
    for (i, profile) in profiles.iter().enumerate() {
        let hello = (profile.generate_hello)();
        let tls_packet = create_tls_packet(&hello, &profile.hostname)?;

        let now = SystemTime::now().duration_since(UNIX_EPOCH)?;
        let packet_header = PcapPacketHeader {
            ts_sec: (now.as_secs() + i as u64) as u32, // Space packets 1 second apart
            ts_usec: now.subsec_micros(),
            incl_len: tls_packet.len() as u32,
            orig_len: tls_packet.len() as u32,
        };

        unsafe {
            let header_bytes = std::slice::from_raw_parts(
                &packet_header as *const _ as *const u8,
                std::mem::size_of::<PcapPacketHeader>(),
            );
            pcap_file.write_all(header_bytes)?;
        }

        pcap_file.write_all(&tls_packet)?;
    }

    Ok(())
}

/// Generate fingerprint analysis report
fn generate_fingerprint_report(fingerprints: &[(String, TlsFingerprint, TlsFingerprint)]) -> Result<(), Box<dyn std::error::Error>> {
    use serde_json::json;

    let mut report = json!({
        "timestamp": SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        "profiles": []
    });

    for (name, ja3, ja4) in fingerprints {
        report["profiles"].as_array_mut().unwrap().push(json!({
            "name": name,
            "ja3": {
                "string": ja3.raw_string,
                "hash": ja3.hash,
            },
            "ja4": {
                "string": ja4.raw_string,
                "hash": ja4.hash,
            }
        }));
    }

    let report_str = serde_json::to_string_pretty(&report)?;
    std::fs::write("artifacts/pcaps/fingerprint_analysis.json", report_str)?;

    Ok(())
}

/// Generate K-S test data for statistical analysis
fn generate_ks_test_data(profiles: &[BrowserProfile]) -> Result<(), Box<dyn std::error::Error>> {
    use serde_json::json;

    let mut distributions = Vec::new();

    // Generate multiple samples for each profile
    for profile in profiles {
        let mut ja3_hashes = Vec::new();
        let mut ja4_hashes = Vec::new();

        // Generate 100 samples with slight variations
        for i in 0..100 {
            let mut hello = (profile.generate_hello)();

            // Add random variation
            hello.random = {
                let mut random = [0u8; 32];
                let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos() as u64;
                for (j, byte) in random.iter_mut().enumerate() {
                    *byte = ((timestamp + i as u64 + j as u64) & 0xFF) as u8;
                }
                random
            };

            let ja3_fp = ja3::generate_ja3(&hello)?;
            let ja4_fp = ja4::generate_ja4(&hello)?;

            ja3_hashes.push(ja3_fp.hash);
            ja4_hashes.push(ja4_fp.hash);
        }

        distributions.push(json!({
            "profile": profile.name,
            "samples": 100,
            "ja3_distribution": ja3_hashes,
            "ja4_distribution": ja4_hashes,
            "ja3_unique": ja3_hashes.iter().collect::<std::collections::HashSet<_>>().len(),
            "ja4_unique": ja4_hashes.iter().collect::<std::collections::HashSet<_>>().len(),
        }));
    }

    let ks_data = json!({
        "timestamp": SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        "test_type": "Kolmogorov-Smirnov",
        "distributions": distributions,
    });

    let ks_str = serde_json::to_string_pretty(&ks_data)?;
    std::fs::write("artifacts/pcaps/ks_test_data.json", ks_str)?;

    Ok(())
}
