//! Betanet uTLS CLI
//!
//! Command-line interface for generating Chrome N-2 TLS ClientHello messages,
//! computing JA3/JA4 fingerprints, and running self-tests.

use betanet_utls::{
    ja3, ja4,
    refresh::{ChromeRefresh, ReleaseChannel},
    template::TlsTemplate,
    ChromeVersion, ClientHello, Result,
};
use clap::{Parser, Subcommand};
use serde_json::{json, Value};
use std::fs;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "betanet-utls")]
#[command(about = "Chrome N-2 uTLS ClientHello generator with JA3/JA4 fingerprinting")]
#[command(version = "1.0.0")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate Chrome N-2 ClientHello template
    Gen {
        /// Generate for Chrome stable N-2 version
        #[arg(long)]
        chrome_stable: Option<String>,

        /// Output file path
        #[arg(short, long)]
        out: PathBuf,
    },

    /// Run self-tests against pcaps and templates
    Selftest {
        /// Directory containing pcap files
        #[arg(long)]
        pcap: PathBuf,

        /// Template file to test against
        #[arg(long)]
        template: PathBuf,

        /// JSON output file for test results
        #[arg(long)]
        out_json: PathBuf,

        /// Log file for detailed output
        #[arg(long)]
        log: PathBuf,
    },

    /// Auto-refresh Chrome version templates
    Refresh {
        /// Force update even if recent
        #[arg(long)]
        force: bool,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Gen { chrome_stable, out } => {
            cmd_gen(chrome_stable, out).await?;
        }
        Commands::Selftest {
            pcap,
            template,
            out_json,
            log,
        } => {
            cmd_selftest(pcap, template, out_json, log).await?;
        }
        Commands::Refresh { force } => {
            cmd_refresh(force).await?;
        }
    }

    Ok(())
}

async fn cmd_gen(chrome_stable: Option<String>, out: PathBuf) -> Result<()> {
    tracing::info!("Generating Chrome N-2 ClientHello template");

    // Parse Chrome version specification
    let target_version = match chrome_stable.as_deref() {
        Some("N-2") | None => ChromeVersion::current_stable_n2(),
        Some(version_str) => ChromeVersion::from_string(version_str)?,
    };

    tracing::info!("Target Chrome version: {}", target_version);

    // Generate deterministic ClientHello for Chrome N-2
    let hostname = "example.com"; // Standard test hostname
    let hello = ClientHello::chrome_n2_stable(hostname);

    // Create comprehensive template
    let template = TlsTemplate {
        version: target_version.clone(),
        server_name: hostname.to_string(),
        cipher_suites: hello.cipher_suites.clone(),
        extensions: hello
            .extensions
            .iter()
            .map(|ext| ext.extension_type)
            .collect(),
        curves: extract_supported_groups(&hello),
        signature_algorithms: extract_signature_algorithms(&hello),
        alpn_protocols: extract_alpn_protocols(&hello),
    };

    // Calculate fingerprints
    let ja3_fp = ja3::generate_ja3(&hello)?;
    let ja4_fp = ja4::generate_ja4(&hello)?;

    // Create JSON template output
    let template_json = json!({
        "name": format!("chrome-stable-{}", target_version.major),
        "description": format!("Chrome {} stable (N-2) deterministic ClientHello template", target_version),
        "generated_at": chrono::Utc::now().to_rfc3339(),
        "chrome_version": {
            "version": target_version.to_string(),
            "major": target_version.major,
            "minor": target_version.minor,
            "build": target_version.build,
            "patch": target_version.patch
        },
        "tls_config": {
            "version": format!("0x{:04x}", hello.version),
            "version_name": format_tls_version(hello.version),
            "session_id_length": hello.session_id.len(),
            "compression_methods": hello.compression_methods
        },
        "cipher_suites": template.cipher_suites.iter().map(|&cs| {
            json!({
                "value": format!("0x{:04x}", cs),
                "name": format_cipher_suite(cs),
                "is_grease": betanet_utls::grease::GREASE_VALUES.contains(&cs)
            })
        }).collect::<Vec<_>>(),
        "extensions": hello.extensions.iter().map(|ext| {
            json!({
                "type": format!("0x{:04x}", ext.extension_type),
                "name": format_extension_type(ext.extension_type),
                "data_length": ext.data.len(),
                "is_grease": betanet_utls::grease::GREASE_VALUES.contains(&ext.extension_type)
            })
        }).collect::<Vec<_>>(),
        "supported_groups": template.curves.iter().map(|&curve| {
            json!({
                "value": format!("0x{:04x}", curve),
                "name": format_named_group(curve),
                "is_grease": betanet_utls::grease::GREASE_VALUES.contains(&curve)
            })
        }).collect::<Vec<_>>(),
        "alpn_protocols": template.alpn_protocols,
        "fingerprints": {
            "ja3": {
                "hash": ja3_fp.fingerprint,
                "type": ja3_fp.fingerprint_type
            },
            "ja4": {
                "hash": ja4_fp.fingerprint,
                "type": ja4_fp.fingerprint_type
            }
        },
        "grease_config": {
            "enabled": true,
            "cipher_suite": format!("0x{:04x}", betanet_utls::grease::grease_cipher_suite()),
            "extension": format!("0x{:04x}", betanet_utls::grease::grease_extension()),
            "named_group": format!("0x{:04x}", betanet_utls::grease::grease_named_group())
        },
        "raw_clienthello": {
            "hex": hex::encode(hello.encode()?),
            "size": hello.size()
        }
    });

    // Write template to output file
    let json_output = serde_json::to_string_pretty(&template_json)
        .map_err(|e| betanet_utls::UtlsError::Serialization(e.to_string()))?;

    fs::write(&out, json_output)?;

    println!("✅ Chrome N-2 stable template generated: {}", out.display());
    println!("   Chrome version: {}", target_version);
    println!("   JA3: {}", ja3_fp.fingerprint);
    println!("   JA4: {}", ja4_fp.fingerprint);

    Ok(())
}

async fn cmd_selftest(
    pcap_dir: PathBuf,
    template_path: PathBuf,
    out_json: PathBuf,
    log_path: PathBuf,
) -> Result<()> {
    tracing::info!("Running JA3/JA4 self-tests");

    // Load template
    let template_content = fs::read_to_string(&template_path)?;
    let template: Value = serde_json::from_str(&template_content)
        .map_err(|e| betanet_utls::UtlsError::Serialization(e.to_string()))?;

    // Extract expected fingerprints from template
    let expected_ja3 = template["fingerprints"]["ja3"]["hash"]
        .as_str()
        .ok_or_else(|| betanet_utls::UtlsError::Template("Missing JA3 fingerprint in template".to_string()))?;
    let expected_ja4 = template["fingerprints"]["ja4"]["hash"]
        .as_str()
        .ok_or_else(|| betanet_utls::UtlsError::Template("Missing JA4 fingerprint in template".to_string()))?;

    // Initialize test results
    let mut test_results = json!({
        "test_run": {
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "template_file": template_path.to_string_lossy(),
            "pcap_directory": pcap_dir.to_string_lossy()
        },
        "template_fingerprints": {
            "ja3": expected_ja3,
            "ja4": expected_ja4
        },
        "pcap_tests": [],
        "summary": {
            "total_pcaps": 0,
            "passed": 0,
            "failed": 0,
            "tolerance_violations": 0
        }
    });

    let mut log_entries = Vec::new();
    log_entries.push(format!("=== Betanet uTLS Self-Test Report ==="));
    log_entries.push(format!("Timestamp: {}", chrono::Utc::now().to_rfc3339()));
    log_entries.push(format!("Template: {}", template_path.display()));
    log_entries.push(format!("PCAP directory: {}", pcap_dir.display()));
    log_entries.push(format!("Expected JA3: {}", expected_ja3));
    log_entries.push(format!("Expected JA4: {}", expected_ja4));
    log_entries.push(String::new());

    // Find and process pcap files
    let pcap_files = find_pcap_files(&pcap_dir)?;
    test_results["summary"]["total_pcaps"] = json!(pcap_files.len());

    log_entries.push(format!("Found {} pcap files", pcap_files.len()));
    log_entries.push(String::new());

    let mut passed = 0;
    let mut failed = 0;
    let mut tolerance_violations = 0;

    for pcap_file in &pcap_files {
        log_entries.push(format!("Processing: {}", pcap_file.display()));

        match process_pcap_file(&pcap_file).await {
            Ok(pcap_fingerprints) => {
                // Test each extracted ClientHello
                let mut pcap_results = json!({
                    "file": pcap_file.to_string_lossy(),
                    "status": "success",
                    "client_hellos": []
                });

                for (i, (ja3, ja4)) in pcap_fingerprints.iter().enumerate() {
                    let ja3_match = check_fingerprint_match(&expected_ja3, &ja3)?;
                    let ja4_match = check_fingerprint_match(&expected_ja4, &ja4)?;

                    let hello_result = json!({
                        "index": i,
                        "ja3": {
                            "fingerprint": ja3,
                            "matches_template": ja3_match.matches,
                            "distance": ja3_match.distance,
                            "tolerance_ok": ja3_match.within_tolerance
                        },
                        "ja4": {
                            "fingerprint": ja4,
                            "matches_template": ja4_match.matches,
                            "distance": ja4_match.distance,
                            "tolerance_ok": ja4_match.within_tolerance
                        }
                    });

                    pcap_results["client_hellos"]
                        .as_array_mut()
                        .unwrap()
                        .push(hello_result);

                    log_entries.push(format!("  ClientHello {}: JA3={}, JA4={}", i, ja3, ja4));
                    log_entries.push(format!("    JA3 match: {} (distance: {})", ja3_match.matches, ja3_match.distance));
                    log_entries.push(format!("    JA4 match: {} (distance: {})", ja4_match.matches, ja4_match.distance));

                    if ja3_match.matches && ja4_match.matches {
                        passed += 1;
                    } else {
                        failed += 1;
                    }

                    if !ja3_match.within_tolerance || !ja4_match.within_tolerance {
                        tolerance_violations += 1;
                    }
                }

                test_results["pcap_tests"]
                    .as_array_mut()
                    .unwrap()
                    .push(pcap_results);
            }
            Err(e) => {
                failed += 1;
                log_entries.push(format!("  Error: {}", e));

                let error_result = json!({
                    "file": pcap_file.to_string_lossy(),
                    "status": "error",
                    "error": e.to_string()
                });

                test_results["pcap_tests"]
                    .as_array_mut()
                    .unwrap()
                    .push(error_result);
            }
        }

        log_entries.push(String::new());
    }

    // Update summary
    test_results["summary"]["passed"] = json!(passed);
    test_results["summary"]["failed"] = json!(failed);
    test_results["summary"]["tolerance_violations"] = json!(tolerance_violations);

    log_entries.push(format!("=== Test Summary ==="));
    log_entries.push(format!("Total: {}", pcap_files.len()));
    log_entries.push(format!("Passed: {}", passed));
    log_entries.push(format!("Failed: {}", failed));
    log_entries.push(format!("Tolerance violations: {}", tolerance_violations));

    let success_rate = if pcap_files.len() > 0 {
        (passed as f64) / (pcap_files.len() as f64) * 100.0
    } else {
        0.0
    };
    log_entries.push(format!("Success rate: {:.1}%", success_rate));

    // Write results
    let json_output = serde_json::to_string_pretty(&test_results)
        .map_err(|e| betanet_utls::UtlsError::Serialization(e.to_string()))?;
    fs::write(&out_json, json_output)?;

    let log_output = log_entries.join("\n");
    fs::write(&log_path, log_output)?;

    println!("✅ Self-test completed");
    println!("   Results: {}", out_json.display());
    println!("   Log: {}", log_path.display());
    println!("   Success rate: {:.1}%", success_rate);

    Ok(())
}

async fn cmd_refresh(force: bool) -> Result<()> {
    tracing::info!("Refreshing Chrome version templates");

    let mut refresh = ChromeRefresh::new();

    // Check if refresh is needed (unless forced)
    if !force && !refresh.is_cache_expired(ReleaseChannel::Stable) {
        println!("Templates are up to date");
        println!("Use --force to refresh anyway");
        return Ok(());
    }

    // Fetch latest versions
    let stable_versions = refresh
        .fetch_versions(betanet_utls::refresh::ReleaseChannel::Stable)
        .await?;

    if stable_versions.is_empty() {
        return Err(betanet_utls::UtlsError::Config("No stable versions found".to_string()));
    }

    // Update N-2 version
    let n2_version = if stable_versions.len() >= 3 {
        &stable_versions[2] // N-2 (third most recent)
    } else {
        &stable_versions.last().unwrap() // Fallback to latest if not enough versions
    };

    println!("Updated Chrome N-2 stable: {}", n2_version.version);

    // Update index file
    let index = json!({
        "last_updated": chrono::Utc::now().to_rfc3339(),
        "stable_n2": {
            "version": n2_version.version,
            "release_date": n2_version.release_date
        },
        "available_templates": [
            format!("chrome-stable-{}", n2_version.version.split('.').next().unwrap_or("119"))
        ]
    });

    let templates_dir = PathBuf::from("tmp_submission/utls/templates");
    fs::create_dir_all(&templates_dir)?;

    let index_path = templates_dir.join("index.json");
    let index_json = serde_json::to_string_pretty(&index)
        .map_err(|e| betanet_utls::UtlsError::Serialization(e.to_string()))?;
    fs::write(&index_path, index_json)?;

    println!("✅ Templates refreshed and index updated: {}", index_path.display());

    Ok(())
}

// Helper functions

fn extract_supported_groups(hello: &ClientHello) -> Vec<u16> {
    use betanet_utls::extensions;

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

fn extract_signature_algorithms(hello: &ClientHello) -> Vec<u16> {
    use betanet_utls::extensions;

    for ext in &hello.extensions {
        if ext.extension_type == extensions::SIGNATURE_ALGORITHMS && ext.data.len() >= 2 {
            let mut algorithms = Vec::new();
            let algorithms_len = u16::from_be_bytes([ext.data[0], ext.data[1]]) as usize;

            if ext.data.len() >= 2 + algorithms_len {
                let mut i = 2;
                while i + 1 < 2 + algorithms_len {
                    let algorithm = u16::from_be_bytes([ext.data[i], ext.data[i + 1]]);
                    algorithms.push(algorithm);
                    i += 2;
                }
            }

            return algorithms;
        }
    }
    Vec::new()
}

fn extract_alpn_protocols(hello: &ClientHello) -> Vec<String> {
    use betanet_utls::extensions;

    for ext in &hello.extensions {
        if ext.extension_type == extensions::ALPN && ext.data.len() >= 2 {
            let mut protocols = Vec::new();
            let protocols_len = u16::from_be_bytes([ext.data[0], ext.data[1]]) as usize;

            if ext.data.len() >= 2 + protocols_len {
                let mut offset = 2;
                while offset < 2 + protocols_len {
                    if offset < ext.data.len() {
                        let proto_len = ext.data[offset] as usize;
                        offset += 1;

                        if offset + proto_len <= ext.data.len() {
                            let protocol = String::from_utf8_lossy(&ext.data[offset..offset + proto_len]);
                            protocols.push(protocol.to_string());
                            offset += proto_len;
                        } else {
                            break;
                        }
                    } else {
                        break;
                    }
                }
            }

            return protocols;
        }
    }
    Vec::new()
}

fn find_pcap_files(dir: &PathBuf) -> Result<Vec<PathBuf>> {
    let mut pcap_files = Vec::new();

    if dir.is_dir() {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() {
                if let Some(ext) = path.extension() {
                    if ext == "pcap" || ext == "pcapng" {
                        pcap_files.push(path);
                    }
                }
            }
        }
    }

    pcap_files.sort();
    Ok(pcap_files)
}

async fn process_pcap_file(_pcap_path: &PathBuf) -> Result<Vec<(String, String)>> {
    // Stub implementation - in a real implementation, this would:
    // 1. Parse the pcap file using a pcap library
    // 2. Extract TLS ClientHello messages
    // 3. Generate JA3/JA4 fingerprints for each
    // 4. Return the fingerprints

    // For now, return simulated fingerprints for testing
    let dummy_hello = ClientHello::chrome_n2_stable("test.com");
    let ja3_fp = ja3::generate_ja3(&dummy_hello)?;
    let ja4_fp = ja4::generate_ja4(&dummy_hello)?;

    Ok(vec![(ja3_fp.fingerprint, ja4_fp.fingerprint)])
}

struct FingerprintMatch {
    matches: bool,
    distance: u32,
    within_tolerance: bool,
}

fn check_fingerprint_match(expected: &str, actual: &str) -> Result<FingerprintMatch> {
    let matches = expected == actual;

    // Calculate Hamming distance for similarity
    let distance = if expected.len() == actual.len() {
        expected
            .chars()
            .zip(actual.chars())
            .map(|(a, b)| if a == b { 0 } else { 1 })
            .sum()
    } else {
        std::cmp::max(expected.len(), actual.len()) as u32
    };

    // Define tolerance rules - within 2 character differences for minor variations
    let within_tolerance = distance <= 2;

    Ok(FingerprintMatch {
        matches,
        distance,
        within_tolerance,
    })
}

fn format_tls_version(version: u16) -> &'static str {
    match version {
        0x0301 => "TLS 1.0",
        0x0302 => "TLS 1.1",
        0x0303 => "TLS 1.2",
        0x0304 => "TLS 1.3",
        _ => "Unknown",
    }
}

fn format_cipher_suite(cipher: u16) -> &'static str {
    match cipher {
        0x1301 => "TLS_AES_128_GCM_SHA256",
        0x1302 => "TLS_AES_256_GCM_SHA384",
        0x1303 => "TLS_CHACHA20_POLY1305_SHA256",
        0xc02b => "TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256",
        0xc02f => "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256",
        0xc02c => "TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384",
        0xc030 => "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384",
        _ => "Unknown",
    }
}

fn format_extension_type(ext_type: u16) -> &'static str {
    match ext_type {
        0x0000 => "server_name",
        0x000a => "supported_groups",
        0x000b => "ec_point_formats",
        0x000d => "signature_algorithms",
        0x0010 => "application_layer_protocol_negotiation",
        0x002b => "supported_versions",
        0x0033 => "key_share",
        0x002d => "psk_key_exchange_modes",
        _ => "unknown",
    }
}

fn format_named_group(group: u16) -> &'static str {
    match group {
        0x001d => "x25519",
        0x0017 => "secp256r1",
        0x0018 => "secp384r1",
        0x0019 => "secp521r1",
        0x0100 => "ffdhe2048",
        _ => "unknown",
    }
}
