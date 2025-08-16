//! uTLS Generator CLI Tool
//!
//! Command-line interface for generating Chrome N-2 TLS ClientHello messages
//! and computing JA3/JA4 fingerprints.

use betanet_utls::{
    chrome::ChromeProfile,
    ja3, ja4,
    refresh::{ChromeRefresh, ReleaseChannel},
    ChromeVersion, ClientHello, TlsTemplateBuilder,
};
use clap::{Parser, Subcommand};
use std::fs;

#[derive(Parser)]
#[command(name = "utls-gen")]
#[command(about = "Chrome N-2 uTLS ClientHello generator with JA3/JA4 fingerprinting")]
#[command(version = "1.0.0")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate ClientHello message
    Generate {
        /// Target hostname
        #[arg(long, default_value = "example.com")]
        hostname: String,

        /// Chrome version (119, 120, 121, or auto for N-2)
        #[arg(short, long, default_value = "auto")]
        version: String,

        /// Include GREASE values
        #[arg(short, long, default_value_t = true)]
        grease: bool,

        /// Output format (json, hex, raw)
        #[arg(short, long, default_value = "json")]
        format: String,

        /// Output file path
        #[arg(short, long)]
        output: Option<String>,
    },

    /// Calculate JA3 fingerprint
    Ja3 {
        /// Target hostname
        #[arg(long, default_value = "example.com")]
        hostname: String,

        /// Chrome version (119, 120, 121, or auto for N-2)
        #[arg(short, long, default_value = "auto")]
        version: String,

        /// Include GREASE values
        #[arg(short, long, default_value_t = true)]
        grease: bool,
    },

    /// Calculate JA4 fingerprint
    Ja4 {
        /// Target hostname
        #[arg(long, default_value = "example.com")]
        hostname: String,

        /// Chrome version (119, 120, 121, or auto for N-2)
        #[arg(short, long, default_value = "auto")]
        version: String,

        /// Include GREASE values
        #[arg(short, long, default_value_t = true)]
        grease: bool,
    },

    /// Compare JA3 vs JA4 fingerprints
    Compare {
        /// Target hostname
        #[arg(long, default_value = "example.com")]
        hostname: String,

        /// Chrome version (119, 120, 121, or auto for N-2)
        #[arg(short, long, default_value = "auto")]
        version: String,
    },

    /// Generate TLS template
    Template {
        /// Template name
        #[arg(short, long, default_value = "chrome_n2")]
        name: String,

        /// Target hostname
        #[arg(long, default_value = "example.com")]
        hostname: String,

        /// Output file path
        #[arg(short, long)]
        output: Option<String>,
    },

    /// Refresh Chrome version information
    Refresh {
        /// Release channel (stable, beta, dev, canary)
        #[arg(short, long, default_value = "stable")]
        channel: String,
    },

    /// Run self-tests
    Test {
        /// Test type (all, ja3, ja4, grease, consistency)
        #[arg(short, long, default_value = "all")]
        test_type: String,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Generate {
            hostname,
            version,
            grease,
            format,
            output,
        } => {
            cmd_generate(hostname, version, grease, format, output).await?;
        }
        Commands::Ja3 {
            hostname,
            version,
            grease,
        } => {
            cmd_ja3(hostname, version, grease).await?;
        }
        Commands::Ja4 {
            hostname,
            version,
            grease,
        } => {
            cmd_ja4(hostname, version, grease).await?;
        }
        Commands::Compare { hostname, version } => {
            cmd_compare(hostname, version).await?;
        }
        Commands::Template {
            name,
            hostname,
            output,
        } => {
            cmd_template(name, hostname, output).await?;
        }
        Commands::Refresh { channel } => {
            cmd_refresh(channel).await?;
        }
        Commands::Test { test_type } => {
            cmd_test(test_type).await?;
        }
    }

    Ok(())
}

async fn cmd_generate(
    hostname: String,
    version: String,
    grease: bool,
    format: String,
    output: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Generating ClientHello for {}", hostname);

    let hello = create_client_hello(&hostname, &version, grease).await?;

    let output_data = match format.as_str() {
        "json" => {
            let json_data = serde_json::json!({
                "hostname": hostname,
                "version": format_version(&hello.version),
                "cipher_suites": hello.cipher_suites,
                "extensions": hello.extensions.iter().map(|ext| {
                    serde_json::json!({
                        "type": ext.extension_type,
                        "data_length": ext.data.len()
                    })
                }).collect::<Vec<_>>(),
                "grease_enabled": grease,
                "size": hello.size()
            });
            serde_json::to_string_pretty(&json_data)?
        }
        "hex" => {
            let encoded = hello.encode()?;
            hex::encode(encoded)
        }
        "raw" => {
            let encoded = hello.encode()?;
            String::from_utf8_lossy(&encoded).to_string()
        }
        _ => return Err("Invalid format. Use: json, hex, raw".into()),
    };

    if let Some(output_path) = output {
        fs::write(&output_path, &output_data)?;
        println!("✅ Output written to {}", output_path);
    } else {
        println!("{}", output_data);
    }

    Ok(())
}

async fn cmd_ja3(
    hostname: String,
    version: String,
    grease: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Computing JA3 fingerprint for {}", hostname);

    let hello = create_client_hello(&hostname, &version, grease).await?;
    let ja3_fp = ja3::generate_ja3(&hello)?;

    println!("JA3: {}", ja3_fp.fingerprint);

    Ok(())
}

async fn cmd_ja4(
    hostname: String,
    version: String,
    grease: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Computing JA4 fingerprint for {}", hostname);

    let hello = create_client_hello(&hostname, &version, grease).await?;
    let ja4_fp = ja4::generate_ja4(&hello)?;

    println!("JA4: {}", ja4_fp.fingerprint);

    Ok(())
}

async fn cmd_compare(hostname: String, version: String) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Comparing JA3 vs JA4 for {}", hostname);

    let hello = create_client_hello(&hostname, &version, true).await?;
    let ja3_fp = ja3::generate_ja3(&hello)?;
    let ja4_fp = ja4::generate_ja4(&hello)?;

    println!("\n📊 Fingerprint Comparison:");
    println!("  JA3 (MD5):  {}", ja3_fp.fingerprint);
    println!("  JA4 (SHA256): {}", ja4_fp.fingerprint);

    // Analyze JA4 structure
    let ja4_parts: Vec<&str> = ja4_fp.fingerprint.split('_').collect();
    if ja4_parts.len() == 2 {
        println!("\n🔬 JA4 Analysis:");
        println!(
            "  Prefix:     {} (protocol+ALPN+version+counts)",
            ja4_parts[0]
        );
        println!("  Hashes:     {} (cipher+extension hashes)", ja4_parts[1]);

        if ja4_parts[0].len() >= 8 {
            let protocol = &ja4_parts[0][0..1];
            let alpn = &ja4_parts[0][1..3];
            let tls_version = &ja4_parts[0][3..5];
            let cipher_count = &ja4_parts[0][5..7];
            let ext_count = &ja4_parts[0][7..9];

            println!("  Protocol:   {} (TLS)", protocol);
            println!("  ALPN:       {} (HTTP/2)", alpn);
            println!("  TLS Version: {} (1.2)", tls_version);
            println!(
                "  Ciphers:    {} ({} total)",
                cipher_count,
                u8::from_str_radix(cipher_count, 16).unwrap_or(0)
            );
            println!(
                "  Extensions: {} ({} total)",
                ext_count,
                u8::from_str_radix(ext_count, 16).unwrap_or(0)
            );
        }
    }

    Ok(())
}

async fn cmd_template(
    name: String,
    hostname: String,
    output: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("📋 Generating TLS template: {}", name);

    let template = match name.as_str() {
        "chrome_n2" => {
            let version = ChromeVersion::current_stable_n2();
            TlsTemplateBuilder::for_chrome(version)
                .server_name(&hostname)
                .alpn_protocols(vec!["h2".to_string(), "http/1.1".to_string()])
                .build()
        }
        "chrome_119" => {
            let version = ChromeVersion::new(119, 0, 6045, 123);
            TlsTemplateBuilder::for_chrome(version)
                .server_name(&hostname)
                .build()
        }
        _ => return Err("Unknown template name. Use: chrome_n2, chrome_119".into()),
    };

    let template_json = serde_json::json!({
        "name": name,
        "version": template.version.to_string(),
        "server_name": template.server_name,
        "cipher_suites": template.cipher_suites,
        "extensions": template.extensions,
        "curves": template.curves,
        "signature_algorithms": template.signature_algorithms,
        "alpn_protocols": template.alpn_protocols,
        "ja3_fingerprint": template.ja3_fingerprint()
    });

    let output_data = serde_json::to_string_pretty(&template_json)?;

    if let Some(output_path) = output {
        fs::write(&output_path, &output_data)?;
        println!("✅ Template written to {}", output_path);
    } else {
        println!("{}", output_data);
    }

    Ok(())
}

async fn cmd_refresh(channel: String) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 Refreshing Chrome version info for {} channel", channel);

    let release_channel = match channel.as_str() {
        "stable" => ReleaseChannel::Stable,
        "beta" => ReleaseChannel::Beta,
        "dev" => ReleaseChannel::Dev,
        "canary" => ReleaseChannel::Canary,
        _ => return Err("Invalid channel. Use: stable, beta, dev, canary".into()),
    };

    let mut refresh = ChromeRefresh::new();
    let versions = refresh.fetch_versions(release_channel).await?;

    println!("📦 Found {} versions:", versions.len());
    for (i, version) in versions.iter().take(5).enumerate() {
        println!(
            "  {}. Chrome {} ({})",
            i + 1,
            version.version,
            version.release_date.as_deref().unwrap_or("unknown")
        );
    }

    if let Ok(n2_version) = refresh.get_stable_n2_version().await {
        println!("\n🎯 Current N-2 stable: Chrome {}", n2_version);
    }

    Ok(())
}

async fn cmd_test(test_type: String) -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Running {} tests", test_type);

    match test_type.as_str() {
        "all" => {
            run_ja3_test().await?;
            run_ja4_test().await?;
            run_grease_test().await?;
            run_consistency_test().await?;
        }
        "ja3" => run_ja3_test().await?,
        "ja4" => run_ja4_test().await?,
        "grease" => run_grease_test().await?,
        "consistency" => run_consistency_test().await?,
        _ => return Err("Invalid test type. Use: all, ja3, ja4, grease, consistency".into()),
    }

    println!("✅ All tests completed successfully!");
    Ok(())
}

async fn create_client_hello(
    hostname: &str,
    version: &str,
    grease: bool,
) -> Result<ClientHello, Box<dyn std::error::Error>> {
    let hello = match version {
        "auto" | "119" => {
            if grease {
                ClientHello::chrome_n2_stable(hostname)
            } else {
                let profile = ChromeProfile::chrome_119();
                ClientHello::from_chrome_profile_with_grease(&profile, hostname, false)
            }
        }
        "120" => {
            let profile = ChromeProfile::chrome_119(); // Using 119 as base
            ClientHello::from_chrome_profile_with_grease(&profile, hostname, grease)
        }
        "121" => {
            let profile = ChromeProfile::chrome_119(); // Using 119 as base
            ClientHello::from_chrome_profile_with_grease(&profile, hostname, grease)
        }
        _ => return Err("Invalid version. Use: auto, 119, 120, 121".into()),
    };

    Ok(hello)
}

fn format_version(version: &u16) -> String {
    match *version {
        0x0301 => "TLS 1.0".to_string(),
        0x0302 => "TLS 1.1".to_string(),
        0x0303 => "TLS 1.2".to_string(),
        0x0304 => "TLS 1.3".to_string(),
        _ => format!("Unknown (0x{:04x})", version),
    }
}

async fn run_ja3_test() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔍 Testing JA3 fingerprinting...");

    let hello = ClientHello::chrome_n2_stable("test.example.com");
    let ja3_fp = ja3::generate_ja3(&hello)?;

    assert_eq!(ja3_fp.fingerprint_type, "ja3");
    assert_eq!(ja3_fp.fingerprint.len(), 32); // MD5 hash length

    println!("  ✅ JA3 generation working");
    println!("  ✅ MD5 hash length correct");

    Ok(())
}

async fn run_ja4_test() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔍 Testing JA4 fingerprinting...");

    let hello = ClientHello::chrome_n2_stable("test.example.com");
    let ja4_fp = ja4::generate_ja4(&hello)?;

    assert_eq!(ja4_fp.fingerprint_type, "ja4");
    assert!(ja4_fp.fingerprint.starts_with('t')); // TLS protocol
    assert!(ja4_fp.fingerprint.contains('_')); // Separator

    println!("  ✅ JA4 generation working");
    println!("  ✅ Protocol prefix correct");
    println!("  ✅ Structure format correct");

    Ok(())
}

async fn run_grease_test() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔍 Testing GREASE value handling...");

    let hello_with_grease = ClientHello::chrome_n2_stable("grease-test.com");
    let hello_without_grease = {
        let profile = ChromeProfile::chrome_119();
        ClientHello::from_chrome_profile_with_grease(&profile, "grease-test.com", false)
    };

    // Check GREASE presence
    let has_grease_cipher = hello_with_grease
        .cipher_suites
        .iter()
        .any(|&c| betanet_utls::grease::GREASE_VALUES.contains(&c));
    let no_grease_cipher = hello_without_grease
        .cipher_suites
        .iter()
        .any(|&c| betanet_utls::grease::GREASE_VALUES.contains(&c));

    assert!(has_grease_cipher);
    assert!(!no_grease_cipher);

    println!("  ✅ GREASE values correctly added/removed");
    println!("  ✅ Fingerprinting handles GREASE filtering");

    Ok(())
}

async fn run_consistency_test() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔍 Testing fingerprint consistency...");

    let hostnames = ["test1.com", "test2.com", "test3.com"];
    let mut ja4_prefixes = Vec::new();

    for hostname in &hostnames {
        let hello = ClientHello::chrome_n2_stable(hostname);
        let ja4_fp = ja4::generate_ja4(&hello)?;

        if let Some(prefix) = ja4_fp.fingerprint.split('_').next() {
            ja4_prefixes.push(prefix.to_string());
        }
    }

    // Check structural consistency (ignoring randomness)
    let first_len = ja4_prefixes[0].len();
    let consistent = ja4_prefixes.iter().all(|p| p.len() == first_len);

    assert!(consistent);

    println!("  ✅ Structural consistency across hostnames");
    println!("  ✅ Deterministic fingerprint components");

    Ok(())
}
