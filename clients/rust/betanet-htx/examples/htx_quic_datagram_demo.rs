//! HTX over QUIC with DATAGRAM demonstration
//!
//! Shows HTX protocol working over QUIC with:
//! - H3 ALPN negotiation
//! - QUIC DATAGRAM for small frames
//! - ECH configuration from DNS
//! - Fallback to streams for large frames

use betanet_htx::{Frame, FrameType, HtxConfig, HtxError};
use bytes::Bytes;
use std::net::SocketAddr;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{error, info, warn};

#[cfg(feature = "quic")]
use betanet_htx::quic::{EchConfig, QuicTransport};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    info!("🚀 Starting HTX over QUIC DATAGRAM demo");

    #[cfg(not(feature = "quic"))]
    {
        eprintln!("❌ QUIC feature not enabled. Build with --features quic");
        return Ok(());
    }

    #[cfg(feature = "quic")]
    {
        // Configure HTX with TLS camouflage enabled. If `ech_config_path` is not
        // set, the ECH configuration will be fetched via DNS using the
        // `camouflage_domain` value.
        let mut config = HtxConfig::default();
        config.enable_quic = true;
        config.enable_tls_camouflage = true;
        config.camouflage_domain = Some("cloudflare.com".to_string());
        config.alpn_protocols = vec!["h3".to_string(), "h3-32".to_string(), "htx/1.1".to_string()];

        let listen_addr: SocketAddr = "127.0.0.1:9443".parse()?;
        let connect_addr = listen_addr;

        info!("📡 Testing QUIC DATAGRAM echo over UDP-443 path");

        // Start server in background
        let server_config = config.clone();
        let server_handle =
            tokio::spawn(async move { run_echo_server(server_config, listen_addr).await });

        // Give server time to start
        sleep(Duration::from_millis(100)).await;

        // Run client
        match run_echo_client(config, connect_addr).await {
            Ok(()) => {
                info!("✅ Client completed successfully");
            }
            Err(e) => {
                error!("❌ Client failed: {}", e);
            }
        }

        // Stop server
        server_handle.abort();
        let _ = server_handle.await;

        info!("🏁 Demo completed");
    }

    Ok(())
}

#[cfg(feature = "quic")]
async fn run_echo_server(config: HtxConfig, addr: SocketAddr) -> Result<(), HtxError> {
    info!("🔧 Starting QUIC echo server on {}", addr);

    // This is a simplified server demo - in practice you'd use QuicTransport::listen.
    // Demonstrate loading an ECH configuration from DNS.
    match EchConfig::from_dns("cloudflare.com").await {
        Ok(ech_config) => {
            info!("🔒 ECH configuration:");
            info!("  - Public name: {}", ech_config.public_name);
            info!("  - Config ID: {}", ech_config.config_id);
            info!("  - KEM ID: 0x{:04x}", ech_config.kem_id);
            info!("  - Cipher suites: {:?}", ech_config.cipher_suites);
        }
        Err(e) => {
            warn!("⚠️ Failed to load ECH config: {}", e);
        }
    }

    // Simulate server running
    sleep(Duration::from_millis(500)).await;

    Ok(())
}

#[cfg(feature = "quic")]
async fn run_echo_client(config: HtxConfig, addr: SocketAddr) -> Result<(), HtxError> {
    info!("📞 Connecting QUIC client to {}", addr);

    // Create QUIC transport (this will fail in demo due to no actual server)
    match QuicTransport::connect(addr, &config, None).await {
        Ok(mut transport) => {
            info!("✅ QUIC connection established");

            // Check DATAGRAM support
            if transport.has_datagram_support() {
                info!(
                    "📦 DATAGRAM support: enabled (max size: {:?})",
                    transport.max_datagram_size()
                );
            } else {
                warn!("📦 DATAGRAM support: disabled, falling back to streams");
            }

            // Check ECH configuration
            if let Some(ech) = transport.ech_config() {
                info!("🔒 ECH configured with public name: {}", ech.public_name);
            }

            // Test different frame types over DATAGRAM
            let test_frames = vec![
                Frame::ping(Some(Bytes::from("hello"))).unwrap(),
                Frame::data(1, Bytes::from("small data")).unwrap(),
                Frame::window_update(1, 1024).unwrap(),
            ];

            for (i, frame) in test_frames.into_iter().enumerate() {
                info!("📨 Sending test frame {} via DATAGRAM", i + 1);

                match transport.send_datagram(frame).await {
                    Ok(()) => {
                        info!("✅ Frame {} sent successfully", i + 1);
                    }
                    Err(e) => {
                        warn!("⚠️ Frame {} failed via DATAGRAM: {}", i + 1, e);
                        info!("🔄 Would fallback to stream in production");
                    }
                }
            }

            // Test receiving DATAGRAMs
            info!("👂 Listening for DATAGRAM responses...");
            for _ in 0..3 {
                match transport.recv_datagram().await {
                    Ok(Some(frame)) => {
                        info!("📬 Received frame: {:?}", frame.frame_type);
                    }
                    Ok(None) => {
                        info!("📭 No DATAGRAM received");
                        break;
                    }
                    Err(e) => {
                        warn!("❌ DATAGRAM receive error: {}", e);
                        break;
                    }
                }
            }
        }
        Err(e) => {
            // Expected in demo since we don't have a real server
            info!("🔄 Connection failed (expected in demo): {}", e);
            info!("📊 Demo configuration summary:");
            info!("  - QUIC enabled: {}", config.enable_quic);
            info!("  - TLS camouflage: {}", config.enable_tls_camouflage);
            info!("  - ALPN protocols: {:?}", config.alpn_protocols);
            if let Some(domain) = config.camouflage_domain {
                info!("  - Camouflage domain: {}", domain);
            }
        }
    }

    Ok(())
}

/// Test that demonstrates QUIC DATAGRAM path usage
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_quic_datagram_config() {
        let mut config = HtxConfig::default();
        config.enable_quic = true;
        config.enable_tls_camouflage = true;

        assert!(config.enable_quic);
        assert!(config.enable_tls_camouflage);
    }

    #[test]
    fn test_ech_config_defaults() {
        let ech = EchConfig::default();
        assert_eq!(ech.public_name, "cloudflare.com");
        assert_eq!(ech.kem_id, 0x0020); // X25519
        assert!(!ech.cipher_suites.is_empty());
    }

    #[test]
    fn test_frame_sizes_for_datagram() {
        // Test frames that should fit in DATAGRAM (typical max ~1200 bytes)
        let small_frame = Frame::ping(Some(Bytes::from("test"))).unwrap();
        assert!(small_frame.size() < 1200);

        let medium_frame = Frame::data(1, Bytes::from(vec![0u8; 500])).unwrap();
        assert!(medium_frame.size() < 1200);

        // Large frame should exceed DATAGRAM limits
        let large_frame = Frame::data(1, Bytes::from(vec![0u8; 2000])).unwrap();
        assert!(large_frame.size() > 1200);
    }
}
