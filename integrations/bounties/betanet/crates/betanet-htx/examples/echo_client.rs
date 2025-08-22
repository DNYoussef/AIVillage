//! HTX echo client example
//!
//! Demonstrates HTX protocol client with:
//! - TCP and QUIC transport support
//! - ECH stub configuration
//! - JA3 template integration
//! - Noise XK handshake
//! - Stream multiplexing

use betanet_htx::{
    dial_tcp, HtxConfig, HtxError, Result, StreamMux,
    TlsCamouflageBuilder, create_tls_connector,
};
use clap::Parser;
use std::net::SocketAddr;
use tokio::sync::mpsc;
use tokio::time::{sleep, Duration};
use tracing::{error, info, warn};

/// Echo client command-line arguments
#[derive(Parser, Debug)]
#[command(author, version, about = "HTX echo client example")]
struct Args {
    /// Server address to connect to
    #[arg(short, long, default_value = "127.0.0.1:9443")]
    server: SocketAddr,

    /// Use TCP transport (default)
    #[arg(long)]
    tcp: bool,

    /// Use QUIC transport
    #[arg(long)]
    quic: bool,

    /// Enable ECH stub
    #[arg(long)]
    ech_stub: bool,

    /// JA3 template to use
    #[arg(long)]
    ja3_template: Option<String>,

    /// Camouflage domain for ECH
    #[arg(long)]
    camouflage_domain: Option<String>,

    /// Path to ECH config file
    #[arg(long)]
    ech_config_path: Option<String>,

    /// Message to send to server
    #[arg(short, long, default_value = "Hello, HTX Server!")]
    message: String,

    /// Number of messages to send
    #[arg(short, long, default_value = "5")]
    count: u32,

    /// Delay between messages (milliseconds)
    #[arg(short, long, default_value = "1000")]
    delay: u64,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize tracing
    let log_level = if args.verbose {
        tracing::Level::DEBUG
    } else {
        tracing::Level::INFO
    };

    tracing_subscriber::fmt()
        .with_max_level(log_level)
        .init();

    info!("HTX Echo Client connecting to {}", args.server);

    // Configure HTX
    let config = HtxConfig {
        listen_addr: "127.0.0.1:0".parse().unwrap(), // Client uses ephemeral port
        enable_tcp: args.tcp || (!args.tcp && !args.quic), // Default to TCP
        enable_quic: args.quic,
        enable_noise_xk: true,
        enable_tls_camouflage: args.ech_stub,
        camouflage_domain: args.camouflage_domain.clone(),
        ech_config_path: args.ech_config_path.clone().map(Into::into),
        alpn_protocols: vec!["htx/1.1".to_string(), "h2".to_string()],
        ..Default::default()
    };

    if args.ech_stub {
        info!("ECH stub enabled for TLS camouflage");
    }

    if let Some(template) = &args.ja3_template {
        info!("Using JA3 template: {}", template);
        // Apply JA3 template (stub implementation)
        if let Err(e) = betanet_htx::tls::apply_ja3_template(template) {
            warn!("Failed to apply JA3 template: {}", e);
        }
    }

    // Connect using the appropriate transport
    if config.enable_quic {
        info!("Connecting via QUIC...");
        run_quic_client(args.server, config, &args).await?;
    } else {
        info!("Connecting via TCP...");
        run_tcp_client(args.server, config, &args).await?;
    }

    Ok(())
}

/// Run TCP-based HTX client
async fn run_tcp_client(server: SocketAddr, config: HtxConfig, args: &Args) -> Result<()> {
    let mut connection = dial_tcp(server, config).await?;
    info!("Connected to TCP server at {}", server);

    // Perform HTX handshake
    info!("Starting HTX handshake");
    connection.handshake().await?;
    info!("HTX handshake completed");

    // Create multiplexer
    let (frame_sender, mut frame_receiver) = mpsc::unbounded_channel();
    let mux = StreamMux::new(true, frame_sender); // Client uses odd stream IDs

    // Create a stream for communication
    let stream_id = connection.create_stream()?;
    info!("Created stream {}", stream_id);

    // Send messages
    for i in 0..args.count {
        let message = format!("{} (message {})", args.message, i + 1);
        info!("Sending: {}", message);

        if let Err(e) = connection.send(stream_id, message.as_bytes()).await {
            error!("Failed to send message: {}", e);
            break;
        }

        // Wait for response
        tokio::select! {
            data_result = connection.recv() => {
                match data_result {
                    Ok(stream_data) => {
                        for (recv_stream_id, data) in stream_data {
                            if recv_stream_id == stream_id {
                                let response = String::from_utf8_lossy(&data);
                                info!("Received: {}", response);
                            }
                        }
                    }
                    Err(e) => {
                        error!("Failed to receive response: {}", e);
                        break;
                    }
                }
            }

            // Handle frames from multiplexer
            frame = frame_receiver.recv() => {
                if let Some(frame) = frame {
                    info!("Multiplexer frame: {:?}", frame.frame_type);
                }
            }

            // Timeout
            _ = sleep(Duration::from_millis(5000)) => {
                warn!("Timeout waiting for response");
            }
        }

        if i < args.count - 1 {
            sleep(Duration::from_millis(args.delay)).await;
        }
    }

    info!("Closing connection");
    connection.close().await?;

    Ok(())
}

/// Run QUIC-based HTX client (stub implementation)
async fn run_quic_client(server: SocketAddr, config: HtxConfig, args: &Args) -> Result<()> {
    #[cfg(feature = "quic")]
    {
        info!("QUIC client connecting to {}", server);

        use betanet_htx::quic::QuicTransport;

        let mut transport = QuicTransport::connect(server, &config).await?;
        info!("Connected to QUIC server at {}", server);

        // Send messages using QUIC DATAGRAM if available
        for i in 0..args.count {
            let message = format!("{} (QUIC message {})", args.message, i + 1);
            info!("Sending via QUIC: {}", message);

            // Create a DATA frame for the message
            let frame = betanet_htx::Frame::data(1, message.into())?;

            if transport.has_datagram_support() {
                if let Err(e) = transport.send_datagram(frame).await {
                    error!("Failed to send DATAGRAM: {}", e);
                    // Fallback to stream
                    if let Err(e2) = transport.send(message.as_bytes()).await {
                        error!("Failed to send via stream: {}", e2);
                        break;
                    }
                }
            } else {
                if let Err(e) = transport.send(message.as_bytes()).await {
                    error!("Failed to send via QUIC stream: {}", e);
                    break;
                }
            }

            // Try to receive response
            tokio::select! {
                frame_result = transport.recv_datagram() => {
                    if let Ok(Some(frame)) = frame_result {
                        info!("Received DATAGRAM frame: {} bytes", frame.payload.len());
                        let response = String::from_utf8_lossy(&frame.payload);
                        info!("Response: {}", response);
                    }
                }

                _ = sleep(Duration::from_millis(2000)) => {
                    warn!("Timeout waiting for QUIC response");
                }
            }

            if i < args.count - 1 {
                sleep(Duration::from_millis(args.delay)).await;
            }
        }

        info!("Closing QUIC connection");
        transport.close().await?;
    }

    #[cfg(not(feature = "quic"))]
    {
        return Err(HtxError::Config("QUIC feature not enabled".to_string()));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_args_parsing() {
        let args = Args::parse_from(&[
            "echo_client",
            "--server", "192.168.1.100:8443",
            "--quic",
            "--ech-stub",
            "--ja3-template", "firefox_latest",
            "--message", "Test message",
            "--count", "3",
            "--delay", "500",
            "--verbose"
        ]);

        assert_eq!(args.server, "192.168.1.100:8443".parse().unwrap());
        assert!(!args.tcp);
        assert!(args.quic);
        assert!(args.ech_stub);
        assert_eq!(args.ja3_template, Some("firefox_latest".to_string()));
        assert_eq!(args.message, "Test message");
        assert_eq!(args.count, 3);
        assert_eq!(args.delay, 500);
        assert!(args.verbose);
    }

    #[tokio::test]
    async fn test_tcp_client_config() {
        let server = "127.0.0.1:9443".parse().unwrap();
        let config = HtxConfig {
            enable_tcp: true,
            enable_quic: false,
            enable_noise_xk: true,
            enable_tls_camouflage: true,
            camouflage_domain: Some("example.com".to_string()),
            ..Default::default()
        };

        // Test configuration validation
        assert!(config.enable_tcp);
        assert!(!config.enable_quic);
        assert!(config.enable_noise_xk);
        assert!(config.enable_tls_camouflage);
        assert_eq!(config.camouflage_domain, Some("example.com".to_string()));
    }

    #[test]
    fn test_message_formatting() {
        let base_message = "Hello";
        let formatted = format!("{} (message {})", base_message, 1);
        assert_eq!(formatted, "Hello (message 1)");
    }
}
