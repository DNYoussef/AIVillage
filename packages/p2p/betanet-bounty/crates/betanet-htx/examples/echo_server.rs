//! HTX echo server example
//!
//! Demonstrates HTX protocol server with:
//! - TCP and QUIC transport support
//! - ECH stub configuration
//! - JA3 template integration
//! - Noise XK handshake
//! - Stream multiplexing

use betanet_htx::{
    accept_tcp, dial_tcp, HtxConfig, HtxError, Result, StreamMux,
    TlsCamouflageBuilder, create_tls_connector,
};
use clap::Parser;
use std::net::SocketAddr;
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tracing::{error, info, warn};

/// Echo server command-line arguments
#[derive(Parser, Debug)]
#[command(author, version, about = "HTX echo server example")]
struct Args {
    /// Listen address
    #[arg(short, long, default_value = "127.0.0.1:9443")]
    listen: SocketAddr,

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

    info!("HTX Echo Server starting on {}", args.listen);

    // Configure HTX
    let mut config = HtxConfig {
        listen_addr: args.listen,
        enable_tcp: args.tcp || (!args.tcp && !args.quic), // Default to TCP
        enable_quic: args.quic,
        enable_noise_xk: true,
        enable_tls_camouflage: args.ech_stub,
        camouflage_domain: args.camouflage_domain,
        ech_config_path: args.ech_config_path.map(Into::into),
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

    // Start the appropriate server
    if config.enable_quic {
        info!("Starting QUIC server...");
        run_quic_server(config).await?;
    } else {
        info!("Starting TCP server...");
        run_tcp_server(config).await?;
    }

    Ok(())
}

/// Run TCP-based HTX server
async fn run_tcp_server(config: HtxConfig) -> Result<()> {
    let listener = TcpListener::bind(config.listen_addr).await
        .map_err(|e| HtxError::Transport(format!("Failed to bind TCP listener: {}", e)))?;

    info!("TCP server listening on {}", config.listen_addr);

    loop {
        match accept_tcp(&listener, config.clone()).await {
            Ok(mut connection) => {
                info!("Accepted TCP connection from {}",
                      connection.session().config.listen_addr);

                tokio::spawn(async move {
                    if let Err(e) = handle_tcp_connection(&mut connection).await {
                        error!("TCP connection error: {}", e);
                    }
                });
            }
            Err(e) => {
                error!("Failed to accept TCP connection: {}", e);
            }
        }
    }
}

/// Handle TCP connection with echo protocol
async fn handle_tcp_connection(
    connection: &mut betanet_htx::HtxTcpConnection
) -> Result<()> {
    // Perform HTX handshake
    info!("Starting HTX handshake");
    connection.handshake().await?;
    info!("HTX handshake completed");

    // Create multiplexer
    let (frame_sender, mut frame_receiver) = mpsc::unbounded_channel();
    let mux = StreamMux::new(false, frame_sender); // Server uses even stream IDs

    loop {
        tokio::select! {
            // Handle incoming data from connection
            data_result = connection.recv() => {
                match data_result {
                    Ok(stream_data) => {
                        for (stream_id, data) in stream_data {
                            info!("Received {} bytes on stream {}", data.len(), stream_id);

                            // Echo the data back
                            let echo_data = format!("Echo: {}",
                                String::from_utf8_lossy(&data));

                            if let Err(e) = connection.send(stream_id, echo_data.as_bytes()).await {
                                error!("Failed to send echo response: {}", e);
                            }
                        }
                    }
                    Err(e) => {
                        error!("Connection receive error: {}", e);
                        break;
                    }
                }
            }

            // Handle outgoing frames from multiplexer
            frame = frame_receiver.recv() => {
                if let Some(frame) = frame {
                    info!("Sending frame: {:?}", frame.frame_type);
                    // In a real implementation, would send frame via connection
                }
            }
        }
    }

    info!("Connection closed");
    Ok(())
}

/// Run QUIC-based HTX server (stub implementation)
async fn run_quic_server(config: HtxConfig) -> Result<()> {
    #[cfg(feature = "quic")]
    {
        info!("QUIC server starting on {}", config.listen_addr);

        // This would use the QuicTransport implementation
        // For now, this is a stub that demonstrates the interface

        use betanet_htx::quic::QuicTransport;

        // Create a dummy handler for QUIC connections
        let handler = |_connection: Box<dyn betanet_htx::HtxConnection>| {
            info!("QUIC connection accepted");
        };

        QuicTransport::listen(config, handler).await?;
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
            "echo_server",
            "--listen", "0.0.0.0:8443",
            "--tcp",
            "--ech-stub",
            "--ja3-template", "chrome_latest",
            "--verbose"
        ]);

        assert_eq!(args.listen, "0.0.0.0:8443".parse().unwrap());
        assert!(args.tcp);
        assert!(!args.quic);
        assert!(args.ech_stub);
        assert_eq!(args.ja3_template, Some("chrome_latest".to_string()));
        assert!(args.verbose);
    }

    #[tokio::test]
    async fn test_config_creation() {
        let config = HtxConfig {
            listen_addr: "127.0.0.1:9443".parse().unwrap(),
            enable_tcp: true,
            enable_quic: false,
            enable_noise_xk: true,
            enable_tls_camouflage: true,
            camouflage_domain: Some("example.com".to_string()),
            ..Default::default()
        };

        assert!(config.enable_tcp);
        assert!(!config.enable_quic);
        assert!(config.enable_noise_xk);
        assert!(config.enable_tls_camouflage);
    }
}
