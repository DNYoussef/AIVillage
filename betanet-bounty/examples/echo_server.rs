//! Echo server example using Betanet HTX transport
//!
//! This example demonstrates integrating with the existing AI Village infrastructure
//! while providing a high-performance echo server.

use std::net::SocketAddr;

use betanet_htx::{HtxConfig, HtxConnection, HtxServer};
use tokio::net::TcpListener;
use tracing::{error, info};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    let listen_addr: SocketAddr = "127.0.0.1:9000".parse()?;
    info!("Starting Betanet HTX echo server on {}", listen_addr);

    // Configure HTX server to integrate with AI Village transport layer
    let config = HtxConfig {
        listen_addr,
        enable_tcp: true,
        enable_quic: false, // Start with TCP for simplicity
        enable_noise_xk: true, // Use Noise for encryption like existing infrastructure
        enable_hybrid_kem: false,
        max_connections: 100,
        connection_timeout_secs: 30,
        keepalive_interval_secs: 10,
    };

    let mut server = HtxServer::new(config);

    // Start server with echo handler
    server
        .start(|mut conn| {
            tokio::spawn(async move {
                let remote_addr = match conn.remote_addr() {
                    Ok(addr) => addr,
                    Err(e) => {
                        error!("Failed to get remote address: {}", e);
                        return;
                    }
                };

                info!("New connection from {}", remote_addr);

                let mut buffer = vec![0u8; 4096];

                loop {
                    match conn.recv(&mut buffer).await {
                        Ok(0) => {
                            info!("Connection closed by {}", remote_addr);
                            break;
                        }
                        Ok(n) => {
                            let received = &buffer[..n];
                            info!("Received {} bytes from {}", n, remote_addr);

                            // Echo the data back
                            if let Err(e) = conn.send(received).await {
                                error!("Failed to echo data to {}: {}", remote_addr, e);
                                break;
                            }

                            info!("Echoed {} bytes to {}", n, remote_addr);
                        }
                        Err(e) => {
                            error!("Failed to receive from {}: {}", remote_addr, e);
                            break;
                        }
                    }
                }

                info!("Connection handler for {} finished", remote_addr);
            });
        })
        .await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use betanet_htx::HtxClient;
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn test_echo_server_integration() {
        // This test would verify integration with AI Village transport layer
        let config = HtxConfig::default();
        let server = HtxServer::new(config);

        // In a real test, we'd start the server and connect with a client
        // to verify the echo functionality works
        assert_eq!(server.address(), config.listen_addr);
    }
}
