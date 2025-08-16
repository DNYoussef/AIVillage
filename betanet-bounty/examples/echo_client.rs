//! Echo client example using Betanet HTX transport
//!
//! This example demonstrates connecting to an HTX server and performing
//! echo operations, integrating with existing AI Village infrastructure.

use std::io;
use std::net::SocketAddr;

use betanet_htx::{HtxClient, HtxConfig};
use tokio::io::{AsyncBufReadExt, BufReader};
use tracing::{error, info};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    let server_addr: SocketAddr = "127.0.0.1:9000".parse()?;
    info!("Connecting to Betanet HTX echo server at {}", server_addr);

    // Configure HTX client with AI Village integration
    let config = HtxConfig {
        listen_addr: "127.0.0.1:0".parse()?, // Use any available port for client
        enable_tcp: true,
        enable_quic: false,
        enable_noise_xk: true, // Match server configuration
        enable_hybrid_kem: false,
        max_connections: 1,
        connection_timeout_secs: 30,
        keepalive_interval_secs: 10,
    };

    let mut client = HtxClient::new(config);

    // Connect to server
    info!("Establishing HTX connection...");
    client.connect(server_addr).await?;
    info!("Connected! Type messages to echo (Ctrl+C to exit)");

    // Interactive echo session
    let stdin = io::stdin();
    let mut reader = BufReader::new(tokio::io::stdin());
    let mut line = String::new();

    loop {
        print!("Enter message: ");
        io::Write::flush(&mut io::stdout())?;

        line.clear();
        match reader.read_line(&mut line).await {
            Ok(0) => {
                info!("EOF received, exiting");
                break;
            }
            Ok(_) => {
                let message = line.trim();
                if message.is_empty() {
                    continue;
                }

                if message == "quit" || message == "exit" {
                    info!("Exiting on user request");
                    break;
                }

                // Send message
                info!("Sending: {}", message);
                if let Err(e) = client.send(message.as_bytes()).await {
                    error!("Failed to send message: {}", e);
                    continue;
                }

                // Receive echo
                let mut buffer = vec![0u8; 4096];
                match client.recv(&mut buffer).await {
                    Ok(n) => {
                        let response = String::from_utf8_lossy(&buffer[..n]);
                        info!("Received: {}", response);

                        if response.trim() == message {
                            println!("✓ Echo successful!");
                        } else {
                            println!("✗ Echo mismatch: sent '{}', got '{}'", message, response);
                        }
                    }
                    Err(e) => {
                        error!("Failed to receive echo: {}", e);
                    }
                }
            }
            Err(e) => {
                error!("Failed to read input: {}", e);
                break;
            }
        }
    }

    // Disconnect
    info!("Disconnecting from server");
    client.disconnect().await?;
    info!("Goodbye!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use betanet_htx::HtxServer;
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn test_echo_client_server() {
        // This test would verify full client-server echo functionality
        // integrated with AI Village transport protocols

        let config = HtxConfig::default();
        let client = HtxClient::new(config.clone());
        let server = HtxServer::new(config);

        // In a real test, we'd:
        // 1. Start the server in a background task
        // 2. Connect with the client
        // 3. Send test messages and verify echoes
        // 4. Test error conditions and reconnection

        assert_eq!(client.remote_addr(), None); // Not connected yet
    }

    #[tokio::test]
    async fn test_chrome_fingerprint_mimicry() {
        // Test that our HTX client properly mimics Chrome browser behavior
        // when integrating with uTLS fingerprinting

        use betanet_utls::{ChromeProfile, ClientHello};

        let profile = ChromeProfile::chrome_119();
        let hello = ClientHello::from_chrome_profile(&profile, "example.com");

        // Verify the fingerprint looks like Chrome
        assert_eq!(hello.version, betanet_utls::tls_version::TLS_1_2);
        assert!(!hello.cipher_suites.is_empty());
        assert!(!hello.extensions.is_empty());
    }
}
