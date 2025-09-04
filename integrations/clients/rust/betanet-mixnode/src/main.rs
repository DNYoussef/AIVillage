//! Betanet Mixnode CLI

use std::net::SocketAddr;

use clap::{Parser, Subcommand};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

use betanet_mixnode::{
    config::MixnodeConfig,
    mixnode::StandardMixnode,
    packet::Packet,
    Mixnode,
    MixnodeError,
    MixnodeStats,
    Result,
};
use bytes::Bytes;

/// Betanet Mixnode CLI
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Subcommand to run
    #[command(subcommand)]
    command: Commands,

    /// Verbose logging
    #[arg(short, long)]
    verbose: bool,

    /// Configuration file path
    #[arg(short, long)]
    config: Option<std::path::PathBuf>,
}

/// Available commands
#[derive(Subcommand)]
enum Commands {
    /// Start the mixnode
    Start {
        /// Listen address
        #[arg(short, long, default_value = "127.0.0.1:9001")]
        listen: SocketAddr,

        /// Number of layers in the mix network
        #[arg(long, default_value = "3")]
        layers: u8,

        /// Enable cover traffic
        #[arg(long)]
        cover_traffic: bool,
    },

    /// Generate keypair
    Keygen {
        /// Output file for private key
        #[arg(short, long, default_value = "mixnode.key")]
        output: std::path::PathBuf,
    },

    /// Show node statistics
    Stats {
        /// Node address to query
        #[arg(short, long)]
        node: SocketAddr,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize tracing subscriber with level based on --verbose flag
    let subscriber = FmtSubscriber::builder()
        .with_max_level(if cli.verbose { Level::DEBUG } else { Level::INFO })
        .finish();
    tracing::subscriber::set_global_default(subscriber)
        .expect("setting default subscriber failed");

    match cli.command {
        Commands::Start {
            listen,
            layers,
            cover_traffic,
        } => {
            info!("Starting Betanet mixnode on {}", listen);

            let config = MixnodeConfig {
                listen_addr: listen,
                layers,
                enable_cover_traffic: cover_traffic,
                ..Default::default()
            };

            let mut mixnode = StandardMixnode::new(config)?;
            mixnode.start().await?;

            // Wait for Ctrl+C
            tokio::signal::ctrl_c()
                .await
                .map_err(betanet_mixnode::MixnodeError::Io)?;

            info!("Shutting down mixnode");
            mixnode.stop().await?;
        }

        Commands::Keygen { output } => {
            info!("Generating keypair to {:?}", output);

            use rand::rngs::OsRng;
            use rand::RngCore;
            let mut private_key = [0u8; 32];
            OsRng.fill_bytes(&mut private_key);

            std::fs::write(output, private_key).map_err(MixnodeError::Io)?;

            info!("Keypair generated successfully");
        }

        Commands::Stats { node } => {
            info!("Querying statistics from {}", node);
            let mut stream = tokio::net::TcpStream::connect(node)
                .await
                .map_err(MixnodeError::Io)?;

            // Send control packet requesting stats
            let packet = Packet::control(Bytes::from_static(b"stats"));
            let request = packet.encode()?;
            stream.write_all(&request).await.map_err(MixnodeError::Io)?;

            // Read response
            let mut buf = vec![0u8; 1024];
            let n = stream.read(&mut buf).await.map_err(MixnodeError::Io)?;
            let response = Packet::parse(&buf[..n])?;
            let stats: MixnodeStats =
                serde_json::from_slice(&response.payload).map_err(|e| MixnodeError::Network(e.to_string()))?;

            println!("Statistics for node {}:", node);
            println!("  Packets processed: {}", stats.packets_processed);
            println!("  Packets forwarded: {}", stats.packets_forwarded);
            println!("  Packets dropped: {}", stats.packets_dropped);
            println!("  Cover traffic sent: {}", stats.cover_traffic_sent);
            println!("  Avg processing time: {:.2} µs", stats.avg_processing_time_us);
            println!("  Uptime: {} s", stats.uptime_secs);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_parsing() {
        use clap::Parser;

        let cli = Cli::try_parse_from(["betanet-mixnode", "start", "--listen", "127.0.0.1:9001"]);
        assert!(cli.is_ok());
    }
}
