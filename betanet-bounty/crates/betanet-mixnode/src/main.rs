//! Betanet Mixnode CLI

use std::net::SocketAddr;

use clap::{Parser, Subcommand};
use tracing::info;
// use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use betanet_mixnode::{config::MixnodeConfig, mixnode::StandardMixnode, Mixnode, Result};

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

    // Initialize tracing (simplified)
    if cli.verbose {
        // Enable verbose logging (placeholder for now)
        println!("Verbose logging enabled");
    }

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
            tokio::signal::ctrl_c().await.map_err(|e| {
                betanet_mixnode::MixnodeError::Io(e)
            })?;

            info!("Shutting down mixnode");
            mixnode.stop().await?;
        }

        Commands::Keygen { output } => {
            info!("Generating keypair to {:?}", output);

            // Use a deterministic key for now (TODO: use proper random generation)
            let private_key = [42u8; 32];

            std::fs::write(output, private_key).map_err(|e| {
                betanet_mixnode::MixnodeError::Io(e)
            })?;

            info!("Keypair generated successfully");
        }

        Commands::Stats { node } => {
            info!("Querying statistics from {}", node);
            // This would connect to the node and fetch stats
            // For now, just print a placeholder
            println!("Statistics for node {}:", node);
            println!("  Packets processed: N/A");
            println!("  Uptime: N/A");
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

        let cli = Cli::try_parse_from(&["betanet-mixnode", "start", "--listen", "127.0.0.1:9001"]);
        assert!(cli.is_ok());
    }
}
