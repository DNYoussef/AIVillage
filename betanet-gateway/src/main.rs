// Betanet Gateway - SCION tunnel termination and HTX control stream processor
// Production implementation with anti-replay, AEAD encryption, and telemetry

use anyhow::{Context, Result};
use clap::Parser;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::signal;
use tracing::{info, warn, error};

mod config;
mod encap;
mod anti_replay;
mod aead;
mod integrated_protection;
mod multipath;
mod metrics;
mod htx_server;
mod scion_client;

// Generated protobuf code will be created by build script
mod generated {
    pub mod betanet {
        pub mod gateway {
            tonic::include_proto!("betanet.gateway");
        }
    }
}

use crate::config::GatewayConfig;
use crate::htx_server::HTXServer;
use crate::scion_client::ScionClient;
use crate::anti_replay::AntiReplayManager;
use crate::aead::AeadManager;
use crate::multipath::MultipathManager;
use crate::metrics::MetricsCollector;

#[derive(Parser)]
#[command(name = "betanet-gateway")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(about = "Betanet Gateway - SCION tunnel termination for AIVillage")]
struct Args {
    /// Configuration file path
    #[arg(short, long, value_name = "FILE")]
    config: Option<PathBuf>,
    
    /// HTX server bind address
    #[arg(long, default_value = "0.0.0.0:8443")]
    htx_bind: SocketAddr,
    
    /// SCION sidecar gRPC address  
    #[arg(long, default_value = "127.0.0.1:8080")]
    scion_addr: SocketAddr,
    
    /// Metrics server address
    #[arg(long, default_value = "0.0.0.0:9090")]
    metrics_addr: SocketAddr,
    
    /// Anti-replay database path
    #[arg(long, default_value = "/var/lib/betanet-gateway/replay.db")]
    replay_db_path: PathBuf,
    
    /// Log level
    #[arg(long, default_value = "info", value_parser = ["trace", "debug", "info", "warn", "error"])]
    log_level: String,
    
    /// Enable JSON logging
    #[arg(long)]
    json_logs: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    
    // Initialize logging
    init_logging(&args.log_level, args.json_logs)?;
    
    info!(
        version = env!("CARGO_PKG_VERSION"),
        "Starting Betanet Gateway"
    );
    
    // Load configuration
    let config = load_config(args.config.as_deref()).await
        .context("Failed to load configuration")?;
    
    // Override config with command line args
    let config = override_config_with_args(config, &args);
    
    info!(?config, "Loaded configuration");
    
    // Initialize components
    let gateway = BetanetGateway::new(config).await
        .context("Failed to initialize Betanet Gateway")?;
    
    // Start the gateway
    gateway.run().await
        .context("Gateway execution failed")?;
    
    info!("Betanet Gateway stopped");
    Ok(())
}

/// Main gateway orchestrator
pub struct BetanetGateway {
    config: Arc<GatewayConfig>,
    metrics: Arc<MetricsCollector>,
    scion_client: Arc<ScionClient>,
    anti_replay: Arc<AntiReplayManager>,
    multipath: Arc<MultipathManager>,
    htx_server: HTXServer,
}

impl BetanetGateway {
    pub async fn new(config: GatewayConfig) -> Result<Self> {
        let config = Arc::new(config);
        
        info!("Initializing Betanet Gateway components");
        
        // Initialize metrics collector
        let metrics = Arc::new(MetricsCollector::new(config.clone())?);
        
        // Initialize SCION client
        let scion_client = Arc::new(
            ScionClient::new(config.scion.clone(), metrics.clone()).await?
        );
        
        // Initialize anti-replay manager
        let anti_replay = Arc::new(
            AntiReplayManager::new(config.anti_replay.clone(), metrics.clone()).await?
        );
        
        // Initialize multipath manager
        let multipath = Arc::new(
            MultipathManager::new(config.multipath.clone(), metrics.clone()).await?
        );
        
        // Initialize HTX server
        let htx_server = HTXServer::new(
            config.clone(),
            scion_client.clone(),
            anti_replay.clone(),
            multipath.clone(),
            metrics.clone(),
        ).await?;
        
        info!("Betanet Gateway components initialized successfully");
        
        Ok(BetanetGateway {
            config,
            metrics,
            scion_client,
            anti_replay,
            multipath,
            htx_server,
        })
    }
    
    pub async fn run(self) -> Result<()> {
        info!("Starting Betanet Gateway services");
        
        // Start metrics server
        let metrics_handle = {
            let metrics = self.metrics.clone();
            let addr = self.config.metrics.bind_addr;
            tokio::spawn(async move {
                if let Err(e) = metrics.start_server(addr).await {
                    error!(error = ?e, "Metrics server failed");
                }
            })
        };
        
        // Start HTX server
        let htx_handle = {
            let server = self.htx_server;
            tokio::spawn(async move {
                if let Err(e) = server.run().await {
                    error!(error = ?e, "HTX server failed");
                }
            })
        };
        
        // Start background tasks
        let multipath_handle = {
            let multipath = self.multipath.clone();
            tokio::spawn(async move {
                multipath.run_background_tasks().await;
            })
        };
        
        let anti_replay_handle = {
            let anti_replay = self.anti_replay.clone();
            tokio::spawn(async move {
                anti_replay.run_background_tasks().await;
            })
        };
        
        info!(
            htx_bind = ?self.config.htx.bind_addr,
            metrics_bind = ?self.config.metrics.bind_addr,
            scion_addr = ?self.config.scion.address,
            "Betanet Gateway started successfully"
        );
        
        // Wait for shutdown signal
        tokio::select! {
            result = metrics_handle => {
                warn!(?result, "Metrics server task completed");
            }
            result = htx_handle => {
                warn!(?result, "HTX server task completed");  
            }
            result = multipath_handle => {
                warn!(?result, "Multipath manager task completed");
            }
            result = anti_replay_handle => {
                warn!(?result, "Anti-replay manager task completed");
            }
            _ = signal::ctrl_c() => {
                info!("Shutdown signal received");
            }
        }
        
        info!("Shutting down Betanet Gateway");
        
        // Graceful shutdown
        if let Err(e) = self.shutdown().await {
            error!(error = ?e, "Error during shutdown");
        }
        
        Ok(())
    }
    
    async fn shutdown(&self) -> Result<()> {
        info!("Performing graceful shutdown");
        
        // Stop components in reverse order
        if let Err(e) = self.multipath.stop().await {
            error!(error = ?e, "Error stopping multipath manager");
        }
        
        if let Err(e) = self.anti_replay.stop().await {
            error!(error = ?e, "Error stopping anti-replay manager");
        }
        
        if let Err(e) = self.scion_client.stop().await {
            error!(error = ?e, "Error stopping SCION client");
        }
        
        if let Err(e) = self.metrics.stop().await {
            error!(error = ?e, "Error stopping metrics collector");
        }
        
        info!("Graceful shutdown completed");
        Ok(())
    }
}

fn init_logging(level: &str, json_logs: bool) -> Result<()> {
    use tracing_subscriber::{EnvFilter, fmt, prelude::*};
    
    let filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new(level))
        .context("Failed to create logging filter")?;
    
    let registry = tracing_subscriber::registry()
        .with(filter);
    
    if json_logs {
        registry
            .with(fmt::layer().json())
            .init();
    } else {
        registry
            .with(fmt::layer().with_target(false))
            .init();
    }
    
    info!(level = level, json = json_logs, "Logging initialized");
    Ok(())
}

async fn load_config(config_path: Option<&std::path::Path>) -> Result<GatewayConfig> {
    match config_path {
        Some(path) => {
            info!(?path, "Loading configuration from file");
            GatewayConfig::from_file(path).await
        }
        None => {
            info!("Using default configuration");
            Ok(GatewayConfig::default())
        }
    }
}

fn override_config_with_args(mut config: GatewayConfig, args: &Args) -> GatewayConfig {
    // Override with command line arguments
    config.htx.bind_addr = args.htx_bind;
    config.scion.address = args.scion_addr;
    config.metrics.bind_addr = args.metrics_addr;
    config.anti_replay.db_path = args.replay_db_path.clone();
    
    config
}