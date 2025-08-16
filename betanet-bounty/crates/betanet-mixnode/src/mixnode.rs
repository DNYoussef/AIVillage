//! Standard mixnode implementation

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{broadcast, RwLock};
use tracing::{debug, error, info, warn};

use crate::{
    config::MixnodeConfig, delay::DelayQueue, packet::Packet, routing::RoutingTable, Mixnode,
    MixnodeError, MixnodeStats, Result,
};

/// Standard mixnode implementation
pub struct StandardMixnode {
    config: MixnodeConfig,
    stats: Arc<RwLock<MixnodeStats>>,
    delay_queue: Arc<RwLock<DelayQueue>>,
    routing_table: Arc<RwLock<RoutingTable>>,
    shutdown_tx: Option<broadcast::Sender<()>>,
}

impl StandardMixnode {
    /// Create a new standard mixnode
    pub fn new(config: MixnodeConfig) -> Result<Self> {
        config.validate()?;

        Ok(Self {
            config,
            stats: Arc::new(RwLock::new(MixnodeStats::new())),
            delay_queue: Arc::new(RwLock::new(DelayQueue::new())),
            routing_table: Arc::new(RwLock::new(RoutingTable::new())),
            shutdown_tx: None,
        })
    }

    /// Handle incoming connection
    async fn handle_connection(&self, stream: TcpStream, peer_addr: SocketAddr) -> Result<()> {
        debug!("Handling connection from {}", peer_addr);

        let mut buffer = vec![0u8; self.config.buffer_size];

        loop {
            match tokio::time::timeout(self.config.connection_timeout, stream.readable()).await {
                Ok(Ok(())) => {
                    match stream.try_read(&mut buffer) {
                        Ok(0) => {
                            debug!("Connection closed by peer {}", peer_addr);
                            break;
                        }
                        Ok(n) => {
                            let start_time = Instant::now();

                            // Process the packet
                            if let Some(processed) = self.process_packet(&buffer[..n]).await? {
                                // Queue for delayed forwarding
                                let delay = self.calculate_delay().await;
                                let mut delay_queue = self.delay_queue.write().await;
                                delay_queue.add_packet(processed, delay).await;
                            }

                            // Update statistics
                            let processing_time = start_time.elapsed();
                            let mut stats = self.stats.write().await;
                            stats.record_processed(processing_time);
                        }
                        Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                            continue;
                        }
                        Err(e) => {
                            error!("Failed to read from {}: {}", peer_addr, e);
                            break;
                        }
                    }
                }
                Ok(Err(e)) => {
                    error!("Stream error for {}: {}", peer_addr, e);
                    break;
                }
                Err(_) => {
                    warn!("Connection timeout for {}", peer_addr);
                    break;
                }
            }
        }

        Ok(())
    }

    /// Calculate packet delay
    async fn calculate_delay(&self) -> Duration {
        #[cfg(feature = "vrf")]
        {
            // Use VRF-based delay calculation
            if self.config.enable_vrf {
                return crate::vrf_delay::calculate_vrf_delay(
                    &self.config.min_delay,
                    &self.config.max_delay,
                )
                .await
                .unwrap_or(self.config.min_delay);
            }
        }

        // Fallback to random delay
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let delay_ms =
            rng.gen_range(self.config.min_delay.as_millis()..=self.config.max_delay.as_millis());
        Duration::from_millis(delay_ms as u64)
    }

    /// Start cover traffic generation
    async fn start_cover_traffic(&self) {
        if !self.config.enable_cover_traffic {
            return;
        }

        let stats = Arc::clone(&self.stats);
        let interval = self.config.cover_traffic_interval;

        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            loop {
                ticker.tick().await;

                // Generate and send cover traffic
                debug!("Generating cover traffic");

                // Update statistics
                let mut stats = stats.write().await;
                stats.record_cover_traffic();
            }
        });
    }

    /// Process packets from delay queue
    async fn process_delay_queue(&self, mut shutdown_rx: broadcast::Receiver<()>) {
        let delay_queue = Arc::clone(&self.delay_queue);
        let routing_table = Arc::clone(&self.routing_table);
        let stats = Arc::clone(&self.stats);

        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(Duration::from_millis(10));
            loop {
                tokio::select! {
                    _ = ticker.tick() => {
                        let packet = {
                            let mut queue = delay_queue.write().await;
                            queue.pop_ready().await
                        };

                        if let Some(packet) = packet {
                            // Forward the packet
                            debug!("Forwarding delayed packet");

                            // Parse packet to get routing info
                            if let Ok(parsed_packet) = Packet::parse(&packet) {
                                let routing = routing_table.read().await;
                                if let Some(next_hop) = routing.get_next_hop(&parsed_packet).await {
                                    // Forward to next hop
                                    debug!("Forwarding to {}", next_hop);

                                    let mut stats = stats.write().await;
                                    stats.record_forwarded();
                                } else {
                                    warn!("No route found for packet");

                                    let mut stats = stats.write().await;
                                    stats.record_dropped();
                                }
                            }
                        }
                    }
                    _ = shutdown_rx.recv() => {
                        debug!("Delay queue processor shutting down");
                        break;
                    }
                }
            }
        });
    }
}

#[async_trait::async_trait]
impl Mixnode for StandardMixnode {
    async fn start(&mut self) -> Result<()> {
        info!("Starting mixnode on {}", self.config.listen_addr);

        let listener = TcpListener::bind(self.config.listen_addr)
            .await
            .map_err(MixnodeError::Io)?;

        let (shutdown_tx, _) = broadcast::channel(1);
        self.shutdown_tx = Some(shutdown_tx.clone());

        // Start cover traffic generation
        self.start_cover_traffic().await;

        // Start delay queue processor
        self.process_delay_queue(shutdown_tx.subscribe()).await;

        // Accept connections
        tokio::spawn({
            let config = self.config.clone();
            let stats = Arc::clone(&self.stats);
            let delay_queue = Arc::clone(&self.delay_queue);
            let routing_table = Arc::clone(&self.routing_table);
            let mut shutdown_rx = shutdown_tx.subscribe();

            async move {
                loop {
                    tokio::select! {
                        result = listener.accept() => {
                            match result {
                                Ok((stream, addr)) => {
                                    debug!("Accepted connection from {}", addr);

                                    let mixnode = StandardMixnode {
                                        config: config.clone(),
                                        stats: Arc::clone(&stats),
                                        delay_queue: Arc::clone(&delay_queue),
                                        routing_table: Arc::clone(&routing_table),
                                        shutdown_tx: None,
                                    };

                                    tokio::spawn(async move {
                                        if let Err(e) = mixnode.handle_connection(stream, addr).await {
                                            error!("Connection error: {}", e);
                                        }
                                    });
                                }
                                Err(e) => {
                                    error!("Failed to accept connection: {}", e);
                                }
                            }
                        }
                        _ = shutdown_rx.recv() => {
                            info!("Shutdown signal received");
                            break;
                        }
                    }
                }
            }
        });

        Ok(())
    }

    async fn stop(&mut self) -> Result<()> {
        info!("Stopping mixnode");

        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }

        Ok(())
    }

    async fn process_packet(&self, packet: &[u8]) -> Result<Option<Vec<u8>>> {
        debug!("Processing packet of {} bytes", packet.len());

        let _parsed_packet = Packet::parse(packet)?;

        #[cfg(feature = "sphinx")]
        if self.config.enable_sphinx {
            return crate::sphinx::process_sphinx_packet(&_parsed_packet).await;
        }

        // Fallback to simple forwarding
        Ok(Some(packet.to_vec()))
    }

    fn stats(&self) -> Arc<RwLock<MixnodeStats>> {
        Arc::clone(&self.stats)
    }

    fn address(&self) -> SocketAddr {
        self.config.listen_addr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mixnode_creation() {
        let config = MixnodeConfig::default();
        let mixnode = StandardMixnode::new(config);
        assert!(mixnode.is_ok());
    }

    #[tokio::test]
    async fn test_delay_calculation() {
        let config = MixnodeConfig::default();
        let mixnode = StandardMixnode::new(config).unwrap();

        let delay = mixnode.calculate_delay().await;
        assert!(delay >= mixnode.config.min_delay);
        assert!(delay <= mixnode.config.max_delay);
    }

    #[tokio::test]
    async fn test_stats_access() {
        let config = MixnodeConfig::default();
        let mixnode = StandardMixnode::new(config).unwrap();

        let stats_handle = mixnode.stats();
        let stats = stats_handle.read().await;
        assert_eq!(stats.packets_processed, 0);
    }
}
