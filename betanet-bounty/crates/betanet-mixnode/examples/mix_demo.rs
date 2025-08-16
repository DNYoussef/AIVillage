//! Three-hop mixnet routing demonstration
//!
//! This example demonstrates packet routing through a 3-hop mixnet:
//! Client -> Mixnode 1 -> Mixnode 2 -> Mixnode 3 -> Destination
//!
//! Usage: cargo run --example mix_demo --no-default-features --features sphinx,cover-traffic

use std::net::SocketAddr;
use std::time::Duration;

use tokio::net::{TcpListener, TcpStream};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::time::{sleep, timeout};
use tracing::{debug, info, warn, error};

use betanet_mixnode::{
    config::MixnodeConfig,
    mixnode::StandardMixnode,
    Mixnode,
};

/// Demo configuration
#[derive(Debug, Clone)]
pub struct DemoConfig {
    /// Enable verbose logging
    pub verbose: bool,
    /// Number of test packets to send
    pub num_packets: usize,
    /// Delay between packets
    pub packet_delay: Duration,
    /// Test timeout
    pub timeout_duration: Duration,
}

impl Default for DemoConfig {
    fn default() -> Self {
        Self {
            verbose: true,
            num_packets: 10,
            packet_delay: Duration::from_millis(100),
            timeout_duration: Duration::from_secs(30),
        }
    }
}

/// Mixnode instance for the demo
pub struct DemoMixnode {
    /// Node ID
    pub id: String,
    /// Listen address
    pub address: SocketAddr,
    /// Mixnode implementation
    pub mixnode: StandardMixnode,
    /// Next hop address (None for exit node)
    pub next_hop: Option<SocketAddr>,
}

impl DemoMixnode {
    /// Create new demo mixnode
    pub fn new(id: String, address: SocketAddr, next_hop: Option<SocketAddr>) -> std::result::Result<Self, Box<dyn std::error::Error>> {
        let config = MixnodeConfig {
            listen_addr: address,
            enable_sphinx: true,
            enable_cover_traffic: false,
            ..Default::default()
        };

        let mixnode = StandardMixnode::new(config).map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;

        Ok(Self {
            id,
            address,
            mixnode,
            next_hop,
        })
    }

    /// Start the mixnode server
    pub async fn start(&mut self) -> std::result::Result<(), Box<dyn std::error::Error>> {
        info!("Starting mixnode {} on {}", self.id, self.address);

        // Start the mixnode
        self.mixnode.start().await?;

        // Start TCP listener for incoming packets
        let listener = TcpListener::bind(self.address).await?;
        info!("Mixnode {} listening on {}", self.id, self.address);

        let next_hop = self.next_hop;
        let _mixnode_ref = &self.mixnode;

        tokio::spawn(async move {
            loop {
                match listener.accept().await {
                    Ok((stream, peer_addr)) => {
                        debug!("Connection from {} to mixnode", peer_addr);

                        let next_hop_addr = next_hop;

                        tokio::spawn(async move {
                            if let Err(e) = handle_connection(stream, next_hop_addr).await {
                                warn!("Connection handling error: {}", e);
                            }
                        });
                    }
                    Err(e) => {
                        error!("Accept error: {}", e);
                        break;
                    }
                }
            }
        });

        Ok(())
    }
}

/// Handle incoming connection
async fn handle_connection(
    mut stream: TcpStream,
    next_hop: Option<SocketAddr>
) -> std::result::Result<(), std::io::Error> {
    let mut buffer = vec![0u8; 4096];

    match stream.read(&mut buffer).await {
        Ok(0) => {
            debug!("Connection closed by peer");
            return Ok(());
        }
        Ok(n) => {
            debug!("Received {} bytes", n);
            buffer.truncate(n);

            // Simple packet processing (for demo purposes)
            // In a real implementation, this would use Sphinx processing
            if let Some(next_addr) = next_hop {
                // Forward to next hop
                debug!("Forwarding to next hop: {}", next_addr);
                if let Err(e) = forward_packet(&buffer, next_addr).await {
                    warn!("Forward error: {}", e);
                    return Err(e.into());
                }
            } else {
                // Final destination - send response back
                debug!("Final destination reached, sending response");
                let response = b"MIXNET_SUCCESS";
                if let Err(e) = stream.write_all(response).await {
                    warn!("Response write error: {}", e);
                }
            }
        }
        Err(e) => {
            warn!("Read error: {}", e);
            return Err(e.into());
        }
    }

    Ok(())
}

/// Forward packet to next hop
async fn forward_packet(data: &[u8], next_hop: SocketAddr) -> std::result::Result<(), std::io::Error> {
    match TcpStream::connect(next_hop).await {
        Ok(mut stream) => {
            stream.write_all(data).await?;
            debug!("Packet forwarded to {}", next_hop);
            Ok(())
        }
        Err(e) => {
            Err(e)
        }
    }
}

/// Test client that sends packets through the mixnet
#[derive(Debug)]
pub struct DemoClient {
    /// Entry point address
    pub entry_address: SocketAddr,
}

impl DemoClient {
    /// Create new demo client
    pub fn new(entry_address: SocketAddr) -> Self {
        Self { entry_address }
    }

    /// Send test packet through mixnet
    pub async fn send_packet(&self, data: &[u8]) -> std::result::Result<Vec<u8>, std::io::Error> {
        info!("Sending packet through mixnet entry: {}", self.entry_address);

        match TcpStream::connect(self.entry_address).await {
            Ok(mut stream) => {
                // Send packet
                stream.write_all(data).await?;

                // Read response (with timeout)
                let mut response = vec![0u8; 1024];
                match timeout(Duration::from_secs(5), stream.read(&mut response)).await {
                    Ok(Ok(n)) => {
                        response.truncate(n);
                        info!("Received response: {} bytes", n);
                        Ok(response)
                    }
                    Ok(Err(e)) => Err(e),
                    Err(_) => Err(std::io::Error::new(std::io::ErrorKind::TimedOut, "Response timeout")),
                }
            }
            Err(e) => Err(e)
        }
    }
}

/// Run the 3-hop mixnet demonstration
#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .compact()
        .init();

    let config = DemoConfig::default();

    info!("🚀 Starting 3-Hop Mixnet Demonstration");
    info!("=====================================");

    // Define the 3-hop topology
    let mixnode1_addr: SocketAddr = "127.0.0.1:8001".parse().unwrap();
    let mixnode2_addr: SocketAddr = "127.0.0.1:8002".parse().unwrap();
    let mixnode3_addr: SocketAddr = "127.0.0.1:8003".parse().unwrap();

    info!("Topology: Client -> Mixnode1({}) -> Mixnode2({}) -> Mixnode3({})",
          mixnode1_addr, mixnode2_addr, mixnode3_addr);

    // Create mixnodes
    let mut mixnode1 = DemoMixnode::new(
        "MixNode1".to_string(),
        mixnode1_addr,
        Some(mixnode2_addr),
    )?;

    let mut mixnode2 = DemoMixnode::new(
        "MixNode2".to_string(),
        mixnode2_addr,
        Some(mixnode3_addr),
    )?;

    let mut mixnode3 = DemoMixnode::new(
        "MixNode3".to_string(),
        mixnode3_addr,
        None, // Exit node
    )?;

    // Start mixnodes (in reverse order so they're ready when needed)
    info!("🔧 Starting mixnodes...");
    mixnode3.start().await?;
    sleep(Duration::from_millis(100)).await;

    mixnode2.start().await?;
    sleep(Duration::from_millis(100)).await;

    mixnode1.start().await?;
    sleep(Duration::from_millis(100)).await;

    info!("✅ All mixnodes started successfully");

    // Create test client
    let client = DemoClient::new(mixnode1_addr);

    // Run packet tests
    info!("📡 Testing packet routing through 3-hop mixnet...");

    let mut success_count = 0;
    let mut total_packets = 0;

    for i in 0..config.num_packets {
        total_packets += 1;

        let test_data = format!("TEST_PACKET_{:03}", i).into_bytes();
        info!("Sending packet {}/{}: {:?}", i + 1, config.num_packets,
              String::from_utf8_lossy(&test_data));

        match client.send_packet(&test_data).await {
            Ok(response) => {
                let response_str = String::from_utf8_lossy(&response);
                if response_str.contains("MIXNET_SUCCESS") {
                    success_count += 1;
                    info!("✅ Packet {} succeeded: {}", i + 1, response_str);
                } else {
                    warn!("❌ Packet {} unexpected response: {}", i + 1, response_str);
                }
            }
            Err(e) => {
                warn!("❌ Packet {} failed: {}", i + 1, e);
            }
        }

        if i < config.num_packets - 1 {
            sleep(config.packet_delay).await;
        }
    }

    // Report results
    info!("📊 Demo Results:");
    info!("================");
    info!("Total packets sent: {}", total_packets);
    info!("Successful packets: {}", success_count);
    info!("Success rate: {:.1}%", (success_count as f64 / total_packets as f64) * 100.0);

    if success_count > 0 {
        info!("🎉 3-hop mixnet routing demonstration SUCCESSFUL!");
        info!("Packets successfully routed through:");
        info!("  Client -> MixNode1 -> MixNode2 -> MixNode3 -> Response");
    } else {
        warn!("⚠️  No packets successfully routed through mixnet");
    }

    // Performance testing
    info!("🏎️  Running performance test...");
    let start_time = std::time::Instant::now();
    let perf_packets = 100;
    let mut perf_success = 0;

    for i in 0..perf_packets {
        let data = format!("PERF_{:03}", i).into_bytes();
        if client.send_packet(&data).await.is_ok() {
            perf_success += 1;
        }
    }

    let elapsed = start_time.elapsed();
    let throughput = perf_success as f64 / elapsed.as_secs_f64();

    info!("Performance Results:");
    info!("- Packets: {}/{}", perf_success, perf_packets);
    info!("- Time: {:.2}s", elapsed.as_secs_f64());
    info!("- Throughput: {:.1} pkt/s", throughput);

    info!("🏁 Demo completed successfully!");
    Ok(())
}

/// High-performance pipeline demonstration
#[cfg(feature = "pipeline-demo")]
async fn demo_pipeline_performance() -> Result<()> {
    info!("🚀 Running pipeline performance demonstration");

    let rate_config = RateLimitingConfig {
        enabled: true,
        burst_capacity: 1000,
        sustained_rate: 500.0,
        output_rate: 25000.0, // Target 25k pkt/s
        shaping_buffer_size: 10000,
    };

    let mut pipeline = PacketPipeline::with_config(4, rate_config);
    pipeline.start().await?;

    let start = std::time::Instant::now();
    let test_packets = 50000;
    let mut sent = 0;

    // Submit test packets
    for i in 0..test_packets {
        let data = format!("PIPELINE_TEST_{}", i).into_bytes();
        let packet = betanet_mixnode::pipeline::PipelinePacket::new(data.into());

        if pipeline.submit_packet(packet).await.is_ok() {
            sent += 1;
        }

        if i % 1000 == 0 {
            tokio::task::yield_now().await;
        }
    }

    // Wait for processing
    sleep(Duration::from_secs(2)).await;

    let processed = pipeline.get_processed_packets(test_packets);
    let elapsed = start.elapsed();

    pipeline.stop().await?;

    let throughput = processed.len() as f64 / elapsed.as_secs_f64();

    info!("Pipeline Performance:");
    info!("- Sent: {} packets", sent);
    info!("- Processed: {} packets", processed.len());
    info!("- Time: {:.2}s", elapsed.as_secs_f64());
    info!("- Throughput: {:.0} pkt/s", throughput);
    info!("- Target: 25,000 pkt/s");
    info!("- Result: {}", if throughput >= 25000.0 { "✅ PASSED" } else { "❌ FAILED" });

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_demo_mixnode_creation() {
        let addr: SocketAddr = "127.0.0.1:9001".parse().unwrap();
        let next_hop: SocketAddr = "127.0.0.1:9002".parse().unwrap();

        let mixnode = DemoMixnode::new(
            "TestNode".to_string(),
            addr,
            Some(next_hop),
        );

        assert!(mixnode.is_ok());
        let node = mixnode.unwrap();
        assert_eq!(node.id, "TestNode");
        assert_eq!(node.address, addr);
        assert_eq!(node.next_hop, Some(next_hop));
    }

    #[tokio::test]
    async fn test_demo_client_creation() {
        let addr: SocketAddr = "127.0.0.1:9003".parse().unwrap();
        let client = DemoClient::new(addr);
        assert_eq!(client.entry_address, addr);
    }

    #[test]
    fn test_demo_config() {
        let config = DemoConfig::default();
        assert!(config.verbose);
        assert_eq!(config.num_packets, 10);
        assert!(config.timeout_duration > Duration::from_secs(0));
    }
}
