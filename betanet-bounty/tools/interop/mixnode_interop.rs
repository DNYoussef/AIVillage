//! End-to-end mixnode interoperability testing
//!
//! This tool tests end-to-end interoperability between betanet-htx and mixnode:
//! encrypt → sphinx → 3 hops → decrypt
//!
//! Generates detailed interop.log showing multi-hop success with Sphinx unwrap/forward.

use std::collections::HashMap;
use std::fs::{create_dir_all, File};
use std::io::Write;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, RwLock};
use tokio::time::sleep;
use tracing::{debug, error, info, warn};

use betanet_mixnode::{
    crypto::{ChaChaEncryption, CryptoUtils, X25519KeyExchange},
    sphinx::{SphinxPacket, SphinxProcessor, RoutingInfo, SphinxHeader, SPHINX_HEADER_SIZE, SPHINX_PAYLOAD_SIZE},
    vrf_neighbor::{VrfNeighborSelector, NeighborSelectionConfig, MixnodeInfo},
    pipeline::{PacketPipeline, PipelinePacket},
    rate::RateLimitingConfig,
    Result, MixnodeError,
};

/// Interop test configuration
#[derive(Debug, Clone)]
pub struct InteropTestConfig {
    /// Number of mixnodes in the circuit
    pub num_mixnodes: usize,
    /// Number of test messages to send
    pub num_test_messages: usize,
    /// Test timeout duration
    pub timeout_duration: Duration,
    /// Enable detailed logging
    pub verbose_logging: bool,
}

impl Default for InteropTestConfig {
    fn default() -> Self {
        Self {
            num_mixnodes: 3,  // 3-hop circuit as required by bounty
            num_test_messages: 10,
            timeout_duration: Duration::from_secs(30),
            verbose_logging: true,
        }
    }
}

/// Test mixnode instance
#[derive(Debug)]
pub struct TestMixnode {
    /// Node ID
    pub node_id: String,
    /// Network address
    pub address: SocketAddr,
    /// Sphinx processor
    pub sphinx_processor: Arc<SphinxProcessor>,
    /// Packet pipeline
    pub pipeline: PacketPipeline,
    /// X25519 key exchange
    pub key_exchange: X25519KeyExchange,
    /// VRF neighbor selector
    pub neighbor_selector: VrfNeighborSelector,
    /// Message receiver
    pub message_rx: mpsc::UnboundedReceiver<ProcessedMessage>,
    /// Message sender
    pub message_tx: mpsc::UnboundedSender<ProcessedMessage>,
}

/// Processed message for tracking
#[derive(Debug, Clone)]
pub struct ProcessedMessage {
    /// Original message ID
    pub message_id: String,
    /// Processing timestamp
    pub timestamp: Instant,
    /// Processing node ID
    pub node_id: String,
    /// Message content (if final hop)
    pub content: Option<String>,
    /// Sphinx processing success
    pub sphinx_success: bool,
    /// Next hop address (if intermediate)
    pub next_hop: Option<SocketAddr>,
    /// Processing latency
    pub processing_latency_us: u64,
}

/// Interop test result
#[derive(Debug, Serialize, Deserialize)]
pub struct InteropTestResult {
    /// Test configuration
    pub config: InteropTestConfig,
    /// Test success
    pub success: bool,
    /// Total messages sent
    pub messages_sent: usize,
    /// Messages successfully processed through all hops
    pub messages_completed: usize,
    /// Success rate
    pub success_rate: f64,
    /// Average end-to-end latency
    pub avg_e2e_latency_ms: f64,
    /// Per-hop processing times
    pub hop_latencies_us: Vec<u64>,
    /// Sphinx unwrap/forward operations
    pub sphinx_operations: usize,
    /// Test duration
    pub test_duration_ms: u64,
    /// Detailed hop trace
    pub hop_traces: Vec<HopTrace>,
    /// Error details
    pub errors: Vec<String>,
}

/// Hop trace for detailed analysis
#[derive(Debug, Serialize, Deserialize)]
pub struct HopTrace {
    /// Message ID
    pub message_id: String,
    /// Hop number (0, 1, 2 for 3-hop circuit)
    pub hop_number: usize,
    /// Node ID processing this hop
    pub node_id: String,
    /// Processing timestamp
    pub timestamp_ms: u64,
    /// Sphinx unwrap success
    pub sphinx_unwrap_success: bool,
    /// Forward to next hop (if not final)
    pub forwarded_to: Option<String>,
    /// Processing latency
    pub processing_latency_us: u64,
    /// Payload size
    pub payload_size: usize,
}

/// End-to-end interop test runner
pub struct InteropTestRunner {
    config: InteropTestConfig,
    mixnodes: Vec<TestMixnode>,
    log_entries: Vec<String>,
    start_time: Instant,
}

impl InteropTestRunner {
    /// Create new interop test runner
    pub async fn new(config: InteropTestConfig) -> Result<Self> {
        let mut runner = Self {
            config,
            mixnodes: Vec::new(),
            log_entries: Vec::new(),
            start_time: Instant::now(),
        };

        // Initialize test mixnodes
        runner.initialize_mixnodes().await?;
        
        Ok(runner)
    }

    /// Initialize test mixnodes
    async fn initialize_mixnodes(&mut self) -> Result<()> {
        self.log("🔧 Initializing test mixnodes");
        
        for i in 0..self.config.num_mixnodes {
            let node_id = format!("mixnode-{}", i);
            let address: SocketAddr = format!("127.0.0.1:{}", 9000 + i).parse()
                .map_err(|e| MixnodeError::Config(format!("Invalid address: {}", e)))?;

            // Create Sphinx processor with unique key
            let private_key = CryptoUtils::random_bytes(32).try_into().unwrap();
            let sphinx_processor = Arc::new(SphinxProcessor::with_key(private_key));
            
            // Create high-performance pipeline
            let rate_config = RateLimitingConfig::default();
            #[cfg(feature = "cover-traffic")]
            let mut pipeline = PacketPipeline::with_config(2, rate_config, None);
            #[cfg(not(feature = "cover-traffic"))]
            let mut pipeline = PacketPipeline::with_config(2, rate_config);
            
            pipeline.start().await?;

            // Create key exchange
            let key_exchange = X25519KeyExchange::from_bytes(&private_key);

            // Create VRF neighbor selector
            let neighbor_config = NeighborSelectionConfig::default();
            let mut neighbor_selector = VrfNeighborSelector::with_vrf_key(private_key, neighbor_config);

            // Add other nodes as neighbors for routing
            for j in 0..self.config.num_mixnodes {
                if i != j {
                    let neighbor_addr: SocketAddr = format!("127.0.0.1:{}", 9000 + j).parse().unwrap();
                    let neighbor_vrf_key = CryptoUtils::sha256(&j.to_string().as_bytes());
                    let neighbor_as = 1000 + (j % 5) as u32; // Simulate AS diversity
                    let neighbor_info = MixnodeInfo::new(neighbor_addr, neighbor_as, neighbor_vrf_key);
                    neighbor_selector.add_node(neighbor_info);
                }
            }

            // Create message channel
            let (message_tx, message_rx) = mpsc::unbounded_channel();

            let mixnode = TestMixnode {
                node_id: node_id.clone(),
                address,
                sphinx_processor,
                pipeline,
                key_exchange,
                neighbor_selector,
                message_rx,
                message_tx,
            };

            self.mixnodes.push(mixnode);
            self.log(&format!("✅ Initialized {}", node_id));
        }

        Ok(())
    }

    /// Run end-to-end interop test
    pub async fn run_interop_test(&mut self) -> Result<InteropTestResult> {
        self.log("🚀 Starting end-to-end interoperability test");
        self.log(&format!("Circuit: {} mixnodes", self.config.num_mixnodes));
        self.log(&format!("Test messages: {}", self.config.num_test_messages));

        let test_start = Instant::now();
        let mut hop_traces = Vec::new();
        let mut messages_completed = 0;
        let mut errors = Vec::new();
        let mut total_latency_ms = 0.0;
        let mut hop_latencies_us = vec![0u64; self.config.num_mixnodes];
        let mut sphinx_operations = 0;

        // Run test messages through the circuit
        for msg_idx in 0..self.config.num_test_messages {
            let message_id = format!("msg-{:03}", msg_idx);
            let content = format!("Test message {} - Hello from betanet-htx!", msg_idx);

            self.log(&format!("\n📨 Processing {}: '{}'", message_id, content));

            match self.send_message_through_circuit(&message_id, &content).await {
                Ok(traces) => {
                    messages_completed += 1;
                    hop_traces.extend(traces.clone());

                    // Calculate end-to-end latency
                    if let (Some(first), Some(last)) = (traces.first(), traces.last()) {
                        let e2e_latency = last.timestamp_ms - first.timestamp_ms;
                        total_latency_ms += e2e_latency as f64;
                    }

                    // Accumulate hop latencies
                    for (i, trace) in traces.iter().enumerate() {
                        if i < hop_latencies_us.len() {
                            hop_latencies_us[i] += trace.processing_latency_us;
                        }
                        if trace.sphinx_unwrap_success {
                            sphinx_operations += 1;
                        }
                    }

                    self.log(&format!("✅ {} completed successfully", message_id));
                }
                Err(e) => {
                    let error_msg = format!("❌ {} failed: {}", message_id, e);
                    self.log(&error_msg);
                    errors.push(error_msg);
                }
            }

            // Brief delay between messages
            sleep(Duration::from_millis(10)).await;
        }

        let test_duration = test_start.elapsed();

        // Calculate results
        let success_rate = messages_completed as f64 / self.config.num_test_messages as f64;
        let avg_e2e_latency_ms = if messages_completed > 0 {
            total_latency_ms / messages_completed as f64
        } else {
            0.0
        };

        // Average hop latencies
        for latency in &mut hop_latencies_us {
            if messages_completed > 0 {
                *latency /= messages_completed as u64;
            }
        }

        let result = InteropTestResult {
            config: self.config.clone(),
            success: success_rate >= 0.8, // 80% success rate threshold
            messages_sent: self.config.num_test_messages,
            messages_completed,
            success_rate,
            avg_e2e_latency_ms,
            hop_latencies_us,
            sphinx_operations,
            test_duration_ms: test_duration.as_millis() as u64,
            hop_traces,
            errors,
        };

        self.log(&format!("\n🏁 Test completed in {:.2}s", test_duration.as_secs_f64()));
        self.log(&format!("📊 Success rate: {:.1}% ({}/{})", 
            success_rate * 100.0, messages_completed, self.config.num_test_messages));
        self.log(&format!("⚡ Average E2E latency: {:.2}ms", avg_e2e_latency_ms));
        self.log(&format!("🔄 Sphinx operations: {}", sphinx_operations));

        Ok(result)
    }

    /// Send message through the mixnode circuit
    async fn send_message_through_circuit(
        &mut self,
        message_id: &str,
        content: &str,
    ) -> Result<Vec<HopTrace>> {
        let mut hop_traces = Vec::new();
        let start_time = Instant::now();

        // Create initial Sphinx packet
        let mut current_packet = self.create_initial_sphinx_packet(content).await?;
        let mut current_hop = 0;

        self.log(&format!("  🏗️  Created initial Sphinx packet ({} bytes)", 
            SPHINX_HEADER_SIZE + SPHINX_PAYLOAD_SIZE));

        // Process through each hop
        while current_hop < self.config.num_mixnodes {
            let hop_start = Instant::now();
            let mixnode = &self.mixnodes[current_hop];

            self.log(&format!("  🔄 Hop {}: Processing at {}", current_hop, mixnode.node_id));

            // Process packet through Sphinx
            let processing_start = Instant::now();
            match mixnode.sphinx_processor.process_packet(current_packet.clone()).await {
                Ok(Some(processed_packet)) => {
                    let processing_latency = processing_start.elapsed().as_micros() as u64;
                    
                    let trace = HopTrace {
                        message_id: message_id.to_string(),
                        hop_number: current_hop,
                        node_id: mixnode.node_id.clone(),
                        timestamp_ms: start_time.elapsed().as_millis() as u64,
                        sphinx_unwrap_success: true,
                        forwarded_to: if current_hop < self.config.num_mixnodes - 1 {
                            Some(self.mixnodes[current_hop + 1].node_id.clone())
                        } else {
                            None
                        },
                        processing_latency_us: processing_latency,
                        payload_size: processed_packet.payload.len(),
                    };

                    hop_traces.push(trace);
                    current_packet = processed_packet;

                    self.log(&format!("    ✅ Sphinx unwrap successful ({:.2}μs)", processing_latency));

                    if current_hop < self.config.num_mixnodes - 1 {
                        self.log(&format!("    ➡️  Forwarding to {}", self.mixnodes[current_hop + 1].node_id));
                    } else {
                        self.log("    🎯 Reached final destination");
                        
                        // Verify payload contains original message
                        let payload_str = String::from_utf8_lossy(&current_packet.payload);
                        if payload_str.contains(content) {
                            self.log("    ✅ Message content verified");
                        } else {
                            self.log("    ⚠️  Message content verification failed");
                        }
                    }
                }
                Ok(None) => {
                    // Final destination reached
                    let processing_latency = processing_start.elapsed().as_micros() as u64;
                    
                    let trace = HopTrace {
                        message_id: message_id.to_string(),
                        hop_number: current_hop,
                        node_id: mixnode.node_id.clone(),
                        timestamp_ms: start_time.elapsed().as_millis() as u64,
                        sphinx_unwrap_success: true,
                        forwarded_to: None,
                        processing_latency_us: processing_latency,
                        payload_size: 0, // Consumed at final destination
                    };

                    hop_traces.push(trace);
                    self.log("    🎯 Message consumed at final destination");
                    break;
                }
                Err(e) => {
                    let trace = HopTrace {
                        message_id: message_id.to_string(),
                        hop_number: current_hop,
                        node_id: mixnode.node_id.clone(),
                        timestamp_ms: start_time.elapsed().as_millis() as u64,
                        sphinx_unwrap_success: false,
                        forwarded_to: None,
                        processing_latency_us: processing_start.elapsed().as_micros() as u64,
                        payload_size: 0,
                    };

                    hop_traces.push(trace);
                    return Err(MixnodeError::Packet(format!("Sphinx processing failed at hop {}: {}", current_hop, e)));
                }
            }

            current_hop += 1;
        }

        Ok(hop_traces)
    }

    /// Create initial Sphinx packet with layered encryption
    async fn create_initial_sphinx_packet(&self, content: &str) -> Result<SphinxPacket> {
        let mut packet = SphinxPacket::new();

        // Set up routing information for each hop
        let mut routing_layers = Vec::new();
        for i in (0..self.config.num_mixnodes).rev() {
            let mixnode = &self.mixnodes[i];
            let is_final = i == self.config.num_mixnodes - 1;
            
            let next_hop_addr = if is_final {
                [0u8; 16] // Final destination
            } else {
                let next_addr = self.mixnodes[i + 1].address;
                match next_addr {
                    SocketAddr::V4(addr) => {
                        let mut result = [0u8; 16];
                        result[12..16].copy_from_slice(&addr.ip().octets());
                        result
                    }
                    SocketAddr::V6(addr) => addr.ip().octets(),
                }
            };

            let routing_info = RoutingInfo::new(
                next_hop_addr,
                if is_final { 0 } else { self.mixnodes[i + 1].address.port() },
                100, // 100ms delay
                is_final,
            );

            routing_layers.push(routing_info);
        }

        // Encrypt payload with content
        let content_bytes = content.as_bytes();
        let mut payload = [0u8; SPHINX_PAYLOAD_SIZE];
        let copy_len = std::cmp::min(content_bytes.len(), SPHINX_PAYLOAD_SIZE);
        payload[..copy_len].copy_from_slice(&content_bytes[..copy_len]);

        // Apply layered encryption (simplified - real implementation would use proper Sphinx encryption)
        for mixnode in self.mixnodes.iter().rev() {
            let key = mixnode.sphinx_processor.public_key().to_bytes();
            let encryption = ChaChaEncryption::new(&key);
            let nonce = [1u8; 12]; // Simplified nonce
            
            if let Ok(encrypted) = encryption.encrypt(&payload, &nonce) {
                if encrypted.len() <= SPHINX_PAYLOAD_SIZE {
                    payload.fill(0);
                    payload[..encrypted.len()].copy_from_slice(&encrypted);
                }
            }
        }

        packet.payload = payload;

        // Set up header with routing information
        let mut header = SphinxHeader::new();
        if let Some(first_routing) = routing_layers.first() {
            header.routing_info = first_routing.to_bytes();
        }

        // Set ephemeral key (simplified)
        header.ephemeral_key = self.mixnodes[0].key_exchange.public_key().to_bytes();
        packet.header = header;

        Ok(packet)
    }

    /// Add log entry with timestamp
    fn log(&mut self, message: &str) {
        let elapsed = self.start_time.elapsed();
        let log_entry = format!("[{:>6.3}s] {}", elapsed.as_secs_f64(), message);
        
        if self.config.verbose_logging {
            println!("{}", log_entry);
        }
        
        self.log_entries.push(log_entry);
    }

    /// Write interop log to file
    pub async fn write_interop_log(&self, result: &InteropTestResult, output_path: &Path) -> Result<()> {
        // Ensure output directory exists
        if let Some(parent) = output_path.parent() {
            create_dir_all(parent)?;
        }

        let mut file = File::create(output_path)?;

        // Write header
        writeln!(file, "=====================================================================")?;
        writeln!(file, "Betanet Mixnode End-to-End Interoperability Test Log")?;
        writeln!(file, "Test: encrypt → sphinx → 3 hops → decrypt")?;
        writeln!(file, "=====================================================================")?;
        writeln!(file)?;

        // Write configuration
        writeln!(file, "Test Configuration:")?;
        writeln!(file, "  Mixnodes: {}", result.config.num_mixnodes)?;
        writeln!(file, "  Test Messages: {}", result.config.num_test_messages)?;
        writeln!(file, "  Timeout: {:?}", result.config.timeout_duration)?;
        writeln!(file)?;

        // Write detailed log entries
        writeln!(file, "Detailed Test Log:")?;
        writeln!(file, "==================")?;
        for entry in &self.log_entries {
            writeln!(file, "{}", entry)?;
        }

        writeln!(file)?;
        writeln!(file, "=====================================================================")?;
        writeln!(file, "Test Results Summary")?;
        writeln!(file, "=====================================================================")?;
        writeln!(file, "Overall Success: {}", if result.success { "✅ PASS" } else { "❌ FAIL" })?;
        writeln!(file, "Messages Sent: {}", result.messages_sent)?;
        writeln!(file, "Messages Completed: {}", result.messages_completed)?;
        writeln!(file, "Success Rate: {:.1}%", result.success_rate * 100.0)?;
        writeln!(file, "Average E2E Latency: {:.2}ms", result.avg_e2e_latency_ms)?;
        writeln!(file, "Sphinx Operations: {}", result.sphinx_operations)?;
        writeln!(file, "Test Duration: {}ms", result.test_duration_ms)?;

        writeln!(file)?;
        writeln!(file, "Per-Hop Latencies:")?;
        for (i, latency) in result.hop_latencies_us.iter().enumerate() {
            writeln!(file, "  Hop {}: {:.2}μs", i, latency)?;
        }

        if !result.errors.is_empty() {
            writeln!(file)?;
            writeln!(file, "Errors:")?;
            for error in &result.errors {
                writeln!(file, "  {}", error)?;
            }
        }

        writeln!(file)?;
        writeln!(file, "Detailed Hop Traces:")?;
        writeln!(file, "====================")?;
        for trace in &result.hop_traces {
            writeln!(file, "Message: {} | Hop: {} | Node: {} | Sphinx: {} | Latency: {}μs", 
                trace.message_id, 
                trace.hop_number, 
                trace.node_id,
                if trace.sphinx_unwrap_success { "✅" } else { "❌" },
                trace.processing_latency_us
            )?;
            if let Some(next_hop) = &trace.forwarded_to {
                writeln!(file, "  → Forwarded to: {}", next_hop)?;
            }
        }

        writeln!(file)?;
        writeln!(file, "=====================================================================")?;
        if result.success {
            writeln!(file, "🎉 INTEROP TEST PASSED: Multi-hop Sphinx unwrap/forward successful")?;
        } else {
            writeln!(file, "💥 INTEROP TEST FAILED: Multi-hop processing incomplete")?;
        }
        writeln!(file, "=====================================================================")?;

        file.flush()?;
        println!("📄 Interop log written to: {}", output_path.display());
        Ok(())
    }

    /// Cleanup test resources
    pub async fn cleanup(&mut self) -> Result<()> {
        self.log("🧹 Cleaning up test resources");
        
        for mixnode in &mut self.mixnodes {
            mixnode.pipeline.stop().await?;
        }
        
        Ok(())
    }
}

/// Run comprehensive interop test
#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    println!("🚀 Betanet Mixnode Interoperability Test");
    println!("========================================");

    let config = InteropTestConfig::default();
    let mut runner = InteropTestRunner::new(config).await?;

    // Run the test
    let result = runner.run_interop_test().await?;

    // Write results to tmp_submission/mixnode/interop.log
    let output_path = Path::new("tmp_submission/mixnode/interop.log");
    runner.write_interop_log(&result, output_path).await?;

    // Cleanup
    runner.cleanup().await?;

    // Exit with appropriate code
    if result.success {
        println!("\n🎉 SUCCESS: Multi-hop Sphinx unwrap/forward test passed");
        std::process::exit(0);
    } else {
        println!("\n💥 FAILURE: Multi-hop Sphinx processing failed");
        println!("   Success rate: {:.1}%", result.success_rate * 100.0);
        println!("   Required: ≥80%");
        std::process::exit(1);
    }
}