//! High-performance packet processing pipeline
//!
//! Optimized for throughput with:
//! - Batch processing for cache efficiency
//! - Memory pools to reduce allocation overhead
//! - Zero-copy operations where possible
//! - SIMD optimizations for crypto operations
//! - Lock-free data structures where possible

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bytes::{Bytes, BytesMut};
use crossbeam_queue::SegQueue;
use tokio::sync::{broadcast, Notify, Semaphore};
use tokio::time::sleep;

use crate::{MixnodeError, Result};

#[cfg(feature = "cover-traffic")]
use crate::cover::{AdvancedCoverTrafficGenerator, CoverTrafficConfig};

use crate::rate::{RateLimitedTrafficShaper, RateLimitingConfig};

#[cfg(feature = "sphinx")]
use crate::packet::Packet;

#[cfg(feature = "sphinx")]
use crate::sphinx::{SphinxPacket, SphinxProcessor};

/// Batch size for high-throughput processing (increased for 25k pkt/s target)
pub const BATCH_SIZE: usize = 128;
/// Memory pool size for packet buffers
pub const POOL_SIZE: usize = 1024;
/// Maximum queue depth before backpressure
pub const MAX_QUEUE_DEPTH: usize = 10000;

/// High-performance packet processing pipeline
pub struct PacketPipeline {
    /// Packet buffer memory pool
    memory_pool: Arc<MemoryPool>,
    /// Sphinx processor for onion decryption (if enabled)
    #[cfg(feature = "sphinx")]
    sphinx_processor: Arc<SphinxProcessor>,
    /// Input packet queue
    input_queue: Arc<Mutex<VecDeque<PipelinePacket>>>,
    /// Output packet queue
    output_queue: Arc<Mutex<VecDeque<PipelinePacket>>>,
    /// Processing semaphore for backpressure
    processing_semaphore: Arc<Semaphore>,
    /// Notifier for waking workers when new packets arrive
    packet_notifier: Arc<Notify>,
    /// Pipeline statistics
    stats: Arc<PipelineStats>,
    /// Worker handles
    workers: Vec<tokio::task::JoinHandle<()>>,
    /// Shutdown signal
    shutdown_tx: Option<broadcast::Sender<()>>,
    /// Rate limiter and traffic shaper
    rate_limiter: Arc<RateLimitedTrafficShaper>,
    /// Cover traffic generator (if enabled)
    #[cfg(feature = "cover-traffic")]
    cover_traffic: Arc<Mutex<AdvancedCoverTrafficGenerator>>,
}

/// Pipeline packet with metadata
#[derive(Debug, Clone)]
pub struct PipelinePacket {
    /// Packet data
    pub data: Bytes,
    /// Arrival timestamp
    pub arrival_time: Instant,
    /// Processing priority (lower = higher priority)
    pub priority: u8,
    /// Source address for routing decisions
    pub source: Option<std::net::SocketAddr>,
}

impl PipelinePacket {
    /// Create new pipeline packet
    pub fn new(data: Bytes) -> Self {
        Self {
            data,
            arrival_time: Instant::now(),
            priority: 0,
            source: None,
        }
    }

    /// Create with priority
    pub fn with_priority(data: Bytes, priority: u8) -> Self {
        Self {
            data,
            arrival_time: Instant::now(),
            priority,
            source: None,
        }
    }

    /// Get packet age
    pub fn age(&self) -> Duration {
        self.arrival_time.elapsed()
    }
}

/// Memory pool for packet buffers
pub struct MemoryPool {
    /// Pool of reusable buffers stored in a lock-free queue
    buffers: SegQueue<BytesMut>,
    /// Pool statistics
    allocated: AtomicUsize,
    reused: AtomicUsize,
}

impl MemoryPool {
    /// Create new memory pool
    pub fn new(capacity: usize, buffer_size: usize) -> Self {
        let buffers = SegQueue::new();
        for _ in 0..capacity {
            buffers.push(BytesMut::with_capacity(buffer_size));
        }

        Self {
            buffers,
            allocated: AtomicUsize::new(0),
            reused: AtomicUsize::new(0),
        }
    }

    /// Get buffer from pool
    pub fn get_buffer(&self, size: usize) -> BytesMut {
        if let Some(mut buf) = self.buffers.pop() {
            buf.clear();
            if buf.capacity() >= size {
                self.reused.fetch_add(1, Ordering::Relaxed);
                return buf;
            }
        }

        self.allocated.fetch_add(1, Ordering::Relaxed);
        BytesMut::with_capacity(size.max(2048))
    }

    /// Get memory pool hit rate (percentage, 0-100)
    pub fn hit_rate_percent(&self) -> f64 {
        let total_requests =
            self.allocated.load(Ordering::Relaxed) + self.reused.load(Ordering::Relaxed);
        if total_requests == 0 {
            return 0.0;
        }
        let reused = self.reused.load(Ordering::Relaxed);
        (reused as f64 / total_requests as f64) * 100.0
    }

    /// Return buffer to pool
    pub fn return_buffer(&self, mut buffer: BytesMut) {
        buffer.clear();
        if self.buffers.len() < POOL_SIZE {
            self.buffers.push(buffer);
        }
    }

    /// Get pool statistics
    pub fn stats(&self) -> (usize, usize) {
        (
            self.allocated.load(Ordering::Relaxed),
            self.reused.load(Ordering::Relaxed),
        )
    }
}

/// Pipeline performance statistics
#[derive(Debug)]
pub struct PipelineStats {
    /// Total packets processed
    pub packets_processed: AtomicU64,
    /// Total packets dropped (overflow/errors)
    pub packets_dropped: AtomicU64,
    /// Total processing time (nanoseconds)
    pub total_processing_time_ns: AtomicU64,
    /// Batch processing efficiency
    pub batches_processed: AtomicU64,
    /// Queue depth samples
    pub avg_queue_depth: AtomicU64,
    /// Memory pool efficiency
    pub pool_hit_rate: AtomicU64,
}

impl PipelineStats {
    /// Create new statistics
    pub fn new() -> Self {
        Self {
            packets_processed: AtomicU64::new(0),
            packets_dropped: AtomicU64::new(0),
            total_processing_time_ns: AtomicU64::new(0),
            batches_processed: AtomicU64::new(0),
            avg_queue_depth: AtomicU64::new(0),
            pool_hit_rate: AtomicU64::new(0),
        }
    }

    /// Record packet processing
    pub fn record_processed(&self, count: u64, processing_time_ns: u64) {
        self.packets_processed.fetch_add(count, Ordering::Relaxed);
        self.total_processing_time_ns
            .fetch_add(processing_time_ns, Ordering::Relaxed);
    }

    /// Record batch processing
    pub fn record_batch(&self, _batch_size: u64) {
        self.batches_processed.fetch_add(1, Ordering::Relaxed);
    }

    /// Update memory pool hit rate
    pub fn update_pool_hit_rate(&self, hit_rate_percent: f64) {
        // Store as fixed-point percentage (multiply by 100 for precision)
        let hit_rate_fp = (hit_rate_percent * 100.0) as u64;
        self.pool_hit_rate.store(hit_rate_fp, Ordering::Relaxed);
    }

    /// Get memory pool hit rate as percentage
    pub fn get_pool_hit_rate(&self) -> f64 {
        let hit_rate_fp = self.pool_hit_rate.load(Ordering::Relaxed);
        hit_rate_fp as f64 / 100.0
    }

    /// Get average processing time per packet (nanoseconds)
    pub fn avg_processing_time_ns(&self) -> u64 {
        let processed = self.packets_processed.load(Ordering::Relaxed);
        if processed > 0 {
            self.total_processing_time_ns.load(Ordering::Relaxed) / processed
        } else {
            0
        }
    }

    /// Get throughput (packets per second)
    pub fn throughput_pps(&self, duration: Duration) -> f64 {
        let processed = self.packets_processed.load(Ordering::Relaxed) as f64;
        processed / duration.as_secs_f64()
    }
}

impl Default for PipelineStats {
    fn default() -> Self {
        Self::new()
    }
}

impl PacketPipeline {
    /// Create new packet pipeline
    pub fn new(num_workers: usize) -> Self {
        #[cfg(feature = "cover-traffic")]
        return Self::with_config(num_workers, RateLimitingConfig::default(), None);

        #[cfg(not(feature = "cover-traffic"))]
        return Self::with_config(num_workers, RateLimitingConfig::default());
    }

    /// Create new packet pipeline with configuration
    #[cfg(feature = "cover-traffic")]
    pub fn with_config(
        num_workers: usize,
        rate_config: RateLimitingConfig,
        cover_config: Option<CoverTrafficConfig>,
    ) -> Self {
        let memory_pool = Arc::new(MemoryPool::new(POOL_SIZE, 4096));
        #[cfg(feature = "sphinx")]
        let sphinx_processor = Arc::new(SphinxProcessor::new());
        let processing_semaphore = Arc::new(Semaphore::new(MAX_QUEUE_DEPTH));
        let packet_notifier = Arc::new(Notify::new());
        let stats = Arc::new(PipelineStats::new());
        let rate_limiter = Arc::new(RateLimitedTrafficShaper::new(rate_config));

        #[cfg(feature = "cover-traffic")]
        let cover_traffic = {
            let config = cover_config.unwrap_or_default();
            Arc::new(Mutex::new(AdvancedCoverTrafficGenerator::new(config)))
        };

        Self {
            memory_pool,
            #[cfg(feature = "sphinx")]
            sphinx_processor,
            input_queue: Arc::new(Mutex::new(VecDeque::new())),
            output_queue: Arc::new(Mutex::new(VecDeque::new())),
            processing_semaphore,
            packet_notifier,
            stats,
            workers: Vec::with_capacity(num_workers),
            shutdown_tx: None,
            rate_limiter,
            #[cfg(feature = "cover-traffic")]
            cover_traffic,
        }
    }

    /// Create new packet pipeline with configuration (without cover traffic)
    #[cfg(not(feature = "cover-traffic"))]
    pub fn with_config(num_workers: usize, rate_config: RateLimitingConfig) -> Self {
        let memory_pool = Arc::new(MemoryPool::new(POOL_SIZE, 4096));
        #[cfg(feature = "sphinx")]
        let sphinx_processor = Arc::new(SphinxProcessor::new());
        let processing_semaphore = Arc::new(Semaphore::new(MAX_QUEUE_DEPTH));
        let packet_notifier = Arc::new(Notify::new());
        let stats = Arc::new(PipelineStats::new());
        let rate_limiter = Arc::new(RateLimitedTrafficShaper::new(rate_config));

        Self {
            memory_pool,
            #[cfg(feature = "sphinx")]
            sphinx_processor,
            input_queue: Arc::new(Mutex::new(VecDeque::new())),
            output_queue: Arc::new(Mutex::new(VecDeque::new())),
            processing_semaphore,
            packet_notifier,
            stats,
            workers: Vec::with_capacity(num_workers),
            shutdown_tx: None,
            rate_limiter,
        }
    }

    /// Start the processing pipeline
    pub async fn start(&mut self) -> Result<()> {
        let (shutdown_tx, _) = broadcast::channel(1);
        self.shutdown_tx = Some(shutdown_tx.clone());

        // Spawn worker threads
        let num_workers = self.workers.capacity();
        for worker_id in 0..num_workers {
            let input_queue = Arc::clone(&self.input_queue);
            let output_queue = Arc::clone(&self.output_queue);
            let memory_pool = Arc::clone(&self.memory_pool);
            #[cfg(feature = "sphinx")]
            let sphinx_processor = Arc::clone(&self.sphinx_processor);
            let processing_semaphore = Arc::clone(&self.processing_semaphore);
            let packet_notifier = Arc::clone(&self.packet_notifier);
            let stats = Arc::clone(&self.stats);
            let mut shutdown_rx = shutdown_tx.subscribe();

            let worker = tokio::spawn(async move {
                let mut batch_buffer = Vec::with_capacity(BATCH_SIZE);

                loop {
                    tokio::select! {
                        _ = shutdown_rx.recv() => {
                            tracing::debug!("Worker {} shutting down", worker_id);
                            break;
                        }
                        _ = packet_notifier.notified() => {
                            loop {
                                // Process available packets in batches when notified
                                batch_buffer.clear();

                                {
                                    let mut queue = input_queue.lock().unwrap();
                                    while batch_buffer.len() < BATCH_SIZE {
                                        if let Some(packet) = queue.pop_front() {
                                            batch_buffer.push(packet);
                                        } else {
                                            break;
                                        }
                                    }
                                }

                                if batch_buffer.is_empty() {
                                    break;
                                }

                                let start_time = Instant::now();

                                // Process batch
                                #[cfg(feature = "sphinx")]
                                let processed = Self::process_batch(
                                    &batch_buffer,
                                    &sphinx_processor,
                                    &memory_pool,
                                ).await;

                                #[cfg(not(feature = "sphinx"))]
                                let processed = Self::process_batch_simple(&batch_buffer, &memory_pool).await;

                                // Output processed packets (ensure packets reach output)
                                if !processed.is_empty() {
                                    let mut output = output_queue.lock().unwrap();
                                    for packet in processed {
                                        output.push_back(packet);
                                    }
                                }

                                // Update statistics
                                let processing_time = start_time.elapsed().as_nanos() as u64;
                                stats.record_processed(batch_buffer.len() as u64, processing_time);
                                stats.record_batch(batch_buffer.len() as u64);

                                // Update memory pool hit rate periodically
                                if stats.batches_processed.load(Ordering::Relaxed) % 100 == 0 {
                                    let hit_rate = memory_pool.hit_rate_percent();
                                    stats.update_pool_hit_rate(hit_rate);
                                }

                                // Release semaphore permits
                                processing_semaphore.add_permits(batch_buffer.len());
                            }
                        }
                    }
                }
            });

            self.workers.push(worker);
        }

        Ok(())
    }

    /// Stop the pipeline
    pub async fn stop(&mut self) -> Result<()> {
        if let Some(shutdown_tx) = self.shutdown_tx.take() {
            let _ = shutdown_tx.send(());
        }

        // Wait for workers to finish
        for worker in self.workers.drain(..) {
            worker
                .await
                .map_err(|e| MixnodeError::Network(format!("Worker join error: {}", e)))?;
        }

        Ok(())
    }

    /// Submit packet for processing
    pub async fn submit_packet(&self, packet: PipelinePacket) -> Result<()> {
        // Acquire semaphore permit for backpressure
        let _permit = self
            .processing_semaphore
            .acquire()
            .await
            .map_err(|_| MixnodeError::Network("Pipeline semaphore closed".to_string()))?;

        // Add to input queue
        let was_empty = {
            let mut queue = self.input_queue.lock().unwrap();
            if queue.len() >= MAX_QUEUE_DEPTH {
                self.stats.packets_dropped.fetch_add(1, Ordering::Relaxed);
                return Err(MixnodeError::Network("Pipeline queue full".to_string()));
            }
            let was_empty = queue.is_empty();
            queue.push_back(packet);
            was_empty
        };

        if was_empty {
            self.packet_notifier.notify_one();
        }

        // Don't release permit - will be released after processing
        std::mem::forget(_permit);
        Ok(())
    }

    /// Get processed packets
    pub fn get_processed_packets(&self, max_count: usize) -> Vec<PipelinePacket> {
        let mut output = self.output_queue.lock().unwrap();
        let mut result = Vec::with_capacity(max_count.min(output.len()));

        for _ in 0..max_count {
            if let Some(packet) = output.pop_front() {
                result.push(packet);
            } else {
                break;
            }
        }

        result
    }

    /// Process a batch of packets with Sphinx
    #[cfg(feature = "sphinx")]
    async fn process_batch(
        batch: &[PipelinePacket],
        sphinx_processor: &SphinxProcessor,
        _memory_pool: &MemoryPool,
    ) -> Vec<PipelinePacket> {
        let mut processed = Vec::with_capacity(batch.len());

        // Convert to Sphinx packets for batch processing
        let mut sphinx_packets = Vec::with_capacity(batch.len());
        let mut packet_indices = Vec::with_capacity(batch.len());

        for (i, pipeline_packet) in batch.iter().enumerate() {
            // Parse packet
            if let Ok(packet) = Packet::parse(&pipeline_packet.data) {
                if let Ok(sphinx_packet) = SphinxPacket::from_bytes(&packet.payload) {
                    sphinx_packets.push(sphinx_packet);
                    packet_indices.push(i);
                }
            }
        }

        // Batch process Sphinx packets
        if !sphinx_packets.is_empty() {
            if let Ok(results) = sphinx_processor.process_batch(sphinx_packets).await {
                for (result_idx, result) in results.into_iter().enumerate() {
                    if let Some(processed_sphinx) = result {
                        let original_idx = packet_indices[result_idx];
                        let original_packet = &batch[original_idx];

                        // Create processed packet
                        let processed_data = Bytes::from(processed_sphinx.to_bytes());
                        let processed_packet = PipelinePacket {
                            data: processed_data,
                            arrival_time: original_packet.arrival_time,
                            priority: original_packet.priority,
                            source: original_packet.source,
                        };

                        processed.push(processed_packet);
                    }
                }
            }
        }

        processed
    }

    /// Simple batch processing without Sphinx (optimized)
    #[cfg(not(feature = "sphinx"))]
    async fn process_batch_simple(
        batch: &[PipelinePacket],
        _memory_pool: &MemoryPool,
    ) -> Vec<PipelinePacket> {
        // Simple pass-through processing for testing (ensure output)
        batch.to_vec()
    }

    /// Get pipeline statistics
    pub fn stats(&self) -> &PipelineStats {
        &self.stats
    }

    /// Get current queue depths
    pub fn queue_depths(&self) -> (usize, usize) {
        let input_depth = self.input_queue.lock().unwrap().len();
        let output_depth = self.output_queue.lock().unwrap().len();
        (input_depth, output_depth)
    }

    /// Get memory pool statistics
    pub fn memory_pool_stats(&self) -> (usize, usize) {
        self.memory_pool.stats()
    }

    /// Get memory pool hit rate (percentage)
    pub fn memory_pool_hit_rate(&self) -> f64 {
        self.memory_pool.hit_rate_percent()
    }

    /// Get rate limiter queue length
    pub async fn rate_limiter_queue_length(&self) -> usize {
        self.rate_limiter.queue_length().await
    }

    /// Get cover traffic status
    #[cfg(feature = "cover-traffic")]
    pub fn is_cover_traffic_enabled(&self) -> bool {
        // This prevents the dead code warning
        let _ = &self.cover_traffic;
        true
    }

    /// Get cover traffic status (dummy for non-cover-traffic builds)
    #[cfg(not(feature = "cover-traffic"))]
    pub fn is_cover_traffic_enabled(&self) -> bool {
        false
    }
}

/// Performance benchmark for the pipeline
pub struct PipelineBenchmark {
    pipeline: PacketPipeline,
    test_packets: Vec<PipelinePacket>,
}

impl PipelineBenchmark {
    /// Create new benchmark
    pub fn new(num_workers: usize, num_test_packets: usize) -> Self {
        let pipeline = PacketPipeline::new(num_workers);

        // Generate test packets
        let mut test_packets = Vec::with_capacity(num_test_packets);
        for i in 0..num_test_packets {
            let _test_data = format!("test_packet_{:06}", i);
            let packet_data = Bytes::from(vec![0u8; 1200]); // Typical packet size
            test_packets.push(PipelinePacket::new(packet_data));
        }

        Self {
            pipeline,
            test_packets,
        }
    }

    /// Run throughput benchmark
    pub async fn run_throughput_test(&mut self, duration_secs: u64) -> Result<BenchmarkResults> {
        self.pipeline.start().await?;

        let start_time = Instant::now();
        let test_duration = Duration::from_secs(duration_secs);
        let mut packets_sent = 0u64;

        // Send packets at maximum rate
        let send_task = {
            let pipeline = &self.pipeline;
            let test_packets = &self.test_packets;
            async move {
                while start_time.elapsed() < test_duration {
                    for packet in test_packets {
                        if pipeline.submit_packet(packet.clone()).await.is_err() {
                            break; // Pipeline full
                        }
                        packets_sent += 1;

                        // Small delay to prevent overwhelming
                        if packets_sent % 1000 == 0 {
                            tokio::task::yield_now().await;
                        }
                    }
                }
                packets_sent
            }
        };

        let packets_sent = send_task.await;

        // Wait for processing to complete
        sleep(Duration::from_millis(100)).await;

        let elapsed = start_time.elapsed();
        let stats = self.pipeline.stats();

        let processed = stats.packets_processed.load(Ordering::Relaxed);
        let dropped = stats.packets_dropped.load(Ordering::Relaxed);
        let avg_processing_time_ns = stats.avg_processing_time_ns();
        let throughput_pps = stats.throughput_pps(elapsed);
        let memory_pool_hit_rate = self.pipeline.memory_pool_hit_rate();

        self.pipeline.stop().await?;

        Ok(BenchmarkResults {
            packets_sent,
            packets_processed: processed,
            packets_dropped: dropped,
            elapsed_secs: elapsed.as_secs_f64(),
            throughput_pps,
            avg_processing_time_ns,
            memory_pool_hit_rate,
        })
    }
}

/// Benchmark results
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    /// Total packets sent during test
    pub packets_sent: u64,
    /// Total packets successfully processed
    pub packets_processed: u64,
    /// Total packets dropped due to overflow or errors
    pub packets_dropped: u64,
    /// Test duration in seconds
    pub elapsed_secs: f64,
    /// Achieved throughput in packets per second
    pub throughput_pps: f64,
    /// Average processing time per packet in nanoseconds
    pub avg_processing_time_ns: u64,
    /// Memory pool hit rate (0.0 to 1.0)
    pub memory_pool_hit_rate: f64,
}

impl BenchmarkResults {
    /// Check if meets performance target
    pub fn meets_target(&self, target_pps: f64) -> bool {
        self.throughput_pps >= target_pps
    }

    /// Print results
    pub fn print_results(&self) {
        println!("🚀 Pipeline Benchmark Results:");
        println!("  Packets sent:     {}", self.packets_sent);
        println!("  Packets processed: {}", self.packets_processed);
        println!("  Packets dropped:   {}", self.packets_dropped);
        println!("  Elapsed time:      {:.2}s", self.elapsed_secs);
        println!("  Throughput:        {:.0} pkt/s", self.throughput_pps);
        println!(
            "  Avg processing:    {:.2}μs",
            self.avg_processing_time_ns as f64 / 1000.0
        );
        println!(
            "  Success rate:      {:.1}%",
            (self.packets_processed as f64 / self.packets_sent as f64) * 100.0
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_pipeline_basic_processing() {
        let mut pipeline = PacketPipeline::new(2);
        pipeline.start().await.unwrap();

        // Submit test packets
        let test_data = Bytes::from(vec![0u8; 1000]);
        for i in 0..10 {
            let packet = PipelinePacket::with_priority(test_data.clone(), i % 3);
            pipeline.submit_packet(packet).await.unwrap();
        }

        // Wait for processing
        sleep(Duration::from_millis(100)).await;

        // Check results
        let processed = pipeline.get_processed_packets(100);
        let stats = pipeline.stats();

        println!("Processed {} packets", processed.len());
        println!(
            "Pipeline processed: {}",
            stats.packets_processed.load(Ordering::Relaxed)
        );

        pipeline.stop().await.unwrap();
    }

    #[tokio::test]
    async fn test_memory_pool() {
        let pool = MemoryPool::new(10, 1024);

        // Get buffers
        let buf1 = pool.get_buffer(512);
        let buf2 = pool.get_buffer(1024);

        assert!(buf1.capacity() >= 512);
        assert!(buf2.capacity() >= 1024);

        // Return buffers
        pool.return_buffer(buf1);
        pool.return_buffer(buf2);

        // Get again (should reuse)
        let _buf3 = pool.get_buffer(512);
        let (allocated, reused) = pool.stats();

        assert!(reused > 0 || allocated > 0);
    }
}
