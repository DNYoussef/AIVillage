//! Rate limiting and traffic shaping
//!
//! Implements token bucket algorithm for rate limiting and traffic shaping
//! to ensure constant-rate output and prevent traffic analysis.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;


use tokio::sync::{Mutex, Semaphore};
use std::collections::VecDeque;
use tokio::time::sleep;
use tracing::{debug, warn};

use crate::{MixnodeError, Result};

/// High-performance token bucket rate limiter
pub struct TokenBucket {
    /// Maximum number of tokens
    capacity: u64,
    /// Current number of tokens (fixed-point: actual_tokens = tokens / PRECISION)
    tokens: AtomicU64,
    /// Rate of token refill (tokens per second, fixed-point)
    refill_rate_fp: u64,
    /// Last refill timestamp (microseconds since epoch)
    last_refill_us: AtomicU64,
    /// Statistics
    stats: Arc<RateLimiterStats>,
}

const TOKEN_PRECISION: u64 = 1_000_000; // 1 million = 1.0 token

impl TokenBucket {
    /// Create new token bucket
    pub fn new(capacity: u64, refill_rate: f64) -> Self {
        let now_us = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        Self {
            capacity,
            tokens: AtomicU64::new(capacity * TOKEN_PRECISION),
            refill_rate_fp: (refill_rate * TOKEN_PRECISION as f64) as u64,
            last_refill_us: AtomicU64::new(now_us),
            stats: Arc::new(RateLimiterStats::new()),
        }
    }

    /// Try to consume tokens (non-blocking, lock-free)
    pub async fn try_consume(&self, tokens: u64) -> bool {
        let tokens_fp = tokens * TOKEN_PRECISION;

        // Fast path: lock-free refill and consume
        self.refill_lockfree();

        // Use compare-and-swap loop for atomic token consumption
        loop {
            let current = self.tokens.load(Ordering::Acquire);
            if current < tokens_fp {
                self.stats.requests_denied.fetch_add(1, Ordering::Relaxed);
                return false;
            }

            let new_value = current - tokens_fp;
            match self.tokens.compare_exchange_weak(
                current,
                new_value,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    self.stats.tokens_consumed.fetch_add(tokens, Ordering::Relaxed);
                    self.stats.requests_allowed.fetch_add(1, Ordering::Relaxed);
                    return true;
                }
                Err(_) => {
                    // Retry on CAS failure
                    std::hint::spin_loop();
                }
            }
        }
    }

    /// Consume tokens (optimized blocking)
    pub async fn consume(&self, tokens: u64) -> Result<()> {
        let mut backoff = 1u64;
        const MAX_BACKOFF: u64 = 64;

        loop {
            if self.try_consume(tokens).await {
                return Ok(());
            }

            // Adaptive backoff instead of fixed sleep
            if backoff <= 8 {
                // Spin for very short waits
                for _ in 0..backoff * 100 {
                    std::hint::spin_loop();
                }
            } else {
                // Use async sleep for longer waits
                let wait_time = Duration::from_micros(backoff * 10);
                sleep(wait_time).await;
            }

            backoff = (backoff * 2).min(MAX_BACKOFF);
        }
    }

    /// Lock-free token refill using atomic operations
    fn refill_lockfree(&self) {
        let now_us = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        let last_refill = self.last_refill_us.load(Ordering::Acquire);
        let elapsed_us = now_us.saturating_sub(last_refill);

        // Only refill if enough time has passed (reduce contention)
        if elapsed_us < 1000 { // 1ms minimum
            return;
        }

        // Try to update the timestamp atomically
        if self.last_refill_us.compare_exchange_weak(
            last_refill,
            now_us,
            Ordering::Release,
            Ordering::Relaxed,
        ).is_ok() {
            // We won the race to update timestamp, now add tokens
            let tokens_to_add_fp = (elapsed_us * self.refill_rate_fp) / 1_000_000;

            if tokens_to_add_fp > 0 {
                let capacity_fp = self.capacity * TOKEN_PRECISION;

                // Atomic add with saturation at capacity
                loop {
                    let current = self.tokens.load(Ordering::Acquire);
                    let new_tokens = (current + tokens_to_add_fp).min(capacity_fp);

                    if current == new_tokens {
                        break; // Already at capacity
                    }

                    match self.tokens.compare_exchange_weak(
                        current,
                        new_tokens,
                        Ordering::Release,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => {
                            let added = (tokens_to_add_fp / TOKEN_PRECISION) as u64;
                            self.stats.tokens_added.fetch_add(added, Ordering::Relaxed);
                            break;
                        }
                        Err(_) => {
                            std::hint::spin_loop();
                        }
                    }
                }
            }
        }
    }

    /// Get current token count
    pub fn available_tokens(&self) -> u64 {
        self.refill_lockfree();
        self.tokens.load(Ordering::Relaxed) / TOKEN_PRECISION
    }

    /// Get statistics
    pub fn stats(&self) -> &RateLimiterStats {
        &self.stats
    }

    /// Update rate parameters
    pub fn update_rate(&mut self, capacity: u64, refill_rate: f64) {
        self.capacity = capacity;
        self.refill_rate_fp = (refill_rate * TOKEN_PRECISION as f64) as u64;

        // Adjust current tokens if new capacity is smaller
        let capacity_fp = capacity * TOKEN_PRECISION;
        let current = self.tokens.load(Ordering::Relaxed);
        if current > capacity_fp {
            self.tokens.store(capacity_fp, Ordering::Relaxed);
        }
    }
}

/// High-performance traffic shaper for constant-rate output
pub struct TrafficShaper {
    /// Target output rate (packets per second)
    target_rate: f64,
    /// High-performance packet queue (FIFO with VecDeque)
    packet_queue: Arc<Mutex<VecDeque<Vec<u8>>>>,
    /// Rate limiter
    rate_limiter: TokenBucket,
    /// Shaping buffer
    buffer: Arc<Semaphore>,
    /// Statistics
    stats: Arc<TrafficShaperStats>,
    /// Queue size limit
    max_queue_size: usize,
}

impl TrafficShaper {
    /// Create new high-performance traffic shaper
    pub fn new(target_rate: f64, buffer_capacity: usize) -> Self {
        let rate_limiter = TokenBucket::new(
            (target_rate * 2.0) as u64, // 2 second burst capacity
            target_rate,
        );

        Self {
            target_rate,
            packet_queue: Arc::new(Mutex::new(VecDeque::with_capacity(buffer_capacity))),
            rate_limiter,
            buffer: Arc::new(Semaphore::new(buffer_capacity)),
            stats: Arc::new(TrafficShaperStats::new()),
            max_queue_size: buffer_capacity.max(10000),
        }
    }

    /// Submit packet for shaped output (optimized)
    pub async fn submit_packet(&self, packet: Vec<u8>) -> Result<()> {
        // Fast path: check queue size without locking first
        {
            let queue = self.packet_queue.lock().await;
            if queue.len() >= self.max_queue_size {
                drop(queue);
                self.stats.packets_dropped.fetch_add(1, Ordering::Relaxed);
                return Err(MixnodeError::Network("Traffic shaper queue full".to_string()));
            }
        }

        // Acquire buffer slot only after queue check
        let _permit = self.buffer.acquire().await.map_err(|_| {
            MixnodeError::Network("Traffic shaper buffer closed".to_string())
        })?;

        // Add to queue with optimized VecDeque
        {
            let mut queue = self.packet_queue.lock().await;
            // Double-check size since queue could have grown
            if queue.len() >= self.max_queue_size {
                self.stats.packets_dropped.fetch_add(1, Ordering::Relaxed);
                return Err(MixnodeError::Network("Traffic shaper queue full".to_string()));
            }
            queue.push_back(packet);
            self.stats.packets_queued.fetch_add(1, Ordering::Relaxed);
        }

        // Don't release permit - it will be released when packet is sent
        std::mem::forget(_permit);
        Ok(())
    }

    /// Get next shaped packet (rate-limited, optimized)
    pub async fn next_packet(&self) -> Result<Option<Vec<u8>>> {
        // Rate limit to target rate (1 token per packet) - now much faster
        self.rate_limiter.consume(1).await?;

        // Get packet from queue with optimized FIFO operations
        let packet = {
            let mut queue = self.packet_queue.lock().await;
            queue.pop_front() // VecDeque::pop_front() is O(1) vs Vec::pop() being O(n)
        };

        if packet.is_some() {
            self.stats.packets_sent.fetch_add(1, Ordering::Relaxed);
            self.buffer.add_permits(1); // Release buffer slot
        }

        Ok(packet)
    }

    /// Start continuous output loop
    pub async fn start_output<F>(&self, mut output_handler: F) -> Result<()>
    where
        F: FnMut(Vec<u8>) -> Result<()> + Send,
    {
        let mut interval = tokio::time::interval(Duration::from_secs_f64(1.0 / self.target_rate));

        loop {
            interval.tick().await;

            if let Some(packet) = self.next_packet().await? {
                if let Err(e) = output_handler(packet) {
                    warn!("Output handler error: {}", e);
                    self.stats.output_errors.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }

    /// Get current queue length
    pub async fn queue_length(&self) -> usize {
        self.packet_queue.lock().await.len()
    }

    /// Get statistics
    pub fn stats(&self) -> &TrafficShaperStats {
        &self.stats
    }

    /// Update target rate
    pub fn update_rate(&mut self, new_rate: f64) {
        self.target_rate = new_rate;
        self.rate_limiter.update_rate((new_rate * 2.0) as u64, new_rate);
    }
}

/// Rate limiter statistics
#[derive(Debug)]
pub struct RateLimiterStats {
    /// Tokens consumed
    pub tokens_consumed: AtomicU64,
    /// Tokens added through refill
    pub tokens_added: AtomicU64,
    /// Requests allowed
    pub requests_allowed: AtomicU64,
    /// Requests denied
    pub requests_denied: AtomicU64,
}

impl RateLimiterStats {
    /// Create new statistics
    pub fn new() -> Self {
        Self {
            tokens_consumed: AtomicU64::new(0),
            tokens_added: AtomicU64::new(0),
            requests_allowed: AtomicU64::new(0),
            requests_denied: AtomicU64::new(0),
        }
    }

    /// Get denial rate (0.0 to 1.0)
    pub fn denial_rate(&self) -> f64 {
        let allowed = self.requests_allowed.load(Ordering::Relaxed) as f64;
        let denied = self.requests_denied.load(Ordering::Relaxed) as f64;
        let total = allowed + denied;
        if total > 0.0 {
            denied / total
        } else {
            0.0
        }
    }
}

impl Default for RateLimiterStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Traffic shaper statistics
#[derive(Debug)]
pub struct TrafficShaperStats {
    /// Packets queued for shaping
    pub packets_queued: AtomicU64,
    /// Packets sent after shaping
    pub packets_sent: AtomicU64,
    /// Packets dropped due to overflow
    pub packets_dropped: AtomicU64,
    /// Output handler errors
    pub output_errors: AtomicU64,
}

impl TrafficShaperStats {
    /// Create new statistics
    pub fn new() -> Self {
        Self {
            packets_queued: AtomicU64::new(0),
            packets_sent: AtomicU64::new(0),
            packets_dropped: AtomicU64::new(0),
            output_errors: AtomicU64::new(0),
        }
    }

    /// Get loss rate (0.0 to 1.0)
    pub fn loss_rate(&self) -> f64 {
        let queued = self.packets_queued.load(Ordering::Relaxed) as f64;
        let dropped = self.packets_dropped.load(Ordering::Relaxed) as f64;
        if queued > 0.0 {
            dropped / queued
        } else {
            0.0
        }
    }

    /// Get throughput rate
    pub fn throughput_rate(&self, duration: Duration) -> f64 {
        let sent = self.packets_sent.load(Ordering::Relaxed) as f64;
        sent / duration.as_secs_f64()
    }
}

impl Default for TrafficShaperStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Combined rate limiting and traffic shaping configuration
#[derive(Debug, Clone)]
pub struct RateLimitingConfig {
    /// Enable rate limiting
    pub enabled: bool,
    /// Maximum burst size (tokens)
    pub burst_capacity: u64,
    /// Sustained rate (tokens/packets per second)
    pub sustained_rate: f64,
    /// Traffic shaping buffer size
    pub shaping_buffer_size: usize,
    /// Constant output rate (packets per second)
    pub output_rate: f64,
}

impl Default for RateLimitingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            burst_capacity: 1000,
            sustained_rate: 500.0,
            shaping_buffer_size: 5000,
            output_rate: 100.0, // Conservative output rate
        }
    }
}

/// Integrated rate limiter and traffic shaper
pub struct RateLimitedTrafficShaper {
    /// Configuration
    config: RateLimitingConfig,
    /// Input rate limiter
    input_limiter: TokenBucket,
    /// Output traffic shaper
    output_shaper: TrafficShaper,
}

impl RateLimitedTrafficShaper {
    /// Create new integrated rate limiter and shaper
    pub fn new(config: RateLimitingConfig) -> Self {
        let input_limiter = TokenBucket::new(config.burst_capacity, config.sustained_rate);
        let output_shaper = TrafficShaper::new(config.output_rate, config.shaping_buffer_size);

        Self {
            config,
            input_limiter,
            output_shaper,
        }
    }

    /// Process packet through rate limiting and shaping
    pub async fn process_packet(&self, packet: Vec<u8>) -> Result<()> {
        if !self.config.enabled {
            // Pass through without limiting
            return self.output_shaper.submit_packet(packet).await;
        }

        // Rate limit input
        let packet_size = packet.len() as u64;
        if !self.input_limiter.try_consume(packet_size).await {
            debug!("Packet rate limited, size: {} bytes", packet_size);
            return Err(MixnodeError::Network("Rate limited".to_string()));
        }

        // Submit to traffic shaper
        self.output_shaper.submit_packet(packet).await
    }

    /// Get next shaped output packet
    pub async fn next_output_packet(&self) -> Result<Option<Vec<u8>>> {
        self.output_shaper.next_packet().await
    }

    /// Get input rate limiter statistics
    pub fn input_stats(&self) -> &RateLimiterStats {
        self.input_limiter.stats()
    }

    /// Get output shaper statistics
    pub fn output_stats(&self) -> &TrafficShaperStats {
        self.output_shaper.stats()
    }

    /// Update configuration
    pub fn update_config(&mut self, config: RateLimitingConfig) {
        self.input_limiter.update_rate(config.burst_capacity, config.sustained_rate);
        self.output_shaper.update_rate(config.output_rate);
        self.config = config;
    }

    /// Get current queue length
    pub async fn queue_length(&self) -> usize {
        self.output_shaper.queue_length().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::timeout;

    #[tokio::test]
    async fn test_token_bucket_basic() {
        let bucket = TokenBucket::new(10, 5.0);

        // Should allow initial burst
        assert!(bucket.try_consume(5).await);
        assert!(bucket.try_consume(3).await);

        // Should deny when empty
        assert!(!bucket.try_consume(5).await);

        // Wait for refill
        sleep(Duration::from_millis(1100)).await; // Allow 5+ tokens
        assert!(bucket.try_consume(5).await);
    }

    #[tokio::test]
    async fn test_token_bucket_refill() {
        let bucket = TokenBucket::new(10, 10.0); // 10 tokens/sec

        // Consume all tokens
        assert!(bucket.try_consume(10).await);
        assert_eq!(bucket.available_tokens(), 0);

        // Wait for partial refill
        sleep(Duration::from_millis(500)).await; // Should add ~5 tokens
        let available = bucket.available_tokens();
        assert!(available >= 4 && available <= 6); // Allow some timing variance
    }

    #[tokio::test]
    async fn test_traffic_shaper_basic() {
        let shaper = TrafficShaper::new(2.0, 100); // 2 pkt/s (4 token burst capacity)

        // Submit packets
        for i in 0..5 {
            let packet = format!("packet_{}", i).into_bytes();
            shaper.submit_packet(packet).await.unwrap();
        }

        assert_eq!(shaper.queue_length().await, 5);

        // Get first few packets (should use burst capacity)
        for _ in 0..4 {
            let packet = shaper.next_packet().await.unwrap();
            assert!(packet.is_some());
        }

        // Next packet should be rate limited
        let packet_delayed = timeout(Duration::from_millis(300), shaper.next_packet()).await;
        assert!(packet_delayed.is_err()); // Should timeout due to rate limiting
    }

    #[tokio::test]
    async fn test_rate_limited_traffic_shaper() {
        let config = RateLimitingConfig {
            enabled: true,
            burst_capacity: 100, // 100 bytes burst
            sustained_rate: 50.0, // 50 bytes/sec
            shaping_buffer_size: 10,
            output_rate: 1.0,
        };

        let shaper = RateLimitedTrafficShaper::new(config);

        // Should accept small packets within burst capacity
        for i in 0..3 {
            let packet = format!("packet_{}", i).into_bytes(); // ~8 bytes each
            assert!(shaper.process_packet(packet).await.is_ok());
        }

        // Should rate limit large packet that exceeds remaining capacity
        let large_packet = vec![0u8; 200]; // 200 bytes > remaining burst capacity
        assert!(shaper.process_packet(large_packet).await.is_err());
    }

    #[test]
    fn test_rate_limiting_config() {
        let config = RateLimitingConfig::default();
        assert!(config.enabled);
        assert_eq!(config.burst_capacity, 1000);
        assert_eq!(config.sustained_rate, 500.0);
        assert_eq!(config.output_rate, 100.0);
    }

    #[test]
    fn test_stats() {
        let stats = RateLimiterStats::new();
        assert_eq!(stats.denial_rate(), 0.0);

        stats.requests_allowed.store(8, Ordering::Relaxed);
        stats.requests_denied.store(2, Ordering::Relaxed);
        assert_eq!(stats.denial_rate(), 0.2);

        let shaper_stats = TrafficShaperStats::new();
        assert_eq!(shaper_stats.loss_rate(), 0.0);

        shaper_stats.packets_queued.store(100, Ordering::Relaxed);
        shaper_stats.packets_dropped.store(5, Ordering::Relaxed);
        assert_eq!(shaper_stats.loss_rate(), 0.05);
    }
}
