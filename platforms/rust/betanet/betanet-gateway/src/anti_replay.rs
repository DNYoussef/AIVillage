// Anti-replay protection using RocksDB-backed sliding window
// Production implementation with 64-bit sequence numbers and persistence

use anyhow::{Context, Result, bail};
use rocksdb::{DB, Options, WriteBatch};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tokio::time::{interval, sleep};
use tracing::{debug, error, info, warn};

use crate::config::AntiReplayConfig;
use crate::metrics::MetricsCollector;

/// Anti-replay validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether the sequence is valid
    pub valid: bool,

    /// Rejection reason if invalid
    pub rejection_reason: String,

    /// Current window state after validation
    pub window_state: WindowState,

    /// Validation time in microseconds
    pub validation_time_us: u64,

    /// Whether window was updated
    pub window_updated: bool,
}

/// Sliding window state for a peer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowState {
    /// Base sequence number of the window
    pub window_base: u64,

    /// Received sequence bitmap (64 bits)
    pub received_bitmap: u64,

    /// Last update timestamp
    pub last_update_ns: u64,

    /// Total sequences processed
    pub sequences_processed: u64,

    /// Number of replays blocked
    pub replays_blocked: u64,
}

impl Default for WindowState {
    fn default() -> Self {
        Self {
            window_base: 0,
            received_bitmap: 0,
            last_update_ns: 0,
            sequences_processed: 0,
            replays_blocked: 0,
        }
    }
}

/// Anti-replay statistics
#[derive(Debug, Clone, Default)]
pub struct AntiReplayStats {
    /// Total validations performed
    pub total_validated: u64,

    /// Number of replays blocked
    pub replays_blocked: u64,

    /// Number of expired sequences rejected
    pub expired_rejected: u64,

    /// Number of far-future sequences rejected
    pub future_rejected: u64,

    /// Average validation time in microseconds
    pub average_validation_time_us: u64,

    /// Number of active peer windows
    pub active_windows: u64,

    /// Database operations performed
    pub db_operations: u64,

    /// Window slides performed
    pub window_slides: u64,
}

/// RocksDB-backed anti-replay manager
pub struct AntiReplayManager {
    config: AntiReplayConfig,
    metrics: Arc<MetricsCollector>,
    db: Arc<DB>,
    windows: Arc<RwLock<HashMap<String, WindowState>>>,
    stats: Arc<RwLock<AntiReplayStats>>,
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
    background_handle: Option<tokio::task::JoinHandle<()>>,
}

impl AntiReplayManager {
    /// Create new anti-replay manager with RocksDB persistence
    pub async fn new(
        config: AntiReplayConfig,
        metrics: Arc<MetricsCollector>,
    ) -> Result<Self> {
        info!(?config.db_path, "Initializing anti-replay manager");

        // Create database directory if needed
        if let Some(parent) = config.db_path.parent() {
            tokio::fs::create_dir_all(parent).await
                .with_context(|| format!("Failed to create directory: {}", parent.display()))?;
        }

        // Configure RocksDB
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.set_max_open_files(1000);
        opts.set_use_fsync(false); // Use fdatasync for better performance
        opts.set_bytes_per_sync(1048576); // 1MB
        opts.set_write_buffer_size(64 * 1024 * 1024); // 64MB
        opts.set_max_write_buffer_number(3);
        opts.set_level_zero_file_num_compaction_trigger(4);

        // Open database
        let db = Arc::new(
            DB::open(&opts, &config.db_path)
                .with_context(|| format!("Failed to open database: {}", config.db_path.display()))?
        );

        info!("Anti-replay database opened successfully");

        // Load existing windows from database
        let windows = Self::load_windows_from_db(&db).await?;
        let window_count = windows.len();

        info!(windows_loaded = window_count, "Loaded existing anti-replay windows");

        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();

        let manager = Self {
            config: config.clone(),
            metrics: metrics.clone(),
            db: db.clone(),
            windows: Arc::new(RwLock::new(windows)),
            stats: Arc::new(RwLock::new(AntiReplayStats::default())),
            shutdown_tx: Some(shutdown_tx),
            background_handle: None,
        };

        // Start background tasks
        let background_handle = tokio::spawn(Self::background_worker(
            config,
            db,
            manager.windows.clone(),
            manager.stats.clone(),
            metrics,
            shutdown_rx,
        ));

        Ok(Self {
            background_handle: Some(background_handle),
            ..manager
        })
    }

    /// Validate sequence number against sliding window
    pub async fn validate_sequence(
        &self,
        peer_id: &str,
        sequence: u64,
        timestamp_ns: u64,
        update_window: bool,
    ) -> ValidationResult {
        let start_time = Instant::now();

        // Get current time for age validation
        let current_time_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        // Check sequence age
        if timestamp_ns > 0 {
            let age_ns = current_time_ns.saturating_sub(timestamp_ns);
            let max_age_ns = self.config.max_sequence_age.as_nanos() as u64;

            if age_ns > max_age_ns {
                let result = ValidationResult {
                    valid: false,
                    rejection_reason: "expired".to_string(),
                    window_state: WindowState::default(),
                    validation_time_us: start_time.elapsed().as_micros() as u64,
                    window_updated: false,
                };

                self.update_stats(|stats| {
                    stats.total_validated += 1;
                    stats.expired_rejected += 1;
                }).await;

                return result;
            }
        }

        // Get or create window state
        let mut windows = self.windows.write().await;
        let window_state = windows.entry(peer_id.to_string())
            .or_insert_with(WindowState::default);

        // Validate sequence against sliding window
        let validation_result = self.validate_against_window(window_state, sequence, update_window);

        let result = ValidationResult {
            valid: validation_result.0,
            rejection_reason: validation_result.1,
            window_state: window_state.clone(),
            validation_time_us: start_time.elapsed().as_micros() as u64,
            window_updated: validation_result.2,
        };

        // Update statistics
        self.update_stats(|stats| {
            stats.total_validated += 1;
            if !result.valid {
                match result.rejection_reason.as_str() {
                    "replay" => stats.replays_blocked += 1,
                    "future" => stats.future_rejected += 1,
                    _ => {}
                }
            }
            stats.average_validation_time_us = (stats.average_validation_time_us + result.validation_time_us) / 2;
        }).await;

        // Persist window state if updated
        if result.window_updated && update_window {
            if let Err(e) = self.persist_window_state(peer_id, window_state).await {
                error!(peer_id = peer_id, error = ?e, "Failed to persist window state");
            }
        }

        result
    }

    /// Get anti-replay statistics
    pub async fn get_stats(&self) -> AntiReplayStats {
        let mut stats = self.stats.read().await.clone();
        stats.active_windows = self.windows.read().await.len() as u64;
        stats
    }

    /// Run background maintenance tasks
    pub async fn run_background_tasks(&self) {
        if let Some(handle) = &self.background_handle {
            if let Err(e) = handle.await {
                error!(error = ?e, "Background task failed");
            }
        }
    }

    /// Stop the anti-replay manager
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping anti-replay manager");

        if let Some(tx) = &self.shutdown_tx {
            let _ = tx.send(());
        }

        if let Some(handle) = &self.background_handle {
            if let Err(e) = handle.await {
                warn!(error = ?e, "Error waiting for background task to stop");
            }
        }

        info!("Anti-replay manager stopped");
        Ok(())
    }

    /// Validate sequence against sliding window algorithm
    fn validate_against_window(
        &self,
        window_state: &mut WindowState,
        sequence: u64,
        update_window: bool,
    ) -> (bool, String, bool) {
        // Handle first sequence for peer
        if window_state.window_base == 0 {
            if update_window {
                window_state.window_base = sequence;
                window_state.received_bitmap = 1u64; // Mark first bit
                window_state.last_update_ns = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos() as u64;
                window_state.sequences_processed += 1;
            }
            return (true, String::new(), update_window);
        }

        let window_size = self.config.window_size as u64;

        // Check if sequence is too far in the future
        if sequence > window_state.window_base + window_size * 2 {
            return (false, "future".to_string(), false);
        }

        // Check if sequence is behind the window (expired)
        if sequence < window_state.window_base {
            return (false, "expired".to_string(), false);
        }

        // Check if sequence is within current window
        if sequence < window_state.window_base + window_size {
            let bit_position = (sequence - window_state.window_base) as u32;
            let bit_mask = 1u64 << bit_position;

            // Check if already received (replay)
            if window_state.received_bitmap & bit_mask != 0 {
                if update_window {
                    window_state.replays_blocked += 1;
                }
                return (false, "replay".to_string(), false);
            }

            // Mark as received
            if update_window {
                window_state.received_bitmap |= bit_mask;
                window_state.sequences_processed += 1;
                window_state.last_update_ns = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos() as u64;
            }

            return (true, String::new(), update_window);
        }

        // Sequence is ahead of window - need to slide
        if update_window {
            let slide_amount = sequence - (window_state.window_base + window_size - 1);

            if slide_amount >= window_size {
                // Complete window replacement
                window_state.window_base = sequence;
                window_state.received_bitmap = 1u64;
            } else {
                // Slide window
                window_state.received_bitmap <<= slide_amount;
                window_state.window_base += slide_amount;

                // Set bit for new sequence
                let bit_position = (sequence - window_state.window_base) as u32;
                window_state.received_bitmap |= 1u64 << bit_position;
            }

            window_state.sequences_processed += 1;
            window_state.last_update_ns = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64;
        }

        (true, String::new(), update_window)
    }

    /// Load window states from RocksDB
    async fn load_windows_from_db(db: &Arc<DB>) -> Result<HashMap<String, WindowState>> {
        let mut windows = HashMap::new();
        let iter = db.iterator(rocksdb::IteratorMode::Start);

        for item in iter {
            let (key, value) = item.context("Database iterator error")?;

            let peer_id = String::from_utf8(key.to_vec())
                .context("Invalid peer ID encoding")?;

            let window_state: WindowState = bincode::deserialize(&value)
                .context("Failed to deserialize window state")?;

            windows.insert(peer_id, window_state);
        }

        Ok(windows)
    }

    /// Persist window state to RocksDB
    async fn persist_window_state(&self, peer_id: &str, window_state: &WindowState) -> Result<()> {
        let serialized = bincode::serialize(window_state)
            .context("Failed to serialize window state")?;

        self.db.put(peer_id.as_bytes(), serialized)
            .context("Failed to write to database")?;

        self.update_stats(|stats| {
            stats.db_operations += 1;
        }).await;

        Ok(())
    }

    /// Update statistics with closure
    async fn update_stats<F>(&self, update_fn: F)
    where
        F: FnOnce(&mut AntiReplayStats),
    {
        let mut stats = self.stats.write().await;
        update_fn(&mut *stats);
    }

    /// Background worker for cleanup and maintenance
    async fn background_worker(
        config: AntiReplayConfig,
        db: Arc<DB>,
        windows: Arc<RwLock<HashMap<String, WindowState>>>,
        stats: Arc<RwLock<AntiReplayStats>>,
        metrics: Arc<MetricsCollector>,
        mut shutdown_rx: tokio::sync::oneshot::Receiver<()>,
    ) {
        info!("Starting anti-replay background worker");

        let mut cleanup_interval = interval(config.cleanup_interval);
        let mut sync_interval = interval(config.sync_interval);

        loop {
            tokio::select! {
                _ = cleanup_interval.tick() => {
                    Self::cleanup_expired_windows(&config, &windows, &stats).await;
                }

                _ = sync_interval.tick() => {
                    Self::sync_windows_to_db(&db, &windows, &stats).await;
                }

                _ = &mut shutdown_rx => {
                    info!("Anti-replay background worker shutting down");

                    // Final sync before shutdown
                    Self::sync_windows_to_db(&db, &windows, &stats).await;
                    break;
                }
            }
        }

        info!("Anti-replay background worker stopped");
    }

    /// Clean up expired windows
    async fn cleanup_expired_windows(
        config: &AntiReplayConfig,
        windows: &Arc<RwLock<HashMap<String, WindowState>>>,
        stats: &Arc<RwLock<AntiReplayStats>>,
    ) {
        let current_time_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let cleanup_threshold_ns = config.cleanup_ttl.as_nanos() as u64;

        let mut windows_guard = windows.write().await;
        let initial_count = windows_guard.len();

        windows_guard.retain(|peer_id, window_state| {
            let age_ns = current_time_ns.saturating_sub(window_state.last_update_ns);
            let should_keep = age_ns < cleanup_threshold_ns;

            if !should_keep {
                debug!(peer_id = peer_id, age_ns = age_ns, "Cleaning up expired window");
            }

            should_keep
        });

        let cleaned_count = initial_count - windows_guard.len();

        if cleaned_count > 0 {
            info!(cleaned_count = cleaned_count, "Cleaned up expired anti-replay windows");
        }

        drop(windows_guard);

        // Update stats
        let mut stats_guard = stats.write().await;
        stats_guard.active_windows = windows.read().await.len() as u64;
    }

    /// Sync in-memory windows to database
    async fn sync_windows_to_db(
        db: &Arc<DB>,
        windows: &Arc<RwLock<HashMap<String, WindowState>>>,
        stats: &Arc<RwLock<AntiReplayStats>>,
    ) {
        let windows_guard = windows.read().await;

        if windows_guard.is_empty() {
            return;
        }

        let mut batch = WriteBatch::default();
        let mut batch_size = 0;

        for (peer_id, window_state) in windows_guard.iter() {
            match bincode::serialize(window_state) {
                Ok(serialized) => {
                    batch.put(peer_id.as_bytes(), serialized);
                    batch_size += 1;
                }
                Err(e) => {
                    error!(peer_id = peer_id, error = ?e, "Failed to serialize window state for sync");
                }
            }
        }

        drop(windows_guard);

        if batch_size > 0 {
            match db.write(batch) {
                Ok(()) => {
                    debug!(batch_size = batch_size, "Synced anti-replay windows to database");

                    let mut stats_guard = stats.write().await;
                    stats_guard.db_operations += batch_size as u64;
                }
                Err(e) => {
                    error!(error = ?e, "Failed to sync windows to database");
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use crate::metrics::MetricsCollector;

    async fn create_test_manager() -> (AntiReplayManager, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");

        let config = AntiReplayConfig {
            db_path,
            window_size: 64,
            cleanup_ttl: Duration::from_secs(3600),
            cleanup_interval: Duration::from_secs(60),
            sync_interval: Duration::from_secs(10),
            max_sequence_age: Duration::from_secs(300),
        };

        let metrics = Arc::new(MetricsCollector::new(Arc::new(
            crate::config::GatewayConfig::default()
        )).unwrap());

        let manager = AntiReplayManager::new(config, metrics).await.unwrap();
        (manager, temp_dir)
    }

    #[tokio::test]
    async fn test_basic_sequence_validation() {
        let (manager, _temp_dir) = create_test_manager().await;
        let peer_id = "test_peer";
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64;

        // First sequence should be accepted
        let result1 = manager.validate_sequence(peer_id, 1, timestamp, true).await;
        assert!(result1.valid);
        assert!(result1.window_updated);

        // Replay should be rejected
        let result2 = manager.validate_sequence(peer_id, 1, timestamp, true).await;
        assert!(!result2.valid);
        assert_eq!(result2.rejection_reason, "replay");

        // Future sequence should be accepted
        let result3 = manager.validate_sequence(peer_id, 10, timestamp, true).await;
        assert!(result3.valid);

        manager.stop().await.unwrap();
    }

    #[tokio::test]
    async fn test_window_sliding() {
        let (manager, _temp_dir) = create_test_manager().await;
        let peer_id = "slide_peer";
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64;

        // Accept first sequence
        let result1 = manager.validate_sequence(peer_id, 1, timestamp, true).await;
        assert!(result1.valid);
        assert_eq!(result1.window_state.window_base, 1);

        // Jump ahead to cause window slide
        let slide_seq = 1 + 64 + 10; // Beyond window size
        let result2 = manager.validate_sequence(peer_id, slide_seq, timestamp, true).await;
        assert!(result2.valid);
        assert!(result2.window_state.window_base > 1);

        // Previous sequence should now be expired
        let result3 = manager.validate_sequence(peer_id, 1, timestamp, true).await;
        assert!(!result3.valid);
        assert_eq!(result3.rejection_reason, "expired");

        manager.stop().await.unwrap();
    }

    #[tokio::test]
    async fn test_sequence_expiration() {
        let (manager, _temp_dir) = create_test_manager().await;
        let peer_id = "expire_peer";

        // Old timestamp (10 minutes ago)
        let old_timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64 - (10 * 60 * 1_000_000_000);

        // Should be rejected as expired
        let result = manager.validate_sequence(peer_id, 1, old_timestamp, true).await;
        assert!(!result.valid);
        assert_eq!(result.rejection_reason, "expired");

        manager.stop().await.unwrap();
    }

    #[tokio::test]
    async fn test_far_future_rejection() {
        let (manager, _temp_dir) = create_test_manager().await;
        let peer_id = "future_peer";
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64;

        // Accept normal sequence
        let result1 = manager.validate_sequence(peer_id, 1, timestamp, true).await;
        assert!(result1.valid);

        // Far future sequence should be rejected
        let far_future_seq = 1 + (64 * 2) + 100; // Way beyond acceptable range
        let result2 = manager.validate_sequence(peer_id, far_future_seq, timestamp, true).await;
        assert!(!result2.valid);
        assert_eq!(result2.rejection_reason, "future");

        manager.stop().await.unwrap();
    }

    #[tokio::test]
    async fn test_statistics_accuracy() {
        let (manager, _temp_dir) = create_test_manager().await;
        let peer_id = "stats_peer";
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64;

        // Valid sequence
        manager.validate_sequence(peer_id, 1, timestamp, true).await;

        // Replay
        manager.validate_sequence(peer_id, 1, timestamp, true).await;

        // Expired
        let old_timestamp = timestamp - (10 * 60 * 1_000_000_000);
        manager.validate_sequence(peer_id, 0, old_timestamp, true).await;

        // Future
        manager.validate_sequence(peer_id, 10000, timestamp, true).await;

        let stats = manager.get_stats().await;
        assert_eq!(stats.total_validated, 4);
        assert_eq!(stats.replays_blocked, 1);
        assert_eq!(stats.expired_rejected, 1);
        assert_eq!(stats.future_rejected, 1);
        assert!(stats.average_validation_time_us > 0);

        manager.stop().await.unwrap();
    }
}
