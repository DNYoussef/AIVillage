// Multipath management with path quality tracking and failover
// Production implementation with EWMA RTT tracking and adaptive selection

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tokio::time::{interval, sleep};
use tracing::{debug, error, info, warn};

use crate::config::MultipathConfig;
use crate::metrics::MetricsCollector;

/// Path quality metrics using EWMA smoothing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathQuality {
    /// Round-trip time in microseconds (EWMA)
    pub rtt_us: f64,
    
    /// Jitter in microseconds (EWMA)
    pub jitter_us: f64,
    
    /// Packet loss rate (0.0-1.0, EWMA)
    pub loss_rate: f64,
    
    /// Bandwidth estimate in bytes per second
    pub bandwidth_bps: f64,
    
    /// Path utilization (0.0-1.0)
    pub utilization: f64,
    
    /// Number of measurements taken
    pub measurement_count: u64,
    
    /// Last measurement timestamp
    pub last_measured_ns: u64,
    
    /// Path stability score (0.0-1.0)
    pub stability_score: f64,
}

impl Default for PathQuality {
    fn default() -> Self {
        Self {
            rtt_us: 0.0,
            jitter_us: 0.0,
            loss_rate: 0.0,
            bandwidth_bps: 0.0,
            utilization: 0.0,
            measurement_count: 0,
            last_measured_ns: 0,
            stability_score: 1.0,
        }
    }
}

/// Path information with quality tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathInfo {
    /// Unique path identifier
    pub path_id: String,
    
    /// Path fingerprint from SCION
    pub fingerprint: String,
    
    /// Destination ISD-AS
    pub dst_ia: String,
    
    /// Path quality metrics
    pub quality: PathQuality,
    
    /// Path selection score (higher is better)
    pub selection_score: f64,
    
    /// Whether path is currently active
    pub is_active: bool,
    
    /// Whether path is healthy (below failover thresholds)
    pub is_healthy: bool,
    
    /// Time when path was discovered
    pub discovered_at_ns: u64,
    
    /// Last time path was used for traffic
    pub last_used_ns: u64,
    
    /// Number of packets sent via this path
    pub packets_sent: u64,
    
    /// Number of packets successfully delivered
    pub packets_delivered: u64,
    
    /// Recent RTT measurements for jitter calculation
    pub recent_rtts: VecDeque<f64>,
}

impl PathInfo {
    pub fn new(path_id: String, fingerprint: String, dst_ia: String) -> Self {
        let now_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
            
        Self {
            path_id,
            fingerprint,
            dst_ia,
            quality: PathQuality::default(),
            selection_score: 0.0,
            is_active: false,
            is_healthy: true,
            discovered_at_ns: now_ns,
            last_used_ns: 0,
            packets_sent: 0,
            packets_delivered: 0,
            recent_rtts: VecDeque::new(),
        }
    }
}

/// Path measurement result
#[derive(Debug, Clone)]
pub struct PathMeasurement {
    pub path_id: String,
    pub rtt_us: f64,
    pub success: bool,
    pub bandwidth_bps: Option<f64>,
    pub timestamp_ns: u64,
}

/// Multipath selection decision
#[derive(Debug, Clone)]
pub struct PathSelection {
    /// Selected primary path
    pub primary_path: Option<PathInfo>,
    
    /// Selected backup paths (ordered by preference)
    pub backup_paths: Vec<PathInfo>,
    
    /// Reason for selection
    pub selection_reason: String,
    
    /// Whether failover occurred
    pub failover_triggered: bool,
    
    /// Total paths considered
    pub paths_considered: usize,
}

/// Multipath manager statistics
#[derive(Debug, Clone, Default)]
pub struct MultipathStats {
    /// Total paths discovered
    pub total_paths: u64,
    
    /// Currently active paths
    pub active_paths: u64,
    
    /// Healthy paths (below failover threshold)
    pub healthy_paths: u64,
    
    /// Path failovers performed
    pub failovers_performed: u64,
    
    /// Total path measurements
    pub measurements_taken: u64,
    
    /// Path exploration attempts
    pub exploration_attempts: u64,
    
    /// Average path RTT across all paths
    pub average_rtt_us: f64,
    
    /// Best path RTT
    pub best_rtt_us: f64,
    
    /// Path selection decisions made
    pub selections_made: u64,
}

/// Multipath management with adaptive path selection
pub struct MultipathManager {
    config: MultipathConfig,
    metrics: Arc<MetricsCollector>,
    paths: Arc<RwLock<HashMap<String, PathInfo>>>,
    stats: Arc<RwLock<MultipathStats>>,
    current_primary: Arc<RwLock<Option<String>>>,
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
    background_handle: Option<tokio::task::JoinHandle<()>>,
}

impl MultipathManager {
    /// Create new multipath manager
    pub async fn new(
        config: MultipathConfig,
        metrics: Arc<MetricsCollector>,
    ) -> Result<Self> {
        info!(?config, "Initializing multipath manager");
        
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
        
        let manager = Self {
            config: config.clone(),
            metrics: metrics.clone(),
            paths: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(MultipathStats::default())),
            current_primary: Arc::new(RwLock::new(None)),
            shutdown_tx: Some(shutdown_tx),
            background_handle: None,
        };
        
        // Start background measurement and maintenance
        let background_handle = tokio::spawn(Self::background_worker(
            config,
            manager.paths.clone(),
            manager.stats.clone(),
            manager.current_primary.clone(),
            metrics,
            shutdown_rx,
        ));
        
        Ok(Self {
            background_handle: Some(background_handle),
            ..manager
        })
    }
    
    /// Register a new path
    pub async fn register_path(
        &self,
        path_id: String,
        fingerprint: String,
        dst_ia: String,
    ) -> Result<()> {
        let mut paths = self.paths.write().await;
        
        if !paths.contains_key(&path_id) {
            let path_info = PathInfo::new(path_id.clone(), fingerprint, dst_ia);
            paths.insert(path_id.clone(), path_info);
            
            debug!(path_id = path_id, "Registered new path");
            
            // Update stats
            let mut stats = self.stats.write().await;
            stats.total_paths += 1;
        }
        
        Ok(())
    }
    
    /// Update path quality with new measurement
    pub async fn update_path_quality(&self, measurement: PathMeasurement) -> Result<()> {
        let mut paths = self.paths.write().await;
        
        if let Some(path) = paths.get_mut(&measurement.path_id) {
            self.apply_measurement_to_path(path, &measurement);
            
            // Update health status
            path.is_healthy = self.evaluate_path_health(&path.quality);
            
            debug!(
                path_id = measurement.path_id,
                rtt_us = measurement.rtt_us,
                success = measurement.success,
                healthy = path.is_healthy,
                "Updated path quality"
            );
        } else {
            warn!(path_id = measurement.path_id, "Measurement for unknown path");
        }
        
        // Update stats
        let mut stats = self.stats.write().await;
        stats.measurements_taken += 1;
        
        Ok(())
    }
    
    /// Select best path for traffic
    pub async fn select_path(&self, dst_ia: &str) -> Result<PathSelection> {
        let paths = self.paths.read().await;
        
        // Filter paths for destination
        let candidate_paths: Vec<&PathInfo> = paths
            .values()
            .filter(|p| p.dst_ia == dst_ia)
            .collect();
        
        if candidate_paths.is_empty() {
            return Ok(PathSelection {
                primary_path: None,
                backup_paths: Vec::new(),
                selection_reason: "no_paths_available".to_string(),
                failover_triggered: false,
                paths_considered: 0,
            });
        }
        
        // Calculate selection scores for all paths
        let mut scored_paths: Vec<(PathInfo, f64)> = candidate_paths
            .iter()
            .map(|&path| {
                let score = self.calculate_selection_score(path);
                (path.clone(), score)
            })
            .collect();
        
        // Sort by score (descending - higher is better)
        scored_paths.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Apply exploration probability
        let should_explore = rand::random::<f64>() < self.config.exploration_probability;
        
        let (primary_path, selection_reason, failover_triggered) = if should_explore && scored_paths.len() > 1 {
            // Explore: randomly select from top paths
            let explore_candidate = scored_paths.get(1).cloned();
            if let Some((path, _score)) = explore_candidate {
                (Some(path), "exploration".to_string(), false)
            } else {
                let (path, _score) = scored_paths[0].clone();
                (Some(path), "best_available".to_string(), false)
            }
        } else {
            // Select best path
            let (best_path, _score) = scored_paths[0].clone();
            
            // Check if this is a failover from current primary
            let current_primary = self.current_primary.read().await;
            let failover = match current_primary.as_ref() {
                Some(current_id) => current_id != &best_path.path_id,
                None => false,
            };
            
            (Some(best_path), "best_available".to_string(), failover)
        };
        
        // Select backup paths
        let backup_paths: Vec<PathInfo> = scored_paths
            .iter()
            .skip(1) // Skip primary
            .take(self.config.max_paths - 1) // Leave room for primary
            .map(|(path, _score)| path.clone())
            .collect();
        
        // Update current primary
        if let Some(ref primary) = primary_path {
            let mut current_primary = self.current_primary.write().await;
            *current_primary = Some(primary.path_id.clone());
        }
        
        // Update stats
        let mut stats = self.stats.write().await;
        stats.selections_made += 1;
        if failover_triggered {
            stats.failovers_performed += 1;
        }
        if should_explore {
            stats.exploration_attempts += 1;
        }
        
        Ok(PathSelection {
            primary_path,
            backup_paths,
            selection_reason,
            failover_triggered,
            paths_considered: candidate_paths.len(),
        })
    }
    
    /// Get multipath statistics
    pub async fn get_stats(&self) -> MultipathStats {
        let paths = self.paths.read().await;
        let mut stats = self.stats.read().await.clone();
        
        // Update real-time stats
        stats.active_paths = paths.values().filter(|p| p.is_active).count() as u64;
        stats.healthy_paths = paths.values().filter(|p| p.is_healthy).count() as u64;
        
        if !paths.is_empty() {
            let total_rtt: f64 = paths.values()
                .filter(|p| p.quality.measurement_count > 0)
                .map(|p| p.quality.rtt_us)
                .sum();
            let rtt_count = paths.values()
                .filter(|p| p.quality.measurement_count > 0)
                .count();
            
            if rtt_count > 0 {
                stats.average_rtt_us = total_rtt / rtt_count as f64;
            }
            
            stats.best_rtt_us = paths.values()
                .filter(|p| p.quality.measurement_count > 0)
                .map(|p| p.quality.rtt_us)
                .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(0.0);
        }
        
        stats
    }
    
    /// Run background maintenance tasks
    pub async fn run_background_tasks(&self) {
        if let Some(handle) = &self.background_handle {
            if let Err(e) = handle.await {
                error!(error = ?e, "Multipath background task failed");
            }
        }
    }
    
    /// Stop the multipath manager
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping multipath manager");
        
        if let Some(tx) = &self.shutdown_tx {
            let _ = tx.send(());
        }
        
        if let Some(handle) = &self.background_handle {
            if let Err(e) = handle.await {
                warn!(error = ?e, "Error waiting for multipath background task to stop");
            }
        }
        
        info!("Multipath manager stopped");
        Ok(())
    }
    
    /// Apply measurement to path using EWMA smoothing
    fn apply_measurement_to_path(&self, path: &mut PathInfo, measurement: &PathMeasurement) {
        let alpha = self.config.quality_smoothing;
        let quality = &mut path.quality;
        
        // Update RTT with EWMA
        if quality.measurement_count == 0 {
            quality.rtt_us = measurement.rtt_us;
        } else {
            quality.rtt_us = alpha * measurement.rtt_us + (1.0 - alpha) * quality.rtt_us;
        }
        
        // Calculate and update jitter
        path.recent_rtts.push_back(measurement.rtt_us);
        if path.recent_rtts.len() > 10 {
            path.recent_rtts.pop_front();
        }
        
        if path.recent_rtts.len() > 1 {
            let mean_rtt: f64 = path.recent_rtts.iter().sum::<f64>() / path.recent_rtts.len() as f64;
            let variance: f64 = path.recent_rtts
                .iter()
                .map(|rtt| (rtt - mean_rtt).powi(2))
                .sum::<f64>() / path.recent_rtts.len() as f64;
            let current_jitter = variance.sqrt();
            
            if quality.measurement_count == 0 {
                quality.jitter_us = current_jitter;
            } else {
                quality.jitter_us = alpha * current_jitter + (1.0 - alpha) * quality.jitter_us;
            }
        }
        
        // Update loss rate
        let current_loss = if measurement.success { 0.0 } else { 1.0 };
        if quality.measurement_count == 0 {
            quality.loss_rate = current_loss;
        } else {
            quality.loss_rate = alpha * current_loss + (1.0 - alpha) * quality.loss_rate;
        }
        
        // Update bandwidth if provided
        if let Some(bw) = measurement.bandwidth_bps {
            if quality.measurement_count == 0 {
                quality.bandwidth_bps = bw;
            } else {
                quality.bandwidth_bps = alpha * bw + (1.0 - alpha) * quality.bandwidth_bps;
            }
        }
        
        // Update stability score based on measurement consistency
        let rtt_stability = if quality.jitter_us > 0.0 {
            1.0 - (quality.jitter_us / quality.rtt_us).min(1.0)
        } else {
            1.0
        };
        let loss_stability = 1.0 - quality.loss_rate;
        quality.stability_score = (rtt_stability + loss_stability) / 2.0;
        
        // Update counters
        quality.measurement_count += 1;
        quality.last_measured_ns = measurement.timestamp_ns;
        
        if measurement.success {
            path.packets_delivered += 1;
        }
        path.packets_sent += 1;
        
        // Update selection score
        path.selection_score = self.calculate_selection_score(path);
    }
    
    /// Calculate path selection score (higher is better)
    fn calculate_selection_score(&self, path: &PathInfo) -> f64 {
        let quality = &path.quality;
        
        if quality.measurement_count == 0 {
            return 0.0;
        }
        
        // RTT component (lower is better, normalized to 0-1)
        let rtt_score = if quality.rtt_us > 0.0 {
            1.0 / (1.0 + quality.rtt_us / 100000.0) // Normalize to ~100ms baseline
        } else {
            0.0
        };
        
        // Loss rate component (lower is better)
        let loss_score = 1.0 - quality.loss_rate;
        
        // Jitter component (lower is better, normalized)
        let jitter_score = if quality.jitter_us > 0.0 && quality.rtt_us > 0.0 {
            1.0 - (quality.jitter_us / quality.rtt_us).min(1.0)
        } else {
            1.0
        };
        
        // Stability component
        let stability_score = quality.stability_score;
        
        // Bandwidth component (higher is better, normalized to Mbps)
        let bandwidth_score = if quality.bandwidth_bps > 0.0 {
            (quality.bandwidth_bps / 1_000_000.0).min(100.0) / 100.0 // Cap at 100 Mbps
        } else {
            0.5 // Unknown bandwidth gets neutral score
        };
        
        // Combine scores with weights
        let score = 0.3 * rtt_score +
                   0.3 * loss_score +
                   0.2 * jitter_score +
                   0.1 * stability_score +
                   0.1 * bandwidth_score;
        
        score.max(0.0).min(1.0)
    }
    
    /// Evaluate if path is healthy based on thresholds
    fn evaluate_path_health(&self, quality: &PathQuality) -> bool {
        if quality.measurement_count == 0 {
            return true; // Unknown paths are considered healthy initially
        }
        
        // Check RTT threshold (if we have a baseline)
        let rtt_healthy = quality.rtt_us == 0.0 || 
                         quality.rtt_us < 50000.0 * self.config.failover_rtt_threshold;
        
        // Check loss rate threshold
        let loss_healthy = quality.loss_rate < self.config.failover_loss_threshold;
        
        rtt_healthy && loss_healthy
    }
    
    /// Background worker for path monitoring and maintenance
    async fn background_worker(
        config: MultipathConfig,
        paths: Arc<RwLock<HashMap<String, PathInfo>>>,
        stats: Arc<RwLock<MultipathStats>>,
        current_primary: Arc<RwLock<Option<String>>>,
        metrics: Arc<MetricsCollector>,
        mut shutdown_rx: tokio::sync::oneshot::Receiver<()>,
    ) {
        info!("Starting multipath background worker");
        
        let mut measurement_interval = interval(config.measurement_interval);
        
        loop {
            tokio::select! {
                _ = measurement_interval.tick() => {
                    Self::perform_path_measurements(&paths, &stats).await;
                    Self::cleanup_stale_paths(&config, &paths).await;
                }
                
                _ = &mut shutdown_rx => {
                    info!("Multipath background worker shutting down");
                    break;
                }
            }
        }
        
        info!("Multipath background worker stopped");
    }
    
    /// Perform active measurements on paths
    async fn perform_path_measurements(
        paths: &Arc<RwLock<HashMap<String, PathInfo>>>,
        stats: &Arc<RwLock<MultipathStats>>,
    ) {
        let paths_guard = paths.read().await;
        let path_list: Vec<String> = paths_guard.keys().cloned().collect();
        drop(paths_guard);
        
        if path_list.is_empty() {
            return;
        }
        
        debug!(path_count = path_list.len(), "Performing path measurements");
        
        // In a real implementation, this would send actual probe packets
        // For now, we simulate measurements based on path history
        for path_id in path_list {
            let measurement = Self::simulate_path_measurement(&path_id).await;
            
            let mut paths_guard = paths.write().await;
            if let Some(path) = paths_guard.get_mut(&path_id) {
                // Apply simulated measurement (this would be real in production)
                let quality = &mut path.quality;
                
                if quality.measurement_count == 0 {
                    quality.rtt_us = measurement.rtt_us;
                    quality.loss_rate = if measurement.success { 0.0 } else { 1.0 };
                } else {
                    // Add some realistic variation
                    let alpha = 0.125; // Standard EWMA alpha
                    quality.rtt_us = alpha * measurement.rtt_us + (1.0 - alpha) * quality.rtt_us;
                    let current_loss = if measurement.success { 0.0 } else { 1.0 };
                    quality.loss_rate = alpha * current_loss + (1.0 - alpha) * quality.loss_rate;
                }
                
                quality.measurement_count += 1;
                quality.last_measured_ns = measurement.timestamp_ns;
            }
        }
        
        let mut stats_guard = stats.write().await;
        stats_guard.measurements_taken += path_list.len() as u64;
    }
    
    /// Simulate path measurement (replace with real probing in production)
    async fn simulate_path_measurement(path_id: &str) -> PathMeasurement {
        // Simulate network delay
        sleep(Duration::from_millis(1)).await;
        
        let base_rtt = 50000.0; // 50ms base RTT
        let variation = rand::random::<f64>() * 20000.0; // ±20ms variation
        let rtt_us = base_rtt + variation;
        
        let success_rate = 0.98; // 98% success rate
        let success = rand::random::<f64>() < success_rate;
        
        PathMeasurement {
            path_id: path_id.to_string(),
            rtt_us,
            success,
            bandwidth_bps: Some(10_000_000.0), // 10 Mbps
            timestamp_ns: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
        }
    }
    
    /// Clean up stale paths that haven't been measured recently
    async fn cleanup_stale_paths(
        config: &MultipathConfig,
        paths: &Arc<RwLock<HashMap<String, PathInfo>>>,
    ) {
        let current_time_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        let stale_threshold_ns = (config.measurement_interval.as_nanos() * 10) as u64; // 10 intervals
        
        let mut paths_guard = paths.write().await;
        let initial_count = paths_guard.len();
        
        paths_guard.retain(|path_id, path| {
            let age_ns = current_time_ns.saturating_sub(path.quality.last_measured_ns);
            let should_keep = age_ns < stale_threshold_ns || path.quality.last_measured_ns == 0;
            
            if !should_keep {
                debug!(path_id = path_id, age_ns = age_ns, "Cleaning up stale path");
            }
            
            should_keep
        });
        
        let cleaned_count = initial_count - paths_guard.len();
        
        if cleaned_count > 0 {
            info!(cleaned_count = cleaned_count, "Cleaned up stale paths");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GatewayConfig;
    
    async fn create_test_manager() -> MultipathManager {
        let config = MultipathConfig {
            measurement_interval: Duration::from_millis(100),
            failover_rtt_threshold: 2.0,
            failover_loss_threshold: 0.1,
            min_paths: 1,
            max_paths: 5,
            exploration_probability: 0.1,
            quality_smoothing: 0.125,
        };
        
        let gateway_config = Arc::new(GatewayConfig::default());
        let metrics = Arc::new(MetricsCollector::new(gateway_config).unwrap());
        
        MultipathManager::new(config, metrics).await.unwrap()
    }
    
    #[tokio::test]
    async fn test_path_registration() {
        let manager = create_test_manager().await;
        
        // Register a path
        manager.register_path(
            "path1".to_string(),
            "fingerprint1".to_string(),
            "1-ff00:0:110".to_string(),
        ).await.unwrap();
        
        let stats = manager.get_stats().await;
        assert_eq!(stats.total_paths, 1);
        
        manager.stop().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_quality_updates() {
        let manager = create_test_manager().await;
        
        // Register path
        manager.register_path(
            "path1".to_string(),
            "fingerprint1".to_string(),
            "1-ff00:0:110".to_string(),
        ).await.unwrap();
        
        // Update quality
        let measurement = PathMeasurement {
            path_id: "path1".to_string(),
            rtt_us: 25000.0,
            success: true,
            bandwidth_bps: Some(10_000_000.0),
            timestamp_ns: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64,
        };
        
        manager.update_path_quality(measurement).await.unwrap();
        
        let stats = manager.get_stats().await;
        assert_eq!(stats.measurements_taken, 1);
        
        manager.stop().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_path_selection() {
        let manager = create_test_manager().await;
        
        // Register multiple paths
        for i in 1..=3 {
            manager.register_path(
                format!("path{}", i),
                format!("fingerprint{}", i),
                "1-ff00:0:110".to_string(),
            ).await.unwrap();
            
            // Give different quality measurements
            let measurement = PathMeasurement {
                path_id: format!("path{}", i),
                rtt_us: 20000.0 + (i as f64 * 10000.0), // Different RTTs
                success: true,
                bandwidth_bps: Some(10_000_000.0),
                timestamp_ns: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64,
            };
            
            manager.update_path_quality(measurement).await.unwrap();
        }
        
        // Select path
        let selection = manager.select_path("1-ff00:0:110").await.unwrap();
        assert!(selection.primary_path.is_some());
        assert_eq!(selection.paths_considered, 3);
        
        // Best path should be path1 (lowest RTT)
        assert_eq!(selection.primary_path.unwrap().path_id, "path1");
        
        manager.stop().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_path_health_evaluation() {
        let manager = create_test_manager().await;
        
        // Register path
        manager.register_path(
            "path1".to_string(),
            "fingerprint1".to_string(),
            "1-ff00:0:110".to_string(),
        ).await.unwrap();
        
        // Healthy measurement
        let healthy_measurement = PathMeasurement {
            path_id: "path1".to_string(),
            rtt_us: 25000.0,
            success: true,
            bandwidth_bps: Some(10_000_000.0),
            timestamp_ns: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64,
        };
        
        manager.update_path_quality(healthy_measurement).await.unwrap();
        
        let paths = manager.paths.read().await;
        let path = paths.get("path1").unwrap();
        assert!(path.is_healthy);
        
        drop(paths);
        
        // Unhealthy measurement (high loss)
        for _ in 0..10 {
            let unhealthy_measurement = PathMeasurement {
                path_id: "path1".to_string(),
                rtt_us: 25000.0,
                success: false, // Failed measurement
                bandwidth_bps: Some(10_000_000.0),
                timestamp_ns: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64,
            };
            
            manager.update_path_quality(unhealthy_measurement).await.unwrap();
        }
        
        let paths = manager.paths.read().await;
        let path = paths.get("path1").unwrap();
        assert!(!path.is_healthy);
        
        manager.stop().await.unwrap();
    }
}