//! Enhanced Chrome fingerprint calibration with per-origin tuning
//!
//! Addresses operational fragility by implementing per-origin calibration
//! with camouflaged pre-flights and mobile budget management.

use crate::{
    config::ChromeFingerprintConfig,
    error::{BetanetError, Result},
};

use bytes::Bytes;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{Mutex, RwLock, Semaphore};
use tracing::{debug, error, info, warn};

/// Chrome fingerprint template with auto-refresh capability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChromeFingerprintTemplate {
    pub ja3_hash: String,
    pub ja4_hash: String,
    pub cipher_suites: Vec<String>,
    pub extensions: Vec<u16>,
    pub elliptic_curves: Vec<u16>,
    pub signature_algorithms: Vec<u16>,
    pub alpn_protocols: Vec<String>,
    pub h2_settings: HashMap<String, u32>,
    pub chrome_version: String,
    pub creation_timestamp: f64,
}

/// Per-origin fingerprint data for precise mimicry
#[derive(Debug, Clone)]
pub struct OriginFingerprint {
    pub hostname: String,
    pub tls_template: ChromeFingerprintTemplate,
    pub cookie_names: Vec<String>,
    pub cookie_length_histogram: HashMap<usize, f64>, // length -> frequency
    pub header_patterns: Vec<Vec<String>>, // ordered header sequences
    pub content_lengths: Vec<usize>,
    pub response_timing_ms: Vec<f64>,
    pub h2_settings_exact: HashMap<String, u32>, // exact H2 settings (not ±15%)
    pub last_calibrated: f64,
    pub calibration_count: u32,
    pub pop_selection_hints: Vec<String>, // CDN POP hints
}

/// Mobile budget manager for cover traffic
pub struct MobileBudgetManager {
    cover_traffic_budget_bytes: usize,
    cover_traffic_budget_time: Duration,
    last_cover_burst: Arc<Mutex<HashMap<String, Instant>>>,
    burst_count: Arc<Mutex<HashMap<String, u32>>>,
}

/// Chrome template manager with auto-refresh
pub struct ChromeTemplateManagerV2 {
    templates: Arc<RwLock<HashMap<String, ChromeFingerprintTemplate>>>,
    supported_versions: Vec<String>,
    auto_refresh_interval: Duration,
}

/// Origin calibrator with camouflaged pre-flights
pub struct OriginCalibratorV2 {
    origins: Arc<RwLock<HashMap<String, OriginFingerprint>>>,
    mobile_budget: Arc<MobileBudgetManager>,
    calibration_semaphore: Arc<Semaphore>,
    client: Client,
}

impl ChromeFingerprintTemplate {
    /// Check if template is stale and needs refresh
    pub fn is_stale(&self, max_age_hours: u64) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
        (now - self.creation_timestamp) > (max_age_hours as f64 * 3600.0)
    }
}

impl OriginFingerprint {
    /// Check if origin needs recalibration
    pub fn needs_recalibration(&self, max_age_hours: u64) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
        (now - self.last_calibrated) > (max_age_hours as f64 * 3600.0)
    }
}

impl MobileBudgetManager {
    pub fn new() -> Self {
        Self {
            cover_traffic_budget_bytes: 150 * 1024, // 150KB per burst
            cover_traffic_budget_time: Duration::from_secs(10 * 60), // 10 min between bursts
            last_cover_burst: Arc::new(Mutex::new(HashMap::new())),
            burst_count: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Check if cover traffic is within mobile budget
    pub async fn can_create_cover_traffic(&self, origin: &str, bytes_needed: usize) -> bool {
        let now = Instant::now();
        let last_burst_map = self.last_cover_burst.lock().await;

        if let Some(last_burst) = last_burst_map.get(origin) {
            // Check time budget
            if now.duration_since(*last_burst) < self.cover_traffic_budget_time {
                return false;
            }
        }

        // Check bytes budget
        bytes_needed <= self.cover_traffic_budget_bytes
    }

    /// Record cover traffic usage
    pub async fn record_cover_traffic(&self, origin: &str, bytes_used: usize) {
        let mut last_burst_map = self.last_cover_burst.lock().await;
        let mut burst_count_map = self.burst_count.lock().await;

        last_burst_map.insert(origin.to_string(), Instant::now());
        *burst_count_map.entry(origin.to_string()).or_insert(0) += 1;

        debug!("Cover traffic burst to {}: {} bytes", origin, bytes_used);
    }
}

impl ChromeTemplateManagerV2 {
    pub fn new() -> Self {
        Self {
            templates: Arc::new(RwLock::new(HashMap::new())),
            supported_versions: vec!["N".to_string(), "N-1".to_string(), "N-2".to_string()],
            auto_refresh_interval: Duration::from_secs(6 * 3600), // 6 hours
        }
    }

    /// Get Chrome template, refreshing if stale
    pub async fn get_template(&self, version: &str) -> Result<ChromeFingerprintTemplate> {
        let templates = self.templates.read().await;

        if let Some(template) = templates.get(version) {
            if !template.is_stale(24) {
                return Ok(template.clone());
            }
        }

        drop(templates);
        self.refresh_template(version).await
    }

    /// Refresh Chrome template from latest Chrome builds
    async fn refresh_template(&self, version: &str) -> Result<ChromeFingerprintTemplate> {
        info!("Refreshing Chrome template for version {}", version);

        // Simulate Chrome 120.0.6099.109 template
        let template = ChromeFingerprintTemplate {
            ja3_hash: "cd08e31494f9531f560d64c695473da9".to_string(),
            ja4_hash: "t13d1516h2_8daaf6152771e_02713d6af862".to_string(),
            cipher_suites: vec![
                "TLS_AES_128_GCM_SHA256".to_string(),
                "TLS_AES_256_GCM_SHA384".to_string(),
                "TLS_CHACHA20_POLY1305_SHA256".to_string(),
                "TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256".to_string(),
                "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256".to_string(),
            ],
            extensions: vec![0, 5, 10, 11, 13, 16, 18, 21, 23, 27, 35, 43, 45, 51],
            elliptic_curves: vec![29, 23, 24], // x25519, secp256r1, secp384r1
            signature_algorithms: vec![0x0804, 0x0403, 0x0805],
            alpn_protocols: vec!["h2".to_string(), "http/1.1".to_string()],
            h2_settings: [
                ("HEADER_TABLE_SIZE".to_string(), 65536),
                ("ENABLE_PUSH".to_string(), 0),
                ("MAX_CONCURRENT_STREAMS".to_string(), 1000),
                ("INITIAL_WINDOW_SIZE".to_string(), 6291456),
                ("MAX_FRAME_SIZE".to_string(), 16777215),
                ("MAX_HEADER_LIST_SIZE".to_string(), 262144),
            ].iter().cloned().collect(),
            chrome_version: "120.0.6099.109".to_string(),
            creation_timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
        };

        let mut templates = self.templates.write().await;
        templates.insert(version.to_string(), template.clone());

        Ok(template)
    }
}

impl OriginCalibratorV2 {
    pub fn new(mobile_budget: Arc<MobileBudgetManager>) -> Self {
        Self {
            origins: Arc::new(RwLock::new(HashMap::new())),
            mobile_budget,
            calibration_semaphore: Arc::new(Semaphore::new(3)), // Limit concurrent calibrations
            client: Client::builder()
                .timeout(Duration::from_secs(10))
                .build()
                .unwrap(),
        }
    }

    /// Get origin fingerprint, calibrating if needed
    pub async fn get_origin_fingerprint(&self, hostname: &str) -> Result<OriginFingerprint> {
        let origins = self.origins.read().await;

        if let Some(fingerprint) = origins.get(hostname) {
            if !fingerprint.needs_recalibration(6) {
                return Ok(fingerprint.clone());
            }
        }

        drop(origins);

        let _permit = self.calibration_semaphore.acquire().await.map_err(|_| {
            BetanetError::Chromeium("Calibration semaphore acquisition failed".to_string())
        })?;

        self.calibrate_origin(hostname).await
    }

    /// Perform camouflaged calibration against origin
    async fn calibrate_origin(&self, hostname: &str) -> Result<OriginFingerprint> {
        info!("Calibrating origin: {}", hostname);

        // Perform normal-looking fetch with timing camouflage
        let timing_data = self.camouflaged_calibration_fetch(hostname).await?;

        // Parse observed patterns
        let fingerprint = self.analyze_origin_patterns(hostname, timing_data).await?;

        // Cache results
        let mut origins = self.origins.write().await;
        origins.insert(hostname.to_string(), fingerprint.clone());

        Ok(fingerprint)
    }

    /// Perform camouflaged calibration that looks like normal browsing
    async fn camouflaged_calibration_fetch(&self, hostname: &str) -> Result<Vec<f64>> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Randomize delay to avoid correlation
        let site_specific_delay = Duration::from_millis(rng.gen_range(200..800));
        tokio::time::sleep(site_specific_delay).await;

        let common_paths = [
            "/favicon.ico",
            "/robots.txt",
            "/manifest.json",
            "/.well-known/security.txt",
        ];

        let path = common_paths[rng.gen_range(0..common_paths.len())];
        let url = format!("https://{}{}", hostname, path);

        let start = Instant::now();
        let result = self.client.get(&url).send().await;
        let timing = start.elapsed().as_millis() as f64;

        match result {
            Ok(response) => {
                // Read response to measure timing
                let _content = response.bytes().await.unwrap_or_default();
                debug!("Calibration fetch: {} -> success", url);
                Ok(vec![timing])
            }
            Err(e) => {
                debug!("Calibration fetch failed (expected): {}", e);
                // Return timing even on failure for baseline measurement
                Ok(vec![timing])
            }
        }
    }

    /// Analyze origin to extract fingerprint patterns
    async fn analyze_origin_patterns(
        &self,
        hostname: &str,
        timing_data: Vec<f64>,
    ) -> Result<OriginFingerprint> {
        // In production, this would analyze captured traffic
        // For now, create realistic patterns based on hostname

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        let fingerprint = OriginFingerprint {
            hostname: hostname.to_string(),
            tls_template: ChromeFingerprintTemplate {
                ja3_hash: format!("origin_{}_hash", hostname.replace('.', "_")),
                ja4_hash: format!("t13d1516h2_origin_{}", hostname.len()),
                cipher_suites: vec!["TLS_AES_128_GCM_SHA256".to_string()],
                extensions: vec![0, 5, 10],
                elliptic_curves: vec![29],
                signature_algorithms: vec![0x0804],
                alpn_protocols: vec!["h2".to_string()],
                h2_settings: [("HEADER_TABLE_SIZE".to_string(), 65536)].iter().cloned().collect(),
                chrome_version: "120.0.6099.109".to_string(),
                creation_timestamp: now,
            },
            cookie_names: vec!["_cfuvid".to_string(), "session_id".to_string(), "csrftoken".to_string()],
            cookie_length_histogram: [(24, 0.3), (32, 0.4), (48, 0.2), (64, 0.1)]
                .iter()
                .cloned()
                .collect(),
            header_patterns: vec![
                vec!["Host".to_string(), "User-Agent".to_string(), "Accept".to_string()],
                vec!["Host".to_string(), "User-Agent".to_string(), "Accept".to_string(), "Accept-Language".to_string()],
            ],
            content_lengths: vec![1024, 2048, 4096, 8192],
            response_timing_ms: timing_data,
            h2_settings_exact: [
                ("HEADER_TABLE_SIZE".to_string(), 65536),
                ("ENABLE_PUSH".to_string(), 0),
                ("MAX_CONCURRENT_STREAMS".to_string(), 1000),
            ].iter().cloned().collect(),
            last_calibrated: now,
            calibration_count: 1,
            pop_selection_hints: vec!["cf-ray".to_string(), "x-served-by".to_string(), "x-cache".to_string()],
        };

        Ok(fingerprint)
    }

    /// Create fallback fingerprint when calibration fails
    fn create_fallback_fingerprint(&self, hostname: &str) -> OriginFingerprint {
        warn!("Using fallback fingerprint for {}", hostname);

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        OriginFingerprint {
            hostname: hostname.to_string(),
            tls_template: ChromeFingerprintTemplate {
                ja3_hash: "fallback_hash".to_string(),
                ja4_hash: "fallback_ja4".to_string(),
                cipher_suites: vec!["TLS_AES_128_GCM_SHA256".to_string()],
                extensions: vec![0, 5, 10, 11, 13],
                elliptic_curves: vec![29, 23],
                signature_algorithms: vec![0x0804, 0x0403],
                alpn_protocols: vec!["h2".to_string(), "http/1.1".to_string()],
                h2_settings: [("HEADER_TABLE_SIZE".to_string(), 65536)].iter().cloned().collect(),
                chrome_version: "120.0.6099.109".to_string(),
                creation_timestamp: now,
            },
            cookie_names: vec!["sessionid".to_string()],
            cookie_length_histogram: [(32, 1.0)].iter().cloned().collect(),
            header_patterns: vec![vec!["Host".to_string(), "User-Agent".to_string(), "Accept".to_string()]],
            content_lengths: vec![1024],
            response_timing_ms: vec![100.0],
            h2_settings_exact: [("HEADER_TABLE_SIZE".to_string(), 65536)].iter().cloned().collect(),
            last_calibrated: now,
            calibration_count: 0,
            pop_selection_hints: vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chrome_template_stale_check() {
        let template = ChromeFingerprintTemplate {
            ja3_hash: "test".to_string(),
            ja4_hash: "test".to_string(),
            cipher_suites: vec![],
            extensions: vec![],
            elliptic_curves: vec![],
            signature_algorithms: vec![],
            alpn_protocols: vec![],
            h2_settings: HashMap::new(),
            chrome_version: "120.0".to_string(),
            creation_timestamp: 0.0, // Very old timestamp
        };

        assert!(template.is_stale(1)); // Should be stale after 1 hour
    }

    #[tokio::test]
    async fn test_mobile_budget_manager() {
        let manager = MobileBudgetManager::new();

        // Should allow first request
        assert!(manager.can_create_cover_traffic("example.com", 100 * 1024).await);

        // Record usage
        manager.record_cover_traffic("example.com", 100 * 1024).await;

        // Should deny immediate second request (time budget)
        assert!(!manager.can_create_cover_traffic("example.com", 100 * 1024).await);
    }

    #[test]
    fn test_origin_fingerprint_recalibration() {
        let fingerprint = OriginFingerprint {
            hostname: "test.com".to_string(),
            tls_template: ChromeFingerprintTemplate {
                ja3_hash: "test".to_string(),
                ja4_hash: "test".to_string(),
                cipher_suites: vec![],
                extensions: vec![],
                elliptic_curves: vec![],
                signature_algorithms: vec![],
                alpn_protocols: vec![],
                h2_settings: HashMap::new(),
                chrome_version: "120.0".to_string(),
                creation_timestamp: 0.0,
            },
            cookie_names: vec![],
            cookie_length_histogram: HashMap::new(),
            header_patterns: vec![],
            content_lengths: vec![],
            response_timing_ms: vec![],
            h2_settings_exact: HashMap::new(),
            last_calibrated: 0.0, // Very old timestamp
            calibration_count: 1,
            pop_selection_hints: vec![],
        };

        assert!(fingerprint.needs_recalibration(1)); // Should need recalibration after 1 hour
    }
}
