//! TLS Camouflage and Template Management
//!
//! Implements realistic TLS camouflage with:
//! - Template caching with TTL and stochastic reuse
//! - Background calibrator for template diversity
//! - Mixture models for timing/size distributions
//! - Fallback reducer with cover traffic patterns
//! - Anti-fingerprinting measures

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};
use rand::Rng;
use thiserror::Error;
use tokio::time::{interval, Interval};
use reqwest::Client;
use betanet_utls::{ChromeVersion, TlsTemplate, UtlsError};

/// TLS camouflage errors
#[derive(Debug, Error)]
pub enum TlsCamouflageError {
    #[error("Template cache error: {0}")]
    TemplateCache(String),

    #[error("Calibration error: {0}")]
    Calibration(String),

    #[error("Mixture model error: {0}")]
    MixtureModel(String),

    #[error("Cover traffic error: {0}")]
    CoverTraffic(String),

    #[error("uTLS error: {0}")]
    Utls(#[from] UtlsError),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

pub type Result<T> = std::result::Result<T, TlsCamouflageError>;

/// Template cache key for {origin, POP, ALPN}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TemplateKey {
    pub origin: String,
    pub pop_location: String, // Point of Presence
    pub alpn: String, // Application Layer Protocol Negotiation
}

impl TemplateKey {
    pub fn new(origin: impl Into<String>, pop: impl Into<String>, alpn: impl Into<String>) -> Self {
        Self {
            origin: origin.into(),
            pop_location: pop.into(),
            alpn: alpn.into(),
        }
    }

    pub fn http2(origin: impl Into<String>, pop: impl Into<String>) -> Self {
        Self::new(origin, pop, "h2")
    }

    pub fn http3(origin: impl Into<String>, pop: impl Into<String>) -> Self {
        Self::new(origin, pop, "h3")
    }
}

/// Cached TLS template with metadata
#[derive(Debug, Clone)]
pub struct CachedTemplate {
    pub template: TlsTemplate,
    pub created_at: SystemTime,
    pub expires_at: SystemTime,
    pub use_count: u32,
    pub last_used: SystemTime,
    pub jitter_factor: f64,
    pub site_class: SiteClass,
}

impl CachedTemplate {
    pub fn new(template: TlsTemplate, ttl: Duration, site_class: SiteClass) -> Self {
        let now = SystemTime::now();
        let mut rng = rand::thread_rng();
        let jitter_factor = rng.gen_range(0.8..1.2); // ±20% jitter
        let actual_ttl = Duration::from_secs_f64(ttl.as_secs_f64() * jitter_factor);

        Self {
            template,
            created_at: now,
            expires_at: now + actual_ttl,
            use_count: 0,
            last_used: now,
            jitter_factor,
            site_class,
        }
    }

    pub fn is_expired(&self) -> bool {
        SystemTime::now() > self.expires_at
    }

    pub fn record_use(&mut self) {
        self.use_count += 1;
        self.last_used = SystemTime::now();
    }

    pub fn time_remaining(&self) -> Duration {
        self.expires_at.duration_since(SystemTime::now()).unwrap_or_default()
    }
}

/// Site classification for different behavioral profiles
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SiteClass {
    CDN,      // Content Delivery Network
    Social,   // Social media platforms
    Commerce, // E-commerce sites
    News,     // News and media sites
    Tech,     // Technology/developer sites
    Finance,  // Financial services
    Gaming,   // Gaming platforms
    Stream,   // Video/audio streaming
    Unknown,  // Unclassified
}

impl SiteClass {
    pub fn from_origin(origin: &str) -> Self {
        let origin_lower = origin.to_lowercase();

        // CDN detection
        if origin_lower.contains("cloudflare")
            || origin_lower.contains("cloudfront")
            || origin_lower.contains("fastly")
            || origin_lower.contains("cdn") {
            return Self::CDN;
        }

        // Social media
        if origin_lower.contains("facebook")
            || origin_lower.contains("twitter")
            || origin_lower.contains("instagram")
            || origin_lower.contains("linkedin") {
            return Self::Social;
        }

        // E-commerce
        if origin_lower.contains("amazon")
            || origin_lower.contains("ebay")
            || origin_lower.contains("shopify") {
            return Self::Commerce;
        }

        // Streaming
        if origin_lower.contains("youtube")
            || origin_lower.contains("netflix")
            || origin_lower.contains("twitch")
            || origin_lower.contains("spotify") {
            return Self::Stream;
        }

        // Finance
        if origin_lower.contains("bank")
            || origin_lower.contains("paypal")
            || origin_lower.contains("visa") {
            return Self::Finance;
        }

        Self::Unknown
    }

    pub fn typical_connection_profile(&self) -> ConnectionProfile {
        match self {
            Self::CDN => ConnectionProfile {
                avg_connection_time: Duration::from_millis(150),
                typical_payload_size: 50_000,
                burst_probability: 0.3,
                keepalive_duration: Duration::from_secs(30),
            },
            Self::Social => ConnectionProfile {
                avg_connection_time: Duration::from_millis(300),
                typical_payload_size: 25_000,
                burst_probability: 0.7,
                keepalive_duration: Duration::from_secs(120),
            },
            Self::Commerce => ConnectionProfile {
                avg_connection_time: Duration::from_millis(400),
                typical_payload_size: 35_000,
                burst_probability: 0.2,
                keepalive_duration: Duration::from_secs(60),
            },
            Self::Stream => ConnectionProfile {
                avg_connection_time: Duration::from_millis(800),
                typical_payload_size: 500_000,
                burst_probability: 0.1,
                keepalive_duration: Duration::from_secs(300),
            },
            _ => ConnectionProfile {
                avg_connection_time: Duration::from_millis(250),
                typical_payload_size: 30_000,
                burst_probability: 0.4,
                keepalive_duration: Duration::from_secs(90),
            },
        }
    }
}

/// Connection behavior profile for site classes
#[derive(Debug, Clone)]
pub struct ConnectionProfile {
    pub avg_connection_time: Duration,
    pub typical_payload_size: usize,
    pub burst_probability: f64,
    pub keepalive_duration: Duration,
}

/// Template cache configuration
#[derive(Debug, Clone)]
pub struct TemplateCacheConfig {
    pub max_cache_size: usize,
    pub default_ttl: Duration,
    pub jitter_range: (f64, f64),
    pub stochastic_reuse_probability: f64,
    pub background_refresh_interval: Duration,
    pub max_use_count: u32,
    pub enable_cover_traffic: bool,
}

impl Default for TemplateCacheConfig {
    fn default() -> Self {
        Self {
            max_cache_size: 500,
            default_ttl: Duration::from_secs(1800), // 30 minutes
            jitter_range: (0.7, 1.3), // ±30% jitter
            stochastic_reuse_probability: 0.85,
            background_refresh_interval: Duration::from_secs(300), // 5 minutes
            max_use_count: 50,
            enable_cover_traffic: true,
        }
    }
}

/// Template cache with TTL and stochastic reuse
pub struct TemplateCache {
    cache: Arc<RwLock<HashMap<TemplateKey, CachedTemplate>>>,
    config: TemplateCacheConfig,
    background_calibrator: Option<BackgroundCalibrator>,
    mixture_models: Arc<Mutex<HashMap<SiteClass, MixtureModel>>>,
    cover_pool: Arc<Mutex<VecDeque<CoverTrafficEntry>>>,
    stats: Arc<Mutex<CacheStats>>,
}

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub evictions: u64,
    pub background_refreshes: u64,
    pub stochastic_reuses: u64,
    pub cover_traffic_generated: u64,
}

impl CacheStats {
    pub fn hit_ratio(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total as f64
        }
    }
}

/// Log-normal component for mixture models
#[derive(Debug, Clone)]
pub struct LogNormalComponent {
    pub weight: f64,
    pub mu: f64,    // log-scale location parameter
    pub sigma: f64, // log-scale parameter
}

impl LogNormalComponent {
    pub fn sample(&self, rng: &mut impl Rng) -> f64 {
        // Stub implementation using normal distribution approximation
        let normal: f64 = rng.gen_range(-3.0..3.0);
        (normal * self.sigma + self.mu).exp()
    }
}

/// Mixture model for timing/size distributions
#[derive(Debug, Clone)]
pub struct MixtureModel {
    pub timing_components: Vec<LogNormalComponent>,
    pub size_components: Vec<LogNormalComponent>,
    pub site_class: SiteClass,
}

impl MixtureModel {
    pub fn new(site_class: SiteClass) -> Self {
        let (timing_components, size_components) = match site_class {
            SiteClass::CDN => (
                vec![
                    LogNormalComponent { weight: 0.6, mu: 4.6, sigma: 0.5 }, // ~100ms fast
                    LogNormalComponent { weight: 0.4, mu: 5.3, sigma: 0.3 }, // ~200ms normal
                ],
                vec![
                    LogNormalComponent { weight: 0.5, mu: 10.0, sigma: 0.8 }, // ~20KB
                    LogNormalComponent { weight: 0.5, mu: 11.5, sigma: 1.0 }, // ~100KB
                ],
            ),
            SiteClass::Social => (
                vec![
                    LogNormalComponent { weight: 0.3, mu: 5.0, sigma: 0.4 }, // ~150ms
                    LogNormalComponent { weight: 0.7, mu: 5.7, sigma: 0.6 }, // ~300ms
                ],
                vec![
                    LogNormalComponent { weight: 0.8, mu: 9.5, sigma: 0.7 }, // ~15KB
                    LogNormalComponent { weight: 0.2, mu: 12.0, sigma: 1.2 }, // ~160KB images
                ],
            ),
            SiteClass::Stream => (
                vec![
                    LogNormalComponent { weight: 0.2, mu: 6.0, sigma: 0.3 }, // ~400ms initial
                    LogNormalComponent { weight: 0.8, mu: 6.7, sigma: 0.8 }, // ~800ms streaming
                ],
                vec![
                    LogNormalComponent { weight: 0.1, mu: 12.5, sigma: 0.5 }, // ~300KB metadata
                    LogNormalComponent { weight: 0.9, mu: 15.5, sigma: 1.5 }, // ~5MB chunks
                ],
            ),
            _ => (
                vec![
                    LogNormalComponent { weight: 1.0, mu: 5.5, sigma: 0.6 }, // ~250ms
                ],
                vec![
                    LogNormalComponent { weight: 1.0, mu: 10.5, sigma: 0.9 }, // ~35KB
                ],
            ),
        };

        Self {
            timing_components,
            size_components,
            site_class,
        }
    }

    pub fn sample_timing(&self, rng: &mut impl Rng) -> Duration {
        let component = self.choose_timing_component(rng);
        let millis = component.sample(rng).max(10.0).min(5000.0); // 10ms to 5s bounds
        Duration::from_millis(millis as u64)
    }

    pub fn sample_size(&self, rng: &mut impl Rng) -> usize {
        let component = self.choose_size_component(rng);
        let bytes = component.sample(rng).max(100.0).min(10_000_000.0); // 100B to 10MB bounds
        bytes as usize
    }

    fn choose_timing_component(&self, rng: &mut impl Rng) -> &LogNormalComponent {
        let r: f64 = rng.gen();
        let mut cumulative_weight = 0.0;

        for component in &self.timing_components {
            cumulative_weight += component.weight;
            if r <= cumulative_weight {
                return component;
            }
        }

        self.timing_components.last().unwrap()
    }

    fn choose_size_component(&self, rng: &mut impl Rng) -> &LogNormalComponent {
        let r: f64 = rng.gen();
        let mut cumulative_weight = 0.0;

        for component in &self.size_components {
            cumulative_weight += component.weight;
            if r <= cumulative_weight {
                return component;
            }
        }

        self.size_components.last().unwrap()
    }
}

/// Cover traffic entry for background pool
#[derive(Debug, Clone)]
pub struct CoverTrafficEntry {
    pub template: TlsTemplate,
    pub timing_delay: Duration,
    pub payload_size: usize,
    pub created_at: SystemTime,
    pub site_class: SiteClass,
}

/// Background calibrator for template diversity
pub struct BackgroundCalibrator {
    consenting_origins: Vec<String>,
    calibration_interval: Interval,
    template_cache: Arc<RwLock<HashMap<TemplateKey, CachedTemplate>>>,
    mixture_models: Arc<Mutex<HashMap<SiteClass, MixtureModel>>>,
    cover_pool: Arc<Mutex<VecDeque<CoverTrafficEntry>>>,
    is_running: Arc<Mutex<bool>>,
}

impl BackgroundCalibrator {
    pub fn new(
        consenting_origins: Vec<String>,
        refresh_interval: Duration,
        template_cache: Arc<RwLock<HashMap<TemplateKey, CachedTemplate>>>,
        mixture_models: Arc<Mutex<HashMap<SiteClass, MixtureModel>>>,
        cover_pool: Arc<Mutex<VecDeque<CoverTrafficEntry>>>,
    ) -> Self {
        Self {
            consenting_origins,
            calibration_interval: interval(refresh_interval),
            template_cache,
            mixture_models,
            cover_pool,
            is_running: Arc::new(Mutex::new(false)),
        }
    }

    pub async fn start(&mut self) -> Result<()> {
        {
            let mut running = self.is_running.lock().unwrap();
            if *running {
                return Ok(()); // Already running
            }
            *running = true;
        }

        // Note: Background calibration is disabled in stub implementation
        Ok(())
    }

    pub fn stop(&self) {
        let mut running = self.is_running.lock().unwrap();
        *running = false;
    }

    async fn calibration_loop(mut self) {
        loop {
            {
                let running = self.is_running.lock().unwrap();
                if !*running {
                    break;
                }
            }

            self.calibration_interval.tick().await;

            if let Err(e) = self.perform_calibration().await {
                eprintln!("Background calibration error: {}", e);
            }
        }
    }

    async fn perform_calibration(&self) -> Result<()> {
        let mut rng = rand::thread_rng();

        // Select random consenting origins for calibration
        let origins_to_calibrate: Vec<_> = self.consenting_origins
            .iter()
            .filter(|_| rng.gen_bool(0.3)) // 30% chance each origin gets calibrated
            .cloned()
            .collect();

        for origin in origins_to_calibrate {
            if let Err(e) = self.calibrate_origin(&origin).await {
                eprintln!("Failed to calibrate origin {}: {}", origin, e);
            }
        }

        // Generate cover traffic entries
        self.generate_cover_entries().await?;

        Ok(())
    }

    async fn calibrate_origin(&self, origin: &str) -> Result<()> {
        let site_class = SiteClass::from_origin(origin);
        let pop_locations = vec!["us-east", "us-west", "eu-west", "ap-southeast"];
        let alpns = vec!["h2", "h3"];

        let mut rng = rand::thread_rng();

        for pop in pop_locations.iter() {
            for alpn in alpns.iter() {
                // Skip some combinations probabilistically
                if rng.gen_bool(0.7) {
                    continue;
                }

                let key = TemplateKey::new(origin, *pop, *alpn);

                // Attempt to gather real network data
                if let Ok((timing, size)) = self.fetch_network_data(origin).await {
                    self.update_mixture_model(site_class, timing, size);
                }

                // Create new template from observed data
                let template = self.create_calibrated_template(&key, site_class).await?;
                let cached_template = CachedTemplate::new(
                    template,
                    Duration::from_secs(rng.gen_range(1200..3600)), // 20-60 min TTL
                    site_class,
                );

                // Insert into cache
                let mut cache = self.template_cache.write().unwrap();
                cache.insert(key, cached_template);
            }
        }

        Ok(())
    }

    async fn fetch_network_data(&self, origin: &str) -> Result<(Duration, usize)> {
        let client = Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .map_err(|e| TlsCamouflageError::Calibration(e.to_string()))?;

        let url = format!("https://{}/", origin);
        let start = Instant::now();
        let resp = client
            .get(&url)
            .send()
            .await
            .map_err(|e| TlsCamouflageError::Calibration(e.to_string()))?;
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| TlsCamouflageError::Calibration(e.to_string()))?;
        let elapsed = start.elapsed();
        Ok((elapsed, bytes.len()))
    }

    fn update_mixture_model(&self, site_class: SiteClass, timing: Duration, size: usize) {
        let mut models = self.mixture_models.lock().unwrap();
        let model = models.entry(site_class).or_insert_with(|| MixtureModel::new(site_class));

        let log_t = (timing.as_millis().max(1) as f64).ln();
        if let Some(comp) = model.timing_components.get_mut(0) {
            let diff = log_t - comp.mu;
            comp.mu += 0.1 * diff;
            comp.sigma = 0.9 * comp.sigma + 0.1 * diff.abs();
        }

        let log_s = (size.max(1) as f64).ln();
        if let Some(comp) = model.size_components.get_mut(0) {
            let diff = log_s - comp.mu;
            comp.mu += 0.1 * diff;
            comp.sigma = 0.9 * comp.sigma + 0.1 * diff.abs();
        }
    }

    async fn create_calibrated_template(&self, key: &TemplateKey, site_class: SiteClass) -> Result<TlsTemplate> {
        let chrome_version = ChromeVersion::current_stable_n2();
        let mut template = TlsTemplate::for_chrome(chrome_version, &key.origin);
        template.cipher_suites = self.generate_site_specific_ciphers(site_class)?;
        template.extensions = self.generate_site_specific_extensions(site_class, &key.alpn)?;
        template.alpn_protocols = vec![key.alpn.clone()];
        template.server_name = key.origin.clone();
        Ok(template)
    }

    fn generate_site_specific_ciphers(&self, site_class: SiteClass) -> Result<Vec<u16>> {
        // TLS 1.3 cipher suites (stub implementation)
        const TLS_AES_128_GCM_SHA256: u16 = 0x1301;
        const TLS_AES_256_GCM_SHA384: u16 = 0x1302;
        const TLS_CHACHA20_POLY1305_SHA256: u16 = 0x1303;
        // TLS 1.2 cipher suites
        const TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256: u16 = 0xc02f;
        const TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384: u16 = 0xc030;
        const TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256: u16 = 0xc02b;
        const TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384: u16 = 0xc02c;

        match site_class {
            SiteClass::Finance => Ok(vec![
                TLS_AES_256_GCM_SHA384,
                TLS_CHACHA20_POLY1305_SHA256,
                TLS_AES_128_GCM_SHA256,
                TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
                TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
            ]),
            SiteClass::CDN => Ok(vec![
                TLS_AES_128_GCM_SHA256,
                TLS_CHACHA20_POLY1305_SHA256,
                TLS_AES_256_GCM_SHA384,
                TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
                TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
            ]),
            _ => Ok(vec![
                TLS_AES_128_GCM_SHA256,
                TLS_AES_256_GCM_SHA384,
                TLS_CHACHA20_POLY1305_SHA256,
                TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
                TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
            ]),
        }
    }

    fn generate_site_specific_extensions(&self, site_class: SiteClass, alpn: &str) -> Result<Vec<u16>> {
        // TLS extension types (stub implementation)
        const SERVER_NAME: u16 = 0x0000;
        const SUPPORTED_GROUPS: u16 = 0x000a;
        const SIGNATURE_ALGORITHMS: u16 = 0x000d;
        const SUPPORTED_VERSIONS: u16 = 0x002b;
        const ALPN: u16 = 0x0010;
        const KEY_SHARE: u16 = 0x0033;
        const PSK_KEY_EXCHANGE_MODES: u16 = 0x002d;

        let mut extensions = vec![
            SERVER_NAME,
            SUPPORTED_GROUPS,
            SIGNATURE_ALGORITHMS,
            SUPPORTED_VERSIONS,
        ];

        if alpn == "h2" || alpn == "h3" {
            extensions.push(ALPN);
        }

        match site_class {
            SiteClass::Social | SiteClass::Stream => {
                extensions.push(KEY_SHARE);
                extensions.push(PSK_KEY_EXCHANGE_MODES);
            }
            _ => {
                extensions.push(KEY_SHARE);
            }
        }

        Ok(extensions)
    }

    async fn generate_cover_entries(&self) -> Result<()> {
        let mut rng = rand::thread_rng();
        let mixture_models = self.mixture_models.lock().unwrap();
        let mut cover_pool = self.cover_pool.lock().unwrap();

        // Generate 5-15 cover entries per calibration cycle
        let num_entries = rng.gen_range(5..=15);

        for _ in 0..num_entries {
            let site_class = match rng.gen_range(0..8) {
                0 => SiteClass::CDN,
                1 => SiteClass::Social,
                2 => SiteClass::Commerce,
                3 => SiteClass::News,
                4 => SiteClass::Tech,
                5 => SiteClass::Finance,
                6 => SiteClass::Gaming,
                7 => SiteClass::Stream,
                _ => SiteClass::Unknown,
            };

            let model = mixture_models.get(&site_class)
                .unwrap_or_else(|| mixture_models.get(&SiteClass::Unknown).unwrap());

            let timing_delay = model.sample_timing(&mut rng);
            let payload_size = model.sample_size(&mut rng);

            // Create synthetic template for cover traffic
            let template = TlsTemplate {
                version: ChromeVersion::current_stable_n2(),
                cipher_suites: vec![0x1301, 0x1302, 0x1303],
                extensions: vec![0x0000, 0x000a, 0x000d, 0x002b],
                curves: vec![0x001d, 0x0017],
                signature_algorithms: vec![0x0804, 0x0805],
                alpn_protocols: vec!["h2".to_string()],
                server_name: format!("cover-{}.example.com", rng.gen::<u32>()),
            };

            let entry = CoverTrafficEntry {
                template,
                timing_delay,
                payload_size,
                created_at: SystemTime::now(),
                site_class,
            };

            cover_pool.push_back(entry);
        }

        // Limit cover pool size
        while cover_pool.len() > 200 {
            cover_pool.pop_front();
        }

        Ok(())
    }
}

impl TemplateCache {
    pub fn new(config: TemplateCacheConfig) -> Self {
        let cache = Arc::new(RwLock::new(HashMap::new()));
        let mixture_models = Arc::new(Mutex::new(HashMap::new()));
        let cover_pool = Arc::new(Mutex::new(VecDeque::new()));

        // Initialize mixture models for all site classes
        {
            let mut models = mixture_models.lock().unwrap();
            for site_class in [
                SiteClass::CDN,
                SiteClass::Social,
                SiteClass::Commerce,
                SiteClass::News,
                SiteClass::Tech,
                SiteClass::Finance,
                SiteClass::Gaming,
                SiteClass::Stream,
                SiteClass::Unknown,
            ] {
                models.insert(site_class, MixtureModel::new(site_class));
            }
        }

        Self {
            cache,
            config,
            background_calibrator: None,
            mixture_models,
            cover_pool,
            stats: Arc::new(Mutex::new(CacheStats::default())),
        }
    }

    pub fn with_background_calibrator(mut self, consenting_origins: Vec<String>) -> Self {
        let calibrator = BackgroundCalibrator::new(
            consenting_origins,
            self.config.background_refresh_interval,
            Arc::clone(&self.cache),
            Arc::clone(&self.mixture_models),
            Arc::clone(&self.cover_pool),
        );
        self.background_calibrator = Some(calibrator);
        self
    }

    pub async fn start(&mut self) -> Result<()> {
        if let Some(ref mut calibrator) = self.background_calibrator {
            calibrator.start().await?;
        }
        Ok(())
    }

    pub fn stop(&self) {
        if let Some(ref calibrator) = self.background_calibrator {
            calibrator.stop();
        }
    }

    pub fn get_template(&self, key: &TemplateKey) -> Option<TlsTemplate> {
        let mut rng = rand::thread_rng();
        let mut stats = self.stats.lock().unwrap();

        // Check for stochastic reuse decision
        if rng.gen_bool(self.config.stochastic_reuse_probability) {
            {
                let cache = self.cache.read().unwrap();
                if let Some(cached) = cache.get(key) {
                    if !cached.is_expired() && cached.use_count < self.config.max_use_count {
                        let template = cached.template.clone();
                        stats.cache_hits += 1;
                        stats.stochastic_reuses += 1;
                        drop(stats);
                        drop(cache);

                        // Update usage counter
                        let mut cache_write = self.cache.write().unwrap();
                        if let Some(cached_mut) = cache_write.get_mut(key) {
                            cached_mut.record_use();
                        }

                        return Some(template);
                    }
                }
            }
        }

        // Cache miss - check if we should draw from cover pool
        stats.cache_misses += 1;
        drop(stats);

        if self.config.enable_cover_traffic && rng.gen_bool(0.3) {
            if let Some(cover_template) = self.get_cover_template(key) {
                return Some(cover_template);
            }
        }

        None
    }

    pub fn insert_template(&self, key: TemplateKey, template: TlsTemplate) -> Result<()> {
        let site_class = SiteClass::from_origin(&key.origin);
        let cached = CachedTemplate::new(template, self.config.default_ttl, site_class);

        let mut cache = self.cache.write().unwrap();

        // Check if cache is full and evict if necessary
        if cache.len() >= self.config.max_cache_size {
            self.evict_expired_or_lru(&mut cache);
        }

        cache.insert(key, cached);
        Ok(())
    }

    fn get_cover_template(&self, key: &TemplateKey) -> Option<TlsTemplate> {
        let mut cover_pool = self.cover_pool.lock().unwrap();
        let site_class = SiteClass::from_origin(&key.origin);

        // Find suitable cover entry
        let matching_entry_index = cover_pool
            .iter()
            .enumerate()
            .find(|(_, entry)| {
                entry.site_class == site_class || entry.site_class == SiteClass::Unknown
            })
            .map(|(index, _)| index);

        if let Some(index) = matching_entry_index {
            let entry = cover_pool.remove(index)?;
            let mut stats = self.stats.lock().unwrap();
            stats.cover_traffic_generated += 1;
            drop(stats);

            Some(entry.template)
        } else {
            None
        }
    }

    fn evict_expired_or_lru(&self, cache: &mut HashMap<TemplateKey, CachedTemplate>) {
        let _now = SystemTime::now();

        // First pass: remove expired entries
        let expired_keys: Vec<_> = cache
            .iter()
            .filter(|(_, cached)| cached.is_expired())
            .map(|(key, _)| key.clone())
            .collect();

        for key in expired_keys {
            cache.remove(&key);
            let mut stats = self.stats.lock().unwrap();
            stats.evictions += 1;
        }

        // If still over capacity, evict LRU entries
        while cache.len() >= self.config.max_cache_size {
            let lru_key = cache
                .iter()
                .min_by_key(|(_, cached)| cached.last_used)
                .map(|(key, _)| key.clone());

            if let Some(key) = lru_key {
                cache.remove(&key);
                let mut stats = self.stats.lock().unwrap();
                stats.evictions += 1;
            } else {
                break;
            }
        }
    }

    pub fn cleanup_expired(&self) -> usize {
        let mut cache = self.cache.write().unwrap();
        let initial_size = cache.len();
        self.evict_expired_or_lru(&mut cache);
        initial_size - cache.len()
    }

    pub fn statistics(&self) -> CacheStats {
        self.stats.lock().unwrap().clone()
    }

    pub fn cache_info(&self) -> CacheInfo {
        let cache = self.cache.read().unwrap();
        let stats = self.stats.lock().unwrap();

        CacheInfo {
            size: cache.len(),
            capacity: self.config.max_cache_size,
            hit_ratio: stats.hit_ratio(),
            total_hits: stats.cache_hits,
            total_misses: stats.cache_misses,
            evictions: stats.evictions,
            stochastic_reuses: stats.stochastic_reuses,
            cover_traffic_generated: stats.cover_traffic_generated,
        }
    }
}

/// Cache information summary
#[derive(Debug, Clone)]
pub struct CacheInfo {
    pub size: usize,
    pub capacity: usize,
    pub hit_ratio: f64,
    pub total_hits: u64,
    pub total_misses: u64,
    pub evictions: u64,
    pub stochastic_reuses: u64,
    pub cover_traffic_generated: u64,
}

/// Fallback reducer to prevent "triple-burst" patterns
pub struct FallbackReducer {
    recent_connections: VecDeque<SystemTime>,
    burst_threshold: usize,
    time_window: Duration,
    cover_pool: Arc<Mutex<VecDeque<CoverTrafficEntry>>>,
}

impl FallbackReducer {
    pub fn new(
        burst_threshold: usize,
        time_window: Duration,
        cover_pool: Arc<Mutex<VecDeque<CoverTrafficEntry>>>,
    ) -> Self {
        Self {
            recent_connections: VecDeque::new(),
            burst_threshold,
            time_window,
            cover_pool,
        }
    }

    pub fn should_allow_connection(&mut self) -> bool {
        let now = SystemTime::now();

        // Clean old entries
        while let Some(&front_time) = self.recent_connections.front() {
            if now.duration_since(front_time).unwrap_or_default() > self.time_window {
                self.recent_connections.pop_front();
            } else {
                break;
            }
        }

        // Check if this would create a burst
        if self.recent_connections.len() >= self.burst_threshold {
            // Triple-burst detected - draw from cover pool instead
            false
        } else {
            self.recent_connections.push_back(now);
            true
        }
    }

    pub fn get_cover_connection(&self) -> Option<CoverTrafficEntry> {
        let mut cover_pool = self.cover_pool.lock().unwrap();
        cover_pool.pop_front()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_template_key_creation() {
        let key = TemplateKey::http2("example.com", "us-east");
        assert_eq!(key.origin, "example.com");
        assert_eq!(key.pop_location, "us-east");
        assert_eq!(key.alpn, "h2");
    }

    #[test]
    fn test_site_classification() {
        assert_eq!(SiteClass::from_origin("cdn.cloudflare.com"), SiteClass::CDN);
        assert_eq!(SiteClass::from_origin("facebook.com"), SiteClass::Social);
        assert_eq!(SiteClass::from_origin("amazon.com"), SiteClass::Commerce);
        assert_eq!(SiteClass::from_origin("unknown-site.com"), SiteClass::Unknown);
    }

    #[test]
    fn test_cached_template_expiration() {
        let template = TlsTemplate {
            version: ChromeVersion::current_stable_n2(),
            cipher_suites: vec![],
            extensions: vec![],
            curves: vec![],
            signature_algorithms: vec![],
            alpn_protocols: vec![],
            server_name: "test".to_string(),
        };

        let cached = CachedTemplate::new(template, Duration::from_millis(100), SiteClass::Unknown);

        assert!(!cached.is_expired());
        std::thread::sleep(Duration::from_millis(150));
        assert!(cached.is_expired());
    }

    #[tokio::test]
    async fn test_template_cache_basic() {
        let cache = TemplateCache::new(TemplateCacheConfig::default());
        let key = TemplateKey::http2("example.com", "us-east");

        // Cache miss first
        assert!(cache.get_template(&key).is_none());

        let template = TlsTemplate {
            version: ChromeVersion::current_stable_n2(),
            cipher_suites: vec![0x1301],
            extensions: vec![0x0000],
            curves: vec![],
            signature_algorithms: vec![],
            alpn_protocols: vec!["h2".to_string()],
            server_name: "example.com".to_string(),
        };

        // Insert template
        cache.insert_template(key.clone(), template.clone()).unwrap();

        // Should get template back (stochastically)
        let _retrieved = cache.get_template(&key);
        // Note: Due to stochastic reuse, this might not always return the template
        // In a real test, you'd want to control the randomness
    }

    #[test]
    fn test_mixture_model_sampling() {
        let mut rng = rand::thread_rng();
        let model = MixtureModel::new(SiteClass::CDN);

        let timing = model.sample_timing(&mut rng);
        let size = model.sample_size(&mut rng);

        assert!(timing.as_millis() >= 10);
        assert!(timing.as_millis() <= 5000);
        assert!(size >= 100);
        assert!(size <= 10_000_000);
    }

    #[test]
    fn test_fallback_reducer() {
        let cover_pool = Arc::new(Mutex::new(VecDeque::new()));
        let mut reducer = FallbackReducer::new(3, Duration::from_secs(1), cover_pool);

        // First few connections should be allowed
        assert!(reducer.should_allow_connection());
        assert!(reducer.should_allow_connection());
        assert!(reducer.should_allow_connection());

        // Fourth connection should trigger burst protection
        assert!(!reducer.should_allow_connection());
    }
}

/// Apply JA3 template for fingerprint resistance
pub fn apply_ja3_template(template: &str) -> Result<()> {
    // Stub implementation - would configure TLS client with JA3 template
    tracing::info!("Applying JA3 template: {}", template);
    Ok(())
}

/// Create TLS connector with HTX camouflage
pub fn create_tls_connector() -> Result<TlsConnectorStub> {
    // Stub implementation - would create real TLS connector
    Ok(TlsConnectorStub::new())
}

/// Stub TLS connector for testing
pub struct TlsConnectorStub {
    enabled: bool,
}

impl TlsConnectorStub {
    pub fn new() -> Self {
        Self { enabled: true }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

/// TLS camouflage builder for configuring TLS settings
pub struct TlsCamouflageBuilder {
    enable_ech: bool,
    ja3_template: Option<String>,
}

impl TlsCamouflageBuilder {
    pub fn new() -> Self {
        Self {
            enable_ech: false,
            ja3_template: None,
        }
    }

    pub fn with_ech(mut self, enable: bool) -> Self {
        self.enable_ech = enable;
        self
    }

    pub fn with_ja3_template(mut self, template: String) -> Self {
        self.ja3_template = Some(template);
        self
    }

    pub fn build(self) -> Result<TlsConnectorStub> {
        // Apply configurations
        if let Some(template) = &self.ja3_template {
            apply_ja3_template(template)?;
        }

        if self.enable_ech {
            tracing::info!("ECH enabled for TLS camouflage");
        }

        create_tls_connector()
    }
}
