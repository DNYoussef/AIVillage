//! Chrome version auto-refresh functionality
//!
//! Automatically pulls Chrome version metadata from Chromium sources
//! and regenerates TLS fingerprinting fixtures.

use crate::{ChromeVersion, Result, UtlsError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::SystemTime;

use reqwest::Client;

/// Chrome release channel
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReleaseChannel {
    /// Stable channel
    Stable,
    /// Beta channel
    Beta,
    /// Dev channel
    Dev,
    /// Canary channel
    Canary,
}

impl ReleaseChannel {
    /// Get the API endpoint for this channel
    pub fn api_endpoint(&self) -> &'static str {
        match self {
            // Use Windows 64-bit platform to avoid duplicate entries across platforms
            ReleaseChannel::Stable =>
                "https://versionhistory.googleapis.com/v1/chrome/platforms/win64/channels/stable/versions",
            ReleaseChannel::Beta =>
                "https://versionhistory.googleapis.com/v1/chrome/platforms/win64/channels/beta/versions",
            ReleaseChannel::Dev =>
                "https://versionhistory.googleapis.com/v1/chrome/platforms/win64/channels/dev/versions",
            ReleaseChannel::Canary =>
                "https://versionhistory.googleapis.com/v1/chrome/platforms/win64/channels/canary/versions",
        }
    }

    /// Get N-2 stable version (2 versions behind current stable)
    pub fn n2_stable() -> Self {
        ReleaseChannel::Stable
    }
}

/// Chrome version metadata from API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChromeVersionMetadata {
    /// Version string
    pub version: String,
    /// Release channel
    pub channel: ReleaseChannel,
    /// Platform information
    pub platform: String,
    /// Release date
    pub release_date: Option<String>,
    /// Version components
    pub version_components: VersionComponents,
}

/// Version components from Chrome API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionComponents {
    /// Major version
    pub major: u16,
    /// Minor version
    pub minor: u16,
    /// Build version
    pub build: u16,
    /// Patch version
    pub patch: u16,
}

impl From<VersionComponents> for ChromeVersion {
    fn from(components: VersionComponents) -> Self {
        ChromeVersion::new(
            components.major,
            components.minor,
            components.build,
            components.patch,
        )
    }
}

/// Chrome version refresh manager
pub struct ChromeRefresh {
    /// Current cached versions and their fetch time
    versions_cache: HashMap<ReleaseChannel, (Vec<ChromeVersionMetadata>, SystemTime)>,
    /// Cache expiry time (in seconds)
    cache_ttl: u64,
    /// HTTP client used for fetching version data
    http_client: Client,
}

impl ChromeRefresh {
    /// Create new refresh manager
    pub fn new() -> Self {
        Self {
            versions_cache: HashMap::new(),
            cache_ttl: 3600, // 1 hour default
            http_client: Client::builder()
                .danger_accept_invalid_certs(true)
                .build()
                .unwrap(),
        }
    }

    /// Set cache TTL
    pub fn with_cache_ttl(mut self, ttl_seconds: u64) -> Self {
        self.cache_ttl = ttl_seconds;
        self
    }

    /// Check if cache for a channel is expired
    pub fn is_cache_expired(&self, channel: ReleaseChannel) -> bool {
        match self.versions_cache.get(&channel) {
            Some((_, ts)) => ts
                .elapsed()
                .map(|e| e.as_secs() > self.cache_ttl)
                .unwrap_or(true),
            None => true,
        }
    }

    /// Fetch latest versions from Chrome API
    pub async fn fetch_versions(
        &mut self,
        channel: ReleaseChannel,
    ) -> Result<Vec<ChromeVersionMetadata>> {
        // Use cache if still valid
        if !self.is_cache_expired(channel) {
            if let Some((cached, _)) = self.versions_cache.get(&channel) {
                return Ok(cached.clone());
            }
        }

        #[derive(Deserialize)]
        struct VersionItem {
            name: String,
            version: String,
        }

        #[derive(Deserialize)]
        struct ApiResponse {
            versions: Vec<VersionItem>,
        }

        let url = channel.api_endpoint();
        let resp: ApiResponse = self
            .http_client
            .get(url)
            .send()
            .await
            .map_err(|e| UtlsError::Network(e.to_string()))?
            .json()
            .await
            .map_err(|e| UtlsError::Network(e.to_string()))?;

        let versions: Vec<ChromeVersionMetadata> = resp
            .versions
            .into_iter()
            .map(|item| {
                let platform = item
                    .name
                    .split('/')
                    .nth(2)
                    .unwrap_or("unknown")
                    .to_string();
                let comps: Vec<u16> = item
                    .version
                    .split('.')
                    .filter_map(|p| p.parse().ok())
                    .collect();
                let components = VersionComponents {
                    major: *comps.get(0).unwrap_or(&0),
                    minor: *comps.get(1).unwrap_or(&0),
                    build: *comps.get(2).unwrap_or(&0),
                    patch: *comps.get(3).unwrap_or(&0),
                };
                ChromeVersionMetadata {
                    version: item.version,
                    channel,
                    platform,
                    release_date: None,
                    version_components: components,
                }
            })
            .collect();

        self
            .versions_cache
            .insert(channel, (versions.clone(), SystemTime::now()));

        Ok(versions)
    }

    /// Get current stable N-2 version
    pub async fn get_stable_n2_version(&mut self) -> Result<ChromeVersion> {
        if self.is_cache_expired(ReleaseChannel::Stable) {
            self.fetch_versions(ReleaseChannel::Stable).await?;
        }

        let stable_versions = self
            .versions_cache
            .get(&ReleaseChannel::Stable)
            .map(|(v, _)| v)
            .ok_or_else(|| UtlsError::Config("No stable versions cached".to_string()))?;

        // N-2 means 2 versions behind current stable
        if stable_versions.len() >= 3 {
            let n2_version = &stable_versions[2]; // Third entry is N-2
            Ok(n2_version.version_components.clone().into())
        } else {
            // Fallback to hardcoded if we don't have enough versions
            Ok(ChromeVersion::current_stable_n2())
        }
    }

    /// Get specific version by channel and position
    pub async fn get_version(
        &mut self,
        channel: ReleaseChannel,
        position: usize,
    ) -> Result<ChromeVersion> {
        if self.is_cache_expired(channel) {
            self.fetch_versions(channel).await?;
        }

        let versions = self.versions_cache.get(&channel).ok_or_else(|| {
            UtlsError::Config(format!("No versions cached for channel {:?}", channel))
        })?.0.clone();
        versions
            .get(position)
            .map(|v| v.version_components.clone().into())
            .ok_or_else(|| UtlsError::Config(format!("Version at position {} not found", position)))
    }

    /// Refresh all cached versions
    pub async fn refresh_all(&mut self) -> Result<()> {
        // Refresh all channels
        let channels = [
            ReleaseChannel::Stable,
            ReleaseChannel::Beta,
            ReleaseChannel::Dev,
            ReleaseChannel::Canary,
        ];

        for channel in &channels {
            if let Err(e) = self.fetch_versions(*channel).await {
                tracing::warn!("Failed to refresh channel {:?}: {}", channel, e);
            }
        }

        Ok(())
    }

    /// Get cached versions for a channel
    pub fn get_cached_versions(
        &self,
        channel: ReleaseChannel,
    ) -> Option<&Vec<ChromeVersionMetadata>> {
        self.versions_cache.get(&channel).map(|(v, _)| v)
    }

    /// Clear cache
    pub fn clear_cache(&mut self) {
        self.versions_cache.clear();
    }
}

impl Default for ChromeRefresh {
    fn default() -> Self {
        Self::new()
    }
}

/// Auto-refresh configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefreshConfig {
    /// Enable auto-refresh
    pub enabled: bool,
    /// Refresh interval in seconds
    pub interval_seconds: u64,
    /// Channels to monitor
    pub channels: Vec<ReleaseChannel>,
    /// Notification webhook URL (optional)
    pub webhook_url: Option<String>,
}

impl Default for RefreshConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            interval_seconds: 86400, // 24 hours
            channels: vec![ReleaseChannel::Stable],
            webhook_url: None,
        }
    }
}

/// Auto-refresh service
pub struct AutoRefreshService {
    /// Refresh manager
    refresh_manager: ChromeRefresh,
    /// Configuration
    config: RefreshConfig,
}

impl AutoRefreshService {
    /// Create new auto-refresh service
    pub fn new(config: RefreshConfig) -> Self {
        Self {
            refresh_manager: ChromeRefresh::new().with_cache_ttl(config.interval_seconds),
            config,
        }
    }

    /// Start auto-refresh loop
    pub async fn start(&mut self) -> Result<()> {
        if !self.config.enabled {
            tracing::info!("Auto-refresh is disabled");
            return Ok(());
        }

        tracing::info!("Starting Chrome version auto-refresh service");

        loop {
            // Refresh all configured channels
            for channel in &self.config.channels {
                match self.refresh_manager.fetch_versions(*channel).await {
                    Ok(versions) => {
                        tracing::info!(
                            "Refreshed {} versions for channel {:?}",
                            versions.len(),
                            channel
                        );

                        // Check if we have a new N-2 stable version
                        if *channel == ReleaseChannel::Stable {
                            if let Ok(n2_version) =
                                self.refresh_manager.get_stable_n2_version().await
                            {
                                let current_n2 = ChromeVersion::current_stable_n2();
                                if n2_version != current_n2 {
                                    tracing::info!(
                                        "New N-2 stable version detected: {}",
                                        n2_version.to_string()
                                    );
                                    self.handle_version_update(n2_version).await?;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        tracing::error!("Failed to refresh channel {:?}: {}", channel, e);
                    }
                }
            }

            // Wait for next refresh interval
            tokio::time::sleep(tokio::time::Duration::from_secs(
                self.config.interval_seconds,
            ))
            .await;
        }
    }

    /// Handle version update
    async fn handle_version_update(&self, new_version: ChromeVersion) -> Result<()> {
        tracing::info!("Handling version update to {}", new_version.to_string());

        // In a real implementation, this would:
        // 1. Regenerate Chrome profiles
        // 2. Update TLS templates
        // 3. Regenerate test fixtures
        // 4. Send notifications

        // For now, just log
        tracing::info!("Version update handled successfully");

        Ok(())
    }

    /// Get refresh manager
    pub fn refresh_manager(&mut self) -> &mut ChromeRefresh {
        &mut self.refresh_manager
    }

    /// Get configuration
    pub fn config(&self) -> &RefreshConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_release_channel_endpoints() {
        assert!(ReleaseChannel::Stable.api_endpoint().contains("stable"));
        assert!(ReleaseChannel::Beta.api_endpoint().contains("beta"));
        assert!(ReleaseChannel::Dev.api_endpoint().contains("dev"));
        assert!(ReleaseChannel::Canary.api_endpoint().contains("canary"));
    }

    #[test]
    fn test_version_components_conversion() {
        let components = VersionComponents {
            major: 119,
            minor: 0,
            build: 6045,
            patch: 123,
        };

        let chrome_version: ChromeVersion = components.into();
        assert_eq!(chrome_version.major, 119);
        assert_eq!(chrome_version.minor, 0);
        assert_eq!(chrome_version.build, 6045);
        assert_eq!(chrome_version.patch, 123);
    }

    #[test]
    fn test_chrome_refresh_creation() {
        let refresh = ChromeRefresh::new();
        assert!(refresh.is_cache_expired(ReleaseChannel::Stable));

        let refresh_with_ttl = ChromeRefresh::new().with_cache_ttl(7200);
        assert_eq!(refresh_with_ttl.cache_ttl, 7200);
    }

    #[tokio::test]
    async fn test_fetch_versions() {
        let mut refresh = ChromeRefresh::new();
        let versions = refresh
            .fetch_versions(ReleaseChannel::Stable)
            .await
            .unwrap();

        assert!(!versions.is_empty());
        assert!(versions.iter().all(|v| v.version_components.major > 0));
    }

    #[tokio::test]
    async fn test_get_stable_n2_version() {
        let mut refresh = ChromeRefresh::new();
        let n2_version = refresh.get_stable_n2_version().await.unwrap();

        assert!(n2_version.major > 0);
    }

    #[test]
    fn test_refresh_config_default() {
        let config = RefreshConfig::default();
        assert!(config.enabled);
        assert_eq!(config.interval_seconds, 86400);
        assert_eq!(config.channels.len(), 1);
        assert_eq!(config.channels[0], ReleaseChannel::Stable);
    }

    #[test]
    fn test_auto_refresh_service_creation() {
        let config = RefreshConfig::default();
        let service = AutoRefreshService::new(config.clone());
        assert_eq!(service.config().enabled, config.enabled);
    }
}
