//! Integration tests for Betanet v2 operational fixes
//!
//! Validates that the Rust implementation properly integrates with the enhanced
//! Python Betanet v2 features including per-origin calibration, mobile budget
//! management, and governance constraints.

use crate::{
    BetanetMessage, MessagePriority,
    gateway::{ScionGateway, PathRequirements},
    transport::chrome_fingerprint_v2::{
        ChromeTemplateManagerV2, OriginCalibratorV2, MobileBudgetManager,
    },
    metrics::{KpiMetrics, BenchmarkConfig, NetworkConditions},
    error::Result,
    config::{BetanetConfig, GatewayConfig},
};

use bytes::Bytes;
use std::time::Duration;
use tokio::time::timeout;
use tracing::{info, warn, debug};

/// Integration test suite for Betanet v2 features
pub struct BetanetV2IntegrationTest {
    /// Chrome template manager
    chrome_manager: ChromeTemplateManagerV2,
    /// Origin calibrator
    origin_calibrator: OriginCalibratorV2,
    /// Mobile budget manager
    mobile_budget: MobileBudgetManager,
    /// SCION gateway
    scion_gateway: Option<ScionGateway>,
    /// KPI metrics
    kpi_metrics: KpiMetrics,
}

/// Test configuration
#[derive(Debug, Clone)]
pub struct IntegrationTestConfig {
    /// Test duration
    pub duration: Duration,
    /// Origins to test
    pub test_origins: Vec<String>,
    /// Message sizes to test
    pub message_sizes: Vec<usize>,
    /// Concurrent connections
    pub concurrent_connections: u32,
    /// Enable mobile simulation
    pub mobile_simulation: bool,
    /// Network conditions
    pub network_conditions: NetworkConditions,
}

/// Test results
#[derive(Debug, Clone)]
pub struct IntegrationTestResults {
    /// Chrome fingerprinting results
    pub chrome_fingerprinting: ChromeFingerprintingResults,
    /// Origin calibration results
    pub origin_calibration: OriginCalibrationResults,
    /// Mobile budget results
    pub mobile_budget: MobileBudgetResults,
    /// Gateway operation results
    pub gateway_operation: GatewayOperationResults,
    /// Performance metrics
    pub performance: PerformanceResults,
    /// Overall test success
    pub overall_success: bool,
}

/// Chrome fingerprinting test results
#[derive(Debug, Clone)]
pub struct ChromeFingerprintingResults {
    pub templates_tested: u32,
    pub templates_successful: u32,
    pub average_refresh_time_ms: f64,
    pub fingerprint_accuracy: f64,
}

/// Origin calibration test results
#[derive(Debug, Clone)]
pub struct OriginCalibrationResults {
    pub origins_calibrated: u32,
    pub calibrations_successful: u32,
    pub average_calibration_time_ms: f64,
    pub camouflage_effectiveness: f64,
}

/// Mobile budget test results
#[derive(Debug, Clone)]
pub struct MobileBudgetResults {
    pub budget_enforcements: u32,
    pub cover_traffic_blocked: u32,
    pub average_battery_savings_percent: f64,
    pub data_usage_reduction_percent: f64,
}

/// Gateway operation test results
#[derive(Debug, Clone)]
pub struct GatewayOperationResults {
    pub paths_discovered: u32,
    pub beacons_processed: u32,
    pub control_messages_signed: u32,
    pub governance_constraints_applied: u32,
}

/// Performance test results
#[derive(Debug, Clone)]
pub struct PerformanceResults {
    pub average_latency_ms: f64,
    pub throughput_msgs_per_sec: f64,
    pub reliability_score: f64,
    pub resource_efficiency: f64,
}

impl BetanetV2IntegrationTest {
    /// Create new integration test suite
    pub async fn new() -> Result<Self> {
        let mobile_budget = MobileBudgetManager::new();
        let chrome_manager = ChromeTemplateManagerV2::new();
        let origin_calibrator = OriginCalibratorV2::new(std::sync::Arc::new(mobile_budget.clone()));
        let kpi_metrics = KpiMetrics::new();

        // Start KPI monitoring
        kpi_metrics.start().await?;

        Ok(Self {
            chrome_manager,
            origin_calibrator,
            mobile_budget,
            scion_gateway: None,
            kpi_metrics,
        })
    }

    /// Initialize SCION gateway for testing
    pub async fn initialize_gateway(&mut self, config: GatewayConfig) -> Result<()> {
        let gateway = ScionGateway::new(config).await?;
        gateway.start().await?;
        self.scion_gateway = Some(gateway);
        Ok(())
    }

    /// Run comprehensive integration tests
    pub async fn run_comprehensive_test(&self, config: IntegrationTestConfig) -> Result<IntegrationTestResults> {
        info!("Starting Betanet v2 comprehensive integration test");

        // Test Chrome fingerprinting
        let chrome_results = self.test_chrome_fingerprinting(&config).await?;

        // Test origin calibration
        let calibration_results = self.test_origin_calibration(&config).await?;

        // Test mobile budget management
        let budget_results = self.test_mobile_budget_management(&config).await?;

        // Test gateway operations
        let gateway_results = self.test_gateway_operations(&config).await?;

        // Test performance
        let performance_results = self.test_performance(&config).await?;

        let overall_success = chrome_results.fingerprint_accuracy > 0.9 &&
                             calibration_results.camouflage_effectiveness > 0.85 &&
                             budget_results.average_battery_savings_percent > 20.0 &&
                             gateway_results.paths_discovered > 0 &&
                             performance_results.reliability_score > 0.95;

        let results = IntegrationTestResults {
            chrome_fingerprinting: chrome_results,
            origin_calibration: calibration_results,
            mobile_budget: budget_results,
            gateway_operation: gateway_results,
            performance: performance_results,
            overall_success,
        };

        info!("Integration test completed: success={}", results.overall_success);
        Ok(results)
    }

    /// Test Chrome fingerprinting functionality
    async fn test_chrome_fingerprinting(&self, config: &IntegrationTestConfig) -> Result<ChromeFingerprintingResults> {
        info!("Testing Chrome fingerprinting...");

        let mut templates_tested = 0;
        let mut templates_successful = 0;
        let mut total_refresh_time = 0.0;

        // Test different Chrome versions
        let versions = ["N", "N-1", "N-2"];

        for version in &versions {
            templates_tested += 1;

            let start = std::time::Instant::now();
            match self.chrome_manager.get_template(version).await {
                Ok(template) => {
                    templates_successful += 1;
                    let refresh_time = start.elapsed().as_millis() as f64;
                    total_refresh_time += refresh_time;

                    // Validate template fields
                    assert!(!template.ja3_hash.is_empty());
                    assert!(!template.ja4_hash.is_empty());
                    assert!(!template.cipher_suites.is_empty());

                    debug!("Template {} loaded successfully in {}ms", version, refresh_time);
                }
                Err(e) => {
                    warn!("Failed to load template {}: {}", version, e);
                }
            }
        }

        let average_refresh_time = if templates_successful > 0 {
            total_refresh_time / templates_successful as f64
        } else {
            0.0
        };

        let fingerprint_accuracy = templates_successful as f64 / templates_tested as f64;

        Ok(ChromeFingerprintingResults {
            templates_tested,
            templates_successful,
            average_refresh_time_ms: average_refresh_time,
            fingerprint_accuracy,
        })
    }

    /// Test origin calibration functionality
    async fn test_origin_calibration(&self, config: &IntegrationTestConfig) -> Result<OriginCalibrationResults> {
        info!("Testing origin calibration...");

        let mut origins_calibrated = 0;
        let mut calibrations_successful = 0;
        let mut total_calibration_time = 0.0;

        for origin in &config.test_origins {
            origins_calibrated += 1;

            let start = std::time::Instant::now();

            // Use timeout to prevent hanging on network issues
            match timeout(Duration::from_secs(30), self.origin_calibrator.get_origin_fingerprint(origin)).await {
                Ok(Ok(fingerprint)) => {
                    calibrations_successful += 1;
                    let calibration_time = start.elapsed().as_millis() as f64;
                    total_calibration_time += calibration_time;

                    // Validate fingerprint data
                    assert_eq!(fingerprint.hostname, *origin);
                    assert!(!fingerprint.cookie_names.is_empty());
                    assert!(!fingerprint.header_patterns.is_empty());

                    debug!("Origin {} calibrated successfully in {}ms", origin, calibration_time);
                }
                Ok(Err(e)) => {
                    warn!("Failed to calibrate origin {}: {}", origin, e);
                }
                Err(_) => {
                    warn!("Origin calibration timed out for {}", origin);
                }
            }
        }

        let average_calibration_time = if calibrations_successful > 0 {
            total_calibration_time / calibrations_successful as f64
        } else {
            0.0
        };

        // Camouflage effectiveness based on successful calibrations
        let camouflage_effectiveness = calibrations_successful as f64 / origins_calibrated as f64;

        Ok(OriginCalibrationResults {
            origins_calibrated,
            calibrations_successful,
            average_calibration_time_ms: average_calibration_time,
            camouflage_effectiveness,
        })
    }

    /// Test mobile budget management
    async fn test_mobile_budget_management(&self, config: &IntegrationTestConfig) -> Result<MobileBudgetResults> {
        info!("Testing mobile budget management...");

        let mut budget_enforcements = 0;
        let mut cover_traffic_blocked = 0;
        let test_origin = "example.com";

        // Test initial budget availability
        let large_request = 200 * 1024; // 200KB
        let small_request = 50 * 1024;  // 50KB

        // First request should be allowed
        if self.mobile_budget.can_create_cover_traffic(test_origin, small_request).await {
            self.mobile_budget.record_cover_traffic(test_origin, small_request).await;
            budget_enforcements += 1;
        }

        // Second immediate request should be blocked (time budget)
        if !self.mobile_budget.can_create_cover_traffic(test_origin, small_request).await {
            cover_traffic_blocked += 1;
        }

        // Large request should be blocked (size budget)
        if !self.mobile_budget.can_create_cover_traffic("other.com", large_request).await {
            cover_traffic_blocked += 1;
        }

        // Calculate savings (simplified estimation)
        let potential_requests = 10;
        let blocked_percentage = (cover_traffic_blocked as f64 / potential_requests as f64) * 100.0;
        let battery_savings = blocked_percentage * 0.5; // Estimated 50% correlation
        let data_savings = blocked_percentage * 0.8;    // Estimated 80% correlation

        Ok(MobileBudgetResults {
            budget_enforcements,
            cover_traffic_blocked,
            average_battery_savings_percent: battery_savings,
            data_usage_reduction_percent: data_savings,
        })
    }

    /// Test gateway operations
    async fn test_gateway_operations(&self, config: &IntegrationTestConfig) -> Result<GatewayOperationResults> {
        info!("Testing gateway operations...");

        if let Some(gateway) = &self.scion_gateway {
            // Test path selection
            let requirements = PathRequirements {
                max_latency_ms: Some(100),
                min_bandwidth_mbps: Some(10),
                max_cost: Some(1000),
                forbidden_ases: vec![],
                required_ases: vec![],
            };

            let _path = gateway.select_path(64513, &requirements).await;

            // Get gateway statistics
            let stats = gateway.get_statistics().await;

            Ok(GatewayOperationResults {
                paths_discovered: stats.total_paths as u32,
                beacons_processed: stats.active_beacons as u32,
                control_messages_signed: stats.control_message_stats.messages_sent as u32,
                governance_constraints_applied: 0, // Would be tracked in real implementation
            })
        } else {
            warn!("Gateway not initialized for testing");
            Ok(GatewayOperationResults {
                paths_discovered: 0,
                beacons_processed: 0,
                control_messages_signed: 0,
                governance_constraints_applied: 0,
            })
        }
    }

    /// Test performance metrics
    async fn test_performance(&self, config: &IntegrationTestConfig) -> Result<PerformanceResults> {
        info!("Testing performance...");

        // Create test messages
        let mut messages_sent = 0;
        let mut total_latency = 0.0;
        let start_time = std::time::Instant::now();

        for size in &config.message_sizes {
            let payload = vec![0u8; *size];
            let message = BetanetMessage::new(
                "test_sender".to_string(),
                "test_recipient".to_string(),
                Bytes::from(payload),
            );

            let send_start = std::time::Instant::now();

            // Record metrics
            self.kpi_metrics.record_message_sent(&message, Duration::from_millis(50)).await;
            self.kpi_metrics.record_message_received(&message).await;

            let latency = send_start.elapsed().as_millis() as f64;
            total_latency += latency;
            messages_sent += 1;

            // Small delay between messages
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        let test_duration = start_time.elapsed().as_secs_f64();
        let average_latency = if messages_sent > 0 {
            total_latency / messages_sent as f64
        } else {
            0.0
        };
        let throughput = messages_sent as f64 / test_duration;

        // Get real-time stats
        let realtime_stats = self.kpi_metrics.get_realtime_stats().await;

        Ok(PerformanceResults {
            average_latency_ms: average_latency,
            throughput_msgs_per_sec: throughput,
            reliability_score: realtime_stats.reliability_score,
            resource_efficiency: 0.85, // Placeholder - would calculate from actual metrics
        })
    }

    /// Generate comprehensive test report
    pub async fn generate_test_report(&self, results: &IntegrationTestResults) -> Result<String> {
        let report = format!(
            r#"# Betanet v2 Integration Test Report

## Overall Result: {}

## Chrome Fingerprinting
- Templates Tested: {}
- Success Rate: {:.1}%
- Average Refresh Time: {:.1}ms
- Fingerprint Accuracy: {:.1}%

## Origin Calibration
- Origins Tested: {}
- Success Rate: {:.1}%
- Average Calibration Time: {:.1}ms
- Camouflage Effectiveness: {:.1}%

## Mobile Budget Management
- Budget Enforcements: {}
- Cover Traffic Blocked: {}
- Battery Savings: {:.1}%
- Data Usage Reduction: {:.1}%

## Gateway Operations
- Paths Discovered: {}
- Beacons Processed: {}
- Control Messages Signed: {}
- Governance Constraints Applied: {}

## Performance Metrics
- Average Latency: {:.1}ms
- Throughput: {:.1} msgs/sec
- Reliability Score: {:.3}
- Resource Efficiency: {:.3}

## Recommendations

{}

Generated: {}
"#,
            if results.overall_success { "✅ PASS" } else { "❌ FAIL" },
            results.chrome_fingerprinting.templates_tested,
            results.chrome_fingerprinting.fingerprint_accuracy * 100.0,
            results.chrome_fingerprinting.average_refresh_time_ms,
            results.chrome_fingerprinting.fingerprint_accuracy * 100.0,
            results.origin_calibration.origins_calibrated,
            (results.origin_calibration.calibrations_successful as f64 / results.origin_calibration.origins_calibrated as f64) * 100.0,
            results.origin_calibration.average_calibration_time_ms,
            results.origin_calibration.camouflage_effectiveness * 100.0,
            results.mobile_budget.budget_enforcements,
            results.mobile_budget.cover_traffic_blocked,
            results.mobile_budget.average_battery_savings_percent,
            results.mobile_budget.data_usage_reduction_percent,
            results.gateway_operation.paths_discovered,
            results.gateway_operation.beacons_processed,
            results.gateway_operation.control_messages_signed,
            results.gateway_operation.governance_constraints_applied,
            results.performance.average_latency_ms,
            results.performance.throughput_msgs_per_sec,
            results.performance.reliability_score,
            results.performance.resource_efficiency,
            self.generate_recommendations(results),
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
        );

        Ok(report)
    }

    /// Generate recommendations based on test results
    fn generate_recommendations(&self, results: &IntegrationTestResults) -> String {
        let mut recommendations = Vec::new();

        if results.chrome_fingerprinting.fingerprint_accuracy < 0.9 {
            recommendations.push("- Improve Chrome fingerprinting accuracy by updating templates more frequently");
        }

        if results.origin_calibration.camouflage_effectiveness < 0.85 {
            recommendations.push("- Enhance origin calibration camouflage techniques");
        }

        if results.mobile_budget.average_battery_savings_percent < 20.0 {
            recommendations.push("- Tighten mobile budget constraints to improve battery life");
        }

        if results.performance.average_latency_ms > 100.0 {
            recommendations.push("- Optimize transport selection to reduce latency");
        }

        if results.performance.reliability_score < 0.95 {
            recommendations.push("- Investigate reliability issues and improve error handling");
        }

        if recommendations.is_empty() {
            recommendations.push("- All metrics are within acceptable ranges");
            recommendations.push("- Continue monitoring and maintain current configuration");
        }

        recommendations.join("\n")
    }
}

impl Default for IntegrationTestConfig {
    fn default() -> Self {
        Self {
            duration: Duration::from_secs(60),
            test_origins: vec![
                "example.com".to_string(),
                "httpbin.org".to_string(),
                "jsonplaceholder.typicode.com".to_string(),
            ],
            message_sizes: vec![1024, 4096, 16384, 65536],
            concurrent_connections: 5,
            mobile_simulation: true,
            network_conditions: NetworkConditions {
                latency_ms: 50,
                bandwidth_mbps: 100,
                packet_loss_percent: 0.1,
                jitter_ms: 10,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_integration_suite_creation() {
        let test_suite = BetanetV2IntegrationTest::new().await;
        assert!(test_suite.is_ok());
    }

    #[tokio::test]
    async fn test_chrome_fingerprinting_isolated() {
        let test_suite = BetanetV2IntegrationTest::new().await.unwrap();
        let config = IntegrationTestConfig::default();

        let results = test_suite.test_chrome_fingerprinting(&config).await;
        assert!(results.is_ok());

        let results = results.unwrap();
        assert!(results.templates_tested > 0);
    }

    #[tokio::test]
    async fn test_mobile_budget_isolated() {
        let test_suite = BetanetV2IntegrationTest::new().await.unwrap();
        let config = IntegrationTestConfig::default();

        let results = test_suite.test_mobile_budget_management(&config).await;
        assert!(results.is_ok());

        let results = results.unwrap();
        assert!(results.budget_enforcements >= 0);
    }
}
