"""Cost management constants for AIVillage.

This module centralizes all cost management and resource optimization
magic literals to eliminate connascence of meaning.
"""

from enum import Enum
from typing import Final

# Cost tracking intervals
COST_UPDATE_INTERVAL_MINUTES: Final[int] = 5
COST_AGGREGATION_INTERVAL_HOURS: Final[int] = 1
COST_REPORT_GENERATION_INTERVAL_HOURS: Final[int] = 24
COST_ALERT_CHECK_INTERVAL_MINUTES: Final[int] = 15

# Budget and billing
DEFAULT_MONTHLY_BUDGET_USD: Final[float] = 1000.0
BUDGET_WARNING_THRESHOLD_PERCENT: Final[int] = 80
BUDGET_CRITICAL_THRESHOLD_PERCENT: Final[int] = 95
COST_ALLOCATION_PRECISION_DIGITS: Final[int] = 4
BILLING_CYCLE_DAYS: Final[int] = 30

# Cloud provider constants
AWS_COST_API_RETRY_ATTEMPTS: Final[int] = 3
GCP_BILLING_API_TIMEOUT_SECONDS: Final[int] = 30
AZURE_COST_MANAGEMENT_DELAY_HOURS: Final[int] = 24
CLOUD_COST_SYNC_BATCH_SIZE: Final[int] = 1000

# Resource optimization
CPU_UTILIZATION_TARGET_PERCENT: Final[int] = 70
MEMORY_UTILIZATION_TARGET_PERCENT: Final[int] = 75
STORAGE_UTILIZATION_TARGET_PERCENT: Final[int] = 80
NETWORK_BANDWIDTH_TARGET_PERCENT: Final[int] = 60

# P2P transport optimization
P2P_COST_PER_GB_TRANSFER: Final[float] = 0.01
P2P_LATENCY_COST_MULTIPLIER: Final[float] = 1.5
P2P_BANDWIDTH_OPTIMIZATION_INTERVAL_MINUTES: Final[int] = 10
P2P_ROUTE_COST_CACHE_TTL_SECONDS: Final[int] = 300

# Edge computing costs
EDGE_NODE_HOURLY_COST_USD: Final[float] = 0.05
EDGE_STORAGE_COST_PER_GB_MONTH: Final[float] = 0.10
EDGE_COMPUTE_COST_MULTIPLIER: Final[float] = 1.2
EDGE_BANDWIDTH_COST_PER_GB: Final[float] = 0.05

# Cost tagging and allocation
MAX_COST_TAGS: Final[int] = 20
COST_TAG_KEY_MAX_LENGTH: Final[int] = 128
COST_TAG_VALUE_MAX_LENGTH: Final[int] = 256
UNALLOCATED_COST_THRESHOLD_PERCENT: Final[int] = 5

# Monitoring and alerts
COST_SPIKE_DETECTION_THRESHOLD_PERCENT: Final[int] = 50
COST_ANOMALY_DETECTION_WINDOW_HOURS: Final[int] = 24
COST_TREND_ANALYSIS_WINDOW_DAYS: Final[int] = 30
COST_FORECAST_HORIZON_DAYS: Final[int] = 90

# Governance and compliance
COST_APPROVAL_THRESHOLD_USD: Final[float] = 100.0
COST_AUDIT_RETENTION_MONTHS: Final[int] = 36
COST_REPORT_ARCHIVE_MONTHS: Final[int] = 12
CHARGEBACK_PROCESSING_DAY_OF_MONTH: Final[int] = 1


class CostCategory(Enum):
    """Cost categories for resource allocation."""

    COMPUTE = "compute"
    STORAGE = "storage"
    NETWORK = "network"
    AI_MODELS = "ai_models"
    DATABASES = "databases"
    MONITORING = "monitoring"
    SECURITY = "security"
    BACKUP = "backup"
    DEVELOPMENT = "development"
    OPERATIONS = "operations"


class BillingModel(Enum):
    """Billing models for cost calculation."""

    PAY_AS_YOU_GO = "pay_as_you_go"
    RESERVED_CAPACITY = "reserved_capacity"
    SPOT_PRICING = "spot_pricing"
    COMMITTED_USE = "committed_use"
    SUBSCRIPTION = "subscription"


class CostOptimizationStrategy(Enum):
    """Cost optimization strategies."""

    RIGHTSIZING = "rightsizing"
    AUTOSCALING = "autoscaling"
    SCHEDULING = "scheduling"
    SPOT_INSTANCES = "spot_instances"
    RESERVED_INSTANCES = "reserved_instances"
    COMPRESSION = "compression"
    CACHING = "caching"
    LOAD_BALANCING = "load_balancing"


class AlertSeverity(Enum):
    """Cost alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


# Dashboard configuration
DASHBOARD_REFRESH_INTERVAL_SECONDS: Final[int] = 30
DASHBOARD_CHART_DATA_POINTS: Final[int] = 100
DASHBOARD_COST_BREAKDOWN_LIMIT: Final[int] = 10
DASHBOARD_TREND_DISPLAY_DAYS: Final[int] = 7

# Cost calculation constants
CURRENCY_PRECISION_DECIMAL_PLACES: Final[int] = 2
PERCENTAGE_PRECISION_DECIMAL_PLACES: Final[int] = 1
COST_ROUNDING_METHOD: Final[str] = "ROUND_HALF_UP"
EXCHANGE_RATE_UPDATE_HOURS: Final[int] = 6

# Resource pricing models
COMPUTE_PRICING_MODEL: Final[str] = "per_hour"
STORAGE_PRICING_MODEL: Final[str] = "per_gb_month"
NETWORK_PRICING_MODEL: Final[str] = "per_gb_transfer"
AI_MODEL_PRICING_MODEL: Final[str] = "per_api_call"

# Cost allocation rules
SHARED_COST_ALLOCATION_METHOD: Final[str] = "proportional"
OVERHEAD_COST_PERCENTAGE: Final[int] = 10
DEPARTMENT_COST_ALLOCATION_ENABLED: Final[bool] = True
PROJECT_COST_TRACKING_ENABLED: Final[bool] = True


class CostMessages:
    """Standardized cost management messages."""

    BUDGET_WARNING: Final[str] = "Budget warning: {percent}% of monthly budget used"
    BUDGET_EXCEEDED: Final[str] = "Budget exceeded: {amount} over {budget} monthly limit"
    COST_SPIKE_DETECTED: Final[str] = "Cost spike detected: {percent}% increase in {category}"
    OPTIMIZATION_SUGGESTION: Final[str] = "Cost optimization suggestion: {suggestion} could save {amount}"
    RESOURCE_UNDERUTILIZED: Final[str] = "Resource {resource_id} underutilized: {utilization}%"
    COST_ALLOCATION_UPDATED: Final[str] = "Cost allocation updated for {entity}: {amount}"
    BILLING_PERIOD_ENDED: Final[str] = "Billing period ended. Total cost: {amount}"
    FORECAST_GENERATED: Final[str] = "Cost forecast generated: projected {amount} for next {period}"


# Performance monitoring
COST_CALCULATION_TIMEOUT_SECONDS: Final[int] = 60
COST_DATA_PROCESSING_BATCH_SIZE: Final[int] = 5000
COST_REPORT_GENERATION_TIMEOUT_MINUTES: Final[int] = 30
COST_API_RATE_LIMIT_PER_MINUTE: Final[int] = 100

# Data retention
DETAILED_COST_DATA_RETENTION_DAYS: Final[int] = 90
AGGREGATED_COST_DATA_RETENTION_MONTHS: Final[int] = 24
COST_AUDIT_TRAIL_RETENTION_YEARS: Final[int] = 7
COST_FORECAST_RETENTION_MONTHS: Final[int] = 6

# Integration constants
EXTERNAL_BILLING_API_TIMEOUT_SECONDS: Final[int] = 45
COST_DATA_EXPORT_FORMAT: Final[str] = "JSON"
COST_WEBHOOK_TIMEOUT_SECONDS: Final[int] = 10
COST_NOTIFICATION_BATCH_SIZE: Final[int] = 50

# Distributed cost tracking
DISTRIBUTED_COST_SYNC_INTERVAL_MINUTES: Final[int] = 15
COST_CONSENSUS_TIMEOUT_SECONDS: Final[int] = 30
COST_REPLICATION_FACTOR: Final[int] = 3
COST_PARTITION_SIZE_HOURS: Final[int] = 4

# Multi-cloud cost management
MULTI_CLOUD_COST_NORMALIZATION: Final[bool] = True
CLOUD_PROVIDER_COST_COMPARISON: Final[bool] = True
CROSS_CLOUD_TRANSFER_COST_TRACKING: Final[bool] = True
HYBRID_CLOUD_COST_ALLOCATION: Final[bool] = True
