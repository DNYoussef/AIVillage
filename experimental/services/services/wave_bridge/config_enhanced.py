"""Enhanced Configuration for WhatsApp Wave Bridge with Prompt Engineering
Part B: Agent Forge Phase 4 - Configuration Management
"""

from dataclasses import dataclass
import os
from typing import Any


@dataclass
class PromptEngineeeringConfig:
    """Configuration for prompt engineering features"""

    # Core feature toggles
    enable_enhanced_prompts: bool = True
    enable_real_time_optimization: bool = True
    enable_advanced_ab_testing: bool = True
    enable_prompt_baking: bool = True

    # Performance thresholds
    prompt_optimization_threshold: float = 0.75
    response_time_target: float = 5.0
    min_sample_size_ab_test: int = 30
    confidence_level: float = 0.95

    # W&B configuration
    wandb_project: str = "aivillage-tutoring"
    wandb_entity: str = "aivillage"
    enable_wandb_logging: bool = True

    # Prompt template settings
    max_active_templates: int = 50
    template_cache_ttl: int = 3600  # 1 hour
    auto_template_cleanup: bool = True

    # A/B testing settings
    max_concurrent_tests: int = 10
    test_duration_days: int = 7
    auto_promote_winners: bool = True
    statistical_significance_threshold: float = 0.95

    # Real-time optimization
    optimization_trigger_threshold: float = 0.8
    optimization_cooldown_minutes: int = 30
    max_optimizations_per_hour: int = 10

    # Language and localization
    supported_languages: list = None
    default_language: str = "en"
    enable_auto_translation: bool = True
    translation_cache_ttl: int = 86400  # 24 hours

    # Subject detection
    subject_detection_confidence: float = 0.6
    enable_subject_specialization: bool = True
    default_subject: str = "general"

    def __post_init__(self):
        if self.supported_languages is None:
            self.supported_languages = [
                "en",
                "es",
                "hi",
                "fr",
                "ar",
                "pt",
                "sw",
                "de",
                "it",
                "zh",
            ]


class EnhancedAppConfig:
    """Main configuration class for enhanced WhatsApp Wave Bridge"""

    def __init__(self):
        # Load from environment variables
        self.load_from_env()

        # Initialize prompt engineering config
        self.prompt_engineering = PromptEngineeeringConfig()

        # Core application settings
        self.app_settings = {
            "title": "WhatsApp Wave Bridge Enhanced",
            "version": "2.0.0",
            "debug": os.getenv("DEBUG", "false").lower() == "true",
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
            "max_workers": int(os.getenv("MAX_WORKERS", "4")),
            "request_timeout": float(os.getenv("REQUEST_TIMEOUT", "30.0")),
        }

        # API settings
        self.api_settings = {
            "host": os.getenv("HOST", "0.0.0.0"),
            "port": int(os.getenv("PORT", "8000")),
            "reload": os.getenv("RELOAD", "false").lower() == "true",
            "access_log": os.getenv("ACCESS_LOG", "true").lower() == "true",
        }

        # Security settings
        self.security_settings = {
            "rate_limit_per_minute": int(os.getenv("RATE_LIMIT_PER_MINUTE", "60")),
            "max_message_length": int(os.getenv("MAX_MESSAGE_LENGTH", "1000")),
            "enable_request_validation": os.getenv(
                "ENABLE_REQUEST_VALIDATION", "true"
            ).lower()
            == "true",
            "allowed_origins": os.getenv("ALLOWED_ORIGINS", "*").split(","),
        }

        # External API settings
        self.external_apis = {
            "twilio": {
                "account_sid": os.getenv("TWILIO_ACCOUNT_SID"),
                "auth_token": os.getenv("TWILIO_AUTH_TOKEN"),
                "whatsapp_number": os.getenv(
                    "TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886"
                ),
            },
            "anthropic": {
                "api_key": os.getenv("ANTHROPIC_API_KEY"),
                "model": os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307"),
                "max_tokens": int(os.getenv("ANTHROPIC_MAX_TOKENS", "1000")),
            },
            "openai": {
                "api_key": os.getenv("OPENAI_API_KEY"),
                "model": os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                "max_tokens": int(os.getenv("OPENAI_MAX_TOKENS", "1000")),
            },
            "wandb": {
                "api_key": os.getenv("WANDB_API_KEY"),
                "project": self.prompt_engineering.wandb_project,
                "entity": self.prompt_engineering.wandb_entity,
            },
        }

        # Performance monitoring
        self.monitoring = {
            "enable_metrics": os.getenv("ENABLE_METRICS", "true").lower() == "true",
            "metrics_port": int(os.getenv("METRICS_PORT", "9090")),
            "health_check_interval": int(os.getenv("HEALTH_CHECK_INTERVAL", "30")),
            "alert_webhook_url": os.getenv("ALERT_WEBHOOK_URL"),
            "enable_performance_alerts": os.getenv(
                "ENABLE_PERFORMANCE_ALERTS", "true"
            ).lower()
            == "true",
        }

        # Caching settings
        self.caching = {
            "enable_response_cache": os.getenv("ENABLE_RESPONSE_CACHE", "true").lower()
            == "true",
            "cache_ttl": int(os.getenv("CACHE_TTL", "300")),  # 5 minutes
            "max_cache_size": int(os.getenv("MAX_CACHE_SIZE", "1000")),
            "cache_backend": os.getenv("CACHE_BACKEND", "memory"),  # memory, redis
        }

    def load_from_env(self):
        """Load configuration from environment variables"""
        # Prompt engineering feature flags
        pe_config = {
            "enable_enhanced_prompts": os.getenv(
                "ENABLE_ENHANCED_PROMPTS", "true"
            ).lower()
            == "true",
            "enable_real_time_optimization": os.getenv(
                "ENABLE_REAL_TIME_OPTIMIZATION", "true"
            ).lower()
            == "true",
            "enable_advanced_ab_testing": os.getenv(
                "ENABLE_ADVANCED_AB_TESTING", "true"
            ).lower()
            == "true",
            "enable_prompt_baking": os.getenv("ENABLE_PROMPT_BAKING", "true").lower()
            == "true",
        }

        # Performance thresholds
        pe_config.update(
            {
                "prompt_optimization_threshold": float(
                    os.getenv("PROMPT_OPTIMIZATION_THRESHOLD", "0.75")
                ),
                "response_time_target": float(os.getenv("RESPONSE_TIME_TARGET", "5.0")),
                "min_sample_size_ab_test": int(
                    os.getenv("MIN_SAMPLE_SIZE_AB_TEST", "30")
                ),
                "confidence_level": float(os.getenv("CONFIDENCE_LEVEL", "0.95")),
            }
        )

        self.prompt_engineering_env = pe_config

    def validate_config(self) -> dict[str, Any]:
        """Validate configuration and return validation results"""
        validation_results = {"valid": True, "errors": [], "warnings": []}

        # Validate required API keys
        required_apis = ["TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN"]
        for api_key in required_apis:
            if not os.getenv(api_key):
                validation_results["errors"].append(
                    f"Missing required environment variable: {api_key}"
                )
                validation_results["valid"] = False

        # Validate AI model API keys (at least one required)
        ai_apis = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"]
        if not any(os.getenv(api_key) for api_key in ai_apis):
            validation_results["errors"].append(
                "At least one AI model API key required (ANTHROPIC_API_KEY or OPENAI_API_KEY)"
            )
            validation_results["valid"] = False

        # Validate W&B API key for prompt engineering
        if self.prompt_engineering.enable_wandb_logging and not os.getenv(
            "WANDB_API_KEY"
        ):
            validation_results["warnings"].append(
                "WANDB_API_KEY not set - prompt engineering tracking will be limited"
            )

        # Validate performance thresholds
        if self.prompt_engineering.response_time_target <= 0:
            validation_results["errors"].append("RESPONSE_TIME_TARGET must be positive")
            validation_results["valid"] = False

        if (
            self.prompt_engineering.prompt_optimization_threshold < 0
            or self.prompt_engineering.prompt_optimization_threshold > 1
        ):
            validation_results["errors"].append(
                "PROMPT_OPTIMIZATION_THRESHOLD must be between 0 and 1"
            )
            validation_results["valid"] = False

        # Validate port settings
        try:
            port = int(os.getenv("PORT", "8000"))
            if port < 1024 or port > 65535:
                validation_results["warnings"].append(
                    f"Port {port} may require special permissions"
                )
        except ValueError:
            validation_results["errors"].append("PORT must be a valid integer")
            validation_results["valid"] = False

        return validation_results

    def get_feature_flags(self) -> dict[str, bool]:
        """Get all feature flags for the application"""
        return {
            "enhanced_prompts": self.prompt_engineering.enable_enhanced_prompts,
            "real_time_optimization": self.prompt_engineering.enable_real_time_optimization,
            "advanced_ab_testing": self.prompt_engineering.enable_advanced_ab_testing,
            "prompt_baking": self.prompt_engineering.enable_prompt_baking,
            "wandb_logging": self.prompt_engineering.enable_wandb_logging,
            "auto_translation": self.prompt_engineering.enable_auto_translation,
            "subject_specialization": self.prompt_engineering.enable_subject_specialization,
            "metrics": self.monitoring["enable_metrics"],
            "performance_alerts": self.monitoring["enable_performance_alerts"],
            "response_cache": self.caching["enable_response_cache"],
            "request_validation": self.security_settings["enable_request_validation"],
        }

    def export_config(self) -> dict[str, Any]:
        """Export complete configuration as dictionary"""
        return {
            "app_settings": self.app_settings,
            "api_settings": self.api_settings,
            "security_settings": self.security_settings,
            "external_apis": {
                # Exclude sensitive keys
                api_name: {
                    k: v
                    for k, v in api_config.items()
                    if "key" not in k.lower() and "token" not in k.lower()
                }
                for api_name, api_config in self.external_apis.items()
            },
            "prompt_engineering": {
                "enable_enhanced_prompts": self.prompt_engineering.enable_enhanced_prompts,
                "enable_real_time_optimization": self.prompt_engineering.enable_real_time_optimization,
                "response_time_target": self.prompt_engineering.response_time_target,
                "supported_languages": self.prompt_engineering.supported_languages,
                "max_active_templates": self.prompt_engineering.max_active_templates,
                "max_concurrent_tests": self.prompt_engineering.max_concurrent_tests,
            },
            "monitoring": self.monitoring,
            "caching": self.caching,
            "feature_flags": self.get_feature_flags(),
        }


# Global configuration instance
config = EnhancedAppConfig()

# Validation on import
validation = config.validate_config()
if not validation["valid"]:
    import warnings

    warnings.warn(f"Configuration validation failed: {validation['errors']}", stacklevel=2)

if validation["warnings"]:
    import warnings

    for warning in validation["warnings"]:
        warnings.warn(warning, stacklevel=2)
