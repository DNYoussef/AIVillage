"""
Configuration for WhatsApp Wave Bridge
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration settings for WhatsApp Wave Bridge"""
    
    # Twilio Configuration
    TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
    TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
    TWILIO_WHATSAPP_NUMBER = os.getenv('TWILIO_WHATSAPP_NUMBER', 'whatsapp:+14155238886')
    
    # AI Model Configuration
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    
    # W&B Configuration
    WANDB_API_KEY = os.getenv('WANDB_API_KEY')
    WANDB_PROJECT = os.getenv('WANDB_PROJECT', 'aivillage-tutoring')
    WANDB_ENTITY = os.getenv('WANDB_ENTITY')
    
    # Performance Settings
    RESPONSE_TIME_TARGET = float(os.getenv('RESPONSE_TIME_TARGET', '5.0'))
    MAX_RESPONSE_LENGTH = int(os.getenv('MAX_RESPONSE_LENGTH', '1000'))
    CACHE_TTL = int(os.getenv('CACHE_TTL', '3600'))  # 1 hour
    
    # Language Settings
    DEFAULT_LANGUAGE = os.getenv('DEFAULT_LANGUAGE', 'en')
    ENABLE_AUTO_TRANSLATION = os.getenv('ENABLE_AUTO_TRANSLATION', 'true').lower() == 'true'
    
    # A/B Testing Settings
    ENABLE_AB_TESTING = os.getenv('ENABLE_AB_TESTING', 'true').lower() == 'true'
    AB_TEST_SAMPLE_SIZE = int(os.getenv('AB_TEST_SAMPLE_SIZE', '100'))
    
    # Security Settings
    WEBHOOK_SECRET = os.getenv('WEBHOOK_SECRET')
    RATE_LIMIT_PER_MINUTE = int(os.getenv('RATE_LIMIT_PER_MINUTE', '30'))
    
    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    ENABLE_DEBUG_MODE = os.getenv('ENABLE_DEBUG_MODE', 'false').lower() == 'true'
    
    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """Validate required configuration"""
        
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Required settings
        required_settings = [
            ('TWILIO_ACCOUNT_SID', cls.TWILIO_ACCOUNT_SID),
            ('TWILIO_AUTH_TOKEN', cls.TWILIO_AUTH_TOKEN),
            ('WANDB_API_KEY', cls.WANDB_API_KEY)
        ]
        
        for setting_name, setting_value in required_settings:
            if not setting_value:
                validation_results['errors'].append(f"Missing required setting: {setting_name}")
                validation_results['valid'] = False
        
        # AI Model settings (at least one required)
        if not cls.ANTHROPIC_API_KEY and not cls.OPENAI_API_KEY:
            validation_results['errors'].append("At least one AI model API key required (ANTHROPIC_API_KEY or OPENAI_API_KEY)")
            validation_results['valid'] = False
        
        # Warnings for optional but recommended settings
        if not cls.ANTHROPIC_API_KEY:
            validation_results['warnings'].append("ANTHROPIC_API_KEY not set - will use OpenAI as primary model")
        
        if not cls.OPENAI_API_KEY:
            validation_results['warnings'].append("OPENAI_API_KEY not set - no fallback model available")
        
        if not cls.WEBHOOK_SECRET:
            validation_results['warnings'].append("WEBHOOK_SECRET not set - webhooks will not be validated")
        
        return validation_results

# Global config instance
config = Config()

# Validate configuration on import
config_validation = config.validate_config()
if not config_validation['valid']:
    raise ValueError(f"Configuration validation failed: {config_validation['errors']}")

if config_validation['warnings']:
    import logging
    logger = logging.getLogger(__name__)
    for warning in config_validation['warnings']:
        logger.warning(f"Configuration warning: {warning}")

__all__ = ['config', 'Config']