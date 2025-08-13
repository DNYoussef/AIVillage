"""HTTP Security Validator for Production Environment.

This module provides runtime guards to prevent insecure HTTP URLs in production.
"""

import logging
import os
import re
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class HTTPSecurityError(ValueError):
    """Raised when HTTP URLs are used in production environment."""


def validate_url_for_production(url: str, context: str = "") -> str:
    """Validate that URL uses HTTPS in production environment.

    Args:
        url: The URL to validate
        context: Optional context description for error messages

    Returns:
        The validated URL

    Raises:
        HTTPSecurityError: If HTTP URL is used in production
    """
    if not url:
        return url

    if os.getenv("AIVILLAGE_ENV") == "production":
        parsed = urlparse(url)
        if parsed.scheme == "http":
            context_msg = f" in {context}" if context else ""
            msg = f"HTTP URL '{url}' is not allowed in production environment{context_msg}. Use HTTPS instead."
            logger.error(msg)
            raise HTTPSecurityError(msg)

    return url


def validate_config_dict_for_production(config: dict[str, Any], path: str = "") -> None:
    """Recursively validate configuration dictionary for HTTP URLs in production.

    Args:
        config: Configuration dictionary to validate
        path: Current path in the configuration (for error context)

    Raises:
        HTTPSecurityError: If HTTP URLs are found in production
    """
    if not isinstance(config, dict):
        return

    for key, value in config.items():
        current_path = f"{path}.{key}" if path else key

        if isinstance(value, dict):
            validate_config_dict_for_production(value, current_path)
        elif isinstance(value, str) and value.startswith("http://"):
            validate_url_for_production(value, f"config key '{current_path}'")
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, str) and item.startswith("http://"):
                    validate_url_for_production(
                        item, f"config key '{current_path}[{i}]'"
                    )
                elif isinstance(item, dict):
                    validate_config_dict_for_production(item, f"{current_path}[{i}]")


def scan_source_for_http_urls(source_code: str, filename: str = "") -> list[str]:
    """Scan source code for hardcoded HTTP URLs.

    Args:
        source_code: Source code to scan
        filename: Optional filename for context

    Returns:
        List of found HTTP URLs
    """
    # Pattern to find HTTP URLs in strings
    http_pattern = r'"http://[^"]*"|\'http://[^\']*\'|`http://[^`]*`'
    matches = re.findall(http_pattern, source_code)

    # Clean up the quotes
    urls = [match.strip("\"'`") for match in matches]

    if urls and filename:
        logger.warning(f"Found {len(urls)} HTTP URLs in {filename}: {urls}")

    return urls


def validate_production_environment() -> None:
    """Validate that production environment is properly configured for HTTPS.

    Raises:
        HTTPSecurityError: If production environment validation fails
    """
    if os.getenv("AIVILLAGE_ENV") != "production":
        return

    logger.info("Validating production environment for HTTPS compliance...")

    # Check common environment variables for HTTP URLs
    env_vars_to_check = [
        "QDRANT_URL",
        "REDIS_URL",
        "PROMETHEUS_ENDPOINT",
        "JAEGER_ENDPOINT",
        "GATEWAY_URL",
        "TWIN_URL",
        "API_BASE_URL",
    ]

    violations = []
    for var_name in env_vars_to_check:
        value = os.getenv(var_name)
        if value and value.startswith("http://"):
            violations.append(f"{var_name}={value}")

    if violations:
        msg = f"HTTP URLs found in production environment variables: {', '.join(violations)}"
        logger.error(msg)
        raise HTTPSecurityError(msg)

    logger.info("Production environment HTTPS validation passed")


def require_https_in_production(func):
    """Decorator to validate HTTPS URLs in production before function execution.

    This decorator can be applied to functions that accept URL parameters.
    """

    def wrapper(*args, **kwargs):
        if os.getenv("AIVILLAGE_ENV") == "production":
            # Check all string arguments for HTTP URLs
            for arg in args:
                if isinstance(arg, str) and arg.startswith("http://"):
                    validate_url_for_production(arg, f"function {func.__name__}")

            # Check keyword arguments for HTTP URLs
            for key, value in kwargs.items():
                if isinstance(value, str) and value.startswith("http://"):
                    validate_url_for_production(
                        value, f"function {func.__name__} parameter '{key}'"
                    )

        return func(*args, **kwargs)

    return wrapper


# Production startup validation
if __name__ == "__main__":
    try:
        validate_production_environment()
        print("✅ Production HTTPS validation passed")
    except HTTPSecurityError as e:
        print(f"❌ Production HTTPS validation failed: {e}")
        exit(1)
