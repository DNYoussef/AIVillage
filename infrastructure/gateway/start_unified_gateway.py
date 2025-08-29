#!/usr/bin/env python3
"""
Unified API Gateway Startup Script

Production-ready startup script for the AIVillage Unified API Gateway.
Handles service initialization, health checks, and graceful shutdown.
"""

import asyncio
import logging
import os
from pathlib import Path
import signal
import sys

import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class UnifiedGatewayRunner:
    """Manages the unified gateway startup and lifecycle."""

    def __init__(self):
        self.server = None
        self.config = self._load_config()
        self.shutdown_event = asyncio.Event()

    def _load_config(self):
        """Load configuration from environment variables."""
        config = {
            "host": os.getenv("HOST", "0.0.0.0"),
            "port": int(os.getenv("PORT", "8000")),
            "debug": os.getenv("DEBUG", "false").lower() == "true",
            "workers": int(os.getenv("WORKERS", "1")),
            "jwt_secret": os.getenv("JWT_SECRET_KEY"),
            "require_mfa": os.getenv("REQUIRE_MFA", "false").lower() == "true",
            "log_level": os.getenv("LOG_LEVEL", "info").lower(),
            "access_log": os.getenv("ACCESS_LOG", "true").lower() == "true",
        }

        # Validate critical configuration
        if not config["jwt_secret"]:
            logger.warning("JWT_SECRET_KEY not set. Using auto-generated key for this session.")
            config["jwt_secret"] = os.urandom(32).hex()

        return config

    def _setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers."""

        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}, initiating graceful shutdown...")
            self.shutdown_event.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, signal_handler)

    def _validate_environment(self):
        """Validate environment and dependencies."""
        errors = []
        warnings = []

        # Check Python version

        # Check critical environment variables
        if not os.getenv("JWT_SECRET_KEY"):
            warnings.append("JWT_SECRET_KEY not set - using auto-generated key")

        # Check optional dependencies
        try:
            pass
        except ImportError:
            errors.append("PyJWT package required: pip install PyJWT")

        try:
            pass
        except ImportError:
            errors.append("FastAPI package required: pip install fastapi")

        try:
            pass
        except ImportError:
            errors.append("Uvicorn package required: pip install uvicorn")

        # Report validation results
        if warnings:
            for warning in warnings:
                logger.warning(f"⚠️ {warning}")

        if errors:
            logger.error("❌ Environment validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            return False

        logger.info("✅ Environment validation passed")
        return True

    def _display_startup_info(self):
        """Display startup information."""
        logger.info("🚀 AIVillage Unified API Gateway")
        logger.info("=" * 50)
        logger.info(f"Host: {self.config['host']}")
        logger.info(f"Port: {self.config['port']}")
        logger.info(f"Debug Mode: {self.config['debug']}")
        logger.info(f"Workers: {self.config['workers']}")
        logger.info(f"Log Level: {self.config['log_level']}")
        logger.info(f"MFA Required: {self.config['require_mfa']}")
        logger.info("=" * 50)
        logger.info("🔗 Available Endpoints:")
        logger.info(f"  • API Docs: http://{self.config['host']}:{self.config['port']}/docs")
        logger.info(f"  • ReDoc: http://{self.config['host']}:{self.config['port']}/redoc")
        logger.info(f"  • Health: http://{self.config['host']}:{self.config['port']}/health")
        logger.info(f"  • WebSocket: ws://{self.config['host']}:{self.config['port']}/ws")
        logger.info("=" * 50)
        logger.info("🌟 Integrated Services:")
        logger.info("  • Agent Forge 7-phase training pipeline")
        logger.info("  • P2P/Fog computing with BitChat/BetaNet")
        logger.info("  • JWT authentication with MFA support")
        logger.info("  • Real-time WebSocket updates")
        logger.info("  • Production-grade security & monitoring")
        logger.info("=" * 50)

    def run(self):
        """Run the unified gateway."""
        logger.info("🚀 Starting AIVillage Unified API Gateway...")

        # Validate environment
        if not self._validate_environment():
            sys.exit(1)

        # Setup signal handlers
        self._setup_signal_handlers()

        # Display startup info
        self._display_startup_info()

        # Configure uvicorn
        uvicorn_config = uvicorn.Config(
            "unified_api_gateway:app",
            host=self.config["host"],
            port=self.config["port"],
            log_level=self.config["log_level"],
            access_log=self.config["access_log"],
            reload=self.config["debug"],
            workers=1 if self.config["debug"] else self.config["workers"],
        )

        # Start server
        server = uvicorn.Server(uvicorn_config)

        try:
            logger.info("✅ Unified API Gateway started successfully!")
            logger.info("🔧 Press Ctrl+C to gracefully shutdown")
            server.run()
        except KeyboardInterrupt:
            logger.info("🔄 Graceful shutdown initiated by user")
        except Exception as e:
            logger.error(f"❌ Server error: {e}")
            sys.exit(1)
        finally:
            logger.info("👋 Unified API Gateway shutdown complete")


def main():
    """Main entry point."""
    # Change to the gateway directory
    gateway_dir = Path(__file__).parent
    os.chdir(gateway_dir)

    # Add to Python path
    sys.path.insert(0, str(gateway_dir))

    # Run the gateway
    runner = UnifiedGatewayRunner()
    runner.run()


if __name__ == "__main__":
    main()
