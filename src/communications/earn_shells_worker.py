#!/usr/bin/env python3
"""Earn Shells Worker - Mints credits based on Prometheus metrics."""

import argparse
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urljoin

try:
    from .credits_ledger import CreditsConfig, CreditsLedger
except ImportError:  # pragma: no cover - allow script execution
    from credits_ledger import CreditsConfig, CreditsLedger  # type: ignore

import requests

# Configure logging
log_path = Path.home() / ".aivillage" / "logs" / "earn_shells_worker.log"
log_path.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path),
    ],
)
logger = logging.getLogger(__name__)


class PrometheusClient:
    """Client for querying Prometheus metrics."""

    def __init__(self, prometheus_url: str) -> None:
        self.prometheus_url = prometheus_url
        self.session = requests.Session()
        self.session.timeout = 30

    def query(self, query: str, timestamp: datetime | None = None) -> dict:
        """Execute a PromQL query."""
        url = urljoin(self.prometheus_url, "/api/v1/query")
        params = {"query": query}

        if timestamp:
            params["time"] = timestamp.timestamp()

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.exception("Prometheus query failed: %s", e)
            raise

    def get_node_metrics(self, node_id: str, timestamp: datetime | None = None) -> dict:
        """Get node metrics for earning calculation."""
        metrics = {}

        # Node uptime (seconds)
        try:
            uptime_query = f'up{{instance="{node_id}"}}'
            uptime_result = self.query(uptime_query, timestamp)
            if uptime_result["status"] == "success" and uptime_result["data"]["result"]:
                metrics["uptime_seconds"] = int(
                    float(uptime_result["data"]["result"][0]["value"][1])
                )
            else:
                metrics["uptime_seconds"] = 0
        except Exception as e:
            logger.warning("Failed to get uptime for %s: %s", node_id, e)
            metrics["uptime_seconds"] = 0

        # FLOPs (floating point operations)
        try:
            flops_query = f'rate(twin_chat_latency_seconds_count{{instance="{node_id}"}}[5m]) * 1000000'
            flops_result = self.query(flops_query, timestamp)
            if flops_result["status"] == "success" and flops_result["data"]["result"]:
                metrics["flops"] = int(
                    float(flops_result["data"]["result"][0]["value"][1])
                )
            else:
                metrics["flops"] = 0
        except Exception as e:
            logger.warning("Failed to get FLOPs for %s: %s", node_id, e)
            metrics["flops"] = 0

        # Bandwidth (bytes)
        try:
            bandwidth_query = (
                f'rate(gw_requests_total{{instance="{node_id}"}}[5m]) * 1024'
            )
            bandwidth_result = self.query(bandwidth_query, timestamp)
            if (
                bandwidth_result["status"] == "success"
                and bandwidth_result["data"]["result"]
            ):
                metrics["bandwidth_bytes"] = int(
                    float(bandwidth_result["data"]["result"][0]["value"][1])
                )
            else:
                metrics["bandwidth_bytes"] = 0
        except Exception as e:
            logger.warning("Failed to get bandwidth for %s: %s", node_id, e)
            metrics["bandwidth_bytes"] = 0

        logger.debug("Node %s metrics: %s", node_id, metrics)
        return metrics

    def get_active_nodes(self) -> list[str]:
        """Get list of active nodes from Prometheus."""
        try:
            query = "up == 1"
            result = self.query(query)

            if result["status"] != "success":
                logger.error("Failed to get active nodes: %s", result)
                return []

            nodes = []
            for metric in result["data"]["result"]:
                instance = metric["metric"].get("instance", "")
                if instance:
                    nodes.append(instance)

            logger.info("Found %d active nodes: %s", len(nodes), nodes)
            return nodes

        except Exception as e:
            logger.exception("Failed to get active nodes: %s", e)
            return []


class EarnShellsWorker:
    """Worker that mints credits based on Prometheus metrics."""

    def __init__(self, prometheus_url: str, credits_api_url: str) -> None:
        self.prometheus_client = PrometheusClient(prometheus_url)
        self.credits_api_url = credits_api_url
        self.config = CreditsConfig()
        self.ledger = CreditsLedger(self.config)
        self.session = requests.Session()
        self.session.timeout = 30

    def ensure_user_exists(self, username: str, node_id: str) -> bool:
        """Ensure user exists in the system."""
        try:
            # Check if user exists
            response = self.session.get(f"{self.credits_api_url}/balance/{username}")
            if response.status_code == 200:
                return True
            if response.status_code == 404:
                # Create user
                create_response = self.session.post(
                    f"{self.credits_api_url}/users",
                    json={"username": username, "node_id": node_id},
                )
                if create_response.status_code == 201:
                    logger.info("Created user %s with node_id %s", username, node_id)
                    return True
                logger.error(
                    "Failed to create user %s: %s", username, create_response.text
                )
                return False
            logger.error("Failed to check user %s: %s", username, response.text)
            return False
        except Exception as e:
            logger.exception("Error ensuring user %s exists: %s", username, e)
            return False

    def mint_credits_for_node(self, node_id: str, scrape_timestamp: datetime) -> bool:
        """Mint credits for a specific node based on its metrics."""
        try:
            # Get node metrics
            metrics = self.prometheus_client.get_node_metrics(node_id, scrape_timestamp)

            # Generate username from node_id (you might want to customize this)
            username = f"node_{node_id.replace(':', '_').replace('.', '_')}"

            # Ensure user exists
            if not self.ensure_user_exists(username, node_id):
                logger.error("Failed to ensure user %s exists", username)
                return False

            # Submit earning request
            earn_request = {
                "username": username,
                "scrape_timestamp": scrape_timestamp.isoformat(),
                "uptime_seconds": metrics["uptime_seconds"],
                "flops": metrics["flops"],
                "bandwidth_bytes": metrics["bandwidth_bytes"],
            }

            response = self.session.post(
                f"{self.credits_api_url}/earn", json=earn_request
            )

            if response.status_code == 200:
                earning_data = response.json()
                logger.info(
                    "Minted %s credits for %s (uptime: %ds, FLOPs: %d, bandwidth: %d bytes)",
                    earning_data["credits_earned"],
                    username,
                    metrics["uptime_seconds"],
                    metrics["flops"],
                    metrics["bandwidth_bytes"],
                )
                return True
            logger.error("Failed to mint credits for %s: %s", username, response.text)
            return False

        except Exception as e:
            logger.exception("Error minting credits for node %s: %s", node_id, e)
            return False

    def run_earning_cycle(self) -> None:
        """Run a single earning cycle for all active nodes."""
        scrape_timestamp = datetime.now(UTC)
        logger.info("Starting earning cycle at %s", scrape_timestamp)

        try:
            # Get active nodes
            active_nodes = self.prometheus_client.get_active_nodes()

            if not active_nodes:
                logger.warning("No active nodes found")
                return

            # Mint credits for each node
            success_count = 0
            for node_id in active_nodes:
                if self.mint_credits_for_node(node_id, scrape_timestamp):
                    success_count += 1

            logger.info(
                "Earning cycle completed: %d/%d nodes processed successfully",
                success_count,
                len(active_nodes),
            )

        except Exception as e:
            logger.exception("Error in earning cycle: %s", e)

    def run_continuous(self, interval_seconds: int = 300) -> None:
        """Run worker continuously with specified interval."""
        logger.info(
            "Starting continuous earning worker with %ds interval", interval_seconds
        )

        while True:
            try:
                self.run_earning_cycle()
                logger.info("Sleeping for %d seconds...", interval_seconds)
                time.sleep(interval_seconds)

            except KeyboardInterrupt:
                logger.info("Received interrupt signal, shutting down...")
                break
            except Exception as e:
                logger.exception("Unexpected error in continuous worker: %s", e)
                time.sleep(60)  # Wait 1 minute before retrying

    def run_once(self) -> None:
        """Run worker once and exit."""
        logger.info("Running single earning cycle...")
        self.run_earning_cycle()
        logger.info("Single earning cycle completed")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Earn Shells Worker")
    parser.add_argument(
        "--prometheus-url",
        default="https://localhost:9090",
        help="Prometheus server URL",
    )
    parser.add_argument(
        "--credits-api-url", default="https://localhost:8002", help="Credits API URL"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Interval between earning cycles in seconds",
    )
    parser.add_argument(
        "--once", action="store_true", help="Run once and exit (useful for cron jobs)"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    worker = EarnShellsWorker(args.prometheus_url, args.credits_api_url)

    if args.once:
        worker.run_once()
    else:
        worker.run_continuous(args.interval)


if __name__ == "__main__":
    main()
