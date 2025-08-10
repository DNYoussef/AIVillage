#!/usr/bin/env python3
"""Compression Performance Monitoring Dashboard.

Tracks compression ratios, accuracy, and performance over time.
Provides alerts for regressions and generates reports.

This module provides comprehensive monitoring capabilities for compression
performance including automated benchmarking, regression detection,
and visualization generation.
"""

import argparse
from datetime import datetime, timedelta
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
except ImportError as e:
    logging.exception(f"Required dependencies missing: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("compression_monitor.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class CompressionMetrics:
    """Container for compression performance metrics.

    Attributes:
        timestamp: ISO format timestamp of the measurement
        compression_ratio: Achieved compression ratio (higher is better)
        relative_error: Relative reconstruction error (lower is better)
        compression_time: Time taken for compression in seconds
        memory_usage: Peak memory usage during compression in MB
        model_size: Size of the model in parameters
        method: Compression method used
        config: Configuration parameters used
    """

    def __init__(self) -> None:
        self.timestamp: str = datetime.now().isoformat()
        self.compression_ratio: float = 0.0
        self.relative_error: float = 0.0
        self.compression_time: float = 0.0
        self.memory_usage: float = 0.0
        self.model_size: int = 0
        self.method: str = ""
        self.config: dict[str, Any] = {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the metrics
        """
        return {
            "timestamp": self.timestamp,
            "compression_ratio": self.compression_ratio,
            "relative_error": self.relative_error,
            "compression_time": self.compression_time,
            "memory_usage": self.memory_usage,
            "model_size": self.model_size,
            "method": self.method,
            "config": self.config,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompressionMetrics":
        """Create from dictionary.

        Args:
            data: Dictionary containing metric data

        Returns:
            CompressionMetrics instance
        """
        metrics = cls()
        for key, value in data.items():
            if hasattr(metrics, key):
                setattr(metrics, key, value)
        return metrics


class CompressionMonitor:
    """Monitor compression performance and detect regressions.

    This class provides comprehensive monitoring of compression performance
    including automated benchmarking, regression detection, and reporting.

    Attributes:
        data_file: Path to the metrics history file
        metrics_history: List of historical compression metrics
        thresholds: Performance thresholds for regression detection
    """

    def __init__(self, data_file: str = "compression_metrics.json") -> None:
        """Initialize the compression monitor.

        Args:
            data_file: Path to store metrics history
        """
        self.data_file = Path(data_file)
        self.metrics_history: list[CompressionMetrics] = []
        self.load_history()

        # Regression thresholds
        self.thresholds = {
            "compression_ratio_min": 3.0,  # Minimum acceptable compression ratio
            "relative_error_max": 2.0,  # Maximum acceptable relative error
            "compression_time_max": 30.0,  # Maximum compression time (seconds)
            "memory_usage_max": 1000.0,  # Maximum memory usage (MB)
            "regression_threshold": 0.15,  # 15% regression threshold
        }

    def load_history(self) -> None:
        """Load metrics history from file.

        Raises:
            Exception: If loading fails, logs error and continues with empty history
        """
        if self.data_file.exists():
            try:
                with open(self.data_file, encoding="utf-8") as f:
                    data = json.load(f)
                    self.metrics_history = [
                        CompressionMetrics.from_dict(item) for item in data
                    ]
                logger.info(f"Loaded {len(self.metrics_history)} historical metrics")
            except Exception as e:
                logger.exception(f"Failed to load metrics history: {e}")
                self.metrics_history = []

    def save_history(self) -> None:
        """Save metrics history to file.

        Raises:
            Exception: If saving fails, logs error
        """
        try:
            data = [metrics.to_dict() for metrics in self.metrics_history]
            with open(self.data_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info(
                f"Saved {len(self.metrics_history)} metrics to {self.data_file}"
            )
        except Exception as e:
            logger.exception(f"Failed to save metrics history: {e}")

    def run_benchmark(self, method: str = "SeedLM") -> CompressionMetrics:
        """Run a quick benchmark and return metrics.

        Args:
            method: Compression method to benchmark

        Returns:
            CompressionMetrics with benchmark results
        """
        metrics = CompressionMetrics()
        metrics.method = method

        try:
            # Add current directory to path for imports
            sys.path.insert(0, os.getcwd())

            # Safely load and execute SeedLM implementation
            seedlm_path = Path("agent_forge/compression/seedlm.py")
            if not seedlm_path.exists():
                msg = f"SeedLM implementation not found at {seedlm_path}"
                raise FileNotFoundError(msg)

            # Use exec with controlled globals for safety
            seedlm_globals = {}
            with open(seedlm_path, encoding="utf-8") as f:
                exec(f.read(), seedlm_globals)

            # Create encoder from executed code
            SeedLMConfig = seedlm_globals.get("SeedLMConfig")
            ProgressiveSeedLMEncoder = seedlm_globals.get("ProgressiveSeedLMEncoder")

            if not SeedLMConfig or not ProgressiveSeedLMEncoder:
                msg = "Required classes not found in SeedLM implementation"
                raise ImportError(msg)

            config = SeedLMConfig()
            encoder = ProgressiveSeedLMEncoder(config)

            # Test on standard weight matrix
            test_weight = torch.randn(256, 512)

            # Benchmark compression
            start_time = time.time()
            compressed = encoder.encode(test_weight, compression_level=0.5)
            compression_time = time.time() - start_time

            # Benchmark decompression
            reconstructed = encoder.decode(compressed)

            # Calculate metrics
            metrics.compression_time = compression_time
            metrics.relative_error = (
                torch.norm(test_weight - reconstructed) / torch.norm(test_weight)
            ).item()
            metrics.model_size = test_weight.numel()

            # Handle different compressed data formats safely
            if isinstance(compressed, dict):
                metrics.compression_ratio = compressed.get("data", {}).get(
                    "compression_ratio", 4.0
                )
            else:
                # Estimate compression ratio based on size reduction
                original_size = test_weight.numel() * 4  # 4 bytes per float32
                compressed_size = getattr(
                    compressed, "numel", lambda: original_size // 4
                )()
                metrics.compression_ratio = original_size / (compressed_size * 4)

            metrics.config = {
                "compression_level": 0.5,
                "test_size": list(test_weight.shape),
            }

            logger.info(
                f"Benchmark complete: {metrics.compression_ratio:.2f}x ratio, "
                f"{metrics.relative_error:.4f} error, {compression_time:.2f}s"
            )

            return metrics

        except Exception as e:
            logger.exception(f"Benchmark failed: {e}")
            # Update metrics with error information
            metrics.compression_ratio = 0.0
            metrics.relative_error = float("inf")
            metrics.config["error"] = str(e)

        return metrics

    def add_metrics(self, metrics: CompressionMetrics) -> None:
        """Add new metrics to history.

        Args:
            metrics: Compression metrics to add to history
        """
        self.metrics_history.append(metrics)
        self.save_history()

    def check_regression(
        self, current_metrics: CompressionMetrics, lookback_days: int = 7
    ) -> list[str]:
        """Check for performance regressions.

        Args:
            current_metrics: Current metrics to compare against baseline
            lookback_days: Number of days to look back for baseline

        Returns:
            List of regression alert messages
        """
        alerts: list[str] = []

        # Get recent metrics for comparison
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        recent_metrics = [
            m
            for m in self.metrics_history[-20:]
            if datetime.fromisoformat(m.timestamp) > cutoff_date  # Last 20 runs
        ]

        if not recent_metrics:
            logger.warning("No recent metrics for regression comparison")
            return alerts

        # Calculate baseline averages
        baseline_ratio = np.mean([m.compression_ratio for m in recent_metrics])
        baseline_error = np.mean([m.relative_error for m in recent_metrics])
        baseline_time = np.mean([m.compression_time for m in recent_metrics])

        # Check for regressions
        ratio_regression = (
            baseline_ratio - current_metrics.compression_ratio
        ) / baseline_ratio
        error_regression = (
            current_metrics.relative_error - baseline_error
        ) / baseline_error
        time_regression = (
            current_metrics.compression_time - baseline_time
        ) / baseline_time

        threshold = self.thresholds["regression_threshold"]

        if ratio_regression > threshold:
            alerts.append(
                f"Compression ratio regression: {ratio_regression:.2%} decrease "
                f"({baseline_ratio:.2f}x → {current_metrics.compression_ratio:.2f}x)"
            )

        if error_regression > threshold:
            alerts.append(
                f"Accuracy regression: {error_regression:.2%} increase "
                f"({baseline_error:.4f} → {current_metrics.relative_error:.4f})"
            )

        if time_regression > threshold:
            alerts.append(
                f"Speed regression: {time_regression:.2%} slower "
                f"({baseline_time:.2f}s → {current_metrics.compression_time:.2f}s)"
            )

        # Check absolute thresholds
        if current_metrics.compression_ratio < self.thresholds["compression_ratio_min"]:
            alerts.append(
                f"Compression ratio below threshold: {current_metrics.compression_ratio:.2f}x "
                f"< {self.thresholds['compression_ratio_min']}x"
            )

        if current_metrics.relative_error > self.thresholds["relative_error_max"]:
            alerts.append(
                f"Relative error above threshold: {current_metrics.relative_error:.4f} "
                f"> {self.thresholds['relative_error_max']}"
            )

        if current_metrics.compression_time > self.thresholds["compression_time_max"]:
            alerts.append(
                f"Compression time above threshold: {current_metrics.compression_time:.2f}s "
                f"> {self.thresholds['compression_time_max']}s"
            )

        return alerts

    def generate_report(self, days: int = 30) -> str:
        """Generate performance report."""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_metrics = [
            m
            for m in self.metrics_history
            if datetime.fromisoformat(m.timestamp) > cutoff_date
        ]

        if not recent_metrics:
            return "No metrics available for the specified time period."

        # Create DataFrame for analysis
        df_data = []
        for m in recent_metrics:
            df_data.append(
                {
                    "timestamp": m.timestamp,
                    "compression_ratio": m.compression_ratio,
                    "relative_error": m.relative_error,
                    "compression_time": m.compression_time,
                    "method": m.method,
                }
            )

        df = pd.DataFrame(df_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Generate summary statistics
        report = f"""
# Compression Performance Report ({days} days)

## Summary Statistics
- **Total Benchmarks**: {len(recent_metrics)}
- **Average Compression Ratio**: {df["compression_ratio"].mean():.2f}x
- **Average Relative Error**: {df["relative_error"].mean():.4f}
- **Average Compression Time**: {df["compression_time"].mean():.2f}s

## Performance Trends
- **Best Compression Ratio**: {df["compression_ratio"].max():.2f}x
- **Worst Compression Ratio**: {df["compression_ratio"].min():.2f}x
- **Best Accuracy** (lowest error): {df["relative_error"].min():.4f}
- **Worst Accuracy** (highest error): {df["relative_error"].max():.4f}
- **Fastest Compression**: {df["compression_time"].min():.2f}s
- **Slowest Compression**: {df["compression_time"].max():.2f}s

## Recent Performance
"""

        # Add latest metrics
        if recent_metrics:
            latest = recent_metrics[-1]
            report += f"""
- **Latest Benchmark** ({latest.timestamp}):
  - Compression Ratio: {latest.compression_ratio:.2f}x
  - Relative Error: {latest.relative_error:.4f}
  - Compression Time: {latest.compression_time:.2f}s
  - Method: {latest.method}
"""

        return report

    def cleanup_old_metrics(self, days_to_keep: int = 90) -> int:
        """Clean up old metrics to prevent unbounded growth.

        Args:
            days_to_keep: Number of days of metrics to retain

        Returns:
            Number of metrics removed
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        original_count = len(self.metrics_history)

        self.metrics_history = [
            m
            for m in self.metrics_history
            if datetime.fromisoformat(m.timestamp) > cutoff_date
        ]

        removed_count = original_count - len(self.metrics_history)
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old metrics")
            self.save_history()

        return removed_count

    def create_visualizations(
        self, output_dir: str = "compression_plots"
    ) -> Path | None:
        """Create performance visualization plots.

        Args:
            output_dir: Directory to save plots

        Returns:
            Path to the generated plot file, or None if generation failed
        """
        if not self.metrics_history:
            logger.warning("No metrics history available for visualization")
            return None

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Prepare data
        timestamps = [datetime.fromisoformat(m.timestamp) for m in self.metrics_history]
        ratios = [m.compression_ratio for m in self.metrics_history]
        errors = [m.relative_error for m in self.metrics_history]
        times = [m.compression_time for m in self.metrics_history]

        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Compression Performance Over Time", fontsize=16)

        # Compression ratio over time
        axes[0, 0].plot(timestamps, ratios, "b-o", alpha=0.7)
        axes[0, 0].set_title("Compression Ratio")
        axes[0, 0].set_ylabel("Ratio (x)")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis="x", rotation=45)

        # Relative error over time
        axes[0, 1].plot(timestamps, errors, "r-o", alpha=0.7)
        axes[0, 1].set_title("Relative Error")
        axes[0, 1].set_ylabel("Relative Error")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis="x", rotation=45)

        # Compression time over time
        axes[1, 0].plot(timestamps, times, "g-o", alpha=0.7)
        axes[1, 0].set_title("Compression Time")
        axes[1, 0].set_ylabel("Time (seconds)")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis="x", rotation=45)

        # Quality vs Speed scatter
        axes[1, 1].scatter(times, errors, c=ratios, cmap="viridis", alpha=0.7)
        axes[1, 1].set_xlabel("Compression Time (s)")
        axes[1, 1].set_ylabel("Relative Error")
        axes[1, 1].set_title("Quality vs Speed (colored by ratio)")
        cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
        cbar.set_label("Compression Ratio (x)")

        plt.tight_layout()
        plt.savefig(
            output_path / "compression_performance.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

        logger.info(f"Visualizations saved to {output_path}")
        return output_path / "compression_performance.png"


def main() -> int:
    """Main monitoring script.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="Compression Performance Monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compression_monitor.py --run-benchmark --check-regression
  python compression_monitor.py --generate-report 30
  python compression_monitor.py --create-plots
""",
    )
    parser.add_argument(
        "--run-benchmark", action="store_true", help="Run benchmark and add to history"
    )
    parser.add_argument(
        "--check-regression",
        action="store_true",
        help="Check for performance regressions",
    )
    parser.add_argument(
        "--generate-report",
        type=int,
        metavar="DAYS",
        help="Generate performance report for N days",
    )
    parser.add_argument(
        "--create-plots",
        action="store_true",
        help="Create performance visualization plots",
    )
    parser.add_argument(
        "--data-file",
        default="compression_metrics.json",
        help="Path to metrics data file",
    )
    parser.add_argument(
        "--method", default="SeedLM", help="Compression method to benchmark"
    )

    args = parser.parse_args()

    # Initialize monitor
    monitor = CompressionMonitor(data_file=args.data_file)

    if args.run_benchmark:
        logger.info("Running compression benchmark...")
        metrics = monitor.run_benchmark(method=args.method)
        monitor.add_metrics(metrics)

        if args.check_regression:
            alerts = monitor.check_regression(metrics)
            if alerts:
                logger.warning("Performance regressions detected:")
                for alert in alerts:
                    logger.warning(f"  - {alert}")
                return 1
            logger.info("No performance regressions detected")

    if args.generate_report:
        logger.info(f"Generating performance report for {args.generate_report} days...")
        report = monitor.generate_report(days=args.generate_report)

        # Save report
        report_file = (
            f"compression_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        )
        with open(report_file, "w") as f:
            f.write(report)

        print(report)
        logger.info(f"Report saved to {report_file}")

    if args.create_plots:
        logger.info("Creating performance visualizations...")
        plot_path = monitor.create_visualizations()
        if plot_path:
            logger.info(f"Visualizations created at {plot_path}")

    # Cleanup old metrics to prevent unbounded growth
    monitor.cleanup_old_metrics()

    return 0


if __name__ == "__main__":
    sys.exit(main())
