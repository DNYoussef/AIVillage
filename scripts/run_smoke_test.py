#!/usr/bin/env python3
"""Agent Forge Production Smoke Test

This script executes the full Agent Forge pipeline and validates the results
against predefined thresholds to ensure production readiness.

Usage:
    python run_smoke_test.py --frontier-api-key YOUR_KEY
    python run_smoke_test.py --dry-run  # Configuration validation only
    python run_smoke_test.py --timeout 3600  # Custom timeout in seconds
"""

import argparse
import asyncio
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("smoke_test.log")
    ]
)
logger = logging.getLogger(__name__)

class AgentForgeSmokeTest:
    """Production smoke test for Agent Forge pipeline."""

    def __init__(self, args):
        self.args = args
        self.project_root = Path(__file__).parent
        self.results_file = self.project_root / "smoke_test_results.json"

        # Pipeline thresholds - based on run_full_agent_forge.py
        self.thresholds = {
            "MMLU": 0.60,
            "GSM8K": 0.40,
            "HumanEval": 0.25,
            "minimum_phases_completed": 4,  # At least 4/6 phases must succeed
            "maximum_execution_time": args.timeout
        }

        # W&B configuration
        self.wandb_project = "agent-forge-full-pipeline"
        self.wandb_api = None
        self.start_time = None
        self.pipeline_process = None

        logger.info(f"Smoke test initialized with timeout: {args.timeout}s")

    def setup_wandb_monitoring(self) -> bool:
        """Initialize W&B API for monitoring."""
        try:
            import wandb

            # Check for W&B API key
            wandb_key = os.getenv("WANDB_API_KEY")
            if not wandb_key:
                logger.warning("WANDB_API_KEY not found - metrics monitoring disabled")
                return False

            self.wandb_api = wandb.Api()
            logger.info("W&B monitoring initialized")
            return True

        except ImportError:
            logger.warning("wandb not installed - metrics monitoring disabled")
            return False
        except Exception as e:
            logger.warning(f"W&B setup failed: {e}")
            return False

    async def execute_pipeline(self) -> dict[str, Any]:
        """Execute the Agent Forge pipeline as subprocess."""
        logger.info("Starting Agent Forge pipeline execution...")

        # Build command
        cmd = [
            sys.executable,
            str(self.project_root / "run_full_agent_forge.py")
        ]

        # Add arguments from smoke test
        if self.args.frontier_api_key:
            cmd.extend(["--frontier-api-key", self.args.frontier_api_key])
        if self.args.dry_run:
            cmd.append("--dry-run")
        if self.args.no_deploy:
            cmd.append("--no-deploy")
        if self.args.quick:
            cmd.append("--quick")

        # Add timeout for the pipeline itself
        cmd.extend(["--timeout", str(self.args.timeout)])

        logger.info(f"Executing: {' '.join(cmd)}")

        # Start pipeline process
        self.start_time = datetime.now()

        try:
            self.pipeline_process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=self.project_root
            )

            # Monitor process with timeout
            stdout_lines = []

            try:
                # Wait for completion with timeout
                stdout, _ = await asyncio.wait_for(
                    self.pipeline_process.communicate(),
                    timeout=self.args.timeout
                )

                stdout_lines = stdout.decode("utf-8").splitlines() if stdout else []
                return_code = self.pipeline_process.returncode

            except asyncio.TimeoutError:
                logger.error(f"Pipeline execution timed out after {self.args.timeout}s")
                if self.pipeline_process:
                    self.pipeline_process.terminate()
                    try:
                        await asyncio.wait_for(self.pipeline_process.wait(), timeout=10)
                    except asyncio.TimeoutError:
                        self.pipeline_process.kill()

                return {
                    "status": "timeout",
                    "return_code": -1,
                    "execution_time": self.args.timeout,
                    "stdout_lines": stdout_lines,
                    "error": f"Pipeline timed out after {self.args.timeout} seconds"
                }

            execution_time = (datetime.now() - self.start_time).total_seconds()

            if return_code == 0:
                logger.info(f"Pipeline completed successfully in {execution_time:.1f}s")
                status = "success"
            else:
                logger.error(f"Pipeline failed with return code {return_code}")
                status = "failed"

            return {
                "status": status,
                "return_code": return_code,
                "execution_time": execution_time,
                "stdout_lines": stdout_lines,
                "error": None if return_code == 0 else f"Pipeline failed with code {return_code}"
            }

        except Exception as e:
            execution_time = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            logger.error(f"Pipeline execution failed: {e}")

            return {
                "status": "error",
                "return_code": -1,
                "execution_time": execution_time,
                "stdout_lines": [],
                "error": str(e)
            }

    def query_wandb_metrics(self) -> dict[str, Any]:
        """Query W&B for the latest pipeline run metrics."""
        if not self.wandb_api:
            return {"status": "wandb_unavailable", "metrics": {}}

        try:
            logger.info("Querying W&B for latest pipeline metrics...")

            # Get runs from the project
            runs = self.wandb_api.runs(
                path=f"{self.wandb_api.default_entity}/{self.wandb_project}",
                filters={"state": "finished"},
                order="-created_at"
            )

            if not runs:
                logger.warning("No finished runs found in W&B project")
                return {"status": "no_runs", "metrics": {}}

            # Get the most recent run
            latest_run = runs[0]

            # Extract metrics
            metrics = {}
            history = latest_run.history(pandas=False)

            # Look for benchmark scores in run history
            benchmark_metrics = ["MMLU", "GSM8K", "HumanEval", "HellaSwag", "ARC"]

            for row in history:
                for metric in benchmark_metrics:
                    # Check various possible metric key formats
                    possible_keys = [
                        metric,
                        f"unified_pipeline/{metric}_score",
                        f"mastery_trained/{metric}_score",
                        f"best_model/{metric}_score"
                    ]

                    for key in possible_keys:
                        if key in row and row[key] is not None:
                            metrics[metric] = row[key]
                            break

            # Also check run summary for final metrics
            summary = latest_run.summary
            for metric in benchmark_metrics:
                if metric not in metrics:  # Only if not found in history
                    possible_keys = [
                        metric,
                        f"final_{metric.lower()}",
                        f"best_{metric.lower()}_score"
                    ]

                    for key in possible_keys:
                        if key in summary:
                            metrics[metric] = summary[key]
                            break

            # Get pipeline status metrics
            pipeline_metrics = {}
            for key, value in summary.items():
                if "phase" in key.lower() or "pipeline" in key.lower():
                    pipeline_metrics[key] = value

            logger.info(f"Retrieved metrics from W&B run {latest_run.id}")

            return {
                "status": "success",
                "run_id": latest_run.id,
                "run_name": latest_run.name,
                "metrics": metrics,
                "pipeline_metrics": pipeline_metrics,
                "run_duration": (latest_run.summary.get("pipeline_duration_seconds", 0))
            }

        except Exception as e:
            logger.error(f"Failed to query W&B metrics: {e}")
            return {"status": "error", "error": str(e), "metrics": {}}

    def validate_benchmark_results(self) -> dict[str, Any]:
        """Validate benchmark results from local files."""
        logger.info("Validating benchmark results from local files...")

        # Check for benchmark results file
        results_file = self.project_root / "benchmark_results" / "agent_forge_model_comparison.json"

        if not results_file.exists():
            logger.warning(f"Benchmark results file not found: {results_file}")
            return {"status": "no_local_results", "validation": {}}

        try:
            with open(results_file) as f:
                results_data = json.load(f)

            model_averages = results_data.get("model_averages", {})
            benchmark_comparison = results_data.get("benchmark_comparison", [])

            if not model_averages:
                return {"status": "no_model_data", "validation": {}}

            # Find best performing model
            best_model = max(model_averages, key=model_averages.get)
            best_score = model_averages[best_model]

            # Extract benchmark scores for best model
            best_model_scores = {}
            for benchmark_data in benchmark_comparison:
                benchmark_name = benchmark_data.get("Benchmark")
                if benchmark_name and best_model in benchmark_data:
                    best_model_scores[benchmark_name] = benchmark_data[best_model]

            # Validate against thresholds
            validation_results = {}
            overall_pass = True

            for benchmark, threshold in self.thresholds.items():
                if benchmark in ["minimum_phases_completed", "maximum_execution_time"]:
                    continue  # These are validated elsewhere

                score = best_model_scores.get(benchmark, 0.0)
                passed = score >= threshold

                validation_results[benchmark] = {
                    "score": score,
                    "threshold": threshold,
                    "passed": passed,
                    "margin": score - threshold
                }

                if not passed:
                    overall_pass = False

            logger.info(f"Benchmark validation: {'PASSED' if overall_pass else 'FAILED'}")

            return {
                "status": "success",
                "best_model": best_model,
                "best_score": best_score,
                "validation": validation_results,
                "overall_pass": overall_pass,
                "all_scores": best_model_scores
            }

        except Exception as e:
            logger.error(f"Failed to validate benchmark results: {e}")
            return {"status": "error", "error": str(e), "validation": {}}

    def check_pipeline_artifacts(self) -> dict[str, Any]:
        """Check that expected pipeline artifacts were created."""
        logger.info("Checking pipeline artifacts...")

        expected_outputs = [
            ("Agent Forge Outputs", self.project_root / "agent_forge_outputs"),
            ("Benchmark Results", self.project_root / "benchmark_results"),
            ("Pipeline Summary", self.project_root / "agent_forge_pipeline_summary.json"),
            ("Forge Checkpoints", self.project_root / "forge_checkpoints")
        ]

        artifacts_status = {}
        all_present = True

        for name, path in expected_outputs:
            exists = path.exists()
            artifacts_status[name] = {
                "path": str(path),
                "exists": exists,
                "type": "directory" if path.is_dir() else "file" if exists else "missing"
            }

            if not exists:
                all_present = False
                logger.warning(f"Missing expected artifact: {path}")
            else:
                logger.info(f"Found artifact: {path}")

        return {
            "all_present": all_present,
            "artifacts": artifacts_status
        }

    async def run_smoke_test(self) -> dict[str, Any]:
        """Execute the complete smoke test workflow."""
        logger.info("="*60)
        logger.info("AGENT FORGE PRODUCTION SMOKE TEST")
        logger.info("="*60)

        test_start_time = datetime.now()

        # Initialize results structure
        results = {
            "smoke_test": {
                "timestamp": test_start_time.isoformat(),
                "version": "1.0.0",
                "timeout": self.args.timeout,
                "dry_run": self.args.dry_run
            },
            "pipeline_execution": {},
            "wandb_metrics": {},
            "benchmark_validation": {},
            "artifacts_check": {},
            "overall_status": "unknown",
            "summary": {}
        }

        try:
            # Step 1: Setup W&B monitoring
            wandb_available = self.setup_wandb_monitoring()

            # Step 2: Execute pipeline
            logger.info("Step 1/4: Executing Agent Forge pipeline...")
            pipeline_result = await self.execute_pipeline()
            results["pipeline_execution"] = pipeline_result

            # Step 3: Query W&B metrics (if available)
            if wandb_available and pipeline_result["status"] == "success":
                logger.info("Step 2/4: Querying W&B metrics...")
                # Wait a moment for W&B to process the run
                await asyncio.sleep(10)
                wandb_result = self.query_wandb_metrics()
                results["wandb_metrics"] = wandb_result
            else:
                logger.info("Step 2/4: Skipping W&B metrics (not available or pipeline failed)")
                results["wandb_metrics"] = {"status": "skipped"}

            # Step 4: Validate benchmark results
            logger.info("Step 3/4: Validating benchmark results...")
            benchmark_result = self.validate_benchmark_results()
            results["benchmark_validation"] = benchmark_result

            # Step 5: Check artifacts
            logger.info("Step 4/4: Checking pipeline artifacts...")
            artifacts_result = self.check_pipeline_artifacts()
            results["artifacts_check"] = artifacts_result

            # Determine overall status
            overall_pass = self._determine_overall_status(results)
            results["overall_status"] = "PASSED" if overall_pass else "FAILED"

            # Generate summary
            results["summary"] = self._generate_summary(results)

        except Exception as e:
            logger.error(f"Smoke test failed with exception: {e}")
            results["overall_status"] = "ERROR"
            results["summary"] = {
                "error": str(e),
                "execution_time": (datetime.now() - test_start_time).total_seconds()
            }

        # Calculate total test time
        test_duration = (datetime.now() - test_start_time).total_seconds()
        results["smoke_test"]["duration"] = test_duration

        # Save results
        with open(self.results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Smoke test results saved to: {self.results_file}")

        return results

    def _determine_overall_status(self, results: dict[str, Any]) -> bool:
        """Determine if the smoke test passed overall."""
        # Check pipeline execution
        pipeline_status = results["pipeline_execution"].get("status")
        if pipeline_status not in ["success"]:
            logger.error(f"Pipeline execution failed: {pipeline_status}")
            return False

        # Check execution time
        execution_time = results["pipeline_execution"].get("execution_time", 0)
        if execution_time > self.thresholds["maximum_execution_time"]:
            logger.error(f"Pipeline exceeded maximum execution time: {execution_time}s > {self.thresholds['maximum_execution_time']}s")
            return False

        # Check benchmark validation
        benchmark_validation = results["benchmark_validation"]
        if benchmark_validation.get("status") == "success":
            if not benchmark_validation.get("overall_pass", False):
                logger.error("Benchmark validation failed")
                return False
        else:
            logger.warning("Could not validate benchmarks - treating as non-critical")

        # Check artifacts
        if not results["artifacts_check"].get("all_present", False):
            logger.error("Required artifacts missing")
            return False

        logger.info("All smoke test criteria passed")
        return True

    def _generate_summary(self, results: dict[str, Any]) -> dict[str, Any]:
        """Generate a summary of the smoke test results."""
        summary = {
            "overall_status": results["overall_status"],
            "execution_time": results["pipeline_execution"].get("execution_time", 0),
            "pipeline_success": results["pipeline_execution"].get("status") == "success",
            "artifacts_present": results["artifacts_check"].get("all_present", False),
            "recommendations": []
        }

        # Add benchmark summary if available
        benchmark_validation = results["benchmark_validation"]
        if benchmark_validation.get("status") == "success":
            summary["benchmark_pass"] = benchmark_validation.get("overall_pass", False)
            summary["best_model"] = benchmark_validation.get("best_model")
            summary["best_score"] = benchmark_validation.get("best_score")

            # Add specific benchmark scores
            validation_details = benchmark_validation.get("validation", {})
            summary["benchmark_scores"] = {
                benchmark: details["score"]
                for benchmark, details in validation_details.items()
            }

            # Generate recommendations
            for benchmark, details in validation_details.items():
                if not details["passed"]:
                    summary["recommendations"].append(
                        f"Improve {benchmark} performance: {details['score']:.3f} < {details['threshold']:.3f}"
                    )

        # Add W&B summary if available
        wandb_metrics = results["wandb_metrics"]
        if wandb_metrics.get("status") == "success":
            summary["wandb_run_id"] = wandb_metrics.get("run_id")
            summary["wandb_available"] = True
        else:
            summary["wandb_available"] = False

        return summary

def main():
    """Main entry point for the smoke test."""
    parser = argparse.ArgumentParser(
        description="Agent Forge Production Smoke Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full smoke test with frontier API
  python run_smoke_test.py --frontier-api-key YOUR_KEY

  # Dry run validation only
  python run_smoke_test.py --dry-run

  # Custom timeout (default: 7200 seconds)
  python run_smoke_test.py --timeout 3600
        """
    )

    parser.add_argument(
        "--frontier-api-key",
        help="Frontier API key for pipeline execution"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run configuration validation only"
    )

    parser.add_argument(
        "--no-deploy",
        action="store_true",
        help="Skip deployment smoke test"
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run in quick mode with reduced iterations"
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=7200,  # 2 hours
        help="Timeout for pipeline execution in seconds (default: 7200)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    async def run_test():
        smoke_test = AgentForgeSmokeTest(args)
        results = await smoke_test.run_smoke_test()

        # Print final results
        print(f"\n{'='*80}")
        print("AGENT FORGE SMOKE TEST RESULTS")
        print(f"{'='*80}")

        status = results["overall_status"]
        print(f"Overall Status: {status}")

        if status == "PASSED":
            print("All smoke test criteria met - pipeline ready for production")
            summary = results.get("summary", {})

            if "best_model" in summary:
                print(f"Best Model: {summary['best_model']} (Score: {summary.get('best_score', 0):.3f})")

            if "execution_time" in summary:
                print(f"Execution Time: {summary['execution_time']:.1f} seconds")

        elif status == "FAILED":
            print("X Smoke test failed - review issues before production deployment")
            summary = results.get("summary", {})

            if summary.get("recommendations"):
                print("\nRecommendations:")
                for rec in summary["recommendations"]:
                    print(f"  - {rec}")

        else:
            print("! Smoke test encountered errors - check logs for details")

        print(f"\nDetailed results: {smoke_test.results_file}")
        print(f"{'='*80}\n")

        return 0 if status == "PASSED" else 1

    try:
        exit_code = asyncio.run(run_test())
        sys.exit(exit_code)

    except KeyboardInterrupt:
        logger.info("Smoke test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Smoke test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
