#!/usr/bin/env python3
"""Agent Forge Complete Execution Script

Master script that sets up and runs the complete Agent Forge pipeline.
This script handles:
1. Environment validation
2. Model and benchmark downloads
3. Pipeline execution
4. Monitoring dashboard launch
5. Results reporting
"""

import argparse
import asyncio
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("agent_forge_execution.log"),
    ],
)
logger = logging.getLogger(__name__)


class AgentForgeRunner:
    """Complete Agent Forge execution manager"""

    def __init__(self):
        self.start_time = datetime.now()
        self.setup_success = False
        self.pipeline_success = False
        self.dashboard_process = None

    def validate_environment(self) -> bool:
        """Validate the execution environment"""
        logger.info("🔍 Validating environment...")

        try:
            # Check Python version
            if sys.version_info < (3, 10):
                logger.error("❌ Python 3.10+ required")
                return False

            # Check key dependencies
            try:
                import datasets
                import torch
                import transformers

                logger.info("✅ Core dependencies available")
            except ImportError as e:
                logger.error(f"❌ Missing dependencies: {e}")
                return False

            # Check GPU
            import torch

            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                memory_gb = torch.cuda.get_device_properties(0).total_memory / (
                    1024**3
                )
                logger.info(f"✅ GPU: {device_name} ({memory_gb:.1f} GB)")

                if memory_gb < 6:
                    logger.warning("⚠️ GPU memory may be insufficient for 1.5B models")
            else:
                logger.warning("⚠️ CUDA not available, will use CPU")

            # Check disk space
            import shutil

            free_gb = shutil.disk_usage(".").free / (1024**3)
            if free_gb < 20:
                logger.error(f"❌ Insufficient disk space: {free_gb:.1f} GB free")
                return False

            logger.info("✅ Environment validation passed")
            return True

        except Exception as e:
            logger.error(f"❌ Environment validation failed: {e}")
            return False

    def setup_environment(self, skip_downloads: bool = False) -> bool:
        """Set up the complete environment"""
        logger.info("🛠️ Setting up Agent Forge environment...")

        try:
            # Run setup script
            setup_cmd = [sys.executable, "scripts/setup_environment.py"]
            if skip_downloads:
                setup_cmd.append("--skip-downloads")

            result = subprocess.run(
                setup_cmd, check=False, capture_output=True, text=True
            )

            if result.returncode == 0:
                logger.info("✅ Environment setup completed")
                logger.info(result.stdout)
                self.setup_success = True
                return True
            logger.error("❌ Environment setup failed")
            logger.error(result.stderr)
            return False

        except Exception as e:
            logger.error(f"❌ Setup error: {e}")
            return False

    def download_resources(self) -> bool:
        """Download models and benchmarks"""
        logger.info("📥 Downloading resources...")

        download_success = True

        # Download models
        try:
            logger.info("Downloading models...")
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/download_models.py",
                    "--models-dir",
                    "D:/agent_forge_models",
                    "--check-space",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=1800,
            )  # 30 min timeout

            if result.returncode == 0:
                logger.info("✅ Models downloaded successfully")
            else:
                logger.error("❌ Model download failed")
                logger.error(result.stderr)
                download_success = False

        except subprocess.TimeoutExpired:
            logger.warning("⚠️ Model download timed out, continuing...")
        except Exception as e:
            logger.error(f"❌ Model download error: {e}")
            download_success = False

        # Download benchmarks
        try:
            logger.info("Downloading benchmarks...")
            result = subprocess.run(
                [sys.executable, "scripts/download_benchmarks.py"],
                check=False,
                capture_output=True,
                text=True,
                timeout=600,
            )  # 10 min timeout

            if result.returncode == 0:
                logger.info("✅ Benchmarks downloaded successfully")
            else:
                logger.warning("⚠️ Benchmark download issues (continuing)")
                logger.warning(result.stderr)

        except Exception as e:
            logger.warning(f"⚠️ Benchmark download error: {e}")

        return download_success

    async def run_pipeline(self) -> bool:
        """Execute the Agent Forge pipeline"""
        logger.info("🚀 Starting Agent Forge pipeline...")

        try:
            # Import and run enhanced orchestrator
            sys.path.append(str(Path("agent_forge")))

            from enhanced_orchestrator import run_enhanced_pipeline

            logger.info("Pipeline starting...")
            results = await run_enhanced_pipeline()

            # Analyze results
            completed_phases = sum(1 for r in results.values() if r.success)
            total_phases = len(results)
            success_rate = completed_phases / total_phases if total_phases > 0 else 0

            logger.info(
                f"Pipeline completed: {completed_phases}/{total_phases} phases ({success_rate:.1%})"
            )

            if success_rate >= 0.6:  # 60% success threshold
                logger.info("✅ Pipeline execution successful")
                self.pipeline_success = True
                return True
            logger.warning("⚠️ Pipeline had significant issues")
            return False

        except Exception as e:
            logger.error(f"❌ Pipeline execution failed: {e}")
            return False

    def launch_dashboard(self) -> bool:
        """Launch monitoring dashboard"""
        logger.info("📊 Launching monitoring dashboard...")

        try:
            # Launch dashboard in background
            dashboard_cmd = [sys.executable, "scripts/run_dashboard.py"]
            self.dashboard_process = subprocess.Popen(
                dashboard_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            # Give it a moment to start
            time.sleep(3)

            if self.dashboard_process.poll() is None:
                logger.info("✅ Dashboard launched at http://localhost:8501")
                return True
            logger.error("❌ Dashboard failed to start")
            return False

        except Exception as e:
            logger.error(f"❌ Dashboard launch error: {e}")
            return False

    def generate_report(self):
        """Generate execution summary report"""
        end_time = datetime.now()
        duration = end_time - self.start_time

        report = f"""
{"=" * 60}
AGENT FORGE EXECUTION SUMMARY
{"=" * 60}

Execution Time: {self.start_time.strftime("%Y-%m-%d %H:%M:%S")} - {end_time.strftime("%Y-%m-%d %H:%M:%S")}
Total Duration: {duration}

Environment Setup: {"✅ SUCCESS" if self.setup_success else "❌ FAILED"}
Pipeline Execution: {"✅ SUCCESS" if self.pipeline_success else "❌ FAILED"}
Dashboard Status: {"✅ RUNNING" if self.dashboard_process and self.dashboard_process.poll() is None else "❌ NOT RUNNING"}

NEXT STEPS:
{self.get_next_steps()}

{"=" * 60}
        """

        logger.info(report)

        # Save report to file
        report_file = (
            f"agent_forge_report_{self.start_time.strftime('%Y%m%d_%H%M%S')}.txt"
        )
        with open(report_file, "w") as f:
            f.write(report)

        logger.info(f"📄 Full report saved to: {report_file}")

    def get_next_steps(self) -> str:
        """Get appropriate next steps based on execution results"""
        if self.setup_success and self.pipeline_success:
            return """
1. 🎉 Agent Forge is running successfully!
2. 📊 View dashboard at: http://localhost:8501
3. 🔬 Check pipeline outputs in: ./forge_output_enhanced/
4. 📈 Monitor W&B runs for detailed metrics
5. 🧪 Run evaluation: python benchmarks/evaluate_model.py
"""
        if self.setup_success:
            return """
1. ✅ Environment setup completed
2. ⚠️ Pipeline had issues - check logs for details
3. 🔧 Try running pipeline manually: python agent_forge/enhanced_orchestrator.py
4. 📊 Dashboard available at: http://localhost:8501
5. 🐛 Debug issues using the monitoring dashboard
"""
        return """
1. ❌ Environment setup failed
2. 🔧 Run setup manually: python scripts/setup_environment.py
3. 📋 Check requirements: pip install -r agent_forge/requirements.txt
4. 💾 Ensure sufficient disk space (20+ GB recommended)
5. 🔍 Check agent_forge_execution.log for detailed errors
"""

    def cleanup(self):
        """Clean up resources"""
        if self.dashboard_process and self.dashboard_process.poll() is None:
            logger.info("🧹 Stopping dashboard...")
            self.dashboard_process.terminate()
            try:
                self.dashboard_process.wait(timeout=5)
            except BaseException:
                self.dashboard_process.kill()


async def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Run complete Agent Forge pipeline")
    parser.add_argument(
        "--skip-setup", action="store_true", help="Skip environment setup"
    )
    parser.add_argument(
        "--skip-downloads", action="store_true", help="Skip model/benchmark downloads"
    )
    parser.add_argument(
        "--skip-pipeline", action="store_true", help="Skip pipeline execution"
    )
    parser.add_argument(
        "--skip-dashboard", action="store_true", help="Skip dashboard launch"
    )
    parser.add_argument(
        "--validate-only", action="store_true", help="Only validate environment"
    )

    args = parser.parse_args()

    runner = AgentForgeRunner()

    try:
        logger.info("🤖 Agent Forge Complete Execution Starting...")

        # Step 1: Validate environment
        if not runner.validate_environment():
            logger.error("❌ Environment validation failed, aborting")
            return 1

        if args.validate_only:
            logger.info("✅ Environment validation completed")
            return 0

        # Step 2: Setup environment
        if not args.skip_setup:
            if not runner.setup_environment(args.skip_downloads):
                logger.error("❌ Environment setup failed")
                return 1

        # Step 3: Download resources
        if not args.skip_downloads and not args.skip_setup:
            runner.download_resources()  # Non-blocking

        # Step 4: Launch dashboard
        if not args.skip_dashboard:
            runner.launch_dashboard()

        # Step 5: Run pipeline
        if not args.skip_pipeline:
            await runner.run_pipeline()

        # Step 6: Generate report
        runner.generate_report()

        # Keep dashboard running if successful
        if (
            runner.dashboard_process
            and runner.dashboard_process.poll() is None
            and runner.pipeline_success
        ):
            logger.info("🎉 Agent Forge execution completed successfully!")
            logger.info("📊 Dashboard running at http://localhost:8501")
            logger.info("Press Ctrl+C to stop and exit")

            try:
                while runner.dashboard_process.poll() is None:
                    await asyncio.sleep(5)
            except KeyboardInterrupt:
                logger.info("👋 Stopping Agent Forge...")

        return 0 if runner.pipeline_success else 1

    except KeyboardInterrupt:
        logger.info("👋 Execution interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return 1
    finally:
        runner.cleanup()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
