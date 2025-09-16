"""
Phase 8 Deployment Readiness Validator

Validates deployment readiness for compressed models across different platforms
and hardware configurations. Ensures real-world production compatibility.
"""

import os
import sys
import time
import json
import logging
import platform
import subprocess
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import psutil
import torch


@dataclass
class HardwareRequirements:
    """Hardware requirements specification"""
    min_ram_gb: float
    min_cpu_cores: int
    gpu_required: bool
    min_gpu_memory_gb: float
    min_disk_space_gb: float
    supported_architectures: List[str]


@dataclass
class PlatformCompatibility:
    """Platform compatibility results"""
    platform_name: str
    os_version: str
    python_version: str
    pytorch_version: str
    cuda_available: bool
    cuda_version: Optional[str]
    compatibility_score: float
    issues: List[str]


@dataclass
class DeploymentMetrics:
    """Deployment readiness metrics"""
    hardware_compatible: bool
    platform_compatible: bool
    inference_speed_acceptable: bool
    memory_usage_acceptable: bool
    startup_time_acceptable: bool
    cross_platform_tested: bool
    production_ready: bool
    deployment_issues: List[str]


class DeploymentReadinessValidator:
    """Validates deployment readiness for compressed models"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.test_results = {}

    def validate_deployment_readiness(
        self,
        model_path: str,
        hardware_requirements: HardwareRequirements,
        target_platforms: List[str],
        performance_targets: Dict[str, float]
    ) -> DeploymentMetrics:
        """
        Comprehensive deployment readiness validation

        Args:
            model_path: Path to compressed model
            hardware_requirements: Hardware requirements spec
            target_platforms: List of target platform names
            performance_targets: Performance targets dict

        Returns:
            DeploymentMetrics with readiness assessment
        """
        self.logger.info("Starting deployment readiness validation")

        try:
            # Validate current hardware compatibility
            hardware_compatible = self._validate_hardware_requirements(hardware_requirements)

            # Validate platform compatibility
            platform_results = self._validate_platform_compatibility(target_platforms)
            platform_compatible = all(r.compatibility_score > 0.8 for r in platform_results)

            # Load and test model
            model = self._load_model_safely(model_path)

            # Validate inference performance
            inference_acceptable = self._validate_inference_performance(
                model, performance_targets.get('max_inference_time', 1.0)
            )

            # Validate memory usage
            memory_acceptable = self._validate_memory_usage(
                model, performance_targets.get('max_memory_mb', 1024)
            )

            # Validate startup time
            startup_acceptable = self._validate_startup_time(
                model_path, performance_targets.get('max_startup_time', 10.0)
            )

            # Cross-platform testing
            cross_platform_tested = self._validate_cross_platform_compatibility(
                model_path, target_platforms
            )

            # Collect deployment issues
            deployment_issues = self._collect_deployment_issues(
                hardware_compatible, platform_results,
                inference_acceptable, memory_acceptable, startup_acceptable
            )

            # Overall production readiness
            production_ready = (
                hardware_compatible and
                platform_compatible and
                inference_acceptable and
                memory_acceptable and
                startup_acceptable and
                len(deployment_issues) == 0
            )

            metrics = DeploymentMetrics(
                hardware_compatible=hardware_compatible,
                platform_compatible=platform_compatible,
                inference_speed_acceptable=inference_acceptable,
                memory_usage_acceptable=memory_acceptable,
                startup_time_acceptable=startup_acceptable,
                cross_platform_tested=cross_platform_tested,
                production_ready=production_ready,
                deployment_issues=deployment_issues
            )

            self._log_deployment_results(metrics)
            return metrics

        except Exception as e:
            self.logger.error(f"Deployment readiness validation failed: {e}")
            raise

    def _validate_hardware_requirements(self, requirements: HardwareRequirements) -> bool:
        """Validate hardware requirements against current system"""
        issues = []

        # Check RAM
        available_ram = psutil.virtual_memory().total / (1024**3)  # GB
        if available_ram < requirements.min_ram_gb:
            issues.append(f"Insufficient RAM: {available_ram:.1f}GB < {requirements.min_ram_gb}GB")

        # Check CPU cores
        cpu_cores = psutil.cpu_count(logical=False)
        if cpu_cores < requirements.min_cpu_cores:
            issues.append(f"Insufficient CPU cores: {cpu_cores} < {requirements.min_cpu_cores}")

        # Check GPU if required
        if requirements.gpu_required:
            if not torch.cuda.is_available():
                issues.append("GPU required but CUDA not available")
            else:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                if gpu_memory < requirements.min_gpu_memory_gb:
                    issues.append(f"Insufficient GPU memory: {gpu_memory:.1f}GB < {requirements.min_gpu_memory_gb}GB")

        # Check disk space
        disk_usage = psutil.disk_usage('/')
        available_disk = disk_usage.free / (1024**3)  # GB
        if available_disk < requirements.min_disk_space_gb:
            issues.append(f"Insufficient disk space: {available_disk:.1f}GB < {requirements.min_disk_space_gb}GB")

        # Check architecture
        current_arch = platform.machine().lower()
        if current_arch not in [arch.lower() for arch in requirements.supported_architectures]:
            issues.append(f"Unsupported architecture: {current_arch}")

        if issues:
            self.logger.warning(f"Hardware compatibility issues: {issues}")
            return False

        self.logger.info("Hardware requirements satisfied")
        return True

    def _validate_platform_compatibility(self, target_platforms: List[str]) -> List[PlatformCompatibility]:
        """Validate compatibility with target platforms"""
        results = []
        current_platform = platform.system().lower()

        for target in target_platforms:
            issues = []
            compatibility_score = 1.0

            # Platform OS compatibility
            if target.lower() != current_platform:
                if not self._is_cross_platform_compatible(current_platform, target.lower()):
                    issues.append(f"Platform mismatch: {current_platform} vs {target}")
                    compatibility_score *= 0.5

            # Python version compatibility
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
            if not self._is_python_version_compatible(python_version):
                issues.append(f"Python version compatibility issue: {python_version}")
                compatibility_score *= 0.8

            # PyTorch compatibility
            pytorch_version = torch.__version__
            if not self._is_pytorch_version_compatible(pytorch_version):
                issues.append(f"PyTorch version compatibility issue: {pytorch_version}")
                compatibility_score *= 0.8

            # CUDA compatibility
            cuda_available = torch.cuda.is_available()
            cuda_version = torch.version.cuda if cuda_available else None

            result = PlatformCompatibility(
                platform_name=target,
                os_version=platform.version(),
                python_version=python_version,
                pytorch_version=pytorch_version,
                cuda_available=cuda_available,
                cuda_version=cuda_version,
                compatibility_score=compatibility_score,
                issues=issues
            )

            results.append(result)

        return results

    def _load_model_safely(self, model_path: str) -> Any:
        """Load model with error handling and validation"""
        try:
            model = torch.load(model_path, map_location='cpu')

            # Validate model structure
            if hasattr(model, 'eval'):
                model.eval()
            else:
                self.logger.warning("Model doesn't have eval() method")

            # Test basic forward pass
            try:
                if hasattr(model, '__call__'):
                    # Create dummy input
                    dummy_input = torch.randn(1, 3, 224, 224)  # Common image size
                    with torch.no_grad():
                        _ = model(dummy_input)
                    self.logger.info("Model forward pass test successful")
            except Exception as e:
                self.logger.warning(f"Model forward pass test failed: {e}")

            return model

        except Exception as e:
            self.logger.error(f"Failed to load model {model_path}: {e}")
            raise

    def _validate_inference_performance(self, model: Any, max_inference_time: float) -> bool:
        """Validate inference performance against targets"""
        try:
            model.eval()
            dummy_input = torch.randn(1, 3, 224, 224)

            # Warm up
            with torch.no_grad():
                for _ in range(5):
                    _ = model(dummy_input)

            # Measure inference time
            times = []
            with torch.no_grad():
                for _ in range(10):
                    start_time = time.perf_counter()
                    _ = model(dummy_input)
                    end_time = time.perf_counter()
                    times.append(end_time - start_time)

            avg_time = sum(times) / len(times)

            if avg_time <= max_inference_time:
                self.logger.info(f"Inference performance acceptable: {avg_time:.4f}s <= {max_inference_time}s")
                return True
            else:
                self.logger.warning(f"Inference too slow: {avg_time:.4f}s > {max_inference_time}s")
                return False

        except Exception as e:
            self.logger.error(f"Inference performance validation failed: {e}")
            return False

    def _validate_memory_usage(self, model: Any, max_memory_mb: float) -> bool:
        """Validate memory usage against targets"""
        try:
            import psutil
            process = psutil.Process()

            # Baseline memory
            baseline = process.memory_info().rss / (1024*1024)  # MB

            # Load model and run inference
            model.eval()
            dummy_input = torch.randn(1, 3, 224, 224)

            with torch.no_grad():
                _ = model(dummy_input)

            # Peak memory
            peak = process.memory_info().rss / (1024*1024)  # MB
            memory_usage = peak - baseline

            if memory_usage <= max_memory_mb:
                self.logger.info(f"Memory usage acceptable: {memory_usage:.1f}MB <= {max_memory_mb}MB")
                return True
            else:
                self.logger.warning(f"Memory usage too high: {memory_usage:.1f}MB > {max_memory_mb}MB")
                return False

        except Exception as e:
            self.logger.error(f"Memory usage validation failed: {e}")
            return False

    def _validate_startup_time(self, model_path: str, max_startup_time: float) -> bool:
        """Validate model startup/loading time"""
        try:
            start_time = time.perf_counter()
            _ = torch.load(model_path, map_location='cpu')
            end_time = time.perf_counter()

            startup_time = end_time - start_time

            if startup_time <= max_startup_time:
                self.logger.info(f"Startup time acceptable: {startup_time:.2f}s <= {max_startup_time}s")
                return True
            else:
                self.logger.warning(f"Startup time too slow: {startup_time:.2f}s > {max_startup_time}s")
                return False

        except Exception as e:
            self.logger.error(f"Startup time validation failed: {e}")
            return False

    def _validate_cross_platform_compatibility(
        self,
        model_path: str,
        target_platforms: List[str]
    ) -> bool:
        """Validate cross-platform compatibility"""
        # This is a simplified check - in practice, you'd test on actual platforms
        current_platform = platform.system().lower()

        compatible_platforms = 0
        for target in target_platforms:
            if self._is_cross_platform_compatible(current_platform, target.lower()):
                compatible_platforms += 1

        compatibility_ratio = compatible_platforms / len(target_platforms)

        if compatibility_ratio >= 0.8:  # 80% compatibility threshold
            self.logger.info(f"Cross-platform compatibility acceptable: {compatibility_ratio:.1%}")
            return True
        else:
            self.logger.warning(f"Cross-platform compatibility insufficient: {compatibility_ratio:.1%}")
            return False

    def _is_cross_platform_compatible(self, current: str, target: str) -> bool:
        """Check if platforms are compatible"""
        # Simplified compatibility matrix
        compatibility_map = {
            'linux': ['linux', 'darwin'],  # Linux can often run on macOS via containers
            'darwin': ['darwin', 'linux'],  # macOS can run Linux containers
            'windows': ['windows']  # Windows is more isolated
        }

        return target in compatibility_map.get(current, [current])

    def _is_python_version_compatible(self, version: str) -> bool:
        """Check Python version compatibility"""
        major, minor = map(int, version.split('.'))

        # Support Python 3.8+
        if major == 3 and minor >= 8:
            return True

        return False

    def _is_pytorch_version_compatible(self, version: str) -> bool:
        """Check PyTorch version compatibility"""
        try:
            major, minor = map(int, version.split('.')[:2])

            # Support PyTorch 1.10+
            if major > 1 or (major == 1 and minor >= 10):
                return True
        except:
            pass

        return False

    def _collect_deployment_issues(
        self,
        hardware_compatible: bool,
        platform_results: List[PlatformCompatibility],
        inference_acceptable: bool,
        memory_acceptable: bool,
        startup_acceptable: bool
    ) -> List[str]:
        """Collect all deployment issues"""
        issues = []

        if not hardware_compatible:
            issues.append("Hardware requirements not met")

        for result in platform_results:
            if result.compatibility_score < 0.8:
                issues.extend(result.issues)

        if not inference_acceptable:
            issues.append("Inference performance below target")

        if not memory_acceptable:
            issues.append("Memory usage exceeds target")

        if not startup_acceptable:
            issues.append("Startup time exceeds target")

        return issues

    def _log_deployment_results(self, metrics: DeploymentMetrics) -> None:
        """Log deployment readiness results"""
        self.logger.info("=== Deployment Readiness Validation Results ===")
        self.logger.info(f"Hardware Compatible: {metrics.hardware_compatible}")
        self.logger.info(f"Platform Compatible: {metrics.platform_compatible}")
        self.logger.info(f"Inference Speed Acceptable: {metrics.inference_speed_acceptable}")
        self.logger.info(f"Memory Usage Acceptable: {metrics.memory_usage_acceptable}")
        self.logger.info(f"Startup Time Acceptable: {metrics.startup_time_acceptable}")
        self.logger.info(f"Cross-Platform Tested: {metrics.cross_platform_tested}")
        self.logger.info(f"Production Ready: {metrics.production_ready}")

        if metrics.deployment_issues:
            self.logger.warning(f"Deployment Issues: {metrics.deployment_issues}")

        if metrics.production_ready:
            self.logger.info("DEPLOYMENT READINESS VALIDATION PASSED")
        else:
            self.logger.error("DEPLOYMENT READINESS VALIDATION FAILED")

    def generate_deployment_report(
        self,
        metrics: DeploymentMetrics,
        platform_results: List[PlatformCompatibility],
        output_path: str
    ) -> None:
        """Generate comprehensive deployment report"""
        report = {
            "validation_timestamp": time.time(),
            "deployment_readiness": {
                "hardware_compatible": metrics.hardware_compatible,
                "platform_compatible": metrics.platform_compatible,
                "inference_speed_acceptable": metrics.inference_speed_acceptable,
                "memory_usage_acceptable": metrics.memory_usage_acceptable,
                "startup_time_acceptable": metrics.startup_time_acceptable,
                "cross_platform_tested": metrics.cross_platform_tested,
                "production_ready": metrics.production_ready,
                "issues": metrics.deployment_issues
            },
            "platform_compatibility": [
                {
                    "platform": result.platform_name,
                    "compatibility_score": result.compatibility_score,
                    "issues": result.issues,
                    "python_version": result.python_version,
                    "pytorch_version": result.pytorch_version,
                    "cuda_available": result.cuda_available
                }
                for result in platform_results
            ],
            "system_info": {
                "platform": platform.system(),
                "architecture": platform.machine(),
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "pytorch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available()
            }
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Deployment report saved to {output_path}")


def main():
    """Main validation function"""
    logging.basicConfig(level=logging.INFO)

    # Example usage
    validator = DeploymentReadinessValidator()

    print("Deployment Readiness Validator initialized")
    print("Use validate_deployment_readiness() with actual models and requirements")


if __name__ == "__main__":
    main()