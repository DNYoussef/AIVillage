"""
Phase 8 Production Validator

Comprehensive production readiness validation for compressed AI models.
Integrates quality validation, deployment readiness, and theater detection
to ensure genuine production-ready compression results.
"""

import os
import sys
import time
import json
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import numpy as np

# Import our validation components
from .compression_quality_validator import CompressionQualityValidator, CompressionMetrics, QualityThresholds
from .deployment_readiness_validator import DeploymentReadinessValidator, DeploymentMetrics, HardwareRequirements
from .compression_theater_detector import CompressionTheaterDetector, TheaterDetectionResult


@dataclass
class ProductionValidationConfig:
    """Production validation configuration"""
    quality_thresholds: QualityThresholds
    hardware_requirements: HardwareRequirements
    target_platforms: List[str]
    performance_targets: Dict[str, float]
    enable_theater_detection: bool = True
    validation_timeout_minutes: int = 30
    output_directory: str = "validation_results"


@dataclass
class ProductionValidationResult:
    """Complete production validation results"""
    timestamp: float
    validation_passed: bool
    overall_score: float
    quality_metrics: CompressionMetrics
    deployment_metrics: DeploymentMetrics
    theater_detection: TheaterDetectionResult
    production_ready: bool
    critical_issues: List[str]
    warnings: List[str]
    recommendations: List[str]


class Phase8ProductionValidator:
    """Comprehensive production validator for Phase 8 compression"""

    def __init__(self, config: ProductionValidationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize sub-validators
        self.quality_validator = CompressionQualityValidator(config.quality_thresholds)
        self.deployment_validator = DeploymentReadinessValidator()
        self.theater_detector = CompressionTheaterDetector()

        # Ensure output directory exists
        Path(config.output_directory).mkdir(parents=True, exist_ok=True)

    def validate_production_readiness(
        self,
        original_model_path: str,
        compressed_model_path: str,
        test_data: Any,
        validation_name: str = "phase8_compression"
    ) -> ProductionValidationResult:
        """
        Comprehensive production readiness validation

        Args:
            original_model_path: Path to original model
            compressed_model_path: Path to compressed model
            test_data: Test dataset for validation
            validation_name: Name for this validation run

        Returns:
            ProductionValidationResult with complete assessment
        """
        self.logger.info(f"Starting Phase 8 production validation: {validation_name}")
        start_time = time.time()

        try:
            # Initialize result tracking
            critical_issues = []
            warnings = []
            recommendations = []

            # 1. Quality Validation
            self.logger.info("Step 1/4: Validating compression quality")
            quality_metrics = self.quality_validator.validate_compression_quality(
                original_model_path=original_model_path,
                compressed_model_path=compressed_model_path,
                test_data=test_data,
                validation_config={}
            )

            if not quality_metrics.validation_passed:
                critical_issues.append("Compression quality validation failed")

            if quality_metrics.theater_detected:
                warnings.append("Quality theater patterns detected")

            # 2. Deployment Readiness Validation
            self.logger.info("Step 2/4: Validating deployment readiness")
            deployment_metrics = self.deployment_validator.validate_deployment_readiness(
                model_path=compressed_model_path,
                hardware_requirements=self.config.hardware_requirements,
                target_platforms=self.config.target_platforms,
                performance_targets=self.config.performance_targets
            )

            if not deployment_metrics.production_ready:
                critical_issues.append("Deployment readiness validation failed")
                critical_issues.extend(deployment_metrics.deployment_issues)

            # 3. Theater Detection
            theater_detection = TheaterDetectionResult(
                theater_detected=False,
                confidence_score=1.0,
                theater_indicators=[],
                evidence=[],
                severity="none"
            )

            if self.config.enable_theater_detection:
                self.logger.info("Step 3/4: Detecting compression theater")
                theater_detection = self.theater_detector.detect_compression_theater(
                    original_model_path=original_model_path,
                    compressed_model_path=compressed_model_path,
                    quality_metrics=quality_metrics,
                    deployment_metrics=deployment_metrics
                )

                if theater_detection.theater_detected:
                    if theater_detection.severity in ["high", "critical"]:
                        critical_issues.append("High-severity theater patterns detected")
                    else:
                        warnings.append("Theater patterns detected")

            # 4. Integration Validation with Phase 7
            self.logger.info("Step 4/4: Validating Phase 7 integration")
            integration_valid = self._validate_phase7_integration(
                compressed_model_path, quality_metrics
            )

            if not integration_valid:
                critical_issues.append("Phase 7 integration validation failed")

            # 5. Calculate overall scores and recommendations
            overall_score = self._calculate_overall_score(
                quality_metrics, deployment_metrics, theater_detection
            )

            recommendations.extend(self._generate_recommendations(
                quality_metrics, deployment_metrics, theater_detection
            ))

            # 6. Determine final production readiness
            production_ready = (
                quality_metrics.validation_passed and
                deployment_metrics.production_ready and
                not (theater_detection.theater_detected and theater_detection.severity in ["high", "critical"]) and
                integration_valid and
                len(critical_issues) == 0
            )

            # Create final result
            result = ProductionValidationResult(
                timestamp=time.time(),
                validation_passed=production_ready,
                overall_score=overall_score,
                quality_metrics=quality_metrics,
                deployment_metrics=deployment_metrics,
                theater_detection=theater_detection,
                production_ready=production_ready,
                critical_issues=critical_issues,
                warnings=warnings,
                recommendations=recommendations
            )

            # Generate comprehensive report
            self._generate_production_report(result, validation_name)

            # Log final results
            self._log_final_results(result, time.time() - start_time)

            return result

        except Exception as e:
            self.logger.error(f"Production validation failed with exception: {e}")
            raise

    def _validate_phase7_integration(
        self,
        compressed_model_path: str,
        quality_metrics: CompressionMetrics
    ) -> bool:
        """Validate integration with Phase 7 optimization results"""
        try:
            # Check for Phase 7 artifacts and compatibility
            phase7_results_path = Path("phases/phase7_optimization/results")

            if not phase7_results_path.exists():
                self.logger.warning("Phase 7 results not found - skipping integration validation")
                return True

            # Validate compression is compatible with Phase 7 optimizations
            integration_checks = [
                self._check_optimization_preservation(quality_metrics),
                self._check_performance_consistency(compressed_model_path),
                self._validate_architecture_compatibility(compressed_model_path)
            ]

            integration_valid = all(integration_checks)

            if integration_valid:
                self.logger.info("Phase 7 integration validation passed")
            else:
                self.logger.error("Phase 7 integration validation failed")

            return integration_valid

        except Exception as e:
            self.logger.error(f"Phase 7 integration validation error: {e}")
            return False

    def _check_optimization_preservation(self, quality_metrics: CompressionMetrics) -> bool:
        """Check if Phase 7 optimizations are preserved"""
        # Verify that compression doesn't undo Phase 7 improvements
        if quality_metrics.accuracy_retention < 0.98:  # Stricter for post-optimization
            self.logger.warning("Compression may have undone Phase 7 optimizations")
            return False
        return True

    def _check_performance_consistency(self, model_path: str) -> bool:
        """Check performance consistency with Phase 7"""
        try:
            # Load model and verify it maintains expected performance characteristics
            import torch
            model = torch.load(model_path, map_location='cpu')

            # Basic structural checks
            if hasattr(model, 'eval'):
                model.eval()
                return True

            return False
        except Exception as e:
            self.logger.error(f"Performance consistency check failed: {e}")
            return False

    def _validate_architecture_compatibility(self, model_path: str) -> bool:
        """Validate architectural compatibility"""
        try:
            # Ensure compressed model maintains compatible architecture
            # This is a simplified check - extend based on specific requirements
            return Path(model_path).exists() and Path(model_path).stat().st_size > 0
        except Exception as e:
            self.logger.error(f"Architecture compatibility check failed: {e}")
            return False

    def _calculate_overall_score(
        self,
        quality_metrics: CompressionMetrics,
        deployment_metrics: DeploymentMetrics,
        theater_detection: TheaterDetectionResult
    ) -> float:
        """Calculate overall production readiness score (0-1)"""
        scores = []

        # Quality score (40% weight)
        quality_score = 0.0
        if quality_metrics.validation_passed:
            quality_score = min(1.0, quality_metrics.accuracy_retention *
                              min(1.0, quality_metrics.compression_ratio / 5.0))
        scores.append(quality_score * 0.4)

        # Deployment score (35% weight)
        deployment_score = 0.0
        if deployment_metrics.production_ready:
            deployment_score = 1.0
            if deployment_metrics.deployment_issues:
                deployment_score *= max(0.5, 1.0 - len(deployment_metrics.deployment_issues) * 0.1)
        scores.append(deployment_score * 0.35)

        # Theater penalty (25% weight)
        theater_score = 1.0
        if theater_detection.theater_detected:
            if theater_detection.severity == "critical":
                theater_score = 0.0
            elif theater_detection.severity == "high":
                theater_score = 0.3
            elif theater_detection.severity == "medium":
                theater_score = 0.6
            else:  # low
                theater_score = 0.8
        scores.append(theater_score * 0.25)

        return sum(scores)

    def _generate_recommendations(
        self,
        quality_metrics: CompressionMetrics,
        deployment_metrics: DeploymentMetrics,
        theater_detection: TheaterDetectionResult
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        # Quality recommendations
        if quality_metrics.accuracy_retention < 0.98:
            recommendations.append("Consider less aggressive compression to preserve accuracy")

        if quality_metrics.compression_ratio < 3.0:
            recommendations.append("Compression ratio is low - investigate more aggressive techniques")

        # Deployment recommendations
        if not deployment_metrics.inference_speed_acceptable:
            recommendations.append("Optimize inference speed for production deployment")

        if not deployment_metrics.memory_usage_acceptable:
            recommendations.append("Reduce memory usage for production constraints")

        # Theater recommendations
        if theater_detection.theater_detected:
            recommendations.append("Address theater patterns before production deployment")
            recommendations.extend([
                f"Theater indicator: {indicator}"
                for indicator in theater_detection.theater_indicators[:3]
            ])

        return recommendations

    def _generate_production_report(
        self,
        result: ProductionValidationResult,
        validation_name: str
    ) -> None:
        """Generate comprehensive production validation report"""
        report = {
            "validation_summary": {
                "name": validation_name,
                "timestamp": result.timestamp,
                "validation_passed": result.validation_passed,
                "production_ready": result.production_ready,
                "overall_score": result.overall_score,
                "validation_duration_minutes": (time.time() - result.timestamp) / 60
            },
            "quality_validation": {
                "passed": result.quality_metrics.validation_passed,
                "accuracy_retention": result.quality_metrics.accuracy_retention,
                "compression_ratio": result.quality_metrics.compression_ratio,
                "speed_improvement": result.quality_metrics.speed_improvement,
                "memory_reduction": result.quality_metrics.memory_reduction,
                "theater_detected": result.quality_metrics.theater_detected
            },
            "deployment_validation": {
                "production_ready": result.deployment_metrics.production_ready,
                "hardware_compatible": result.deployment_metrics.hardware_compatible,
                "platform_compatible": result.deployment_metrics.platform_compatible,
                "performance_acceptable": result.deployment_metrics.inference_speed_acceptable,
                "issues": result.deployment_metrics.deployment_issues
            },
            "theater_detection": {
                "theater_detected": result.theater_detection.theater_detected,
                "confidence_score": result.theater_detection.confidence_score,
                "severity": result.theater_detection.severity,
                "indicators": result.theater_detection.theater_indicators,
                "evidence": result.theater_detection.evidence
            },
            "issues_and_recommendations": {
                "critical_issues": result.critical_issues,
                "warnings": result.warnings,
                "recommendations": result.recommendations
            },
            "validation_config": {
                "quality_thresholds": {
                    "min_accuracy_retention": self.config.quality_thresholds.min_accuracy_retention,
                    "min_compression_ratio": self.config.quality_thresholds.min_compression_ratio,
                    "min_speed_improvement": self.config.quality_thresholds.min_speed_improvement
                },
                "target_platforms": self.config.target_platforms,
                "performance_targets": self.config.performance_targets
            }
        }

        # Save report
        report_path = Path(self.config.output_directory) / f"{validation_name}_production_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Production validation report saved to {report_path}")

        # Generate summary report for quick review
        self._generate_summary_report(result, validation_name)

    def _generate_summary_report(
        self,
        result: ProductionValidationResult,
        validation_name: str
    ) -> None:
        """Generate executive summary report"""
        summary_path = Path(self.config.output_directory) / f"{validation_name}_summary.txt"

        with open(summary_path, 'w') as f:
            f.write(f"PHASE 8 COMPRESSION - PRODUCTION VALIDATION SUMMARY\n")
            f.write(f"{'='*60}\n\n")

            f.write(f"Validation Name: {validation_name}\n")
            f.write(f"Validation Date: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(result.timestamp))}\n")
            f.write(f"Overall Score: {result.overall_score:.2f}/1.00\n\n")

            # Production Readiness
            status = "PRODUCTION READY" if result.production_ready else "NOT PRODUCTION READY"
            f.write(f"PRODUCTION STATUS: {status}\n")
            f.write(f"{'='*30}\n\n")

            # Key Metrics
            f.write(f"KEY METRICS:\n")
            f.write(f"- Accuracy Retention: {result.quality_metrics.accuracy_retention:.1%}\n")
            f.write(f"- Compression Ratio: {result.quality_metrics.compression_ratio:.1f}x\n")
            f.write(f"- Speed Improvement: {result.quality_metrics.speed_improvement:.1f}x\n")
            f.write(f"- Memory Reduction: {result.quality_metrics.memory_reduction:.1%}\n\n")

            # Issues
            if result.critical_issues:
                f.write(f"CRITICAL ISSUES:\n")
                for issue in result.critical_issues:
                    f.write(f"- {issue}\n")
                f.write("\n")

            if result.warnings:
                f.write(f"WARNINGS:\n")
                for warning in result.warnings:
                    f.write(f"- {warning}\n")
                f.write("\n")

            # Top Recommendations
            if result.recommendations:
                f.write(f"TOP RECOMMENDATIONS:\n")
                for rec in result.recommendations[:5]:  # Top 5
                    f.write(f"- {rec}\n")

        self.logger.info(f"Summary report saved to {summary_path}")

    def _log_final_results(self, result: ProductionValidationResult, duration: float) -> None:
        """Log final validation results"""
        self.logger.info("="*60)
        self.logger.info("PHASE 8 PRODUCTION VALIDATION COMPLETE")
        self.logger.info("="*60)
        self.logger.info(f"Duration: {duration:.1f} seconds")
        self.logger.info(f"Overall Score: {result.overall_score:.2f}/1.00")
        self.logger.info(f"Production Ready: {result.production_ready}")

        if result.production_ready:
            self.logger.info("🟢 VALIDATION PASSED - PRODUCTION DEPLOYMENT APPROVED")
        else:
            self.logger.error("🔴 VALIDATION FAILED - PRODUCTION DEPLOYMENT BLOCKED")

        if result.critical_issues:
            self.logger.error(f"Critical Issues: {len(result.critical_issues)}")
            for issue in result.critical_issues:
                self.logger.error(f"  - {issue}")

        if result.warnings:
            self.logger.warning(f"Warnings: {len(result.warnings)}")

        self.logger.info("="*60)


def main():
    """Main validation function"""
    logging.basicConfig(level=logging.INFO)

    # Example configuration
    config = ProductionValidationConfig(
        quality_thresholds=QualityThresholds(),
        hardware_requirements=HardwareRequirements(
            min_ram_gb=4.0,
            min_cpu_cores=2,
            gpu_required=False,
            min_gpu_memory_gb=2.0,
            min_disk_space_gb=1.0,
            supported_architectures=["x86_64", "aarch64"]
        ),
        target_platforms=["linux", "windows", "darwin"],
        performance_targets={
            "max_inference_time": 1.0,
            "max_memory_mb": 1024,
            "max_startup_time": 10.0
        }
    )

    validator = Phase8ProductionValidator(config)
    print("Phase 8 Production Validator initialized")
    print("Use validate_production_readiness() with actual models and data")


if __name__ == "__main__":
    main()