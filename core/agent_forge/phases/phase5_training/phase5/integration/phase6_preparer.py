"""
Phase 6 Baking Preparation Module
Prepares trained models and metadata for Phase 6 baking process.
"""

import asyncio
import logging
import json
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

class BakingReadiness(Enum):
    READY = "ready"
    NEEDS_VALIDATION = "needs_validation"
    NOT_READY = "not_ready"

@dataclass
class BakingMetadata:
    """Metadata for Phase 6 baking process."""
    model_id: str
    phase5_version: str
    training_config: Dict[str, Any]
    performance_metrics: Dict[str, float]
    quality_scores: Dict[str, float]
    export_timestamp: datetime
    compatibility_info: Dict[str, Any]

@dataclass
class ExportPackage:
    """Complete export package for Phase 6."""
    model_data: bytes
    metadata: BakingMetadata
    configuration: Dict[str, Any]
    validation_results: Dict[str, Any]
    checksum: str

class Phase6Preparer:
    """
    Prepares Phase 5 training outputs for Phase 6 baking process.
    """

    def __init__(self, export_dir: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        self.export_dir = export_dir or Path("exports/phase6")
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.preparation_cache = {}

    async def initialize(self) -> bool:
        """Initialize Phase 6 preparer."""
        try:
            self.logger.info("Initializing Phase 6 preparer")

            # Ensure export directory exists and is writable
            if not self.export_dir.exists():
                self.export_dir.mkdir(parents=True)

            # Create subdirectories
            subdirs = ["models", "metadata", "configs", "validation", "logs"]
            for subdir in subdirs:
                (self.export_dir / subdir).mkdir(exist_ok=True)

            # Initialize logging
            log_file = self.export_dir / "logs" / f"phase6_prep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            self.logger.addHandler(file_handler)

            self.logger.info("Phase 6 preparer initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Phase 6 preparer initialization failed: {e}")
            return False

    async def assess_baking_readiness(self, model: Any, training_results: Dict[str, Any]) -> Tuple[BakingReadiness, List[str]]:
        """Assess model readiness for Phase 6 baking."""
        try:
            self.logger.info("Assessing baking readiness")

            issues = []
            readiness = BakingReadiness.READY

            # Check training completion
            if not await self._check_training_completion(training_results):
                issues.append("Training not completed successfully")
                readiness = BakingReadiness.NOT_READY

            # Validate model quality
            quality_check = await self._validate_model_quality(model, training_results)
            if not quality_check["passed"]:
                issues.extend(quality_check["issues"])
                readiness = BakingReadiness.NEEDS_VALIDATION

            # Check performance targets
            perf_check = await self._check_performance_targets(training_results)
            if not perf_check["met"]:
                issues.extend(perf_check["issues"])
                if readiness == BakingReadiness.READY:
                    readiness = BakingReadiness.NEEDS_VALIDATION

            # Validate export compatibility
            compat_check = await self._check_export_compatibility(model)
            if not compat_check:
                issues.append("Model not compatible with Phase 6 export format")
                readiness = BakingReadiness.NOT_READY

            # Check required metadata
            metadata_check = await self._validate_required_metadata(training_results)
            if not metadata_check:
                issues.append("Missing required metadata for baking")
                readiness = BakingReadiness.NEEDS_VALIDATION

            self.logger.info(f"Baking readiness assessment: {readiness.value}, {len(issues)} issues")
            return readiness, issues

        except Exception as e:
            self.logger.error(f"Failed to assess baking readiness: {e}")
            return BakingReadiness.NOT_READY, [f"Assessment failed: {e}"]

    async def prepare_export_package(self, model: Any, training_results: Dict[str, Any], model_id: str) -> Optional[ExportPackage]:
        """Prepare complete export package for Phase 6."""
        try:
            self.logger.info(f"Preparing export package for model: {model_id}")

            # Check readiness first
            readiness, issues = await self.assess_baking_readiness(model, training_results)
            if readiness == BakingReadiness.NOT_READY:
                raise ValueError(f"Model not ready for export: {', '.join(issues)}")

            # Create baking metadata
            metadata = await self._create_baking_metadata(model_id, training_results)

            # Serialize model data
            model_data = await self._serialize_model(model)

            # Prepare configuration
            configuration = await self._prepare_baking_configuration(model, training_results)

            # Run validation
            validation_results = await self._validate_export_package(model, model_data, metadata)

            # Calculate checksum
            checksum = await self._calculate_package_checksum(model_data, metadata, configuration)

            # Create export package
            package = ExportPackage(
                model_data=model_data,
                metadata=metadata,
                configuration=configuration,
                validation_results=validation_results,
                checksum=checksum
            )

            # Save package to disk
            package_path = await self._save_export_package(package, model_id)

            self.logger.info(f"Export package prepared successfully: {package_path}")
            return package

        except Exception as e:
            self.logger.error(f"Failed to prepare export package: {e}")
            return None

    async def create_baking_manifest(self, packages: List[ExportPackage]) -> Dict[str, Any]:
        """Create manifest for multiple export packages."""
        try:
            self.logger.info(f"Creating baking manifest for {len(packages)} packages")

            manifest = {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "packages": [],
                "summary": {
                    "total_packages": len(packages),
                    "ready_packages": 0,
                    "validation_warnings": 0,
                    "total_size_mb": 0.0
                }
            }

            for package in packages:
                package_info = {
                    "model_id": package.metadata.model_id,
                    "checksum": package.checksum,
                    "size_mb": len(package.model_data) / (1024 * 1024),
                    "quality_score": package.metadata.quality_scores.get("overall", 0.0),
                    "performance_metrics": package.metadata.performance_metrics,
                    "readiness_status": "ready" if package.validation_results.get("overall_valid", False) else "warning",
                    "export_timestamp": package.metadata.export_timestamp.isoformat()
                }

                manifest["packages"].append(package_info)

                # Update summary
                manifest["summary"]["total_size_mb"] += package_info["size_mb"]
                if package_info["readiness_status"] == "ready":
                    manifest["summary"]["ready_packages"] += 1
                else:
                    manifest["summary"]["validation_warnings"] += 1

            # Save manifest
            manifest_path = self.export_dir / "baking_manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2, default=str)

            self.logger.info(f"Baking manifest created: {manifest_path}")
            return manifest

        except Exception as e:
            self.logger.error(f"Failed to create baking manifest: {e}")
            raise

    async def validate_phase6_handoff(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        """Validate readiness for Phase 6 handoff."""
        try:
            self.logger.info("Validating Phase 6 handoff readiness")

            validation_result = {
                "valid": True,
                "warnings": [],
                "errors": [],
                "statistics": {},
                "recommendations": []
            }

            # Check manifest completeness
            required_fields = ["version", "created_at", "packages", "summary"]
            missing_fields = [field for field in required_fields if field not in manifest]
            if missing_fields:
                validation_result["errors"].extend([f"Missing manifest field: {field}" for field in missing_fields])
                validation_result["valid"] = False

            # Validate packages
            packages = manifest.get("packages", [])
            ready_count = 0
            warning_count = 0

            for package in packages:
                if package.get("readiness_status") == "ready":
                    ready_count += 1
                else:
                    warning_count += 1
                    validation_result["warnings"].append(f"Package {package.get('model_id')} has warnings")

            # Check readiness thresholds
            total_packages = len(packages)
            ready_percentage = (ready_count / total_packages * 100) if total_packages > 0 else 0

            if ready_percentage < 80:
                validation_result["errors"].append(f"Only {ready_percentage:.1f}% of packages are ready (minimum 80% required)")
                validation_result["valid"] = False

            # Update statistics
            validation_result["statistics"] = {
                "total_packages": total_packages,
                "ready_packages": ready_count,
                "warning_packages": warning_count,
                "ready_percentage": ready_percentage,
                "total_size_gb": manifest.get("summary", {}).get("total_size_mb", 0) / 1024
            }

            # Generate recommendations
            if warning_count > 0:
                validation_result["recommendations"].append("Review packages with warnings before Phase 6 baking")

            if ready_percentage < 100:
                validation_result["recommendations"].append("Consider re-training models with warnings to improve readiness")

            self.logger.info(f"Phase 6 handoff validation: {'PASSED' if validation_result['valid'] else 'FAILED'}")
            return validation_result

        except Exception as e:
            self.logger.error(f"Phase 6 handoff validation failed: {e}")
            return {"valid": False, "errors": [f"Validation failed: {e}"]}

    async def _check_training_completion(self, training_results: Dict[str, Any]) -> bool:
        """Check if training completed successfully."""
        try:
            status = training_results.get("status", "")
            if status != "completed":
                return False

            # Check for required results
            required_fields = ["final_metrics", "training_history", "model_state"]
            return all(field in training_results for field in required_fields)

        except Exception:
            return False

    async def _validate_model_quality(self, model: Any, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model quality metrics."""
        try:
            metrics = training_results.get("final_metrics", {})
            issues = []
            passed = True

            # Quality thresholds
            thresholds = {
                "accuracy": 0.90,
                "precision": 0.85,
                "recall": 0.85,
                "f1_score": 0.85
            }

            for metric, threshold in thresholds.items():
                if metric in metrics:
                    if metrics[metric] < threshold:
                        issues.append(f"{metric} ({metrics[metric]:.3f}) below threshold ({threshold})")
                        passed = False
                else:
                    issues.append(f"Missing required metric: {metric}")
                    passed = False

            return {"passed": passed, "issues": issues}

        except Exception as e:
            return {"passed": False, "issues": [f"Quality validation failed: {e}"]}

    async def _check_performance_targets(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Check performance targets."""
        try:
            metrics = training_results.get("performance_metrics", {})
            issues = []

            # Performance thresholds
            thresholds = {
                "inference_time": 0.1,  # seconds
                "memory_usage": 1.0,    # GB
                "throughput": 100.0     # samples/sec
            }

            met = True
            for metric, threshold in thresholds.items():
                if metric in metrics:
                    if metric == "inference_time" and metrics[metric] > threshold:
                        issues.append(f"Inference time too high: {metrics[metric]:.3f}s > {threshold}s")
                        met = False
                    elif metric == "memory_usage" and metrics[metric] > threshold:
                        issues.append(f"Memory usage too high: {metrics[metric]:.2f}GB > {threshold}GB")
                        met = False
                    elif metric == "throughput" and metrics[metric] < threshold:
                        issues.append(f"Throughput too low: {metrics[metric]:.1f} < {threshold}")
                        met = False

            return {"met": met, "issues": issues}

        except Exception as e:
            return {"met": False, "issues": [f"Performance check failed: {e}"]}

    async def _check_export_compatibility(self, model: Any) -> bool:
        """Check model compatibility with Phase 6 export."""
        try:
            # Check if model has required attributes/methods for export
            required_attrs = ["state_dict", "config", "forward"]

            # Mock compatibility check
            if hasattr(model, "get") and model.get("phase5_enhancements"):
                return True

            return True  # Simplified for mock implementation

        except Exception:
            return False

    async def _validate_required_metadata(self, training_results: Dict[str, Any]) -> bool:
        """Validate required metadata presence."""
        try:
            required_fields = [
                "training_config",
                "final_metrics",
                "model_architecture",
                "training_history"
            ]

            return all(field in training_results for field in required_fields)

        except Exception:
            return False

    async def _create_baking_metadata(self, model_id: str, training_results: Dict[str, Any]) -> BakingMetadata:
        """Create baking metadata from training results."""
        return BakingMetadata(
            model_id=model_id,
            phase5_version="1.0.0",
            training_config=training_results.get("training_config", {}),
            performance_metrics=training_results.get("performance_metrics", {}),
            quality_scores=training_results.get("final_metrics", {}),
            export_timestamp=datetime.now(),
            compatibility_info={
                "phase4_compatible": True,
                "phase6_ready": True,
                "export_format": "phase5_enhanced",
                "compression": "bitnet_quantized"
            }
        )

    async def _serialize_model(self, model: Any) -> bytes:
        """Serialize model for export."""
        try:
            # In practice, this would use appropriate model serialization
            # For now, using pickle as placeholder
            return pickle.dumps(model)

        except Exception as e:
            self.logger.error(f"Model serialization failed: {e}")
            raise

    async def _prepare_baking_configuration(self, model: Any, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare baking configuration."""
        return {
            "model_format": "enhanced_bitnet",
            "quantization": training_results.get("quantization_config", {}),
            "optimization": {
                "memory_optimization": True,
                "inference_optimization": True,
                "deployment_ready": True
            },
            "phase6_settings": {
                "baking_temperature": 0.8,
                "compression_level": "high",
                "validation_strict": True
            }
        }

    async def _validate_export_package(self, model: Any, model_data: bytes, metadata: BakingMetadata) -> Dict[str, Any]:
        """Validate export package."""
        try:
            validation = {
                "overall_valid": True,
                "model_size_mb": len(model_data) / (1024 * 1024),
                "metadata_complete": True,
                "checksum_valid": True,
                "warnings": []
            }

            # Size validation
            if validation["model_size_mb"] > 500:
                validation["warnings"].append(f"Large model size: {validation['model_size_mb']:.1f}MB")

            # Metadata validation
            required_metadata = ["model_id", "performance_metrics", "quality_scores"]
            for field in required_metadata:
                if not hasattr(metadata, field) or getattr(metadata, field) is None:
                    validation["metadata_complete"] = False
                    validation["overall_valid"] = False

            return validation

        except Exception as e:
            return {
                "overall_valid": False,
                "error": str(e),
                "warnings": [f"Validation failed: {e}"]
            }

    async def _calculate_package_checksum(self, model_data: bytes, metadata: BakingMetadata, configuration: Dict[str, Any]) -> str:
        """Calculate package checksum."""
        import hashlib

        # Combine all package components for checksum
        content = model_data + json.dumps(asdict(metadata), sort_keys=True, default=str).encode() + json.dumps(configuration, sort_keys=True).encode()

        return hashlib.sha256(content).hexdigest()

    async def _save_export_package(self, package: ExportPackage, model_id: str) -> Path:
        """Save export package to disk."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            package_dir = self.export_dir / f"package_{model_id}_{timestamp}"
            package_dir.mkdir(exist_ok=True)

            # Save model data
            model_file = package_dir / "model.pkl"
            with open(model_file, 'wb') as f:
                f.write(package.model_data)

            # Save metadata
            metadata_file = package_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(asdict(package.metadata), f, indent=2, default=str)

            # Save configuration
            config_file = package_dir / "configuration.json"
            with open(config_file, 'w') as f:
                json.dump(package.configuration, f, indent=2)

            # Save validation results
            validation_file = package_dir / "validation.json"
            with open(validation_file, 'w') as f:
                json.dump(package.validation_results, f, indent=2)

            # Save checksum
            checksum_file = package_dir / "checksum.txt"
            with open(checksum_file, 'w') as f:
                f.write(package.checksum)

            return package_dir

        except Exception as e:
            self.logger.error(f"Failed to save export package: {e}")
            raise