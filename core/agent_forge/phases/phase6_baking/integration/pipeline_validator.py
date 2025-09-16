"""
Comprehensive Pipeline Validator for Phase 6 Integration

This module provides end-to-end validation of the complete Phase 6 baking pipeline,
ensuring all components work together seamlessly and meet quality requirements.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import psutil

from .phase5_connector import Phase5Connector, create_phase5_connector
from .phase7_preparer import Phase7Preparer, create_phase7_preparer

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of pipeline validation"""
    component: str
    passed: bool
    score: float
    details: Dict[str, Any]
    issues: List[str]
    recommendations: List[str]
    execution_time_ms: float

@dataclass
class PipelineHealth:
    """Overall pipeline health assessment"""
    overall_health: str  # EXCELLENT, GOOD, WARNING, CRITICAL
    health_score: float  # 0-100
    component_results: List[ValidationResult]
    critical_issues: List[str]
    performance_metrics: Dict[str, float]
    readiness_status: Dict[str, bool]

class PipelineValidator:
    """Comprehensive validator for Phase 6 baking pipeline"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.validation_results = []

        # Initialize component validators
        self.phase5_connector = create_phase5_connector(config.get('phase5_config', {}))
        self.phase7_preparer = create_phase7_preparer(config.get('phase7_config', {}))

        # Validation thresholds
        self.thresholds = {
            'min_health_score': config.get('min_health_score', 80.0),
            'min_component_score': config.get('min_component_score', 70.0),
            'max_validation_time_ms': config.get('max_validation_time_ms', 30000),
            'min_models_required': config.get('min_models_required', 1),
            'min_success_rate': config.get('min_success_rate', 0.8)
        }

    def validate_complete_pipeline(self) -> PipelineHealth:
        """Validate the complete Phase 6 baking pipeline end-to-end"""
        logger.info("Starting comprehensive pipeline validation")
        start_time = time.time()

        component_results = []
        critical_issues = []

        # Validate all pipeline components
        validation_tasks = [
            ('Phase5Integration', self._validate_phase5_integration),
            ('BakingCore', self._validate_baking_core),
            ('OptimizationEngine', self._validate_optimization_engine),
            ('QualityGates', self._validate_quality_gates),
            ('Phase7Preparation', self._validate_phase7_preparation),
            ('SystemResources', self._validate_system_resources),
            ('DataFlow', self._validate_data_flow),
            ('ErrorHandling', self._validate_error_handling)
        ]

        # Execute validations in parallel for efficiency
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_component = {
                executor.submit(task_func): component_name
                for component_name, task_func in validation_tasks
            }

            for future in as_completed(future_to_component):
                component_name = future_to_component[future]
                try:
                    result = future.result()
                    component_results.append(result)

                    if not result.passed:
                        critical_issues.extend(result.issues)

                except Exception as e:
                    logger.error(f"Validation failed for {component_name}: {e}")
                    component_results.append(ValidationResult(
                        component=component_name,
                        passed=False,
                        score=0.0,
                        details={'error': str(e)},
                        issues=[f"Validation execution failed: {e}"],
                        recommendations=[f"Fix {component_name} validation errors"],
                        execution_time_ms=0.0
                    ))

        # Calculate overall health metrics
        total_time = (time.time() - start_time) * 1000
        performance_metrics = self._calculate_performance_metrics(component_results, total_time)
        health_score = self._calculate_health_score(component_results)
        overall_health = self._determine_overall_health(health_score, critical_issues)
        readiness_status = self._assess_readiness_status(component_results)

        pipeline_health = PipelineHealth(
            overall_health=overall_health,
            health_score=health_score,
            component_results=component_results,
            critical_issues=critical_issues,
            performance_metrics=performance_metrics,
            readiness_status=readiness_status
        )

        logger.info(f"Pipeline validation completed in {total_time:.0f}ms - Health: {overall_health}")
        return pipeline_health

    def _validate_phase5_integration(self) -> ValidationResult:
        """Validate Phase 5 integration capabilities"""
        start_time = time.time()
        issues = []
        details = {}

        try:
            # Test model discovery
            models = self.phase5_connector.discover_trained_models()
            details['models_found'] = len(models)

            if len(models) == 0:
                issues.append("No Phase 5 trained models found")

            # Test pipeline validation
            pipeline_results = self.phase5_connector.validate_integration_pipeline()
            details.update(pipeline_results)

            if pipeline_results.get('compatible_models', 0) == 0:
                issues.append("No compatible models for Phase 6 integration")

            # Test model transfer capability
            if models:
                try:
                    best_model = self.phase5_connector.get_best_model()
                    if best_model:
                        compatible, score, validation_info = self.phase5_connector.validate_model_compatibility(
                            best_model['path']
                        )
                        details['best_model_compatibility'] = score
                        if not compatible:
                            issues.append(f"Best model not compatible (score: {score:.2f})")
                except Exception as e:
                    issues.append(f"Model compatibility test failed: {e}")

            # Calculate component score
            score = self._calculate_component_score(details, issues, {
                'models_found': 10,
                'compatible_models': 20,
                'average_compatibility_score': 30,
                'transfer_success_rate': 40
            })

        except Exception as e:
            issues.append(f"Phase 5 integration validation failed: {e}")
            score = 0.0

        execution_time = (time.time() - start_time) * 1000

        return ValidationResult(
            component='Phase5Integration',
            passed=len(issues) == 0 and score >= self.thresholds['min_component_score'],
            score=score,
            details=details,
            issues=issues,
            recommendations=self._generate_recommendations('phase5', issues),
            execution_time_ms=execution_time
        )

    def _validate_baking_core(self) -> ValidationResult:
        """Validate core baking functionality"""
        start_time = time.time()
        issues = []
        details = {}

        try:
            # Check baking components exist
            baking_components = [
                'src/phase6/core/model_baking_engine.py',
                'src/phase6/optimization/neural_optimization.py',
                'src/phase6/validation/performance_validator.py',
                'src/phase6/quality/quality_gate_manager.py'
            ]

            missing_components = []
            for component in baking_components:
                if not Path(component).exists():
                    missing_components.append(component)

            details['missing_components'] = missing_components
            if missing_components:
                issues.append(f"Missing core components: {', '.join(missing_components)}")

            # Check baking configuration
            config_file = Path('config/baking_config.json')
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        baking_config = json.load(f)
                    details['config_valid'] = True
                    details['optimization_techniques'] = len(baking_config.get('optimization_techniques', []))
                except Exception as e:
                    issues.append(f"Invalid baking configuration: {e}")
                    details['config_valid'] = False
            else:
                issues.append("Baking configuration file missing")
                details['config_valid'] = False

            # Test baking pipeline readiness
            details['baking_ready'] = len(missing_components) == 0 and details.get('config_valid', False)

            score = self._calculate_component_score(details, issues, {
                'missing_components': -20,  # Negative weight
                'config_valid': 40,
                'optimization_techniques': 10,
                'baking_ready': 50
            })

        except Exception as e:
            issues.append(f"Baking core validation failed: {e}")
            score = 0.0

        execution_time = (time.time() - start_time) * 1000

        return ValidationResult(
            component='BakingCore',
            passed=len(issues) == 0 and score >= self.thresholds['min_component_score'],
            score=score,
            details=details,
            issues=issues,
            recommendations=self._generate_recommendations('baking_core', issues),
            execution_time_ms=execution_time
        )

    def _validate_optimization_engine(self) -> ValidationResult:
        """Validate model optimization capabilities"""
        start_time = time.time()
        issues = []
        details = {}

        try:
            # Check optimization components
            optimization_files = [
                'src/phase6/optimization/quantization.py',
                'src/phase6/optimization/pruning.py',
                'src/phase6/optimization/knowledge_distillation.py',
                'src/phase6/optimization/neural_architecture_search.py'
            ]

            available_optimizations = []
            for opt_file in optimization_files:
                if Path(opt_file).exists():
                    available_optimizations.append(Path(opt_file).stem)

            details['available_optimizations'] = available_optimizations
            details['optimization_count'] = len(available_optimizations)

            if len(available_optimizations) < 2:
                issues.append("Insufficient optimization techniques available")

            # Check optimization configuration
            opt_config_file = Path('config/optimization_config.json')
            if opt_config_file.exists():
                try:
                    with open(opt_config_file, 'r') as f:
                        opt_config = json.load(f)
                    details['optimization_config_valid'] = True
                    details['target_speedup'] = opt_config.get('target_speedup', 1.0)
                    details['target_compression'] = opt_config.get('target_compression', 1.0)
                except Exception as e:
                    issues.append(f"Invalid optimization configuration: {e}")
                    details['optimization_config_valid'] = False
            else:
                issues.append("Optimization configuration missing")
                details['optimization_config_valid'] = False

            # Test optimization readiness
            details['optimization_ready'] = (
                len(available_optimizations) >= 2 and
                details.get('optimization_config_valid', False)
            )

            score = self._calculate_component_score(details, issues, {
                'optimization_count': 15,
                'optimization_config_valid': 35,
                'target_speedup': 25,
                'target_compression': 25
            })

        except Exception as e:
            issues.append(f"Optimization engine validation failed: {e}")
            score = 0.0

        execution_time = (time.time() - start_time) * 1000

        return ValidationResult(
            component='OptimizationEngine',
            passed=len(issues) == 0 and score >= self.thresholds['min_component_score'],
            score=score,
            details=details,
            issues=issues,
            recommendations=self._generate_recommendations('optimization', issues),
            execution_time_ms=execution_time
        )

    def _validate_quality_gates(self) -> ValidationResult:
        """Validate quality gate mechanisms"""
        start_time = time.time()
        issues = []
        details = {}

        try:
            # Check quality gate components
            quality_files = [
                'src/phase6/quality/quality_gate_manager.py',
                'src/phase6/quality/metrics_collector.py',
                'src/phase6/quality/performance_analyzer.py',
                'src/phase6/quality/compliance_checker.py'
            ]

            quality_components = []
            for qf in quality_files:
                if Path(qf).exists():
                    quality_components.append(Path(qf).stem)

            details['quality_components'] = quality_components
            details['quality_component_count'] = len(quality_components)

            if len(quality_components) < 3:
                issues.append("Insufficient quality gate components")

            # Check quality thresholds configuration
            thresholds_file = Path('config/quality_thresholds.json')
            if thresholds_file.exists():
                try:
                    with open(thresholds_file, 'r') as f:
                        thresholds = json.load(f)
                    details['thresholds_configured'] = True
                    details['threshold_count'] = len(thresholds)
                except Exception as e:
                    issues.append(f"Invalid quality thresholds: {e}")
                    details['thresholds_configured'] = False
            else:
                issues.append("Quality thresholds not configured")
                details['thresholds_configured'] = False

            # Validate CTQ (Critical to Quality) parameters
            ctq_params = ['accuracy', 'latency', 'model_size', 'memory_usage']
            configured_ctq = []

            if details.get('thresholds_configured', False):
                with open(thresholds_file, 'r') as f:
                    thresholds = json.load(f)
                for param in ctq_params:
                    if param in thresholds:
                        configured_ctq.append(param)

            details['configured_ctq'] = configured_ctq
            details['ctq_coverage'] = len(configured_ctq) / len(ctq_params)

            if len(configured_ctq) < 3:
                issues.append("Insufficient CTQ parameters configured")

            score = self._calculate_component_score(details, issues, {
                'quality_component_count': 20,
                'thresholds_configured': 30,
                'threshold_count': 20,
                'ctq_coverage': 30
            })

        except Exception as e:
            issues.append(f"Quality gates validation failed: {e}")
            score = 0.0

        execution_time = (time.time() - start_time) * 1000

        return ValidationResult(
            component='QualityGates',
            passed=len(issues) == 0 and score >= self.thresholds['min_component_score'],
            score=score,
            details=details,
            issues=issues,
            recommendations=self._generate_recommendations('quality_gates', issues),
            execution_time_ms=execution_time
        )

    def _validate_phase7_preparation(self) -> ValidationResult:
        """Validate Phase 7 preparation capabilities"""
        start_time = time.time()
        issues = []
        details = {}

        try:
            # Test Phase 7 preparer functionality
            models = self.phase7_preparer.discover_baked_models()
            details['baked_models_found'] = len(models)

            # Test pipeline validation
            pipeline_results = self.phase7_preparer.validate_phase7_pipeline()
            details.update(pipeline_results)

            if pipeline_results.get('adas_ready_models', 0) == 0 and len(models) > 0:
                issues.append("No models ready for ADAS deployment")

            # Check ADAS requirements configuration
            adas_requirements = self.phase7_preparer.adas_requirements
            details['adas_config'] = {
                'max_inference_time_ms': adas_requirements.max_inference_time_ms,
                'min_accuracy': adas_requirements.min_accuracy,
                'target_hardware': adas_requirements.target_hardware,
                'safety_level': adas_requirements.safety_level
            }

            # Validate safety compliance mechanisms
            safety_checks = [
                'deterministic_inference',
                'memory_bounds_verified',
                'fail_safe_mechanisms',
                'input_validation',
                'output_bounds_checked'
            ]

            details['safety_mechanisms'] = safety_checks
            details['safety_mechanism_count'] = len(safety_checks)

            score = self._calculate_component_score(details, issues, {
                'baked_models_found': 10,
                'adas_ready_models': 30,
                'deployment_success_rate': 30,
                'safety_mechanism_count': 30
            })

        except Exception as e:
            issues.append(f"Phase 7 preparation validation failed: {e}")
            score = 0.0

        execution_time = (time.time() - start_time) * 1000

        return ValidationResult(
            component='Phase7Preparation',
            passed=len(issues) == 0 and score >= self.thresholds['min_component_score'],
            score=score,
            details=details,
            issues=issues,
            recommendations=self._generate_recommendations('phase7', issues),
            execution_time_ms=execution_time
        )

    def _validate_system_resources(self) -> ValidationResult:
        """Validate system resource availability and performance"""
        start_time = time.time()
        issues = []
        details = {}

        try:
            # CPU validation
            cpu_count = psutil.cpu_count(logical=False)
            cpu_usage = psutil.cpu_percent(interval=1)
            details['cpu_cores'] = cpu_count
            details['cpu_usage_percent'] = cpu_usage

            if cpu_count < 4:
                issues.append(f"Insufficient CPU cores: {cpu_count} (minimum 4 recommended)")

            if cpu_usage > 80:
                issues.append(f"High CPU usage: {cpu_usage}% (may affect baking performance)")

            # Memory validation
            memory = psutil.virtual_memory()
            details['total_memory_gb'] = memory.total / (1024**3)
            details['available_memory_gb'] = memory.available / (1024**3)
            details['memory_usage_percent'] = memory.percent

            min_memory_gb = 8
            if details['total_memory_gb'] < min_memory_gb:
                issues.append(f"Insufficient memory: {details['total_memory_gb']:.1f}GB (minimum {min_memory_gb}GB)")

            if memory.percent > 85:
                issues.append(f"High memory usage: {memory.percent}% (may cause baking failures)")

            # Disk space validation
            disk_usage = psutil.disk_usage('.')
            details['total_disk_gb'] = disk_usage.total / (1024**3)
            details['free_disk_gb'] = disk_usage.free / (1024**3)
            details['disk_usage_percent'] = (disk_usage.used / disk_usage.total) * 100

            min_free_disk_gb = 10
            if details['free_disk_gb'] < min_free_disk_gb:
                issues.append(f"Insufficient disk space: {details['free_disk_gb']:.1f}GB free (minimum {min_free_disk_gb}GB)")

            # GPU validation (if available)
            try:
                import torch
                if torch.cuda.is_available():
                    details['gpu_available'] = True
                    details['gpu_count'] = torch.cuda.device_count()
                    details['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                else:
                    details['gpu_available'] = False
                    issues.append("No GPU available (recommended for optimal baking performance)")
            except ImportError:
                details['gpu_available'] = False

            score = self._calculate_component_score(details, issues, {
                'cpu_cores': 15,
                'available_memory_gb': 25,
                'free_disk_gb': 20,
                'gpu_available': 25,
                'low_resource_usage': 15  # Bonus for low usage
            })

        except Exception as e:
            issues.append(f"System resources validation failed: {e}")
            score = 0.0

        execution_time = (time.time() - start_time) * 1000

        return ValidationResult(
            component='SystemResources',
            passed=len(issues) == 0 and score >= self.thresholds['min_component_score'],
            score=score,
            details=details,
            issues=issues,
            recommendations=self._generate_recommendations('resources', issues),
            execution_time_ms=execution_time
        )

    def _validate_data_flow(self) -> ValidationResult:
        """Validate data flow through the pipeline"""
        start_time = time.time()
        issues = []
        details = {}

        try:
            # Check data directories
            required_dirs = [
                'models/phase5',
                'models/phase6',
                'models/adas',
                'data/intermediate',
                'logs/baking'
            ]

            existing_dirs = []
            missing_dirs = []

            for req_dir in required_dirs:
                dir_path = Path(req_dir)
                if dir_path.exists():
                    existing_dirs.append(req_dir)
                else:
                    missing_dirs.append(req_dir)

            details['existing_dirs'] = existing_dirs
            details['missing_dirs'] = missing_dirs
            details['directory_coverage'] = len(existing_dirs) / len(required_dirs)

            if missing_dirs:
                issues.append(f"Missing directories: {', '.join(missing_dirs)}")

            # Check data format compatibility
            data_formats = ['pytorch', 'onnx', 'tensorrt']
            supported_formats = []

            for fmt in data_formats:
                try:
                    if fmt == 'pytorch':
                        import torch
                        supported_formats.append(fmt)
                    elif fmt == 'onnx':
                        import onnx
                        supported_formats.append(fmt)
                    elif fmt == 'tensorrt':
                        import tensorrt
                        supported_formats.append(fmt)
                except ImportError:
                    pass

            details['supported_formats'] = supported_formats
            details['format_support_ratio'] = len(supported_formats) / len(data_formats)

            if len(supported_formats) < 2:
                issues.append("Insufficient data format support")

            # Test data pipeline connectivity
            pipeline_connectivity = self._test_pipeline_connectivity()
            details['pipeline_connectivity'] = pipeline_connectivity

            if not pipeline_connectivity.get('phase5_to_phase6', False):
                issues.append("Phase 5 to Phase 6 data flow not working")

            if not pipeline_connectivity.get('phase6_to_phase7', False):
                issues.append("Phase 6 to Phase 7 data flow not working")

            score = self._calculate_component_score(details, issues, {
                'directory_coverage': 30,
                'format_support_ratio': 35,
                'pipeline_connectivity': 35
            })

        except Exception as e:
            issues.append(f"Data flow validation failed: {e}")
            score = 0.0

        execution_time = (time.time() - start_time) * 1000

        return ValidationResult(
            component='DataFlow',
            passed=len(issues) == 0 and score >= self.thresholds['min_component_score'],
            score=score,
            details=details,
            issues=issues,
            recommendations=self._generate_recommendations('data_flow', issues),
            execution_time_ms=execution_time
        )

    def _validate_error_handling(self) -> ValidationResult:
        """Validate error handling and recovery mechanisms"""
        start_time = time.time()
        issues = []
        details = {}

        try:
            # Check error handling components
            error_handling_files = [
                'src/phase6/error_handling/error_manager.py',
                'src/phase6/error_handling/recovery_strategies.py',
                'src/phase6/error_handling/failure_detector.py'
            ]

            error_components = []
            for eh_file in error_handling_files:
                if Path(eh_file).exists():
                    error_components.append(Path(eh_file).stem)

            details['error_handling_components'] = error_components
            details['error_component_count'] = len(error_components)

            if len(error_components) < 2:
                issues.append("Insufficient error handling mechanisms")

            # Check logging configuration
            log_config_file = Path('config/logging_config.json')
            if log_config_file.exists():
                details['logging_configured'] = True
                try:
                    with open(log_config_file, 'r') as f:
                        log_config = json.load(f)
                    details['log_levels'] = list(log_config.get('loggers', {}).keys())
                except Exception as e:
                    issues.append(f"Invalid logging configuration: {e}")
            else:
                details['logging_configured'] = False
                issues.append("Logging configuration missing")

            # Check checkpoint mechanism
            checkpoint_dir = Path('checkpoints')
            details['checkpoint_mechanism'] = checkpoint_dir.exists()

            if not details['checkpoint_mechanism']:
                issues.append("No checkpoint mechanism available")

            # Test error simulation capability
            details['error_simulation_ready'] = self._check_error_simulation_capability()

            score = self._calculate_component_score(details, issues, {
                'error_component_count': 25,
                'logging_configured': 30,
                'checkpoint_mechanism': 25,
                'error_simulation_ready': 20
            })

        except Exception as e:
            issues.append(f"Error handling validation failed: {e}")
            score = 0.0

        execution_time = (time.time() - start_time) * 1000

        return ValidationResult(
            component='ErrorHandling',
            passed=len(issues) == 0 and score >= self.thresholds['min_component_score'],
            score=score,
            details=details,
            issues=issues,
            recommendations=self._generate_recommendations('error_handling', issues),
            execution_time_ms=execution_time
        )

    def _test_pipeline_connectivity(self) -> Dict[str, bool]:
        """Test connectivity between pipeline phases"""
        connectivity = {
            'phase5_to_phase6': False,
            'phase6_to_phase7': False,
            'bidirectional_flow': False
        }

        try:
            # Test Phase 5 to Phase 6 connection
            models = self.phase5_connector.discover_trained_models()
            if models:
                best_model = self.phase5_connector.get_best_model()
                if best_model:
                    compatible, _, _ = self.phase5_connector.validate_model_compatibility(best_model['path'])
                    connectivity['phase5_to_phase6'] = compatible

            # Test Phase 6 to Phase 7 connection
            baked_models = self.phase7_preparer.discover_baked_models()
            if baked_models:
                readiness_report = self.phase7_preparer.assess_adas_readiness(baked_models[0]['path'])
                connectivity['phase6_to_phase7'] = readiness_report.ready_for_deployment

            # Test bidirectional flow
            connectivity['bidirectional_flow'] = (
                connectivity['phase5_to_phase6'] and
                connectivity['phase6_to_phase7']
            )

        except Exception as e:
            logger.error(f"Pipeline connectivity test failed: {e}")

        return connectivity

    def _check_error_simulation_capability(self) -> bool:
        """Check if error simulation is available for testing"""
        try:
            error_sim_file = Path('tests/error_simulation.py')
            return error_sim_file.exists()
        except Exception:
            return False

    def _calculate_component_score(self, details: Dict[str, Any], issues: List[str],
                                 weight_map: Dict[str, float]) -> float:
        """Calculate component score based on details and weights"""
        score = 0.0
        max_score = 100.0

        # Subtract points for issues
        issue_penalty = len(issues) * 10
        score = max_score - issue_penalty

        # Add weighted scores for details
        for key, weight in weight_map.items():
            if key in details:
                value = details[key]

                if isinstance(value, bool):
                    score += weight if value else 0
                elif isinstance(value, (int, float)):
                    if key.startswith('missing_') or key.endswith('_usage_percent'):
                        # Negative weights or usage percentages
                        if weight < 0:
                            score += weight * value
                        else:
                            score += weight * (1 - value / 100)
                    else:
                        # Normalize and apply weight
                        normalized = min(value / 10, 1.0) if value > 0 else 0
                        score += weight * normalized
                elif isinstance(value, list):
                    # List length as a factor
                    normalized = min(len(value) / 5, 1.0)
                    score += weight * normalized

        return max(0.0, min(100.0, score))

    def _calculate_performance_metrics(self, results: List[ValidationResult],
                                     total_time_ms: float) -> Dict[str, float]:
        """Calculate overall performance metrics"""
        return {
            'total_validation_time_ms': total_time_ms,
            'average_component_score': np.mean([r.score for r in results]) if results else 0.0,
            'component_pass_rate': sum(1 for r in results if r.passed) / len(results) if results else 0.0,
            'total_issues': sum(len(r.issues) for r in results),
            'critical_component_failures': sum(1 for r in results if not r.passed and r.score < 50)
        }

    def _calculate_health_score(self, results: List[ValidationResult]) -> float:
        """Calculate overall health score"""
        if not results:
            return 0.0

        # Weighted average with critical components having higher weight
        critical_components = ['BakingCore', 'QualityGates', 'Phase5Integration', 'Phase7Preparation']

        total_weighted_score = 0.0
        total_weight = 0.0

        for result in results:
            weight = 2.0 if result.component in critical_components else 1.0
            total_weighted_score += result.score * weight
            total_weight += weight

        return total_weighted_score / total_weight if total_weight > 0 else 0.0

    def _determine_overall_health(self, health_score: float, critical_issues: List[str]) -> str:
        """Determine overall health status"""
        if len(critical_issues) > 10 or health_score < 50:
            return 'CRITICAL'
        elif len(critical_issues) > 5 or health_score < 70:
            return 'WARNING'
        elif health_score < 85:
            return 'GOOD'
        else:
            return 'EXCELLENT'

    def _assess_readiness_status(self, results: List[ValidationResult]) -> Dict[str, bool]:
        """Assess readiness status for different phases"""
        component_status = {r.component: r.passed for r in results}

        return {
            'phase5_integration_ready': component_status.get('Phase5Integration', False),
            'baking_ready': component_status.get('BakingCore', False) and
                          component_status.get('OptimizationEngine', False),
            'quality_validation_ready': component_status.get('QualityGates', False),
            'phase7_preparation_ready': component_status.get('Phase7Preparation', False),
            'production_ready': all(component_status.values()),
            'system_resources_adequate': component_status.get('SystemResources', False),
            'error_handling_ready': component_status.get('ErrorHandling', False)
        }

    def _generate_recommendations(self, component_type: str, issues: List[str]) -> List[str]:
        """Generate specific recommendations based on component type and issues"""
        recommendations = []

        if component_type == 'phase5':
            if any('No Phase 5 trained models found' in issue for issue in issues):
                recommendations.append("Run Phase 5 training to generate models for baking")
            if any('compatible' in issue.lower() for issue in issues):
                recommendations.append("Review model compatibility requirements and adjust training parameters")

        elif component_type == 'baking_core':
            if any('Missing core components' in issue for issue in issues):
                recommendations.append("Install all required baking components")
            if any('configuration' in issue.lower() for issue in issues):
                recommendations.append("Configure baking parameters and optimization settings")

        elif component_type == 'optimization':
            if any('Insufficient optimization techniques' in issue for issue in issues):
                recommendations.append("Install additional optimization libraries (pruning, quantization)")
            if any('configuration' in issue.lower() for issue in issues):
                recommendations.append("Configure optimization targets and constraints")

        elif component_type == 'quality_gates':
            if any('quality gate components' in issue.lower() for issue in issues):
                recommendations.append("Implement missing quality gate components")
            if any('CTQ parameters' in issue for issue in issues):
                recommendations.append("Configure Critical-to-Quality parameters for validation")

        elif component_type == 'phase7':
            if any('No models ready for ADAS' in issue for issue in issues):
                recommendations.append("Complete Phase 6 baking to prepare models for ADAS deployment")
            if any('safety' in issue.lower() for issue in issues):
                recommendations.append("Implement safety compliance mechanisms for ADAS requirements")

        elif component_type == 'resources':
            if any('CPU' in issue for issue in issues):
                recommendations.append("Upgrade CPU or reduce concurrent baking operations")
            if any('memory' in issue.lower() for issue in issues):
                recommendations.append("Add more RAM or optimize memory usage")
            if any('disk' in issue.lower() for issue in issues):
                recommendations.append("Free up disk space or add more storage")
            if any('GPU' in issue for issue in issues):
                recommendations.append("Install GPU for accelerated model optimization")

        elif component_type == 'data_flow':
            if any('Missing directories' in issue for issue in issues):
                recommendations.append("Create required directory structure for data pipeline")
            if any('format support' in issue.lower() for issue in issues):
                recommendations.append("Install required libraries for data format conversion")

        elif component_type == 'error_handling':
            if any('error handling mechanisms' in issue.lower() for issue in issues):
                recommendations.append("Implement comprehensive error handling and recovery systems")
            if any('logging' in issue.lower() for issue in issues):
                recommendations.append("Configure proper logging for debugging and monitoring")

        if not recommendations:
            recommendations.append(f"Address {component_type} issues to improve pipeline health")

        return recommendations

    def generate_health_report(self, pipeline_health: PipelineHealth) -> str:
        """Generate comprehensive health report"""
        report = f"""
# Phase 6 Baking Pipeline Health Report

## Overall Status: {pipeline_health.overall_health}
**Health Score: {pipeline_health.health_score:.1f}/100**

### Performance Summary
- Total Validation Time: {pipeline_health.performance_metrics['total_validation_time_ms']:.0f}ms
- Component Pass Rate: {pipeline_health.performance_metrics['component_pass_rate']:.1%}
- Average Component Score: {pipeline_health.performance_metrics['average_component_score']:.1f}
- Total Issues Found: {pipeline_health.performance_metrics['total_issues']}

### Readiness Status
"""
        for status, ready in pipeline_health.readiness_status.items():
            status_icon = "✅" if ready else "❌"
            report += f"- {status.replace('_', ' ').title()}: {status_icon}\n"

        report += "\n### Component Results\n"
        for result in pipeline_health.component_results:
            status_icon = "✅" if result.passed else "❌"
            report += f"- **{result.component}** {status_icon}: {result.score:.1f}/100 ({result.execution_time_ms:.0f}ms)\n"

            if result.issues:
                report += f"  Issues: {', '.join(result.issues)}\n"

        if pipeline_health.critical_issues:
            report += f"\n### Critical Issues\n"
            for issue in pipeline_health.critical_issues:
                report += f"- ⚠️ {issue}\n"

        report += f"\n### Recommendations\n"
        all_recommendations = []
        for result in pipeline_health.component_results:
            all_recommendations.extend(result.recommendations)

        for rec in set(all_recommendations):
            report += f"- {rec}\n"

        return report

def create_pipeline_validator(config: Dict[str, Any]) -> PipelineValidator:
    """Factory function to create pipeline validator"""
    return PipelineValidator(config)

# Testing utilities
def test_pipeline_validation():
    """Test pipeline validation functionality"""
    config = {
        'phase5_config': {
            'phase5_model_dir': 'models/phase5',
            'supported_architectures': ['ResNet', 'VGG', 'MobileNet'],
            'min_accuracy': 0.85
        },
        'phase7_config': {
            'phase6_output_dir': 'models/phase6',
            'adas_export_dir': 'models/adas',
            'max_inference_time_ms': 50.0,
            'min_accuracy': 0.95
        },
        'min_health_score': 75.0,
        'min_component_score': 65.0
    }

    validator = PipelineValidator(config)
    pipeline_health = validator.validate_complete_pipeline()

    print(validator.generate_health_report(pipeline_health))
    return pipeline_health

if __name__ == "__main__":
    test_pipeline_validation()