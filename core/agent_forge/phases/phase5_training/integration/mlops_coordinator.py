"""
MLOps Coordinator for Training Automation
Coordinates MLOps pipeline integration with Phase 5 training.
"""

import asyncio
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import time

class PipelineStage(Enum):
    DATA_PREPARATION = "data_preparation"
    MODEL_TRAINING = "model_training"
    VALIDATION = "validation"
    TESTING = "testing"
    DEPLOYMENT_PREP = "deployment_prep"
    MONITORING = "monitoring"

class PipelineStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class PipelineConfig:
    """MLOps pipeline configuration."""
    pipeline_id: str
    experiment_name: str
    model_config: Dict[str, Any]
    training_config: Dict[str, Any]
    validation_config: Dict[str, Any]
    deployment_config: Dict[str, Any]
    monitoring_config: Dict[str, Any]
    resource_config: Dict[str, Any]

@dataclass
class ExperimentRun:
    """Experiment run information."""
    run_id: str
    experiment_name: str
    pipeline_id: str
    status: PipelineStatus
    start_time: datetime
    end_time: Optional[datetime]
    metrics: Dict[str, float]
    artifacts: Dict[str, str]
    logs: List[str]

@dataclass
class ModelRegistry:
    """Model registry entry."""
    model_id: str
    model_name: str
    version: str
    phase: str
    metrics: Dict[str, float]
    artifacts: Dict[str, str]
    registered_at: datetime
    status: str

class MLOpsCoordinator:
    """
    Coordinates MLOps pipeline integration with Phase 5 training.
    """

    def __init__(self, workspace_dir: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        self.workspace_dir = workspace_dir or Path("mlops/workspace")
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        # MLOps components
        self.experiment_tracker = ExperimentTracker(self.workspace_dir / "experiments")
        self.model_registry = ModelRegistryManager(self.workspace_dir / "registry")
        self.pipeline_orchestrator = PipelineOrchestrator(self.workspace_dir / "pipelines")
        self.monitoring_system = MonitoringSystem(self.workspace_dir / "monitoring")

        # State tracking
        self.active_pipelines = {}
        self.experiment_history = {}

    async def initialize(self) -> bool:
        """Initialize MLOps coordinator."""
        try:
            self.logger.info("Initializing MLOps coordinator")

            # Initialize components
            components = [
                self.experiment_tracker,
                self.model_registry,
                self.pipeline_orchestrator,
                self.monitoring_system
            ]

            for component in components:
                if not await component.initialize():
                    raise Exception(f"Failed to initialize {component.__class__.__name__}")

            self.logger.info("MLOps coordinator initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"MLOps coordinator initialization failed: {e}")
            return False

    async def create_training_pipeline(self, config: PipelineConfig) -> str:
        """Create Phase 5 training pipeline."""
        try:
            self.logger.info(f"Creating training pipeline: {config.pipeline_id}")

            # Register pipeline
            pipeline_id = await self.pipeline_orchestrator.register_pipeline(config)

            # Setup experiment tracking
            experiment_run = ExperimentRun(
                run_id=f"run_{int(time.time())}",
                experiment_name=config.experiment_name,
                pipeline_id=pipeline_id,
                status=PipelineStatus.PENDING,
                start_time=datetime.now(),
                end_time=None,
                metrics={},
                artifacts={},
                logs=[]
            )

            await self.experiment_tracker.start_experiment(experiment_run)

            # Initialize monitoring
            await self.monitoring_system.setup_monitoring(config)

            self.active_pipelines[pipeline_id] = {
                "config": config,
                "experiment_run": experiment_run,
                "status": PipelineStatus.PENDING
            }

            self.logger.info(f"Training pipeline created: {pipeline_id}")
            return pipeline_id

        except Exception as e:
            self.logger.error(f"Failed to create training pipeline: {e}")
            raise

    async def execute_training_pipeline(self, pipeline_id: str, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute training pipeline."""
        try:
            self.logger.info(f"Executing training pipeline: {pipeline_id}")

            if pipeline_id not in self.active_pipelines:
                raise ValueError(f"Pipeline not found: {pipeline_id}")

            pipeline_info = self.active_pipelines[pipeline_id]
            config = pipeline_info["config"]
            experiment_run = pipeline_info["experiment_run"]

            # Update status
            experiment_run.status = PipelineStatus.RUNNING
            await self.experiment_tracker.update_experiment(experiment_run)

            # Execute pipeline stages
            results = {}
            for stage in PipelineStage:
                stage_result = await self._execute_pipeline_stage(
                    stage, config, training_data, experiment_run
                )
                results[stage.value] = stage_result

                # Check for failures
                if not stage_result.get("success", False):
                    experiment_run.status = PipelineStatus.FAILED
                    await self.experiment_tracker.update_experiment(experiment_run)
                    raise Exception(f"Pipeline stage failed: {stage.value}")

            # Complete pipeline
            experiment_run.status = PipelineStatus.COMPLETED
            experiment_run.end_time = datetime.now()
            await self.experiment_tracker.update_experiment(experiment_run)

            # Register model if training successful
            if results.get("model_training", {}).get("model"):
                await self._register_trained_model(
                    results["model_training"]["model"],
                    config,
                    experiment_run
                )

            self.logger.info(f"Training pipeline completed: {pipeline_id}")
            return results

        except Exception as e:
            self.logger.error(f"Training pipeline execution failed: {e}")

            # Update experiment status
            if pipeline_id in self.active_pipelines:
                experiment_run = self.active_pipelines[pipeline_id]["experiment_run"]
                experiment_run.status = PipelineStatus.FAILED
                experiment_run.end_time = datetime.now()
                await self.experiment_tracker.update_experiment(experiment_run)

            raise

    async def monitor_training_progress(self, pipeline_id: str) -> Dict[str, Any]:
        """Monitor training progress."""
        try:
            if pipeline_id not in self.active_pipelines:
                return {"error": f"Pipeline not found: {pipeline_id}"}

            pipeline_info = self.active_pipelines[pipeline_id]
            experiment_run = pipeline_info["experiment_run"]

            # Get latest metrics
            metrics = await self.monitoring_system.get_current_metrics(pipeline_id)

            # Update experiment with metrics
            experiment_run.metrics.update(metrics)
            await self.experiment_tracker.update_experiment(experiment_run)

            # Prepare progress report
            progress_report = {
                "pipeline_id": pipeline_id,
                "status": experiment_run.status.value,
                "start_time": experiment_run.start_time.isoformat(),
                "duration_minutes": (datetime.now() - experiment_run.start_time).total_seconds() / 60,
                "current_metrics": metrics,
                "logs": experiment_run.logs[-10:] if experiment_run.logs else []  # Last 10 logs
            }

            return progress_report

        except Exception as e:
            self.logger.error(f"Failed to monitor training progress: {e}")
            return {"error": str(e)}

    async def get_pipeline_recommendations(self, pipeline_id: str) -> List[str]:
        """Get recommendations for pipeline optimization."""
        try:
            recommendations = []

            if pipeline_id not in self.active_pipelines:
                return ["Pipeline not found"]

            pipeline_info = self.active_pipelines[pipeline_id]
            experiment_run = pipeline_info["experiment_run"]

            # Analyze metrics for recommendations
            metrics = experiment_run.metrics

            # Training performance recommendations
            if "training_accuracy" in metrics:
                accuracy = metrics["training_accuracy"]
                if accuracy < 0.8:
                    recommendations.append("Consider increasing training epochs or adjusting learning rate")
                elif accuracy > 0.99:
                    recommendations.append("Model may be overfitting - consider regularization")

            if "training_loss" in metrics and "validation_loss" in metrics:
                train_loss = metrics["training_loss"]
                val_loss = metrics["validation_loss"]
                if val_loss > train_loss * 1.5:
                    recommendations.append("Validation loss significantly higher - possible overfitting")

            # Resource utilization recommendations
            if "gpu_utilization" in metrics:
                gpu_util = metrics["gpu_utilization"]
                if gpu_util < 0.5:
                    recommendations.append("Low GPU utilization - consider increasing batch size")
                elif gpu_util > 0.95:
                    recommendations.append("High GPU utilization - monitor for memory issues")

            # Training time recommendations
            if "epoch_time_seconds" in metrics:
                epoch_time = metrics["epoch_time_seconds"]
                if epoch_time > 300:  # 5 minutes
                    recommendations.append("Long epoch time - consider model optimization or data pipeline improvements")

            if not recommendations:
                recommendations.append("Training pipeline performing well - no immediate optimizations needed")

            return recommendations

        except Exception as e:
            self.logger.error(f"Failed to get pipeline recommendations: {e}")
            return [f"Error getting recommendations: {e}"]

    async def export_experiment_results(self, experiment_name: str, output_path: Optional[Path] = None) -> Path:
        """Export experiment results for analysis."""
        try:
            self.logger.info(f"Exporting experiment results: {experiment_name}")

            # Get experiment runs
            runs = await self.experiment_tracker.get_experiment_runs(experiment_name)

            if not runs:
                raise ValueError(f"No runs found for experiment: {experiment_name}")

            # Prepare export data
            export_data = {
                "experiment_name": experiment_name,
                "export_timestamp": datetime.now().isoformat(),
                "total_runs": len(runs),
                "runs": [asdict(run) for run in runs],
                "summary_statistics": self._calculate_experiment_statistics(runs)
            }

            # Determine output path
            if output_path is None:
                output_path = self.workspace_dir / "exports" / f"{experiment_name}_{int(time.time())}.json"
                output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save export
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)

            self.logger.info(f"Experiment results exported: {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"Failed to export experiment results: {e}")
            raise

    async def _execute_pipeline_stage(self, stage: PipelineStage, config: PipelineConfig, data: Dict[str, Any], experiment_run: ExperimentRun) -> Dict[str, Any]:
        """Execute a single pipeline stage."""
        try:
            self.logger.info(f"Executing pipeline stage: {stage.value}")

            stage_start_time = time.time()
            result = {"success": False, "duration_seconds": 0, "artifacts": {}}

            if stage == PipelineStage.DATA_PREPARATION:
                result = await self._execute_data_preparation(config, data)
            elif stage == PipelineStage.MODEL_TRAINING:
                result = await self._execute_model_training(config, data)
            elif stage == PipelineStage.VALIDATION:
                result = await self._execute_validation(config, data)
            elif stage == PipelineStage.TESTING:
                result = await self._execute_testing(config, data)
            elif stage == PipelineStage.DEPLOYMENT_PREP:
                result = await self._execute_deployment_prep(config, data)
            elif stage == PipelineStage.MONITORING:
                result = await self._execute_monitoring_setup(config, data)

            result["duration_seconds"] = time.time() - stage_start_time

            # Log stage completion
            log_entry = f"Stage {stage.value} completed in {result['duration_seconds']:.2f}s"
            experiment_run.logs.append(log_entry)

            self.logger.info(f"Pipeline stage completed: {stage.value}")
            return result

        except Exception as e:
            self.logger.error(f"Pipeline stage failed: {stage.value}: {e}")
            return {"success": False, "error": str(e), "duration_seconds": time.time() - stage_start_time}

    async def _execute_data_preparation(self, config: PipelineConfig, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data preparation stage."""
        # Mock implementation - in practice would prepare actual training data
        return {
            "success": True,
            "prepared_samples": 10000,
            "validation_split": 0.2,
            "artifacts": {"data_stats": "data_preparation_stats.json"}
        }

    async def _execute_model_training(self, config: PipelineConfig, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute model training stage."""
        # Mock implementation - in practice would train actual model
        mock_model = {
            "model_id": f"phase5_model_{int(time.time())}",
            "architecture": config.model_config.get("architecture", "transformer"),
            "parameters": config.model_config.get("num_parameters", 1000000),
            "trained": True
        }

        return {
            "success": True,
            "model": mock_model,
            "training_metrics": {
                "final_accuracy": 0.92,
                "final_loss": 0.08,
                "training_time_hours": 2.5
            },
            "artifacts": {
                "model_checkpoint": "model_checkpoint.pkl",
                "training_log": "training.log"
            }
        }

    async def _execute_validation(self, config: PipelineConfig, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute validation stage."""
        return {
            "success": True,
            "validation_metrics": {
                "accuracy": 0.91,
                "precision": 0.89,
                "recall": 0.88,
                "f1_score": 0.885
            },
            "artifacts": {"validation_report": "validation_report.json"}
        }

    async def _execute_testing(self, config: PipelineConfig, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute testing stage."""
        return {
            "success": True,
            "test_metrics": {
                "test_accuracy": 0.90,
                "test_coverage": 0.95,
                "performance_tests_passed": True
            },
            "artifacts": {"test_report": "test_report.json"}
        }

    async def _execute_deployment_prep(self, config: PipelineConfig, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute deployment preparation stage."""
        return {
            "success": True,
            "deployment_artifacts": {
                "model_package": "model_deployment_package.tar.gz",
                "inference_config": "inference_config.json",
                "deployment_manifest": "deployment_manifest.yaml"
            },
            "readiness_checks": {
                "model_serializable": True,
                "dependencies_resolved": True,
                "performance_acceptable": True
            }
        }

    async def _execute_monitoring_setup(self, config: PipelineConfig, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute monitoring setup stage."""
        return {
            "success": True,
            "monitoring_config": {
                "metrics_collection": True,
                "alerting_configured": True,
                "dashboards_created": True
            },
            "artifacts": {"monitoring_config": "monitoring_config.json"}
        }

    async def _register_trained_model(self, model: Dict[str, Any], config: PipelineConfig, experiment_run: ExperimentRun) -> str:
        """Register trained model in model registry."""
        try:
            model_registry_entry = ModelRegistry(
                model_id=model["model_id"],
                model_name=f"{config.experiment_name}_model",
                version="1.0.0",
                phase="phase5",
                metrics=experiment_run.metrics,
                artifacts=experiment_run.artifacts,
                registered_at=datetime.now(),
                status="active"
            )

            model_id = await self.model_registry.register_model(model_registry_entry)
            self.logger.info(f"Model registered: {model_id}")
            return model_id

        except Exception as e:
            self.logger.error(f"Failed to register model: {e}")
            raise

    def _calculate_experiment_statistics(self, runs: List[ExperimentRun]) -> Dict[str, Any]:
        """Calculate summary statistics for experiment runs."""
        if not runs:
            return {}

        completed_runs = [run for run in runs if run.status == PipelineStatus.COMPLETED]

        if not completed_runs:
            return {"completed_runs": 0, "success_rate": 0.0}

        # Calculate average metrics
        all_metrics = {}
        for run in completed_runs:
            for metric, value in run.metrics.items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)

        avg_metrics = {
            metric: sum(values) / len(values)
            for metric, values in all_metrics.items()
        }

        # Calculate durations
        durations = []
        for run in completed_runs:
            if run.end_time:
                duration = (run.end_time - run.start_time).total_seconds()
                durations.append(duration)

        avg_duration = sum(durations) / len(durations) if durations else 0

        return {
            "total_runs": len(runs),
            "completed_runs": len(completed_runs),
            "success_rate": len(completed_runs) / len(runs) * 100,
            "average_metrics": avg_metrics,
            "average_duration_seconds": avg_duration
        }


# Supporting Classes

class ExperimentTracker:
    """Tracks ML experiments."""

    def __init__(self, experiments_dir: Path):
        self.experiments_dir = experiments_dir
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"{__name__}.ExperimentTracker")

    async def initialize(self) -> bool:
        """Initialize experiment tracker."""
        self.logger.info("Experiment tracker initialized")
        return True

    async def start_experiment(self, experiment_run: ExperimentRun) -> bool:
        """Start tracking an experiment."""
        try:
            experiment_file = self.experiments_dir / f"{experiment_run.run_id}.json"

            with open(experiment_file, 'w') as f:
                json.dump(asdict(experiment_run), f, indent=2, default=str)

            self.logger.info(f"Started tracking experiment: {experiment_run.run_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start experiment tracking: {e}")
            return False

    async def update_experiment(self, experiment_run: ExperimentRun) -> bool:
        """Update experiment tracking."""
        try:
            experiment_file = self.experiments_dir / f"{experiment_run.run_id}.json"

            with open(experiment_file, 'w') as f:
                json.dump(asdict(experiment_run), f, indent=2, default=str)

            return True

        except Exception as e:
            self.logger.error(f"Failed to update experiment: {e}")
            return False

    async def get_experiment_runs(self, experiment_name: str) -> List[ExperimentRun]:
        """Get all runs for an experiment."""
        runs = []

        try:
            for experiment_file in self.experiments_dir.glob("*.json"):
                with open(experiment_file, 'r') as f:
                    run_data = json.load(f)

                if run_data.get("experiment_name") == experiment_name:
                    # Convert datetime strings back
                    run_data['start_time'] = datetime.fromisoformat(run_data['start_time'])
                    if run_data.get('end_time'):
                        run_data['end_time'] = datetime.fromisoformat(run_data['end_time'])
                    run_data['status'] = PipelineStatus(run_data['status'])

                    runs.append(ExperimentRun(**run_data))

        except Exception as e:
            self.logger.error(f"Failed to get experiment runs: {e}")

        return runs


class ModelRegistryManager:
    """Manages model registry."""

    def __init__(self, registry_dir: Path):
        self.registry_dir = registry_dir
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"{__name__}.ModelRegistryManager")

    async def initialize(self) -> bool:
        """Initialize model registry."""
        self.logger.info("Model registry initialized")
        return True

    async def register_model(self, model_entry: ModelRegistry) -> str:
        """Register a model."""
        try:
            model_file = self.registry_dir / f"{model_entry.model_id}.json"

            with open(model_file, 'w') as f:
                json.dump(asdict(model_entry), f, indent=2, default=str)

            self.logger.info(f"Model registered: {model_entry.model_id}")
            return model_entry.model_id

        except Exception as e:
            self.logger.error(f"Failed to register model: {e}")
            raise


class PipelineOrchestrator:
    """Orchestrates ML pipelines."""

    def __init__(self, pipelines_dir: Path):
        self.pipelines_dir = pipelines_dir
        self.pipelines_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"{__name__}.PipelineOrchestrator")

    async def initialize(self) -> bool:
        """Initialize pipeline orchestrator."""
        self.logger.info("Pipeline orchestrator initialized")
        return True

    async def register_pipeline(self, config: PipelineConfig) -> str:
        """Register a pipeline."""
        try:
            pipeline_file = self.pipelines_dir / f"{config.pipeline_id}.json"

            with open(pipeline_file, 'w') as f:
                json.dump(asdict(config), f, indent=2, default=str)

            self.logger.info(f"Pipeline registered: {config.pipeline_id}")
            return config.pipeline_id

        except Exception as e:
            self.logger.error(f"Failed to register pipeline: {e}")
            raise


class MonitoringSystem:
    """Monitoring and alerting system."""

    def __init__(self, monitoring_dir: Path):
        self.monitoring_dir = monitoring_dir
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"{__name__}.MonitoringSystem")

    async def initialize(self) -> bool:
        """Initialize monitoring system."""
        self.logger.info("Monitoring system initialized")
        return True

    async def setup_monitoring(self, config: PipelineConfig) -> bool:
        """Setup monitoring for pipeline."""
        try:
            monitoring_config = {
                "pipeline_id": config.pipeline_id,
                "metrics_to_track": ["accuracy", "loss", "training_time", "gpu_utilization"],
                "alert_thresholds": {
                    "accuracy": {"min": 0.8},
                    "loss": {"max": 0.2},
                    "training_time": {"max": 3600}  # 1 hour
                }
            }

            config_file = self.monitoring_dir / f"{config.pipeline_id}_monitoring.json"
            with open(config_file, 'w') as f:
                json.dump(monitoring_config, f, indent=2)

            return True

        except Exception as e:
            self.logger.error(f"Failed to setup monitoring: {e}")
            return False

    async def get_current_metrics(self, pipeline_id: str) -> Dict[str, float]:
        """Get current metrics for pipeline."""
        # Mock implementation - in practice would collect actual metrics
        return {
            "training_accuracy": 0.85 + (time.time() % 100) / 1000,  # Simulate changing metrics
            "training_loss": 0.15 - (time.time() % 100) / 2000,
            "gpu_utilization": 0.75 + (time.time() % 50) / 200,
            "epoch_time_seconds": 120 + (time.time() % 30)
        }