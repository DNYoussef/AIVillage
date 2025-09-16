"""
Agent Forge Phase 5 - Master Pipeline Coordinator
Orchestrates the complete training pipeline with BitNet and Grokfast integration
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
import logging
import time
import json
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import defaultdict

# Import pipeline components
from .data_loader import DataLoaderFactory, DataConfig, DataLoadingProfiler
from .training_loop import TrainingLoop, TrainingConfig, TrainingState
from .bitnet_optimizer import BitNetOptimizer, BitNetConfig, convert_model_to_bitnet
from .grokfast_trainer import GrokfastTrainer, GrokfastConfig
from .loss_functions import LossManager, LossConfig, create_loss_function
from .scheduler import SchedulerFactory, SchedulerConfig, SchedulerType
from .validation import RealTimeValidator, ValidationConfig

class PipelineState(Enum):
    INITIALIZING = "initializing"
    READY = "ready"
    TRAINING = "training"
    VALIDATING = "validating"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"

class PipelinePhase(Enum):
    SETUP = "setup"
    DATA_LOADING = "data_loading"
    MODEL_PREPARATION = "model_preparation"
    TRAINING = "training"
    VALIDATION = "validation"
    COMPLETION = "completion"

@dataclass
class PipelineConfig:
    """Complete pipeline configuration"""
    # Core settings
    experiment_name: str = "agent_forge_training"
    output_dir: str = "./training_output"
    device: str = "auto"  # auto, cpu, cuda
    mixed_precision: bool = True
    distributed: bool = False

    # Data configuration
    data_config: DataConfig = field(default_factory=DataConfig)

    # Model configuration
    model_config: Dict[str, Any] = field(default_factory=dict)

    # Training configuration
    training_config: TrainingConfig = field(default_factory=TrainingConfig)

    # BitNet configuration
    bitnet_config: Optional[BitNetConfig] = None

    # Grokfast configuration
    grokfast_config: Optional[GrokfastConfig] = None

    # Loss configuration
    loss_config: LossConfig = field(default_factory=LossConfig)

    # Scheduler configuration
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)

    # Validation configuration
    validation_config: ValidationConfig = field(default_factory=ValidationConfig)

    # Pipeline specific
    max_training_time: Optional[float] = None  # seconds
    save_intermediate_checkpoints: bool = True
    auto_resume: bool = True
    performance_profiling: bool = True

class PipelineMetrics:
    """Comprehensive pipeline metrics tracking"""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.phase_times = {}
        self.training_metrics = defaultdict(list)
        self.validation_metrics = defaultdict(list)
        self.system_metrics = defaultdict(list)
        self.pipeline_events = []

    def start_phase(self, phase: PipelinePhase):
        """Start timing a pipeline phase"""
        self.phase_times[phase.value] = {'start': time.time()}
        self.pipeline_events.append({
            'timestamp': time.time(),
            'event': f"phase_start_{phase.value}"
        })

    def end_phase(self, phase: PipelinePhase):
        """End timing a pipeline phase"""
        if phase.value in self.phase_times:
            self.phase_times[phase.value]['end'] = time.time()
            duration = self.phase_times[phase.value]['end'] - self.phase_times[phase.value]['start']
            self.phase_times[phase.value]['duration'] = duration

        self.pipeline_events.append({
            'timestamp': time.time(),
            'event': f"phase_end_{phase.value}"
        })

    def record_training_metrics(self, metrics: Dict[str, float], step: int):
        """Record training metrics"""
        for key, value in metrics.items():
            self.training_metrics[key].append((step, value, time.time()))

    def record_validation_metrics(self, metrics: Dict[str, float], step: int):
        """Record validation metrics"""
        for key, value in metrics.items():
            self.validation_metrics[key].append((step, value, time.time()))

    def record_system_metrics(self, metrics: Dict[str, float]):
        """Record system performance metrics"""
        timestamp = time.time()
        for key, value in metrics.items():
            self.system_metrics[key].append((timestamp, value))

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        total_time = None
        if self.start_time and self.end_time:
            total_time = self.end_time - self.start_time

        summary = {
            'total_time': total_time,
            'phase_durations': {
                phase: times.get('duration', 0)
                for phase, times in self.phase_times.items()
            },
            'total_events': len(self.pipeline_events)
        }

        # Training metrics summary
        if self.training_metrics:
            summary['training_summary'] = {}
            for metric, values in self.training_metrics.items():
                if values:
                    latest_value = values[-1][1]
                    summary['training_summary'][metric] = {
                        'latest': latest_value,
                        'count': len(values)
                    }

        # Validation metrics summary
        if self.validation_metrics:
            summary['validation_summary'] = {}
            for metric, values in self.validation_metrics.items():
                if values:
                    latest_value = values[-1][1]
                    summary['validation_summary'][metric] = {
                        'latest': latest_value,
                        'count': len(values)
                    }

        return summary

class TrainingPipeline:
    """Master training pipeline coordinator"""

    def __init__(
        self,
        model: nn.Module,
        config: PipelineConfig,
        train_data_path: str,
        val_data_path: Optional[str] = None
    ):
        self.model = model
        self.config = config
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path

        # Pipeline state
        self.state = PipelineState.INITIALIZING
        self.current_phase = None

        # Metrics and monitoring
        self.metrics = PipelineMetrics()
        self.profiler = DataLoadingProfiler() if config.performance_profiling else None

        # Pipeline components (initialized during setup)
        self.device = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.loss_manager = None
        self.trainer = None
        self.validator = None

        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup comprehensive logging"""
        log_file = self.output_dir / f"{self.config.experiment_name}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger(f"Pipeline_{self.config.experiment_name}")

    def setup(self):
        """Setup the complete training pipeline"""
        self.state = PipelineState.INITIALIZING
        self.metrics.start_time = time.time()

        try:
            self._setup_device()
            self._setup_data_loaders()
            self._setup_model()
            self._setup_optimization()
            self._setup_loss_functions()
            self._setup_validation()

            self.state = PipelineState.READY
            self.logger.info("Pipeline setup completed successfully")

        except Exception as e:
            self.state = PipelineState.ERROR
            self.logger.error(f"Pipeline setup failed: {e}")
            raise

    def _setup_device(self):
        """Setup compute device"""
        self.metrics.start_phase(PipelinePhase.SETUP)

        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

        self.logger.info(f"Using device: {self.device}")

        if self.device.type == "cuda":
            self.logger.info(f"GPU: {torch.cuda.get_device_name()}")
            self.logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        self.metrics.end_phase(PipelinePhase.SETUP)

    def _setup_data_loaders(self):
        """Setup data loading pipeline"""
        self.metrics.start_phase(PipelinePhase.DATA_LOADING)

        # Create training data loader
        self.train_loader = DataLoaderFactory.create_loader(
            self.train_data_path,
            self.config.data_config
        )

        # Create validation data loader if path provided
        if self.val_data_path:
            val_config = self.config.data_config
            val_config.shuffle = False
            val_config.drop_last = False

            self.val_loader = DataLoaderFactory.create_loader(
                self.val_data_path,
                val_config
            )
        else:
            # Create validation split from training data
            self.train_loader, self.val_loader = DataLoaderFactory.create_validation_split(
                self.train_data_path,
                self.config.data_config
            )

        self.logger.info(f"Data loaders created - Train batches: {len(self.train_loader)}")
        if self.val_loader:
            self.logger.info(f"Validation batches: {len(self.val_loader)}")

        self.metrics.end_phase(PipelinePhase.DATA_LOADING)

    def _setup_model(self):
        """Setup model with BitNet conversion if configured"""
        self.metrics.start_phase(PipelinePhase.MODEL_PREPARATION)

        # Move model to device
        self.model = self.model.to(self.device)

        # Convert to BitNet if configured
        if self.config.bitnet_config:
            self.logger.info("Converting model to BitNet")
            self.model = convert_model_to_bitnet(self.model, self.config.bitnet_config)

        # Enable mixed precision if configured
        if self.config.mixed_precision and torch.cuda.is_available():
            self.model = self.model.half()
            self.logger.info("Enabled mixed precision training")

        # Model compilation if available
        if hasattr(torch, 'compile') and self.config.training_config.compile_model:
            self.model = torch.compile(self.model)
            self.logger.info("Model compiled for optimization")

        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.logger.info(f"Model parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")

        self.metrics.end_phase(PipelinePhase.MODEL_PREPARATION)

    def _setup_optimization(self):
        """Setup optimizer and scheduler"""
        # Create optimizer
        if self.config.bitnet_config:
            self.optimizer = BitNetOptimizer(
                self.model.parameters(),
                self.config.bitnet_config
            )
            self.logger.info("Using BitNet optimizer")
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.training_config.learning_rate,
                weight_decay=self.config.training_config.weight_decay
            )
            self.logger.info("Using AdamW optimizer")

        # Create scheduler
        self.scheduler = SchedulerFactory.create_scheduler(
            self.optimizer,
            self.config.scheduler_config
        )
        self.logger.info(f"Created {self.config.scheduler_config.scheduler_type.value} scheduler")

    def _setup_loss_functions(self):
        """Setup loss function management"""
        self.loss_manager = LossManager(self.config.loss_config)

        # Create primary loss function
        if self.config.bitnet_config:
            loss_fn = create_loss_function(
                "bitnet_classification",
                self.config.loss_config,
                num_classes=self.config.model_config.get('num_classes', 10)
            )
        elif self.config.grokfast_config:
            loss_fn = create_loss_function(
                "grokfast_classification",
                self.config.loss_config,
                num_classes=self.config.model_config.get('num_classes', 10)
            )
        else:
            loss_fn = create_loss_function(
                "classification",
                self.config.loss_config,
                num_classes=self.config.model_config.get('num_classes', 10)
            )

        self.loss_manager.register_loss("primary", loss_fn, 1.0)
        self.logger.info("Loss functions configured")

    def _setup_validation(self):
        """Setup validation system"""
        self.validator = RealTimeValidator(
            self.model,
            self.config.validation_config,
            self.device
        )

        if self.config.validation_config.memory_monitoring:
            self.validator.start_monitoring()

        self.logger.info("Validation system initialized")

    def train(self):
        """Execute the complete training pipeline"""
        if self.state != PipelineState.READY:
            raise RuntimeError("Pipeline not ready. Call setup() first.")

        self.state = PipelineState.TRAINING
        self.metrics.start_phase(PipelinePhase.TRAINING)

        try:
            # Create training loop
            if self.config.grokfast_config:
                # Use Grokfast trainer
                base_optimizer = self.optimizer
                if hasattr(self.optimizer, 'optimizer'):
                    base_optimizer = self.optimizer.optimizer

                self.trainer = GrokfastTrainer(
                    self.model,
                    base_optimizer,
                    self.config.grokfast_config,
                    self.device
                )
                self.logger.info("Using Grokfast training system")
            else:
                # Use standard training loop
                self.trainer = TrainingLoop(
                    self.model,
                    self.config.training_config,
                    self.device,
                    str(self.output_dir / "checkpoints")
                )
                self.trainer.setup_optimizer()
                self.logger.info("Using standard training loop")

            # Define loss function for trainer
            def training_loss_fn(batch):
                if hasattr(self.trainer, 'train_step'):
                    # Grokfast trainer
                    predictions = self.model(batch['input'])
                    return self.loss_manager.compute_total_loss(
                        predictions, batch['target'], model=self.model
                    )['total_loss']
                else:
                    # Standard trainer
                    return self.loss_manager.compute_total_loss(
                        self.model(batch['input']), batch['target'], model=self.model
                    )['total_loss']

            # Training execution
            if hasattr(self.trainer, 'train'):
                # Standard training loop
                self.trainer.train(
                    self.train_loader,
                    training_loss_fn,
                    self.val_loader
                )
            else:
                # Grokfast trainer - manual loop
                self._execute_grokfast_training(training_loss_fn)

            self.state = PipelineState.COMPLETED
            self.logger.info("Training completed successfully")

        except KeyboardInterrupt:
            self.state = PipelineState.PAUSED
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.state = PipelineState.ERROR
            self.logger.error(f"Training failed: {e}")
            raise
        finally:
            self.metrics.end_phase(PipelinePhase.TRAINING)
            self.metrics.end_time = time.time()
            self._save_final_results()

    def _execute_grokfast_training(self, loss_fn: Callable):
        """Execute Grokfast training loop"""
        for epoch in range(self.config.training_config.epochs):
            self.logger.info(f"Starting epoch {epoch + 1}")

            for batch_idx, batch in enumerate(self.train_loader):
                # Training step
                metrics = self.trainer.train_step(batch, loss_fn)

                # Record metrics
                self.metrics.record_training_metrics(metrics, self.trainer.global_step)

                # Validation
                if (self.val_loader and
                    self.trainer.global_step % self.config.validation_config.validation_frequency == 0):
                    val_metrics = self._run_validation()
                    self.metrics.record_validation_metrics(val_metrics, self.trainer.global_step)

                # Logging
                if self.trainer.global_step % self.config.training_config.log_interval == 0:
                    self._log_training_progress(metrics, self.trainer.global_step)

                # Early stopping check
                if hasattr(self.validator, 'should_stop_early') and self.validator.should_stop_early():
                    self.logger.info("Early stopping triggered")
                    return

                # Maximum time check
                if (self.config.max_training_time and
                    time.time() - self.metrics.start_time > self.config.max_training_time):
                    self.logger.info("Maximum training time reached")
                    return

    def _run_validation(self) -> Dict[str, float]:
        """Run validation and return metrics"""
        self.metrics.start_phase(PipelinePhase.VALIDATION)

        # Define validation loss function
        def val_loss_fn(batch):
            predictions = self.model(batch['input'])
            return self.loss_manager.compute_total_loss(
                predictions, batch['target'], model=self.model
            )['total_loss']

        val_metrics = self.validator.run_validation(self.val_loader, val_loss_fn)

        self.metrics.end_phase(PipelinePhase.VALIDATION)
        return val_metrics

    def _log_training_progress(self, metrics: Dict[str, float], step: int):
        """Log training progress"""
        log_msg = f"Step {step:6d} | "

        if 'loss' in metrics:
            log_msg += f"Loss: {metrics['loss']:.6f} | "
        if 'learning_rate' in metrics:
            log_msg += f"LR: {metrics['learning_rate']:.2e} | "
        if 'grokfast_phase' in metrics:
            log_msg += f"Phase: {metrics['grokfast_phase']} | "

        # Add memory usage if available
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1e9
            log_msg += f"GPU: {memory_used:.1f}GB"

        self.logger.info(log_msg)

    def _save_final_results(self):
        """Save final training results"""
        self.metrics.start_phase(PipelinePhase.COMPLETION)

        # Save metrics
        metrics_file = self.output_dir / f"{self.config.experiment_name}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics.get_summary(), f, indent=2, default=str)

        # Save model
        model_file = self.output_dir / f"{self.config.experiment_name}_final_model.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config.__dict__,
            'metrics_summary': self.metrics.get_summary()
        }, model_file)

        # Export validation results
        if self.validator:
            val_file = self.output_dir / f"{self.config.experiment_name}_validation.json"
            self.validator.export_results(str(val_file))

        self.logger.info(f"Results saved to {self.output_dir}")
        self.metrics.end_phase(PipelinePhase.COMPLETION)

    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        status = {
            'state': self.state.value,
            'current_phase': self.current_phase.value if self.current_phase else None,
            'metrics_summary': self.metrics.get_summary()
        }

        if self.trainer and hasattr(self.trainer, 'get_training_summary'):
            status['training_summary'] = self.trainer.get_training_summary()
        elif self.trainer and hasattr(self.trainer, 'get_state'):
            status['training_state'] = self.trainer.get_state()

        if self.validator:
            status['validation_summary'] = self.validator.get_validation_summary()

        return status

    def cleanup(self):
        """Cleanup pipeline resources"""
        if self.validator:
            self.validator.stop_monitoring_thread()

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info("Pipeline cleanup completed")

def create_training_pipeline(
    model: nn.Module,
    train_data_path: str,
    experiment_name: str = "agent_forge_training",
    val_data_path: Optional[str] = None,
    use_bitnet: bool = False,
    use_grokfast: bool = False,
    **kwargs
) -> TrainingPipeline:
    """Factory function to create training pipeline"""

    # Create configuration
    config = PipelineConfig(
        experiment_name=experiment_name,
        **kwargs
    )

    # Configure BitNet if requested
    if use_bitnet:
        config.bitnet_config = BitNetConfig()

    # Configure Grokfast if requested
    if use_grokfast:
        config.grokfast_config = GrokfastConfig()

    # Create and return pipeline
    pipeline = TrainingPipeline(
        model=model,
        config=config,
        train_data_path=train_data_path,
        val_data_path=val_data_path
    )

    return pipeline

if __name__ == "__main__":
    # Example usage and testing
    import torch.nn as nn
    import tempfile
    import json

    # Create test model
    class TestModel(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, num_classes)
            )

        def forward(self, x):
            return self.layers(x)

    # Create test data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        test_data = []
        for _ in range(1000):
            test_data.append({
                'input': torch.randn(128).tolist(),
                'target': torch.randint(0, 10, (1,)).item()
            })
        json.dump(test_data, f)
        test_file = f.name

    # Create model
    model = TestModel(num_classes=10)

    # Test standard pipeline
    print("Testing standard training pipeline...")
    pipeline = create_training_pipeline(
        model=model,
        train_data_path=test_file,
        experiment_name="test_standard",
        output_dir="./test_output"
    )

    # Update configuration for quick test
    pipeline.config.training_config.epochs = 2
    pipeline.config.data_config.batch_size = 16
    pipeline.config.validation_config.validation_frequency = 50

    # Setup and run
    pipeline.setup()
    print(f"Pipeline state: {pipeline.state.value}")

    # Get status
    status = pipeline.get_status()
    print(f"Pipeline status: {status}")

    # Cleanup
    pipeline.cleanup()

    # Test BitNet pipeline
    print("\nTesting BitNet training pipeline...")
    bitnet_pipeline = create_training_pipeline(
        model=TestModel(num_classes=10),
        train_data_path=test_file,
        experiment_name="test_bitnet",
        use_bitnet=True,
        output_dir="./test_output"
    )

    bitnet_pipeline.config.training_config.epochs = 1
    bitnet_pipeline.setup()
    print(f"BitNet pipeline state: {bitnet_pipeline.state.value}")

    bitnet_pipeline.cleanup()

    # Cleanup test file
    import os
    os.unlink(test_file)

    print("Pipeline coordinator test completed successfully!")