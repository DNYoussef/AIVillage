"""
Agent Forge Phase 5 - Training Pipeline
Comprehensive training system with BitNet and Grokfast optimization
"""

from .pipeline.data_loader import (
    DataLoaderFactory,
    DataConfig,
    CachedDataset,
    StreamingDataset,
    QualityValidator,
    DataLoadingProfiler,
    create_optimized_loader
)

from .pipeline.training_loop import (
    TrainingLoop,
    TrainingConfig,
    TrainingState,
    TrainingMetrics,
    CheckpointManager,
    EarlyStopping,
    MemoryMonitor
)

from .pipeline.bitnet_optimizer import (
    BitNetOptimizer,
    BitNetConfig,
    BitNetLayer,
    BitNetLossFunction,
    StraightThroughEstimator,
    convert_model_to_bitnet
)

from .pipeline.grokfast_trainer import (
    GrokfastTrainer,
    GrokfastConfig,
    GrokfastOptimizer,
    GrokfastScheduler,
    CapabilityTracker,
    KnowledgeConsolidation
)

from .pipeline.loss_functions import (
    LossManager,
    LossConfig,
    BaseLoss,
    ClassificationLoss,
    BitNetLoss,
    GrokfastLoss,
    ContrastiveLoss,
    MultiTaskLoss,
    AdaptiveLoss,
    create_loss_function
)

from .pipeline.scheduler import (
    SchedulerFactory,
    SchedulerConfig,
    SchedulerType,
    BaseScheduler,
    LinearScheduler,
    CosineScheduler,
    OneCycleScheduler,
    CyclicScheduler,
    AdaptiveScheduler,
    GrokfastScheduler,
    BitNetScheduler,
    CompositeScheduler,
    SchedulerMonitor
)

from .pipeline.validation import (
    RealTimeValidator,
    ValidationConfig,
    ValidationMode,
    MetricCalculator,
    PerformanceMonitor
)

from .pipeline.pipeline_coordinator import (
    TrainingPipeline,
    PipelineConfig,
    PipelineState,
    PipelinePhase,
    PipelineMetrics,
    create_training_pipeline
)

__all__ = [
    # Data Loading
    'DataLoaderFactory',
    'DataConfig',
    'CachedDataset',
    'StreamingDataset',
    'QualityValidator',
    'DataLoadingProfiler',
    'create_optimized_loader',

    # Training Loop
    'TrainingLoop',
    'TrainingConfig',
    'TrainingState',
    'TrainingMetrics',
    'CheckpointManager',
    'EarlyStopping',
    'MemoryMonitor',

    # BitNet
    'BitNetOptimizer',
    'BitNetConfig',
    'BitNetLayer',
    'BitNetLossFunction',
    'StraightThroughEstimator',
    'convert_model_to_bitnet',

    # Grokfast
    'GrokfastTrainer',
    'GrokfastConfig',
    'GrokfastOptimizer',
    'GrokfastScheduler',
    'CapabilityTracker',
    'KnowledgeConsolidation',

    # Loss Functions
    'LossManager',
    'LossConfig',
    'BaseLoss',
    'ClassificationLoss',
    'BitNetLoss',
    'GrokfastLoss',
    'ContrastiveLoss',
    'MultiTaskLoss',
    'AdaptiveLoss',
    'create_loss_function',

    # Schedulers
    'SchedulerFactory',
    'SchedulerConfig',
    'SchedulerType',
    'BaseScheduler',
    'LinearScheduler',
    'CosineScheduler',
    'OneCycleScheduler',
    'CyclicScheduler',
    'AdaptiveScheduler',
    'GrokfastScheduler',
    'BitNetScheduler',
    'CompositeScheduler',
    'SchedulerMonitor',

    # Validation
    'RealTimeValidator',
    'ValidationConfig',
    'ValidationMode',
    'MetricCalculator',
    'PerformanceMonitor',

    # Pipeline Coordinator
    'TrainingPipeline',
    'PipelineConfig',
    'PipelineState',
    'PipelinePhase',
    'PipelineMetrics',
    'create_training_pipeline'
]

# Version information
__version__ = "1.0.0"
__author__ = "Agent Forge Phase 5 Training Pipeline"
__description__ = "Comprehensive training pipeline with BitNet and Grokfast optimization"