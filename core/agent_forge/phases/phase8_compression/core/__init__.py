"""
Phase 8 Compression Core Algorithms
Advanced neural network compression techniques for model optimization.
"""

from .pruning_algorithms import (
    BasePruner,
    MagnitudePruner,
    GradientBasedPruner,
    SNIPPruner,
    GraSPPruner,
    StructuredPruner,
    AdaptivePruner,
    PruningOrchestrator,
    PruningConfig,
    create_pruner
)

from .quantization_engine import (
    BaseQuantizer,
    PostTrainingQuantizer,
    QuantizationAwareTrainer,
    DynamicQuantizer,
    MixedPrecisionOptimizer,
    CustomQuantizationScheme,
    QuantizationOrchestrator,
    QuantizationConfig,
    QuantizationMode,
    create_quantizer,
    benchmark_quantized_model
)

from .knowledge_distillation import (
    BaseDistiller,
    ResponseDistiller,
    FeatureDistiller,
    AttentionDistiller,
    ProgressiveDistiller,
    MultiTeacherDistiller,
    OnlineDistiller,
    DistillationOrchestrator,
    DistillationConfig,
    create_distiller,
    calculate_model_similarity
)

from .architecture_optimization import (
    BaseArchitectureOptimizer,
    MobileNetOptimizer,
    EfficientNetOptimizer,
    HardwareAwareOptimizer,
    ArchitectureOrchestrator,
    ArchitectureConfig,
    create_architecture_optimizer,
    visualize_architecture
)

__version__ = "1.0.0"

__all__ = [
    # Pruning
    "BasePruner",
    "MagnitudePruner",
    "GradientBasedPruner",
    "SNIPPruner",
    "GraSPPruner",
    "StructuredPruner",
    "AdaptivePruner",
    "PruningOrchestrator",
    "PruningConfig",
    "create_pruner",

    # Quantization
    "BaseQuantizer",
    "PostTrainingQuantizer",
    "QuantizationAwareTrainer",
    "DynamicQuantizer",
    "MixedPrecisionOptimizer",
    "CustomQuantizationScheme",
    "QuantizationOrchestrator",
    "QuantizationConfig",
    "QuantizationMode",
    "create_quantizer",
    "benchmark_quantized_model",

    # Knowledge Distillation
    "BaseDistiller",
    "ResponseDistiller",
    "FeatureDistiller",
    "AttentionDistiller",
    "ProgressiveDistiller",
    "MultiTeacherDistiller",
    "OnlineDistiller",
    "DistillationOrchestrator",
    "DistillationConfig",
    "create_distiller",
    "calculate_model_similarity",

    # Architecture Optimization
    "BaseArchitectureOptimizer",
    "MobileNetOptimizer",
    "EfficientNetOptimizer",
    "HardwareAwareOptimizer",
    "ArchitectureOrchestrator",
    "ArchitectureConfig",
    "create_architecture_optimizer",
    "visualize_architecture"
]

# Quick access factory functions
def compress_model(model, compression_type="pruning", **kwargs):
    """
    Quick model compression using different techniques.

    Args:
        model: PyTorch model to compress
        compression_type: Type of compression ('pruning', 'quantization', 'distillation', 'nas')
        **kwargs: Additional arguments for specific compression methods

    Returns:
        Compressed model and compression statistics
    """
    if compression_type == "pruning":
        pruner = create_pruner(
            algorithm=kwargs.get('algorithm', 'magnitude'),
            sparsity_ratio=kwargs.get('sparsity_ratio', 0.5)
        )
        dataloader = kwargs.get('dataloader')
        stats = pruner.prune_model(model, dataloader)
        return model, stats

    elif compression_type == "quantization":
        quantizer = create_quantizer(
            mode=kwargs.get('mode', 'ptq'),
            weight_bits=kwargs.get('weight_bits', 8)
        )
        orchestrator = QuantizationOrchestrator()
        config = QuantizationConfig(
            mode=QuantizationMode.PTQ if kwargs.get('mode', 'ptq') == 'ptq' else QuantizationMode.QAT
        )
        dataloader = kwargs.get('dataloader')
        return orchestrator.quantize_model(model, dataloader, config)

    elif compression_type == "distillation":
        teacher_model = kwargs.get('teacher_model')
        if teacher_model is None:
            raise ValueError("Teacher model required for distillation")

        distiller = create_distiller(
            teacher_model=teacher_model,
            student_model=model,
            distillation_type=kwargs.get('distillation_type', 'response')
        )

        dataloader = kwargs.get('dataloader')
        optimizer = kwargs.get('optimizer')
        criterion = kwargs.get('criterion')

        if any(x is None for x in [dataloader, optimizer, criterion]):
            raise ValueError("dataloader, optimizer, and criterion required for distillation")

        history = distiller.distill(dataloader, optimizer, criterion,
                                  epochs=kwargs.get('epochs', 10))
        return model, history

    elif compression_type == "nas":
        optimizer = create_architecture_optimizer(
            optimizer_type=kwargs.get('optimizer_type', 'mobilenet'),
            max_params=kwargs.get('max_params', 10_000_000)
        )
        dataloader = kwargs.get('dataloader')
        best_arch, best_fitness = optimizer.optimize(dataloader,
                                                    num_generations=kwargs.get('generations', 20))
        return best_arch, best_fitness

    else:
        raise ValueError(f"Unknown compression type: {compression_type}")

# Compression pipeline for combining multiple techniques
def compression_pipeline(model, techniques, dataloader, **kwargs):
    """
    Apply multiple compression techniques in sequence.

    Args:
        model: PyTorch model to compress
        techniques: List of compression techniques to apply in order
        dataloader: DataLoader for calibration/training
        **kwargs: Additional arguments

    Returns:
        Compressed model and combined statistics
    """
    current_model = model
    all_stats = {}

    for i, technique in enumerate(techniques):
        print(f"Applying compression technique {i+1}/{len(techniques)}: {technique}")

        if technique == "pruning":
            pruner = create_pruner(sparsity_ratio=kwargs.get('sparsity_ratio', 0.5))
            stats = pruner.prune_model(current_model, dataloader)
            all_stats['pruning'] = stats

        elif technique == "quantization":
            orchestrator = QuantizationOrchestrator()
            config = QuantizationConfig(mode=QuantizationMode.PTQ)
            current_model, stats = orchestrator.quantize_model(current_model, dataloader, config)
            all_stats['quantization'] = stats

        else:
            print(f"Warning: Unknown technique {technique}, skipping...")

    return current_model, all_stats

# Information about the compression module
def get_compression_info():
    """Get information about available compression techniques."""
    return {
        "version": __version__,
        "techniques": {
            "pruning": {
                "algorithms": ["magnitude", "gradient", "snip", "grasp", "structured", "adaptive"],
                "description": "Remove unimportant weights/neurons from the model"
            },
            "quantization": {
                "modes": ["ptq", "qat", "dynamic"],
                "description": "Reduce precision of weights and activations"
            },
            "distillation": {
                "types": ["response", "feature", "attention", "progressive", "multi_teacher", "online"],
                "description": "Transfer knowledge from teacher to smaller student model"
            },
            "architecture_optimization": {
                "optimizers": ["mobilenet", "efficientnet", "hardware_aware"],
                "description": "Find optimal architecture through neural architecture search"
            }
        },
        "quick_start": {
            "pruning": "compress_model(model, 'pruning', dataloader=dl, sparsity_ratio=0.5)",
            "quantization": "compress_model(model, 'quantization', dataloader=dl, mode='ptq')",
            "distillation": "compress_model(student, 'distillation', teacher_model=teacher, dataloader=dl, optimizer=opt, criterion=crit)",
            "nas": "compress_model(None, 'nas', dataloader=dl, max_params=5000000)"
        }
    }