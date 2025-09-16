import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import logging
import time
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
import copy

# Import all compression agents
from .model_analyzer import ModelAnalyzer, ModelAnalysis
from .pruning_agent import PruningAgent, PruningConfig, PruningResult
from .quantization_agent import QuantizationAgent, QuantizationConfig, QuantizationResult
from .knowledge_distiller import KnowledgeDistiller, DistillationConfig, DistillationResult
from .architecture_optimizer import ArchitectureOptimizer, ArchitectureConfig, OptimizationResult
from .compression_validator import CompressionValidator, ValidationConfig, ValidationMetrics
from .deployment_packager import DeploymentPackager, DeploymentConfig, DeploymentPackage
from .performance_profiler import PerformanceProfiler, ProfilingConfig, PerformanceMetrics

@dataclass
class CompressionStrategy:
    """Defines a compression strategy."""
    name: str
    techniques: List[str]  # ['pruning', 'quantization', 'distillation', 'architecture_optimization']
    target_compression: float
    accuracy_threshold: float
    priority: str = 'balanced'  # 'speed', 'size', 'accuracy', 'balanced'
    
@dataclass
class CompressionPipeline:
    """Defines a compression pipeline configuration."""
    strategy: CompressionStrategy
    validation_enabled: bool = True
    profiling_enabled: bool = True
    deployment_packaging: bool = True
    intermediate_validation: bool = True
    
@dataclass
class CompressionResults:
    """Comprehensive compression results."""
    original_model: nn.Module
    compressed_model: nn.Module
    compression_ratio: float
    accuracy_retention: float
    speedup_achieved: float
    memory_reduction: float
    
    # Detailed results from each agent
    analysis_results: Optional[ModelAnalysis] = None
    pruning_results: Optional[PruningResult] = None
    quantization_results: Optional[QuantizationResult] = None
    distillation_results: Optional[DistillationResult] = None
    optimization_results: Optional[OptimizationResult] = None
    validation_results: Optional[ValidationMetrics] = None
    deployment_package: Optional[DeploymentPackage] = None
    performance_metrics: Optional[PerformanceMetrics] = None
    
    # Pipeline metadata
    processing_time: float = 0.0
    strategy_used: Optional[str] = None
    techniques_applied: List[str] = field(default_factory=list)
    optimization_suggestions: List[str] = field(default_factory=list)
    
class CompressionOrchestrator:
    """Main orchestrator for coordinating all compression agents."""
    
    def __init__(self, device: torch.device = torch.device('cpu'), logger: Optional[logging.Logger] = None):
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize all agents
        self.analyzer = ModelAnalyzer(logger)
        self.pruner = PruningAgent(logger)
        self.quantizer = QuantizationAgent(logger)
        self.distiller = KnowledgeDistiller(device, logger)
        self.optimizer = ArchitectureOptimizer(device, logger)
        self.validator = CompressionValidator(device, logger)
        self.packager = DeploymentPackager(device, logger)
        self.profiler = PerformanceProfiler(device, logger)
        
        # Predefined strategies
        self.strategies = self._initialize_strategies()
        
    def _initialize_strategies(self) -> Dict[str, CompressionStrategy]:
        """Initialize predefined compression strategies."""
        return {
            'aggressive': CompressionStrategy(
                name='Aggressive Compression',
                techniques=['pruning', 'quantization', 'architecture_optimization'],
                target_compression=0.1,  # 10x compression
                accuracy_threshold=0.85,
                priority='size'
            ),
            'balanced': CompressionStrategy(
                name='Balanced Compression',
                techniques=['pruning', 'quantization'],
                target_compression=0.25,  # 4x compression
                accuracy_threshold=0.95,
                priority='balanced'
            ),
            'conservative': CompressionStrategy(
                name='Conservative Compression',
                techniques=['quantization'],
                target_compression=0.5,  # 2x compression
                accuracy_threshold=0.98,
                priority='accuracy'
            ),
            'speed_optimized': CompressionStrategy(
                name='Speed Optimized',
                techniques=['pruning', 'architecture_optimization'],
                target_compression=0.3,  # ~3x compression
                accuracy_threshold=0.92,
                priority='speed'
            ),
            'mobile_optimized': CompressionStrategy(
                name='Mobile Optimized',
                techniques=['pruning', 'quantization', 'distillation'],
                target_compression=0.1,  # 10x compression
                accuracy_threshold=0.88,
                priority='size'
            ),
            'knowledge_transfer': CompressionStrategy(
                name='Knowledge Transfer',
                techniques=['distillation'],
                target_compression=0.2,  # 5x compression
                accuracy_threshold=0.93,
                priority='accuracy'
            )
        }
        
    def compress_model(self, model: nn.Module,
                      train_loader: Optional[DataLoader] = None,
                      val_loader: Optional[DataLoader] = None,
                      test_loader: Optional[DataLoader] = None,
                      strategy: Union[str, CompressionStrategy] = 'balanced',
                      pipeline_config: Optional[CompressionPipeline] = None) -> CompressionResults:
        """Main compression orchestration method."""
        try:
            start_time = time.time()
            
            # Resolve strategy
            if isinstance(strategy, str):
                if strategy not in self.strategies:
                    raise ValueError(f"Unknown strategy: {strategy}. Available: {list(self.strategies.keys())}")
                strategy_obj = self.strategies[strategy]
            else:
                strategy_obj = strategy
                
            # Create pipeline config if not provided
            if pipeline_config is None:
                pipeline_config = CompressionPipeline(strategy=strategy_obj)
                
            self.logger.info(f"Starting compression with strategy: {strategy_obj.name}")
            self.logger.info(f"Techniques: {', '.join(strategy_obj.techniques)}")
            
            # Phase 1: Analysis
            self.logger.info("Phase 1: Model Analysis")
            sample_input = self._get_sample_input(val_loader or test_loader)
            analysis_results = self.analyzer.analyze_model(model, sample_input)
            
            self.logger.info(f"Original model: {analysis_results.total_params:,} parameters, "
                           f"{analysis_results.total_size_mb:.2f} MB")
            
            # Phase 2: Sequential Compression
            compressed_model = copy.deepcopy(model)
            techniques_applied = []
            detailed_results = {}
            
            # Apply compression techniques in order
            for technique in strategy_obj.techniques:
                self.logger.info(f"Applying {technique}...")
                
                if technique == 'pruning':
                    compressed_model, pruning_results = self._apply_pruning(
                        compressed_model, strategy_obj, val_loader
                    )
                    detailed_results['pruning'] = pruning_results
                    techniques_applied.append('pruning')
                    
                elif technique == 'quantization':
                    compressed_model, quantization_results = self._apply_quantization(
                        compressed_model, strategy_obj, val_loader
                    )
                    detailed_results['quantization'] = quantization_results
                    techniques_applied.append('quantization')
                    
                elif technique == 'distillation':
                    if train_loader is None:
                        self.logger.warning("Distillation requires training data, skipping")
                        continue
                    compressed_model, distillation_results = self._apply_distillation(
                        model, compressed_model, train_loader, val_loader, strategy_obj
                    )
                    detailed_results['distillation'] = distillation_results
                    techniques_applied.append('distillation')
                    
                elif technique == 'architecture_optimization':
                    if train_loader is None:
                        self.logger.warning("Architecture optimization requires training data, skipping")
                        continue
                    compressed_model, optimization_results = self._apply_architecture_optimization(
                        compressed_model, train_loader, val_loader, strategy_obj
                    )
                    detailed_results['optimization'] = optimization_results
                    techniques_applied.append('architecture_optimization')
                    
                # Intermediate validation
                if pipeline_config.intermediate_validation and val_loader is not None:
                    quick_validation = self.validator.quick_validation(
                        model, compressed_model, val_loader
                    )
                    if not quick_validation:
                        self.logger.warning(f"Intermediate validation failed after {technique}")
                        
            # Phase 3: Validation
            validation_results = None
            if pipeline_config.validation_enabled and val_loader is not None:
                self.logger.info("Phase 3: Validation")
                validation_config = ValidationConfig(
                    accuracy_threshold=strategy_obj.accuracy_threshold
                )
                validation_results = self.validator.validate_compression(
                    model, compressed_model, val_loader, validation_config
                )
                detailed_results['validation'] = validation_results
                
            # Phase 4: Performance Profiling
            performance_metrics = None
            if pipeline_config.profiling_enabled:
                self.logger.info("Phase 4: Performance Profiling")
                profiling_config = ProfilingConfig()
                performance_metrics = self.profiler.comprehensive_profile(
                    compressed_model, sample_input, profiling_config
                )
                detailed_results['performance'] = performance_metrics
                
            # Phase 5: Deployment Packaging
            deployment_package = None
            if pipeline_config.deployment_packaging:
                self.logger.info("Phase 5: Deployment Packaging")
                deployment_config = DeploymentConfig()
                # Create temporary directory for packaging
                import tempfile
                with tempfile.TemporaryDirectory() as temp_dir:
                    deployment_package = self.packager.create_deployment_package(
                        compressed_model, sample_input, temp_dir, deployment_config
                    )
                detailed_results['deployment'] = deployment_package
                
            # Calculate final metrics
            final_metrics = self._calculate_final_metrics(
                model, compressed_model, analysis_results, detailed_results, val_loader
            )
            
            # Generate optimization suggestions
            optimization_suggestions = self._generate_optimization_suggestions(
                analysis_results, detailed_results, validation_results
            )
            
            # Create comprehensive results
            processing_time = time.time() - start_time
            
            results = CompressionResults(
                original_model=model,
                compressed_model=compressed_model,
                compression_ratio=final_metrics['compression_ratio'],
                accuracy_retention=final_metrics['accuracy_retention'],
                speedup_achieved=final_metrics['speedup_achieved'],
                memory_reduction=final_metrics['memory_reduction'],
                analysis_results=analysis_results,
                pruning_results=detailed_results.get('pruning'),
                quantization_results=detailed_results.get('quantization'),
                distillation_results=detailed_results.get('distillation'),
                optimization_results=detailed_results.get('optimization'),
                validation_results=validation_results,
                deployment_package=deployment_package,
                performance_metrics=performance_metrics,
                processing_time=processing_time,
                strategy_used=strategy_obj.name,
                techniques_applied=techniques_applied,
                optimization_suggestions=optimization_suggestions
            )
            
            self.logger.info(f"Compression completed in {processing_time:.2f} seconds")
            self.logger.info(f"Achieved {final_metrics['compression_ratio']:.2f}x compression")
            self.logger.info(f"Accuracy retention: {final_metrics['accuracy_retention']:.3f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Compression orchestration failed: {e}")
            raise
            
    def _get_sample_input(self, data_loader: Optional[DataLoader]) -> torch.Tensor:
        """Get sample input for analysis."""
        if data_loader is None:
            # Default sample input (adjust based on typical use case)
            return torch.randn(1, 3, 224, 224)
        else:
            return next(iter(data_loader))[0][:1]
            
    def _apply_pruning(self, model: nn.Module, 
                      strategy: CompressionStrategy,
                      val_loader: Optional[DataLoader]) -> Tuple[nn.Module, PruningResult]:
        """Apply pruning compression."""
        # Determine pruning configuration based on strategy
        if strategy.priority == 'speed':
            pruning_type = 'structured'
            sparsity = 0.5
        elif strategy.priority == 'size':
            pruning_type = 'magnitude'
            sparsity = 0.8
        else:
            pruning_type = 'magnitude'
            sparsity = 0.6
            
        config = PruningConfig(
            pruning_type=pruning_type,
            sparsity_target=sparsity
        )
        
        result = self.pruner.prune_model(model, config)
        
        # Evaluate impact if validation data available
        if val_loader is not None:
            try:
                impact_metrics = self.pruner.evaluate_pruning_impact(
                    copy.deepcopy(model), model, val_loader, self.device
                )
                result.performance_metrics.update(impact_metrics)
            except Exception as e:
                self.logger.warning(f"Pruning impact evaluation failed: {e}")
                
        return model, result
        
    def _apply_quantization(self, model: nn.Module,
                           strategy: CompressionStrategy,
                           val_loader: Optional[DataLoader]) -> Tuple[nn.Module, QuantizationResult]:
        """Apply quantization compression."""
        # Determine quantization configuration
        if strategy.priority == 'speed':
            quant_type = 'dynamic'
        elif strategy.priority == 'size':
            quant_type = 'static'
        else:
            quant_type = 'qat'
            
        config = QuantizationConfig(
            quantization_type=quant_type,
            calibration_dataset=val_loader if quant_type == 'static' else None
        )
        
        quantized_model, result = self.quantizer.quantize_model(model, config)
        
        # Evaluate impact
        if val_loader is not None:
            try:
                impact_metrics = self.quantizer.evaluate_quantization_impact(
                    copy.deepcopy(model), quantized_model, val_loader, self.device
                )
                result.performance_metrics.update(impact_metrics)
            except Exception as e:
                self.logger.warning(f"Quantization impact evaluation failed: {e}")
                
        # Replace original model with quantized version
        return quantized_model, result
        
    def _apply_distillation(self, teacher_model: nn.Module,
                           student_model: nn.Module,
                           train_loader: DataLoader,
                           val_loader: DataLoader,
                           strategy: CompressionStrategy) -> Tuple[nn.Module, DistillationResult]:
        """Apply knowledge distillation."""
        # Create smaller student if needed
        if student_model is teacher_model:
            # Create compressed student model
            compression_ratio = strategy.target_compression
            student_model = self.distiller.create_student_model(teacher_model, compression_ratio)
            
        # Configure distillation
        distillation_type = 'response'  # Could be made strategy-dependent
        
        config = DistillationConfig(
            distillation_type=distillation_type,
            num_epochs=50,  # Reduced for orchestration
            learning_rate=0.001
        )
        
        result = self.distiller.distill_knowledge(
            teacher_model, student_model, train_loader, val_loader, config
        )
        
        return student_model, result
        
    def _apply_architecture_optimization(self, model: nn.Module,
                                        train_loader: DataLoader,
                                        val_loader: DataLoader,
                                        strategy: CompressionStrategy) -> Tuple[nn.Module, OptimizationResult]:
        """Apply architecture optimization."""
        # Configure optimization based on strategy
        if strategy.priority == 'speed':
            search_strategy = 'progressive'
            generations = 20
        else:
            search_strategy = 'evolutionary'
            generations = 30
            
        config = ArchitectureConfig(
            search_strategy=search_strategy,
            generations=generations,
            target_compression=strategy.target_compression,
            preserve_accuracy=strategy.accuracy_threshold
        )
        
        result = self.optimizer.optimize_architecture(
            model, train_loader, val_loader, config
        )
        
        # Build optimized model
        if result.best_architecture:
            # This would need to be implemented based on the architecture format
            # For now, return the original model
            optimized_model = model
        else:
            optimized_model = model
            
        return optimized_model, result
        
    def _calculate_final_metrics(self, original_model: nn.Module,
                                compressed_model: nn.Module,
                                analysis_results: ModelAnalysis,
                                detailed_results: Dict[str, Any],
                                val_loader: Optional[DataLoader]) -> Dict[str, float]:
        """Calculate final compression metrics."""
        # Model size comparison
        original_params = sum(p.numel() for p in original_model.parameters())
        compressed_params = sum(p.numel() for p in compressed_model.parameters())
        compression_ratio = original_params / compressed_params
        
        original_size_mb = sum(p.numel() * p.element_size() for p in original_model.parameters()) / (1024 * 1024)
        compressed_size_mb = sum(p.numel() * p.element_size() for p in compressed_model.parameters()) / (1024 * 1024)
        memory_reduction = 1 - (compressed_size_mb / original_size_mb)
        
        # Accuracy retention
        accuracy_retention = 1.0  # Default if no validation
        if val_loader is not None:
            try:
                original_acc = self._evaluate_accuracy(original_model, val_loader)
                compressed_acc = self._evaluate_accuracy(compressed_model, val_loader)
                accuracy_retention = compressed_acc / original_acc if original_acc > 0 else 0.0
            except Exception as e:
                self.logger.warning(f"Accuracy evaluation failed: {e}")
                
        # Speedup estimation
        speedup_achieved = compression_ratio  # Rough approximation
        
        # Refine with performance metrics if available
        performance_results = detailed_results.get('performance')
        if performance_results and hasattr(performance_results, 'throughput_metrics'):
            throughput_improvement = performance_results.throughput_metrics.get('samples_per_second', compression_ratio)
            speedup_achieved = max(speedup_achieved, throughput_improvement)
            
        return {
            'compression_ratio': compression_ratio,
            'accuracy_retention': accuracy_retention,
            'speedup_achieved': speedup_achieved,
            'memory_reduction': memory_reduction,
            'original_params': original_params,
            'compressed_params': compressed_params
        }
        
    def _evaluate_accuracy(self, model: nn.Module, val_loader: DataLoader) -> float:
        """Evaluate model accuracy."""
        model.eval()
        model.to(self.device)
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
        return correct / total
        
    def _generate_optimization_suggestions(self, analysis_results: ModelAnalysis,
                                         detailed_results: Dict[str, Any],
                                         validation_results: Optional[ValidationMetrics]) -> List[str]:
        """Generate optimization suggestions based on results."""
        suggestions = []
        
        # Analyze compression opportunities
        opportunities = analysis_results.compression_opportunities
        
        if opportunities['pruning_potential'] > 0.5:
            suggestions.append("High pruning potential detected - consider more aggressive pruning")
            
        if opportunities['quantization_potential'] > 0.7:
            suggestions.append("Model well-suited for quantization - consider INT8 or mixed precision")
            
        if opportunities['knowledge_distillation_potential'] > 0.6:
            suggestions.append("Large model detected - knowledge distillation could be very effective")
            
        # Analyze bottlenecks
        if analysis_results.bottlenecks:
            top_bottleneck = analysis_results.bottlenecks[0]
            suggestions.append(f"Focus optimization on bottleneck layer: {top_bottleneck}")
            
        # Performance-based suggestions
        performance_results = detailed_results.get('performance')
        if performance_results and hasattr(performance_results, 'optimization_suggestions'):
            suggestions.extend(performance_results.optimization_suggestions)
            
        # Validation-based suggestions
        if validation_results and not validation_results.passed_validation:
            if validation_results.accuracy_metrics['accuracy_retention'] < 0.9:
                suggestions.append("Low accuracy retention - consider less aggressive compression")
            if validation_results.performance_metrics['speedup'] < 1.5:
                suggestions.append("Low speedup achieved - consider different compression techniques")
                
        return suggestions
        
    def auto_compress(self, model: nn.Module,
                     train_loader: Optional[DataLoader] = None,
                     val_loader: Optional[DataLoader] = None,
                     test_loader: Optional[DataLoader] = None,
                     target_compression: float = 4.0,
                     accuracy_threshold: float = 0.95) -> CompressionResults:
        """Automatically determine and apply best compression strategy."""
        try:
            self.logger.info("Starting automatic compression strategy selection")
            
            # Analyze model to determine best strategy
            sample_input = self._get_sample_input(val_loader or test_loader)
            analysis = self.analyzer.analyze_model(model, sample_input)
            
            # Strategy selection based on analysis
            strategy = self._select_optimal_strategy(analysis, target_compression, accuracy_threshold)
            
            self.logger.info(f"Selected strategy: {strategy.name}")
            
            # Apply selected strategy
            return self.compress_model(
                model, train_loader, val_loader, test_loader, strategy
            )
            
        except Exception as e:
            self.logger.error(f"Auto compression failed: {e}")
            raise
            
    def _select_optimal_strategy(self, analysis: ModelAnalysis,
                                target_compression: float,
                                accuracy_threshold: float) -> CompressionStrategy:
        """Select optimal strategy based on model analysis."""
        opportunities = analysis.compression_opportunities
        
        # Score each strategy based on suitability
        strategy_scores = {}
        
        for name, strategy in self.strategies.items():
            score = 0.0
            
            # Alignment with target compression
            compression_alignment = 1.0 - abs(1.0/target_compression - 1.0/strategy.target_compression)
            score += compression_alignment * 0.3
            
            # Accuracy threshold compatibility
            accuracy_alignment = 1.0 - abs(accuracy_threshold - strategy.accuracy_threshold)
            score += accuracy_alignment * 0.2
            
            # Technique suitability
            technique_score = 0.0
            if 'pruning' in strategy.techniques:
                technique_score += opportunities['pruning_potential'] * 0.3
            if 'quantization' in strategy.techniques:
                technique_score += opportunities['quantization_potential'] * 0.3
            if 'distillation' in strategy.techniques:
                technique_score += opportunities['knowledge_distillation_potential'] * 0.2
            if 'architecture_optimization' in strategy.techniques:
                technique_score += opportunities['architecture_optimization_potential'] * 0.2
                
            score += technique_score * 0.5
            
            strategy_scores[name] = score
            
        # Select best strategy
        best_strategy_name = max(strategy_scores.keys(), key=lambda k: strategy_scores[k])
        
        # Create custom strategy with target parameters
        best_strategy = copy.deepcopy(self.strategies[best_strategy_name])
        best_strategy.target_compression = 1.0 / target_compression
        best_strategy.accuracy_threshold = accuracy_threshold
        best_strategy.name = f"Auto-Selected: {best_strategy.name}"
        
        return best_strategy
        
    def benchmark_strategies(self, model: nn.Module,
                           train_loader: Optional[DataLoader] = None,
                           val_loader: Optional[DataLoader] = None,
                           strategies: Optional[List[str]] = None) -> Dict[str, CompressionResults]:
        """Benchmark multiple compression strategies."""
        try:
            if strategies is None:
                strategies = ['conservative', 'balanced', 'aggressive']
                
            results = {}
            
            for strategy_name in strategies:
                try:
                    self.logger.info(f"Benchmarking strategy: {strategy_name}")
                    
                    # Use copy of model for each strategy
                    model_copy = copy.deepcopy(model)
                    
                    result = self.compress_model(
                        model_copy, train_loader, val_loader, None, strategy_name
                    )
                    
                    results[strategy_name] = result
                    
                    self.logger.info(f"{strategy_name}: {result.compression_ratio:.2f}x compression, "
                                   f"{result.accuracy_retention:.3f} accuracy retention")
                    
                except Exception as e:
                    self.logger.error(f"Strategy {strategy_name} failed: {e}")
                    
            return results
            
        except Exception as e:
            self.logger.error(f"Strategy benchmarking failed: {e}")
            raise
            
    def export_compression_report(self, results: CompressionResults, output_dir: str) -> None:
        """Export comprehensive compression report."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Main report
            report = {
                'compression_summary': {
                    'strategy_used': results.strategy_used,
                    'techniques_applied': results.techniques_applied,
                    'compression_ratio': results.compression_ratio,
                    'accuracy_retention': results.accuracy_retention,
                    'speedup_achieved': results.speedup_achieved,
                    'memory_reduction': results.memory_reduction,
                    'processing_time': results.processing_time,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                },
                'optimization_suggestions': results.optimization_suggestions
            }
            
            # Add detailed results if available
            if results.analysis_results:
                report['model_analysis'] = {
                    'total_parameters': results.analysis_results.total_params,
                    'model_size_mb': results.analysis_results.total_size_mb,
                    'compression_opportunities': results.analysis_results.compression_opportunities,
                    'bottlenecks': results.analysis_results.bottlenecks
                }
                
            if results.validation_results:
                report['validation_results'] = {
                    'passed_validation': results.validation_results.passed_validation,
                    'overall_score': results.validation_results.overall_score,
                    'accuracy_metrics': results.validation_results.accuracy_metrics,
                    'performance_metrics': results.validation_results.performance_metrics
                }
                
            # Save main report
            report_path = os.path.join(output_dir, 'compression_report.json')
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
                
            # Export individual agent reports
            if results.validation_results:
                validation_report_path = os.path.join(output_dir, 'validation_report.json')
                self.validator.generate_validation_report(results.validation_results, validation_report_path)
                
            if results.performance_metrics:
                performance_report_path = os.path.join(output_dir, 'performance_report.json')
                self.profiler.export_profiling_report(results.performance_metrics, performance_report_path)
                
            self.logger.info(f"Compression report exported to {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to export compression report: {e}")
            raise
            
    def create_custom_strategy(self, name: str,
                             techniques: List[str],
                             target_compression: float,
                             accuracy_threshold: float,
                             priority: str = 'balanced') -> CompressionStrategy:
        """Create custom compression strategy."""
        return CompressionStrategy(
            name=name,
            techniques=techniques,
            target_compression=1.0 / target_compression,  # Convert to ratio
            accuracy_threshold=accuracy_threshold,
            priority=priority
        )
        
    def get_available_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available strategies."""
        strategy_info = {}
        
        for name, strategy in self.strategies.items():
            strategy_info[name] = {
                'description': strategy.name,
                'techniques': strategy.techniques,
                'target_compression': f"{1.0/strategy.target_compression:.1f}x",
                'accuracy_threshold': f"{strategy.accuracy_threshold:.1%}",
                'priority': strategy.priority
            }
            
        return strategy_info
        
    def quick_compress(self, model: nn.Module,
                      target_size_mb: Optional[float] = None,
                      target_speedup: Optional[float] = None) -> CompressionResults:
        """Quick compression with simple targets."""
        # Determine strategy based on targets
        if target_size_mb is not None:
            current_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
            target_compression = current_size / target_size_mb
            strategy_name = 'aggressive' if target_compression > 5 else 'balanced'
        elif target_speedup is not None:
            target_compression = target_speedup
            strategy_name = 'speed_optimized' if target_speedup > 3 else 'balanced'
        else:
            strategy_name = 'balanced'
            
        return self.compress_model(model, strategy=strategy_name)
