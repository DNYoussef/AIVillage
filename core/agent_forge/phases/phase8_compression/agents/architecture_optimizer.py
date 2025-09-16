import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import copy
import logging
import random
from dataclasses import dataclass
from abc import ABC, abstractmethod
import itertools

@dataclass
class ArchitectureConfig:
    """Configuration for architecture optimization."""
    search_strategy: str  # 'evolutionary', 'random', 'progressive', 'efficiency'
    population_size: int = 20
    generations: int = 50
    mutation_rate: float = 0.1
    target_compression: float = 0.5
    preserve_accuracy: float = 0.95  # Minimum accuracy retention
    efficiency_weight: float = 0.3   # Weight for efficiency vs accuracy
    
@dataclass
class ArchitectureCandidate:
    """Candidate architecture with performance metrics."""
    architecture: Dict[str, Any]
    parameters: int
    flops: int
    accuracy: float
    inference_time: float
    memory_usage: float
    efficiency_score: float
    
@dataclass
class OptimizationResult:
    """Results from architecture optimization."""
    best_architecture: ArchitectureCandidate
    optimization_history: List[ArchitectureCandidate]
    compression_achieved: float
    accuracy_retained: float
    speedup_achieved: float
    
class ArchitectureSearchStrategy(ABC):
    """Abstract base class for architecture search strategies."""
    
    @abstractmethod
    def search(self, base_model: nn.Module, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              config: ArchitectureConfig) -> OptimizationResult:
        pass
        
class EvolutionarySearch(ArchitectureSearchStrategy):
    """Evolutionary architecture search."""
    
    def __init__(self, device: torch.device, logger: Optional[logging.Logger] = None):
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        
    def search(self, base_model: nn.Module, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              config: ArchitectureConfig) -> OptimizationResult:
        """Perform evolutionary architecture search."""
        try:
            # Initialize population
            population = self._initialize_population(base_model, config)
            
            # Evaluate initial population
            for candidate in population:
                self._evaluate_candidate(candidate, val_loader)
                
            optimization_history = []
            best_candidate = max(population, key=lambda x: x.efficiency_score)
            
            # Evolution loop
            for generation in range(config.generations):
                self.logger.info(f"Generation {generation + 1}/{config.generations}")
                
                # Selection
                selected = self._selection(population, config.population_size // 2)
                
                # Crossover and mutation
                offspring = []
                for i in range(0, len(selected), 2):
                    if i + 1 < len(selected):
                        child1, child2 = self._crossover(selected[i], selected[i + 1])
                        offspring.extend([child1, child2])
                        
                # Mutate offspring
                for child in offspring:
                    if random.random() < config.mutation_rate:
                        self._mutate(child, config)
                        
                # Evaluate offspring
                for child in offspring:
                    self._evaluate_candidate(child, val_loader)
                    
                # Combine and select next generation
                population = self._survival_selection(population + offspring, config.population_size)
                
                # Track best candidate
                current_best = max(population, key=lambda x: x.efficiency_score)
                if current_best.efficiency_score > best_candidate.efficiency_score:
                    best_candidate = current_best
                    
                optimization_history.extend(population)
                
                self.logger.info(f"Best efficiency score: {best_candidate.efficiency_score:.4f}")
                
            # Calculate final metrics
            original_params = sum(p.numel() for p in base_model.parameters())
            compression_achieved = original_params / best_candidate.parameters
            
            # Evaluate original model accuracy
            original_accuracy = self._evaluate_model_accuracy(base_model, val_loader)
            accuracy_retained = best_candidate.accuracy / original_accuracy
            
            # Estimate speedup (inverse of parameter ratio as approximation)
            speedup_achieved = compression_achieved
            
            return OptimizationResult(
                best_architecture=best_candidate,
                optimization_history=optimization_history,
                compression_achieved=compression_achieved,
                accuracy_retained=accuracy_retained,
                speedup_achieved=speedup_achieved
            )
            
        except Exception as e:
            self.logger.error(f"Evolutionary search failed: {e}")
            raise
            
    def _initialize_population(self, base_model: nn.Module, config: ArchitectureConfig) -> List[ArchitectureCandidate]:
        """Initialize population with random architecture variations."""
        population = []
        
        for _ in range(config.population_size):
            # Create architecture by modifying base model
            arch_config = self._generate_random_architecture(base_model, config)
            model = self._build_model_from_config(arch_config, base_model)
            
            candidate = ArchitectureCandidate(
                architecture=arch_config,
                parameters=sum(p.numel() for p in model.parameters()),
                flops=0,  # Will be calculated during evaluation
                accuracy=0.0,
                inference_time=0.0,
                memory_usage=0.0,
                efficiency_score=0.0
            )
            
            population.append(candidate)
            
        return population
        
    def _generate_random_architecture(self, base_model: nn.Module, config: ArchitectureConfig) -> Dict[str, Any]:
        """Generate random architecture configuration."""
        arch_config = {'layers': []}
        
        for name, module in base_model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Randomly modify conv layers
                new_channels = int(module.out_channels * random.uniform(0.3, 1.0))
                new_channels = max(1, new_channels)
                
                layer_config = {
                    'type': 'conv2d',
                    'name': name,
                    'in_channels': module.in_channels,
                    'out_channels': new_channels,
                    'kernel_size': module.kernel_size,
                    'stride': module.stride,
                    'padding': module.padding
                }
                
            elif isinstance(module, nn.Linear):
                # Randomly modify linear layers
                new_features = int(module.out_features * random.uniform(0.3, 1.0))
                new_features = max(1, new_features)
                
                layer_config = {
                    'type': 'linear',
                    'name': name,
                    'in_features': module.in_features,
                    'out_features': new_features
                }
                
            else:
                continue
                
            arch_config['layers'].append(layer_config)
            
        return arch_config
        
    def _build_model_from_config(self, arch_config: Dict[str, Any], base_model: nn.Module) -> nn.Module:
        """Build model from architecture configuration."""
        model = copy.deepcopy(base_model)
        
        # Apply architecture modifications
        for layer_config in arch_config['layers']:
            name = layer_config['name']
            layer_type = layer_config['type']
            
            # Find the layer in the model
            module_dict = dict(model.named_modules())
            if name in module_dict:
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                if layer_type == 'conv2d':
                    new_layer = nn.Conv2d(
                        layer_config['in_channels'],
                        layer_config['out_channels'],
                        layer_config['kernel_size'],
                        layer_config['stride'],
                        layer_config['padding']
                    )
                    
                elif layer_type == 'linear':
                    new_layer = nn.Linear(
                        layer_config['in_features'],
                        layer_config['out_features']
                    )
                    
                # Replace the layer
                if parent_name:
                    parent = module_dict[parent_name]
                    setattr(parent, child_name, new_layer)
                    
        return model
        
    def _evaluate_candidate(self, candidate: ArchitectureCandidate, val_loader: DataLoader) -> None:
        """Evaluate architecture candidate."""
        try:
            # Build model from architecture
            model = self._build_model_from_config(candidate.architecture, None)
            model.to(self.device)
            
            # Evaluate accuracy (simplified - would need proper training)
            candidate.accuracy = self._evaluate_model_accuracy(model, val_loader)
            
            # Calculate FLOPs
            candidate.flops = self._calculate_flops(model, next(iter(val_loader))[0][:1])
            
            # Measure inference time
            candidate.inference_time = self._measure_inference_time(model, val_loader)
            
            # Estimate memory usage
            candidate.memory_usage = self._estimate_memory_usage(model)
            
            # Calculate efficiency score
            candidate.efficiency_score = self._calculate_efficiency_score(candidate)
            
        except Exception as e:
            self.logger.warning(f"Failed to evaluate candidate: {e}")
            candidate.efficiency_score = 0.0
            
    def _calculate_efficiency_score(self, candidate: ArchitectureCandidate) -> float:
        """Calculate efficiency score combining accuracy and efficiency metrics."""
        # Normalize metrics
        accuracy_score = candidate.accuracy
        
        # Efficiency metrics (higher is better, so take inverse)
        param_efficiency = 1.0 / (candidate.parameters / 1e6 + 1)  # Inverse of millions of params
        flop_efficiency = 1.0 / (candidate.flops / 1e9 + 1)  # Inverse of GFLOPs
        time_efficiency = 1.0 / (candidate.inference_time + 1e-6)  # Inverse of time
        
        # Combined efficiency score
        efficiency = (param_efficiency + flop_efficiency + time_efficiency) / 3
        
        # Weighted combination
        score = 0.7 * accuracy_score + 0.3 * efficiency
        
        return score
        
    def _selection(self, population: List[ArchitectureCandidate], num_selected: int) -> List[ArchitectureCandidate]:
        """Tournament selection."""
        selected = []
        
        for _ in range(num_selected):
            # Tournament size of 3
            tournament = random.sample(population, min(3, len(population)))
            winner = max(tournament, key=lambda x: x.efficiency_score)
            selected.append(winner)
            
        return selected
        
    def _crossover(self, parent1: ArchitectureCandidate, parent2: ArchitectureCandidate) -> Tuple[ArchitectureCandidate, ArchitectureCandidate]:
        """Single-point crossover."""
        # Simple crossover by mixing layer configurations
        arch1 = copy.deepcopy(parent1.architecture)
        arch2 = copy.deepcopy(parent2.architecture)
        
        if len(arch1['layers']) > 1 and len(arch2['layers']) > 1:
            crossover_point = random.randint(1, min(len(arch1['layers']), len(arch2['layers'])) - 1)
            
            # Swap layer configurations after crossover point
            arch1['layers'][crossover_point:], arch2['layers'][crossover_point:] = \
                arch2['layers'][crossover_point:], arch1['layers'][crossover_point:]
                
        child1 = ArchitectureCandidate(
            architecture=arch1,
            parameters=0, flops=0, accuracy=0.0,
            inference_time=0.0, memory_usage=0.0, efficiency_score=0.0
        )
        
        child2 = ArchitectureCandidate(
            architecture=arch2,
            parameters=0, flops=0, accuracy=0.0,
            inference_time=0.0, memory_usage=0.0, efficiency_score=0.0
        )
        
        return child1, child2
        
    def _mutate(self, candidate: ArchitectureCandidate, config: ArchitectureConfig) -> None:
        """Mutate architecture candidate."""
        if not candidate.architecture['layers']:
            return
            
        # Randomly select a layer to mutate
        layer_idx = random.randint(0, len(candidate.architecture['layers']) - 1)
        layer = candidate.architecture['layers'][layer_idx]
        
        if layer['type'] == 'conv2d':
            # Mutate output channels
            current_channels = layer['out_channels']
            mutation_factor = random.uniform(0.8, 1.2)
            new_channels = max(1, int(current_channels * mutation_factor))
            layer['out_channels'] = new_channels
            
        elif layer['type'] == 'linear':
            # Mutate output features
            current_features = layer['out_features']
            mutation_factor = random.uniform(0.8, 1.2)
            new_features = max(1, int(current_features * mutation_factor))
            layer['out_features'] = new_features
            
    def _survival_selection(self, population: List[ArchitectureCandidate], population_size: int) -> List[ArchitectureCandidate]:
        """Select survivors for next generation."""
        # Sort by efficiency score and take top candidates
        sorted_population = sorted(population, key=lambda x: x.efficiency_score, reverse=True)
        return sorted_population[:population_size]
        
    def _evaluate_model_accuracy(self, model: nn.Module, val_loader: DataLoader) -> float:
        """Evaluate model accuracy (simplified)."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i, (data, target) in enumerate(val_loader):
                if i >= 10:  # Limit for efficiency during search
                    break
                    
                data, target = data.to(self.device), target.to(self.device)
                
                try:
                    output = model(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)
                except Exception:
                    # Model might have incompatible dimensions
                    return 0.0
                    
        return correct / total if total > 0 else 0.0
        
    def _calculate_flops(self, model: nn.Module, sample_input: torch.Tensor) -> int:
        """Calculate FLOPs for the model."""
        total_flops = 0
        
        def flop_hook(name):
            def hook(module, input, output):
                nonlocal total_flops
                if isinstance(module, nn.Conv2d):
                    if isinstance(output, torch.Tensor):
                        kernel_flops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
                        output_elements = output.numel()
                        total_flops += output_elements * kernel_flops
                        
                elif isinstance(module, nn.Linear):
                    if isinstance(output, torch.Tensor):
                        batch_size = output.shape[0]
                        total_flops += batch_size * module.in_features * module.out_features
            return hook
            
        # Register hooks
        handles = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                handle = module.register_forward_hook(flop_hook(name))
                handles.append(handle)
                
        try:
            with torch.no_grad():
                model(sample_input.to(self.device))
        except Exception:
            total_flops = 0
            
        # Clean up
        for handle in handles:
            handle.remove()
            
        return total_flops
        
    def _measure_inference_time(self, model: nn.Module, val_loader: DataLoader) -> float:
        """Measure inference time."""
        import time
        
        model.eval()
        times = []
        
        with torch.no_grad():
            for i, (data, _) in enumerate(val_loader):
                if i >= 10:  # Limit for efficiency
                    break
                    
                data = data.to(self.device)
                
                try:
                    start_time = time.time()
                    model(data)
                    end_time = time.time()
                    times.append(end_time - start_time)
                except Exception:
                    return float('inf')
                    
        return np.mean(times) if times else float('inf')
        
    def _estimate_memory_usage(self, model: nn.Module) -> float:
        """Estimate memory usage in MB."""
        total_memory = 0
        
        for param in model.parameters():
            total_memory += param.numel() * param.element_size()
            
        return total_memory / (1024 * 1024)  # Convert to MB
        
class ProgressiveSearch(ArchitectureSearchStrategy):
    """Progressive architecture search with gradual compression."""
    
    def __init__(self, device: torch.device, logger: Optional[logging.Logger] = None):
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        
    def search(self, base_model: nn.Module, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              config: ArchitectureConfig) -> OptimizationResult:
        """Perform progressive architecture search."""
        try:
            current_model = copy.deepcopy(base_model)
            optimization_history = []
            
            # Progressive compression stages
            compression_stages = [0.9, 0.7, 0.5, 0.3]  # Gradual compression
            
            for stage, target_compression in enumerate(compression_stages):
                self.logger.info(f"Compression stage {stage + 1}: target = {target_compression}")
                
                # Find best architecture for this compression level
                best_arch = self._find_best_architecture_for_compression(
                    current_model, val_loader, target_compression, config
                )
                
                if best_arch:
                    current_model = self._build_model_from_config(best_arch.architecture, current_model)
                    optimization_history.append(best_arch)
                    
                    # Early stopping if accuracy drops too much
                    original_accuracy = self._evaluate_model_accuracy(base_model, val_loader)
                    if best_arch.accuracy < original_accuracy * config.preserve_accuracy:
                        self.logger.info(f"Stopping early due to accuracy constraint")
                        break
                        
            best_candidate = optimization_history[-1] if optimization_history else None
            
            if best_candidate is None:
                raise ValueError("No valid architecture found")
                
            # Calculate final metrics
            original_params = sum(p.numel() for p in base_model.parameters())
            compression_achieved = original_params / best_candidate.parameters
            
            original_accuracy = self._evaluate_model_accuracy(base_model, val_loader)
            accuracy_retained = best_candidate.accuracy / original_accuracy
            
            speedup_achieved = compression_achieved
            
            return OptimizationResult(
                best_architecture=best_candidate,
                optimization_history=optimization_history,
                compression_achieved=compression_achieved,
                accuracy_retained=accuracy_retained,
                speedup_achieved=speedup_achieved
            )
            
        except Exception as e:
            self.logger.error(f"Progressive search failed: {e}")
            raise
            
    def _find_best_architecture_for_compression(self, 
                                              base_model: nn.Module,
                                              val_loader: DataLoader,
                                              target_compression: float,
                                              config: ArchitectureConfig) -> Optional[ArchitectureCandidate]:
        """Find best architecture for specific compression ratio."""
        candidates = []
        
        # Generate multiple candidates
        for _ in range(20):
            arch_config = self._generate_compressed_architecture(base_model, target_compression)
            
            candidate = ArchitectureCandidate(
                architecture=arch_config,
                parameters=0, flops=0, accuracy=0.0,
                inference_time=0.0, memory_usage=0.0, efficiency_score=0.0
            )
            
            try:
                model = self._build_model_from_config(arch_config, base_model)
                candidate.parameters = sum(p.numel() for p in model.parameters())
                candidate.accuracy = self._evaluate_model_accuracy(model, val_loader)
                candidate.efficiency_score = candidate.accuracy  # Simplified
                
                candidates.append(candidate)
                
            except Exception as e:
                self.logger.warning(f"Failed to evaluate candidate: {e}")
                continue
                
        return max(candidates, key=lambda x: x.efficiency_score) if candidates else None
        
    def _generate_compressed_architecture(self, base_model: nn.Module, compression_ratio: float) -> Dict[str, Any]:
        """Generate architecture with specific compression ratio."""
        arch_config = {'layers': []}
        
        for name, module in base_model.named_modules():
            if isinstance(module, nn.Conv2d):
                new_channels = max(1, int(module.out_channels * compression_ratio))
                
                layer_config = {
                    'type': 'conv2d',
                    'name': name,
                    'in_channels': module.in_channels,
                    'out_channels': new_channels,
                    'kernel_size': module.kernel_size,
                    'stride': module.stride,
                    'padding': module.padding
                }
                
            elif isinstance(module, nn.Linear):
                new_features = max(1, int(module.out_features * compression_ratio))
                
                layer_config = {
                    'type': 'linear',
                    'name': name,
                    'in_features': module.in_features,
                    'out_features': new_features
                }
                
            else:
                continue
                
            arch_config['layers'].append(layer_config)
            
        return arch_config
        
    def _build_model_from_config(self, arch_config: Dict[str, Any], base_model: nn.Module) -> nn.Module:
        """Build model from architecture configuration."""
        model = copy.deepcopy(base_model)
        
        # Apply modifications (same as evolutionary search)
        module_dict = dict(model.named_modules())
        
        for layer_config in arch_config['layers']:
            name = layer_config['name']
            layer_type = layer_config['type']
            
            if name in module_dict:
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                if layer_type == 'conv2d':
                    new_layer = nn.Conv2d(
                        layer_config['in_channels'],
                        layer_config['out_channels'],
                        layer_config['kernel_size'],
                        layer_config['stride'],
                        layer_config['padding']
                    )
                    
                elif layer_type == 'linear':
                    new_layer = nn.Linear(
                        layer_config['in_features'],
                        layer_config['out_features']
                    )
                    
                if parent_name and parent_name in module_dict:
                    parent = module_dict[parent_name]
                    setattr(parent, child_name, new_layer)
                    
        return model
        
    def _evaluate_model_accuracy(self, model: nn.Module, val_loader: DataLoader) -> float:
        """Evaluate model accuracy."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i, (data, target) in enumerate(val_loader):
                if i >= 10:  # Limit for efficiency
                    break
                    
                data, target = data.to(self.device), target.to(self.device)
                
                try:
                    output = model(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)
                except Exception:
                    return 0.0
                    
        return correct / total if total > 0 else 0.0
        
class ArchitectureOptimizer:
    """Main architecture optimization agent."""
    
    def __init__(self, device: torch.device = torch.device('cpu'), logger: Optional[logging.Logger] = None):
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        self.strategies = {
            'evolutionary': EvolutionarySearch(device, logger),
            'progressive': ProgressiveSearch(device, logger)
        }
        
    def optimize_architecture(self, model: nn.Module,
                            train_loader: DataLoader,
                            val_loader: DataLoader,
                            config: ArchitectureConfig) -> OptimizationResult:
        """Optimize model architecture using specified strategy."""
        try:
            if config.search_strategy not in self.strategies:
                raise ValueError(f"Unknown search strategy: {config.search_strategy}")
                
            strategy = self.strategies[config.search_strategy]
            result = strategy.search(model, train_loader, val_loader, config)
            
            self.logger.info(f"Architecture optimization completed: "
                           f"{result.compression_achieved:.2f}x compression, "
                           f"{result.accuracy_retained:.3f} accuracy retention")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Architecture optimization failed: {e}")
            raise
            
    def create_config(self, search_strategy: str, **kwargs) -> ArchitectureConfig:
        """Create architecture optimization configuration."""
        return ArchitectureConfig(
            search_strategy=search_strategy,
            **kwargs
        )
        
    def analyze_architecture_efficiency(self, model: nn.Module, 
                                      sample_input: torch.Tensor,
                                      test_loader: DataLoader) -> Dict[str, Any]:
        """Analyze architecture efficiency metrics."""
        try:
            model.to(self.device)
            model.eval()
            
            analysis = {}
            
            # Parameter count
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            analysis['total_parameters'] = total_params
            analysis['trainable_parameters'] = trainable_params
            
            # Model size
            model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
            analysis['model_size_mb'] = model_size
            
            # FLOPs calculation
            flops = self._calculate_detailed_flops(model, sample_input)
            analysis['flops'] = flops
            analysis['gflops'] = flops / 1e9
            
            # Layer-wise analysis
            layer_analysis = self._analyze_layers(model)
            analysis['layer_analysis'] = layer_analysis
            
            # Efficiency metrics
            accuracy = self._evaluate_accuracy(model, test_loader)
            analysis['accuracy'] = accuracy
            analysis['parameters_per_accuracy'] = total_params / (accuracy + 1e-8)
            analysis['flops_per_accuracy'] = flops / (accuracy + 1e-8)
            
            # Inference speed
            inference_time = self._measure_detailed_inference_time(model, test_loader)
            analysis['inference_time_ms'] = inference_time
            analysis['throughput_samples_per_sec'] = 1000.0 / inference_time if inference_time > 0 else 0
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Architecture analysis failed: {e}")
            raise
            
    def _calculate_detailed_flops(self, model: nn.Module, sample_input: torch.Tensor) -> int:
        """Calculate detailed FLOPs."""
        total_flops = 0
        
        def flop_hook(name):
            def hook(module, input, output):
                nonlocal total_flops
                if isinstance(module, nn.Conv2d):
                    if isinstance(output, torch.Tensor):
                        kernel_flops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
                        output_elements = output.numel()
                        total_flops += output_elements * kernel_flops
                        if module.bias is not None:
                            total_flops += output_elements
                            
                elif isinstance(module, nn.Linear):
                    if isinstance(output, torch.Tensor):
                        batch_size = output.shape[0]
                        total_flops += batch_size * module.in_features * module.out_features
                        if module.bias is not None:
                            total_flops += batch_size * module.out_features
                            
                elif isinstance(module, nn.BatchNorm2d):
                    if isinstance(output, torch.Tensor):
                        total_flops += output.numel() * 2  # Scale and shift
                        
            return hook
            
        # Register hooks
        handles = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                handle = module.register_forward_hook(flop_hook(name))
                handles.append(handle)
                
        # Forward pass
        with torch.no_grad():
            model(sample_input.to(self.device))
            
        # Clean up
        for handle in handles:
            handle.remove()
            
        return total_flops
        
    def _analyze_layers(self, model: nn.Module) -> Dict[str, Dict[str, Any]]:
        """Analyze individual layers."""
        layer_analysis = {}
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                params = sum(p.numel() for p in module.parameters())
                
                layer_info = {
                    'type': type(module).__name__,
                    'parameters': params
                }
                
                if isinstance(module, nn.Conv2d):
                    layer_info.update({
                        'in_channels': module.in_channels,
                        'out_channels': module.out_channels,
                        'kernel_size': module.kernel_size,
                        'stride': module.stride,
                        'padding': module.padding
                    })
                    
                elif isinstance(module, nn.Linear):
                    layer_info.update({
                        'in_features': module.in_features,
                        'out_features': module.out_features
                    })
                    
                layer_analysis[name] = layer_info
                
        return layer_analysis
        
    def _evaluate_accuracy(self, model: nn.Module, test_loader: DataLoader) -> float:
        """Evaluate model accuracy."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
        return correct / total
        
    def _measure_detailed_inference_time(self, model: nn.Module, test_loader: DataLoader) -> float:
        """Measure detailed inference time."""
        import time
        
        model.eval()
        times = []
        
        # Warm up
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                if i >= 5:
                    break
                data = data.to(self.device)
                model(data)
                
        # Measure
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                if i >= 100:
                    break
                    
                data = data.to(self.device)
                
                start_time = time.time()
                model(data)
                end_time = time.time()
                
                batch_time = (end_time - start_time) * 1000  # Convert to ms
                per_sample_time = batch_time / data.size(0)
                times.append(per_sample_time)
                
        return np.mean(times) if times else 0.0
        
    def compare_architectures(self, architectures: List[nn.Module],
                            names: List[str],
                            sample_input: torch.Tensor,
                            test_loader: DataLoader) -> Dict[str, Dict[str, Any]]:
        """Compare multiple architectures."""
        results = {}
        
        for model, name in zip(architectures, names):
            try:
                analysis = self.analyze_architecture_efficiency(model, sample_input, test_loader)
                results[name] = analysis
            except Exception as e:
                self.logger.error(f"Failed to analyze {name}: {e}")
                results[name] = {'error': str(e)}
                
        return results
