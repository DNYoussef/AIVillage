"""
Neural Architecture Optimization
Implements neural architecture search and optimization for efficient model compression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
from typing import Dict, List, Tuple, Optional, Union, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
import random
import math

logger = logging.getLogger(__name__)

@dataclass
class ArchitectureConfig:
    """Configuration for architecture optimization."""
    search_space: Dict[str, List] = field(default_factory=lambda: {
        'depth': [8, 16, 32, 50],
        'width_multiplier': [0.25, 0.5, 0.75, 1.0],
        'kernel_sizes': [3, 5, 7],
        'activation': ['relu', 'gelu', 'swish'],
        'normalization': ['batch', 'layer', 'none']
    })
    max_params: int = 10_000_000  # 10M parameters
    max_flops: float = 1e9  # 1G FLOPs
    target_accuracy: float = 0.9
    population_size: int = 50
    generations: int = 20
    mutation_rate: float = 0.1
    hardware_constraint: str = 'mobile'  # mobile, edge, server

class BaseArchitectureOptimizer(ABC):
    """Base class for neural architecture optimization."""

    def __init__(self, config: ArchitectureConfig):
        self.config = config
        self.population = []
        self.fitness_history = []
        self.best_architecture = None
        self.best_fitness = float('-inf')

    @abstractmethod
    def generate_architecture(self, constraints: Optional[Dict] = None) -> Dict:
        """Generate a single architecture."""
        pass

    @abstractmethod
    def evaluate_architecture(self, architecture: Dict,
                            dataloader: torch.utils.data.DataLoader) -> float:
        """Evaluate architecture fitness."""
        pass

    @abstractmethod
    def mutate_architecture(self, architecture: Dict) -> Dict:
        """Mutate an architecture."""
        pass

    def optimize(self, dataloader: torch.utils.data.DataLoader,
                num_generations: int = None) -> Tuple[Dict, float]:
        """Run architecture optimization."""
        generations = num_generations or self.config.generations

        # Initialize population
        logger.info("Initializing population...")
        self.population = [self.generate_architecture()
                          for _ in range(self.config.population_size)]

        for generation in range(generations):
            logger.info(f"Generation {generation + 1}/{generations}")

            # Evaluate population
            fitness_scores = []
            for i, arch in enumerate(self.population):
                try:
                    fitness = self.evaluate_architecture(arch, dataloader)
                    fitness_scores.append(fitness)

                    # Track best
                    if fitness > self.best_fitness:
                        self.best_fitness = fitness
                        self.best_architecture = arch.copy()

                except Exception as e:
                    logger.warning(f"Architecture {i} failed evaluation: {e}")
                    fitness_scores.append(float('-inf'))

            self.fitness_history.append(max(fitness_scores))

            # Selection and reproduction
            new_population = self._evolve_population(fitness_scores)
            self.population = new_population

            logger.info(f"Best fitness: {self.best_fitness:.4f}")

        return self.best_architecture, self.best_fitness

    def _evolve_population(self, fitness_scores: List[float]) -> List[Dict]:
        """Evolve population using selection, crossover, and mutation."""
        # Selection (tournament selection)
        selected = []
        for _ in range(self.config.population_size):
            tournament_size = 3
            tournament_indices = random.sample(range(len(self.population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
            selected.append(self.population[winner_idx].copy())

        # Mutation
        new_population = []
        for arch in selected:
            if random.random() < self.config.mutation_rate:
                arch = self.mutate_architecture(arch)
            new_population.append(arch)

        return new_population

class MobileNetOptimizer(BaseArchitectureOptimizer):
    """Optimizer for MobileNet-like architectures."""

    def generate_architecture(self, constraints: Optional[Dict] = None) -> Dict:
        """Generate MobileNet architecture configuration."""
        search_space = self.config.search_space

        # Base architecture parameters
        arch = {
            'input_channels': 3,
            'num_classes': 1000,
            'width_multiplier': random.choice(search_space['width_multiplier']),
            'depth_multiplier': 1.0,
            'blocks': []
        }

        # Generate depthwise separable blocks
        in_channels = int(32 * arch['width_multiplier'])
        block_configs = [
            # (out_channels, stride, num_blocks)
            (64, 1, 1),
            (128, 2, 2),
            (256, 2, 2),
            (512, 2, 6),
            (1024, 2, 2)
        ]

        for out_channels, stride, num_blocks in block_configs:
            out_channels = int(out_channels * arch['width_multiplier'])

            for i in range(num_blocks):
                block = {
                    'type': 'depthwise_separable',
                    'in_channels': in_channels,
                    'out_channels': out_channels,
                    'stride': stride if i == 0 else 1,
                    'kernel_size': random.choice(search_space['kernel_sizes']),
                    'activation': random.choice(search_space['activation']),
                    'normalization': random.choice(search_space['normalization'])
                }
                arch['blocks'].append(block)
                in_channels = out_channels

        return arch

    def mutate_architecture(self, architecture: Dict) -> Dict:
        """Mutate MobileNet architecture."""
        arch = architecture.copy()
        search_space = self.config.search_space

        # Possible mutations
        mutations = ['width_multiplier', 'kernel_size', 'activation', 'normalization']
        mutation = random.choice(mutations)

        if mutation == 'width_multiplier':
            arch['width_multiplier'] = random.choice(search_space['width_multiplier'])
            # Update all block channels
            for i, block in enumerate(arch['blocks']):
                if i == 0:
                    block['in_channels'] = int(32 * arch['width_multiplier'])
                block['out_channels'] = int(block['out_channels'] / architecture['width_multiplier'] * arch['width_multiplier'])

        elif mutation in ['kernel_size', 'activation', 'normalization']:
            # Mutate random block
            block_idx = random.randint(0, len(arch['blocks']) - 1)
            if mutation == 'kernel_size':
                arch['blocks'][block_idx]['kernel_size'] = random.choice(search_space['kernel_sizes'])
            elif mutation == 'activation':
                arch['blocks'][block_idx]['activation'] = random.choice(search_space['activation'])
            elif mutation == 'normalization':
                arch['blocks'][block_idx]['normalization'] = random.choice(search_space['normalization'])

        return arch

    def evaluate_architecture(self, architecture: Dict,
                            dataloader: torch.utils.data.DataLoader) -> float:
        """Evaluate MobileNet architecture."""
        try:
            # Build model from architecture
            model = self._build_mobilenet(architecture)

            # Calculate model statistics
            params = self._count_parameters(model)
            flops = self._estimate_flops(model, (1, 3, 224, 224))

            # Check constraints
            if params > self.config.max_params:
                return float('-inf')
            if flops > self.config.max_flops:
                return float('-inf')

            # Estimate accuracy (simplified - in practice, train/validate)
            accuracy = self._estimate_accuracy(model, architecture)

            # Multi-objective fitness (accuracy, efficiency)
            efficiency_score = 1.0 / (params / 1e6 + flops / 1e9)  # Smaller is better
            fitness = accuracy + 0.1 * efficiency_score

            return fitness

        except Exception as e:
            logger.error(f"Failed to evaluate architecture: {e}")
            return float('-inf')

    def _build_mobilenet(self, architecture: Dict) -> nn.Module:
        """Build MobileNet model from architecture description."""
        class DepthwiseSeparableBlock(nn.Module):
            def __init__(self, in_channels, out_channels, stride, kernel_size,
                        activation, normalization):
                super().__init__()

                # Depthwise convolution
                self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                         stride=stride, padding=kernel_size//2,
                                         groups=in_channels, bias=False)

                # Normalization
                if normalization == 'batch':
                    self.dw_norm = nn.BatchNorm2d(in_channels)
                elif normalization == 'layer':
                    self.dw_norm = nn.GroupNorm(1, in_channels)
                else:
                    self.dw_norm = nn.Identity()

                # Pointwise convolution
                self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)

                if normalization == 'batch':
                    self.pw_norm = nn.BatchNorm2d(out_channels)
                elif normalization == 'layer':
                    self.pw_norm = nn.GroupNorm(1, out_channels)
                else:
                    self.pw_norm = nn.Identity()

                # Activation
                if activation == 'relu':
                    self.activation = nn.ReLU(inplace=True)
                elif activation == 'gelu':
                    self.activation = nn.GELU()
                elif activation == 'swish':
                    self.activation = nn.SiLU()
                else:
                    self.activation = nn.ReLU(inplace=True)

            def forward(self, x):
                x = self.depthwise(x)
                x = self.dw_norm(x)
                x = self.activation(x)
                x = self.pointwise(x)
                x = self.pw_norm(x)
                x = self.activation(x)
                return x

        class MobileNet(nn.Module):
            def __init__(self, arch):
                super().__init__()

                # Initial convolution
                self.conv1 = nn.Conv2d(3, int(32 * arch['width_multiplier']),
                                     3, stride=2, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(int(32 * arch['width_multiplier']))
                self.relu = nn.ReLU(inplace=True)

                # Depthwise separable blocks
                self.blocks = nn.ModuleList()
                for block_config in arch['blocks']:
                    block = DepthwiseSeparableBlock(
                        block_config['in_channels'],
                        block_config['out_channels'],
                        block_config['stride'],
                        block_config['kernel_size'],
                        block_config['activation'],
                        block_config['normalization']
                    )
                    self.blocks.append(block)

                # Global average pooling and classifier
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                final_channels = arch['blocks'][-1]['out_channels']
                self.classifier = nn.Linear(final_channels, arch['num_classes'])

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)

                for block in self.blocks:
                    x = block(x)

                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)

                return x

        return MobileNet(architecture)

    def _count_parameters(self, model: nn.Module) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _estimate_flops(self, model: nn.Module, input_shape: Tuple[int, ...]) -> float:
        """Estimate FLOPs (simplified calculation)."""
        total_flops = 0

        def flop_count_hook(module, input, output):
            nonlocal total_flops

            if isinstance(module, nn.Conv2d):
                # FLOPs = output_elements * (kernel_size^2 * in_channels + bias)
                output_dims = output.shape[2:]
                kernel_flops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
                total_flops += np.prod(output_dims) * kernel_flops * module.out_channels

            elif isinstance(module, nn.Linear):
                total_flops += module.in_features * module.out_features

        # Register hooks
        hooks = []
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hook = module.register_forward_hook(flop_count_hook)
                hooks.append(hook)

        # Forward pass
        dummy_input = torch.randn(input_shape)
        model.eval()
        with torch.no_grad():
            _ = model(dummy_input)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return total_flops

    def _estimate_accuracy(self, model: nn.Module, architecture: Dict) -> float:
        """Estimate model accuracy (heuristic-based)."""
        # Simple heuristic based on architecture properties
        width_mult = architecture['width_multiplier']
        num_blocks = len(architecture['blocks'])

        # Baseline accuracy estimates
        base_accuracy = 0.7  # Assume 70% baseline

        # Width multiplier impact
        width_bonus = (width_mult - 0.25) * 0.2  # More width = better accuracy

        # Depth impact (with diminishing returns)
        depth_bonus = math.log(num_blocks / 10 + 1) * 0.1

        # Random noise to simulate training variance
        noise = random.gauss(0, 0.05)

        estimated_accuracy = base_accuracy + width_bonus + depth_bonus + noise
        return max(0, min(1.0, estimated_accuracy))  # Clamp to [0, 1]

class EfficientNetOptimizer(BaseArchitectureOptimizer):
    """Optimizer for EfficientNet-like architectures with compound scaling."""

    def generate_architecture(self, constraints: Optional[Dict] = None) -> Dict:
        """Generate EfficientNet architecture with compound scaling."""
        search_space = self.config.search_space

        # Compound scaling coefficients
        phi = random.uniform(0, 3)  # Compound coefficient
        alpha = 1.2  # Depth scaling
        beta = 1.1   # Width scaling
        gamma = 1.15 # Resolution scaling

        depth_multiplier = alpha ** phi
        width_multiplier = beta ** phi

        arch = {
            'compound_coeff': phi,
            'depth_multiplier': depth_multiplier,
            'width_multiplier': width_multiplier,
            'input_size': int(224 * (gamma ** phi)),
            'blocks': []
        }

        # Base EfficientNet blocks (MBConv)
        base_blocks = [
            # (out_channels, num_blocks, stride, expand_ratio)
            (16, 1, 1, 1),
            (24, 2, 2, 6),
            (40, 2, 2, 6),
            (80, 3, 2, 6),
            (112, 3, 1, 6),
            (192, 4, 2, 6),
            (320, 1, 1, 6)
        ]

        in_channels = int(32 * width_multiplier)

        for out_channels, num_blocks, stride, expand_ratio in base_blocks:
            out_channels = int(out_channels * width_multiplier)
            num_blocks = int(math.ceil(num_blocks * depth_multiplier))

            for i in range(num_blocks):
                block = {
                    'type': 'mbconv',
                    'in_channels': in_channels,
                    'out_channels': out_channels,
                    'stride': stride if i == 0 else 1,
                    'expand_ratio': expand_ratio,
                    'kernel_size': random.choice(search_space['kernel_sizes']),
                    'se_ratio': 0.25,  # Squeeze-and-excitation ratio
                    'activation': random.choice(search_space['activation']),
                    'normalization': random.choice(search_space['normalization'])
                }
                arch['blocks'].append(block)
                in_channels = out_channels

        return arch

    def mutate_architecture(self, architecture: Dict) -> Dict:
        """Mutate EfficientNet architecture."""
        arch = architecture.copy()

        # Mutate compound coefficient
        if random.random() < 0.3:
            arch['compound_coeff'] += random.gauss(0, 0.2)
            arch['compound_coeff'] = max(0, min(3, arch['compound_coeff']))

            # Recalculate multipliers
            phi = arch['compound_coeff']
            arch['depth_multiplier'] = 1.2 ** phi
            arch['width_multiplier'] = 1.1 ** phi
            arch['input_size'] = int(224 * (1.15 ** phi))

        # Mutate individual block properties
        if random.random() < 0.5:
            block_idx = random.randint(0, len(arch['blocks']) - 1)
            block = arch['blocks'][block_idx]

            mutations = ['kernel_size', 'expand_ratio', 'se_ratio']
            mutation = random.choice(mutations)

            if mutation == 'kernel_size':
                block['kernel_size'] = random.choice(self.config.search_space['kernel_sizes'])
            elif mutation == 'expand_ratio':
                block['expand_ratio'] = random.choice([1, 3, 6])
            elif mutation == 'se_ratio':
                block['se_ratio'] = random.choice([0, 0.25])

        return arch

    def evaluate_architecture(self, architecture: Dict,
                            dataloader: torch.utils.data.DataLoader) -> float:
        """Evaluate EfficientNet architecture."""
        try:
            # Calculate model complexity
            params = self._estimate_params(architecture)
            flops = self._estimate_flops_efficient(architecture)

            # Check constraints
            if params > self.config.max_params:
                return float('-inf')
            if flops > self.config.max_flops:
                return float('-inf')

            # Estimate accuracy based on compound scaling
            accuracy = self._estimate_efficientnet_accuracy(architecture)

            # Multi-objective fitness
            efficiency_score = 1.0 / (params / 1e6 + flops / 1e9)
            fitness = accuracy + 0.05 * efficiency_score

            return fitness

        except Exception as e:
            logger.error(f"Failed to evaluate EfficientNet architecture: {e}")
            return float('-inf')

    def _estimate_params(self, architecture: Dict) -> int:
        """Estimate parameter count for EfficientNet."""
        total_params = 0

        # Initial conv
        total_params += 3 * 32 * 3 * 3  # Rough estimate

        # MBConv blocks
        for block in architecture['blocks']:
            in_ch = block['in_channels']
            out_ch = block['out_channels']
            expand = block['expand_ratio']

            # Expansion conv
            if expand != 1:
                total_params += in_ch * (in_ch * expand)

            # Depthwise conv
            total_params += (in_ch * expand) * block['kernel_size'] ** 2

            # SE module
            if block['se_ratio'] > 0:
                se_ch = max(1, int(in_ch * block['se_ratio']))
                total_params += (in_ch * expand) * se_ch * 2

            # Projection conv
            total_params += (in_ch * expand) * out_ch

        # Final classifier
        total_params += architecture['blocks'][-1]['out_channels'] * 1000

        return int(total_params * architecture['width_multiplier'] ** 2)

    def _estimate_flops_efficient(self, architecture: Dict) -> float:
        """Estimate FLOPs for EfficientNet."""
        input_size = architecture['input_size']
        total_flops = 0

        current_size = input_size // 2  # After initial conv

        for block in architecture['blocks']:
            in_ch = block['in_channels']
            out_ch = block['out_channels']
            expand = block['expand_ratio']
            stride = block['stride']

            # Update spatial dimensions
            if stride == 2:
                current_size //= 2

            # MBConv FLOPs
            expanded_ch = in_ch * expand

            # Expansion
            if expand != 1:
                total_flops += current_size ** 2 * in_ch * expanded_ch

            # Depthwise
            total_flops += current_size ** 2 * expanded_ch * block['kernel_size'] ** 2

            # SE module
            if block['se_ratio'] > 0:
                se_ch = max(1, int(in_ch * block['se_ratio']))
                total_flops += expanded_ch * se_ch * 2

            # Projection
            total_flops += current_size ** 2 * expanded_ch * out_ch

        return total_flops

    def _estimate_efficientnet_accuracy(self, architecture: Dict) -> float:
        """Estimate EfficientNet accuracy based on compound scaling."""
        phi = architecture['compound_coeff']

        # Base accuracy for EfficientNet-B0
        base_accuracy = 0.772

        # Compound scaling impact (diminishing returns)
        scaling_bonus = (1 - math.exp(-phi * 0.5)) * 0.1

        # Add some randomness
        noise = random.gauss(0, 0.03)

        estimated_accuracy = base_accuracy + scaling_bonus + noise
        return max(0.5, min(0.95, estimated_accuracy))

class HardwareAwareOptimizer(BaseArchitectureOptimizer):
    """Hardware-aware neural architecture optimization."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        self.hardware_profiles = self._load_hardware_profiles()

    def _load_hardware_profiles(self) -> Dict:
        """Load hardware constraint profiles."""
        return {
            'mobile': {
                'max_params': 5_000_000,
                'max_flops': 500_000_000,
                'max_memory': 100 * 1024 * 1024,  # 100MB
                'preferred_ops': ['conv2d', 'depthwise_conv2d', 'linear'],
                'latency_weights': {'conv2d': 1.0, 'depthwise_conv2d': 0.3, 'linear': 0.8}
            },
            'edge': {
                'max_params': 20_000_000,
                'max_flops': 2_000_000_000,
                'max_memory': 500 * 1024 * 1024,  # 500MB
                'preferred_ops': ['conv2d', 'group_conv2d', 'linear', 'attention'],
                'latency_weights': {'conv2d': 1.0, 'group_conv2d': 0.7, 'linear': 0.8, 'attention': 2.0}
            },
            'server': {
                'max_params': 100_000_000,
                'max_flops': 10_000_000_000,
                'max_memory': 4 * 1024 * 1024 * 1024,  # 4GB
                'preferred_ops': ['conv2d', 'linear', 'attention', 'layer_norm'],
                'latency_weights': {'conv2d': 1.0, 'linear': 0.5, 'attention': 1.5, 'layer_norm': 0.3}
            }
        }

    def generate_architecture(self, constraints: Optional[Dict] = None) -> Dict:
        """Generate hardware-aware architecture."""
        profile = self.hardware_profiles[self.config.hardware_constraint]

        # Use MobileNet as base but adapt to hardware constraints
        mobilenet_optimizer = MobileNetOptimizer(self.config)
        arch = mobilenet_optimizer.generate_architecture()

        # Adjust for hardware constraints
        target_params = min(profile['max_params'], self.config.max_params)
        target_flops = min(profile['max_flops'], self.config.max_flops)

        # Scale width multiplier based on parameter budget
        current_params = self._estimate_params(arch)
        if current_params > target_params:
            scale_factor = math.sqrt(target_params / current_params)
            arch['width_multiplier'] *= scale_factor

        return arch

    def evaluate_architecture(self, architecture: Dict,
                            dataloader: torch.utils.data.DataLoader) -> float:
        """Evaluate with hardware-aware metrics."""
        profile = self.hardware_profiles[self.config.hardware_constraint]

        try:
            # Standard metrics
            params = self._estimate_params(architecture)
            flops = self._estimate_flops(architecture)
            memory = self._estimate_memory(architecture)

            # Hardware constraints
            if params > profile['max_params']:
                return float('-inf')
            if flops > profile['max_flops']:
                return float('-inf')
            if memory > profile['max_memory']:
                return float('-inf')

            # Estimate latency based on operations
            latency = self._estimate_latency(architecture, profile)

            # Estimate accuracy
            accuracy = self._estimate_accuracy_hardware_aware(architecture)

            # Hardware-aware fitness function
            efficiency = 1.0 / (latency + 1e-6)
            fitness = accuracy + 0.2 * efficiency

            return fitness

        except Exception as e:
            logger.error(f"Hardware-aware evaluation failed: {e}")
            return float('-inf')

    def _estimate_latency(self, architecture: Dict, profile: Dict) -> float:
        """Estimate inference latency."""
        total_latency = 0
        latency_weights = profile['latency_weights']

        for block in architecture['blocks']:
            if block['type'] == 'depthwise_separable':
                # Depthwise + pointwise
                dw_latency = latency_weights.get('depthwise_conv2d', 1.0)
                pw_latency = latency_weights.get('conv2d', 1.0) * 0.5  # 1x1 conv
                total_latency += (dw_latency + pw_latency) * block['out_channels']

            elif block['type'] == 'mbconv':
                # Expansion + depthwise + projection
                exp_latency = latency_weights.get('conv2d', 1.0) * 0.5
                dw_latency = latency_weights.get('depthwise_conv2d', 1.0)
                proj_latency = latency_weights.get('conv2d', 1.0) * 0.5
                total_latency += (exp_latency + dw_latency + proj_latency) * block['out_channels']

        return total_latency

    def _estimate_memory(self, architecture: Dict) -> int:
        """Estimate memory usage."""
        # Simplified memory estimation
        input_size = 224  # Default input size
        peak_memory = 0
        current_memory = input_size ** 2 * 3 * 4  # Input tensor (float32)

        for block in architecture['blocks']:
            # Feature map memory
            spatial_size = input_size ** 2
            channels = block['out_channels']
            feature_memory = spatial_size * channels * 4  # float32

            current_memory = max(current_memory, feature_memory)
            peak_memory = max(peak_memory, current_memory)

            # Update spatial size for stride
            if block.get('stride', 1) == 2:
                input_size //= 2

        return int(peak_memory)

    def _estimate_params(self, architecture: Dict) -> int:
        """Estimate parameter count."""
        mobilenet_opt = MobileNetOptimizer(self.config)
        return mobilenet_opt._count_parameters(mobilenet_opt._build_mobilenet(architecture))

    def _estimate_flops(self, architecture: Dict) -> float:
        """Estimate FLOPs."""
        mobilenet_opt = MobileNetOptimizer(self.config)
        model = mobilenet_opt._build_mobilenet(architecture)
        return mobilenet_opt._estimate_flops(model, (1, 3, 224, 224))

    def _estimate_accuracy_hardware_aware(self, architecture: Dict) -> float:
        """Hardware-aware accuracy estimation."""
        # Base accuracy
        mobilenet_opt = MobileNetOptimizer(self.config)
        base_accuracy = mobilenet_opt._estimate_accuracy(mobilenet_opt, architecture)

        # Hardware-specific adjustments
        profile = self.hardware_profiles[self.config.hardware_constraint]

        # Penalize if using non-preferred operations
        penalty = 0
        for block in architecture['blocks']:
            if block['type'] not in profile.get('preferred_ops', []):
                penalty += 0.01

        return max(0, base_accuracy - penalty)

    def mutate_architecture(self, architecture: Dict) -> Dict:
        """Hardware-aware mutation."""
        # Delegate to MobileNet mutator but add hardware-specific mutations
        mobilenet_opt = MobileNetOptimizer(self.config)
        arch = mobilenet_opt.mutate_architecture(architecture)

        # Hardware-specific mutations
        if random.random() < 0.2:
            profile = self.hardware_profiles[self.config.hardware_constraint]

            # Adjust width multiplier to fit hardware budget
            current_params = self._estimate_params(arch)
            target_params = profile['max_params'] * 0.8  # Use 80% of budget

            if current_params > target_params:
                scale_factor = math.sqrt(target_params / current_params)
                arch['width_multiplier'] *= scale_factor

        return arch

class ArchitectureOrchestrator:
    """Orchestrates different architecture optimization techniques."""

    def __init__(self):
        self.optimizers = {
            'mobilenet': MobileNetOptimizer,
            'efficientnet': EfficientNetOptimizer,
            'hardware_aware': HardwareAwareOptimizer
        }

    def create_optimizer(self, optimizer_type: str, config: ArchitectureConfig) -> BaseArchitectureOptimizer:
        """Create architecture optimizer."""
        if optimizer_type not in self.optimizers:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

        return self.optimizers[optimizer_type](config)

    def compare_optimizers(self, dataloader: torch.utils.data.DataLoader,
                          optimizer_types: List[str],
                          config: ArchitectureConfig,
                          generations: int = 5) -> Dict[str, Dict]:
        """Compare different architecture optimizers."""
        results = {}

        for opt_type in optimizer_types:
            logger.info(f"Testing {opt_type} optimizer...")

            try:
                optimizer = self.create_optimizer(opt_type, config)
                best_arch, best_fitness = optimizer.optimize(dataloader, generations)

                results[opt_type] = {
                    'best_architecture': best_arch,
                    'best_fitness': best_fitness,
                    'fitness_history': optimizer.fitness_history
                }

            except Exception as e:
                logger.error(f"Failed optimization with {opt_type}: {e}")
                results[opt_type] = {'error': str(e)}

        return results

# Utility functions
def visualize_architecture(architecture: Dict) -> str:
    """Create a text visualization of the architecture."""
    viz = f"Architecture Visualization\n"
    viz += f"========================\n"
    viz += f"Width Multiplier: {architecture.get('width_multiplier', 1.0):.2f}\n"
    viz += f"Depth Multiplier: {architecture.get('depth_multiplier', 1.0):.2f}\n"
    viz += f"Total Blocks: {len(architecture.get('blocks', []))}\n\n"

    viz += "Block Details:\n"
    for i, block in enumerate(architecture.get('blocks', [])):
        viz += f"Block {i+1}: {block['type']} "
        viz += f"({block['in_channels']} -> {block['out_channels']}) "
        viz += f"stride={block.get('stride', 1)} "
        viz += f"kernel={block.get('kernel_size', 3)}\n"

    return viz

# Factory function
def create_architecture_optimizer(optimizer_type: str = "mobilenet",
                                max_params: int = 10_000_000,
                                hardware_constraint: str = 'mobile') -> BaseArchitectureOptimizer:
    """Factory function to create architecture optimizers."""
    config = ArchitectureConfig(
        max_params=max_params,
        hardware_constraint=hardware_constraint
    )

    orchestrator = ArchitectureOrchestrator()
    return orchestrator.create_optimizer(optimizer_type, config)

if __name__ == "__main__":
    # Example usage
    from torch.utils.data import DataLoader, TensorDataset

    # Create sample data loader
    dummy_data = TensorDataset(
        torch.randn(100, 3, 224, 224),
        torch.randint(0, 1000, (100,))
    )
    dataloader = DataLoader(dummy_data, batch_size=32)

    # Test architecture optimization
    config = ArchitectureConfig(
        max_params=5_000_000,  # 5M parameters
        hardware_constraint='mobile',
        generations=3  # Quick test
    )

    orchestrator = ArchitectureOrchestrator()
    optimizers_to_test = ['mobilenet', 'hardware_aware']

    results = orchestrator.compare_optimizers(dataloader, optimizers_to_test, config, generations=3)

    for opt_type, result in results.items():
        if 'error' not in result:
            print(f"{opt_type}: Best fitness {result['best_fitness']:.4f}")
            if result['best_architecture']:
                print(visualize_architecture(result['best_architecture']))
        else:
            print(f"{opt_type}: Error - {result['error']}")