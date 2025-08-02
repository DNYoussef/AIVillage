"""Adaptive Model Loader for Resource-Constrained Devices"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import json
import hashlib

from .device_profiler import DeviceProfiler, DeviceType, ResourceSnapshot
from .constraint_manager import ConstraintManager, ResourceConstraints

logger = logging.getLogger(__name__)


class LoadingStrategy(Enum):
    """Model loading strategies"""
    FULL = "full"               # Load full model
    COMPRESSED = "compressed"   # Load compressed model
    QUANTIZED = "quantized"     # Load quantized model
    LAYERED = "layered"         # Load model in layers
    STREAMING = "streaming"     # Stream model from storage/network
    CACHED = "cached"           # Use cached/preloaded model


class ModelSize(Enum):
    """Model size categories"""
    TINY = "tiny"       # < 100MB
    SMALL = "small"     # 100MB - 500MB
    MEDIUM = "medium"   # 500MB - 2GB
    LARGE = "large"     # 2GB - 8GB
    XLARGE = "xlarge"   # > 8GB

@dataclass
class ModelVariant:
    """Represents a model variant with different resource requirements"""
    name: str
    strategy: LoadingStrategy
    size_mb: int
    memory_requirement_mb: int
    cpu_requirement_percent: float
    quality_score: float  # 0-1, higher = better quality
    loading_time_estimate_seconds: float
    supports_streaming: bool = False
    supports_quantization: bool = False
    supports_compression: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'strategy': self.strategy.value,
            'size_mb': self.size_mb,
            'memory_requirement_mb': self.memory_requirement_mb,
            'cpu_requirement_percent': self.cpu_requirement_percent,
            'quality_score': self.quality_score,
            'loading_time_estimate_seconds': self.loading_time_estimate_seconds,
            'supports_streaming': self.supports_streaming,
            'supports_quantization': self.supports_quantization,
            'supports_compression': self.supports_compression
        }

@dataclass
class LoadingContext:
    """Context for model loading decisions"""
    task_type: str  # nightly, breakthrough, emergency, etc.
    priority_level: int  # 1-5
    max_loading_time_seconds: float
    quality_preference: float  # 0-1, higher = prefer quality over speed
    resource_constraints: ResourceConstraints
    allow_degraded_quality: bool = True
    cache_enabled: bool = True
    

class AdaptiveLoader:
    """Adaptive model loader that selects optimal loading strategy based on device resources"""
    
    def __init__(self, device_profiler: DeviceProfiler, constraint_manager: ConstraintManager):
        self.device_profiler = device_profiler
        self.constraint_manager = constraint_manager
        
        # Model variants registry
        self.model_variants: Dict[str, List[ModelVariant]] = {}
        
        # Loading cache
        self.loaded_models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict[str, Any]] = {}
        self.cache_stats: Dict[str, Dict[str, Any]] = {}
        
        # Loading strategies by device type
        self.preferred_strategies = self._initialize_preferred_strategies()
        
        # Adaptive parameters
        self.quality_degradation_threshold = 0.7  # Below this, prefer speed
        self.memory_safety_margin = 0.8  # Use 80% of available memory max
        self.cpu_safety_margin = 0.75     # Use 75% of available CPU max
        
        # Statistics
        self.stats = {
            'models_loaded': 0,
            'loading_failures': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'strategy_adaptations': 0,
            'quality_degradations': 0,
            'loading_time_total': 0.0,
            'by_strategy': {strategy.value: 0 for strategy in LoadingStrategy}
        }
        
        # Initialize built-in model variants
        self._initialize_builtin_variants()
        
    def _initialize_preferred_strategies(self) -> Dict[DeviceType, List[LoadingStrategy]]:
        """Initialize preferred loading strategies by device type"""
        return {
            DeviceType.PHONE: [
                LoadingStrategy.QUANTIZED,
                LoadingStrategy.COMPRESSED,
                LoadingStrategy.CACHED,
                LoadingStrategy.STREAMING
            ],
            DeviceType.TABLET: [
                LoadingStrategy.COMPRESSED,
                LoadingStrategy.QUANTIZED,
                LoadingStrategy.CACHED,
                LoadingStrategy.FULL
            ],
            DeviceType.LAPTOP: [
                LoadingStrategy.FULL,
                LoadingStrategy.COMPRESSED,
                LoadingStrategy.LAYERED,
                LoadingStrategy.CACHED
            ],
            DeviceType.DESKTOP: [
                LoadingStrategy.FULL,
                LoadingStrategy.LAYERED,
                LoadingStrategy.COMPRESSED,
                LoadingStrategy.CACHED
            ],
            DeviceType.SERVER: [
                LoadingStrategy.FULL,
                LoadingStrategy.LAYERED,
                LoadingStrategy.STREAMING,
                LoadingStrategy.CACHED
            ],
            DeviceType.EMBEDDED: [
                LoadingStrategy.QUANTIZED,
                LoadingStrategy.STREAMING,
                LoadingStrategy.CACHED,
                LoadingStrategy.COMPRESSED
            ]
        }
        
    def _initialize_builtin_variants(self):
        """Initialize built-in model variants for common evolution models"""
        # Example model variants for evolution systems
        self.register_model_variants("base_evolution_model", [
            ModelVariant(
                name="base_evolution_model_tiny",
                strategy=LoadingStrategy.QUANTIZED,
                size_mb=50,
                memory_requirement_mb=100,
                cpu_requirement_percent=20.0,
                quality_score=0.6,
                loading_time_estimate_seconds=2.0,
                supports_quantization=True,
                supports_compression=True
            ),
            ModelVariant(
                name="base_evolution_model_small",
                strategy=LoadingStrategy.COMPRESSED,
                size_mb=200,
                memory_requirement_mb=400,
                cpu_requirement_percent=35.0,
                quality_score=0.75,
                loading_time_estimate_seconds=5.0,
                supports_compression=True,
                supports_quantization=True
            ),
            ModelVariant(
                name="base_evolution_model_medium",
                strategy=LoadingStrategy.FULL,
                size_mb=800,
                memory_requirement_mb=1600,
                cpu_requirement_percent=50.0,
                quality_score=0.85,
                loading_time_estimate_seconds=15.0,
                supports_streaming=True
            ),
            ModelVariant(
                name="base_evolution_model_large",
                strategy=LoadingStrategy.LAYERED,
                size_mb=3200,
                memory_requirement_mb=4800,
                cpu_requirement_percent=70.0,
                quality_score=0.95,
                loading_time_estimate_seconds=45.0,
                supports_streaming=True,
                supports_compression=True
            )
        ])
        
    def register_model_variants(self, model_name: str, variants: List[ModelVariant]):
        """Register model variants for a model"""
        self.model_variants[model_name] = variants
        logger.info(f"Registered {len(variants)} variants for model {model_name}")
        
    async def load_model_adaptive(self, model_name: str, context: LoadingContext) -> Tuple[Optional[Any], Dict[str, Any]]:
        """Adaptively load model based on current resource availability"""
        start_time = time.time()
        
        try:
            # Check if model is cached
            if context.cache_enabled and model_name in self.loaded_models:
                self.stats['cache_hits'] += 1
                loading_info = {
                    'strategy': 'cached',
                    'loading_time': 0.0,
                    'from_cache': True,
                    'variant_used': self.model_metadata.get(model_name, {}).get('variant_name', 'unknown')
                }
                return self.loaded_models[model_name], loading_info
                
            self.stats['cache_misses'] += 1
            
            # Get optimal variant
            variant = await self._select_optimal_variant(model_name, context)
            if not variant:
                logger.error(f"No suitable variant found for model {model_name}")
                self.stats['loading_failures'] += 1
                return None, {'error': 'No suitable variant found'}
                
            # Load model using selected variant
            model, loading_info = await self._load_model_variant(model_name, variant, context)
            
            if model is not None:
                # Cache model if successful
                if context.cache_enabled:
                    self.loaded_models[model_name] = model
                    self.model_metadata[model_name] = {
                        'variant_name': variant.name,
                        'strategy': variant.strategy.value,
                        'loaded_at': time.time(),
                        'quality_score': variant.quality_score,
                        'size_mb': variant.size_mb
                    }
                    
                self.stats['models_loaded'] += 1
                self.stats['by_strategy'][variant.strategy.value] += 1
                
                # Update loading time stats
                loading_time = time.time() - start_time
                self.stats['loading_time_total'] += loading_time
                loading_info['loading_time'] = loading_time
                
                logger.info(f"Successfully loaded {model_name} using {variant.strategy.value} strategy")
                
            else:
                self.stats['loading_failures'] += 1
                
            return model, loading_info
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            self.stats['loading_failures'] += 1
            return None, {'error': str(e)}
            
    async def _select_optimal_variant(self, model_name: str, context: LoadingContext) -> Optional[ModelVariant]:
        """Select optimal model variant based on current resources and context"""
        if model_name not in self.model_variants:
            logger.warning(f"No variants registered for model {model_name}")
            return None
            
        variants = self.model_variants[model_name]
        current_snapshot = self.device_profiler.current_snapshot
        
        if not current_snapshot:
            logger.warning("No current resource snapshot available")
            return None
            
        # Calculate available resources
        available_memory_mb = current_snapshot.memory_available / (1024 * 1024)
        available_cpu_percent = max(0, 100 - current_snapshot.cpu_percent)
        
        # Apply safety margins
        usable_memory_mb = available_memory_mb * self.memory_safety_margin
        usable_cpu_percent = available_cpu_percent * self.cpu_safety_margin
        
        # Filter variants that meet resource constraints
        suitable_variants = []
        for variant in variants:
            if (variant.memory_requirement_mb <= usable_memory_mb and
                variant.cpu_requirement_percent <= usable_cpu_percent and
                variant.loading_time_estimate_seconds <= context.max_loading_time_seconds):
                suitable_variants.append(variant)
                
        if not suitable_variants:
            # Try with relaxed constraints if allowed
            if context.allow_degraded_quality:
                self.stats['quality_degradations'] += 1
                
                # Relax memory constraints slightly
                relaxed_memory = available_memory_mb * 0.9
                relaxed_cpu = available_cpu_percent * 0.85
                
                for variant in variants:
                    if (variant.memory_requirement_mb <= relaxed_memory and
                        variant.cpu_requirement_percent <= relaxed_cpu):
                        suitable_variants.append(variant)
                        
        if not suitable_variants:
            logger.warning(f"No suitable variants found for {model_name} with current resources")
            return None
            
        # Score variants based on context preferences
        scored_variants = []
        for variant in suitable_variants:
            score = self._calculate_variant_score(variant, context, current_snapshot)
            scored_variants.append((score, variant))
            
        # Sort by score (higher is better)
        scored_variants.sort(reverse=True)
        
        selected_variant = scored_variants[0][1]
        self.stats['strategy_adaptations'] += 1
        
        logger.debug(f"Selected variant {selected_variant.name} with score {scored_variants[0][0]:.2f}")
        
        return selected_variant
        
    def _calculate_variant_score(self, variant: ModelVariant, context: LoadingContext,
                               snapshot: ResourceSnapshot) -> float:
        """Calculate score for a model variant"""
        score = 0.0
        
        # Quality preference (0-1)
        quality_score = variant.quality_score * context.quality_preference
        score += quality_score * 40  # Quality worth up to 40 points
        
        # Loading time preference (lower is better)
        time_ratio = min(variant.loading_time_estimate_seconds / context.max_loading_time_seconds, 1.0)
        time_score = (1.0 - time_ratio) * (1.0 - context.quality_preference)
        score += time_score * 30  # Speed worth up to 30 points
        
        # Resource efficiency
        memory_ratio = variant.memory_requirement_mb / (snapshot.memory_available / (1024 * 1024))
        cpu_ratio = variant.cpu_requirement_percent / max(100 - snapshot.cpu_percent, 1)
        
        resource_efficiency = 1.0 - (memory_ratio + cpu_ratio) / 2
        score += resource_efficiency * 20  # Efficiency worth up to 20 points
        
        # Strategy preference based on device type
        device_type = self.device_profiler.profile.device_type
        preferred_strategies = self.preferred_strategies.get(device_type, [])
        
        if variant.strategy in preferred_strategies:
            strategy_index = preferred_strategies.index(variant.strategy)
            strategy_bonus = (len(preferred_strategies) - strategy_index) / len(preferred_strategies)
            score += strategy_bonus * 10  # Strategy preference worth up to 10 points
            
        return score
        
    async def _load_model_variant(self, model_name: str, variant: ModelVariant,
                                context: LoadingContext) -> Tuple[Optional[Any], Dict[str, Any]]:
        """Load model using specific variant"""
        loading_info = {
            'variant_name': variant.name,
            'strategy': variant.strategy.value,
            'size_mb': variant.size_mb,
            'quality_score': variant.quality_score,
            'from_cache': False
        }
        
        try:
            if variant.strategy == LoadingStrategy.FULL:
                model = await self._load_full_model(model_name, variant)
            elif variant.strategy == LoadingStrategy.COMPRESSED:
                model = await self._load_compressed_model(model_name, variant)
            elif variant.strategy == LoadingStrategy.QUANTIZED:
                model = await self._load_quantized_model(model_name, variant)
            elif variant.strategy == LoadingStrategy.LAYERED:
                model = await self._load_layered_model(model_name, variant)
            elif variant.strategy == LoadingStrategy.STREAMING:
                model = await self._load_streaming_model(model_name, variant)
            elif variant.strategy == LoadingStrategy.CACHED:
                model = await self._load_cached_model(model_name, variant)
            else:
                raise ValueError(f"Unknown loading strategy: {variant.strategy}")
                
            return model, loading_info
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name} with strategy {variant.strategy}: {e}")
            loading_info['error'] = str(e)
            return None, loading_info
            
    async def _load_full_model(self, model_name: str, variant: ModelVariant) -> Any:
        """Load full model"""
        # Simulate model loading
        await asyncio.sleep(variant.loading_time_estimate_seconds * 0.1)  # Simulated loading
        
        # In real implementation, this would load the actual model
        # For now, return a mock model object
        return {
            'type': 'full_model',
            'name': model_name,
            'variant': variant.name,
            'size_mb': variant.size_mb,
            'quality_score': variant.quality_score
        }
        
    async def _load_compressed_model(self, model_name: str, variant: ModelVariant) -> Any:
        """Load compressed model"""
        await asyncio.sleep(variant.loading_time_estimate_seconds * 0.1)
        
        return {
            'type': 'compressed_model',
            'name': model_name,
            'variant': variant.name,
            'size_mb': variant.size_mb,
            'quality_score': variant.quality_score,
            'compression_ratio': 0.6  # Example compression ratio
        }
        
    async def _load_quantized_model(self, model_name: str, variant: ModelVariant) -> Any:
        """Load quantized model"""
        await asyncio.sleep(variant.loading_time_estimate_seconds * 0.1)
        
        return {
            'type': 'quantized_model',
            'name': model_name,
            'variant': variant.name,
            'size_mb': variant.size_mb,
            'quality_score': variant.quality_score,
            'quantization_bits': 8  # Example quantization
        }
        
    async def _load_layered_model(self, model_name: str, variant: ModelVariant) -> Any:
        """Load model in layers"""
        await asyncio.sleep(variant.loading_time_estimate_seconds * 0.1)
        
        return {
            'type': 'layered_model',
            'name': model_name,
            'variant': variant.name,
            'size_mb': variant.size_mb,
            'quality_score': variant.quality_score,
            'layers_loaded': 12  # Example layer count
        }
        
    async def _load_streaming_model(self, model_name: str, variant: ModelVariant) -> Any:
        """Load streaming model"""
        await asyncio.sleep(variant.loading_time_estimate_seconds * 0.1)
        
        return {
            'type': 'streaming_model',
            'name': model_name,
            'variant': variant.name,
            'size_mb': variant.size_mb,
            'quality_score': variant.quality_score,
            'streaming_enabled': True
        }
        
    async def _load_cached_model(self, model_name: str, variant: ModelVariant) -> Any:
        """Load cached model"""
        # Check if we have a cached version
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
            
        # If not cached, fall back to another strategy
        fallback_variant = ModelVariant(
            name=f"{variant.name}_fallback",
            strategy=LoadingStrategy.COMPRESSED,
            size_mb=variant.size_mb,
            memory_requirement_mb=variant.memory_requirement_mb,
            cpu_requirement_percent=variant.cpu_requirement_percent,
            quality_score=variant.quality_score * 0.9,  # Slightly lower quality
            loading_time_estimate_seconds=variant.loading_time_estimate_seconds * 1.2
        )
        
        return await self._load_compressed_model(model_name, fallback_variant)
        
    def unload_model(self, model_name: str) -> bool:
        """Unload model from cache"""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            if model_name in self.model_metadata:
                del self.model_metadata[model_name]
            logger.info(f"Unloaded model {model_name}")
            return True
        return False
        
    def clear_cache(self):
        """Clear all cached models"""
        count = len(self.loaded_models)
        self.loaded_models.clear()
        self.model_metadata.clear()
        logger.info(f"Cleared {count} models from cache")
        
    def get_cache_usage(self) -> Dict[str, Any]:
        """Get cache usage statistics"""
        total_size_mb = 0
        model_count = len(self.loaded_models)
        
        for model_name in self.model_metadata:
            metadata = self.model_metadata[model_name]
            total_size_mb += metadata.get('size_mb', 0)
            
        return {
            'models_cached': model_count,
            'total_size_mb': total_size_mb,
            'cache_hit_rate': (
                self.stats['cache_hits'] /
                max(self.stats['cache_hits'] + self.stats['cache_misses'], 1)
            ),
            'models': list(self.model_metadata.keys())
        }
        
    def get_optimal_strategy_for_context(self, model_name: str, context: LoadingContext) -> Optional[LoadingStrategy]:
        """Get optimal loading strategy for given context without actually loading"""
        if model_name not in self.model_variants:
            return None
            
        # Use the same selection logic but only return strategy
        variant = asyncio.run(self._select_optimal_variant(model_name, context))
        return variant.strategy if variant else None
        
    def get_model_variants_info(self, model_name: str) -> List[Dict[str, Any]]:
        """Get information about all variants of a model"""
        if model_name not in self.model_variants:
            return []
            
        return [variant.to_dict() for variant in self.model_variants[model_name]]
        
    def estimate_loading_resources(self, model_name: str, strategy: LoadingStrategy) -> Optional[Dict[str, Any]]:
        """Estimate resource requirements for loading a model with specific strategy"""
        if model_name not in self.model_variants:
            return None
            
        variants = self.model_variants[model_name]
        matching_variants = [v for v in variants if v.strategy == strategy]
        
        if not matching_variants:
            return None
            
        variant = matching_variants[0]  # Take first matching variant
        
        return {
            'memory_requirement_mb': variant.memory_requirement_mb,
            'cpu_requirement_percent': variant.cpu_requirement_percent,
            'loading_time_estimate_seconds': variant.loading_time_estimate_seconds,
            'disk_space_mb': variant.size_mb,
            'quality_score': variant.quality_score
        }
        
    def get_loading_stats(self) -> Dict[str, Any]:
        """Get loading statistics"""
        total_loads = max(self.stats['models_loaded'] + self.stats['loading_failures'], 1)
        avg_loading_time = self.stats['loading_time_total'] / max(self.stats['models_loaded'], 1)
        
        return {
            **self.stats,
            'success_rate': self.stats['models_loaded'] / total_loads,
            'average_loading_time': avg_loading_time,
            'cache_usage': self.get_cache_usage(),
            'registered_models': len(self.model_variants),
            'total_variants': sum(len(variants) for variants in self.model_variants.values())
        }

