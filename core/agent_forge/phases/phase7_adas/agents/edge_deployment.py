"""
ADAS Edge Deployment Agent - Phase 7
Edge device optimization and deployment management
"""

import asyncio
import logging
import numpy as np
import threading
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import time
from concurrent.futures import ThreadPoolExecutor
import queue
import psutil
import platform

class EdgeDevice(Enum):
    NVIDIA_JETSON = "nvidia_jetson"
    QUALCOMM_SNAPDRAGON = "qualcomm_snapdragon"
    INTEL_MYRIAD = "intel_myriad"
    AUTOMOTIVE_ECU = "automotive_ecu"
    GENERIC_ARM = "generic_arm"

class OptimizationLevel(Enum):
    MINIMAL = "minimal"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"

class DeploymentStatus(Enum):
    PENDING = "pending"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    FAILED = "failed"
    UPDATING = "updating"

@dataclass
class EdgeResource:
    """Edge device resource information"""
    device_type: EdgeDevice
    cpu_cores: int
    cpu_frequency: float  # GHz
    memory_total: int  # MB
    memory_available: int  # MB
    gpu_available: bool
    gpu_memory: int  # MB
    storage_available: int  # MB
    power_limit: float  # Watts
    thermal_limit: float  # Celsius

@dataclass
class ModelDeployment:
    """Model deployment configuration"""
    model_id: str
    model_name: str
    model_type: str  # inference, training, hybrid
    framework: str  # tensorflow, pytorch, onnx, tensorrt
    precision: str  # fp32, fp16, int8
    batch_size: int
    input_shape: Tuple[int, ...]
    memory_requirement: int  # MB
    compute_requirement: float  # GFLOPS
    latency_requirement: float  # ms
    optimization_level: OptimizationLevel
    deployment_status: DeploymentStatus
    timestamp: float

@dataclass
class EdgePerformance:
    """Edge deployment performance metrics"""
    model_id: str
    inference_latency: float  # ms
    throughput: float  # FPS
    cpu_utilization: float  # %
    memory_utilization: float  # %
    gpu_utilization: float  # %
    power_consumption: float  # Watts
    temperature: float  # Celsius
    accuracy_score: float
    timestamp: float

class EdgeDeploymentAgent:
    """
    Advanced edge deployment agent for ADAS systems
    Optimizes and manages ML model deployment on edge devices
    """

    def __init__(self, agent_id: str = "edge_deployment_001"):
        self.agent_id = agent_id
        self.logger = self._setup_logging()

        # Real-time constraints
        self.max_processing_time = 0.050  # 50ms target
        self.monitoring_frequency = 10  # 10Hz

        # Edge device detection
        self.edge_device = self._detect_edge_device()
        self.edge_resources = self._get_edge_resources()

        # Model management
        self.deployed_models: Dict[str, ModelDeployment] = {}
        self.model_performance: Dict[str, EdgePerformance] = {}
        self.optimization_cache = {}

        # Optimization strategies
        self.optimization_strategies = {
            'quantization': True,
            'pruning': True,
            'knowledge_distillation': True,
            'tensorrt_optimization': True,
            'dynamic_batching': True,
            'model_parallelism': False,
            'pipeline_parallelism': False
        }

        # Processing pipeline
        self.input_queue = queue.Queue(maxsize=100)
        self.output_queue = queue.Queue(maxsize=50)
        self.processing_thread = None
        self.is_running = False

        # Performance monitoring
        self.performance_metrics = {
            'deployment_latency': [],
            'optimization_time': [],
            'inference_performance': [],
            'resource_utilization': [],
            'deployment_success_rate': 0.0
        }

        # Resource thresholds
        self.resource_thresholds = {
            'max_cpu_utilization': 0.8,  # 80%
            'max_memory_utilization': 0.85,  # 85%
            'max_gpu_utilization': 0.9,  # 90%
            'max_temperature': 85.0,  # Celsius
            'max_power_consumption': self.edge_resources.power_limit * 0.9
        }

        self.executor = ThreadPoolExecutor(max_workers=4)

    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging"""
        logger = logging.getLogger(f"ADAS.EdgeDeployment.{self.agent_id}")
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def _detect_edge_device(self) -> EdgeDevice:
        """Detect edge device type"""
        try:
            # Check for NVIDIA Jetson
            if platform.machine().startswith('aarch64'):
                try:
                    with open('/proc/device-tree/model', 'r') as f:
                        model = f.read().lower()
                        if 'jetson' in model:
                            return EdgeDevice.NVIDIA_JETSON
                except:
                    pass

            # Check for ARM architecture
            if platform.machine().startswith('arm') or platform.machine().startswith('aarch'):
                return EdgeDevice.GENERIC_ARM

            # Default to automotive ECU
            return EdgeDevice.AUTOMOTIVE_ECU

        except Exception as e:
            self.logger.error(f"Edge device detection failed: {e}")
            return EdgeDevice.AUTOMOTIVE_ECU

    def _get_edge_resources(self) -> EdgeResource:
        """Get edge device resource information"""
        try:
            # CPU information
            cpu_cores = psutil.cpu_count(logical=False)
            cpu_freq = psutil.cpu_freq()
            cpu_frequency = cpu_freq.current / 1000.0 if cpu_freq else 2.0  # GHz

            # Memory information
            memory = psutil.virtual_memory()
            memory_total = int(memory.total / (1024 * 1024))  # MB
            memory_available = int(memory.available / (1024 * 1024))  # MB

            # Storage information
            disk = psutil.disk_usage('/')
            storage_available = int(disk.free / (1024 * 1024))  # MB

            # GPU information (simplified)
            gpu_available = self._check_gpu_availability()
            gpu_memory = self._get_gpu_memory() if gpu_available else 0

            # Power and thermal limits (device-specific defaults)
            power_limit, thermal_limit = self._get_power_thermal_limits()

            return EdgeResource(
                device_type=self.edge_device,
                cpu_cores=cpu_cores,
                cpu_frequency=cpu_frequency,
                memory_total=memory_total,
                memory_available=memory_available,
                gpu_available=gpu_available,
                gpu_memory=gpu_memory,
                storage_available=storage_available,
                power_limit=power_limit,
                thermal_limit=thermal_limit
            )

        except Exception as e:
            self.logger.error(f"Resource detection failed: {e}")
            # Return default minimal resources
            return EdgeResource(
                device_type=self.edge_device,
                cpu_cores=4,
                cpu_frequency=2.0,
                memory_total=4096,
                memory_available=2048,
                gpu_available=False,
                gpu_memory=0,
                storage_available=10240,
                power_limit=50.0,
                thermal_limit=80.0
            )

    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available"""
        try:
            # Try to detect NVIDIA GPU
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False

    def _get_gpu_memory(self) -> int:
        """Get GPU memory in MB"""
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        except:
            pass
        return 0

    def _get_power_thermal_limits(self) -> Tuple[float, float]:
        """Get power and thermal limits based on device type"""
        limits = {
            EdgeDevice.NVIDIA_JETSON: (50.0, 80.0),
            EdgeDevice.QUALCOMM_SNAPDRAGON: (15.0, 70.0),
            EdgeDevice.INTEL_MYRIAD: (2.5, 60.0),
            EdgeDevice.AUTOMOTIVE_ECU: (30.0, 85.0),
            EdgeDevice.GENERIC_ARM: (25.0, 75.0)
        }
        return limits.get(self.edge_device, (30.0, 80.0))

    async def initialize(self) -> bool:
        """Initialize edge deployment system"""
        try:
            self.logger.info("Initializing ADAS Edge Deployment Agent")

            # Initialize optimization frameworks
            await self._initialize_optimization_frameworks()

            # Setup deployment pipelines
            await self._setup_deployment_pipelines()

            # Initialize monitoring
            await self._initialize_monitoring()

            # Start processing thread
            self.is_running = True
            self.processing_thread = threading.Thread(
                target=self._deployment_processing_loop,
                daemon=True
            )
            self.processing_thread.start()

            self.logger.info("Edge Deployment Agent initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False

    async def _initialize_optimization_frameworks(self):
        """Initialize optimization frameworks"""
        # TensorRT optimization (if available)
        self.tensorrt_available = self._check_tensorrt_availability()

        # ONNX Runtime optimization
        self.onnx_available = self._check_onnx_availability()

        # Quantization tools
        self.quantization_tools = {
            'tensorflow': True,  # TensorFlow Lite
            'pytorch': True,     # PyTorch quantization
            'onnx': self.onnx_available
        }

        # Model compression tools
        self.compression_tools = {
            'pruning': True,
            'knowledge_distillation': True,
            'neural_architecture_search': False  # Advanced optimization
        }

        self.logger.info("Optimization frameworks initialized")

    async def _setup_deployment_pipelines(self):
        """Setup deployment pipelines"""
        # Deployment stages
        self.deployment_pipeline = [
            'model_validation',
            'optimization_selection',
            'model_optimization',
            'performance_validation',
            'deployment_execution',
            'monitoring_setup'
        ]

        # Pipeline configurations
        self.pipeline_configs = {
            'parallel_optimization': True,
            'validation_threshold': 0.95,  # 95% accuracy retention
            'performance_target': 50.0,    # 50ms max latency
            'resource_limit': 0.8          # 80% resource usage
        }

        self.logger.info("Deployment pipelines setup complete")

    async def _initialize_monitoring(self):
        """Initialize performance monitoring"""
        # Monitoring components
        self.monitoring_components = {
            'system_resources': True,
            'model_performance': True,
            'thermal_monitoring': True,
            'power_monitoring': True,
            'network_monitoring': False  # Edge-specific
        }

        # Monitoring intervals
        self.monitoring_intervals = {
            'resource_check': 1.0,      # 1 second
            'performance_check': 0.1,   # 100ms
            'thermal_check': 5.0,       # 5 seconds
            'health_check': 10.0        # 10 seconds
        }

        self.logger.info("Monitoring initialized")

    def _check_tensorrt_availability(self) -> bool:
        """Check TensorRT availability"""
        try:
            import tensorrt
            return True
        except ImportError:
            return False

    def _check_onnx_availability(self) -> bool:
        """Check ONNX Runtime availability"""
        try:
            import onnxruntime
            return True
        except ImportError:
            return False

    def _deployment_processing_loop(self):
        """Main deployment processing loop"""
        self.logger.info("Starting deployment processing loop")

        while self.is_running:
            try:
                # Get deployment request
                try:
                    deployment_request = self.input_queue.get(timeout=0.1)
                    processing_start = time.perf_counter()

                    # Process deployment
                    deployment_result = self._process_deployment(deployment_request)

                    # Check processing time
                    processing_time = time.perf_counter() - processing_start
                    if processing_time > self.max_processing_time:
                        self.logger.warning(
                            f"Deployment processing exceeded time limit: {processing_time*1000:.2f}ms"
                        )

                    # Update performance metrics
                    self.performance_metrics['deployment_latency'].append(processing_time)
                    if len(self.performance_metrics['deployment_latency']) > 100:
                        self.performance_metrics['deployment_latency'].pop(0)

                    # Output deployment result
                    if deployment_result:
                        self.output_queue.put(deployment_result)

                    self.input_queue.task_done()

                except queue.Empty:
                    # Perform periodic monitoring
                    await self._perform_periodic_monitoring()
                    continue

            except Exception as e:
                self.logger.error(f"Deployment processing error: {e}")
                continue

    def _process_deployment(self, deployment_request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process model deployment request"""
        try:
            processing_start = time.perf_counter()

            # Extract deployment request
            model_info = deployment_request.get('model_info', {})
            optimization_config = deployment_request.get('optimization_config', {})
            deployment_config = deployment_request.get('deployment_config', {})

            # Validate model deployment
            validation_result = self._validate_model_deployment(model_info)
            if not validation_result['valid']:
                return {
                    'status': 'failed',
                    'reason': validation_result['reason'],
                    'timestamp': time.time()
                }

            # Select optimization strategy
            optimization_strategy = self._select_optimization_strategy(
                model_info, optimization_config
            )

            # Optimize model
            optimization_result = self._optimize_model(model_info, optimization_strategy)

            # Validate optimized model performance
            performance_validation = self._validate_optimized_performance(
                optimization_result
            )

            # Deploy model
            deployment_result = self._deploy_optimized_model(
                optimization_result, deployment_config
            )

            # Setup monitoring
            monitoring_setup = self._setup_model_monitoring(deployment_result)

            processing_time = time.perf_counter() - processing_start

            return {
                'status': 'success',
                'model_id': model_info.get('model_id'),
                'deployment_info': deployment_result,
                'optimization_info': optimization_result,
                'performance_info': performance_validation,
                'monitoring_info': monitoring_setup,
                'processing_time': processing_time,
                'timestamp': time.time()
            }

        except Exception as e:
            self.logger.error(f"Deployment processing failed: {e}")
            return {
                'status': 'failed',
                'reason': str(e),
                'timestamp': time.time()
            }

    def _validate_model_deployment(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model deployment requirements"""
        validation_result = {'valid': True, 'reason': ''}

        try:
            # Check model format
            supported_formats = ['onnx', 'tensorflow', 'pytorch', 'tensorrt']
            model_format = model_info.get('format', '').lower()
            if model_format not in supported_formats:
                return {
                    'valid': False,
                    'reason': f"Unsupported model format: {model_format}"
                }

            # Check memory requirements
            memory_requirement = model_info.get('memory_requirement', 0)
            if memory_requirement > self.edge_resources.memory_available:
                return {
                    'valid': False,
                    'reason': f"Insufficient memory: required {memory_requirement}MB, available {self.edge_resources.memory_available}MB"
                }

            # Check compute requirements
            compute_requirement = model_info.get('compute_requirement', 0)
            available_compute = self._estimate_available_compute()
            if compute_requirement > available_compute:
                return {
                    'valid': False,
                    'reason': f"Insufficient compute: required {compute_requirement} GFLOPS, available {available_compute} GFLOPS"
                }

            # Check latency requirements
            latency_requirement = model_info.get('latency_requirement', 1000.0)
            if latency_requirement < 10.0:  # Less than 10ms is challenging
                validation_result['reason'] = 'Very aggressive latency requirement'

            return validation_result

        except Exception as e:
            self.logger.error(f"Model validation failed: {e}")
            return {'valid': False, 'reason': 'Validation error'}

    def _estimate_available_compute(self) -> float:
        """Estimate available compute capacity in GFLOPS"""
        # Simplified compute estimation
        base_compute = self.edge_resources.cpu_cores * self.edge_resources.cpu_frequency * 10  # GFLOPS

        # Add GPU compute if available
        if self.edge_resources.gpu_available:
            # Rough GPU compute estimation based on device type
            gpu_compute = {
                EdgeDevice.NVIDIA_JETSON: 1000.0,
                EdgeDevice.QUALCOMM_SNAPDRAGON: 500.0,
                EdgeDevice.INTEL_MYRIAD: 100.0
            }.get(self.edge_device, 200.0)
            base_compute += gpu_compute

        return base_compute

    def _select_optimization_strategy(self, model_info: Dict[str, Any],
                                     optimization_config: Dict[str, Any]) -> Dict[str, Any]:
        """Select optimization strategy based on model and requirements"""
        strategy = {
            'quantization': False,
            'pruning': False,
            'knowledge_distillation': False,
            'tensorrt_optimization': False,
            'batch_optimization': False,
            'precision': 'fp32'
        }

        try:
            # Get optimization level
            opt_level = optimization_config.get('level', OptimizationLevel.BALANCED)

            # Memory-constrained optimization
            memory_ratio = model_info.get('memory_requirement', 0) / self.edge_resources.memory_available
            if memory_ratio > 0.6:
                strategy['quantization'] = True
                strategy['pruning'] = True
                strategy['precision'] = 'int8'

            # Latency-constrained optimization
            latency_requirement = model_info.get('latency_requirement', 1000.0)
            if latency_requirement < 50.0:  # Less than 50ms
                strategy['tensorrt_optimization'] = self.tensorrt_available
                strategy['batch_optimization'] = True
                strategy['precision'] = 'fp16'

            # Optimization level adjustments
            if opt_level == OptimizationLevel.AGGRESSIVE:
                strategy['quantization'] = True
                strategy['pruning'] = True
                strategy['precision'] = 'int8'
            elif opt_level == OptimizationLevel.MAXIMUM:
                strategy['quantization'] = True
                strategy['pruning'] = True
                strategy['knowledge_distillation'] = True
                strategy['precision'] = 'int8'

            # GPU-specific optimizations
            if self.edge_resources.gpu_available:
                strategy['tensorrt_optimization'] = self.tensorrt_available

            return strategy

        except Exception as e:
            self.logger.error(f"Optimization strategy selection failed: {e}")
            return strategy

    def _optimize_model(self, model_info: Dict[str, Any], strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize model based on strategy"""
        optimization_result = {
            'status': 'success',
            'optimizations_applied': [],
            'original_size': model_info.get('model_size', 0),
            'optimized_size': 0,
            'compression_ratio': 1.0,
            'estimated_speedup': 1.0
        }

        try:
            current_size = model_info.get('model_size', 100)  # MB
            speedup_factor = 1.0

            # Apply quantization
            if strategy.get('quantization', False):
                if strategy['precision'] == 'int8':
                    current_size *= 0.25  # 4x compression
                    speedup_factor *= 2.0
                elif strategy['precision'] == 'fp16':
                    current_size *= 0.5   # 2x compression
                    speedup_factor *= 1.5
                optimization_result['optimizations_applied'].append('quantization')

            # Apply pruning
            if strategy.get('pruning', False):
                current_size *= 0.7  # 30% reduction
                speedup_factor *= 1.3
                optimization_result['optimizations_applied'].append('pruning')

            # Apply TensorRT optimization
            if strategy.get('tensorrt_optimization', False):
                speedup_factor *= 2.5  # Significant speedup on GPU
                optimization_result['optimizations_applied'].append('tensorrt')

            # Apply knowledge distillation
            if strategy.get('knowledge_distillation', False):
                current_size *= 0.5  # 50% reduction
                speedup_factor *= 1.8
                optimization_result['optimizations_applied'].append('knowledge_distillation')

            optimization_result['optimized_size'] = current_size
            optimization_result['compression_ratio'] = optimization_result['original_size'] / current_size
            optimization_result['estimated_speedup'] = speedup_factor

            # Simulate optimization time
            optimization_time = len(optimization_result['optimizations_applied']) * 30  # seconds
            optimization_result['optimization_time'] = optimization_time

            return optimization_result

        except Exception as e:
            self.logger.error(f"Model optimization failed: {e}")
            optimization_result['status'] = 'failed'
            return optimization_result

    def _validate_optimized_performance(self, optimization_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate optimized model performance"""
        validation_result = {
            'valid': True,
            'estimated_latency': 0.0,
            'estimated_throughput': 0.0,
            'estimated_accuracy': 0.0,
            'resource_usage': {}
        }

        try:
            # Estimate performance based on optimization
            base_latency = 100.0  # ms
            speedup = optimization_result.get('estimated_speedup', 1.0)

            estimated_latency = base_latency / speedup
            estimated_throughput = 1000.0 / estimated_latency  # FPS

            # Estimate accuracy impact
            optimizations = optimization_result.get('optimizations_applied', [])
            accuracy_retention = 1.0

            for opt in optimizations:
                if opt == 'quantization':
                    accuracy_retention *= 0.98  # 2% accuracy loss
                elif opt == 'pruning':
                    accuracy_retention *= 0.97  # 3% accuracy loss
                elif opt == 'knowledge_distillation':
                    accuracy_retention *= 0.95  # 5% accuracy loss

            estimated_accuracy = accuracy_retention

            # Estimate resource usage
            model_size = optimization_result.get('optimized_size', 100)
            memory_usage = min(0.8, model_size / self.edge_resources.memory_total)
            cpu_usage = min(0.7, estimated_latency / 1000.0)

            validation_result.update({
                'estimated_latency': estimated_latency,
                'estimated_throughput': estimated_throughput,
                'estimated_accuracy': estimated_accuracy,
                'resource_usage': {
                    'memory': memory_usage,
                    'cpu': cpu_usage,
                    'gpu': 0.6 if self.edge_resources.gpu_available else 0.0
                }
            })

            # Check if performance meets requirements
            if estimated_latency > 100.0:  # 100ms threshold
                validation_result['valid'] = False

            return validation_result

        except Exception as e:
            self.logger.error(f"Performance validation failed: {e}")
            validation_result['valid'] = False
            return validation_result

    def _deploy_optimized_model(self, optimization_result: Dict[str, Any],
                               deployment_config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy optimized model"""
        deployment_result = {
            'status': 'success',
            'deployment_id': f"deploy_{int(time.time()*1000)}",
            'model_endpoint': '',
            'deployment_time': 0.0
        }

        try:
            deployment_start = time.perf_counter()

            # Simulate model deployment
            model_id = deployment_config.get('model_id', f"model_{int(time.time())}")

            # Create model deployment record
            model_deployment = ModelDeployment(
                model_id=model_id,
                model_name=deployment_config.get('model_name', 'adas_model'),
                model_type=deployment_config.get('model_type', 'inference'),
                framework=deployment_config.get('framework', 'onnx'),
                precision=deployment_config.get('precision', 'fp32'),
                batch_size=deployment_config.get('batch_size', 1),
                input_shape=tuple(deployment_config.get('input_shape', [1, 224, 224, 3])),
                memory_requirement=optimization_result.get('optimized_size', 100),
                compute_requirement=deployment_config.get('compute_requirement', 100),
                latency_requirement=deployment_config.get('latency_requirement', 50.0),
                optimization_level=OptimizationLevel.BALANCED,
                deployment_status=DeploymentStatus.DEPLOYED,
                timestamp=time.time()
            )

            # Store deployment
            self.deployed_models[model_id] = model_deployment

            deployment_time = time.perf_counter() - deployment_start
            deployment_result['deployment_time'] = deployment_time
            deployment_result['model_endpoint'] = f"/models/{model_id}/predict"

            return deployment_result

        except Exception as e:
            self.logger.error(f"Model deployment failed: {e}")
            deployment_result['status'] = 'failed'
            return deployment_result

    def _setup_model_monitoring(self, deployment_result: Dict[str, Any]) -> Dict[str, Any]:
        """Setup monitoring for deployed model"""
        monitoring_setup = {
            'status': 'success',
            'monitoring_enabled': True,
            'metrics_collected': []
        }

        try:
            # Setup performance monitoring
            metrics_to_collect = [
                'inference_latency',
                'throughput',
                'cpu_utilization',
                'memory_utilization',
                'accuracy_score'
            ]

            if self.edge_resources.gpu_available:
                metrics_to_collect.append('gpu_utilization')

            monitoring_setup['metrics_collected'] = metrics_to_collect

            return monitoring_setup

        except Exception as e:
            self.logger.error(f"Monitoring setup failed: {e}")
            monitoring_setup['status'] = 'failed'
            return monitoring_setup

    async def _perform_periodic_monitoring(self):
        """Perform periodic monitoring of deployed models"""
        try:
            current_time = time.time()

            # Monitor system resources
            system_metrics = self._get_system_metrics()

            # Monitor deployed models
            for model_id, deployment in self.deployed_models.items():
                if deployment.deployment_status == DeploymentStatus.DEPLOYED:
                    # Simulate performance metrics
                    performance = EdgePerformance(
                        model_id=model_id,
                        inference_latency=np.random.normal(30.0, 5.0),  # ms
                        throughput=np.random.normal(25.0, 3.0),  # FPS
                        cpu_utilization=np.random.uniform(0.4, 0.7),  # %
                        memory_utilization=np.random.uniform(0.3, 0.6),  # %
                        gpu_utilization=np.random.uniform(0.5, 0.8) if self.edge_resources.gpu_available else 0.0,
                        power_consumption=np.random.uniform(20.0, 35.0),  # Watts
                        temperature=np.random.uniform(50.0, 70.0),  # Celsius
                        accuracy_score=np.random.uniform(0.92, 0.98),
                        timestamp=current_time
                    )

                    self.model_performance[model_id] = performance

                    # Check for performance issues
                    self._check_performance_issues(performance)

        except Exception as e:
            self.logger.error(f"Periodic monitoring failed: {e}")

    def _get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics"""
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Try to get temperature
            try:
                temps = psutil.sensors_temperatures()
                temp = 45.0  # Default
                if temps:
                    for name, entries in temps.items():
                        if entries:
                            temp = entries[0].current
                            break
            except:
                temp = 45.0

            return {
                'cpu_utilization': cpu_percent / 100.0,
                'memory_utilization': memory_percent / 100.0,
                'temperature': temp,
                'power_consumption': 25.0  # Estimated
            }

        except Exception as e:
            self.logger.error(f"System metrics collection failed: {e}")
            return {}

    def _check_performance_issues(self, performance: EdgePerformance):
        """Check for performance issues"""
        try:
            issues = []

            # Check latency
            if performance.inference_latency > 100.0:  # 100ms threshold
                issues.append(f"High latency: {performance.inference_latency:.1f}ms")

            # Check resource utilization
            if performance.cpu_utilization > self.resource_thresholds['max_cpu_utilization']:
                issues.append(f"High CPU utilization: {performance.cpu_utilization:.1%}")

            if performance.memory_utilization > self.resource_thresholds['max_memory_utilization']:
                issues.append(f"High memory utilization: {performance.memory_utilization:.1%}")

            # Check temperature
            if performance.temperature > self.resource_thresholds['max_temperature']:
                issues.append(f"High temperature: {performance.temperature:.1f}°C")

            # Check accuracy
            if performance.accuracy_score < 0.9:  # 90% threshold
                issues.append(f"Low accuracy: {performance.accuracy_score:.2%}")

            if issues:
                self.logger.warning(f"Performance issues for {performance.model_id}: {', '.join(issues)}")

        except Exception as e:
            self.logger.error(f"Performance issue check failed: {e}")

    async def deploy_model(self, deployment_request: Dict[str, Any]) -> bool:
        """Deploy model to edge device"""
        try:
            if not deployment_request:
                return False

            # Add to processing queue
            try:
                self.input_queue.put_nowait(deployment_request)
                return True
            except queue.Full:
                self.logger.warning("Deployment queue full, dropping request")
                return False

        except Exception as e:
            self.logger.error(f"Error deploying model: {e}")
            return False

    async def get_deployment_result(self, timeout: float = 0.1) -> Optional[Dict[str, Any]]:
        """Get deployment result"""
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    async def get_model_performance(self, model_id: str) -> Optional[EdgePerformance]:
        """Get performance metrics for specific model"""
        return self.model_performance.get(model_id)

    async def get_deployed_models(self) -> Dict[str, ModelDeployment]:
        """Get all deployed models"""
        return self.deployed_models.copy()

    async def get_edge_resources(self) -> EdgeResource:
        """Get edge device resources"""
        # Update available memory
        memory = psutil.virtual_memory()
        self.edge_resources.memory_available = int(memory.available / (1024 * 1024))
        return self.edge_resources

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        metrics = self.performance_metrics.copy()

        if metrics['deployment_latency']:
            metrics['avg_deployment_latency'] = np.mean(metrics['deployment_latency'])
            metrics['max_deployment_latency'] = np.max(metrics['deployment_latency'])

        metrics['deployed_models_count'] = len(self.deployed_models)
        metrics['device_type'] = self.edge_device.value
        metrics['gpu_available'] = self.edge_resources.gpu_available

        return metrics

    async def shutdown(self):
        """Shutdown edge deployment agent"""
        self.logger.info("Shutting down Edge Deployment Agent")

        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)

        self.executor.shutdown(wait=True)

        self.logger.info("Edge Deployment Agent shutdown complete")

# Example usage
if __name__ == "__main__":
    async def test_edge_deployment():
        agent = EdgeDeploymentAgent()

        if await agent.initialize():
            print("Edge Deployment Agent initialized successfully")

            # Simulate deployment request
            deployment_request = {
                'model_info': {
                    'model_id': 'adas_perception_v1',
                    'model_name': 'ADAS Perception Model',
                    'format': 'onnx',
                    'model_size': 150,  # MB
                    'memory_requirement': 200,  # MB
                    'compute_requirement': 500,  # GFLOPS
                    'latency_requirement': 25.0  # ms
                },
                'optimization_config': {
                    'level': OptimizationLevel.BALANCED
                },
                'deployment_config': {
                    'model_type': 'inference',
                    'framework': 'onnx',
                    'batch_size': 1,
                    'input_shape': [1, 3, 224, 224]
                }
            }

            await agent.deploy_model(deployment_request)
            await asyncio.sleep(1.0)  # Wait for processing

            result = await agent.get_deployment_result()
            if result:
                print(f"Deployment result: {result['status']}")
                if result['status'] == 'success':
                    print(f"Model ID: {result['model_id']}")
                    print(f"Processing time: {result['processing_time']:.3f}s")

            # Check resources
            resources = await agent.get_edge_resources()
            print(f"Edge device: {resources.device_type.value}")
            print(f"Available memory: {resources.memory_available}MB")

            await agent.shutdown()

    asyncio.run(test_edge_deployment())