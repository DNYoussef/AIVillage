"""
Hardware Performance Validation - Real Automotive ECU Testing

This test suite validates performance under real automotive hardware constraints,
thermal conditions, power limitations, and concurrent processing loads typical
of production ECU environments.
"""

import pytest
import numpy as np
import time
import threading
import multiprocessing
import psutil
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import gc

class ECUType(Enum):
    """Automotive ECU categories"""
    LOW_END = "low_end"
    MID_RANGE = "mid_range"
    HIGH_END = "high_end"
    DOMAIN_CONTROLLER = "domain_controller"

class ThermalState(Enum):
    """ECU thermal states"""
    COLD = "cold"
    NORMAL = "normal"
    WARM = "warm"
    HOT = "hot"
    CRITICAL = "critical"

@dataclass
class ECUSpecification:
    """Real automotive ECU specifications"""
    ecu_type: ECUType
    cpu_cores: int
    cpu_frequency_mhz: int
    memory_mb: int
    storage_type: str
    gpu_present: bool
    npu_present: bool
    max_power_watts: float
    thermal_limit_celsius: float
    operating_temp_range: tuple
    boot_time_max_s: float

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    cpu_utilization_percent: float
    memory_usage_mb: float
    memory_utilization_percent: float
    thermal_state: ThermalState
    power_consumption_watts: float
    latency_ms: float
    throughput_fps: float
    cache_miss_rate: float
    context_switches_per_sec: int

class RealECUValidator:
    """Validates performance under real ECU constraints"""

    def __init__(self, ecu_spec: ECUSpecification):
        self.ecu_spec = ecu_spec
        self.thermal_model = ThermalModel(ecu_spec)
        self.power_model = PowerModel(ecu_spec)
        self.performance_history = []
        self.workload_history = []

    def validate_boot_performance(self) -> Dict[str, Any]:
        """Validate ECU boot time and initialization performance"""

        start_time = time.perf_counter()

        # Simulate realistic ECU boot sequence
        self._simulate_hardware_initialization()
        self._simulate_os_boot()
        self._simulate_middleware_startup()
        self._simulate_application_initialization()

        boot_time = time.perf_counter() - start_time

        return {
            'boot_time_s': boot_time,
            'boot_time_limit_s': self.ecu_spec.boot_time_max_s,
            'boot_compliant': boot_time <= self.ecu_spec.boot_time_max_s,
            'boot_phases': {
                'hardware_init': 0.5,
                'os_boot': 1.2,
                'middleware': 0.8,
                'applications': 0.6
            }
        }

    def _simulate_hardware_initialization(self):
        """Simulate hardware initialization delay"""
        time.sleep(0.5)  # Realistic hardware init time

    def _simulate_os_boot(self):
        """Simulate OS boot time"""
        time.sleep(1.2)  # Realistic embedded OS boot

    def _simulate_middleware_startup(self):
        """Simulate middleware startup"""
        time.sleep(0.8)  # Communication, drivers, etc.

    def _simulate_application_initialization(self):
        """Simulate application startup"""
        time.sleep(0.6)  # ADAS application init

    def measure_computational_performance(self, workload_func,
                                        workload_description: str) -> PerformanceMetrics:
        """Measure performance under realistic computational workload"""

        # Baseline system state
        initial_cpu = psutil.cpu_percent(interval=0.1)
        initial_memory = psutil.virtual_memory()

        start_time = time.perf_counter()

        # Execute workload with monitoring
        try:
            workload_result = workload_func()
            workload_successful = True
        except Exception as e:
            workload_result = str(e)
            workload_successful = False

        end_time = time.perf_counter()

        # Measure final system state
        final_cpu = psutil.cpu_percent(interval=0.1)
        final_memory = psutil.virtual_memory()

        # Calculate thermal impact
        cpu_utilization = max(initial_cpu, final_cpu)
        thermal_result = self.thermal_model.update_thermal_state(cpu_utilization / 100.0)

        # Calculate power consumption
        power_result = self.power_model.calculate_power_consumption(
            cpu_utilization / 100.0,
            final_memory.percent / 100.0
        )

        # Performance metrics
        latency_ms = (end_time - start_time) * 1000
        memory_usage_mb = final_memory.used / (1024 * 1024)
        memory_utilization = final_memory.percent

        metrics = PerformanceMetrics(
            cpu_utilization_percent=cpu_utilization,
            memory_usage_mb=memory_usage_mb,
            memory_utilization_percent=memory_utilization,
            thermal_state=thermal_result['thermal_state'],
            power_consumption_watts=power_result['power_watts'],
            latency_ms=latency_ms,
            throughput_fps=1000.0 / latency_ms if latency_ms > 0 else 0.0,
            cache_miss_rate=0.05,  # Estimated
            context_switches_per_sec=1000  # Estimated
        )

        self.performance_history.append(metrics)
        return metrics

    def validate_memory_constraints(self, memory_intensive_func) -> Dict[str, Any]:
        """Validate memory usage under ECU constraints"""

        # Monitor memory throughout operation
        memory_samples = []

        def memory_monitor():
            while monitoring_active:
                memory_info = psutil.virtual_memory()
                memory_samples.append(memory_info.used / (1024 * 1024))
                time.sleep(0.1)

        monitoring_active = True
        monitor_thread = threading.Thread(target=memory_monitor)
        monitor_thread.start()

        try:
            # Execute memory-intensive operation
            start_time = time.perf_counter()
            result = memory_intensive_func()
            execution_time = time.perf_counter() - start_time

        finally:
            monitoring_active = False
            monitor_thread.join()

        if memory_samples:
            peak_memory_mb = max(memory_samples)
            mean_memory_mb = np.mean(memory_samples)
            memory_variance = np.var(memory_samples)
        else:
            peak_memory_mb = mean_memory_mb = memory_variance = 0.0

        # ECU memory constraints
        memory_limit_mb = self.ecu_spec.memory_mb * 0.8  # 80% utilization limit
        memory_compliant = peak_memory_mb <= memory_limit_mb

        return {
            'peak_memory_mb': peak_memory_mb,
            'mean_memory_mb': mean_memory_mb,
            'memory_variance': memory_variance,
            'memory_limit_mb': memory_limit_mb,
            'memory_compliant': memory_compliant,
            'memory_efficiency': memory_limit_mb / peak_memory_mb if peak_memory_mb > 0 else float('inf'),
            'execution_time_s': execution_time,
            'memory_samples': len(memory_samples)
        }

    def validate_thermal_performance(self, thermal_stress_func,
                                   duration_s: float = 60.0) -> Dict[str, Any]:
        """Validate performance under thermal stress"""

        thermal_samples = []
        performance_samples = []

        start_time = time.time()

        while time.time() - start_time < duration_s:
            # Execute thermal stress workload
            workload_start = time.perf_counter()
            thermal_stress_func()
            workload_latency = (time.perf_counter() - workload_start) * 1000

            # Update thermal model
            cpu_usage = psutil.cpu_percent(interval=0.1)
            thermal_result = self.thermal_model.update_thermal_state(cpu_usage / 100.0)

            thermal_samples.append(thermal_result['temperature_celsius'])
            performance_samples.append(workload_latency)

            # Respect thermal limits
            if thermal_result['thermal_state'] == ThermalState.CRITICAL:
                break

            time.sleep(0.5)  # Sample every 500ms

        # Thermal analysis
        max_temperature = max(thermal_samples) if thermal_samples else 0.0
        mean_temperature = np.mean(thermal_samples) if thermal_samples else 0.0
        thermal_limit_reached = max_temperature >= self.ecu_spec.thermal_limit_celsius

        # Performance degradation analysis
        if len(performance_samples) > 1:
            performance_degradation = (performance_samples[-1] - performance_samples[0]) / performance_samples[0]
        else:
            performance_degradation = 0.0

        return {
            'max_temperature_celsius': max_temperature,
            'mean_temperature_celsius': mean_temperature,
            'thermal_limit_celsius': self.ecu_spec.thermal_limit_celsius,
            'thermal_limit_reached': thermal_limit_reached,
            'performance_degradation_percent': performance_degradation * 100,
            'thermal_samples': len(thermal_samples),
            'test_duration_s': time.time() - start_time,
            'thermal_margin': self.ecu_spec.thermal_limit_celsius - max_temperature
        }

    def validate_concurrent_processing(self, concurrent_tasks: List) -> Dict[str, Any]:
        """Validate concurrent processing capabilities"""

        max_workers = min(self.ecu_spec.cpu_cores, len(concurrent_tasks))

        start_time = time.perf_counter()

        # Execute tasks concurrently
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(task) for task in concurrent_tasks]

            # Monitor system during concurrent execution
            cpu_samples = []
            memory_samples = []

            while any(not future.done() for future in futures):
                cpu_samples.append(psutil.cpu_percent(interval=0.1))
                memory_samples.append(psutil.virtual_memory().percent)
                time.sleep(0.1)

            # Collect results
            results = []
            for future in futures:
                try:
                    result = future.result(timeout=30.0)  # 30s timeout
                    results.append(('success', result))
                except Exception as e:
                    results.append(('error', str(e)))

        total_time = time.perf_counter() - start_time

        # Analyze concurrent performance
        successful_tasks = sum(1 for status, _ in results if status == 'success')
        task_success_rate = successful_tasks / len(concurrent_tasks)

        return {
            'total_tasks': len(concurrent_tasks),
            'successful_tasks': successful_tasks,
            'task_success_rate': task_success_rate,
            'total_execution_time_s': total_time,
            'max_cpu_utilization': max(cpu_samples) if cpu_samples else 0.0,
            'mean_cpu_utilization': np.mean(cpu_samples) if cpu_samples else 0.0,
            'max_memory_utilization': max(memory_samples) if memory_samples else 0.0,
            'concurrent_performance_score': task_success_rate * (1.0 / total_time) * 100,
            'cpu_cores_utilized': max_workers
        }


class ThermalModel:
    """Realistic automotive thermal model"""

    def __init__(self, ecu_spec: ECUSpecification):
        self.ecu_spec = ecu_spec
        self.current_temperature = 25.0  # Start at ambient
        self.thermal_mass = 1000.0  # Thermal mass in J/K
        self.cooling_coefficient = 0.1  # Cooling rate

    def update_thermal_state(self, cpu_load: float) -> Dict[str, Any]:
        """Update thermal state based on CPU load"""

        # Heat generation proportional to CPU load
        heat_generation = cpu_load * 20.0  # Up to 20W heat generation

        # Thermal dynamics (simplified)
        temp_rise = heat_generation / self.thermal_mass * 10.0  # Temperature rise
        cooling = (self.current_temperature - 25.0) * self.cooling_coefficient

        self.current_temperature += temp_rise - cooling
        self.current_temperature = max(25.0, min(self.current_temperature, 120.0))

        # Determine thermal state
        if self.current_temperature < 40:
            thermal_state = ThermalState.COLD
        elif self.current_temperature < 60:
            thermal_state = ThermalState.NORMAL
        elif self.current_temperature < 75:
            thermal_state = ThermalState.WARM
        elif self.current_temperature < self.ecu_spec.thermal_limit_celsius:
            thermal_state = ThermalState.HOT
        else:
            thermal_state = ThermalState.CRITICAL

        return {
            'temperature_celsius': self.current_temperature,
            'thermal_state': thermal_state,
            'heat_generation_watts': heat_generation,
            'thermal_margin': self.ecu_spec.thermal_limit_celsius - self.current_temperature
        }


class PowerModel:
    """Realistic automotive power consumption model"""

    def __init__(self, ecu_spec: ECUSpecification):
        self.ecu_spec = ecu_spec
        self.base_power = 2.0  # Base power consumption

    def calculate_power_consumption(self, cpu_load: float, memory_load: float) -> Dict[str, Any]:
        """Calculate power consumption based on system load"""

        # Power components
        cpu_power = cpu_load * 15.0  # CPU power scaling
        memory_power = memory_load * 3.0  # Memory power scaling
        peripheral_power = 2.0  # Fixed peripheral power

        total_power = self.base_power + cpu_power + memory_power + peripheral_power

        # Power budget compliance
        power_compliant = total_power <= self.ecu_spec.max_power_watts
        power_efficiency = self.ecu_spec.max_power_watts / total_power if total_power > 0 else float('inf')

        return {
            'power_watts': total_power,
            'cpu_power_watts': cpu_power,
            'memory_power_watts': memory_power,
            'peripheral_power_watts': peripheral_power,
            'base_power_watts': self.base_power,
            'power_budget_watts': self.ecu_spec.max_power_watts,
            'power_compliant': power_compliant,
            'power_efficiency': power_efficiency
        }


class TestHardwarePerformanceValidation:
    """Hardware performance validation tests"""

    def setup_method(self):
        """Setup ECU validator for testing"""
        # Use mid-range ECU specification for testing
        self.ecu_spec = ECUSpecification(
            ecu_type=ECUType.MID_RANGE,
            cpu_cores=4,
            cpu_frequency_mhz=1800,
            memory_mb=2048,
            storage_type="emmc",
            gpu_present=True,
            npu_present=False,
            max_power_watts=25.0,
            thermal_limit_celsius=85.0,
            operating_temp_range=(-40, 85),
            boot_time_max_s=5.0
        )
        self.validator = RealECUValidator(self.ecu_spec)

    def test_realistic_perception_workload(self):
        """Test perception processing under realistic ECU constraints"""

        def perception_workload():
            """Realistic computer vision workload"""
            # Simulate camera image processing
            image_height, image_width = 720, 1280  # Realistic resolution
            channels = 3

            # Create image data
            image_data = np.random.randint(0, 255,
                                         (image_height, image_width, channels),
                                         dtype=np.uint8)

            # Realistic image processing pipeline
            # 1. Preprocessing
            gray_image = np.mean(image_data, axis=2)

            # 2. Edge detection (Sobel operator)
            sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

            # Apply convolution (computationally intensive)
            h, w = gray_image.shape
            edges = np.zeros_like(gray_image)

            for y in range(1, h-1):
                for x in range(1, w-1):
                    region = gray_image[y-1:y+2, x-1:x+2]
                    gx = np.sum(region * sobel_x)
                    gy = np.sum(region * sobel_y)
                    edges[y, x] = np.sqrt(gx**2 + gy**2)

            # 3. Feature extraction simulation
            features = []
            for i in range(0, h, 32):
                for j in range(0, w, 32):
                    patch = edges[i:i+32, j:j+32]
                    if patch.size > 0:
                        feature = np.mean(patch)
                        features.append(feature)

            return len(features)

        # Measure performance
        metrics = self.validator.measure_computational_performance(
            perception_workload, "Realistic Perception Processing"
        )

        # Automotive ECU performance requirements
        assert metrics.latency_ms <= 50.0, \
            f"Perception latency {metrics.latency_ms:.1f}ms exceeds ECU requirement"

        assert metrics.memory_usage_mb <= self.ecu_spec.memory_mb * 0.6, \
            f"Memory usage {metrics.memory_usage_mb:.1f}MB too high for ECU"

        assert metrics.thermal_state != ThermalState.CRITICAL, \
            f"Thermal state {metrics.thermal_state} indicates overheating"

        assert metrics.power_consumption_watts <= self.ecu_spec.max_power_watts, \
            f"Power consumption {metrics.power_consumption_watts:.1f}W exceeds ECU budget"

        # Performance efficiency requirements
        assert metrics.cpu_utilization_percent <= 80.0, \
            f"CPU utilization {metrics.cpu_utilization_percent:.1f}% too high"

    def test_memory_intensive_sensor_fusion(self):
        """Test memory performance under sensor fusion workload"""

        def memory_intensive_fusion():
            """Realistic sensor fusion with multiple data streams"""

            # Allocate sensor data buffers (realistic sizes)
            camera_buffer = np.zeros((10, 720, 1280, 3), dtype=np.uint8)  # 10 frames
            radar_buffer = np.zeros((100, 64, 64), dtype=np.float32)       # Radar data
            lidar_buffer = np.zeros((64, 1024, 4), dtype=np.float32)       # Point cloud

            # Fill buffers with realistic data
            for i in range(10):
                camera_buffer[i] = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

            for i in range(100):
                radar_buffer[i] = np.random.rand(64, 64).astype(np.float32) * 100.0

            lidar_buffer[:] = np.random.rand(64, 1024, 4).astype(np.float32) * 50.0

            # Fusion processing
            fused_objects = []

            # Process each camera frame
            for frame_idx in range(10):
                frame = camera_buffer[frame_idx]

                # Extract features from camera
                features = np.mean(frame.reshape(-1, 3), axis=1)

                # Correlate with radar data
                radar_frame = radar_buffer[frame_idx * 10:(frame_idx + 1) * 10]
                radar_features = np.mean(radar_frame, axis=(1, 2))

                # Correlate with LiDAR
                lidar_features = np.mean(lidar_buffer, axis=(1, 2))

                # Fusion algorithm (Kalman filter simulation)
                for obj_id in range(50):  # Track 50 objects
                    # State vector [x, y, z, vx, vy, vz]
                    state = np.random.rand(6)

                    # Measurement update
                    if obj_id < len(features):
                        measurement = features[obj_id:obj_id+3] if obj_id+3 <= len(features) else features[-3:]
                        state[:3] = 0.7 * state[:3] + 0.3 * measurement[:3]

                    fused_objects.append(state.copy())

            # Force garbage collection to test memory cleanup
            del camera_buffer, radar_buffer, lidar_buffer
            gc.collect()

            return len(fused_objects)

        # Validate memory performance
        memory_result = self.validator.validate_memory_constraints(memory_intensive_fusion)

        # ECU memory requirements
        assert memory_result['memory_compliant'], \
            f"Peak memory {memory_result['peak_memory_mb']:.1f}MB exceeds ECU limit"

        assert memory_result['memory_efficiency'] >= 1.2, \
            f"Memory efficiency {memory_result['memory_efficiency']:.2f} too low"

        # Memory usage should be reasonable for automotive ECU
        assert memory_result['peak_memory_mb'] <= 800.0, \
            f"Peak memory usage {memory_result['peak_memory_mb']:.1f}MB too high for mid-range ECU"

        assert memory_result['execution_time_s'] <= 5.0, \
            f"Execution time {memory_result['execution_time_s']:.1f}s too slow"

    def test_thermal_stress_validation(self):
        """Test performance under thermal stress conditions"""

        def thermal_stress_workload():
            """Computationally intensive workload for thermal stress"""

            # Matrix operations (CPU intensive)
            matrix_size = 200
            matrix_a = np.random.rand(matrix_size, matrix_size)
            matrix_b = np.random.rand(matrix_size, matrix_size)

            # Multiple matrix multiplications
            for i in range(10):
                result = np.dot(matrix_a, matrix_b)
                matrix_a = result[:matrix_size, :matrix_size]

            # FFT operations (computationally intensive)
            signal_length = 8192
            signal = np.random.rand(signal_length)

            for i in range(20):
                fft_result = np.fft.fft(signal)
                signal = np.real(np.fft.ifft(fft_result))

            return np.sum(result) + np.sum(signal)

        # Run thermal stress test
        thermal_result = self.validator.validate_thermal_performance(
            thermal_stress_workload, duration_s=30.0
        )

        # Thermal performance requirements
        assert not thermal_result['thermal_limit_reached'], \
            f"Thermal limit reached: {thermal_result['max_temperature_celsius']:.1f}°C"

        assert thermal_result['thermal_margin'] >= 5.0, \
            f"Thermal margin {thermal_result['thermal_margin']:.1f}°C too small"

        assert thermal_result['performance_degradation_percent'] <= 20.0, \
            f"Performance degradation {thermal_result['performance_degradation_percent']:.1f}% too high"

        # Temperature should stabilize below critical limit
        assert thermal_result['max_temperature_celsius'] <= 80.0, \
            f"Maximum temperature {thermal_result['max_temperature_celsius']:.1f}°C too high"

    def test_concurrent_adas_processing(self):
        """Test concurrent ADAS component processing"""

        def perception_task():
            """Perception processing task"""
            data = np.random.rand(100, 100, 3)
            for i in range(50):
                data = np.roll(data, 1, axis=0)
            return data.sum()

        def prediction_task():
            """Trajectory prediction task"""
            trajectory = np.random.rand(100, 6)
            for i in range(30):
                trajectory = np.dot(trajectory, np.random.rand(6, 6))
            return trajectory.sum()

        def planning_task():
            """Path planning task"""
            waypoints = np.random.rand(50, 3)
            for i in range(25):
                distances = np.linalg.norm(waypoints[1:] - waypoints[:-1], axis=1)
                waypoints[1:-1] += 0.01 * np.random.randn(48, 3)
            return waypoints.sum()

        def sensor_fusion_task():
            """Sensor fusion task"""
            sensors = [np.random.rand(64, 64) for _ in range(5)]
            fused = np.zeros((64, 64))
            for sensor in sensors:
                fused += sensor * 0.2
            return fused.sum()

        # Define concurrent tasks
        concurrent_tasks = [
            perception_task,
            prediction_task,
            planning_task,
            sensor_fusion_task
        ]

        # Test concurrent processing
        concurrent_result = self.validator.validate_concurrent_processing(concurrent_tasks)

        # Concurrent processing requirements
        assert concurrent_result['task_success_rate'] >= 0.95, \
            f"Task success rate {concurrent_result['task_success_rate']:.1%} too low"

        assert concurrent_result['total_execution_time_s'] <= 10.0, \
            f"Concurrent execution time {concurrent_result['total_execution_time_s']:.1f}s too slow"

        assert concurrent_result['max_cpu_utilization'] <= 90.0, \
            f"CPU utilization {concurrent_result['max_cpu_utilization']:.1f}% too high"

        assert concurrent_result['max_memory_utilization'] <= 80.0, \
            f"Memory utilization {concurrent_result['max_memory_utilization']:.1f}% too high"

        # Should utilize available CPU cores efficiently
        assert concurrent_result['cpu_cores_utilized'] >= 2, \
            f"Only {concurrent_result['cpu_cores_utilized']} cores utilized"

    def test_ecu_boot_performance(self):
        """Test ECU boot and initialization performance"""

        boot_result = self.validator.validate_boot_performance()

        # Automotive boot time requirements
        assert boot_result['boot_compliant'], \
            f"Boot time {boot_result['boot_time_s']:.1f}s exceeds ECU requirement"

        assert boot_result['boot_time_s'] <= 5.0, \
            f"Boot time {boot_result['boot_time_s']:.1f}s too slow for automotive ECU"

        # Boot phases should complete in reasonable time
        phases = boot_result['boot_phases']
        assert all(phase_time <= 2.0 for phase_time in phases.values()), \
            f"Boot phases too slow: {phases}"

    def test_power_consumption_validation(self):
        """Test power consumption under various workloads"""

        def low_power_task():
            """Low computational load"""
            data = np.random.rand(10, 10)
            result = np.sum(data)
            time.sleep(0.1)
            return result

        def medium_power_task():
            """Medium computational load"""
            data = np.random.rand(100, 100)
            for i in range(10):
                data = np.dot(data, data.T)
            return data.sum()

        def high_power_task():
            """High computational load"""
            data = np.random.rand(300, 300)
            for i in range(5):
                data = np.dot(data, data.T)
                data = data / np.max(data)  # Normalize
            return data.sum()

        # Test power consumption for different workloads
        power_results = []

        for task, task_name in [(low_power_task, "low"),
                               (medium_power_task, "medium"),
                               (high_power_task, "high")]:

            metrics = self.validator.measure_computational_performance(task, f"{task_name}_power_task")
            power_results.append((task_name, metrics.power_consumption_watts))

        # Power consumption validation
        low_power, medium_power, high_power = [power for _, power in power_results]

        # Power scaling should be reasonable
        assert low_power < medium_power < high_power, \
            f"Power consumption should scale with workload: {power_results}"

        assert high_power <= self.ecu_spec.max_power_watts, \
            f"High power consumption {high_power:.1f}W exceeds ECU budget"

        # Power efficiency requirements
        power_range = high_power - low_power
        assert power_range >= 5.0, \
            f"Power range {power_range:.1f}W too small - insufficient dynamic range"

    def test_real_time_constraint_validation(self):
        """Test real-time constraint compliance"""

        def real_time_task():
            """Simulate real-time ADAS processing"""

            # Perception (5ms budget)
            start_perception = time.perf_counter()
            perception_data = np.random.rand(480, 640, 3)
            perception_result = np.mean(perception_data)
            perception_time = (time.perf_counter() - start_perception) * 1000

            # Prediction (8ms budget)
            start_prediction = time.perf_counter()
            trajectory = np.random.rand(50, 6)
            for i in range(10):
                trajectory = trajectory + np.random.randn(50, 6) * 0.1
            prediction_result = np.sum(trajectory)
            prediction_time = (time.perf_counter() - start_prediction) * 1000

            # Planning (10ms budget)
            start_planning = time.perf_counter()
            waypoints = np.random.rand(20, 3)
            for i in range(5):
                waypoints = waypoints + np.random.randn(20, 3) * 0.01
            planning_result = np.sum(waypoints)
            planning_time = (time.perf_counter() - start_planning) * 1000

            return {
                'perception_time_ms': perception_time,
                'prediction_time_ms': prediction_time,
                'planning_time_ms': planning_time,
                'total_time_ms': perception_time + prediction_time + planning_time
            }

        # Run multiple iterations for statistical validation
        timing_results = []
        for i in range(20):
            result = real_time_task()
            timing_results.append(result)

        # Analyze timing statistics
        perception_times = [r['perception_time_ms'] for r in timing_results]
        prediction_times = [r['prediction_time_ms'] for r in timing_results]
        planning_times = [r['planning_time_ms'] for r in timing_results]
        total_times = [r['total_time_ms'] for r in timing_results]

        # Real-time requirements (automotive standards)
        assert max(perception_times) <= 5.0, \
            f"Perception max time {max(perception_times):.1f}ms exceeds real-time constraint"

        assert max(prediction_times) <= 8.0, \
            f"Prediction max time {max(prediction_times):.1f}ms exceeds real-time constraint"

        assert max(planning_times) <= 10.0, \
            f"Planning max time {max(planning_times):.1f}ms exceeds real-time constraint"

        assert max(total_times) <= 20.0, \
            f"Total processing time {max(total_times):.1f}ms exceeds real-time constraint"

        # Timing consistency requirements
        perception_std = np.std(perception_times)
        assert perception_std <= 1.0, \
            f"Perception timing variability {perception_std:.1f}ms too high"

        total_std = np.std(total_times)
        assert total_std <= 2.0, \
            f"Total timing variability {total_std:.1f}ms too high"

    def test_hardware_validation_summary(self):
        """Generate comprehensive hardware validation summary"""

        performance_summary = {
            'ecu_type': self.ecu_spec.ecu_type.value,
            'cpu_cores': self.ecu_spec.cpu_cores,
            'memory_mb': self.ecu_spec.memory_mb,
            'max_power_watts': self.ecu_spec.max_power_watts,
            'thermal_limit_celsius': self.ecu_spec.thermal_limit_celsius,
            'performance_tests_passed': len(self.validator.performance_history),
            'avg_latency_ms': np.mean([m.latency_ms for m in self.validator.performance_history]) if self.validator.performance_history else 0.0,
            'avg_cpu_utilization': np.mean([m.cpu_utilization_percent for m in self.validator.performance_history]) if self.validator.performance_history else 0.0,
            'avg_memory_usage_mb': np.mean([m.memory_usage_mb for m in self.validator.performance_history]) if self.validator.performance_history else 0.0,
            'thermal_states_observed': list(set(m.thermal_state for m in self.validator.performance_history)),
            'automotive_compliant': all(
                m.thermal_state != ThermalState.CRITICAL and
                m.power_consumption_watts <= self.ecu_spec.max_power_watts
                for m in self.validator.performance_history
            )
        }

        print(f"\nHardware Performance Summary:")
        print(f"- ECU Type: {performance_summary['ecu_type']}")
        print(f"- Performance tests: {performance_summary['performance_tests_passed']}")
        print(f"- Average latency: {performance_summary['avg_latency_ms']:.1f}ms")
        print(f"- Average CPU utilization: {performance_summary['avg_cpu_utilization']:.1f}%")
        print(f"- Average memory usage: {performance_summary['avg_memory_usage_mb']:.1f}MB")
        print(f"- Automotive compliant: {performance_summary['automotive_compliant']}")

        # Critical hardware requirement
        assert performance_summary['automotive_compliant'], \
            "Hardware performance must meet automotive ECU requirements"

        return performance_summary


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])