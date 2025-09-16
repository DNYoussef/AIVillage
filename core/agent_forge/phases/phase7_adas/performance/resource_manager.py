"""
ADAS Resource Manager - CPU/GPU/Memory/Power Management
Automotive-grade resource allocation and thermal management
"""

import psutil
import threading
import time
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
from collections import deque
import subprocess
import os


class ResourceType(Enum):
    """Types of system resources"""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    POWER = "power"
    THERMAL = "thermal"
    NETWORK = "network"
    STORAGE = "storage"


class PowerMode(Enum):
    """Power management modes for automotive ECUs"""
    ECO = "eco"                    # Maximum efficiency
    BALANCED = "balanced"          # Balance performance/power
    PERFORMANCE = "performance"    # Maximum performance
    CRITICAL = "critical"          # Emergency maximum performance


class ThermalState(Enum):
    """Thermal management states"""
    NORMAL = "normal"          # < 60C
    ELEVATED = "elevated"      # 60-75C
    HIGH = "high"             # 75-85C
    CRITICAL = "critical"     # > 85C
    EMERGENCY = "emergency"   # > 95C


@dataclass
class ResourceUsage:
    """Resource usage snapshot"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    gpu_percent: float
    gpu_memory_mb: float
    temperature_c: float
    power_watts: float
    network_mbps: float
    storage_iops: float


@dataclass
class ResourceLimits:
    """Resource allocation limits"""
    max_cpu_percent: float = 80.0
    max_memory_percent: float = 75.0
    max_gpu_percent: float = 90.0
    max_temperature_c: float = 85.0
    max_power_watts: float = 100.0
    thermal_throttle_temp: float = 80.0


@dataclass
class ResourceAllocation:
    """Resource allocation configuration"""
    cpu_cores: List[int]
    memory_mb: int
    gpu_memory_mb: int
    power_mode: PowerMode
    priority: int
    thermal_policy: str


class ResourceManager:
    """
    Automotive-grade resource manager for ADAS systems
    Handles CPU/GPU allocation, memory management, power optimization, thermal control
    """
    
    def __init__(self, 
                 limits: Optional[ResourceLimits] = None,
                 monitoring_interval: float = 0.1):
        self.limits = limits or ResourceLimits()
        self.monitoring_interval = monitoring_interval
        
        # Resource tracking
        self.usage_history: deque = deque(maxlen=1000)
        self.allocations: Dict[str, ResourceAllocation] = {}
        self.resource_pools: Dict[ResourceType, Any] = {}
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Alerts and violations
        self.alerts: List[str] = []
        self.violations: List[Dict] = []
        
        # Hardware detection
        self.hardware_info = self._detect_hardware()
        self.platform_type = self._detect_platform()
        
        # Power management
        self.current_power_mode = PowerMode.BALANCED
        self.thermal_state = ThermalState.NORMAL
        self.throttling_active = False
        
        # Performance counters
        self.allocation_count = 0
        self.deallocation_count = 0
        self.throttle_events = 0
        
        print(f"ResourceManager initialized for {self.platform_type}")
        print(f"Hardware: {self.hardware_info}")
        print(f"Limits: CPU {self.limits.max_cpu_percent}%, "
              f"Memory {self.limits.max_memory_percent}%, "
              f"Thermal {self.limits.max_temperature_c}C")
    
    def _detect_hardware(self) -> Dict[str, Any]:
        """Detect hardware configuration"""
        return {
            "cpu_count": psutil.cpu_count(logical=False),
            "cpu_logical": psutil.cpu_count(logical=True),
            "memory_gb": psutil.virtual_memory().total // (1024**3),
            "cpu_freq_ghz": psutil.cpu_freq().current / 1000 if psutil.cpu_freq() else 0,
            "architecture": os.uname().machine if hasattr(os, 'uname') else 'unknown'
        }
    
    def _detect_platform(self) -> str:
        """Detect platform type for automotive ECUs"""
        # In real implementation, would detect specific platforms
        arch = self.hardware_info.get("architecture", "").lower()
        
        if "aarch64" in arch or "arm" in arch:
            if self.hardware_info.get("cpu_count", 0) >= 8:
                return "snapdragon_ride"  # High-performance ARM
            else:
                return "automotive_ecu"   # Standard automotive ECU
        elif "x86" in arch:
            if self.hardware_info.get("memory_gb", 0) >= 16:
                return "nvidia_drive"     # x86 with high memory
            else:
                return "industrial_pc"    # Standard industrial PC
        else:
            return "generic_ecu"
    
    def allocate_resources(self, 
                          task_id: str,
                          cpu_cores: Optional[List[int]] = None,
                          memory_mb: Optional[int] = None,
                          gpu_memory_mb: Optional[int] = None,
                          power_mode: PowerMode = PowerMode.BALANCED,
                          priority: int = 0) -> ResourceAllocation:
        """
        Allocate resources for a specific task
        
        Args:
            task_id: Unique task identifier
            cpu_cores: Specific CPU cores to allocate
            memory_mb: Memory allocation in MB
            gpu_memory_mb: GPU memory allocation in MB
            power_mode: Power management mode
            priority: Task priority (higher = more resources)
            
        Returns:
            ResourceAllocation object
        """
        
        # Validate current resource availability
        current_usage = self._get_current_usage()
        if not self._validate_allocation_request(current_usage, memory_mb, gpu_memory_mb):
            raise ResourceError("Insufficient resources available")
        
        # Determine optimal allocation
        allocation = self._calculate_optimal_allocation(
            cpu_cores, memory_mb, gpu_memory_mb, power_mode, priority
        )
        
        # Apply allocation
        self._apply_resource_allocation(task_id, allocation)
        
        # Store allocation
        self.allocations[task_id] = allocation
        self.allocation_count += 1
        
        print(f"Allocated resources for {task_id}: "
              f"CPU cores {allocation.cpu_cores}, "
              f"Memory {allocation.memory_mb}MB, "
              f"Power mode {allocation.power_mode.value}")
        
        return allocation
    
    def deallocate_resources(self, task_id: str) -> bool:
        """
        Deallocate resources for a specific task
        
        Args:
            task_id: Task identifier to deallocate
            
        Returns:
            True if successful, False otherwise
        """
        if task_id not in self.allocations:
            return False
        
        allocation = self.allocations[task_id]
        
        # Remove allocation
        self._remove_resource_allocation(task_id, allocation)
        del self.allocations[task_id]
        self.deallocation_count += 1
        
        print(f"Deallocated resources for {task_id}")
        return True
    
    def _calculate_optimal_allocation(self,
                                    cpu_cores: Optional[List[int]],
                                    memory_mb: Optional[int],
                                    gpu_memory_mb: Optional[int],
                                    power_mode: PowerMode,
                                    priority: int) -> ResourceAllocation:
        """Calculate optimal resource allocation"""
        
        # CPU core allocation
        if cpu_cores is None:
            available_cores = list(range(self.hardware_info["cpu_count"]))
            # Allocate based on priority and current usage
            if priority > 5:
                cpu_cores = available_cores[:2]  # High priority gets 2 cores
            else:
                cpu_cores = available_cores[:1]  # Normal priority gets 1 core
        
        # Memory allocation
        if memory_mb is None:
            total_memory = self.hardware_info["memory_gb"] * 1024
            if priority > 5:
                memory_mb = min(2048, int(total_memory * 0.3))  # 30% for high priority
            else:
                memory_mb = min(1024, int(total_memory * 0.2))  # 20% for normal
        
        # GPU memory allocation
        if gpu_memory_mb is None:
            gpu_memory_mb = 512 if priority > 5 else 256
        
        # Thermal policy based on current state
        thermal_policy = self._get_thermal_policy()
        
        return ResourceAllocation(
            cpu_cores=cpu_cores,
            memory_mb=memory_mb,
            gpu_memory_mb=gpu_memory_mb,
            power_mode=power_mode,
            priority=priority,
            thermal_policy=thermal_policy
        )
    
    def _apply_resource_allocation(self, task_id: str, allocation: ResourceAllocation):
        """Apply resource allocation to system"""
        
        # CPU affinity (platform specific)
        if self.platform_type in ["nvidia_drive", "snapdragon_ride"]:
            self._set_cpu_affinity(task_id, allocation.cpu_cores)
        
        # Memory limits (using cgroups on Linux)
        if hasattr(os, 'uname') and 'linux' in os.uname().sysname.lower():
            self._set_memory_limit(task_id, allocation.memory_mb)
        
        # Power mode
        self._apply_power_mode(allocation.power_mode)
        
        # GPU memory (platform specific)
        self._allocate_gpu_memory(task_id, allocation.gpu_memory_mb)
    
    def _remove_resource_allocation(self, task_id: str, allocation: ResourceAllocation):
        """Remove resource allocation from system"""
        
        # Remove CPU affinity
        self._clear_cpu_affinity(task_id)
        
        # Remove memory limits
        self._clear_memory_limit(task_id)
        
        # Free GPU memory
        self._free_gpu_memory(task_id)
    
    def _set_cpu_affinity(self, task_id: str, cpu_cores: List[int]):
        """Set CPU affinity for task"""
        try:
            # In real implementation, would set affinity for specific process
            print(f"Set CPU affinity for {task_id}: cores {cpu_cores}")
        except Exception as e:
            print(f"Failed to set CPU affinity: {e}")
    
    def _clear_cpu_affinity(self, task_id: str):
        """Clear CPU affinity for task"""
        print(f"Cleared CPU affinity for {task_id}")
    
    def _set_memory_limit(self, task_id: str, memory_mb: int):
        """Set memory limit using cgroups"""
        try:
            # In real implementation, would use cgroups
            print(f"Set memory limit for {task_id}: {memory_mb}MB")
        except Exception as e:
            print(f"Failed to set memory limit: {e}")
    
    def _clear_memory_limit(self, task_id: str):
        """Clear memory limit"""
        print(f"Cleared memory limit for {task_id}")
    
    def _allocate_gpu_memory(self, task_id: str, gpu_memory_mb: int):
        """Allocate GPU memory"""
        try:
            # Platform-specific GPU memory allocation
            if self.platform_type == "nvidia_drive":
                print(f"Allocated CUDA memory for {task_id}: {gpu_memory_mb}MB")
            elif self.platform_type == "snapdragon_ride":
                print(f"Allocated Adreno memory for {task_id}: {gpu_memory_mb}MB")
            else:
                print(f"Allocated generic GPU memory for {task_id}: {gpu_memory_mb}MB")
        except Exception as e:
            print(f"Failed to allocate GPU memory: {e}")
    
    def _free_gpu_memory(self, task_id: str):
        """Free GPU memory"""
        print(f"Freed GPU memory for {task_id}")
    
    def _apply_power_mode(self, power_mode: PowerMode):
        """Apply power management mode"""
        self.current_power_mode = power_mode
        
        if self.platform_type == "snapdragon_ride":
            # Snapdragon-specific power scaling
            self._apply_snapdragon_power_mode(power_mode)
        elif self.platform_type == "nvidia_drive":
            # NVIDIA Drive power management
            self._apply_nvidia_power_mode(power_mode)
        else:
            # Generic power management
            self._apply_generic_power_mode(power_mode)
    
    def _apply_snapdragon_power_mode(self, power_mode: PowerMode):
        """Apply Snapdragon-specific power mode"""
        modes = {
            PowerMode.ECO: "powersave",
            PowerMode.BALANCED: "balanced",
            PowerMode.PERFORMANCE: "performance",
            PowerMode.CRITICAL: "performance_max"
        }
        print(f"Applied Snapdragon power mode: {modes[power_mode]}")
    
    def _apply_nvidia_power_mode(self, power_mode: PowerMode):
        """Apply NVIDIA Drive power mode"""
        modes = {
            PowerMode.ECO: "MIN_POWER",
            PowerMode.BALANCED: "BALANCED",
            PowerMode.PERFORMANCE: "MAX_PERFORMANCE",
            PowerMode.CRITICAL: "UNRESTRICTED"
        }
        print(f"Applied NVIDIA Drive power mode: {modes[power_mode]}")
    
    def _apply_generic_power_mode(self, power_mode: PowerMode):
        """Apply generic power mode"""
        print(f"Applied generic power mode: {power_mode.value}")
    
    def start_monitoring(self):
        """Start resource monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        print("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        print("Resource monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                usage = self._get_current_usage()
                self.usage_history.append(usage)
                
                # Check for violations
                self._check_resource_violations(usage)
                
                # Update thermal state
                self._update_thermal_state(usage.temperature_c)
                
                # Apply thermal throttling if needed
                self._apply_thermal_management(usage)
                
            except Exception as e:
                print(f"Monitoring error: {e}")
            
            time.sleep(self.monitoring_interval)
    
    def _get_current_usage(self) -> ResourceUsage:
        """Get current resource usage"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=None)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_mb = memory.used / 1024 / 1024
        
        # GPU usage (simulated)
        gpu_percent = 0.0
        gpu_memory_mb = 0.0
        
        # Temperature (simulated)
        temperature_c = self._get_temperature()
        
        # Power (simulated)
        power_watts = self._estimate_power_consumption(cpu_percent, gpu_percent)
        
        # Network (simulated)
        network_mbps = 0.0
        
        # Storage (simulated)
        storage_iops = 0.0
        
        return ResourceUsage(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_mb=memory_mb,
            gpu_percent=gpu_percent,
            gpu_memory_mb=gpu_memory_mb,
            temperature_c=temperature_c,
            power_watts=power_watts,
            network_mbps=network_mbps,
            storage_iops=storage_iops
        )
    
    def _get_temperature(self) -> float:
        """Get system temperature"""
        try:
            # Try to get thermal sensor data
            if hasattr(psutil, 'sensors_temperatures'):
                temps = psutil.sensors_temperatures()
                if temps:
                    # Get CPU temperature
                    for name, entries in temps.items():
                        if 'cpu' in name.lower() or 'core' in name.lower():
                            if entries:
                                return entries[0].current
            
            # Fallback: estimate based on CPU usage
            cpu_usage = psutil.cpu_percent(interval=None)
            base_temp = 45.0  # Base temperature
            return base_temp + (cpu_usage * 0.5)  # Rough estimation
            
        except Exception:
            return 50.0  # Default safe temperature
    
    def _estimate_power_consumption(self, cpu_percent: float, gpu_percent: float) -> float:
        """Estimate power consumption"""
        # Base power consumption
        base_power = 20.0  # Watts
        
        # CPU power scaling
        cpu_power = (cpu_percent / 100.0) * 30.0  # Max 30W for CPU
        
        # GPU power scaling
        gpu_power = (gpu_percent / 100.0) * 50.0  # Max 50W for GPU
        
        return base_power + cpu_power + gpu_power
    
    def _check_resource_violations(self, usage: ResourceUsage):
        """Check for resource limit violations"""
        violations = []
        
        if usage.cpu_percent > self.limits.max_cpu_percent:
            violations.append(f"CPU usage {usage.cpu_percent:.1f}% > {self.limits.max_cpu_percent}%")
        
        if usage.memory_percent > self.limits.max_memory_percent:
            violations.append(f"Memory usage {usage.memory_percent:.1f}% > {self.limits.max_memory_percent}%")
        
        if usage.temperature_c > self.limits.max_temperature_c:
            violations.append(f"Temperature {usage.temperature_c:.1f}C > {self.limits.max_temperature_c}C")
        
        if usage.power_watts > self.limits.max_power_watts:
            violations.append(f"Power {usage.power_watts:.1f}W > {self.limits.max_power_watts}W")
        
        # Store violations
        for violation in violations:
            self.violations.append({
                "timestamp": usage.timestamp,
                "type": "resource_violation",
                "message": violation
            })
            self.alerts.append(violation)
        
        # Limit alerts to last 100
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
    
    def _update_thermal_state(self, temperature_c: float):
        """Update thermal management state"""
        old_state = self.thermal_state
        
        if temperature_c < 60:
            self.thermal_state = ThermalState.NORMAL
        elif temperature_c < 75:
            self.thermal_state = ThermalState.ELEVATED
        elif temperature_c < 85:
            self.thermal_state = ThermalState.HIGH
        elif temperature_c < 95:
            self.thermal_state = ThermalState.CRITICAL
        else:
            self.thermal_state = ThermalState.EMERGENCY
        
        if old_state != self.thermal_state:
            print(f"Thermal state changed: {old_state.value} -> {self.thermal_state.value}")
    
    def _apply_thermal_management(self, usage: ResourceUsage):
        """Apply thermal management policies"""
        if self.thermal_state in [ThermalState.CRITICAL, ThermalState.EMERGENCY]:
            if not self.throttling_active:
                self._enable_thermal_throttling()
                self.throttling_active = True
                self.throttle_events += 1
        elif self.thermal_state == ThermalState.NORMAL:
            if self.throttling_active:
                self._disable_thermal_throttling()
                self.throttling_active = False
    
    def _enable_thermal_throttling(self):
        """Enable thermal throttling"""
        print("THERMAL THROTTLING ENABLED - Reducing performance to prevent overheating")
        
        # Reduce power mode for all allocations
        for task_id, allocation in self.allocations.items():
            if allocation.power_mode == PowerMode.PERFORMANCE:
                allocation.power_mode = PowerMode.BALANCED
                self._apply_power_mode(PowerMode.BALANCED)
    
    def _disable_thermal_throttling(self):
        """Disable thermal throttling"""
        print("Thermal throttling disabled - Normal performance restored")
    
    def _validate_allocation_request(self, 
                                   current_usage: ResourceUsage,
                                   memory_mb: Optional[int],
                                   gpu_memory_mb: Optional[int]) -> bool:
        """Validate if allocation request can be satisfied"""
        
        # Check memory availability
        if memory_mb:
            available_memory = (100 - current_usage.memory_percent) * self.hardware_info["memory_gb"] * 1024 / 100
            if memory_mb > available_memory * 0.8:  # Keep 20% buffer
                return False
        
        # Check thermal state
        if self.thermal_state in [ThermalState.CRITICAL, ThermalState.EMERGENCY]:
            return False
        
        return True
    
    def _get_thermal_policy(self) -> str:
        """Get current thermal policy"""
        policies = {
            ThermalState.NORMAL: "normal",
            ThermalState.ELEVATED: "conservative",
            ThermalState.HIGH: "aggressive_cooling",
            ThermalState.CRITICAL: "emergency_throttle",
            ThermalState.EMERGENCY: "emergency_shutdown"
        }
        return policies[self.thermal_state]
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get comprehensive resource summary"""
        if not self.usage_history:
            return {"error": "No usage data available"}
        
        latest_usage = self.usage_history[-1]
        
        return {
            "current_usage": asdict(latest_usage),
            "limits": asdict(self.limits),
            "thermal_state": self.thermal_state.value,
            "power_mode": self.current_power_mode.value,
            "active_allocations": len(self.allocations),
            "throttling_active": self.throttling_active,
            "platform": self.platform_type,
            "hardware": self.hardware_info,
            "statistics": {
                "allocations": self.allocation_count,
                "deallocations": self.deallocation_count,
                "throttle_events": self.throttle_events,
                "violations": len(self.violations),
                "alerts": len(self.alerts)
            }
        }
    
    def generate_resource_report(self) -> str:
        """Generate comprehensive resource management report"""
        summary = self.get_resource_summary()
        
        report = f"""
ADAS Resource Management Report
==============================

Platform: {summary['platform']}
Hardware: {summary['hardware']['cpu_count']} cores, {summary['hardware']['memory_gb']}GB RAM

Current Status:
- CPU Usage: {summary['current_usage']['cpu_percent']:.1f}%
- Memory Usage: {summary['current_usage']['memory_percent']:.1f}%
- Temperature: {summary['current_usage']['temperature_c']:.1f}C
- Power: {summary['current_usage']['power_watts']:.1f}W
- Thermal State: {summary['thermal_state']}
- Power Mode: {summary['power_mode']}
- Throttling: {'ACTIVE' if summary['throttling_active'] else 'INACTIVE'}

Active Allocations: {summary['active_allocations']}

Statistics:
- Total Allocations: {summary['statistics']['allocations']}
- Total Deallocations: {summary['statistics']['deallocations']}
- Throttle Events: {summary['statistics']['throttle_events']}
- Violations: {summary['statistics']['violations']}
- Active Alerts: {summary['statistics']['alerts']}

Resource Limits:
- CPU: {summary['limits']['max_cpu_percent']}%
- Memory: {summary['limits']['max_memory_percent']}%
- Temperature: {summary['limits']['max_temperature_c']}C
- Power: {summary['limits']['max_power_watts']}W
"""
        
        if self.alerts:
            report += f"\nRecent Alerts:\n"
            for alert in self.alerts[-5:]:  # Last 5 alerts
                report += f"- {alert}\n"
        
        return report
    
    def cleanup(self):
        """Cleanup resources and stop monitoring"""
        self.stop_monitoring()
        
        # Deallocate all resources
        for task_id in list(self.allocations.keys()):
            self.deallocate_resources(task_id)
        
        print("ResourceManager cleanup completed")


class ResourceError(Exception):
    """Resource management error"""
    pass


# Demo function
def demo_resource_management():
    """Demonstrate automotive resource management"""
    
    print("=== ADAS Resource Management Demo ===")
    
    # Initialize resource manager
    limits = ResourceLimits(
        max_cpu_percent=75.0,
        max_memory_percent=80.0,
        max_temperature_c=80.0,
        max_power_watts=120.0
    )
    
    manager = ResourceManager(limits)
    manager.start_monitoring()
    
    try:
        # Allocate resources for different ADAS tasks
        print("\n1. Allocating resources for object detection...")
        obj_detection = manager.allocate_resources(
            "object_detection",
            memory_mb=1024,
            power_mode=PowerMode.PERFORMANCE,
            priority=8
        )
        
        print("\n2. Allocating resources for lane tracking...")
        lane_tracking = manager.allocate_resources(
            "lane_tracking",
            memory_mb=512,
            power_mode=PowerMode.BALANCED,
            priority=6
        )
        
        print("\n3. Allocating resources for path planning...")
        path_planning = manager.allocate_resources(
            "path_planning",
            memory_mb=256,
            power_mode=PowerMode.ECO,
            priority=4
        )
        
        # Monitor for a few seconds
        print("\n4. Monitoring resources...")
        time.sleep(3)
        
        # Generate report
        print("\n" + manager.generate_resource_report())
        
        # Cleanup
        print("\n5. Deallocating resources...")
        manager.deallocate_resources("object_detection")
        manager.deallocate_resources("lane_tracking")
        manager.deallocate_resources("path_planning")
        
    finally:
        manager.cleanup()


if __name__ == "__main__":
    demo_resource_management()