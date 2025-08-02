"""Dynamic resource allocation for mobile devices - Sprint 6 Enhanced"""

import asyncio
import os
import platform
import resource as sys_resource
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

from .device_profiler import DeviceProfile

logger = logging.getLogger(__name__)

@dataclass
class ResourceAllocation:
    """Resource allocation decision with detailed constraints"""
    cpu_limit_percent: float
    memory_limit_mb: int
    network_limit_kbps: Optional[int]
    gpu_enabled: bool
    background_processing: bool
    priority_level: str  # "high", "medium", "low"
    thermal_mitigation: bool
    battery_conservation: bool
    reason: str
    constraints: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'limits': {
                'cpu_percent': self.cpu_limit_percent,
                'memory_mb': self.memory_limit_mb,
                'network_kbps': self.network_limit_kbps
            },
            'features': {
                'gpu_enabled': self.gpu_enabled,
                'background_processing': self.background_processing,
                'thermal_mitigation': self.thermal_mitigation,
                'battery_conservation': self.battery_conservation
            },
            'meta': {
                'priority_level': self.priority_level,
                'reason': self.reason,
                'constraints': self.constraints
            }
        }


class ResourceAllocator:
    """Intelligent resource allocation based on device state and policy"""
    
    def __init__(self):
        # Threshold configurations
        self.thermal_threshold = 40.0  # Celsius
        self.battery_low_threshold = 20  # Percent
        self.battery_critical_threshold = 10  # Percent
        self.memory_pressure_threshold = 85  # Percent
        self.memory_critical_threshold = 95  # Percent
        
        # Performance profiles
        self.profiles = {
            "performance": {
                "cpu_base": 100,
                "memory_base": 0.9,  # 90% of available
                "gpu_enabled": True,
                "background_processing": True
            },
            "balanced": {
                "cpu_base": 80,
                "memory_base": 0.7,  # 70% of available
                "gpu_enabled": True,
                "background_processing": True
            },
            "power_save": {
                "cpu_base": 50,
                "memory_base": 0.5,  # 50% of available
                "gpu_enabled": False,
                "background_processing": False
            },
            "critical": {
                "cpu_base": 25,
                "memory_base": 0.3,  # 30% of available
                "gpu_enabled": False,
                "background_processing": False
            }
        }
        
        # Platform-specific handlers
        self.platform_handlers = {
            "Linux": self._apply_linux_limits,
            "Darwin": self._apply_macos_limits,
            "Windows": self._apply_windows_limits,
            "Android": self._apply_android_limits
        }
        
    def calculate_allocation(self, profile: DeviceProfile) -> ResourceAllocation:
        """Calculate optimal resource allocation based on device state"""
        
        # Determine base profile
        base_profile = self._select_base_profile(profile)
        
        # Initialize allocation with base profile
        allocation = ResourceAllocation(
            cpu_limit_percent=base_profile["cpu_base"],
            memory_limit_mb=int(profile.ram_available_mb * base_profile["memory_base"]),
            network_limit_kbps=None,
            gpu_enabled=base_profile["gpu_enabled"],
            background_processing=base_profile["background_processing"],
            priority_level="medium",
            thermal_mitigation=False,
            battery_conservation=False,
            reason="base_profile",
            constraints={}
        )
        
        # Apply thermal constraints
        allocation = self._apply_thermal_constraints(allocation, profile)
        
        # Apply battery constraints
        allocation = self._apply_battery_constraints(allocation, profile)
        
        # Apply memory pressure constraints
        allocation = self._apply_memory_constraints(allocation, profile)
        
        # Apply network constraints
        allocation = self._apply_network_constraints(allocation, profile)
        
        # Final validation and adjustments
        allocation = self._validate_allocation(allocation, profile)
        
        return allocation
        
    def _select_base_profile(self, profile: DeviceProfile) -> Dict[str, Any]:
        """Select base performance profile"""
        
        # Critical battery - always use critical profile
        if profile.battery_percent and profile.battery_percent <= self.battery_critical_threshold:
            return self.profiles["critical"]
            
        # High thermal load
        if profile.cpu_temp_celsius and profile.cpu_temp_celsius > self.thermal_threshold + 10:
            return self.profiles["power_save"]
            
        # Low battery and not charging
        if (profile.battery_percent and
            profile.battery_percent <= self.battery_low_threshold and
            not profile.battery_charging):
            return self.profiles["power_save"]
            
        # Memory pressure
        memory_usage = (profile.ram_used_mb / profile.ram_total_mb) * 100
        if memory_usage > self.memory_critical_threshold:
            return self.profiles["critical"]
        elif memory_usage > self.memory_pressure_threshold:
            return self.profiles["power_save"]
            
        # Device type considerations
        if profile.device_type in ["phone", "tablet"]:
            # Mobile devices default to balanced
            return self.profiles["balanced"]
        else:
            # Laptops/desktops can use performance
            return self.profiles["performance"]
            
    def _apply_thermal_constraints(self, allocation: ResourceAllocation,
                                 profile: DeviceProfile) -> ResourceAllocation:
        """Apply thermal throttling constraints"""
        
        if not profile.cpu_temp_celsius:
            return allocation
            
        temp = profile.cpu_temp_celsius
        
        if temp > self.thermal_threshold:
            allocation.thermal_mitigation = True
            
            # Progressive throttling based on temperature
            if temp > self.thermal_threshold + 15:  # 55°C+
                throttle_factor = 0.3
                allocation.reason = f"critical_thermal_{temp:.1f}C"
            elif temp > self.thermal_threshold + 10:  # 50°C+
                throttle_factor = 0.5
                allocation.reason = f"high_thermal_{temp:.1f}C"
            else:  # 40-50°C
                throttle_factor = 0.7
                allocation.reason = f"thermal_throttle_{temp:.1f}C"
                
            allocation.cpu_limit_percent = min(
                allocation.cpu_limit_percent,
                allocation.cpu_limit_percent * throttle_factor
            )
            
            # Disable GPU under high thermal load
            if temp > self.thermal_threshold + 10:
                allocation.gpu_enabled = False
                allocation.background_processing = False
                
            allocation.constraints["thermal_throttle"] = throttle_factor
            
        return allocation
        
    def _apply_battery_constraints(self, allocation: ResourceAllocation,
                                 profile: DeviceProfile) -> ResourceAllocation:
        """Apply battery conservation constraints"""
        
        if not profile.battery_percent:
            return allocation
            
        battery = profile.battery_percent
        charging = profile.battery_charging
        
        if battery <= self.battery_critical_threshold:
            # Critical battery - maximum conservation
            allocation.battery_conservation = True
            allocation.cpu_limit_percent = min(allocation.cpu_limit_percent, 25)
            allocation.memory_limit_mb = int(allocation.memory_limit_mb * 0.5)
            allocation.network_limit_kbps = 10  # Very low bandwidth
            allocation.gpu_enabled = False
            allocation.background_processing = False
            allocation.priority_level = "low"
            allocation.reason = f"critical_battery_{battery}%"
            
        elif battery <= self.battery_low_threshold and not charging:
            # Low battery - moderate conservation
            allocation.battery_conservation = True
            allocation.cpu_limit_percent = min(allocation.cpu_limit_percent, 50)
            allocation.memory_limit_mb = int(allocation.memory_limit_mb * 0.7)
            allocation.network_limit_kbps = 50  # Limited bandwidth
            allocation.priority_level = "low"
            allocation.reason = f"low_battery_{battery}%"
            
        elif charging and battery < 50:
            # Charging but low - moderate performance
            allocation.cpu_limit_percent = min(allocation.cpu_limit_percent, 75)
            allocation.reason = f"charging_low_battery_{battery}%"
            
        allocation.constraints["battery_level"] = battery
        allocation.constraints["charging"] = charging
        
        return allocation
        
    def _apply_memory_constraints(self, allocation: ResourceAllocation,
                                profile: DeviceProfile) -> ResourceAllocation:
        """Apply memory pressure constraints"""
        
        memory_usage_percent = (profile.ram_used_mb / profile.ram_total_mb) * 100
        
        if memory_usage_percent > self.memory_critical_threshold:
            # Critical memory pressure
            allocation.memory_limit_mb = int(profile.ram_available_mb * 0.3)
            allocation.background_processing = False
            allocation.priority_level = "low"
            allocation.reason = f"critical_memory_pressure_{memory_usage_percent:.0f}%"
            
        elif memory_usage_percent > self.memory_pressure_threshold:
            # High memory pressure
            allocation.memory_limit_mb = int(profile.ram_available_mb * 0.6)
            allocation.reason = f"memory_pressure_{memory_usage_percent:.0f}%"
            
        allocation.constraints["memory_usage_percent"] = memory_usage_percent
        
        return allocation
        
    def _apply_network_constraints(self, allocation: ResourceAllocation,
                                 profile: DeviceProfile) -> ResourceAllocation:
        """Apply network-based constraints"""
        
        # Cellular networks - be more conservative
        if profile.network_type in ["cellular", "3g", "4g", "5g"]:
            if not allocation.network_limit_kbps:
                # Default cellular limit
                allocation.network_limit_kbps = 500  # 500 kbps
            else:
                # Apply more restrictive limit
                allocation.network_limit_kbps = min(allocation.network_limit_kbps, 500)
                
        # High latency networks
        if profile.network_latency_ms and profile.network_latency_ms > 200:
            allocation.network_limit_kbps = 100  # Very conservative for high latency
            
        allocation.constraints["network_type"] = profile.network_type
        allocation.constraints["network_latency_ms"] = profile.network_latency_ms
        
        return allocation
        
    def _validate_allocation(self, allocation: ResourceAllocation,
                           profile: DeviceProfile) -> ResourceAllocation:
        """Validate and adjust allocation for sanity"""
        
        # Minimum viable limits
        allocation.cpu_limit_percent = max(allocation.cpu_limit_percent, 10)
        allocation.memory_limit_mb = max(allocation.memory_limit_mb, 64)  # At least 64MB
        
        # Maximum limits based on hardware
        allocation.cpu_limit_percent = min(allocation.cpu_limit_percent, 100)
        allocation.memory_limit_mb = min(allocation.memory_limit_mb, profile.ram_available_mb)
        
        # Consistency checks
        if allocation.cpu_limit_percent <= 25:
            allocation.priority_level = "low"
            allocation.background_processing = False
            
        if allocation.memory_limit_mb <= 128:
            allocation.gpu_enabled = False
            
        return allocation
        
    async def apply_allocation(self, allocation: ResourceAllocation) -> bool:
        """Apply resource limits to current process"""
        
        current_platform = platform.system()
        handler = self.platform_handlers.get(current_platform)
        
        if handler:
            try:
                success = await handler(allocation)
                if success:
                    logger.info(f"Applied resource allocation: {allocation.reason} "
                               f"(CPU: {allocation.cpu_limit_percent}%, "
                               f"RAM: {allocation.memory_limit_mb}MB)")
                return success
            except Exception as e:
                logger.error(f"Failed to apply resource allocation: {e}")
                return False
        else:
            logger.warning(f"No resource allocation handler for {current_platform}")
            return False
            
    async def _apply_linux_limits(self, allocation: ResourceAllocation) -> bool:
        """Apply resource limits on Linux"""
        try:
            # CPU limiting via nice value and cgroups
            nice_value = int(20 * (1 - allocation.cpu_limit_percent / 100))
            os.nice(nice_value)
            
            # Memory limiting
            memory_bytes = allocation.memory_limit_mb * 1024 * 1024
            sys_resource.setrlimit(sys_resource.RLIMIT_AS, (memory_bytes, -1))
            
            # Network limiting would require tc (traffic control) - skip for now
            
            return True
        except Exception as e:
            logger.error(f"Linux resource limiting failed: {e}")
            return False
            
    async def _apply_macos_limits(self, allocation: ResourceAllocation) -> bool:
        """Apply resource limits on macOS"""
        try:
            # Similar to Linux but with macOS-specific adjustments
            nice_value = int(20 * (1 - allocation.cpu_limit_percent / 100))
            os.nice(nice_value)
            
            # Memory limiting (limited support on macOS)
            memory_bytes = allocation.memory_limit_mb * 1024 * 1024
            try:
                sys_resource.setrlimit(sys_resource.RLIMIT_AS, (memory_bytes, -1))
            except:
                # macOS may not support RLIMIT_AS
                pass
                
            return True
        except Exception as e:
            logger.error(f"macOS resource limiting failed: {e}")
            return False
            
    async def _apply_windows_limits(self, allocation: ResourceAllocation) -> bool:
        """Apply resource limits on Windows"""
        try:
            # Windows has limited process resource control from Python
            # Would typically use Job Objects or WMI
            
            # For now, just adjust process priority
            import psutil
            current_process = psutil.Process()
            
            if allocation.cpu_limit_percent <= 25:
                current_process.nice(psutil.IDLE_PRIORITY_CLASS)
            elif allocation.cpu_limit_percent <= 50:
                current_process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            elif allocation.cpu_limit_percent <= 75:
                current_process.nice(psutil.NORMAL_PRIORITY_CLASS)
            else:
                current_process.nice(psutil.HIGH_PRIORITY_CLASS)
                
            return True
        except Exception as e:
            logger.error(f"Windows resource limiting failed: {e}")
            return False
            
    async def _apply_android_limits(self, allocation: ResourceAllocation) -> bool:
        """Apply resource limits on Android"""
        try:
            # Android-specific resource management would go here
            # This would typically involve Android APIs through PyJNIus
            
            logger.info("Android resource limiting not yet implemented")
            return True
        except Exception as e:
            logger.error(f"Android resource limiting failed: {e}")
            return False
            
    def get_allocation_history(self) -> List[Dict[str, Any]]:
        """Get history of resource allocations"""
        # This would be implemented with a circular buffer
        # For now, return empty list
        return []
        
    def predict_resource_needs(self, profile_history: List[DeviceProfile],
                             upcoming_tasks: List[Dict]) -> ResourceAllocation:
        """Predict future resource needs based on history and planned tasks"""
        
        if not profile_history:
            # No history - use current profile with balanced settings
            return ResourceAllocation(
                cpu_limit_percent=75,
                memory_limit_mb=2048,
                network_limit_kbps=None,
                gpu_enabled=True,
                background_processing=True,
                priority_level="medium",
                thermal_mitigation=False,
                battery_conservation=False,
                reason="prediction_no_history",
                constraints={}
            )
            
        # Analyze trends in the history
        recent_profiles = profile_history[-10:]  # Last 10 profiles
        
        # Average resource usage
        avg_cpu = sum(p.cpu_percent for p in recent_profiles) / len(recent_profiles)
        avg_memory_usage = sum(p.ram_used_mb for p in recent_profiles) / len(recent_profiles)
        
        # Predict based on trends and upcoming tasks
        predicted_cpu = min(100, avg_cpu * 1.2)  # 20% buffer
        predicted_memory = int(avg_memory_usage * 1.3)  # 30% buffer
        
        return ResourceAllocation(
            cpu_limit_percent=predicted_cpu,
            memory_limit_mb=predicted_memory,
            network_limit_kbps=None,
            gpu_enabled=True,
            background_processing=True,
            priority_level="medium",
            thermal_mitigation=False,
            battery_conservation=False,
            reason="prediction_based",
            constraints={"prediction_confidence": 0.7}
        )

