"""Enhanced Device Profiler for Sprint 6 Resource Management"""

import asyncio
import json
import logging
import platform
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Tuple
import threading
import queue

# System monitoring
import psutil

logger = logging.getLogger(__name__)

class DeviceType(Enum):
    """Device type classification for resource management"""
    PHONE = "phone"
    TABLET = "tablet"
    LAPTOP = "laptop"
    DESKTOP = "desktop"
    SERVER = "server"
    EMBEDDED = "embedded"
    UNKNOWN = "unknown"

class PowerState(Enum):
    """Device power state for evolution scheduling"""
    PLUGGED_IN = "plugged_in"
    BATTERY_HIGH = "battery_high"      # > 80%
    BATTERY_MEDIUM = "battery_medium"  # 20-80%  
    BATTERY_LOW = "battery_low"        # 5-20%
    BATTERY_CRITICAL = "battery_critical"  # < 5%
    UNKNOWN = "unknown"

class ThermalState(Enum):
    """Device thermal state for performance management"""
    NORMAL = "normal"           # < 60°C
    WARM = "warm"              # 60-75°C
    HOT = "hot"                # 75-85°C
    CRITICAL = "critical"      # > 85°C
    THROTTLING = "throttling"  # CPU throttling detected
    UNKNOWN = "unknown"

@dataclass
class ResourceSnapshot:
    """Real-time resource usage snapshot"""
    timestamp: float
    
    # Memory (bytes)
    memory_total: int
    memory_available: int
    memory_used: int
    memory_percent: float
    
    # CPU
    cpu_percent: float
    cpu_cores: int
    cpu_freq_current: Optional[float] = None
    cpu_freq_max: Optional[float] = None
    cpu_temp: Optional[float] = None
    
    # Storage (bytes)
    storage_total: int = 0
    storage_used: int = 0
    storage_free: int = 0
    storage_percent: float = 0.0
    
    # Power
    battery_percent: Optional[float] = None
    power_plugged: Optional[bool] = None
    power_state: PowerState = PowerState.UNKNOWN
    
    # Thermal
    thermal_state: ThermalState = ThermalState.UNKNOWN
    
    # Network
    network_sent: int = 0
    network_received: int = 0
    network_connections: int = 0
    
    # Process metrics
    process_count: int = 0
    
    # GPU (if available)
    gpu_memory_used: Optional[int] = None
    gpu_memory_total: Optional[int] = None
    gpu_utilization: Optional[float] = None
    
    @property
    def memory_usage_gb(self) -> float:
        """Memory usage in GB"""
        return self.memory_used / (1024**3)
        
    @property
    def memory_available_gb(self) -> float:
        """Available memory in GB"""
        return self.memory_available / (1024**3)
        
    @property
    def storage_used_gb(self) -> float:
        """Storage used in GB"""
        return self.storage_used / (1024**3)

    @property
    def is_resource_constrained(self) -> bool:
        """Check if device is resource constrained"""
        return (
            self.memory_percent > 85 or
            self.cpu_percent > 90 or
            self.storage_percent > 90 or
            self.power_state in [PowerState.BATTERY_LOW, PowerState.BATTERY_CRITICAL] or
            self.thermal_state in [ThermalState.HOT, ThermalState.CRITICAL, ThermalState.THROTTLING]
        )
        
    @property
    def evolution_suitability_score(self) -> float:
        """Calculate suitability for evolution tasks (0-1, higher = better)"""
        score = 1.0
        
        # Memory penalty
        if self.memory_percent > 80:
            score -= (self.memory_percent - 80) / 20 * 0.3
            
        # CPU penalty
        if self.cpu_percent > 70:
            score -= (self.cpu_percent - 70) / 30 * 0.2
            
        # Power penalty
        if self.battery_percent is not None and not self.power_plugged:
            if self.battery_percent < 30:
                score -= (30 - self.battery_percent) / 30 * 0.3
                
        # Thermal penalty
        thermal_penalties = {
            ThermalState.WARM: 0.1,
            ThermalState.HOT: 0.3,
            ThermalState.CRITICAL: 0.7,
            ThermalState.THROTTLING: 0.9
        }
        score -= thermal_penalties.get(self.thermal_state, 0)
        
        return max(0.0, min(1.0, score))

@dataclass
class DeviceProfile:
    """Comprehensive device profile for resource planning"""
    device_id: str
    
    # Hardware classification
    device_type: DeviceType
    os_type: str
    os_version: str
    architecture: str
    
    # Core capabilities
    total_memory_gb: float
    cpu_cores: int
    cpu_model: str
    
    # Feature support
    supports_gpu: bool = False
    supports_bluetooth: bool = False
    supports_wifi: bool = False
    supports_cellular: bool = False
    
    # Performance characteristics
    benchmark_score: Optional[float] = None
    typical_performance: Optional[float] = None
    performance_tier: str = "medium"  # low, medium, high, premium
    
    # Evolution-specific constraints
    max_evolution_memory_mb: Optional[int] = None
    max_evolution_cpu_percent: Optional[float] = None
    evolution_capable: bool = True
    preferred_evolution_types: List[str] = field(default_factory=list)
    
    # Resource limits
    memory_warning_threshold: float = 85.0
    memory_critical_threshold: float = 95.0
    cpu_warning_threshold: float = 80.0
    cpu_critical_threshold: float = 95.0
    
    # Operational settings
    monitoring_interval: float = 10.0
    battery_optimization: bool = False
    thermal_throttling: bool = True
    
    # Metadata
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    
    def update_profile(self, **kwargs):
        """Update profile with new information"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.last_updated = time.time()
        
    def get_evolution_constraints(self) -> Dict[str, Any]:
        """Get constraints for evolution tasks"""
        return {
            'max_memory_mb': self.max_evolution_memory_mb or int(self.total_memory_gb * 1024 * 0.5),
            'max_cpu_percent': self.max_evolution_cpu_percent or 70.0,
            'evolution_capable': self.evolution_capable,
            'preferred_types': self.preferred_evolution_types,
            'device_tier': self.performance_tier,
            'thermal_throttling': self.thermal_throttling,
            'battery_optimization': self.battery_optimization
        }
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary"""
        return {
            "device_id": self.device_id,
            "device_type": self.device_type.value,
            "os_type": self.os_type,
            "os_version": self.os_version,
            "architecture": self.architecture,
            "total_memory_gb": self.total_memory_gb,
            "cpu_cores": self.cpu_cores,
            "cpu_model": self.cpu_model,
            "supports_gpu": self.supports_gpu,
            "supports_bluetooth": self.supports_bluetooth,
            "supports_wifi": self.supports_wifi,
            "supports_cellular": self.supports_cellular,
            "benchmark_score": self.benchmark_score,
            "typical_performance": self.typical_performance,
            "performance_tier": self.performance_tier,
            "max_evolution_memory_mb": self.max_evolution_memory_mb,
            "max_evolution_cpu_percent": self.max_evolution_cpu_percent,
            "evolution_capable": self.evolution_capable,
            "preferred_evolution_types": self.preferred_evolution_types,
            "memory_warning_threshold": self.memory_warning_threshold,
            "memory_critical_threshold": self.memory_critical_threshold,
            "cpu_warning_threshold": self.cpu_warning_threshold,
            "cpu_critical_threshold": self.cpu_critical_threshold,
            "monitoring_interval": self.monitoring_interval,
            "battery_optimization": self.battery_optimization,
            "thermal_throttling": self.thermal_throttling,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
        }

class DeviceProfiler:
    """Enhanced device profiler for mobile resource monitoring"""
    
    def __init__(
        self,
        device_id: Optional[str] = None,
        monitoring_interval: float = 5.0,
        history_size: int = 1000,
        enable_background_monitoring: bool = True,
    ):
        self.device_id = device_id or self._generate_device_id()
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        self.enable_background_monitoring = enable_background_monitoring
        
        # Device profile
        self.profile = self._create_device_profile()
        
        # Resource monitoring
        self.snapshots: List[ResourceSnapshot] = []
        self.current_snapshot: Optional[ResourceSnapshot] = None
        
        # Background monitoring
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.snapshot_queue = queue.Queue(maxsize=100)
        
        # Callbacks for resource events
        self.alert_callbacks: List[Callable[[str, ResourceSnapshot], None]] = []
        self.threshold_callbacks: Dict[str, List[Callable[[ResourceSnapshot], None]]] = {}
        
        # Configurable thresholds
        self.thresholds = {
            "memory_critical": self.profile.memory_critical_threshold,
            "memory_warning": self.profile.memory_warning_threshold,
            "cpu_critical": self.profile.cpu_critical_threshold,
            "cpu_warning": self.profile.cpu_warning_threshold,
            "battery_critical": 5.0,
            "battery_low": 15.0,
            "temperature_critical": 85.0,
            "temperature_warning": 75.0,
        }
        
        # Statistics
        self.stats = {
            "snapshots_taken": 0,
            "alerts_triggered": 0,
            "monitoring_uptime": 0.0,
            "last_alert_time": None,
            "evolution_tasks_run": 0,
            "resource_constraints_hit": 0
        }
        
        logger.info(f"Enhanced device profiler initialized for {self.profile.device_type.value} device")
        
    def _generate_device_id(self) -> str:
        """Generate unique device ID"""
        import hashlib
        
        try:
            # Create consistent device ID based on system characteristics
            mac = hex(uuid.getnode())[2:]
            hostname = platform.node()
            system = platform.system()
            device_string = f"{mac}-{hostname}-{system}"
            return hashlib.md5(device_string.encode()).hexdigest()[:16]
        except:
            return str(uuid.uuid4())[:16]
            
    def _create_device_profile(self) -> DeviceProfile:
        """Create comprehensive device profile"""
        # Detect device type
        device_type = self._detect_device_type()
        
        # Get system information
        memory = psutil.virtual_memory()
        cpu_info = self._get_cpu_info()
        
        # Determine performance tier
        performance_tier = self._calculate_performance_tier()
        
        # Set evolution constraints based on device type and performance
        max_evolution_memory = self._calculate_max_evolution_memory(memory.total, device_type)
        max_evolution_cpu = self._calculate_max_evolution_cpu(device_type)
        
        return DeviceProfile(
            device_id=self.device_id,
            device_type=device_type,
            os_type=platform.system(),
            os_version=platform.version(),
            architecture=platform.machine(),
            total_memory_gb=memory.total / (1024**3),
            cpu_cores=psutil.cpu_count(logical=True),
            cpu_model=cpu_info.get("model", "Unknown"),
            supports_gpu=self._detect_gpu_support(),
            supports_bluetooth=self._detect_bluetooth_support(),
            supports_wifi=self._detect_wifi_support(),
            supports_cellular=self._detect_cellular_support(device_type),
            performance_tier=performance_tier,
            max_evolution_memory_mb=max_evolution_memory,
            max_evolution_cpu_percent=max_evolution_cpu,
            evolution_capable=self._is_evolution_capable(device_type, memory.total),
            preferred_evolution_types=self._get_preferred_evolution_types(device_type, performance_tier),
            battery_optimization=device_type in [DeviceType.PHONE, DeviceType.TABLET],
            monitoring_interval=self.monitoring_interval,
        )
        
    def _detect_device_type(self) -> DeviceType:
        """Detect device type based on system characteristics"""
        system = platform.system().lower()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        cpu_cores = psutil.cpu_count(logical=True)
        
        # Platform-specific detection
        if system == "android":
            return DeviceType.PHONE if memory_gb < 6 else DeviceType.TABLET
        elif system == "darwin":
            if platform.machine().startswith("iP"):
                return DeviceType.PHONE if "iPhone" in platform.machine() else DeviceType.TABLET
            else:
                return DeviceType.LAPTOP if memory_gb < 16 else DeviceType.DESKTOP
        elif system in ["linux", "windows"]:
            if memory_gb < 2:
                return DeviceType.EMBEDDED
            elif memory_gb < 8 or cpu_cores < 4:
                return DeviceType.LAPTOP
            elif memory_gb >= 32 and cpu_cores >= 8:
                return DeviceType.SERVER
            else:
                return DeviceType.DESKTOP
        else:
            return DeviceType.UNKNOWN
            
    def _calculate_performance_tier(self) -> str:
        """Calculate device performance tier"""
        memory_gb = psutil.virtual_memory().total / (1024**3)
        cpu_cores = psutil.cpu_count(logical=True)
        
        # Score based on memory and CPU
        memory_score = min(memory_gb / 32, 1.0)  # Normalize to 32GB
        cpu_score = min(cpu_cores / 16, 1.0)     # Normalize to 16 cores
        
        combined_score = (memory_score + cpu_score) / 2
        
        if combined_score >= 0.8:
            return "premium"
        elif combined_score >= 0.6:
            return "high"
        elif combined_score >= 0.3:
            return "medium"
        else:
            return "low"
            
    def _calculate_max_evolution_memory(self, total_memory: int, device_type: DeviceType) -> int:
        """Calculate maximum memory for evolution tasks"""
        total_gb = total_memory / (1024**3)
        
        # Conservative memory allocation based on device type
        allocation_ratios = {
            DeviceType.PHONE: 0.3,      # 30% of memory
            DeviceType.TABLET: 0.4,     # 40% of memory
            DeviceType.LAPTOP: 0.5,     # 50% of memory
            DeviceType.DESKTOP: 0.6,    # 60% of memory
            DeviceType.SERVER: 0.7,     # 70% of memory
            DeviceType.EMBEDDED: 0.2,   # 20% of memory
        }
        
        ratio = allocation_ratios.get(device_type, 0.4)
        max_memory_gb = total_gb * ratio
        
        return int(max_memory_gb * 1024)  # Convert to MB
        
    def _calculate_max_evolution_cpu(self, device_type: DeviceType) -> float:
        """Calculate maximum CPU usage for evolution tasks"""
        # Conservative CPU allocation
        cpu_limits = {
            DeviceType.PHONE: 60.0,     # 60% CPU
            DeviceType.TABLET: 70.0,    # 70% CPU
            DeviceType.LAPTOP: 75.0,    # 75% CPU
            DeviceType.DESKTOP: 80.0,   # 80% CPU
            DeviceType.SERVER: 90.0,    # 90% CPU
            DeviceType.EMBEDDED: 50.0,  # 50% CPU
        }
        
        return cpu_limits.get(device_type, 70.0)
        
    def _is_evolution_capable(self, device_type: DeviceType, total_memory: int) -> bool:
        """Determine if device is capable of evolution tasks"""
        memory_gb = total_memory / (1024**3)
        
        # Minimum requirements for evolution
        if device_type == DeviceType.EMBEDDED and memory_gb < 1:
            return False
        elif device_type in [DeviceType.PHONE, DeviceType.TABLET] and memory_gb < 2:
            return False
        elif memory_gb < 4:  # Minimum 4GB for other devices
            return False
            
        return True
        
    def _get_preferred_evolution_types(self, device_type: DeviceType, performance_tier: str) -> List[str]:
        """Get preferred evolution types for this device"""
        preferences = []
        
        # Base preferences by device type
        if device_type in [DeviceType.PHONE, DeviceType.TABLET]:
            preferences.extend(["nightly", "lightweight"])
            if performance_tier in ["high", "premium"]:
                preferences.append("breakthrough")
        elif device_type in [DeviceType.LAPTOP, DeviceType.DESKTOP]:
            preferences.extend(["nightly", "breakthrough"])
            if performance_tier in ["high", "premium"]:
                preferences.append("experimental")
        elif device_type == DeviceType.SERVER:
            preferences.extend(["nightly", "breakthrough", "experimental", "distributed"])
            
        return preferences
        
    def _get_cpu_info(self) -> Dict[str, Any]:
        """Get CPU information"""
        try:
            if platform.system() == "Windows":
                import subprocess
                result = subprocess.run(
                    ["wmic", "cpu", "get", "name"],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if len(lines) > 1:
                        return {"model": lines[1].strip()}
            elif platform.system() in ["Linux", "Darwin"]:
                try:
                    with open("/proc/cpuinfo", "r") as f:
                        for line in f:
                            if line.startswith("model name"):
                                return {"model": line.split(":")[1].strip()}
                except:
                    pass
        except:
            pass
            
        return {"model": f"{platform.processor()} ({psutil.cpu_count()} cores)"}
        
    def _detect_gpu_support(self) -> bool:
        """Detect if GPU is available"""
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "-L"],
                capture_output=True, text=True
            )
            return result.returncode == 0
        except:
            return False
            
    def _detect_bluetooth_support(self) -> bool:
        """Detect Bluetooth support"""
        system = platform.system().lower()
        return system in ["android", "darwin", "linux"]
        
    def _detect_wifi_support(self) -> bool:
        """Detect WiFi support"""
        return True  # Most modern devices have WiFi
        
    def _detect_cellular_support(self, device_type: DeviceType) -> bool:
        """Detect cellular support"""
        return device_type in [DeviceType.PHONE, DeviceType.TABLET]
        
    def take_snapshot(self) -> ResourceSnapshot:
        """Take enhanced resource snapshot"""
        timestamp = time.time()
        
        # Memory information
        memory = psutil.virtual_memory()
        
        # CPU information
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_freq = None
        cpu_temp = None
        
        try:
            freq = psutil.cpu_freq()
            if freq:
                cpu_freq = freq.current
        except:
            pass
            
        # Get CPU temperature
        cpu_temp = self._get_cpu_temperature()
        
        # Storage information
        storage_total = storage_used = storage_free = 0
        storage_percent = 0.0
        
        try:
            disk = psutil.disk_usage('/')
            storage_total = disk.total
            storage_used = disk.used
            storage_free = disk.free
            storage_percent = (disk.used / disk.total) * 100
        except:
            pass
            
        # Battery and power information
        battery_percent, power_plugged, power_state = self._get_power_info()
        
        # Thermal state
        thermal_state = self._determine_thermal_state(cpu_temp, cpu_percent)
        
        # Network information
        network_stats = psutil.net_io_counters()
        network_connections = len(psutil.net_connections())
        
        # Process count
        process_count = len(psutil.pids())
        
        # GPU information (if available)
        gpu_memory_used, gpu_memory_total, gpu_utilization = self._get_gpu_info()
        
        snapshot = ResourceSnapshot(
            timestamp=timestamp,
            memory_total=memory.total,
            memory_available=memory.available,
            memory_used=memory.used,
            memory_percent=memory.percent,
            cpu_percent=cpu_percent,
            cpu_cores=psutil.cpu_count(),
            cpu_freq_current=cpu_freq,
            cpu_temp=cpu_temp,
            storage_total=storage_total,
            storage_used=storage_used,
            storage_free=storage_free,
            storage_percent=storage_percent,
            battery_percent=battery_percent,
            power_plugged=power_plugged,
            power_state=power_state,
            thermal_state=thermal_state,
            network_sent=network_stats.bytes_sent,
            network_received=network_stats.bytes_recv,
            network_connections=network_connections,
            process_count=process_count,
            gpu_memory_used=gpu_memory_used,
            gpu_memory_total=gpu_memory_total,
            gpu_utilization=gpu_utilization,
        )
        
        # Store snapshot
        self.current_snapshot = snapshot
        self.snapshots.append(snapshot)
        
        # Maintain history size
        if len(self.snapshots) > self.history_size:
            self.snapshots = self.snapshots[-self.history_size:]
            
        # Update stats
        self.stats["snapshots_taken"] += 1
        
        # Check thresholds and trigger alerts
        self._check_thresholds(snapshot)
        
        # Queue for background processing
        try:
            self.snapshot_queue.put_nowait(snapshot)
        except queue.Full:
            pass
            
        return snapshot
        
    def _get_cpu_temperature(self) -> Optional[float]:
        """Get CPU temperature"""
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                for sensor_name, sensor_list in temps.items():
                    if any(keyword in sensor_name.lower() for keyword in ["cpu", "core", "processor"]):
                        if sensor_list:
                            return sensor_list[0].current
        except:
            pass
        return None
        
    def _get_power_info(self) -> Tuple[Optional[float], Optional[bool], PowerState]:
        """Get battery and power information"""
        battery_percent = None
        power_plugged = None
        power_state = PowerState.UNKNOWN
        
        try:
            battery = psutil.sensors_battery()
            if battery:
                battery_percent = battery.percent
                power_plugged = battery.power_plugged
                
                if power_plugged:
                    power_state = PowerState.PLUGGED_IN
                elif battery_percent > 80:
                    power_state = PowerState.BATTERY_HIGH
                elif battery_percent > 20:
                    power_state = PowerState.BATTERY_MEDIUM
                elif battery_percent > 5:
                    power_state = PowerState.BATTERY_LOW
                else:
                    power_state = PowerState.BATTERY_CRITICAL
        except:
            pass
            
        return battery_percent, power_plugged, power_state
        
    def _determine_thermal_state(self, cpu_temp: Optional[float], cpu_percent: float) -> ThermalState:
        """Determine thermal state"""
        if cpu_temp is None:
            # Estimate based on CPU usage
            if cpu_percent > 95:
                return ThermalState.HOT
            elif cpu_percent > 80:
                return ThermalState.WARM
            else:
                return ThermalState.NORMAL
        else:
            if cpu_temp > 85:
                return ThermalState.CRITICAL
            elif cpu_temp > 75:
                return ThermalState.HOT
            elif cpu_temp > 60:
                return ThermalState.WARM
            else:
                return ThermalState.NORMAL
                
    def _get_gpu_info(self) -> Tuple[Optional[int], Optional[int], Optional[float]]:
        """Get GPU information if available"""
        try:
            # This would use nvidia-ml-py or similar for NVIDIA GPUs
            # For now, return None values
            return None, None, None
        except:
            return None, None, None
            
    def _check_thresholds(self, snapshot: ResourceSnapshot):
        """Check resource thresholds and trigger alerts"""
        alerts_triggered = []
        
        # Memory checks
        if snapshot.memory_percent > self.thresholds["memory_critical"]:
            alerts_triggered.append("memory_critical")
        elif snapshot.memory_percent > self.thresholds["memory_warning"]:
            alerts_triggered.append("memory_warning")
            
        # CPU checks  
        if snapshot.cpu_percent > self.thresholds["cpu_critical"]:
            alerts_triggered.append("cpu_critical")
        elif snapshot.cpu_percent > self.thresholds["cpu_warning"]:
            alerts_triggered.append("cpu_warning")
            
        # Battery checks
        if snapshot.battery_percent is not None:
            if snapshot.battery_percent < self.thresholds["battery_critical"]:
                alerts_triggered.append("battery_critical")
            elif snapshot.battery_percent < self.thresholds["battery_low"]:
                alerts_triggered.append("battery_low")
                
        # Temperature checks
        if snapshot.cpu_temp is not None:
            if snapshot.cpu_temp > self.thresholds["temperature_critical"]:
                alerts_triggered.append("temperature_critical")
            elif snapshot.cpu_temp > self.thresholds["temperature_warning"]:
                alerts_triggered.append("temperature_warning")
                
        # Thermal state checks
        if snapshot.thermal_state in [ThermalState.CRITICAL, ThermalState.THROTTLING]:
            alerts_triggered.append("thermal_critical")
            
        # Resource constraint tracking
        if snapshot.is_resource_constrained:
            self.stats["resource_constraints_hit"] += 1
            
        # Trigger alerts
        for alert_type in alerts_triggered:
            self._trigger_alert(alert_type, snapshot)
            
    def _trigger_alert(self, alert_type: str, snapshot: ResourceSnapshot):
        """Trigger alert for resource threshold violation"""
        self.stats["alerts_triggered"] += 1
        self.stats["last_alert_time"] = time.time()
        
        logger.warning(f"Resource alert: {alert_type} - {self._format_alert_message(alert_type, snapshot)}")
        
        # Call registered callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_type, snapshot)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
                
        # Call threshold-specific callbacks
        for callback in self.threshold_callbacks.get(alert_type, []):
            try:
                callback(snapshot)
            except Exception as e:
                logger.error(f"Error in threshold callback: {e}")
                
    def _format_alert_message(self, alert_type: str, snapshot: ResourceSnapshot) -> str:
        """Format alert message"""
        if "memory" in alert_type:
            return f"Memory usage: {snapshot.memory_percent:.1f}% ({snapshot.memory_usage_gb:.1f}GB used)"
        elif "cpu" in alert_type:
            return f"CPU usage: {snapshot.cpu_percent:.1f}%"
        elif "battery" in alert_type:
            return f"Battery level: {snapshot.battery_percent:.1f}%"
        elif "temperature" in alert_type or "thermal" in alert_type:
            return f"CPU temperature: {snapshot.cpu_temp:.1f}°C, thermal state: {snapshot.thermal_state.value}"
        else:
            return f"Resource threshold exceeded: {alert_type}"
            
    def start_monitoring(self):
        """Start background monitoring"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
            
        self.monitoring_active = True
        
        if self.enable_background_monitoring:
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True
            )
            self.monitoring_thread.start()
            
        logger.info("Enhanced device monitoring started")
        
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
            
        logger.info("Device monitoring stopped")
        
    def _monitoring_loop(self):
        """Background monitoring loop"""
        start_time = time.time()
        
        while self.monitoring_active:
            try:
                self.take_snapshot()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
                
        self.stats["monitoring_uptime"] = time.time() - start_time
        
    def is_suitable_for_evolution(self, evolution_type: str = "nightly") -> bool:
        """Check if device is currently suitable for evolution"""
        if not self.current_snapshot:
            return False
            
        if not self.profile.evolution_capable:
            return False
            
        # Check if evolution type is preferred
        if evolution_type not in self.profile.preferred_evolution_types and evolution_type != "emergency":
            return False
            
        # Get evolution suitability score
        suitability = self.current_snapshot.evolution_suitability_score
        
        # Different thresholds for different evolution types
        thresholds = {
            "nightly": 0.6,
            "breakthrough": 0.7,
            "experimental": 0.8,
            "emergency": 0.3
        }
        
        return suitability >= thresholds.get(evolution_type, 0.6)
        
    def get_evolution_resource_allocation(self) -> Dict[str, Any]:
        """Get recommended resource allocation for evolution"""
        if not self.current_snapshot:
            return {}
            
        constraints = self.profile.get_evolution_constraints()
        current_usage = self.current_snapshot
        
        # Calculate available resources
        available_memory_mb = int(current_usage.memory_available / (1024 * 1024))
        available_cpu_percent = max(0, 100 - current_usage.cpu_percent)
        
        # Apply constraints
        allocated_memory = min(
            available_memory_mb,
            constraints['max_memory_mb']
        )
        
        allocated_cpu = min(
            available_cpu_percent,
            constraints['max_cpu_percent']
        )
        
        return {
            'memory_mb': allocated_memory,
            'cpu_percent': allocated_cpu,
            'device_tier': constraints['device_tier'],
            'evolution_capable': constraints['evolution_capable'],
            'thermal_throttling': constraints['thermal_throttling'],
            'battery_optimization': constraints['battery_optimization'],
            'suitability_score': current_usage.evolution_suitability_score,
            'current_constraints': current_usage.is_resource_constrained
        }
        
    def register_alert_callback(self, callback: Callable[[str, ResourceSnapshot], None]):
        """Register callback for all alerts"""
        self.alert_callbacks.append(callback)
        
    def register_threshold_callback(self, threshold: str, callback: Callable[[ResourceSnapshot], None]):
        """Register callback for specific threshold"""
        if threshold not in self.threshold_callbacks:
            self.threshold_callbacks[threshold] = []
        self.threshold_callbacks[threshold].append(callback)
        
    def get_current_status(self) -> Dict[str, Any]:
        """Get comprehensive current device status"""
        current = self.current_snapshot
        
        if not current:
            return {"status": "no_data"}
            
        return {
            "device_id": self.device_id,
            "device_type": self.profile.device_type.value,
            "performance_tier": self.profile.performance_tier,
            "timestamp": current.timestamp,
            "resource_constrained": current.is_resource_constrained,
            "evolution_suitability": current.evolution_suitability_score,
            "evolution_capable": self.profile.evolution_capable,
            "memory": {
                "used_gb": current.memory_usage_gb,
                "available_gb": current.memory_available_gb,
                "percent": current.memory_percent,
                "total_gb": self.profile.total_memory_gb
            },
            "cpu": {
                "percent": current.cpu_percent,
                "cores": current.cpu_cores,
                "temperature": current.cpu_temp,
                "model": self.profile.cpu_model
            },
            "power": {
                "battery_percent": current.battery_percent,
                "plugged_in": current.power_plugged,
                "state": current.power_state.value,
            },
            "thermal": {
                "state": current.thermal_state.value,
                "temperature": current.cpu_temp
            },
            "storage": {
                "used_gb": current.storage_used_gb,
                "percent": current.storage_percent,
            },
            "evolution": self.get_evolution_resource_allocation()
        }
        
    def export_profile(self, include_history: bool = False) -> Dict[str, Any]:
        """Export device profile and optionally history"""
        data = {
            "profile": self.profile.to_dict(),
            "current_status": self.get_current_status(),
            "statistics": self.stats.copy(),
            "thresholds": self.thresholds.copy(),
        }
        
        if include_history:
            data["history"] = [
                {
                    "timestamp": s.timestamp,
                    "memory_percent": s.memory_percent,
                    "cpu_percent": s.cpu_percent,
                    "battery_percent": s.battery_percent,
                    "power_state": s.power_state.value,
                    "thermal_state": s.thermal_state.value,
                    "evolution_suitability": s.evolution_suitability_score,
                }
                for s in self.snapshots[-100:]  # Last 100 snapshots
            ]
            
        return data