# Fog Edge Beacon Capability Discovery System
# Production-ready capability discovery for edge devices

import asyncio
import json
import logging
import socket
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable, Tuple
from collections import defaultdict
import uuid


logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Types of edge devices."""
    MOBILE = "mobile"
    LAPTOP = "laptop"
    DESKTOP = "desktop"
    SERVER = "server"
    IOT_SENSOR = "iot_sensor"
    IOT_GATEWAY = "iot_gateway"
    RASPBERRY_PI = "raspberry_pi"
    EDGE_AI = "edge_ai"
    DRONE = "drone"
    VEHICLE = "vehicle"
    INDUSTRIAL = "industrial"
    UNKNOWN = "unknown"


class PowerProfile(Enum):
    """Power consumption profiles for edge devices."""
    ULTRA_LOW = "ultra_low"   # < 1W
    LOW = "low"               # 1-10W
    MODERATE = "moderate"     # 10-50W
    HIGH = "high"             # 50-200W
    VERY_HIGH = "very_high"   # > 200W
    BATTERY = "battery"       # Battery-powered
    UNLIMITED = "unlimited"   # Unlimited power


class ConnectionType(Enum):
    """Network connection types."""
    WIFI = "wifi"
    ETHERNET = "ethernet"
    CELLULAR_4G = "cellular_4g"
    CELLULAR_5G = "cellular_5g"
    BLUETOOTH = "bluetooth"
    ZIGBEE = "zigbee"
    LORA = "lora"
    SATELLITE = "satellite"
    MIXED = "mixed"


@dataclass
class HardwareCapabilities:
    """Hardware capabilities of an edge device."""
    
    cpu_cores: int = 1
    cpu_architecture: str = "unknown"
    cpu_frequency_mhz: int = 1000
    memory_mb: int = 1024
    storage_gb: int = 32
    gpu_available: bool = False
    gpu_memory_mb: int = 0
    tpu_available: bool = False
    accelerators: List[str] = field(default_factory=list)
    sensors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cpu_cores": self.cpu_cores,
            "cpu_architecture": self.cpu_architecture,
            "cpu_frequency_mhz": self.cpu_frequency_mhz,
            "memory_mb": self.memory_mb,
            "storage_gb": self.storage_gb,
            "gpu_available": self.gpu_available,
            "gpu_memory_mb": self.gpu_memory_mb,
            "tpu_available": self.tpu_available,
            "accelerators": self.accelerators,
            "sensors": self.sensors
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HardwareCapabilities':
        return cls(
            cpu_cores=data.get("cpu_cores", 1),
            cpu_architecture=data.get("cpu_architecture", "unknown"),
            cpu_frequency_mhz=data.get("cpu_frequency_mhz", 1000),
            memory_mb=data.get("memory_mb", 1024),
            storage_gb=data.get("storage_gb", 32),
            gpu_available=data.get("gpu_available", False),
            gpu_memory_mb=data.get("gpu_memory_mb", 0),
            tpu_available=data.get("tpu_available", False),
            accelerators=data.get("accelerators", []),
            sensors=data.get("sensors", [])
        )


@dataclass
class EdgeDevice:
    """Complete edge device information."""
    
    device_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    device_name: str = "Unknown Device"
    device_type: DeviceType = DeviceType.UNKNOWN
    power_profile: PowerProfile = PowerProfile.MODERATE
    location: Optional[str] = None
    coordinates: Optional[Tuple[float, float]] = None
    owner_id: Optional[str] = None
    hardware: HardwareCapabilities = field(default_factory=HardwareCapabilities)
    last_seen: float = field(default_factory=time.time)
    first_seen: float = field(default_factory=time.time)
    trust_score: float = 5.0
    reputation_score: float = 5.0
    
    def is_online(self, timeout_seconds: float = 300) -> bool:
        return (time.time() - self.last_seen) < timeout_seconds
    
    def calculate_capability_score(self) -> float:
        hw_score = min(self.hardware.cpu_cores / 8, 1.0) * 0.5 + \
                   min(self.hardware.memory_mb / 8192, 1.0) * 0.5
        return hw_score
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "device_name": self.device_name,
            "device_type": self.device_type.value,
            "power_profile": self.power_profile.value,
            "location": self.location,
            "coordinates": self.coordinates,
            "owner_id": self.owner_id,
            "hardware": self.hardware.to_dict(),
            "last_seen": self.last_seen,
            "first_seen": self.first_seen,
            "trust_score": self.trust_score,
            "reputation_score": self.reputation_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EdgeDevice':
        return cls(
            device_id=data.get("device_id", str(uuid.uuid4())),
            device_name=data.get("device_name", "Unknown Device"),
            device_type=DeviceType(data.get("device_type", DeviceType.UNKNOWN.value)),
            power_profile=PowerProfile(data.get("power_profile", PowerProfile.MODERATE.value)),
            location=data.get("location"),
            coordinates=data.get("coordinates"),
            owner_id=data.get("owner_id"),
            hardware=HardwareCapabilities.from_dict(data.get("hardware", {})),
            last_seen=data.get("last_seen", time.time()),
            first_seen=data.get("first_seen", time.time()),
            trust_score=data.get("trust_score", 5.0),
            reputation_score=data.get("reputation_score", 5.0)
        )


class CapabilityBeacon:
    """Edge device capability discovery beacon."""
    
    BROADCAST_PORT = 8765
    DISCOVERY_INTERVAL = 30.0
    
    def __init__(self, device: EdgeDevice, discovery_interval: float = None):
        self.device = device
        self.discovery_interval = discovery_interval or self.DISCOVERY_INTERVAL
        self.discovered_devices: Dict[str, EdgeDevice] = {}
        self.discovery_callbacks: List[Callable] = []
        self.is_running = False
        self.enable_broadcast = True
        self.broadcast_socket: Optional[socket.socket] = None
        
        logger.info(f"Initialized capability beacon for device {device.device_name}")
    
    async def start(self, bind_address: str = "0.0.0.0") -> bool:
        try:
            self.is_running = True
            
            if self.enable_broadcast:
                self.broadcast_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.broadcast_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                self.broadcast_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.broadcast_socket.bind((bind_address, self.BROADCAST_PORT))
                self.broadcast_socket.setblocking(False)
            
            asyncio.create_task(self._discovery_broadcast_loop())
            asyncio.create_task(self._discovery_listen_loop())
            
            logger.info(f"Started capability beacon on {bind_address}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start capability beacon: {e}")
            await self.stop()
            return False
    
    async def stop(self):
        self.is_running = False
        
        if self.broadcast_socket:
            self.broadcast_socket.close()
            self.broadcast_socket = None
        
        logger.info("Stopped capability beacon")
    
    def add_discovery_callback(self, callback: Callable[[EdgeDevice], None]):
        self.discovery_callbacks.append(callback)
    
    def get_discovered_devices(self, device_type: DeviceType = None, online_only: bool = True) -> List[EdgeDevice]:
        devices = []
        for device in self.discovered_devices.values():
            if online_only and not device.is_online():
                continue
            if device_type and device.device_type != device_type:
                continue
            devices.append(device)
        
        return sorted(devices, key=lambda d: d.calculate_capability_score(), reverse=True)
    
    async def _discovery_broadcast_loop(self):
        while self.is_running and self.enable_broadcast:
            try:
                await asyncio.sleep(self.discovery_interval)
                
                discovery_message = {
                    "type": "discovery_announce",
                    "device": self.device.to_dict(),
                    "timestamp": time.time()
                }
                
                message_data = json.dumps(discovery_message).encode('utf-8')
                
                if self.broadcast_socket:
                    try:
                        self.broadcast_socket.sendto(message_data, ('<broadcast>', self.BROADCAST_PORT))
                        logger.debug(f"Broadcasted discovery announcement")
                    except Exception as e:
                        logger.warning(f"Failed to broadcast discovery: {e}")
                
            except Exception as e:
                logger.error(f"Error in discovery broadcast loop: {e}")
    
    async def _discovery_listen_loop(self):
        while self.is_running and self.enable_broadcast:
            try:
                if not self.broadcast_socket:
                    await asyncio.sleep(1)
                    continue
                
                try:
                    data, addr = self.broadcast_socket.recvfrom(4096)
                    await self._handle_discovery_message(data, addr)
                except socket.error:
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Error in discovery listen loop: {e}")
    
    async def _handle_discovery_message(self, data: bytes, addr: tuple):
        try:
            message = json.loads(data.decode('utf-8'))
            message_type = message.get("type")
            
            if message_type == "discovery_announce":
                device_data = message.get("device")
                if device_data:
                    device = EdgeDevice.from_dict(device_data)
                    
                    if device.device_id == self.device.device_id:
                        return
                    
                    was_new = device.device_id not in self.discovered_devices
                    self.discovered_devices[device.device_id] = device
                    
                    if was_new:
                        logger.info(f"Discovered new device: {device.device_name}")
                        
                        for callback in self.discovery_callbacks:
                            try:
                                callback(device)
                            except Exception as e:
                                logger.error(f"Error in discovery callback: {e}")
            
        except Exception as e:
            logger.error(f"Error handling discovery message: {e}")
    
    def get_network_statistics(self) -> Dict[str, Any]:
        online_devices = sum(1 for device in self.discovered_devices.values() if device.is_online())
        
        device_types = defaultdict(int)
        for device in self.discovered_devices.values():
            if device.is_online():
                device_types[device.device_type.value] += 1
        
        return {
            "total_discovered_devices": len(self.discovered_devices),
            "online_devices": online_devices,
            "device_types": dict(device_types),
            "discovery_interval": self.discovery_interval,
            "broadcast_enabled": self.enable_broadcast
        }


# Backward compatibility - try to import from actual infrastructure locations first
try:
    from infrastructure.fog.edge.beacon import *
except ImportError:
    # Use the implementations defined above
    pass
