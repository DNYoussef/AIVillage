"""Dynamic resource allocation for mobile devices."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
import threading
import heapq
from collections import defaultdict, deque

from .device_profiler import DeviceProfiler, ResourceSnapshot, PowerState, ThermalState

logger = logging.getLogger(__name__)


class AllocationStrategy(Enum):
    """Resource allocation strategies."""
    GREEDY = "greedy"                    # Allocate to highest priority first
    FAIR_SHARE = "fair_share"           # Equal allocation across tasks
    PERFORMANCE_BASED = "performance_based"  # Allocate based on device capability
    BATTERY_AWARE = "battery_aware"     # Optimize for battery life
    THERMAL_AWARE = "thermal_aware"     # Consider thermal constraints
    ADAPTIVE = "adaptive"               # Adapt strategy based on conditions


class ResourceType(Enum):
    """Types of resources that can be allocated."""
    MEMORY = "memory"
    CPU = "cpu"
    GPU = "gpu"
    STORAGE = "storage"
    NETWORK = "network"
    BATTERY = "battery"


class Priority(Enum):
    """Task priority levels."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class ResourceRequest:
    """Request for resource allocation."""
    request_id: str
    task_id: str
    priority: Priority
    
    # Resource requirements
    memory_mb: Optional[int] = None
    cpu_percent: Optional[float] = None
    gpu_memory_mb: Optional[int] = None
    storage_mb: Optional[int] = None
    network_kbps: Optional[float] = None
    
    # Constraints
    max_duration_seconds: Optional[int] = None
    battery_impact: float = 1.0  # 0.0 (no impact) to 1.0 (high impact)
    thermal_impact: float = 1.0  # 0.0 (no impact) to 1.0 (high impact)
    
    # Flexibility
    memory_flexible: bool = False  # Can accept less memory
    cpu_flexible: bool = False     # Can accept less CPU
    can_be_preempted: bool = False # Can be paused/stopped
    can_be_delayed: bool = False   # Can wait for resources
    
    # Metadata
    created_at: float = field(default_factory=time.time)
    estimated_duration: Optional[int] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """Compare by priority for heap operations."""
        return self.priority.value < other.priority.value


@dataclass
class ResourceAllocation:
    """Allocated resources for a task."""
    allocation_id: str
    request_id: str
    task_id: str
    
    # Allocated amounts
    memory_mb: int = 0
    cpu_percent: float = 0.0
    gpu_memory_mb: int = 0
    storage_mb: int = 0
    network_kbps: float = 0.0
    
    # Allocation metadata
    allocated_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    strategy_used: Optional[AllocationStrategy] = None
    
    # Tracking
    active: bool = True
    preempted: bool = False
    
    def is_expired(self) -> bool:
        """Check if allocation has expired."""
        return self.expires_at is not None and time.time() > self.expires_at
        
    def get_resource_footprint(self) -> Dict[str, float]:
        """Get normalized resource footprint (0-1)."""
        return {
            "memory": self.memory_mb / 8192 if self.memory_mb > 0 else 0,  # Normalize to 8GB
            "cpu": self.cpu_percent / 100 if self.cpu_percent > 0 else 0,
            "gpu": self.gpu_memory_mb / 4096 if self.gpu_memory_mb > 0 else 0,
            "storage": self.storage_mb / 32768 if self.storage_mb > 0 else 0,  # 32GB
            "network": self.network_kbps / 10000 if self.network_kbps > 0 else 0,  # 10Mbps
        }


@dataclass 
class ResourcePool:
    """Pool of available resources."""
    # Total capacity
    total_memory_mb: int
    total_cpu_percent: float = 100.0
    total_gpu_memory_mb: int = 0
    total_storage_mb: int = 0
    total_network_kbps: float = 10000.0  # 10 Mbps default
    
    # Currently allocated
    allocated_memory_mb: int = 0
    allocated_cpu_percent: float = 0.0
    allocated_gpu_memory_mb: int = 0
    allocated_storage_mb: int = 0
    allocated_network_kbps: float = 0.0
    
    # Reserved (for system/critical tasks)
    reserved_memory_mb: int = 0
    reserved_cpu_percent: float = 20.0  # Keep 20% CPU for system
    reserved_gpu_memory_mb: int = 0
    reserved_storage_mb: int = 0
    reserved_network_kbps: float = 1000.0  # 1 Mbps for system
    
    def available_memory_mb(self) -> int:
        """Available memory for allocation."""
        return max(0, self.total_memory_mb - self.allocated_memory_mb - self.reserved_memory_mb)
        
    def available_cpu_percent(self) -> float:
        """Available CPU for allocation."""
        return max(0, self.total_cpu_percent - self.allocated_cpu_percent - self.reserved_cpu_percent)
        
    def available_gpu_memory_mb(self) -> int:
        """Available GPU memory for allocation."""
        return max(0, self.total_gpu_memory_mb - self.allocated_gpu_memory_mb - self.reserved_gpu_memory_mb)
        
    def available_storage_mb(self) -> int:
        """Available storage for allocation."""
        return max(0, self.total_storage_mb - self.allocated_storage_mb - self.reserved_storage_mb)
        
    def available_network_kbps(self) -> float:
        """Available network bandwidth for allocation."""
        return max(0, self.total_network_kbps - self.allocated_network_kbps - self.reserved_network_kbps)
        
    def can_allocate(self, request: ResourceRequest) -> bool:
        """Check if request can be satisfied."""
        if request.memory_mb and request.memory_mb > self.available_memory_mb():
            return False
        if request.cpu_percent and request.cpu_percent > self.available_cpu_percent():
            return False
        if request.gpu_memory_mb and request.gpu_memory_mb > self.available_gpu_memory_mb():
            return False
        if request.storage_mb and request.storage_mb > self.available_storage_mb():
            return False
        if request.network_kbps and request.network_kbps > self.available_network_kbps():
            return False
        return True
        
    def allocate_resources(self, allocation: ResourceAllocation) -> None:
        """Allocate resources from the pool."""
        self.allocated_memory_mb += allocation.memory_mb
        self.allocated_cpu_percent += allocation.cpu_percent
        self.allocated_gpu_memory_mb += allocation.gpu_memory_mb
        self.allocated_storage_mb += allocation.storage_mb
        self.allocated_network_kbps += allocation.network_kbps
        
    def deallocate_resources(self, allocation: ResourceAllocation) -> None:
        """Deallocate resources back to the pool."""
        self.allocated_memory_mb = max(0, self.allocated_memory_mb - allocation.memory_mb)
        self.allocated_cpu_percent = max(0, self.allocated_cpu_percent - allocation.cpu_percent)
        self.allocated_gpu_memory_mb = max(0, self.allocated_gpu_memory_mb - allocation.gpu_memory_mb)
        self.allocated_storage_mb = max(0, self.allocated_storage_mb - allocation.storage_mb)
        self.allocated_network_kbps = max(0, self.allocated_network_kbps - allocation.network_kbps)
        
    def get_utilization(self) -> Dict[str, float]:
        """Get current resource utilization percentages."""
        return {
            "memory": (self.allocated_memory_mb / max(1, self.total_memory_mb)) * 100,
            "cpu": (self.allocated_cpu_percent / max(1, self.total_cpu_percent)) * 100,
            "gpu": (self.allocated_gpu_memory_mb / max(1, self.total_gpu_memory_mb)) * 100 if self.total_gpu_memory_mb > 0 else 0,
            "storage": (self.allocated_storage_mb / max(1, self.total_storage_mb)) * 100 if self.total_storage_mb > 0 else 0,
            "network": (self.allocated_network_kbps / max(1, self.total_network_kbps)) * 100,
        }


class ResourceAllocator:
    """Dynamic resource allocator for mobile devices."""
    
    def __init__(
        self,
        device_profiler: DeviceProfiler,
        default_strategy: AllocationStrategy = AllocationStrategy.ADAPTIVE,
        enable_preemption: bool = True,
        enable_prediction: bool = True,
    ):
        self.device_profiler = device_profiler
        self.default_strategy = default_strategy
        self.enable_preemption = enable_preemption
        self.enable_prediction = enable_prediction
        
        # Initialize resource pool from device profile
        self.resource_pool = self._create_resource_pool()
        
        # Request management
        self.pending_requests: List[ResourceRequest] = []  # Priority heap
        self.active_allocations: Dict[str, ResourceAllocation] = {}
        self.allocation_history: deque = deque(maxlen=1000)
        
        # Strategy management
        self.current_strategy = default_strategy
        self.strategy_performance: Dict[AllocationStrategy, float] = {}
        
        # Monitoring
        self.allocation_stats = {
            "requests_received": 0,
            "requests_granted": 0,
            "requests_denied": 0,
            "preemptions_performed": 0,
            "strategy_changes": 0,
            "avg_allocation_time": 0.0,
        }
        
        # Background tasks
        self.allocator_active = False
        self.allocator_thread: Optional[threading.Thread] = None
        self.allocation_callbacks: List[Callable[[str, ResourceAllocation], None]] = []
        
        # Prediction models (simplified)
        self.resource_usage_history: Dict[str, deque] = {
            "memory": deque(maxlen=100),
            "cpu": deque(maxlen=100),
            "battery": deque(maxlen=100),
        }
        
        logger.info(f"Resource allocator initialized with {default_strategy.value} strategy")
        
    def _create_resource_pool(self) -> ResourcePool:
        """Create resource pool from device profile."""
        profile = self.device_profiler.profile
        current_snapshot = self.device_profiler.current_snapshot
        
        # Base on device profile
        total_memory_mb = int(profile.total_memory_gb * 1024)
        
        # Adjust based on current usage if available
        if current_snapshot:
            # Consider current system usage
            system_memory_mb = int(current_snapshot.memory_used / (1024 * 1024))
            available_memory_mb = int(current_snapshot.memory_available / (1024 * 1024))
            
            # Use available memory as total pool
            total_memory_mb = available_memory_mb + 1024  # Add some buffer
            
        return ResourcePool(
            total_memory_mb=total_memory_mb,
            total_cpu_percent=100.0,
            total_gpu_memory_mb=4096 if profile.supports_gpu else 0,
            total_storage_mb=32768,  # 32GB default
            total_network_kbps=10000.0,  # 10 Mbps
        )
        
    async def request_resources(self, request: ResourceRequest) -> Optional[str]:
        """Request resource allocation."""
        self.allocation_stats["requests_received"] += 1
        
        logger.debug(f"Resource request: {request.task_id} - {request.priority.name}")
        
        # Validate request
        if not self._validate_request(request):
            logger.warning(f"Invalid resource request: {request.request_id}")
            self.allocation_stats["requests_denied"] += 1
            return None
            
        # Add to pending queue (priority heap)
        heapq.heappush(self.pending_requests, request)
        
        # Try immediate allocation
        allocation = await self._try_allocate(request)
        
        if allocation:
            self._activate_allocation(allocation)
            self.allocation_stats["requests_granted"] += 1
            
            # Notify callbacks
            for callback in self.allocation_callbacks:
                try:
                    callback("allocated", allocation)
                except Exception as e:
                    logger.error(f"Error in allocation callback: {e}")
                    
            return allocation.allocation_id
        else:
            logger.info(f"Request {request.request_id} queued for later allocation")
            return None
            
    async def release_resources(self, allocation_id: str) -> bool:
        """Release allocated resources."""
        if allocation_id not in self.active_allocations:
            logger.warning(f"Allocation {allocation_id} not found")
            return False
            
        allocation = self.active_allocations[allocation_id]
        
        # Return resources to pool
        self.resource_pool.deallocate_resources(allocation)
        
        # Deactivate allocation
        allocation.active = False
        del self.active_allocations[allocation_id]
        
        # Add to history
        self.allocation_history.append(allocation)
        
        logger.debug(f"Released resources for allocation {allocation_id}")
        
        # Try to allocate pending requests
        await self._process_pending_requests()
        
        # Notify callbacks
        for callback in self.allocation_callbacks:
            try:
                callback("released", allocation)
            except Exception as e:
                logger.error(f"Error in release callback: {e}")
                
        return True
        
    async def _try_allocate(self, request: ResourceRequest) -> Optional[ResourceAllocation]:
        """Try to allocate resources for a request."""
        start_time = time.time()
        
        # Choose allocation strategy
        strategy = self._choose_strategy(request)
        
        # Try allocation based on strategy
        allocation = None
        
        if strategy == AllocationStrategy.GREEDY:
            allocation = self._allocate_greedy(request)
        elif strategy == AllocationStrategy.FAIR_SHARE:
            allocation = self._allocate_fair_share(request)
        elif strategy == AllocationStrategy.PERFORMANCE_BASED:
            allocation = self._allocate_performance_based(request)
        elif strategy == AllocationStrategy.BATTERY_AWARE:
            allocation = self._allocate_battery_aware(request)
        elif strategy == AllocationStrategy.THERMAL_AWARE:
            allocation = self._allocate_thermal_aware(request)
        elif strategy == AllocationStrategy.ADAPTIVE:
            allocation = self._allocate_adaptive(request)
            
        if allocation:
            allocation.strategy_used = strategy
            allocation_time = time.time() - start_time
            
            # Update stats
            self.allocation_stats["avg_allocation_time"] = (
                self.allocation_stats["avg_allocation_time"] * 0.9 + allocation_time * 0.1
            )
            
        return allocation
        
    def _allocate_greedy(self, request: ResourceRequest) -> Optional[ResourceAllocation]:
        """Greedy allocation - give exactly what's requested if available."""
        if not self.resource_pool.can_allocate(request):
            # Try preemption if enabled
            if self.enable_preemption:
                if self._try_preemption(request):
                    return self._allocate_greedy(request)
            return None
            
        allocation = ResourceAllocation(
            allocation_id=f"alloc_{int(time.time() * 1000)}_{request.task_id}",
            request_id=request.request_id,
            task_id=request.task_id,
            memory_mb=request.memory_mb or 0,
            cpu_percent=request.cpu_percent or 0,
            gpu_memory_mb=request.gpu_memory_mb or 0,
            storage_mb=request.storage_mb or 0,
            network_kbps=request.network_kbps or 0,
        )
        
        # Set expiration if specified
        if request.max_duration_seconds:
            allocation.expires_at = time.time() + request.max_duration_seconds
            
        return allocation
        
    def _allocate_fair_share(self, request: ResourceRequest) -> Optional[ResourceAllocation]:
        """Fair share allocation - distribute resources equally."""
        active_tasks = len(self.active_allocations) + 1  # +1 for new request
        
        # Calculate fair share of each resource
        fair_memory = self.resource_pool.available_memory_mb() // active_tasks
        fair_cpu = self.resource_pool.available_cpu_percent() / active_tasks
        
        allocation = ResourceAllocation(
            allocation_id=f"alloc_{int(time.time() * 1000)}_{request.task_id}",
            request_id=request.request_id,
            task_id=request.task_id,
            memory_mb=min(request.memory_mb or fair_memory, fair_memory),
            cpu_percent=min(request.cpu_percent or fair_cpu, fair_cpu),
            gpu_memory_mb=min(request.gpu_memory_mb or 0, self.resource_pool.available_gpu_memory_mb()),
            storage_mb=min(request.storage_mb or 0, self.resource_pool.available_storage_mb()),
            network_kbps=min(request.network_kbps or 0, self.resource_pool.available_network_kbps()),
        )
        
        return allocation
        
    def _allocate_performance_based(self, request: ResourceRequest) -> Optional[ResourceAllocation]:
        """Performance-based allocation - allocate based on device capability."""
        current_snapshot = self.device_profiler.current_snapshot
        
        if not current_snapshot:
            return self._allocate_greedy(request)
            
        # Scale allocation based on current performance
        performance_factor = current_snapshot.performance_score
        
        allocation = ResourceAllocation(
            allocation_id=f"alloc_{int(time.time() * 1000)}_{request.task_id}",
            request_id=request.request_id,
            task_id=request.task_id,
            memory_mb=int((request.memory_mb or 0) * performance_factor),
            cpu_percent=(request.cpu_percent or 0) * performance_factor,
            gpu_memory_mb=int((request.gpu_memory_mb or 0) * performance_factor),
            storage_mb=request.storage_mb or 0,
            network_kbps=request.network_kbps or 0,
        )
        
        # Ensure we don't exceed available resources
        if not self.resource_pool.can_allocate(ResourceRequest(
            request_id="temp",
            task_id="temp",
            priority=request.priority,
            memory_mb=allocation.memory_mb,
            cpu_percent=allocation.cpu_percent,
            gpu_memory_mb=allocation.gpu_memory_mb,
        )):
            return None
            
        return allocation
        
    def _allocate_battery_aware(self, request: ResourceRequest) -> Optional[ResourceAllocation]:
        """Battery-aware allocation - reduce resources when battery is low."""
        current_snapshot = self.device_profiler.current_snapshot
        
        if not current_snapshot:
            return self._allocate_greedy(request)
            
        # Scale down based on battery state
        battery_factor = 1.0
        
        if current_snapshot.power_state == PowerState.BATTERY_CRITICAL:
            battery_factor = 0.3
        elif current_snapshot.power_state == PowerState.BATTERY_LOW:
            battery_factor = 0.5
        elif current_snapshot.power_state == PowerState.BATTERY_MEDIUM:
            battery_factor = 0.8
        elif current_snapshot.power_plugged:
            battery_factor = 1.0
            
        # Also consider battery impact of request
        battery_factor *= (1.0 - request.battery_impact * 0.5)
        
        allocation = ResourceAllocation(
            allocation_id=f"alloc_{int(time.time() * 1000)}_{request.task_id}",
            request_id=request.request_id,
            task_id=request.task_id,
            memory_mb=int((request.memory_mb or 0) * battery_factor),
            cpu_percent=(request.cpu_percent or 0) * battery_factor,
            gpu_memory_mb=int((request.gpu_memory_mb or 0) * battery_factor),
            storage_mb=request.storage_mb or 0,
            network_kbps=request.network_kbps or 0,
        )
        
        return allocation
        
    def _allocate_thermal_aware(self, request: ResourceRequest) -> Optional[ResourceAllocation]:
        """Thermal-aware allocation - reduce CPU/GPU when device is hot."""
        current_snapshot = self.device_profiler.current_snapshot
        
        if not current_snapshot:
            return self._allocate_greedy(request)
            
        # Scale down CPU/GPU based on temperature
        thermal_factor = 1.0
        
        if current_snapshot.cpu_temp:
            if current_snapshot.cpu_temp > 85:  # Critical
                thermal_factor = 0.3
            elif current_snapshot.cpu_temp > 75:  # Hot
                thermal_factor = 0.6
            elif current_snapshot.cpu_temp > 60:  # Warm
                thermal_factor = 0.8
                
        # Consider thermal impact of request
        thermal_factor *= (1.0 - request.thermal_impact * 0.3)
        
        allocation = ResourceAllocation(
            allocation_id=f"alloc_{int(time.time() * 1000)}_{request.task_id}",
            request_id=request.request_id,
            task_id=request.task_id,
            memory_mb=request.memory_mb or 0,  # Memory not affected by thermal
            cpu_percent=(request.cpu_percent or 0) * thermal_factor,
            gpu_memory_mb=int((request.gpu_memory_mb or 0) * thermal_factor),
            storage_mb=request.storage_mb or 0,
            network_kbps=request.network_kbps or 0,
        )
        
        return allocation
        
    def _allocate_adaptive(self, request: ResourceRequest) -> Optional[ResourceAllocation]:
        """Adaptive allocation - choose best strategy based on current conditions."""
        current_snapshot = self.device_profiler.current_snapshot
        
        if not current_snapshot:
            return self._allocate_greedy(request)
            
        # Choose strategy based on current conditions
        if current_snapshot.power_state in [PowerState.BATTERY_LOW, PowerState.BATTERY_CRITICAL]:
            return self._allocate_battery_aware(request)
        elif current_snapshot.cpu_temp and current_snapshot.cpu_temp > 75:
            return self._allocate_thermal_aware(request)
        elif current_snapshot.performance_score < 0.5:
            return self._allocate_performance_based(request)
        elif len(self.active_allocations) > 5:
            return self._allocate_fair_share(request)
        else:
            return self._allocate_greedy(request)
            
    def _choose_strategy(self, request: ResourceRequest) -> AllocationStrategy:
        """Choose allocation strategy for a request."""
        # High priority requests get greedy allocation
        if request.priority in [Priority.CRITICAL, Priority.HIGH]:
            return AllocationStrategy.GREEDY
            
        # Otherwise use current strategy
        return self.current_strategy
        
    def _validate_request(self, request: ResourceRequest) -> bool:
        """Validate resource request."""
        # Check for reasonable resource amounts
        max_memory_mb = self.resource_pool.total_memory_mb
        
        if request.memory_mb and request.memory_mb > max_memory_mb:
            return False
        if request.cpu_percent and request.cpu_percent > 100:
            return False
        if request.gpu_memory_mb and request.gpu_memory_mb > self.resource_pool.total_gpu_memory_mb:
            return False
            
        return True
        
    def _try_preemption(self, request: ResourceRequest) -> bool:
        """Try to preempt lower priority allocations."""
        if not self.enable_preemption:
            return False
            
        # Find preemptable allocations with lower priority
        candidates = []
        for allocation in self.active_allocations.values():
            # Find the original request priority (simplified)
            # In practice, would store priority with allocation
            candidates.append(allocation)
            
        # Sort by some criteria (age, priority, etc.)
        candidates.sort(key=lambda a: a.allocated_at)
        
        # Try preempting candidates until we have enough resources
        preempted = []
        for candidate in candidates:
            if self._should_preempt(candidate, request):
                self.resource_pool.deallocate_resources(candidate)
                candidate.active = False
                candidate.preempted = True
                preempted.append(candidate)
                
                self.allocation_stats["preemptions_performed"] += 1
                
                # Check if we now have enough resources
                if self.resource_pool.can_allocate(request):
                    # Move preempted allocations back to pending
                    for preempted_alloc in preempted:
                        del self.active_allocations[preempted_alloc.allocation_id]
                        # Would recreate request and add to pending queue
                        
                    return True
                    
        # Couldn't free enough resources
        # Restore preempted allocations
        for preempted_alloc in preempted:
            self.resource_pool.allocate_resources(preempted_alloc)
            preempted_alloc.active = True
            preempted_alloc.preempted = False
            
        return False
        
    def _should_preempt(self, allocation: ResourceAllocation, new_request: ResourceRequest) -> bool:
        """Determine if an allocation should be preempted."""
        # Simplified logic - would consider priorities, ages, etc.
        return new_request.priority.value < Priority.MEDIUM.value
        
    def _activate_allocation(self, allocation: ResourceAllocation) -> None:
        """Activate an allocation."""
        self.resource_pool.allocate_resources(allocation)
        self.active_allocations[allocation.allocation_id] = allocation
        
        logger.debug(f"Activated allocation {allocation.allocation_id} for task {allocation.task_id}")
        
    async def _process_pending_requests(self) -> None:
        """Process pending resource requests."""
        processed = []
        
        while self.pending_requests:
            request = heapq.heappop(self.pending_requests)
            
            allocation = await self._try_allocate(request)
            
            if allocation:
                self._activate_allocation(allocation)
                self.allocation_stats["requests_granted"] += 1
                processed.append(allocation)
                
                # Notify callbacks
                for callback in self.allocation_callbacks:
                    try:
                        callback("allocated", allocation)
                    except Exception as e:
                        logger.error(f"Error in allocation callback: {e}")
            else:
                # Put back in queue
                heapq.heappush(self.pending_requests, request)
                break  # Can't allocate this one, likely can't allocate others
                
        if processed:
            logger.info(f"Processed {len(processed)} pending requests")
            
    def start_allocator(self) -> None:
        """Start background allocator."""
        if self.allocator_active:
            return
            
        self.allocator_active = True
        self.allocator_thread = threading.Thread(
            target=self._allocator_loop,
            daemon=True
        )
        self.allocator_thread.start()
        
        logger.info("Resource allocator started")
        
    def stop_allocator(self) -> None:
        """Stop background allocator."""
        self.allocator_active = False
        
        if self.allocator_thread and self.allocator_thread.is_alive():
            self.allocator_thread.join(timeout=5.0)
            
        logger.info("Resource allocator stopped")
        
    def _allocator_loop(self) -> None:
        """Background allocator loop."""
        while self.allocator_active:
            try:
                # Check for expired allocations
                self._cleanup_expired_allocations()
                
                # Process pending requests
                asyncio.run(self._process_pending_requests())
                
                # Update strategy if adaptive
                if self.current_strategy == AllocationStrategy.ADAPTIVE:
                    self._update_adaptive_strategy()
                    
                # Collect resource usage data
                self._collect_usage_data()
                
                time.sleep(5.0)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in allocator loop: {e}")
                time.sleep(10.0)
                
    def _cleanup_expired_allocations(self) -> None:
        """Clean up expired allocations."""
        expired = []
        
        for allocation_id, allocation in self.active_allocations.items():
            if allocation.is_expired():
                expired.append(allocation_id)
                
        for allocation_id in expired:
            asyncio.run(self.release_resources(allocation_id))
            logger.info(f"Released expired allocation {allocation_id}")
            
    def _update_adaptive_strategy(self) -> None:
        """Update strategy for adaptive allocation."""
        current_snapshot = self.device_profiler.current_snapshot
        
        if not current_snapshot:
            return
            
        # Simple adaptive logic
        new_strategy = self.current_strategy
        
        if current_snapshot.battery_percent and current_snapshot.battery_percent < 20:
            new_strategy = AllocationStrategy.BATTERY_AWARE
        elif current_snapshot.cpu_temp and current_snapshot.cpu_temp > 75:
            new_strategy = AllocationStrategy.THERMAL_AWARE
        elif current_snapshot.memory_percent > 85:
            new_strategy = AllocationStrategy.FAIR_SHARE
        else:
            new_strategy = AllocationStrategy.PERFORMANCE_BASED
            
        if new_strategy != self.current_strategy:
            self.current_strategy = new_strategy
            self.allocation_stats["strategy_changes"] += 1
            logger.info(f"Switched allocation strategy to {new_strategy.value}")
            
    def _collect_usage_data(self) -> None:
        """Collect resource usage data for prediction."""
        current_snapshot = self.device_profiler.current_snapshot
        
        if current_snapshot:
            self.resource_usage_history["memory"].append(current_snapshot.memory_percent)
            self.resource_usage_history["cpu"].append(current_snapshot.cpu_percent)
            
            if current_snapshot.battery_percent:
                self.resource_usage_history["battery"].append(current_snapshot.battery_percent)
                
    def register_allocation_callback(
        self, 
        callback: Callable[[str, ResourceAllocation], None]
    ) -> None:
        """Register callback for allocation events."""
        self.allocation_callbacks.append(callback)
        
    def get_allocation_status(self) -> Dict[str, Any]:
        """Get current allocation status."""
        utilization = self.resource_pool.get_utilization()
        
        return {
            "strategy": self.current_strategy.value,
            "active_allocations": len(self.active_allocations),
            "pending_requests": len(self.pending_requests),
            "resource_utilization": utilization,
            "available_resources": {
                "memory_mb": self.resource_pool.available_memory_mb(),
                "cpu_percent": self.resource_pool.available_cpu_percent(),
                "gpu_memory_mb": self.resource_pool.available_gpu_memory_mb(),
                "storage_mb": self.resource_pool.available_storage_mb(),
                "network_kbps": self.resource_pool.available_network_kbps(),
            },
            "statistics": self.allocation_stats.copy(),
        }
        
    def get_allocation_history(self) -> List[Dict[str, Any]]:
        """Get allocation history."""
        return [
            {
                "allocation_id": alloc.allocation_id,
                "task_id": alloc.task_id,
                "strategy_used": alloc.strategy_used.value if alloc.strategy_used else None,
                "allocated_at": alloc.allocated_at,
                "expires_at": alloc.expires_at,
                "resources": alloc.get_resource_footprint(),
                "preempted": alloc.preempted,
            }
            for alloc in self.allocation_history
        ]